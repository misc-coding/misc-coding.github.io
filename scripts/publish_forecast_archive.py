#!/usr/bin/env python3
"""Build a rolling, static archive for the India multi-model forecast site.

This command deliberately runs on the workstation that has access to the private
WeatherNext archives.  It creates a complete temporary site and replaces the
published files only after every selected run has passed validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import re
import signal
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

SITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REALTIME_ROOT = Path("/home/saptarishi.dhanuka_asp25/weather/real_time")
DEFAULT_PYTHON = Path("/Datastorage/saptarishi.dhanuka_asp25/conda_envs/realtime_dash/bin/python")
LEAD_DAYS = (1, 3, 5)
DAILY_LEAD_DAYS = (1, 2, 3, 4, 5)
RANGE_VARIABLES = (
    ("temperature_high", "Daily high 2 m temperature", "maximum"),
    ("temperature_low", "Daily low 2 m temperature", "minimum"),
)
ARTIFACTS_PER_RUN = 84
GRID_VARIABLES = ("temperature", "precipitation", "temperature_high", "temperature_low")
ALL_MODEL_IDS = ("weathernext2", "gencast", "gfs", "gefs", "aifs", "ifs_ens")


def stamp(init: pd.Timestamp) -> str:
    return pd.Timestamp(init).strftime("%Y%m%d_%H")


def utc_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_renderer(realtime_root: Path):
    """Import the maintained model adapters and cartopy renderer without copying them."""
    scripts = realtime_root / "scripts"
    source = realtime_root / "src"
    if not (scripts / "publish_forecast_site.py").is_file() or not source.is_dir():
        raise RuntimeError(f"real-time project is incomplete: {realtime_root}")
    sys.path[:0] = [str(scripts), str(source)]
    import publish_forecast_site as renderer  # type: ignore
    from realtime_dash.config import load_config  # type: ignore
    from realtime_dash.india import load as india_load  # type: ignore
    from realtime_dash.sources import openmeteo  # type: ignore
    # The shared renderer is intentionally parameterized by this module-level
    # sequence; retain its tested rendering machinery while publishing 1/3/5-day
    # products instead of its historical 1/2/3-day default.
    renderer.LEAD_DAYS = LEAD_DAYS
    return renderer, load_config, india_load, openmeteo


@contextmanager
def source_timeout(seconds: int):
    """Bound a synchronous source-catalog request without leaking worker threads."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("source lookup timed out")))
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def model_availability(models, cfg, india_load, *, timeout_seconds: int = 90) -> tuple[dict[str, set[pd.Timestamp]], dict[str, str]]:
    """Return independent source availability so one late model cannot block publication."""
    availability: dict[str, set[pd.Timestamp]] = {}
    errors: dict[str, str] = {}
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for model in models:
        try:
            with source_timeout(timeout_seconds):
                raw = india_load.available_inits(model, cfg)
            values = {
                pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)
                for value in raw
            }
            availability[model] = {value for value in values if value.hour == 0 and value <= now}
            print(f"[{model}] newest available init: {max(availability[model]) if availability[model] else 'none'}", flush=True)
        except Exception as exc:  # noqa: BLE001 - catalog failures are isolated by source
            availability[model] = set()
            errors[model] = f"{type(exc).__name__}: {exc}"
            print(f"[{model}] availability unavailable: {errors[model]}", file=sys.stderr, flush=True)
    return availability, errors


def common_midnight_inits(models, cfg, india_load) -> list[pd.Timestamp]:
    """Compatibility helper retained for tests and strict comparisons."""
    availability, _ = model_availability(models, cfg, india_load)
    sets = [availability[model] for model in models]
    return sorted(set.intersection(*sets), reverse=True) if sets else []


def candidate_initializations(
    availability: dict[str, set[pd.Timestamp]], existing: list[dict], *, backfill: bool = False,
) -> list[pd.Timestamp]:
    """Choose fresh runs only; also revisit the latest partial run as sources arrive."""
    union = sorted(set().union(*availability.values()) if availability else set(), reverse=True)
    if not existing:
        return union
    if backfill:
        by_id = {run["id"]: run for run in existing}
        candidates = []
        for value in union:
            run = by_id.get(stamp(value))
            have = {model["id"] for model in run.get("models", [])} if run else set()
            ready = {model for model, values in availability.items() if value in values}
            if not run or ready - have:
                candidates.append(value)
        return candidates
    latest = pd.Timestamp(max(run["initialization_utc"] for run in existing)).tz_localize(None)
    latest_run = next(run for run in existing if pd.Timestamp(run["initialization_utc"]).tz_localize(None) == latest)
    have = {model["id"] for model in latest_run.get("models", [])}
    available_latest = {model for model, values in availability.items() if latest in values}
    candidates = [value for value in union if value > latest]
    if available_latest - have:
        candidates.append(latest)
    return sorted(set(candidates), reverse=True)


def read_archive(site: Path) -> dict:
    path = site / "assets" / "forecast_archive.json"
    if not path.is_file():
        return {"schema_version": 1, "runs": []}
    archive = json.loads(path.read_text())
    if archive.get("schema_version") not in (1, 2) or not isinstance(archive.get("runs"), list):
        raise RuntimeError(f"unsupported archive manifest: {path}")
    return archive


def valid_existing_runs(site: Path, archive: dict, renderer) -> list[dict]:
    """Keep only archive records whose entire asset set is present and valid."""
    runs = []
    for run in archive["runs"]:
        try:
            init = pd.Timestamp(run["initialization_utc"])
            run_leads = tuple(item["day"] for item in run.get("lead_days", []))
            model_ids = {model["id"] for model in run.get("models", [])}
            artifact_models = {artifact.get("model") for artifact in run.get("artifacts", [])}
            if stamp(init) != run["id"] or not model_ids or artifact_models != model_ids or run_leads != LEAD_DAYS:
                continue
            for artifact in run["artifacts"]:
                path = site / artifact["path"]
                if artifact.get("kind") != "grid" or not path.is_file() or path.stat().st_size < 50_000:
                    raise ValueError(f"invalid map payload: {path}")
            runs.append(run)
        except (KeyError, OSError, ValueError):
            continue
    return runs


def render_run(init, models, cfg, renderer, india_load, stage: Path, attempts: int) -> dict:
    """Render every usable model for an init and retain a valid partial run."""
    datasets = {}
    for model in models:
        print(f"[{stamp(init)}] loading {renderer.MODEL_META[model]['label']}", flush=True)
        try:
            with source_timeout(20 * 60):
                datasets[model] = renderer.load_with_retries(
                    model, cfg, init, max_members=8, attempts=attempts,
                )
        except Exception as exc:  # noqa: BLE001 - partial publication is intentional
            print(f"[{stamp(init)}] {model} unavailable: {exc}", file=sys.stderr, flush=True)
    endpoint_models = tuple(model for model in models if model in datasets)
    if not endpoint_models:
        raise RuntimeError("no model produced a complete five-day forecast")
    artifacts = []
    grid_metadata = None
    grid_models = []
    for model in endpoint_models:
        try:
            with source_timeout(5 * 60):
                records, metadata = write_map_payloads(
                    init, {model: datasets[model]}, (model,), cfg, india_load, stage,
                )
            artifacts.extend(records)
            grid_metadata = grid_metadata or metadata
            grid_models.append(model)
        except Exception as exc:  # noqa: BLE001 - a slow native series must not block fresher models
            print(f"[{stamp(init)}] {model} daily fields unavailable: {exc}", file=sys.stderr, flush=True)
    loaded_models = tuple(grid_models)
    if not loaded_models or grid_metadata is None:
        raise RuntimeError("no model produced complete map and daily weather fields")
    datasets = {model: datasets[model] for model in loaded_models}
    manifest = renderer.build_manifest(datasets, loaded_models, init, cfg, artifacts)
    manifest["lead_semantics"] = {
        "temperature": "Exact 2 m temperature snapshot at T+24, T+72, and T+120 hours.",
        "precipitation": "Cumulative precipitation from initialization through T+24, T+72, and T+120 hours.",
        "temperature_high": "Maximum native-step 2 m temperature during the 24 hours ending at each selected lead.",
        "temperature_low": "Minimum native-step 2 m temperature during the 24 hours ending at each selected lead.",
    }
    return {
        "id": stamp(init),
        "initialization_utc": manifest["initialization_utc"],
        "generated_at_utc": manifest["generated_at_utc"],
        "lead_days": manifest["lead_days"],
        "models": manifest["models"],
        "variables": manifest["variables"],
        "lead_semantics": manifest["lead_semantics"],
        "grid_metadata": grid_metadata,
        "bounding_box": manifest["bounding_box"],
        "disclaimer": manifest["disclaimer"],
        "artifacts": artifacts,
        "available_models": list(loaded_models),
        "missing_models": [model for model in ALL_MODEL_IDS if model not in loaded_models],
        "status": "complete" if len(loaded_models) == len(ALL_MODEL_IDS) else "partial",
    }


def _daily_temperature_ranges(model: str, init, cfg, india_load, reference: xr.Dataset) -> dict[str, xr.Dataset]:
    """Derive daily extrema from every available forecast time step, not endpoints.

    The public archive samples day 1, 3, and 5.  For each selected day, high and
    low are calculated over that *calendar forecast day* (e.g. T+48..T+72 for
    day 3), using the source model's native time steps after ensemble reduction.
    """
    series = india_load.load_india_series_cached(
        model, cfg, init, horizon_days=max(LEAD_DAYS), max_members=8,
    ).load()
    valid = pd.to_datetime(series["valid_time"].values).tz_localize(None)
    fields: dict[str, list[xr.DataArray]] = {"maximum": [], "minimum": []}
    targets = []
    init_time = pd.Timestamp(init).tz_localize(None)
    for day in LEAD_DAYS:
        start = init_time + pd.Timedelta(days=day - 1)
        end = init_time + pd.Timedelta(days=day)
        chosen = np.flatnonzero((valid > start) & (valid <= end))
        if not len(chosen):
            raise ValueError(f"{model}: no temperature samples in {start}..{end}")
        daily = series["t2m_C"].isel(valid_time=chosen)
        fields["maximum"].append(daily.max("valid_time"))
        fields["minimum"].append(daily.min("valid_time"))
        targets.append(np.datetime64(end, "ns"))

    out = {}
    for kind, values in fields.items():
        data = xr.concat(
            values,
            dim=xr.DataArray(np.asarray(LEAD_DAYS, dtype=np.int16), dims="lead_day", name="lead_day"),
        )
        dataset = xr.Dataset({"t2m_C": data})
        dataset = dataset.assign_coords(valid_time=("lead_day", np.asarray(targets)))
        dataset.attrs = dict(reference.attrs)
        dataset.attrs["temperature_range_definition"] = (
            f"daily {kind} of native-step 2 m temperature over the 24 hours ending at each displayed lead"
        )
        out[kind] = dataset
    return out


def render_daily_temperature_ranges(init, datasets, models, cfg, renderer, india_load, stage: Path) -> list[dict]:
    """Render daily high/low map layers while retaining the shared renderer style."""
    extrema = {
        model: _daily_temperature_ranges(model, init, cfg, india_load, datasets[model])
        for model in models
    }
    tag = stamp(init)
    records = []
    original_meta = dict(renderer.VAR_META["temperature"])
    original_caption = renderer._lead_caption
    try:
        for variable, label, kind in RANGE_VARIABLES:
            renderer.VAR_META["temperature"].update({
                "label": label,
                "short_label": "Daily high" if kind == "maximum" else "Daily low",
                "description": (
                    "Maximum" if kind == "maximum" else "Minimum"
                ) + " 2 m temperature over the 24 hours ending at the selected lead.",
            })

            def range_caption(_variable, run_init, day, *, _kind=kind):
                start = pd.Timestamp(run_init) + pd.Timedelta(days=day - 1)
                end = pd.Timestamp(run_init) + pd.Timedelta(days=day)
                word = "maximum" if _kind == "maximum" else "minimum"
                return f"Daily {word} · {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC"

            renderer._lead_caption = range_caption
            range_datasets = {model: extrema[model][kind] for model in models}
            for day in LEAD_DAYS:
                rel = Path("assets") / "forecasts" / tag / "comparisons" / f"{variable}_day{day}.png"
                out = stage / rel
                renderer.render_comparison(
                    range_datasets, models, "temperature", day,
                    init=init, bbox=cfg.india_bbox, out=out,
                )
                renderer.validate_png(out)
                records.append({"kind": "comparison", "variable": variable, "day": day, "path": rel.as_posix()})
                for model in models:
                    rel = Path("assets") / "forecasts" / tag / model / f"{variable}_day{day}.png"
                    out = stage / rel
                    renderer.render_individual(
                        range_datasets[model], model, "temperature", day,
                        init=init, bbox=cfg.india_bbox, out=out,
                    )
                    renderer.validate_png(out)
                    records.append({
                        "kind": "individual", "model": model, "variable": variable,
                        "day": day, "path": rel.as_posix(),
                    })
    finally:
        renderer.VAR_META["temperature"].clear()
        renderer.VAR_META["temperature"].update(original_meta)
        renderer._lead_caption = original_caption
    if len(records) != 42:
        raise AssertionError(f"expected 42 daily-range PNGs, produced {len(records)}")
    return records


def _encode_grid(values: np.ndarray, variable: str) -> np.ndarray:
    """Quantize map fields to compact uint16 payloads for client-side rendering."""
    values = np.asarray(values, dtype=np.float32)
    if variable == "precipitation":
        encoded = np.rint(np.clip(values, 0, 6500.0) * 10.0)
    else:
        encoded = np.rint(np.clip(values, -50.0, 65.0) * 100.0 + 5000.0)
    encoded = np.where(np.isfinite(values), encoded, 65535).astype("<u2")
    return encoded


def write_map_payloads(init, datasets, models, cfg, india_load, stage: Path) -> tuple[list[dict], dict]:
    """Write compact browser payloads instead of pre-rendered PNG map galleries."""
    tag = stamp(init)
    target = stage / "assets" / "map_data" / tag
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for model in models:
        ranges = _daily_temperature_ranges(model, init, cfg, india_load, datasets[model])
        fields = {
            "temperature": datasets[model]["t2m_C"].values,
            "precipitation": datasets[model]["precip_cumulative_mm"].values,
            "temperature_high": ranges["maximum"]["t2m_C"].values,
            "temperature_low": ranges["minimum"]["t2m_C"].values,
        }
        payload = np.concatenate([_encode_grid(fields[variable], variable).reshape(-1) for variable in GRID_VARIABLES])
        relative = Path("assets") / "map_data" / tag / f"{model}.bin"
        (stage / relative).write_bytes(payload.tobytes())
        records.append({"kind": "grid", "model": model, "path": relative.as_posix(), "bytes": int(payload.nbytes)})
    metadata = {
        "variables": list(GRID_VARIABLES), "lead_days": list(LEAD_DAYS),
        "shape": [len(LEAD_DAYS), int(datasets[models[0]].sizes["lat"]), int(datasets[models[0]].sizes["lon"])],
        "bounding_box": {"lat_min": cfg.india_bbox.lat_min, "lat_max": cfg.india_bbox.lat_max, "lon_min": cfg.india_bbox.lon_min, "lon_max": cfg.india_bbox.lon_max},
        "encoding": {"temperature": "uint16: (value - 5000) / 100 °C; 65535 = missing", "precipitation": "uint16: value / 10 mm; 65535 = missing"},
    }
    return records, metadata


def has_daily_temperature_ranges(run: dict) -> bool:
    available = {artifact.get("variable") for artifact in run.get("artifacts", [])}
    return all(variable in available for variable, _, _ in RANGE_VARIABLES)


def add_available_daily_ranges(retained: list[dict], stage: Path, renderer) -> list[dict]:
    """Register already-rendered range assets copied into the atomic stage."""
    updated = []
    for run in retained:
        if has_daily_temperature_ranges(run):
            updated.append(run)
            continue
        records = []
        complete = True
        for variable, _, _ in RANGE_VARIABLES:
            for day in LEAD_DAYS:
                comparison = Path("assets") / "forecasts" / run["id"] / "comparisons" / f"{variable}_day{day}.png"
                if not (stage / comparison).is_file():
                    complete = False
                    break
                renderer.validate_png(stage / comparison)
                records.append({"kind": "comparison", "variable": variable, "day": day, "path": comparison.as_posix()})
                for model in run["models"]:
                    individual = Path("assets") / "forecasts" / run["id"] / model["id"] / f"{variable}_day{day}.png"
                    if not (stage / individual).is_file():
                        complete = False
                        break
                    renderer.validate_png(stage / individual)
                    records.append({"kind": "individual", "model": model["id"], "variable": variable, "day": day, "path": individual.as_posix()})
                if not complete:
                    break
            if not complete:
                break
        if complete and len(records) == 42:
            refreshed = dict(run)
            refreshed["artifacts"] = list(run["artifacts"]) + records
            refreshed["lead_semantics"] = {
                **run["lead_semantics"],
                "temperature_high": "Maximum native-step 2 m temperature during the 24 hours ending at each selected lead.",
                "temperature_low": "Minimum native-step 2 m temperature during the 24 hours ending at each selected lead.",
            }
            updated.append(refreshed)
        else:
            updated.append(run)
    return updated


def add_latest_daily_ranges(retained: list[dict], cfg, renderer, india_load, stage: Path) -> list[dict]:
    """Add native-step high/low layers to the latest run without discarding history.

    Historical endpoint products remain immediately available.  The range layer is
    introduced on the latest initialization first, then is emitted for every new
    initialization by ``render_run``.
    """
    if not retained:
        return retained
    latest = dict(retained[0])
    if has_daily_temperature_ranges(latest):
        latest["lead_semantics"] = {
            **latest["lead_semantics"],
            "temperature_high": "Maximum native-step 2 m temperature during the 24 hours ending at each selected lead.",
            "temperature_low": "Minimum native-step 2 m temperature during the 24 hours ending at each selected lead.",
        }
        return [latest, *retained[1:]]
    datasets = {
        model["id"]: _open_run_dataset(cfg, latest, model["id"])
        for model in latest["models"]
    }
    latest["artifacts"] = list(latest["artifacts"]) + render_daily_temperature_ranges(
        pd.Timestamp(latest["initialization_utc"]), datasets,
        tuple(model["id"] for model in latest["models"]), cfg, renderer, india_load, stage,
    )
    latest["lead_semantics"] = {
        **latest["lead_semantics"],
        "temperature_high": "Maximum native-step 2 m temperature during the 24 hours ending at each selected lead.",
        "temperature_low": "Minimum native-step 2 m temperature during the 24 hours ending at each selected lead.",
    }
    return [latest, *retained[1:]]


def archive_manifest(runs: list[dict]) -> dict:
    runs = sorted(runs, key=lambda run: run["initialization_utc"], reverse=True)
    return {
        "schema_version": 2,
        "title": "India Multi-Model Forecast Archive",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "retention_runs": len(runs),
        "latest_initialization_utc": runs[0]["initialization_utc"],
        "runs": runs,
    }


MODEL_COLORS = {
    "weathernext2": "#3b82f6", "gencast": "#a855f7", "gfs": "#e05d44",
    "gefs": "#f59e0b", "aifs": "#0f9b8e", "ifs_ens": "#374151",
}


def _truth_lookup(frame: pd.DataFrame, time_col: str, value_col: str) -> dict:
    return {
        pd.Timestamp(row[time_col]).tz_localize(None): float(row[value_col])
        for _, row in frame.iterrows() if pd.notna(row[value_col])
    }


def _open_run_dataset(cfg, run: dict, model: str):
    path = cfg.cache_root / "india" / run["id"] / f"{model}_lead_days_1-3-5.nc"
    if not path.is_file():
        raise RuntimeError(f"missing cached map data for validation: {path}")
    with xr.open_dataset(path) as opened:
        return opened.load()


def _model_catalog(archive: dict) -> list[dict]:
    """Return the union of model metadata in stable public order."""
    found = {model["id"]: model for run in archive["runs"] for model in run.get("models", [])}
    return [found[model] for model in ALL_MODEL_IDS if model in found]


def _validation_records(archive: dict, cfg, openmeteo) -> dict:
    """Match published point forecasts to Open-Meteo observation time windows."""
    truth = openmeteo.load_truth(cfg, cfg.cities, past_days=90, forecast_days=1)
    records = {}
    for city in cfg.cities:
        temp_truth, precip_truth = truth[city.name]
        temperatures = _truth_lookup(temp_truth, "valid_time", "t2m_C")
        daily_rain = _truth_lookup(precip_truth, "valid_date", "precip_mm_day")
        city_data = {"temperature": [], "precipitation": []}
        for run in archive["runs"]:
            init = pd.Timestamp(run["initialization_utc"]).tz_localize(None)
            datasets = {model["id"]: _open_run_dataset(cfg, run, model["id"])
                        for model in run["models"]}
            for day in LEAD_DAYS:
                valid = init + pd.Timedelta(days=day)
                temp_obs = temperatures.get(valid)
                rain_days = pd.date_range(init.floor("D"), valid.floor("D"), inclusive="left")
                rain_values = [daily_rain.get(pd.Timestamp(date)) for date in rain_days]
                rain_obs = sum(rain_values) if len(rain_values) == day and all(v is not None for v in rain_values) else None
                temp_forecasts, rain_forecasts = {}, {}
                for model, dataset in datasets.items():
                    point = dataset.sel(lat=city.lat, lon=city.lon, method="nearest")
                    temp_forecasts[model] = float(point["t2m_C"].sel(lead_day=day).item())
                    rain_forecasts[model] = float(point["precip_cumulative_mm"].sel(lead_day=day).item())
                if temp_obs is not None:
                    city_data["temperature"].append({
                        "run": run["id"], "lead_day": day, "valid_time_utc": utc_text(valid),
                        "observed": temp_obs, "forecasts": temp_forecasts,
                    })
                if rain_obs is not None:
                    city_data["precipitation"].append({
                        "run": run["id"], "lead_day": day, "valid_time_utc": utc_text(valid),
                        "observed": rain_obs, "forecasts": rain_forecasts,
                    })
        records[city.name] = city_data
    return records


def _plot_validation(records: list[dict], city, variable: str, models: list[dict], out: Path) -> dict:
    label = "2 m temperature" if variable == "temperature" else "Cumulative precipitation"
    unit = "°C" if variable == "temperature" else "mm"
    fig, (scatter_ax, skill_ax) = plt.subplots(1, 2, figsize=(13.2, 5.6), facecolor="#f5f8f7")
    fig.subplots_adjust(left=.07, right=.98, bottom=.19, top=.80, wspace=.28)
    values = []
    skill = {}
    for model in models:
        model_id = model["id"]
        pairs = [(float(row["observed"]), float(row["forecasts"][model_id]), int(row["lead_day"]))
                 for row in records if model_id in row["forecasts"]]
        if not pairs:
            continue
        obs, forecast, leads = map(np.asarray, zip(*pairs))
        values.extend(obs.tolist() + forecast.tolist())
        scatter_ax.scatter(obs, forecast, s=34, alpha=.74, color=MODEL_COLORS[model_id],
                           edgecolor="white", linewidth=.45, label=model["label"])
        skill[model_id] = {
            "label": model["label"], "n": int(len(obs)),
            "mae_by_lead": {str(lead): float(np.mean(np.abs(forecast[leads == lead] - obs[leads == lead])))
                            for lead in LEAD_DAYS if np.any(leads == lead)},
        }
    if values:
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * .08, 1.0 if variable == "temperature" else 2.0)
        scatter_ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#74838a", lw=1, ls="--", zorder=0)
        scatter_ax.set_xlim(lo - pad, hi + pad)
        scatter_ax.set_ylim(lo - pad, hi + pad)
    scatter_ax.set_xlabel(f"Open-Meteo observed ({unit})")
    scatter_ax.set_ylabel(f"Forecast ({unit})")
    scatter_ax.grid(alpha=.2)
    scatter_ax.legend(loc="best", fontsize=7.8, frameon=False)
    for model in models:
        item = skill.get(model["id"])
        if not item:
            continue
        leads = sorted(int(key) for key in item["mae_by_lead"])
        skill_ax.plot(leads, [item["mae_by_lead"][str(lead)] for lead in leads], marker="o", lw=2,
                      color=MODEL_COLORS[model["id"]], label=model["label"])
    skill_ax.set_xticks(LEAD_DAYS, ["Day 1", "Day 3", "Day 5"])
    skill_ax.set_xlabel("Forecast lead")
    skill_ax.set_ylabel(f"Mean absolute error ({unit})")
    skill_ax.grid(alpha=.2)
    skill_ax.legend(loc="best", fontsize=7.8, frameon=False)
    fig.suptitle(f"{city.name} · {label} verification", fontsize=16, fontweight="bold", color="#132a35")
    detail = "exact valid-time temperature" if variable == "temperature" else "rain accumulated from initialization to each valid endpoint"
    fig.text(.5, .05, f"Forecast values sampled at {city.lat:.2f}°N, {city.lon:.2f}°E · {detail} · observations: Open-Meteo", ha="center", fontsize=8.5, color="#53636b")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"matched_points": max((item["n"] for item in skill.values()), default=0), "models": skill}


def _plot_matched_timeseries(records: list[dict], city, variable: str, run: dict, models: list[dict], out: Path) -> dict:
    """Show a single initialization's model values and matched truth over its three leads."""
    rows = sorted((row for row in records if row["run"] == run["id"]), key=lambda row: row["lead_day"])
    label = "2 m temperature" if variable == "temperature" else "Cumulative precipitation"
    unit = "°C" if variable == "temperature" else "mm"
    fig, ax = plt.subplots(figsize=(10.8, 5.5), facecolor="#f5f8f7")
    fig.subplots_adjust(left=.10, right=.97, bottom=.22, top=.80)
    leads = np.asarray([row["lead_day"] for row in rows], dtype=int)
    observations = np.asarray([row["observed"] for row in rows], dtype=float)
    if len(rows):
        ax.plot(leads, observations, color="#121f2a", marker="o", markersize=6, lw=2.8,
                label="Open-Meteo observed", zorder=5)
    for model in models:
        model_rows = [(row["lead_day"], row["forecasts"].get(model["id"])) for row in rows]
        model_rows = [(lead, value) for lead, value in model_rows if value is not None]
        if model_rows:
            model_leads, values = map(np.asarray, zip(*model_rows))
            ax.plot(model_leads, values, color=MODEL_COLORS[model["id"]], marker="o", markersize=4.5,
                    lw=1.7, alpha=.92, label=model["label"])
    ax.set_xticks(LEAD_DAYS, ["Day 1 · +24h", "Day 3 · +72h", "Day 5 · +120h"])
    ax.set_xlabel("Matched valid time")
    ax.set_ylabel(f"{label} ({unit})")
    ax.grid(alpha=.22)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8.5, frameon=False, ncols=2)
    init = pd.Timestamp(run["initialization_utc"])
    fig.suptitle(f"{city.name} · {label} · init {init:%d %b %Y, 00 UTC}", fontsize=15.5, fontweight="bold", color="#132a35")
    detail = "exact valid-time values" if variable == "temperature" else "accumulated from initialization through each valid endpoint"
    fig.text(.5, .055, f"Forecast and Open-Meteo ground truth matched at each lead · {detail}", ha="center", fontsize=8.5, color="#53636b")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"matched_leads": [int(value) for value in leads]}


def render_validation(archive: dict, cfg, openmeteo, stage: Path) -> dict:
    records = _validation_records(archive, cfg, openmeteo)
    models = _model_catalog(archive)
    validation = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "truth_source": "Open-Meteo hourly temperature_2m and precipitation",
        "temperature_definition": "Forecast 2 m temperature at the exact valid time, matched to Open-Meteo hourly temperature.",
        "precipitation_definition": "Forecast precipitation accumulated from initialization through each valid endpoint, matched to the sum of Open-Meteo hourly precipitation over the same interval.",
        "cities": {},
    }
    for city in cfg.cities:
        city_info = {"latitude": city.lat, "longitude": city.lon, "images": {}, "summary": {}, "timeseries": {}}
        for variable in ("temperature", "precipitation"):
            filename = f"{city.name.lower().replace(' ', '-')}-{variable}.png"
            relative = Path("assets") / "validation" / filename
            summary = _plot_validation(records[city.name][variable], city, variable, models, stage / relative)
            city_info["images"][variable] = {"path": relative.as_posix(), "alt": f"{city.name} {variable} forecast verification against Open-Meteo observations"}
            city_info["summary"][variable] = summary
        for run in archive["runs"]:
            run_info = {}
            for variable in ("temperature", "precipitation"):
                filename = f"{city.name.lower().replace(' ', '-')}-{variable}.png"
                relative = Path("assets") / "validation" / "timeseries" / run["id"] / filename
                summary = _plot_matched_timeseries(records[city.name][variable], city, variable, run, models, stage / relative)
                run_info[variable] = {
                    "path": relative.as_posix(),
                    "alt": f"{city.name} {variable} forecast and Open-Meteo ground truth for initialization {run['id']}",
                    **summary,
                }
            city_info["timeseries"][run["id"]] = run_info
        validation["cities"][city.name] = city_info
    return validation


def render_online_combination(cfg) -> dict:
    """Publish a compact, city-level temperature blend from the online learner."""
    from realtime_dash.combine import backtest  # type: ignore

    models = ("weathernext2", "gencast", "aifs", "gefs", "ifs_ens")
    cities = {}
    for city in cfg.cities:
        result = backtest.run(cfg, city.name, "t2m", models, window_days=10)
        if not result.get("ok"):
            continue
        forward = backtest.live_forecast(cfg, city.name, "t2m", result["present"], result["final_weights"])
        points = []
        for valid_time, row in forward.iterrows():
            expert_values = {model: float(row[model]) for model in result["present"] if model in row and pd.notna(row[model])}
            if not expert_values or pd.isna(row.get("combined")):
                continue
            points.append({
                "valid_time_utc": utc_text(valid_time),
                "combined_c": float(row["combined"]),
                "low_c": min(expert_values.values()),
                "high_c": max(expert_values.values()),
                "experts_c": expert_values,
            })
        cities[city.name] = {
            "method": result["best"],
            "backtest_steps": int(len(result["y"])),
            "backtest_rmse_c": float(result["method_rmse"][result["best"]]),
            "uniform_rmse_c": float(result["method_rmse"]["uniform"]),
            "weights": {model: float(weight) for model, weight in result["final_weights"].items()},
            "points": points,
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "variable": "2 m temperature",
        "units": "degree_Celsius",
        "definition": "Experimental online convex combination of city-level source-model forecasts. Weights are learned causally from the most recent 10-day matched history; the shaded range is the spread of contributing source-model forecasts.",
        "cities": cities,
    }


def _normalized_weights(raw: dict[str, float], available: list[str]) -> dict[str, float]:
    """Restrict learned weights to the models in a run and preserve a convex blend."""
    clean = {model: max(0.0, float(raw.get(model, 0.0))) for model in available}
    total = sum(clean.values())
    if total <= 0 and available:
        return {model: 1.0 / len(available) for model in available}
    return {model: value / total for model, value in clean.items()}


def _learned_city_weights(cfg, city_name: str, variable: str, available: list[str]) -> tuple[dict[str, float], str]:
    from realtime_dash.combine import backtest  # type: ignore

    try:
        result = backtest.run(cfg, city_name, variable, tuple(ALL_MODEL_IDS), window_days=10)
    except Exception as exc:  # noqa: BLE001 - the public weather card has a uniform fallback
        print(f"[{city_name}] {variable} learner unavailable: {exc}", file=sys.stderr, flush=True)
        result = {"ok": False}
    if result.get("ok"):
        return _normalized_weights(result["final_weights"], available), str(result["best"])
    return _normalized_weights({}, available), "uniform"


def _daily_city_series(series: xr.Dataset, city, init: pd.Timestamp) -> dict[int, dict[str, float]]:
    """Reduce a native-step model series to daily high/low and 24-hour rain."""
    point = series.sel(lat=city.lat, lon=city.lon, method="nearest")
    times = pd.to_datetime(point["valid_time"].values).tz_localize(None)
    previous = np.concatenate([
        np.asarray([np.datetime64(init, "ns")]),
        times.values[:-1].astype("datetime64[ns]"),
    ])
    hours = (times.values.astype("datetime64[ns]") - previous) / np.timedelta64(1, "h")
    out: dict[int, dict[str, float]] = {}
    for day in DAILY_LEAD_DAYS:
        start = init + pd.Timedelta(days=day - 1)
        end = init + pd.Timedelta(days=day)
        chosen = np.flatnonzero((times > start) & (times <= end))
        if not len(chosen):
            continue
        temperatures = np.asarray(point["t2m_C"].isel(valid_time=chosen).values, dtype=float)
        rates = np.asarray(point["precip_mmday"].isel(valid_time=chosen).values, dtype=float)
        rain = float(np.sum(np.clip(rates, 0, None) * np.clip(hours[chosen], 0, None) / 24.0))
        out[day] = {
            "high_c": float(np.nanmax(temperatures)),
            "low_c": float(np.nanmin(temperatures)),
            "mean_c": float(np.nanmean(temperatures)),
            "precip_mm": max(0.0, rain),
        }
    return out


def _weather_symbol(rain_mm: float) -> tuple[str, str]:
    if rain_mm >= 20:
        return "🌧️", "Heavy rain"
    if rain_mm >= 5:
        return "🌦️", "Rain"
    if rain_mm >= 0.2:
        return "🌤️", "Light rain"
    return "☀️", "Mostly dry"


def render_weather_forecasts(archive: dict, cfg, india_load) -> dict:
    """Create a five-day city product from available source models and online weights."""
    runs = {}
    learned = {
        city.name: {
            "temperature": _learned_city_weights(cfg, city.name, "t2m", list(ALL_MODEL_IDS)),
            "precipitation": _learned_city_weights(cfg, city.name, "precip", list(ALL_MODEL_IDS)),
        }
        for city in cfg.cities
    }
    for run in archive["runs"]:
        init = pd.Timestamp(run["initialization_utc"]).tz_localize(None)
        series_by_model = {}
        for model in (item["id"] for item in run.get("models", [])):
            try:
                with india_load.load_india_series_cached(
                    model, cfg, init, horizon_days=max(DAILY_LEAD_DAYS), max_members=8,
                ) as opened:
                    series_by_model[model] = opened.load()
            except Exception as exc:  # noqa: BLE001 - a weather card can use the remaining experts
                print(f"[{run['id']}] weather series unavailable for {model}: {exc}", file=sys.stderr, flush=True)
        cities = {}
        for city in cfg.cities:
            expert_days = {
                model: _daily_city_series(series, city, init)
                for model, series in series_by_model.items()
            }
            available = [model for model in ALL_MODEL_IDS if len(expert_days.get(model, {})) == len(DAILY_LEAD_DAYS)]
            raw_temp_weights, temp_method = learned[city.name]["temperature"]
            raw_rain_weights, rain_method = learned[city.name]["precipitation"]
            temp_weights = _normalized_weights(raw_temp_weights, available)
            rain_weights = _normalized_weights(raw_rain_weights, available)
            days = []
            for day in DAILY_LEAD_DAYS:
                present = [model for model in available if day in expert_days[model]]
                if not present:
                    continue
                tw = _normalized_weights(temp_weights, present)
                rw = _normalized_weights(rain_weights, present)
                blend = {
                    key: sum(tw[model] * expert_days[model][day][key] for model in present)
                    for key in ("high_c", "low_c", "mean_c")
                }
                blend["precip_mm"] = sum(rw[model] * expert_days[model][day]["precip_mm"] for model in present)
                symbol, condition = _weather_symbol(blend["precip_mm"])
                days.append({
                    "day": day,
                    "valid_date": (init + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                    **{key: round(float(value), 2) for key, value in blend.items()},
                    "symbol": symbol,
                    "condition": condition,
                    "experts": {
                        model: {key: round(float(value), 2) for key, value in expert_days[model][day].items()}
                        for model in present
                    },
                })
            cities[city.name] = {
                "latitude": city.lat,
                "longitude": city.lon,
                "temperature_method": temp_method,
                "precipitation_method": rain_method,
                "temperature_weights": temp_weights,
                "precipitation_weights": rain_weights,
                "available_models": available,
                "days": days,
            }
        runs[run["id"]] = {
            "initialization_utc": run["initialization_utc"],
            "status": run.get("status", "complete"),
            "available_models": run.get("available_models", [model["id"] for model in run["models"]]),
            "missing_models": run.get("missing_models", []),
            "cities": cities,
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "temperature_definition": "Online-weighted daily 2 m temperature high and low from native forecast steps.",
        "precipitation_definition": "Online-weighted precipitation accumulated within each 24-hour forecast day.",
        "runs": runs,
    }


def run_sections(run: dict, renderer) -> str:
    manifest = dict(run)
    html = renderer._view_sections(manifest)
    return re.sub(
        r'<section class="forecast-view"',
        f'<section class="forecast-view" data-init="{run["id"]}"',
        html,
    )


def range_sections(run: dict) -> str:
    """HTML views for the daily extrema layers produced alongside base maps."""
    if any("id" not in model for model in run["models"]) or not has_daily_temperature_ranges(run):
        return ""
    tag = run["id"]
    sections = []
    for variable, label, kind in RANGE_VARIABLES:
        for day in LEAD_DAYS:
            end = pd.Timestamp(run["initialization_utc"]) + pd.Timedelta(days=day)
            start = end - pd.Timedelta(days=1)
            cards = []
            for model in run["models"]:
                path = f"assets/forecasts/{tag}/{model['id']}/{variable}_day{day}.png"
                cards.append(
                    f'<article class="model-card"><a class="image-link" href="{path}"><img loading="lazy" src="{path}" alt="{model["label"]} {label.lower()} over India for forecast day {day}"></a><div class="card-copy"><h3>{model["label"]}</h3><p>{"Maximum" if kind == "maximum" else "Minimum"} 2 m temperature from native forecast steps.</p><a class="download" href="{path}" download>Download PNG</a></div></article>'
                )
            comparison = f"assets/forecasts/{tag}/comparisons/{variable}_day{day}.png"
            sections.append(
                f'<section class="forecast-view" data-init="{tag}" data-variable="{variable}" data-day="{day}" hidden>'
                f'<div class="section-title"><div><p class="kicker">Derived temperature range · Day {day}</p><h2>{label}</h2></div>'
                f'<p>Derived from every native forecast time step in the 24-hour window {start:%d %b %H:%M}–{end:%d %b %H:%M} UTC.</p></div>'
                f'<figure class="comparison"><a class="image-link" href="{comparison}"><img loading="lazy" src="{comparison}" alt="Six-model comparison of {label.lower()} over India for forecast day {day}"></a><figcaption><span>Six-model comparison · common temperature scale</span><a class="download" href="{comparison}" download>Download comparison PNG</a></figcaption></figure><div class="model-grid">{"".join(cards)}</div></section>'
            )
    return "\n".join(sections)


LEGACY_ARCHIVE_JS = r"""
(() => {
  const variableButtons = [...document.querySelectorAll("[data-variable-button]")];
  const dayButtons = [...document.querySelectorAll("[data-day-button]")];
  const validationCityButtons = [...document.querySelectorAll("[data-validation-city]")];
  const validationVariableButtons = [...document.querySelectorAll("[data-validation-variable]")];
  const validationImage = document.querySelector("#validation-image");
  const validationSummary = document.querySelector("#validation-summary");
  const matchInitSelect = document.querySelector("#match-init-select");
  const matchVariableButtons = [...document.querySelectorAll("[data-match-variable]")];
  const matchImage = document.querySelector("#match-image");
  const runSelect = document.querySelector("#run-select");
  const runSummary = document.querySelector("#run-summary");
  const views = [...document.querySelectorAll(".forecast-view")];
  const siteData = JSON.parse(document.querySelector("#archive-data").textContent);
  const runs = siteData.runs;
  const validation = siteData.validation;
  const mapModelButtons = [...document.querySelectorAll("[data-map-model]")];
  const canvas = document.querySelector("#forecast-canvas");
  const mapTitle = document.querySelector("#map-title");
  const mapDescription = document.querySelector("#map-description");
  const mapReadout = document.querySelector("#map-readout");
  const cityReadout = document.querySelector("#city-readout");
  const combination = siteData.combination || { cities: {} };
  const combinationCityButtons = [...document.querySelectorAll("[data-combination-city]")];
  const combinationTitle = document.querySelector("#combination-title");
  const combinationSummary = document.querySelector("#combination-summary");
  const combinationChart = document.querySelector("#combination-chart");
  const combinationWeights = document.querySelector("#combination-weights");
  const params = new URLSearchParams(window.location.search);
  const allowedVariables = new Set(["temperature", "temperature_high", "temperature_low", "precipitation"]);
  const allowedDays = new Set(["1", "3", "5"]);
  const allowedInits = new Set(runs.map((run) => run.id));
  let variable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let day = allowedDays.has(params.get("day")) ? params.get("day") : "1";
  let init = allowedInits.has(params.get("init")) ? params.get("init") : runs[0].id;
  let validationCity = Object.keys(validation.cities).includes(params.get("city")) ? params.get("city") : Object.keys(validation.cities)[0];
  let validationVariable = allowedVariables.has(params.get("validation")) ? params.get("validation") : "temperature";
  let matchInit = allowedInits.has(params.get("match_init")) ? params.get("match_init") : runs[0].id;
  let matchVariable = allowedVariables.has(params.get("match_variable")) ? params.get("match_variable") : "precipitation";
  let combinationCity = Object.keys(combination.cities).includes(params.get("combination_city")) ? params.get("combination_city") : Object.keys(combination.cities)[0];
  let mapModel = params.get("model") || mapModelButtons[0]?.dataset.mapModel;
  let payload = null;
  let view = { scale: 1, x: 0, y: 0 };
  let drag = null;

  function render(updateUrl = true) {
    const active = runs.find((run) => run.id === init);
    variableButtons.forEach((button) => {
      button.disabled = false;
      button.title = "";
      button.setAttribute("aria-pressed", String(button.dataset.variableButton === variable));
    });
    dayButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.dayButton === day)));
    mapModelButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mapModel === mapModel)));
    runSelect.value = init;
    runSummary.textContent = `Initialized ${new Date(active.initialization_utc).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false })} UTC · 6 experts · 5-day horizon`;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("init", init);
      next.searchParams.set("variable", variable);
      next.searchParams.set("day", day);
      next.searchParams.set("model", mapModel);
      history.replaceState(null, "", next);
    }
    loadMap(active);
  }

  function color(value) {
    if (variable === "precipitation") {
      const t = Math.max(0, Math.min(1, value / 120));
      return `hsl(${205 - t * 150} 78% ${92 - t * 47}%)`;
    }
    const t = Math.max(0, Math.min(1, (value - 5) / 40));
    return `hsl(${235 - t * 235} 82% ${35 + 27 * t}%)`;
  }

  function rgb(value) {
    const hsl = color(value).match(/([\d.]+)/g).map(Number);
    let [h, s, l] = hsl; h /= 360; s /= 100; l /= 100;
    const hue = (n) => { const k = (n + h * 12) % 12; return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return [Math.round(255 * hue(0)), Math.round(255 * hue(8)), Math.round(255 * hue(4))];
  }

  async function loadMap(active) {
    if (!canvas || !active.grid_metadata?.shape) return;
    const url = `assets/map_data/${init}/${mapModel}.bin`;
    mapReadout.textContent = "Loading compact grid…";
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(response.statusText);
      payload = new Uint16Array(await response.arrayBuffer());
      drawMap(active);
    } catch (error) {
      mapReadout.textContent = "Grid unavailable for this selection.";
      console.error(error);
    }
  }

  function drawMap(active) {
    if (!payload || !canvas) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(640, Math.round(rect.width * devicePixelRatio));
    const height = Math.max(420, Math.round(rect.height * devicePixelRatio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const metadata = active.grid_metadata;
    const [nLead, nLat, nLon] = metadata.shape;
    const variableIndex = metadata.variables.indexOf(variable);
    const dayIndex = metadata.lead_days.indexOf(Number(day));
    const count = nLead * nLat * nLon;
    const start = variableIndex * count + dayIndex * nLat * nLon;
    const image = ctx.createImageData(nLon, nLat);
    for (let y = 0; y < nLat; y++) for (let x = 0; x < nLon; x++) {
      const encoded = payload[start + y * nLon + x];
      const offset = ((nLat - 1 - y) * nLon + x) * 4;
      if (encoded === 65535) { image.data[offset + 3] = 0; continue; }
      const value = variable === "precipitation" ? encoded / 10 : (encoded - 5000) / 100;
      const pixel = rgb(value);
      image.data[offset] = pixel[0]; image.data[offset + 1] = pixel[1]; image.data[offset + 2] = pixel[2]; image.data[offset + 3] = 255;
    }
    const raster = document.createElement("canvas"); raster.width = nLon; raster.height = nLat; raster.getContext("2d").putImageData(image, 0, 0);
    ctx.fillStyle = "#071923"; ctx.fillRect(0, 0, width, height);
    const scale = view.scale; const x0 = view.x; const y0 = view.y;
    ctx.save(); ctx.translate(x0, y0); ctx.scale(scale, scale); ctx.imageSmoothingEnabled = true;
    ctx.drawImage(raster, 0, 0, width, height);
    ctx.strokeStyle = "rgba(255,255,255,.22)"; ctx.lineWidth = 1 / scale;
    for (let fraction = .2; fraction < 1; fraction += .2) { ctx.beginPath(); ctx.moveTo(width * fraction, 0); ctx.lineTo(width * fraction, height); ctx.moveTo(0, height * fraction); ctx.lineTo(width, height * fraction); ctx.stroke(); }
    const cities = Object.entries(validation.cities);
    cities.forEach(([name, city]) => {
      const x = width * (city.longitude - metadata.bounding_box.lon_min) / (metadata.bounding_box.lon_max - metadata.bounding_box.lon_min);
      const y = height * (metadata.bounding_box.lat_max - city.latitude) / (metadata.bounding_box.lat_max - metadata.bounding_box.lat_min);
      ctx.beginPath(); ctx.arc(x, y, 7 / scale, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#f2553d"; ctx.lineWidth = 3 / scale; ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = `${13 / scale}px Inter, sans-serif`; ctx.fillText(name, x + 10 / scale, y - 9 / scale);
    });
    ctx.restore();
    const label = variable.replace("temperature_high", "daily high").replace("temperature_low", "daily low").replace("precipitation", "rain accumulation");
    mapTitle.textContent = `${label} · Day ${day}`;
    mapDescription.textContent = `${mapModelButtons.find((button) => button.dataset.mapModel === mapModel)?.textContent || mapModel} · ${metadata.shape[1]} × ${metadata.shape[2]} grid · drag / scroll to navigate`;
    mapReadout.textContent = `T+${Number(day) * 24} h · ${metadata.bounding_box.lat_min}–${metadata.bounding_box.lat_max}°N · ${metadata.bounding_box.lon_min}–${metadata.bounding_box.lon_max}°E`;
  }

  function renderValidation(updateUrl = true) {
    validationCityButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.validationCity === validationCity)));
    validationVariableButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.validationVariable === validationVariable)));
    const active = validation.cities[validationCity];
    const image = active.images[validationVariable];
    const points = active.summary[validationVariable].matched_points;
    validationImage.src = image.path;
    validationImage.alt = image.alt;
    validationSummary.textContent = `${validationCity} · ${points} matched forecast–observation pairs per model · Open-Meteo ground truth`;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("city", validationCity);
      next.searchParams.set("validation", validationVariable);
      history.replaceState(null, "", next);
    }
    renderMatchedTimeseries(updateUrl);
  }

  function renderMatchedTimeseries(updateUrl = true) {
    matchVariableButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.matchVariable === matchVariable)));
    const image = validation.cities[validationCity].timeseries[matchInit][matchVariable];
    matchInitSelect.value = matchInit;
    matchImage.src = image.path;
    matchImage.alt = image.alt;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("match_init", matchInit);
      next.searchParams.set("match_variable", matchVariable);
      history.replaceState(null, "", next);
    }
  }

  function renderCombination(updateUrl = true) {
    if (!combinationCity || !combination.cities[combinationCity]) return;
    const active = combination.cities[combinationCity];
    combinationCityButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.combinationCity === combinationCity)));
    combinationTitle.textContent = `${combinationCity} · 2 m temperature`;
    combinationSummary.textContent = `${active.method.toUpperCase()} learner · ${active.backtest_steps} matched steps · blend RMSE ${active.backtest_rmse_c.toFixed(2)} °C · uniform ${active.uniform_rmse_c.toFixed(2)} °C`;
    const points = active.points;
    if (!points.length) {
      combinationChart.innerHTML = '<p class="tag">No forward points are currently available.</p>';
    } else {
      const width = 900, height = 290, pad = { left: 46, right: 18, top: 20, bottom: 42 };
      const values = points.flatMap((point) => [point.low_c, point.high_c, point.combined_c]);
      const lo = Math.floor((Math.min(...values) - 1) / 2) * 2;
      const hi = Math.ceil((Math.max(...values) + 1) / 2) * 2;
      const x = (i) => pad.left + (i / Math.max(points.length - 1, 1)) * (width - pad.left - pad.right);
      const y = (value) => pad.top + (hi - value) / Math.max(hi - lo, 1) * (height - pad.top - pad.bottom);
      const line = points.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point.combined_c).toFixed(1)}`).join(" ");
      const upper = points.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point.high_c).toFixed(1)}`).join(" ");
      const lower = [...points].reverse().map((point, reverseIndex) => `L${x(points.length - 1 - reverseIndex).toFixed(1)},${y(point.low_c).toFixed(1)}`).join(" ");
      const grid = [lo, (lo + hi) / 2, hi].map((value) => `<g><line x1="${pad.left}" x2="${width - pad.right}" y1="${y(value)}" y2="${y(value)}"/><text x="${pad.left - 8}" y="${y(value) + 4}" text-anchor="end">${value.toFixed(0)}°</text></g>`).join("");
      const labels = points.map((point, index) => `<text x="${x(index)}" y="${height - 16}" text-anchor="middle">${new Date(point.valid_time_utc).toLocaleDateString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short" })}</text>`).join("");
      combinationChart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><g class="combo-grid">${grid}</g><path class="combo-range" d="${upper} ${lower} Z"/><path class="combo-line" d="${line}"/>${labels}</svg><p class="chart-key"><span></span>AdaWeather online blend <i></i>source-model spread</p>`;
    }
    combinationWeights.innerHTML = Object.entries(active.weights).map(([model, weight]) => `<div><span>${model.replace("weathernext2", "WeatherNext 2").replace("ifs_ens", "IFS-ENS").toUpperCase()}</span><strong>${(weight * 100).toFixed(1)}%</strong></div>`).join("");
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("combination_city", combinationCity);
      history.replaceState(null, "", next);
    }
  }

  variableButtons.forEach((button) => button.addEventListener("click", () => { variable = button.dataset.variableButton; render(); }));
  dayButtons.forEach((button) => button.addEventListener("click", () => { day = button.dataset.dayButton; render(); }));
  runSelect.addEventListener("change", () => { init = runSelect.value; render(); });
  validationCityButtons.forEach((button) => button.addEventListener("click", () => { validationCity = button.dataset.validationCity; renderValidation(); }));
  validationVariableButtons.forEach((button) => button.addEventListener("click", () => { validationVariable = button.dataset.validationVariable; renderValidation(); }));
  matchInitSelect.addEventListener("change", () => { matchInit = matchInitSelect.value; renderMatchedTimeseries(); });
  matchVariableButtons.forEach((button) => button.addEventListener("click", () => { matchVariable = button.dataset.matchVariable; renderMatchedTimeseries(); }));
  combinationCityButtons.forEach((button) => button.addEventListener("click", () => { combinationCity = button.dataset.combinationCity; renderCombination(); }));
  mapModelButtons.forEach((button) => button.addEventListener("click", () => { mapModel = button.dataset.mapModel; view = { scale: 1, x: 0, y: 0 }; render(); }));
  document.querySelector("#map-reset")?.addEventListener("click", () => { view = { scale: 1, x: 0, y: 0 }; drawMap(runs.find((run) => run.id === init)); });
  canvas?.addEventListener("pointerdown", (event) => { drag = { x: event.clientX, y: event.clientY, moved: false }; canvas.setPointerCapture(event.pointerId); });
  canvas?.addEventListener("pointermove", (event) => { if (!drag) return; const dx = (event.clientX - drag.x) * devicePixelRatio; const dy = (event.clientY - drag.y) * devicePixelRatio; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true; view.x += dx; view.y += dy; drag.x = event.clientX; drag.y = event.clientY; drawMap(runs.find((run) => run.id === init)); });
  canvas?.addEventListener("pointerup", (event) => {
    if (!drag?.moved) {
      const active = runs.find((run) => run.id === init); const meta = active.grid_metadata; const box = canvas.getBoundingClientRect();
      const px = (event.clientX - box.left) * devicePixelRatio; const py = (event.clientY - box.top) * devicePixelRatio;
      const gx = (px - view.x) / view.scale / canvas.width; const gy = (py - view.y) / view.scale / canvas.height;
      const lon = meta.bounding_box.lon_min + gx * (meta.bounding_box.lon_max - meta.bounding_box.lon_min); const lat = meta.bounding_box.lat_max - gy * (meta.bounding_box.lat_max - meta.bounding_box.lat_min);
      const nearest = Object.entries(validation.cities).map(([name, city]) => [name, Math.hypot((city.longitude - lon) * .9, city.latitude - lat)]).sort((a, b) => a[1] - b[1])[0];
      if (nearest && nearest[1] < 1.5) { validationCity = nearest[0]; combinationCity = nearest[0]; renderValidation(); renderCombination(); cityReadout.innerHTML = `<dt>${nearest[0]}</dt><dd>Matched validation and online-combination panel selected.</dd>`; document.querySelector("#combination").scrollIntoView({ behavior: "smooth", block: "start" }); }
    }
    drag = null;
  });
  canvas?.addEventListener("wheel", (event) => { event.preventDefault(); const factor = event.deltaY < 0 ? 1.15 : .87; view.scale = Math.max(1, Math.min(4, view.scale * factor)); drawMap(runs.find((run) => run.id === init)); }, { passive: false });

  const pinLocations = [
    ["Delhi", "31%", "38%"], ["Mumbai", "23%", "56%"],
    ["Bengaluru", "35%", "70%"], ["Kolkata", "69%", "49%"],
  ];
  document.querySelectorAll("#forecast-views .comparison").forEach((figure) => {
    const map = figure.querySelector(".image-link");
    if (!map) return;
    map.classList.add("map-canvas");
    figure.classList.add("map-figure");
    const controls = document.createElement("div");
    controls.className = "map-tools";
    controls.innerHTML = '<button type="button" data-map-zoom="in" aria-label="Enlarge map">+</button><button type="button" data-map-zoom="out" aria-label="Reduce map">−</button><button type="button" data-map-zoom="reset" aria-label="Reset map zoom">⌂</button>';
    figure.append(controls);
    const pins = document.createElement("div");
    pins.className = "map-pins";
    pinLocations.forEach(([city, left, top]) => {
      const pin = document.createElement("button");
      pin.type = "button"; pin.className = "map-pin"; pin.dataset.mapCity = city;
      pin.style.left = left; pin.style.top = top; pin.textContent = city;
      pins.append(pin);
    });
    figure.append(pins);
    let zoom = 1;
    controls.addEventListener("click", (event) => {
      const action = event.target.closest("button")?.dataset.mapZoom;
      if (!action) return;
      zoom = action === "in" ? Math.min(1.7, zoom + .15) : action === "out" ? Math.max(1, zoom - .15) : 1;
      map.querySelector("img").style.transform = `scale(${zoom})`;
    });
  });
  document.querySelectorAll("[data-map-city]").forEach((pin) => pin.addEventListener("click", () => {
    validationCity = pin.dataset.mapCity;
    renderValidation();
    document.querySelector("#validation").scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  render(false);
  renderValidation(false);
  renderCombination(false);
})();
"""


ARCHIVE_JS = r"""
(() => {
  "use strict";
  const dataNode = document.querySelector("#site-data");
  if (!dataNode) return;
  const site = JSON.parse(dataNode.textContent);
  const archive = site.archive;
  const validation = site.validation;
  const weather = site.weather;
  const runs = archive.runs;
  const params = new URLSearchParams(location.search);
  const q = (selector) => document.querySelector(selector);
  const qa = (selector) => [...document.querySelectorAll(selector)];
  const allowedTabs = new Set(["weather", "maps", "validation", "method"]);
  const allowedVariables = new Set(["temperature", "temperature_high", "temperature_low", "precipitation"]);
  const allowedDays = new Set(["1", "3", "5"]);
  const runIds = new Set(runs.map((run) => run.id));
  const cityNames = Object.keys(validation.cities);
  let tab = allowedTabs.has(params.get("tab")) ? params.get("tab") : "weather";
  let init = runIds.has(params.get("init")) ? params.get("init") : runs[0].id;
  let city = cityNames.includes(params.get("city")) ? params.get("city") : cityNames[0];
  let weatherVariable = params.get("weather") === "precipitation" ? "precipitation" : "temperature";
  let mapVariable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let mapDay = allowedDays.has(params.get("day")) ? params.get("day") : "1";
  let mapModel = params.get("model") || runs[0].available_models?.[0] || runs[0].models[0].id;
  let validationVariable = params.get("validation") === "precipitation" ? "precipitation" : "temperature";
  let matchVariable = params.get("match_variable") === "temperature" ? "temperature" : "precipitation";
  let matchInit = runIds.has(params.get("match_init")) ? params.get("match_init") : runs[0].id;
  let payload = null;
  let coastlines = [];
  let coastlinePromise = null;
  let mapRequest = 0;
  let view = { scale: 1, x: 0, y: 0 };
  let drag = null;

  function setUrl() {
    const next = new URL(location.href);
    const values = { tab, init, city, weather: weatherVariable, variable: mapVariable, day: mapDay,
      model: mapModel, validation: validationVariable, match_init: matchInit, match_variable: matchVariable };
    Object.entries(values).forEach(([key, value]) => next.searchParams.set(key, value));
    history.replaceState(null, "", next);
  }

  function selectButton(selector, value, key) {
    qa(selector).forEach((button) => button.setAttribute("aria-pressed", String(button.dataset[key] === value)));
  }

  function activateTab(next, update = true) {
    tab = allowedTabs.has(next) ? next : "weather";
    qa("[data-tab]").forEach((button) => {
      const active = button.dataset.tab === tab;
      button.setAttribute("aria-selected", String(active));
      button.classList.toggle("is-active", active);
    });
    qa("[data-panel]").forEach((panel) => { panel.hidden = panel.dataset.panel !== tab; });
    if (tab === "maps") requestAnimationFrame(() => drawMap(activeRun()));
    if (update) setUrl();
  }

  function activeRun() { return runs.find((run) => run.id === init) || runs[0]; }
  function runModels(run = activeRun()) { return run.available_models || run.models.map((model) => model.id); }
  function modelLabel(model) {
    const item = site.models.find((candidate) => candidate.id === model);
    return item?.label || model;
  }
  function formatInit(value) {
    return new Date(value).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short",
      year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) + " UTC";
  }

  function renderRun() {
    const run = activeRun();
    q("#init-select").value = init;
    q("#run-status").textContent = `${formatInit(run.initialization_utc)} · ${runModels(run).length} of ${site.models.length} models`;
    q("#availability-note").textContent = run.status === "partial"
      ? `Partial run. Waiting for: ${(run.missing_models || []).map(modelLabel).join(", ")}.`
      : "All configured models are available for this initialization.";
    const available = runModels(run);
    if (!available.includes(mapModel)) mapModel = available[0];
    qa("[data-map-model]").forEach((button) => {
      const enabled = available.includes(button.dataset.mapModel);
      button.disabled = !enabled;
      button.title = enabled ? "" : "Not available for this initialization";
      button.setAttribute("aria-pressed", String(button.dataset.mapModel === mapModel));
    });
    renderWeather();
    loadMap();
    setUrl();
  }

  function path(points) {
    if (!points.length) return "";
    return points.map((point, index) => `${index ? "L" : "M"}${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" ");
  }

  function weatherChart(days) {
    const width = 920, height = 260, pad = { l: 45, r: 20, t: 22, b: 42 };
    const value = (day) => weatherVariable === "temperature" ? day.mean_c : day.precip_mm;
    const values = days.map(value);
    if (!values.length) return '<p class="empty-state">No five-day city forecast is available for this run.</p>';
    const low = weatherVariable === "temperature" ? Math.floor(Math.min(...days.map((day) => day.low_c)) - 2) : 0;
    const high = weatherVariable === "temperature" ? Math.ceil(Math.max(...days.map((day) => day.high_c)) + 2) : Math.max(5, Math.ceil(Math.max(...values) * 1.25));
    const x = (index) => pad.l + index * (width - pad.l - pad.r) / Math.max(days.length - 1, 1);
    const y = (number) => pad.t + (high - number) * (height - pad.t - pad.b) / Math.max(high - low, 1);
    const line = path(days.map((day, index) => [x(index), y(value(day))]));
    const area = `${line} L${x(days.length - 1)},${height - pad.b} L${x(0)},${height - pad.b} Z`;
    const gridValues = [low, (low + high) / 2, high];
    const grid = gridValues.map((number) => `<g><line x1="${pad.l}" x2="${width - pad.r}" y1="${y(number)}" y2="${y(number)}"/><text x="${pad.l - 8}" y="${y(number) + 4}" text-anchor="end">${number.toFixed(weatherVariable === "temperature" ? 0 : 1)}</text></g>`).join("");
    const dots = days.map((day, index) => `<g><circle cx="${x(index)}" cy="${y(value(day))}" r="4"/><text x="${x(index)}" y="${y(value(day)) - 12}" text-anchor="middle">${value(day).toFixed(1)}${weatherVariable === "temperature" ? "°" : " mm"}</text><text class="date" x="${x(index)}" y="${height - 15}" text-anchor="middle">${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { weekday: "short", day: "numeric" })}</text></g>`).join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Five-day ${weatherVariable} forecast"><g class="chart-grid">${grid}</g><path class="weather-area" d="${area}"/><path class="weather-line" d="${line}"/><g class="weather-points">${dots}</g></svg>`;
  }

  function renderWeather() {
    q("#city-select").value = city;
    selectButton("[data-weather-variable]", weatherVariable, "weatherVariable");
    const run = weather.runs[init];
    const item = run?.cities?.[city];
    const days = item?.days || [];
    q("#weather-location").textContent = city;
    q("#weather-meta").textContent = item
      ? `Initialized ${formatInit(run.initialization_utc)} · ${item.available_models.length} contributing model${item.available_models.length === 1 ? "" : "s"}`
      : "Forecast unavailable for this selection";
    const first = days[0];
    q("#weather-now").innerHTML = first
      ? `<span aria-hidden="true">${first.symbol}</span><strong>${Math.round(first.mean_c)}°C</strong><small>${first.condition}<br>${first.precip_mm.toFixed(1)} mm in 24 h</small>`
      : "";
    q("#weather-chart").innerHTML = weatherChart(days);
    q("#daily-cards").innerHTML = days.map((day, index) => `<article class="day-card${index === 0 ? " is-first" : ""}"><strong>${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { weekday: "short" })}</strong><time>${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short" })}</time><span class="weather-icon" aria-label="${day.condition}">${day.symbol}</span><p><b>${Math.round(day.high_c)}°</b> <span>${Math.round(day.low_c)}°</span></p><small>${day.precip_mm.toFixed(1)} mm</small></article>`).join("");
    q("#blend-note").textContent = item
      ? `Temperature: ${item.temperature_method} weights · rainfall: ${item.precipitation_method} weights. Rainfall is a 24-hour accumulation, not a probability.`
      : "";
  }

  function mapColor(number) {
    if (mapVariable === "precipitation") {
      const t = Math.max(0, Math.min(1, number / 120));
      return [225 - 185 * t, 241 - 80 * t, 248 - 25 * t];
    }
    const t = Math.max(0, Math.min(1, (number - 5) / 40));
    return [45 + 205 * t, 110 + 70 * (1 - Math.abs(t - .5) * 2), 190 - 145 * t];
  }

  function loadCoastlines() {
    if (!coastlinePromise) {
      coastlinePromise = fetch("assets/coastlines.json")
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then((data) => { coastlines = Array.isArray(data.lines) ? data.lines : []; })
        .catch((error) => { console.warn("Coastline overlay unavailable.", error); });
    }
    return coastlinePromise;
  }

  async function loadMap() {
    const run = activeRun();
    if (!q("#forecast-canvas") || !run.grid_metadata?.shape || !runModels(run).includes(mapModel)) return;
    const request = ++mapRequest;
    q("#map-readout").textContent = "Loading map…";
    try {
      const [response] = await Promise.all([
        fetch(`assets/map_data/${init}/${mapModel}.bin`),
        loadCoastlines(),
      ]);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const next = new Uint16Array(await response.arrayBuffer());
      if (request !== mapRequest) return;
      payload = next;
      drawMap(run);
    } catch (error) {
      if (request === mapRequest) q("#map-readout").textContent = "Map data unavailable.";
      console.error(error);
    }
  }

  function drawCoastlines(ctx, meta, width, height) {
    const bounds = meta.bounding_box;
    const x = (longitude) => width * (longitude - bounds.lon_min) / (bounds.lon_max - bounds.lon_min);
    const y = (latitude) => height * (bounds.lat_max - latitude) / (bounds.lat_max - bounds.lat_min);
    ctx.strokeStyle = "rgba(19, 44, 57, .9)";
    ctx.lineWidth = 1.35 / view.scale;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    coastlines.forEach((line) => {
      if (line.length < 2) return;
      ctx.beginPath();
      line.forEach(([longitude, latitude], index) => {
        if (index === 0) ctx.moveTo(x(longitude), y(latitude));
        else ctx.lineTo(x(longitude), y(latitude));
      });
      ctx.stroke();
    });
  }

  function drawMap(run = activeRun()) {
    const canvas = q("#forecast-canvas");
    if (!payload || !canvas || !run.grid_metadata?.shape) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(600, Math.round(rect.width * ratio));
    const height = Math.max(420, Math.round(Math.max(rect.width * .56, 420) * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const meta = run.grid_metadata;
    const [nLead, nLat, nLon] = meta.shape;
    const variableIndex = meta.variables.indexOf(mapVariable);
    const dayIndex = meta.lead_days.indexOf(Number(mapDay));
    const count = nLead * nLat * nLon;
    const start = variableIndex * count + dayIndex * nLat * nLon;
    const image = ctx.createImageData(nLon, nLat);
    for (let yIndex = 0; yIndex < nLat; yIndex += 1) for (let xIndex = 0; xIndex < nLon; xIndex += 1) {
      const encoded = payload[start + yIndex * nLon + xIndex];
      const offset = ((nLat - 1 - yIndex) * nLon + xIndex) * 4;
      if (encoded === 65535) { image.data[offset + 3] = 0; continue; }
      const number = mapVariable === "precipitation" ? encoded / 10 : (encoded - 5000) / 100;
      const rgb = mapColor(number);
      image.data[offset] = rgb[0]; image.data[offset + 1] = rgb[1]; image.data[offset + 2] = rgb[2]; image.data[offset + 3] = 255;
    }
    const raster = document.createElement("canvas");
    raster.width = nLon; raster.height = nLat; raster.getContext("2d").putImageData(image, 0, 0);
    ctx.fillStyle = "#e8f1f5"; ctx.fillRect(0, 0, width, height);
    ctx.save(); ctx.translate(view.x, view.y); ctx.scale(view.scale, view.scale); ctx.imageSmoothingEnabled = true;
    ctx.drawImage(raster, 0, 0, width, height);
    ctx.strokeStyle = "rgba(255,255,255,.38)"; ctx.lineWidth = 1 / view.scale;
    for (let fraction = .2; fraction < 1; fraction += .2) { ctx.beginPath(); ctx.moveTo(width * fraction, 0); ctx.lineTo(width * fraction, height); ctx.moveTo(0, height * fraction); ctx.lineTo(width, height * fraction); ctx.stroke(); }
    drawCoastlines(ctx, meta, width, height);
    Object.entries(validation.cities).forEach(([name, item]) => {
      const x = width * (item.longitude - meta.bounding_box.lon_min) / (meta.bounding_box.lon_max - meta.bounding_box.lon_min);
      const y = height * (meta.bounding_box.lat_max - item.latitude) / (meta.bounding_box.lat_max - meta.bounding_box.lat_min);
      ctx.beginPath(); ctx.arc(x, y, 6 / view.scale, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#c51d3b"; ctx.lineWidth = 2.5 / view.scale; ctx.stroke();
      ctx.fillStyle = "#173f63"; ctx.font = `${12 / view.scale}px system-ui`; ctx.fillText(name, x + 10 / view.scale, y - 8 / view.scale);
    });
    ctx.restore();
    const names = { temperature: "Temperature", temperature_high: "Daily high", temperature_low: "Daily low", precipitation: "Accumulated rainfall" };
    q("#map-title").textContent = `${names[mapVariable]} · Day ${mapDay}`;
    q("#map-description").textContent = `${modelLabel(mapModel)} · initialized ${formatInit(run.initialization_utc)}`;
    q("#map-readout").textContent = `T+${Number(mapDay) * 24} h · drag to pan · scroll to zoom`;
  }

  function renderMapControls() {
    selectButton("[data-map-variable]", mapVariable, "mapVariable");
    selectButton("[data-map-day]", mapDay, "mapDay");
    view = { scale: 1, x: 0, y: 0 };
    loadMap();
    setUrl();
  }

  function renderValidation() {
    selectButton("[data-validation-city]", city, "validationCity");
    selectButton("[data-validation-variable]", validationVariable, "validationVariable");
    selectButton("[data-match-variable]", matchVariable, "matchVariable");
    q("#match-init-select").value = matchInit;
    const item = validation.cities[city];
    const overview = item.images[validationVariable];
    const matched = item.timeseries[matchInit][matchVariable];
    q("#validation-image").src = overview.path; q("#validation-image").alt = overview.alt;
    q("#match-image").src = matched.path; q("#match-image").alt = matched.alt;
    const points = item.summary[validationVariable].matched_points;
    q("#validation-summary").textContent = `${city} · ${points} matched points per available model · Open-Meteo observations`;
    setUrl();
  }

  qa("[data-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
  q("#init-select").addEventListener("change", (event) => { init = event.target.value; view = { scale: 1, x: 0, y: 0 }; renderRun(); });
  q("#city-select").addEventListener("change", (event) => { city = event.target.value; renderWeather(); renderValidation(); setUrl(); });
  qa("[data-weather-variable]").forEach((button) => button.addEventListener("click", () => { weatherVariable = button.dataset.weatherVariable; renderWeather(); setUrl(); }));
  qa("[data-map-variable]").forEach((button) => button.addEventListener("click", () => { mapVariable = button.dataset.mapVariable; renderMapControls(); }));
  qa("[data-map-day]").forEach((button) => button.addEventListener("click", () => { mapDay = button.dataset.mapDay; renderMapControls(); }));
  qa("[data-map-model]").forEach((button) => button.addEventListener("click", () => { mapModel = button.dataset.mapModel; renderMapControls(); }));
  qa("[data-validation-city]").forEach((button) => button.addEventListener("click", () => { city = button.dataset.validationCity; q("#city-select").value = city; renderWeather(); renderValidation(); }));
  qa("[data-validation-variable]").forEach((button) => button.addEventListener("click", () => { validationVariable = button.dataset.validationVariable; renderValidation(); }));
  qa("[data-match-variable]").forEach((button) => button.addEventListener("click", () => { matchVariable = button.dataset.matchVariable; renderValidation(); }));
  q("#match-init-select").addEventListener("change", (event) => { matchInit = event.target.value; renderValidation(); });
  q("#map-reset").addEventListener("click", () => { view = { scale: 1, x: 0, y: 0 }; drawMap(); });
  const canvas = q("#forecast-canvas");
  canvas.addEventListener("pointerdown", (event) => { drag = { x: event.clientX, y: event.clientY, moved: false }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!drag) return; const ratio = window.devicePixelRatio || 1; const dx = (event.clientX - drag.x) * ratio; const dy = (event.clientY - drag.y) * ratio; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true; view.x += dx; view.y += dy; drag.x = event.clientX; drag.y = event.clientY; drawMap(); });
  canvas.addEventListener("pointerup", (event) => {
    if (!drag?.moved) {
      const run = activeRun(), meta = run.grid_metadata, box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
      const gx = ((event.clientX - box.left) * ratio - view.x) / view.scale / canvas.width;
      const gy = ((event.clientY - box.top) * ratio - view.y) / view.scale / canvas.height;
      const lon = meta.bounding_box.lon_min + gx * (meta.bounding_box.lon_max - meta.bounding_box.lon_min);
      const lat = meta.bounding_box.lat_max - gy * (meta.bounding_box.lat_max - meta.bounding_box.lat_min);
      const nearest = Object.entries(validation.cities).map(([name, item]) => [name, Math.hypot((item.longitude - lon) * .9, item.latitude - lat)]).sort((a, b) => a[1] - b[1])[0];
      if (nearest && nearest[1] < 1.5) { city = nearest[0]; q("#city-select").value = city; renderWeather(); renderValidation(); activateTab("validation"); }
    }
    drag = null;
  });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); view.scale = Math.max(1, Math.min(4, view.scale * (event.deltaY < 0 ? 1.15 : .87))); drawMap(); }, { passive: false });
  window.addEventListener("resize", () => { if (tab === "maps") drawMap(); });

  activateTab(tab, false);
  renderRun();
  renderValidation();
})();
"""


def legacy_build_html(archive: dict, renderer, validation: dict, combination: dict | None = None) -> str:
    latest = archive["runs"][0]
    validation_cities = list(validation["cities"])
    default_city = validation_cities[0]
    combination = combination or {"cities": {}, "definition": "No combination data available."}
    combination_cities = list(combination["cities"])
    default_combination_city = combination_cities[0] if combination_cities else ""
    options = "".join(
        f'<option value="{run["id"]}">{pd.Timestamp(run["initialization_utc"]):%d %b %Y · 00 UTC}</option>'
        for run in archive["runs"]
    )
    map_model_controls = "".join(
        f'<button type="button" data-map-model="{model.get("id", model["label"].lower())}" aria-pressed="{str(index == 0).lower()}">{model["label"]}</button>'
        for index, model in enumerate(latest["models"])
    )
    source_rows = "".join(
        "<tr><th scope=\"row\">{label}</th><td>{provider}</td><td>{members}</td>"
        "<td><a href=\"{url}\">Source details</a></td></tr>".format(
            label=model["label"],
            provider=model["provider"],
            members=(
                "Deterministic"
                if model["members_total"] == 1
                else f"{model['members_used']} / {model['members_total']} members"
            ),
            url=model["source_url"],
        )
        for model in latest["models"]
    )
    validation_city_controls = "".join(
        f'<button type="button" data-validation-city="{city}" aria-pressed="{str(city == default_city).lower()}">{city}</button>'
        for city in validation_cities
    )
    combination_city_controls = "".join(
        f'<button type="button" data-combination-city="{city}" aria-pressed="{str(city == default_combination_city).lower()}">{city}</button>'
        for city in combination_cities
    ) or '<span class="tag">No learner output is currently available.</span>'
    default_image = validation["cities"][default_city]["images"]["temperature"]
    default_match_image = validation["cities"][default_city]["timeseries"][archive["runs"][0]["id"]]["precipitation"]
    product_count = sum(len(run.get("artifacts", [])) for run in archive["runs"])
    data = json.dumps({
        "runs": [{"id": run["id"], "initialization_utc": run["initialization_utc"], "grid_metadata": run.get("grid_metadata", {})} for run in archive["runs"]],
        "validation": validation,
        "combination": combination,
    })
    return f'''<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="AdaWeather experimental research system for India-region multi-model forecast analysis.">
  <title>AdaWeather · India Forecast Research</title><link rel="stylesheet" href="assets/style.css">
  <script defer src="assets/app.js"></script>
</head><body>
  <header class="masthead"><div class="shell nav-shell"><a class="brand" href="#top">ADAWEATHER <span>RESEARCH</span></a><nav aria-label="Primary navigation"><a href="#maps">Explorer</a><a href="#combination">Combination</a><a href="#validation">Verification</a><a href="#method">Method</a><a href="#resources">Resources</a></nav></div></header>
  <main id="top"><section class="experimental-banner"><div class="shell">EXPERIMENTAL RESEARCH SYSTEM · NOT AN OFFICIAL FORECAST, WARNING, OR SAFETY PRODUCT</div></section><section class="hero"><div class="shell hero-grid"><div><p class="eyebrow">India-region forecast research · rolling seven-initialization archive</p><h1>AdaWeather<br>forecast laboratory</h1><p class="lede">An experimental AI ensemble research system. AdaWeather examines a transparent, online-learned mixture of global forecast experts alongside the individual source models. Products are harmonized for research comparison, not operational decision-making.</p><div class="hero-actions"><a class="primary-action" href="#maps">Open map explorer</a><a class="text-action" href="#method">Read the method</a></div></div><dl class="run-card"><div><dt>Research status</dt><dd>Experimental · non-operational</dd></div><div><dt>Initialization</dt><dd id="run-summary">Loading archive…</dd></div><div><dt>Archive run</dt><dd><label class="sr-only" for="run-select">Choose forecast initialization</label><select id="run-select">{options}</select></dd></div><div><dt>Forecast windows</dt><dd>T+24 · T+72 · T+120 h</dd></div></dl></div></section>
  <section class="run-strip" aria-label="Forecast summary"><div class="shell stats"><div><strong>6</strong><span>global forecast experts</span></div><div><strong>7</strong><span>retained initializations</span></div><div><strong>3</strong><span>sampled lead days</span></div><div><strong>{product_count}</strong><span>rendered research maps</span></div></div></section>
  <section class="maps shell" id="maps"><div class="intro-row"><div><p class="kicker">Interactive archive explorer</p><h2>Explore the forecast field.</h2></div><p>Canvas-rendered model fields replace the image gallery. Drag to pan, use the wheel to zoom, and select a city marker to open matched evidence. Coordinates and marker placement use the published India-region bounding box.</p></div><div class="controls explorer-controls" aria-label="Forecast map controls"><fieldset><legend>Layer</legend><div class="segmented"><button type="button" data-variable-button="temperature" aria-pressed="true">Temperature</button><button type="button" data-variable-button="temperature_high" aria-pressed="false">Daily high</button><button type="button" data-variable-button="temperature_low" aria-pressed="false">Daily low</button><button type="button" data-variable-button="precipitation" aria-pressed="false">Rain accumulation</button></div></fieldset><fieldset><legend>Forecast endpoint</legend><div class="segmented"><button type="button" data-day-button="1" aria-pressed="true">Day 1 · +24h</button><button type="button" data-day-button="3" aria-pressed="false">Day 3 · +72h</button><button type="button" data-day-button="5" aria-pressed="false">Day 5 · +120h</button></div></fieldset><fieldset><legend>Expert model</legend><div class="segmented model-select">{map_model_controls}</div></fieldset></div><div class="map-workbench"><canvas id="forecast-canvas" aria-label="Interactive India forecast map"></canvas><div class="canvas-tools"><button type="button" id="map-reset">Reset view</button><span id="map-readout">Loading field…</span></div><aside><p class="kicker">Field explorer</p><h3 id="map-title">2 m temperature</h3><p id="map-description">Model field at selected valid time.</p><dl id="city-readout"><dt>Click a city marker</dt><dd>Open its verification and online-combination evidence.</dd></dl></aside></div></section>
  <section class="maps shell" id="maps"><div class="intro-row"><div><p class="kicker">Interactive archive explorer</p><h2>Explore the forecast field.</h2></div><p>Canvas-rendered model fields replace the image gallery. Drag to pan, use the wheel to zoom, and select a city marker to open matched evidence. Coordinates and marker placement use the published India-region bounding box.</p></div><div class="controls explorer-controls" aria-label="Forecast map controls"><fieldset><legend>Layer</legend><div class="segmented"><button type="button" data-variable-button="temperature" aria-pressed="true">Temperature</button><button type="button" data-variable-button="temperature_high" aria-pressed="false">Daily high</button><button type="button" data-variable-button="temperature_low" aria-pressed="false">Daily low</button><button type="button" data-variable-button="precipitation" aria-pressed="false">Rain accumulation</button></div></fieldset><fieldset><legend>Forecast endpoint</legend><div class="segmented"><button type="button" data-day-button="1" aria-pressed="true">Day 1 · +24h</button><button type="button" data-day-button="3" aria-pressed="false">Day 3 · +72h</button><button type="button" data-day-button="5" aria-pressed="false">Day 5 · +120h</button></div></fieldset><fieldset><legend>Expert model</legend><div class="segmented model-select">{map_model_controls}</div></fieldset></div><div class="map-workbench"><canvas id="forecast-canvas" aria-label="Interactive India forecast map"></canvas><div class="canvas-tools"><button type="button" id="map-reset">Reset view</button><span id="map-readout">Loading field…</span></div><aside><p class="kicker">Field explorer</p><h3 id="map-title">2 m temperature</h3><p id="map-description">Model field at selected valid time.</p><dl id="city-readout"><dt>Click a city marker</dt><dd>Open its verification and online-combination evidence.</dd></dl></aside></div><div class="sr-only">{"".join(f'<span data-init="{run["id"]}"></span>' for run in archive["runs"])}</div></section>
  <section class="combination shell" id="combination"><div class="intro-row"><div><p class="kicker">Experimental online combination</p><h2>One learned blend, shown with its uncertainty.</h2></div><p>For each city, AdaWeather applies a causal online learner to recent matched 2 m-temperature forecasts. The line is the convex blend; the envelope is the contributing model range—not a calibrated prediction interval.</p></div><div class="controls combination-controls"><fieldset><legend>City</legend><div class="segmented">{combination_city_controls}</div></fieldset></div><div class="combination-panel"><div class="combination-head"><div><span class="kicker">Current research blend</span><h3 id="combination-title">{default_combination_city or "Combination unavailable"}</h3></div><p id="combination-summary"></p></div><div id="combination-chart" role="img" aria-label="Experimental online combination temperature forecast"></div><div id="combination-weights" class="weight-grid"></div><p class="combination-note">{combination["definition"]}</p></div></section>
  <section class="validation shell" id="validation"><div class="intro-row"><div><p class="kicker">Realized forecast validation</p><h2>Test claims against observations.</h2></div><p>Each point pairs a published source-model forecast with Open-Meteo ground truth at the same city and valid time. Precipitation is accumulated over the identical initialization-to-endpoint window.</p></div><div class="controls validation-controls" aria-label="Validation chart controls"><fieldset><legend>City</legend><div class="segmented">{validation_city_controls}</div></fieldset><fieldset><legend>Variable</legend><div class="segmented"><button type="button" data-validation-variable="temperature" aria-pressed="true">Temperature</button><button type="button" data-validation-variable="precipitation" aria-pressed="false">Rain accumulation</button></div></fieldset></div><p class="validation-summary" id="validation-summary"></p><figure class="comparison validation-figure"><img id="validation-image" src="{default_image['path']}" alt="{default_image['alt']}"><figcaption><span>Left: forecast vs. matched observation. Right: mean absolute error by lead.</span><a class="download" href="assets/validation_manifest.json">Validation metadata</a></figcaption></figure><div class="single-init-head"><div><p class="kicker">Matched trajectory</p><h3>One initialization, matched leads.</h3></div><label>Initialization <select id="match-init-select">{options}</select></label></div><div class="controls match-controls" aria-label="Single-initialization variable controls"><fieldset><legend>Matched variable</legend><div class="segmented"><button type="button" data-match-variable="temperature" aria-pressed="false">Temperature</button><button type="button" data-match-variable="precipitation" aria-pressed="true">Rain accumulation</button></div></fieldset></div><figure class="comparison validation-figure"><img id="match-image" src="{default_match_image['path']}" alt="{default_match_image['alt']}"><figcaption><span>Each source-model trace is compared with the matched Open-Meteo observation at days 1, 3, and 5.</span><a class="download" href="assets/validation_manifest.json">Validation metadata</a></figcaption></figure></section>
  <section class="method-band" id="method"><div class="shell"><div class="intro-row light"><div><p class="kicker">AdaWeather method</p><h2>Ensemble research with a visible audit trail.</h2></div><p>AdaWeather is an experimental mixture-of-experts study, not a claim of superior operational skill. It evaluates a convex, online-learned combination of available global-model experts against recent observations, while retaining every individual expert for comparison.</p></div><div class="method-grid"><article><span>01</span><h3>Align</h3><p>Only shared 00 UTC initializations are admitted. Fields use the same India-region box, endpoints, and units.</p></article><article><span>02</span><h3>Derive</h3><p>Daily high and low layers are maxima and minima of every available native 2 m-temperature step in each 24-hour forecast day.</p></article><article><span>03</span><h3>Combine</h3><p>The AdaWeather research combiner learns non-negative expert weights from recent matched history; weights sum to one.</p></article><article><span>04</span><h3>Verify</h3><p>Published city samples are matched to Open-Meteo ground truth. Missing source data rejects a run rather than silently substituting one.</p></article></div></div></section>
  <section class="sources shell" id="sources"><div class="intro-row"><div><p class="kicker">Data provenance</p><h2>Source by source.</h2></div><p>WeatherNext products are read from private GCS Zarr archives. NOAA and ECMWF products are read from dynamical.org’s analysis-ready Icechunk archives.</p></div><div class="table-wrap"><table><thead><tr><th>Model</th><th>Archive</th><th>Map reduction</th><th>Documentation</th></tr></thead><tbody>{source_rows}</tbody></table></div><aside class="notice"><strong>Experimental guidance.</strong><p>All visualizations, prototype interfaces, and derived fields on this site are experimental research outputs. They are not official forecasts, warnings, or public-safety products. Consult the India Meteorological Department and relevant authorities for operational guidance.</p></aside></section>
  <section class="resources shell" id="resources"><div class="intro-row"><div><p class="kicker">Learning resources</p><h2>Atmosphere, models, and evidence.</h2></div><p>Selected open educational references for readers who want to connect forecast products with the underlying physical and chemical atmosphere.</p></div><div class="resource-grid"><a href="https://www2.acom.ucar.edu/atmos-chem-class"><span>UCAR / ACOM</span><strong>Atmospheric chemistry class</strong><em>Lecture series spanning chemistry, kinetics, aerosols, and measurement.</em></a><a href="https://csl.noaa.gov/learn/"><span>NOAA CSL</span><strong>Atmospheric chemistry &amp; composition</strong><em>Research-grounded learning materials on air chemistry and atmospheric composition.</em></a><a href="https://www.nesdis.noaa.gov/about/k-12-education/atmosphere-educational-resources"><span>NOAA NESDIS</span><strong>Atmosphere educational resources</strong><em>Weather, rain, clouds, wind, and observation-oriented introductory material.</em></a><a href="https://www.nasa.gov/stem-content/aura-atmospheric-chemistry-education-and-outreach/"><span>NASA Aura</span><strong>Atmospheric chemistry education</strong><em>Satellite-era atmospheric chemistry resources and outreach links.</em></a></div></section></main>
  <footer><div class="shell footer-row"><p>AdaWeather · experimental India forecast research system</p><a href="#top">Back to top ↑</a></div></footer><script id="archive-data" type="application/json">{data}</script></body></html>\n'''


def build_html(
    archive: dict,
    renderer,
    validation: dict,
    combination: dict | None = None,
    weather: dict | None = None,
) -> str:
    """Build one accessible, light-mode application with unique control IDs."""
    latest = archive["runs"][0]
    models = _model_catalog(archive)
    cities = list(validation["cities"])
    default_city = cities[0]
    weather = weather or {"runs": {run["id"]: {"cities": {}} for run in archive["runs"]}}
    options = "".join(
        f'<option value="{run["id"]}">{pd.Timestamp(run["initialization_utc"]):%d %b %Y · 00 UTC}</option>'
        for run in archive["runs"]
    )
    city_options = "".join(f'<option value="{city}">{city}</option>' for city in cities)
    city_buttons = "".join(
        f'<button type="button" data-validation-city="{city}" aria-pressed="{str(city == default_city).lower()}">{city}</button>'
        for city in cities
    )
    model_buttons = "".join(
        f'<button type="button" data-map-model="{model["id"]}" aria-pressed="{str(index == 0).lower()}">{model["label"]}</button>'
        for index, model in enumerate(models)
    )
    source_rows = "".join(
        "<tr><th scope=\"row\">{label}</th><td>{provider}</td><td>{members}</td>"
        "<td><a href=\"{url}\">Documentation</a></td></tr>".format(
            label=model["label"], provider=model["provider"],
            members="Deterministic" if model["members_total"] == 1 else f"{model['members_used']} / {model['members_total']} members",
            url=model["source_url"],
        ) for model in models
    )
    default_overview = validation["cities"][default_city]["images"]["temperature"]
    default_match = validation["cities"][default_city]["timeseries"][latest["id"]]["precipitation"]
    embedded = json.dumps({
        "archive": archive,
        "validation": validation,
        "combination": combination or {"cities": {}},
        "weather": weather,
        "models": models,
    }).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Five-day India weather forecasts, interactive model maps, and validation against observations.">
  <meta name="theme-color" content="#ffffff"><title>India Weather Forecasts · SCDLDS</title>
  <link rel="icon" href="assets/scdlds-logo.jpeg"><link rel="stylesheet" href="assets/style.css">
  <script defer src="assets/app.js"></script>
</head><body>
  <a class="skip-link" href="#content">Skip to content</a>
  <header class="site-header"><div class="shell header-row">
    <a class="brand" href="https://scdlds.ashoka.edu.in/" aria-label="SCDLDS at Ashoka University"><img src="assets/scdlds-logo.jpeg" alt="Safexpress Centre for Data, Learning and Decision Sciences, Ashoka University"></a>
    <nav class="tabs" role="tablist" aria-label="Forecast sections">
      <button type="button" role="tab" data-tab="weather" aria-selected="true">Weather</button>
      <button type="button" role="tab" data-tab="maps" aria-selected="false">Maps</button>
      <button type="button" role="tab" data-tab="validation" aria-selected="false">Validation</button>
      <button type="button" role="tab" data-tab="method" aria-selected="false">About</button>
    </nav>
  </div></header>
  <div class="notice-bar">Experimental research forecasts. For official warnings and decisions, use guidance from the India Meteorological Department.</div>
  <main id="content" class="shell">
    <section class="page-intro"><div><p class="eyebrow">India multi-model forecast</p><h1>Weather forecasts and model comparisons</h1><p>Five-day city forecasts, India-region maps, and matched validation against observations.</p></div>
      <div class="run-control"><label for="init-select">Forecast initialization</label><select id="init-select">{options}</select><strong id="run-status">Loading…</strong><small id="availability-note"></small></div>
    </section>

    <section class="panel" data-panel="weather" aria-label="City weather forecast">
      <div class="panel-heading"><div><p class="eyebrow">Five-day outlook</p><h2 id="weather-location">{default_city}</h2><p id="weather-meta"></p></div><label class="select-control" for="city-select">City<select id="city-select">{city_options}</select></label></div>
      <div class="weather-summary"><div id="weather-now" class="weather-now"></div><div class="segmented compact"><button type="button" data-weather-variable="temperature" aria-pressed="true">Temperature</button><button type="button" data-weather-variable="precipitation" aria-pressed="false">Accumulated rainfall</button></div></div>
      <div id="weather-chart" class="weather-chart"></div><div id="daily-cards" class="daily-cards"></div><p id="blend-note" class="data-note"></p>
    </section>

    <section class="panel" data-panel="maps" hidden aria-label="Interactive forecast maps">
      <div class="panel-heading"><div><p class="eyebrow">Forecast maps</p><h2>India field explorer</h2><p>Choose a model, variable, and endpoint. Drag to pan and scroll to zoom.</p></div></div>
      <div class="control-grid">
        <fieldset><legend>Variable</legend><div class="segmented"><button type="button" data-map-variable="temperature" aria-pressed="true">Temperature</button><button type="button" data-map-variable="temperature_high" aria-pressed="false">Daily high</button><button type="button" data-map-variable="temperature_low" aria-pressed="false">Daily low</button><button type="button" data-map-variable="precipitation" aria-pressed="false">Accumulated rainfall</button></div></fieldset>
        <fieldset><legend>Endpoint</legend><div class="segmented"><button type="button" data-map-day="1" aria-pressed="true">Day 1</button><button type="button" data-map-day="3" aria-pressed="false">Day 3</button><button type="button" data-map-day="5" aria-pressed="false">Day 5</button></div></fieldset>
        <fieldset class="model-fieldset"><legend>Model</legend><div class="segmented">{model_buttons}</div></fieldset>
      </div>
      <div class="map-layout"><div class="map-frame"><canvas id="forecast-canvas" aria-label="Interactive India forecast field with coastline overlay"></canvas><div class="map-tools"><button type="button" id="map-reset">Reset view</button><span id="map-readout">Loading map…</span></div></div><aside><p class="eyebrow">Selected field</p><h3 id="map-title">Temperature · Day 1</h3><p id="map-description"></p><small>Click a city marker to open its validation. Coastlines: <a href="https://www.naturalearthdata.com/">Natural Earth</a>.</small></aside></div>
    </section>

    <section class="panel" data-panel="validation" hidden aria-label="Forecast validation">
      <div class="panel-heading"><div><p class="eyebrow">Open-Meteo observations</p><h2>Forecast validation</h2><p>Forecasts and observations are matched at the same city and valid time. Rainfall uses the same accumulation window.</p></div></div>
      <div class="control-grid"><fieldset><legend>City</legend><div class="segmented">{city_buttons}</div></fieldset><fieldset><legend>Variable</legend><div class="segmented"><button type="button" data-validation-variable="temperature" aria-pressed="true">Temperature</button><button type="button" data-validation-variable="precipitation" aria-pressed="false">Accumulated rainfall</button></div></fieldset></div>
      <p id="validation-summary" class="data-note"></p><figure class="chart-image"><img id="validation-image" src="{default_overview['path']}" alt="{default_overview['alt']}"><figcaption>Forecast versus observation and mean absolute error by lead.</figcaption></figure>
      <div class="subheading"><div><h3>One initialization over matched leads</h3><p>Compare each forecast directly with its observation at days 1, 3, and 5.</p></div><label class="select-control" for="match-init-select">Initialization<select id="match-init-select">{options}</select></label></div>
      <div class="segmented compact"><button type="button" data-match-variable="temperature" aria-pressed="false">Temperature</button><button type="button" data-match-variable="precipitation" aria-pressed="true">Accumulated rainfall</button></div>
      <figure class="chart-image"><img id="match-image" src="{default_match['path']}" alt="{default_match['alt']}"><figcaption>Source-model traces and matched Open-Meteo observations.</figcaption></figure>
    </section>

    <section class="panel" data-panel="method" hidden aria-label="Methods and sources">
      <div class="panel-heading"><div><p class="eyebrow">About the data</p><h2>Method and sources</h2><p>The site updates from the newest available 00 UTC initialization. Late models are added when they become available.</p></div></div>
      <div class="method-grid"><article><strong>1. Load</strong><p>Model fields are reduced to a common India grid and consistent units.</p></article><article><strong>2. Combine</strong><p>Recent matched observations set separate online weights for temperature and rainfall.</p></article><article><strong>3. Verify</strong><p>Open-Meteo temperature and rainfall are matched to the same forecast valid periods.</p></article></div>
      <div class="table-wrap"><table><thead><tr><th>Model</th><th>Source</th><th>Members used</th><th>Reference</th></tr></thead><tbody>{source_rows}</tbody></table></div>
      <p class="method-note">Daily city rainfall is accumulated within each 24-hour forecast day. Map rainfall is cumulative from initialization through the selected endpoint. This is a research product, not an official forecast or warning.</p>
    </section>
  </main>
  <footer><div class="shell footer-row"><span>India Weather Forecasts · SCDLDS research</span><a href="https://scdlds.ashoka.edu.in/">Safexpress Centre for Data, Learning and Decision Sciences</a></div></footer>
  <script id="site-data" type="application/json">{embedded}</script>
</body></html>\n'''


LEGACY_ARCHIVE_CSS = r"""
.run-card select { width: 100%; border: 1px solid rgba(255,255,255,.3); border-radius: 2px; padding: 8px; color: #eef7f5; background: #0d3a48; font: inherit; font-size: .82rem; }
.run-card label { display: block; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.validation { padding-top: 92px; padding-bottom: 96px; scroll-margin-top: 20px; }
.validation-summary { margin: -25px 0 20px; color: var(--muted); font-size: .9rem; }
.validation-figure { margin-bottom: 0; }
.single-init-head { display: flex; justify-content: space-between; align-items: end; gap: 24px; margin: 54px 0 20px; }
.single-init-head .kicker { margin-bottom: 7px; }
.single-init-head h3 { margin: 0; font-size: 1.6rem; letter-spacing: -.04em; }
.single-init-head label { display: grid; gap: 7px; color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
.single-init-head select { min-width: 185px; border: 1px solid var(--line); border-radius: 2px; padding: 9px; color: var(--ink); background: white; font: inherit; font-size: .82rem; letter-spacing: normal; text-transform: none; }
.match-controls { margin-bottom: 20px; }
.brand span { color: #8fd7c4; font-weight: 620; }
.experimental-banner { padding: 9px 0; color: #1b2528; background: #f6bd3b; font-size: .69rem; font-weight: 850; letter-spacing: .1em; text-align: center; }
.hero { padding-top: 78px; background: radial-gradient(circle at 80% 16%, rgba(71, 143, 168, .52), transparent 26%), radial-gradient(circle at 44% 110%, rgba(17, 144, 126, .35), transparent 34%), linear-gradient(132deg, #07121a, #0b2735 57%, #123d50); }
.hero h1 { max-width: 850px; letter-spacing: -.075em; }
.hero .lede { max-width: 780px; }
.run-card { border-color: rgba(162, 226, 215, .36); background: rgba(5, 19, 27, .66); }
.run-card dt { color: #8fd7c4; }
.explorer-controls { position: sticky; top: 10px; z-index: 3; border-color: #aebdc2; box-shadow: 0 12px 34px rgba(6, 22, 30, .12); }
.segmented button:disabled { color: #8b999d; border-color: #dbe1e2; background: #edf0f0; cursor: not-allowed; }
.map-workbench { position: relative; display: grid; grid-template-columns: minmax(0, 1fr) 260px; overflow: hidden; border: 1px solid #173d49; background: #071923; box-shadow: var(--shadow); }
#forecast-canvas { display: block; width: 100%; min-height: 590px; touch-action: none; cursor: grab; }
#forecast-canvas:active { cursor: grabbing; }
.map-workbench aside { padding: 26px 22px; color: #d7e8e7; background: #0b2b36; }
.map-workbench aside .kicker { color: #80d2bb; }
.map-workbench h3 { color: white; font-size: 1.35rem; line-height: 1.05; }
.map-workbench aside p { color: #abc2c2; font-size: .84rem; }
#city-readout { margin-top: 38px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,.17); }
#city-readout dt { color: #83d3bb; font-size: .78rem; font-weight: 800; }
#city-readout dd { margin: 7px 0 0; color: #bed1d1; font-size: .82rem; }
.canvas-tools { position: absolute; z-index: 2; display: flex; align-items: center; gap: 10px; margin: 14px; padding: 7px; border: 1px solid rgba(255,255,255,.3); color: #d8e9e8; background: rgba(4,20,27,.78); font-size: .72rem; }
.canvas-tools button { border: 1px solid rgba(255,255,255,.32); padding: 6px 9px; color: white; background: transparent; font: inherit; cursor: pointer; }
.combination { padding: 90px 0 96px; scroll-margin-top: 20px; }
.combination-controls { margin-bottom: 18px; }
.combination-panel { border: 1px solid #1a3943; color: #eaf3f2; background: linear-gradient(125deg, #081c25, #103947); box-shadow: var(--shadow); }
.combination-head { display: flex; align-items: end; justify-content: space-between; gap: 30px; padding: 25px 28px 20px; border-bottom: 1px solid rgba(255,255,255,.16); }
.combination-head .kicker { margin-bottom: 4px; color: #83d3bb; }
.combination-head h3 { margin: 0; color: white; font-size: 1.65rem; letter-spacing: -.04em; }
.combination-head p { max-width: 440px; margin: 0; color: #b8ccce; font-size: .82rem; text-align: right; }
#combination-chart { padding: 12px 20px 0; }
#combination-chart svg { display: block; width: 100%; height: 290px; overflow: visible; }
.combo-grid line { stroke: rgba(211,236,232,.18); stroke-width: 1; }
.combo-grid text, #combination-chart svg text { fill: #a9c2c2; font: 12px Inter, sans-serif; }
.combo-range { fill: rgba(106, 216, 180, .18); stroke: none; }
.combo-line { fill: none; stroke: #79e1bd; stroke-width: 3.2; stroke-linejoin: round; stroke-linecap: round; }
.chart-key { margin: 0; padding: 0 28px 22px; color: #b8ccce; font-size: .78rem; }
.chart-key span, .chart-key i { display: inline-block; width: 20px; height: 3px; margin: 0 7px 2px 0; background: #79e1bd; vertical-align: middle; }
.chart-key i { height: 10px; margin-left: 18px; background: rgba(106, 216, 180, .3); }
.weight-grid { display: grid; grid-template-columns: repeat(5, 1fr); border-top: 1px solid rgba(255,255,255,.16); }
.weight-grid div { padding: 17px 20px; border-right: 1px solid rgba(255,255,255,.16); }
.weight-grid div:last-child { border-right: 0; }
.weight-grid span { display: block; color: #9eb8bb; font-size: .65rem; font-weight: 800; letter-spacing: .07em; }
.weight-grid strong { display: block; margin-top: 3px; color: white; font-size: 1.45rem; letter-spacing: -.05em; }
.combination-note { margin: 0; padding: 16px 28px; color: #aac2c2; border-top: 1px solid rgba(255,255,255,.16); font-size: .77rem; }
.map-figure { position: relative; border-color: #b7c5c8; background: #dce8e8; }
.map-canvas { overflow: hidden; cursor: zoom-in; }
.map-canvas img { transform-origin: 50% 50%; transition: transform .2s ease; }
.map-tools { position: absolute; top: 14px; left: 14px; z-index: 2; display: grid; overflow: hidden; border: 1px solid #8aa0a4; border-radius: 4px; box-shadow: 0 3px 14px rgba(0,0,0,.2); }
.map-tools button { width: 32px; height: 30px; padding: 0; border: 0; border-bottom: 1px solid #c7d2d4; color: #102a34; background: rgba(255,255,255,.94); font-size: 1.1rem; cursor: pointer; }
.map-tools button:last-child { border-bottom: 0; font-size: .9rem; }
.map-tools button:hover { color: white; background: var(--blue); }
.map-pins { position: absolute; inset: 0 0 44px; z-index: 2; pointer-events: none; }
.map-pin { position: absolute; transform: translate(-50%, -50%); padding: 4px 7px 4px 17px; border: 1px solid rgba(7,38,47,.65); border-radius: 999px; color: #0b2833; background: rgba(255,255,255,.92); box-shadow: 0 2px 7px rgba(0,0,0,.24); font: 700 .67rem/1 Inter, sans-serif; cursor: pointer; pointer-events: auto; }
.map-pin::before { content: ""; position: absolute; left: 6px; top: 50%; width: 6px; height: 6px; border-radius: 50%; background: #ee5a3f; transform: translateY(-50%); }
.map-pin:hover { color: white; background: #0c4354; }
.method-band { background: linear-gradient(120deg, #061820, #0c3544); }
.resource-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.resource-grid a { min-height: 190px; display: flex; flex-direction: column; padding: 22px; border: 1px solid var(--line); background: white; box-shadow: 0 8px 25px rgba(17,43,53,.06); text-decoration: none; transition: transform .18s ease, border-color .18s ease; }
.resource-grid a:hover { border-color: var(--teal); transform: translateY(-3px); }
.resource-grid span { margin-bottom: 35px; color: var(--teal); font-size: .7rem; font-weight: 800; letter-spacing: .1em; }
.resource-grid strong { font-size: 1.08rem; line-height: 1.18; }
.resource-grid em { margin-top: auto; color: var(--muted); font-size: .82rem; font-style: normal; }
@media (max-width: 900px) { .resource-grid { grid-template-columns: repeat(2, 1fr); } .map-workbench { grid-template-columns: 1fr; } #forecast-canvas { min-height: 470px; } }
@media (max-width: 650px) { .single-init-head { display: grid; align-items: start; } }
@media (max-width: 650px) { .validation, .combination { padding-top: 65px; padding-bottom: 68px; } .experimental-banner { font-size: .58rem; } .resource-grid { grid-template-columns: 1fr; } .map-pin { font-size: .58rem; } .combination-head { display: block; } .combination-head p { margin-top: 12px; text-align: left; } .weight-grid { grid-template-columns: repeat(2, 1fr); } .weight-grid div:nth-child(2) { border-right: 0; } .weight-grid div:nth-child(-n+2) { border-bottom: 1px solid rgba(255,255,255,.16); } #combination-chart { padding-inline: 8px; } }
"""


ARCHIVE_CSS = r"""
:root {
  color-scheme: light;
  --ink: #172b3a;
  --muted: #607080;
  --blue: #155fa0;
  --blue-dark: #104b7c;
  --blue-pale: #eaf3fa;
  --red: #c51d3b;
  --paper: #f6f8fa;
  --surface: #ffffff;
  --line: #dbe2e8;
  --shadow: 0 10px 30px rgba(26, 48, 68, .08);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; min-width: 320px; color: var(--ink); background: var(--paper); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; }
button, select { font: inherit; }
a { color: inherit; }
.shell { width: min(1180px, calc(100% - 40px)); margin-inline: auto; }
.skip-link { position: fixed; top: 8px; left: 8px; z-index: 100; padding: 9px 13px; color: white; background: var(--blue-dark); transform: translateY(-160%); }
.skip-link:focus { transform: translateY(0); }
.site-header { position: sticky; top: 0; z-index: 20; background: rgba(255,255,255,.97); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); }
.header-row { min-height: 76px; display: flex; align-items: center; justify-content: space-between; gap: 30px; }
.brand { display: block; flex: 0 1 360px; }
.brand img { display: block; width: min(360px, 100%); height: 54px; object-fit: contain; object-position: left center; }
.tabs { align-self: stretch; display: flex; gap: 4px; }
.tabs button { position: relative; min-width: 82px; padding: 0 14px; border: 0; color: var(--muted); background: transparent; font-size: .88rem; font-weight: 650; cursor: pointer; }
.tabs button::after { position: absolute; right: 12px; bottom: 0; left: 12px; height: 3px; background: var(--blue); content: ""; opacity: 0; }
.tabs button:hover, .tabs button.is-active { color: var(--blue-dark); }
.tabs button.is-active::after { opacity: 1; }
.notice-bar { padding: 8px 20px; color: #72540f; background: #fff8dc; border-bottom: 1px solid #eadc9e; font-size: .76rem; text-align: center; }
.page-intro { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(310px, .5fr); gap: 70px; align-items: end; padding-top: 54px; padding-bottom: 34px; }
.eyebrow { margin: 0 0 8px; color: var(--blue); font-size: .7rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
h1, h2, h3, p { margin-top: 0; }
h1 { max-width: 760px; margin-bottom: 12px; font-size: clamp(2.3rem, 5vw, 4rem); line-height: 1.02; letter-spacing: -.045em; }
.page-intro > div > p:last-child, .panel-heading p, .subheading p { max-width: 700px; margin-bottom: 0; color: var(--muted); }
.run-control { display: grid; gap: 7px; padding: 17px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }
.run-control label, .select-control, legend { color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
select { width: 100%; padding: 9px 34px 9px 10px; color: var(--ink); background: white; border: 1px solid #bdc8d1; border-radius: 5px; }
.run-control strong { font-size: .79rem; }
.run-control small { min-height: 2.5em; color: var(--muted); font-size: .72rem; }
.panel { margin-bottom: 70px; padding: 30px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }
.panel[hidden] { display: none; }
.panel-heading { display: flex; align-items: end; justify-content: space-between; gap: 30px; margin-bottom: 24px; }
.panel-heading h2 { margin-bottom: 5px; font-size: clamp(1.8rem, 4vw, 2.7rem); letter-spacing: -.035em; }
.select-control { display: grid; min-width: 190px; gap: 7px; }
.weather-summary { display: flex; align-items: center; justify-content: space-between; gap: 25px; min-height: 92px; padding: 18px 0; border-top: 1px solid var(--line); }
.weather-now { display: flex; align-items: center; gap: 17px; }
.weather-now > span { font-size: 3.6rem; line-height: 1; }
.weather-now strong { font-size: 3.5rem; font-weight: 500; line-height: 1; letter-spacing: -.06em; }
.weather-now small { color: var(--muted); font-size: .83rem; }
.segmented { display: flex; flex-wrap: wrap; gap: 6px; }
.segmented button { padding: 8px 12px; color: #465b6a; background: #f8fafb; border: 1px solid #ccd5dc; border-radius: 5px; font-size: .78rem; font-weight: 650; cursor: pointer; }
.segmented button:hover:not(:disabled) { color: var(--blue-dark); border-color: #7ba5c5; }
.segmented button[aria-pressed="true"] { color: white; background: var(--blue); border-color: var(--blue); }
.segmented button:disabled { color: #9ba7af; background: #f0f2f3; cursor: not-allowed; text-decoration: line-through; }
.compact { width: max-content; max-width: 100%; }
.weather-chart { min-height: 260px; margin-top: 8px; overflow-x: auto; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.weather-chart svg { display: block; width: 100%; min-width: 600px; height: 260px; }
.chart-grid line { stroke: #dfe5e9; stroke-width: 1; }
.chart-grid text, .weather-points text { fill: #71808b; font: 11px system-ui, sans-serif; }
.weather-points .date { font-weight: 650; }
.weather-points circle { fill: white; stroke: var(--blue); stroke-width: 2.5; }
.weather-area { fill: rgba(54, 133, 192, .12); }
.weather-line { fill: none; stroke: var(--blue); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
.daily-cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; padding-top: 20px; }
.day-card { display: grid; justify-items: center; min-height: 190px; padding: 16px 10px; border: 1px solid transparent; border-radius: 8px; }
.day-card.is-first { background: #f0f5fa; border-color: #dbe8f2; }
.day-card > strong { font-size: 1rem; }
.day-card time { color: var(--muted); font-size: .7rem; }
.weather-icon { margin: 12px 0 7px; font-size: 2.6rem; }
.day-card p { margin: 0 0 5px; }
.day-card p span, .day-card small { color: #7c8992; }
.day-card small { font-size: .74rem; }
.data-note { margin: 20px 0 0; color: var(--muted); font-size: .76rem; }
.empty-state { padding: 60px 20px; color: var(--muted); text-align: center; }
.control-grid { display: flex; flex-wrap: wrap; gap: 20px 32px; margin: 25px 0; padding: 17px; background: #f7f9fa; border: 1px solid var(--line); border-radius: 7px; }
fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
legend { margin-bottom: 8px; }
.model-fieldset { flex: 1 1 100%; }
.map-layout { display: grid; grid-template-columns: minmax(0, 1fr) 230px; border: 1px solid #cbd7df; border-radius: 7px; overflow: hidden; }
.map-frame { position: relative; min-width: 0; background: #e8f1f5; }
#forecast-canvas { display: block; width: 100%; min-height: 520px; touch-action: none; cursor: grab; }
#forecast-canvas:active { cursor: grabbing; }
.map-tools { position: absolute; top: 12px; left: 12px; display: flex; align-items: center; gap: 10px; padding: 7px; color: #45606f; background: rgba(255,255,255,.92); border: 1px solid #becbd3; border-radius: 5px; font-size: .7rem; box-shadow: 0 3px 12px rgba(29,51,66,.12); }
.map-tools button { padding: 6px 9px; color: var(--blue-dark); background: white; border: 1px solid #aebec9; border-radius: 4px; cursor: pointer; }
.map-layout aside { padding: 25px 20px; background: #f8fafb; border-left: 1px solid var(--line); }
.map-layout aside h3 { font-size: 1.25rem; }
.map-layout aside p, .map-layout aside small { color: var(--muted); }
.chart-image { margin: 18px 0 0; overflow: hidden; border: 1px solid var(--line); border-radius: 7px; }
.chart-image img { display: block; width: 100%; height: auto; background: #f5f8f7; }
.chart-image figcaption { padding: 11px 15px; color: var(--muted); border-top: 1px solid var(--line); font-size: .75rem; }
.subheading { display: flex; justify-content: space-between; align-items: end; gap: 30px; margin: 42px 0 17px; padding-top: 30px; border-top: 1px solid var(--line); }
.subheading h3 { margin-bottom: 5px; font-size: 1.35rem; }
.method-grid { display: grid; grid-template-columns: repeat(3, 1fr); margin: 30px 0; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
.method-grid article { min-height: 145px; padding: 22px; border-right: 1px solid var(--line); }
.method-grid article:last-child { border-right: 0; }
.method-grid strong { color: var(--blue-dark); }
.method-grid p { margin: 14px 0 0; color: var(--muted); font-size: .84rem; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 7px; }
table { width: 100%; border-collapse: collapse; text-align: left; }
th, td { padding: 13px 15px; border-bottom: 1px solid var(--line); font-size: .8rem; }
thead th { color: var(--muted); background: #f3f6f8; font-size: .68rem; text-transform: uppercase; letter-spacing: .05em; }
tbody td { color: var(--muted); }
tbody a { color: var(--blue); }
.method-note { margin: 22px 0 0; padding: 16px; color: #5f5b47; background: #fffbea; border-left: 3px solid #d7b34b; font-size: .8rem; }
footer { padding: 26px 0; color: #5e6f7b; background: white; border-top: 1px solid var(--line); }
.footer-row { display: flex; justify-content: space-between; gap: 25px; font-size: .78rem; }
footer a { color: var(--blue); }
@media (max-width: 850px) {
  .header-row { align-items: flex-start; flex-direction: column; gap: 4px; padding-top: 10px; }
  .brand img { height: 44px; }
  .tabs { width: 100%; min-height: 48px; overflow-x: auto; }
  .tabs button { flex: 1; min-width: 76px; }
  .page-intro { grid-template-columns: 1fr; gap: 25px; padding-top: 38px; }
  .panel-heading, .subheading { align-items: start; flex-direction: column; }
  .map-layout { grid-template-columns: 1fr; }
  .map-layout aside { border-top: 1px solid var(--line); border-left: 0; }
  .method-grid { grid-template-columns: 1fr; }
  .method-grid article { min-height: 0; border-right: 0; border-bottom: 1px solid var(--line); }
  .method-grid article:last-child { border-bottom: 0; }
}
@media (max-width: 620px) {
  .shell { width: min(100% - 22px, 1180px); }
  .notice-bar { font-size: .68rem; }
  h1 { font-size: 2.35rem; }
  .panel { padding: 18px; }
  .weather-summary { align-items: flex-start; flex-direction: column; }
  .weather-now strong { font-size: 2.8rem; }
  .daily-cards { grid-template-columns: repeat(5, minmax(108px, 1fr)); overflow-x: auto; }
  .control-grid { display: grid; }
  #forecast-canvas { min-height: 420px; }
  .footer-row { flex-direction: column; }
}
"""


def write_stage(stage: Path, archive: dict, validation: dict, combination: dict, weather: dict, renderer) -> None:
    assets = stage / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    logo = SITE_ROOT / "assets" / "scdlds-logo.jpeg"
    coastlines = SITE_ROOT / "assets" / "coastlines.json"
    if not logo.is_file() or not coastlines.is_file():
        raise RuntimeError("missing static brand or coastline asset")
    shutil.copy2(logo, assets / logo.name)
    shutil.copy2(coastlines, assets / coastlines.name)
    (assets / "style.css").write_text(ARCHIVE_CSS.strip() + "\n")
    (assets / "app.js").write_text(ARCHIVE_JS.strip() + "\n")
    (assets / "forecast_archive.json").write_text(json.dumps(archive, indent=2) + "\n")
    (assets / "forecast_manifest.json").write_text(json.dumps(archive["runs"][0], indent=2) + "\n")
    (assets / "validation_manifest.json").write_text(json.dumps(validation, indent=2) + "\n")
    (assets / "online_combination.json").write_text(json.dumps(combination, indent=2) + "\n")
    (assets / "weather_forecast.json").write_text(json.dumps(weather, indent=2) + "\n")
    (stage / "index.html").write_text(build_html(archive, renderer, validation, combination, weather))
    latest = archive["runs"][0]
    available = ", ".join(model["label"] for model in latest["models"])
    missing = ", ".join(latest.get("missing_models", [])) or "None"
    (stage / "README.md").write_text(
        "# India Weather Forecasts\n\n"
        "A rolling seven-initialization SCDLDS research dashboard with five-day city forecasts, "
        "interactive India maps, and Open-Meteo validation.\n\n"
        f"- Last successful build: `{archive['generated_at_utc']}`\n"
        f"- Latest initialization: `{latest['initialization_utc']}`\n"
        f"- Available models: {available}\n"
        f"- Models still pending: {missing}\n"
        "- Daily publisher: `india-forecast-pages.timer` at 14:00 Asia/Kolkata\n\n"
        "## Tests\n\n"
        "```bash\n"
        "/Datastorage/saptarishi.dhanuka_asp25/conda_envs/realtime_dash/bin/python -m pytest -q\n"
        "node --check assets/app.js\n"
        "```\n\n"
        "Live visit counting is intentionally disabled because no authenticated analytics backend is configured.\n\n"
        "Map coastlines use the public-domain [Natural Earth 1:50m coastline](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-coastline/).\n\n"
        "See [`assets/forecast_archive.json`](assets/forecast_archive.json), "
        "[`assets/weather_forecast.json`](assets/weather_forecast.json), and "
        "[`assets/validation_manifest.json`](assets/validation_manifest.json) for provenance.\n"
    )


def validate_stage(stage: Path, archive: dict, validation: dict, renderer) -> None:
    if len(archive["runs"]) != 7:
        raise RuntimeError(f"expected seven retained runs, found {len(archive['runs'])}")
    html = (stage / "index.html").read_text()
    ids = re.findall(r'\sid="([^"]+)"', html)
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate HTML ids: {', '.join(duplicates)}")
    seen = set()
    for run in archive["runs"]:
        model_ids = {model["id"] for model in run.get("models", [])}
        artifact_models = {artifact.get("model") for artifact in run.get("artifacts", [])}
        if run["id"] in seen or not model_ids or artifact_models != model_ids:
            raise RuntimeError(f"invalid artifact record for run {run.get('id')}")
        seen.add(run["id"])
        for artifact in run["artifacts"]:
            path = stage / artifact["path"]
            if artifact.get("kind") != "grid" or not path.is_file() or path.stat().st_size < 50_000:
                raise RuntimeError(f"invalid grid artifact: {artifact['path']}")
    for relative in ("assets/style.css", "assets/app.js", "assets/scdlds-logo.jpeg", "assets/coastlines.json", "assets/forecast_archive.json", "assets/forecast_manifest.json", "assets/online_combination.json", "assets/weather_forecast.json"):
        if not (stage / relative).is_file():
            raise RuntimeError(f"missing staged asset: {relative}")
    for city in validation["cities"].values():
        for image in city["images"].values():
            renderer.validate_png(stage / image["path"])
            if image["path"] not in html:
                raise RuntimeError(f"unlinked validation image: {image['path']}")
        for run_images in city["timeseries"].values():
            for image in run_images.values():
                renderer.validate_png(stage / image["path"])
                if image["path"] not in html:
                    raise RuntimeError(f"unlinked matched-time-series image: {image['path']}")
    weather = json.loads((stage / "assets" / "weather_forecast.json").read_text())
    for run in archive["runs"]:
        for city_name in validation["cities"]:
            days = weather["runs"].get(run["id"], {}).get("cities", {}).get(city_name, {}).get("days", [])
            if len(days) != len(DAILY_LEAD_DAYS):
                raise RuntimeError(f"incomplete five-day weather product: {run['id']} / {city_name}")


def publish_stage(stage: Path, output_site: Path) -> None:
    if not (output_site / ".git").is_dir():
        raise RuntimeError(f"not a Git Pages repository: {output_site}")
    target_assets = output_site / "assets"
    backup = output_site / ".assets.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if target_assets.exists():
        target_assets.rename(backup)
    try:
        shutil.copytree(stage / "assets", target_assets)
        shutil.copy2(stage / "index.html", output_site / "index.html")
        shutil.copy2(stage / "README.md", output_site / "README.md")
    except Exception:
        if target_assets.exists():
            shutil.rmtree(target_assets)
        if backup.exists():
            backup.rename(target_assets)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-site", type=Path, default=SITE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--history-runs", type=int, default=7)
    parser.add_argument("--backfill", action="store_true", help="fill the archive to the requested retention")
    parser.add_argument("--validation-only", action="store_true", help="regenerate validation from the retained runs without loading a new forecast")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.history_runs != 7:
        raise SystemExit("this public archive is intentionally fixed at seven retained runs")
    renderer, load_config, india_load, openmeteo = load_renderer(args.realtime_root.resolve())
    cfg = load_config()
    models = tuple(renderer.DEFAULT_MODELS)
    prior_archive = read_archive(args.output_site)
    existing = prior_archive["runs"] if args.validation_only else valid_existing_runs(args.output_site, prior_archive, renderer)
    if args.backfill and args.validation_only:
        raise SystemExit("--backfill and --validation-only cannot be combined")
    availability, source_errors = ({}, {}) if args.validation_only else model_availability(models, cfg, india_load)
    candidates = [] if args.validation_only else candidate_initializations(availability, existing, backfill=args.backfill)
    if args.backfill:
        newest_seven = set(sorted(set().union(*availability.values()), reverse=True)[:7])
        candidates = [value for value in candidates if value in newest_seven]
    else:
        candidates = candidates[:3]
    if not args.validation_only and not candidates and prior_archive.get("schema_version") == 2:
        print("no newer or newly completed initialization is available; site unchanged")
        return

    with tempfile.TemporaryDirectory(prefix="forecast-archive-", dir="/tmp") as tmp:
        stage = Path(tmp)
        stage_map_data = stage / "assets" / "map_data"
        stage_map_data.mkdir(parents=True)
        retained = existing[:7]
        if args.validation_only:
            rebuilt = []
            for run in retained:
                run_init = pd.Timestamp(run["initialization_utc"])
                run_models = tuple(model["id"] for model in run["models"])
                datasets = {model: _open_run_dataset(cfg, run, model) for model in run_models}
                artifacts, grid_metadata = write_map_payloads(run_init, datasets, run_models, cfg, india_load, stage)
                rebuilt.append({
                    **run, "artifacts": artifacts, "grid_metadata": grid_metadata,
                    "lead_semantics": {
                        **run["lead_semantics"],
                        "temperature_high": "Maximum native-step 2 m temperature during the 24 hours ending at each selected lead.",
                        "temperature_low": "Minimum native-step 2 m temperature during the 24 hours ending at each selected lead.",
                    },
                })
            retained = rebuilt
        else:
            for run in retained:
                source = args.output_site / "assets" / "map_data" / run["id"]
                if source.is_dir():
                    shutil.copytree(source, stage_map_data / run["id"])
        for init in candidates:
            existing_run = next((entry for entry in retained if entry["id"] == stamp(init)), None)
            models_for_init = {
                model for model, values in availability.items() if init in values
            }
            if existing_run:
                models_for_init.update(model["id"] for model in existing_run.get("models", []))
            ordered_models = tuple(model for model in models if model in models_for_init)
            if args.backfill and not existing_run:
                # Backfill the seven dates promptly with the fastest source. Daily
                # reconciliation can add late/slow experts without delaying history.
                preferred = next((model for model in ("weathernext2", "gfs", *models) if model in ordered_models), None)
                ordered_models = (preferred,) if preferred else ordered_models[:1]
            try:
                run = render_run(init, ordered_models, cfg, renderer, india_load, stage, args.attempts)
            except Exception as exc:  # noqa: BLE001 - keep last good archive intact
                print(f"[{stamp(init)}] rejected: {exc}", file=sys.stderr, flush=True)
                continue
            if existing_run:
                old_models = {model["id"] for model in existing_run.get("models", [])}
                new_models = {model["id"] for model in run.get("models", [])}
                if not new_models.issuperset(old_models):
                    print(f"[{stamp(init)}] rejected because it would remove existing models", file=sys.stderr, flush=True)
                    continue
            retained = [entry for entry in retained if entry["id"] != run["id"]] + [run]
            retained = sorted(retained, key=lambda entry: entry["initialization_utc"], reverse=True)[:7]
            if not args.backfill:
                break
        if len(retained) != 7:
            raise RuntimeError(f"could not build a complete seven-run archive (have {len(retained)})")
        archive = archive_manifest(retained)
        archive["source_status"] = {
            model: {
                "newest_initialization_utc": utc_text(max(values)) if values else None,
                "error": source_errors.get(model),
            }
            for model, values in availability.items()
        } if availability else prior_archive.get("source_status", {})
        validation = render_validation(archive, cfg, openmeteo, stage)
        combination = render_online_combination(cfg)
        weather = render_weather_forecasts(archive, cfg, india_load)
        write_stage(stage, archive, validation, combination, weather, renderer)
        validate_stage(stage, archive, validation, renderer)
        if args.dry_run:
            print("validated archive build; dry-run leaves the site unchanged")
        else:
            publish_stage(stage, args.output_site)
            print(f"published seven validated forecast runs to {args.output_site}")


if __name__ == "__main__":
    main()
