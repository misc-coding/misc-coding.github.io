#!/usr/bin/env python3
"""IMERG and native-time precipitation products for the static forecast site.

The module keeps acquisition, interval matching, encoding, bias correction, and
plotting separate from the site publisher.  IMERG ``time`` coordinates name the
*start* of a 30-minute granule.  Forecast ``valid_time`` coordinates name the
*end* of an interval-average precipitation rate.  Matching therefore compares
forecast ``(start, end]`` with the equivalent IMERG ``[start, end)`` samples.
"""
from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


IMERG_DATASETS = {
    "early": "nasa-imerg-analysis-early",
    "late": "nasa-imerg-analysis-late",
}
NATIVE_INTERVAL = pd.Timedelta(minutes=30)
NATIVE_INTERVAL_SECONDS = 1_800.0
OBSERVATION_DAYS = 3
RECENT_FORECAST_RUNS = 3
IMERG_COMBINED_MODEL_ID = "imerg_combined"
IMERG_COMBINED_MODEL_LABEL = "IMERG-calibrated combination · 6 h"
PAYLOAD_SCALE_MM = 100.0
MISSING_UINT16 = np.uint16(65_535)


def utc_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _naive(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp


def imerg_interval(time_value) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the exact validity interval represented by an IMERG time value."""
    start = _naive(time_value)
    return start, start + NATIVE_INTERVAL


def rate_to_native_accumulation(rate: xr.DataArray) -> xr.DataArray:
    """Convert IMERG kg m-2 s-1 half-hour mean rate to millimetres/granule."""
    result = (rate.clip(min=0) * NATIVE_INTERVAL_SECONDS).astype(np.float32)
    result.name = "precip_mm_30min"
    result.attrs = {
        "long_name": "precipitation accumulated during the named 30-minute interval",
        "units": "mm",
        "interval_seconds": int(NATIVE_INTERVAL_SECONDS),
    }
    return result


def _bbox_values(bbox) -> dict[str, float]:
    return {
        "lat_min": float(bbox.lat_min), "lat_max": float(bbox.lat_max),
        "lon_min": float(bbox.lon_min), "lon_max": float(bbox.lon_max),
    }


def _select_imerg(ds: xr.Dataset, start, end, bbox) -> xr.Dataset:
    """Select exact half-hour starts in ``[start, end)`` and the India crop."""
    start, end = _naive(start), _naive(end)
    expected = pd.date_range(start, end, freq="30min", inclusive="left")
    if not len(expected):
        raise ValueError("IMERG request must contain at least one half-hour interval")
    selected = ds[["precipitation_surface"]].sel(
        time=slice(np.datetime64(start), np.datetime64(end - NATIVE_INTERVAL)),
        latitude=slice(float(bbox.lat_max), float(bbox.lat_min)),
        longitude=slice(float(bbox.lon_min), float(bbox.lon_max)),
    )
    actual = pd.to_datetime(selected["time"].values).tz_localize(None)
    if not actual.equals(expected):
        missing = expected.difference(actual)
        raise ValueError(
            f"IMERG interval is incomplete: expected {len(expected)} half-hours, "
            f"found {len(actual)}; first missing={missing[0] if len(missing) else 'unknown'}"
        )
    rain = rate_to_native_accumulation(selected["precipitation_surface"]).load()
    rain = rain.rename({"latitude": "lat", "longitude": "lon"}).sortby("lat").sortby("lon")
    result = xr.Dataset({"precip_mm_30min": rain})
    result = result.reset_coords(drop=True)
    for name in result.coords:
        result[name].attrs = {}
    result.attrs = {
        "interval_start_utc": utc_text(start),
        "interval_end_exclusive_utc": utc_text(end),
        "native_time_semantics": "time is the inclusive start of a 30-minute interval",
    }
    return result


def _cache_path(cache_root: Path, product: str) -> Path:
    return Path(cache_root) / "imerg" / f"{product}_recent.nc"


def _read_exact_cache(path: Path, start, end) -> xr.Dataset | None:
    if not path.is_file():
        return None
    try:
        with xr.open_dataset(path) as opened:
            if (
                opened.attrs.get("interval_start_utc") != utc_text(start)
                or opened.attrs.get("interval_end_exclusive_utc") != utc_text(end)
            ):
                return None
            return opened.load()
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.nc")
    dataset.to_netcdf(temporary)
    temporary.replace(path)


def fetch_recent_imerg(
    open_dataset: Callable,
    bbox,
    cache_root: Path,
    *,
    days: int = OBSERVATION_DAYS,
) -> tuple[dict[str, xr.Dataset], dict]:
    """Load a common rolling Early/Late window ending at their latest shared granule."""
    if days <= 0:
        raise ValueError("days must be positive")
    opened = {
        product: open_dataset(dataset_id, decode_timedelta=False, chunks=None)
        for product, dataset_id in IMERG_DATASETS.items()
    }
    try:
        latest_starts = {
            product: _naive(dataset["time"].values[-1])
            for product, dataset in opened.items()
        }
        common_end = min(latest_starts.values()) + NATIVE_INTERVAL
        common_start = common_end - pd.Timedelta(days=days)
        products = {}
        for product, dataset in opened.items():
            path = _cache_path(cache_root, product)
            cached = _read_exact_cache(path, common_start, common_end)
            if cached is None:
                cached = _select_imerg(dataset, common_start, common_end, bbox)
                cached.attrs.update({
                    "product": product,
                    "dataset_id": IMERG_DATASETS[product],
                    "source_url": f"https://dynamical.org/catalog/{IMERG_DATASETS[product]}/",
                })
                _write_cache(cached, path)
            products[product] = cached
        window = {
            "start_utc": utc_text(common_start),
            "end_exclusive_utc": utc_text(common_end),
            "half_hour_intervals": int(days * 48),
            "latest_source_interval_starts_utc": {
                product: utc_text(value) for product, value in latest_starts.items()
            },
        }
        return products, window
    finally:
        for dataset in opened.values():
            dataset.close()


def aligned_accumulations(dataset: xr.Dataset, *, hours: int = 6) -> xr.DataArray:
    """Sum complete UTC-aligned IMERG windows without resampling partial bins."""
    if hours <= 0 or hours * 2 != int(hours * 2):
        raise ValueError("hours must be a positive half-hour multiple")
    starts = pd.to_datetime(dataset["time"].values).tz_localize(None)
    if not len(starts):
        raise ValueError("IMERG dataset is empty")
    available_start = starts[0]
    available_end = starts[-1] + NATIVE_INTERVAL
    first = available_start.ceil(f"{hours}h")
    fields, window_starts, window_ends = [], [], []
    expected_count = int(hours * 2)
    cursor = first
    while cursor + pd.Timedelta(hours=hours) <= available_end:
        end = cursor + pd.Timedelta(hours=hours)
        chosen = np.flatnonzero((starts >= cursor) & (starts < end))
        expected = pd.date_range(cursor, end, freq="30min", inclusive="left")
        actual = starts[chosen]
        if len(chosen) != expected_count or not actual.equals(expected):
            raise ValueError(f"incomplete IMERG {hours}-hour interval {cursor}..{end}")
        fields.append(dataset["precip_mm_30min"].isel(time=chosen).sum("time"))
        window_starts.append(np.datetime64(cursor, "ns"))
        window_ends.append(np.datetime64(end, "ns"))
        cursor = end
    if not fields:
        raise ValueError(f"no complete aligned {hours}-hour IMERG intervals")
    result = xr.concat(
        fields,
        dim=xr.DataArray(np.asarray(window_ends), dims="valid_time", name="valid_time"),
    )
    result = result.assign_coords(interval_start=("valid_time", np.asarray(window_starts)))
    result.name = f"precip_mm_{hours}h"
    result.attrs = {
        "units": "mm",
        "accumulation": f"sum of {expected_count} exact native half-hours",
    }
    return result


def encode_precip(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    encoded = np.rint(np.clip(values, 0, (int(MISSING_UINT16) - 1) / PAYLOAD_SCALE_MM) * PAYLOAD_SCALE_MM)
    return np.where(np.isfinite(values), encoded, MISSING_UINT16).astype("<u2")


def encode_temperature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    encoded = np.rint(np.clip(values, -50, 65) * 100 + 5_000)
    return np.where(np.isfinite(values), encoded, MISSING_UINT16).astype("<u2")


def _write_gzip_payload(path: Path, payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            compressed.write(np.asarray(payload, dtype="<u2").tobytes())


def _grid_metadata(data: xr.DataArray) -> dict:
    lat = np.asarray(data["lat"].values, dtype=float)
    lon = np.asarray(data["lon"].values, dtype=float)
    return {
        "shape": [int(len(lat)), int(len(lon))],
        "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        "latitude_spacing_degrees": float(np.median(np.diff(lat))) if len(lat) > 1 else None,
        "longitude_spacing_degrees": float(np.median(np.diff(lon))) if len(lon) > 1 else None,
    }


def write_imerg_observation_payloads(products: dict[str, xr.Dataset], stage: Path) -> dict:
    """Write native 30-minute and aligned six-hour maps, retaining native 0.1° cells."""
    manifest = {}
    for product, dataset in products.items():
        native = dataset["precip_mm_30min"]
        dates = pd.to_datetime(native["time"].values).tz_localize(None).floor("D")
        daily_assets = []
        for date in pd.Index(dates).unique():
            chosen = np.flatnonzero(dates == date)
            subset = native.isel(time=chosen)
            relative = Path("assets") / "imerg" / product / f"{pd.Timestamp(date):%Y%m%d}_30min.bin.gz"
            _write_gzip_payload(stage / relative, encode_precip(subset.values))
            starts = pd.to_datetime(subset["time"].values).tz_localize(None)
            daily_assets.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "path": relative.as_posix(),
                "shape": [int(len(starts)), *list(_grid_metadata(subset)["shape"])],
                "intervals": [
                    {"start_utc": utc_text(value), "end_utc": utc_text(value + NATIVE_INTERVAL)}
                    for value in starts
                ],
            })
        six_hour = aligned_accumulations(dataset, hours=6)
        relative = Path("assets") / "imerg" / product / "recent_6h.bin.gz"
        _write_gzip_payload(stage / relative, encode_precip(six_hour.values))
        manifest[product] = {
            "dataset_id": IMERG_DATASETS[product],
            "label": f"IMERG {product.capitalize()} Run",
            "source_url": f"https://dynamical.org/catalog/{IMERG_DATASETS[product]}/",
            "grid": _grid_metadata(native),
            "native": daily_assets,
            "six_hour": {
                "path": relative.as_posix(),
                "shape": [int(six_hour.sizes["valid_time"]), *list(_grid_metadata(six_hour)["shape"])],
                "intervals": [
                    {"start_utc": utc_text(start), "end_utc": utc_text(end)}
                    for start, end in zip(six_hour["interval_start"].values, six_hour["valid_time"].values)
                ],
            },
        }
    return manifest


def forecast_interval_fields(series: xr.Dataset, init, *, horizon_days: int = 5) -> xr.Dataset:
    """Normalize a native forecast series to exact interval rain and temperature snapshots."""
    init = _naive(init)
    times = pd.to_datetime(series["valid_time"].values).tz_localize(None)
    chosen = np.flatnonzero((times > init) & (times <= init + pd.Timedelta(days=horizon_days)))
    if not len(chosen):
        raise ValueError("forecast series has no positive native lead times")
    valid = times[chosen]
    starts = pd.DatetimeIndex([init, *valid[:-1]])
    hours = np.asarray((valid.values - starts.values) / np.timedelta64(1, "h"), dtype=float)
    if np.any(hours <= 0):
        raise ValueError("forecast valid times must be strictly increasing")
    temperature = series["t2m_C"].isel(valid_time=chosen).astype(np.float32)
    rate = series["precip_mmday"].isel(valid_time=chosen).clip(min=0).astype(np.float32)
    weights = xr.DataArray(hours / 24.0, dims="valid_time", coords={"valid_time": temperature["valid_time"]})
    rain = rate * weights
    result = xr.Dataset({"temperature_c": temperature, "precip_interval_mm": rain})
    result = result.assign_coords(interval_start=("valid_time", starts.values.astype("datetime64[ns]")))
    result.attrs = {
        "initialization_utc": utc_text(init),
        "rain_semantics": "interval-average rate integrated over (interval_start, valid_time]",
    }
    return result


def write_forecast_temporal_payload(
    prepared: xr.Dataset,
    stage: Path,
    run_id: str,
    model: str,
) -> dict:
    """Write one model's five-day native-time temperature/rainfall map payload."""
    payload = np.concatenate([
        encode_temperature(prepared["temperature_c"].values).reshape(-1),
        encode_precip(prepared["precip_interval_mm"].values).reshape(-1),
    ])
    relative = Path("assets") / "temporal_forecasts" / run_id / f"{model}.bin.gz"
    _write_gzip_payload(stage / relative, payload)
    valid = pd.to_datetime(prepared["valid_time"].values).tz_localize(None)
    starts = pd.to_datetime(prepared["interval_start"].values).tz_localize(None)
    grid = _grid_metadata(prepared["temperature_c"])
    return {
        "path": relative.as_posix(),
        "shape": [int(len(valid)), *list(grid["shape"])],
        "grid": grid,
        "variables": ["temperature", "precipitation"],
        "times": [
            {
                "interval_start_utc": utc_text(start),
                "valid_time_utc": utc_text(end),
                "interval_hours": float((end - start) / pd.Timedelta(hours=1)),
            }
            for start, end in zip(starts, valid)
        ],
    }


def forecast_city_records(prepared: xr.Dataset, city) -> list[dict]:
    point = prepared.sel(lat=float(city.lat), lon=float(city.lon), method="nearest")
    valid = pd.to_datetime(point["valid_time"].values).tz_localize(None)
    starts = pd.to_datetime(point["interval_start"].values).tz_localize(None)
    return [
        {
            "interval_start_utc": utc_text(start),
            "valid_time_utc": utc_text(end),
            "interval_hours": float((end - start) / pd.Timedelta(hours=1)),
            "temperature_c": float(point["temperature_c"].isel(valid_time=index).item()),
            "forecast_mm": float(point["precip_interval_mm"].isel(valid_time=index).item()),
            "grid_latitude": float(point["lat"].item()),
            "grid_longitude": float(point["lon"].item()),
        }
        for index, (start, end) in enumerate(zip(starts, valid))
    ]


def _observation_point_lookup(dataset: xr.Dataset, city) -> dict[pd.Timestamp, float]:
    point = dataset["precip_mm_30min"].sel(lat=float(city.lat), lon=float(city.lon), method="nearest")
    times = pd.to_datetime(point["time"].values).tz_localize(None)
    values = np.asarray(point.values, dtype=float)
    return {pd.Timestamp(time): float(value) for time, value in zip(times, values) if np.isfinite(value)}


def match_city_records(
    records: list[dict],
    observation_lookups: dict[str, dict[pd.Timestamp, float]],
) -> list[dict]:
    """Attach exact Early/Late sums only when every half-hour is present."""
    matched = []
    for record in records:
        start = _naive(record["interval_start_utc"])
        end = _naive(record["valid_time_utc"])
        expected = pd.date_range(start, end, freq="30min", inclusive="left")
        if not len(expected) or end - start != len(expected) * NATIVE_INTERVAL:
            continue
        observations = {}
        for product, lookup in observation_lookups.items():
            values = [lookup.get(pd.Timestamp(time)) for time in expected]
            observations[product] = float(sum(values)) if all(value is not None for value in values) else None
        if any(value is not None for value in observations.values()):
            matched.append({**record, "imerg_early_mm": observations.get("early"), "imerg_late_mm": observations.get("late")})
    return matched


def causal_run_bias(
    prior_rows: list[dict],
    as_of,
    *,
    half_life_days: float = 7.0,
    prior_weight: float = 3.0,
) -> tuple[float, int]:
    """Estimate a shrunken additive bias using Late-Run truth realized by issue time."""
    cutoff = _naive(as_of)
    eligible = [
        row for row in prior_rows
        if row.get("imerg_late_mm") is not None and _naive(row["valid_time_utc"]) <= cutoff
    ]
    if not eligible:
        return 0.0, 0
    ages = np.asarray([
        max(0.0, float((cutoff - _naive(row["valid_time_utc"])) / pd.Timedelta(days=1)))
        for row in eligible
    ])
    weights = np.exp(-np.log(2.0) * ages / half_life_days)
    residuals = np.asarray([
        float(row["imerg_late_mm"]) - float(row["forecast_mm"])
        for row in eligible
    ])
    bias = float(np.sum(weights * residuals) / (prior_weight + np.sum(weights)))
    return float(np.clip(bias, -100.0, 100.0)), len(eligible)


def _rmse(pairs: list[tuple[float, float]]) -> float | None:
    return float(np.sqrt(np.mean([(a - b) ** 2 for a, b in pairs]))) if pairs else None


def apply_causal_bias(matches: dict) -> dict:
    """Apply one issue-time-safe city/model bias to every row of each initialization."""
    summaries = {}
    cities = list(matches)
    for city in cities:
        summaries[city] = {}
        models = sorted({model for run in matches[city].values() for model in run})
        for model in models:
            history: list[dict] = []
            summaries[city][model] = {}
            ordered_runs = sorted(matches[city], key=lambda run_id: matches[city][run_id].get(model, {}).get("initialization_utc", ""))
            for run_id in ordered_runs:
                item = matches[city][run_id].get(model)
                if not item:
                    continue
                bias, training_count = (
                    (0.0, 0) if item.get("skip_bias")
                    else causal_run_bias(history, item["initialization_utc"])
                )
                for row in item["rows"]:
                    row["bias_correction_mm"] = bias
                    row["bias_corrected_mm"] = max(0.0, float(row["forecast_mm"]) + bias)
                late_rows = [row for row in item["rows"] if row.get("imerg_late_mm") is not None]
                raw_pairs = [(float(row["forecast_mm"]), float(row["imerg_late_mm"])) for row in late_rows]
                corrected_pairs = [(float(row["bias_corrected_mm"]), float(row["imerg_late_mm"])) for row in late_rows]
                summaries[city][model][run_id] = {
                    "bias_mm": bias,
                    "training_intervals": training_count,
                    "matched_late_intervals": len(late_rows),
                    "raw_rmse_mm": _rmse(raw_pairs),
                    "bias_corrected_rmse_mm": _rmse(corrected_pairs),
                }
                if not item.get("skip_bias"):
                    history.extend(item["rows"])
    return summaries


def render_city_validation_plot(
    rows: list[dict],
    *,
    city_name: str,
    model_label: str,
    initialization,
    summary: dict,
    out: Path,
) -> None:
    """Plot native-interval forecast, Early/Late truth, and causal bias correction."""
    fig, (value_ax, error_ax) = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=True, facecolor="#f7f9fa")
    fig.subplots_adjust(left=.09, right=.98, bottom=.14, top=.86, hspace=.16)
    if rows:
        times = pd.to_datetime([row["valid_time_utc"] for row in rows], utc=True).tz_localize(None)
        forecast = np.asarray([row["forecast_mm"] for row in rows], dtype=float)
        corrected = np.asarray([row["bias_corrected_mm"] for row in rows], dtype=float)
        value_ax.plot(times, forecast, color="#d4573b", marker="o", ms=3.5, lw=1.6, label=f"{model_label} forecast")
        value_ax.plot(times, corrected, color="#315f80", marker="o", ms=3, lw=1.5, ls="--", label="Causal IMERG-Late bias correction")
        for key, label, color in (
            ("imerg_early_mm", "IMERG Early", "#2a9d8f"),
            ("imerg_late_mm", "IMERG Late", "#172b3a"),
        ):
            selected = [(time, row[key]) for time, row in zip(times, rows) if row.get(key) is not None]
            if selected:
                x, y = zip(*selected)
                value_ax.plot(x, y, color=color, marker="o", ms=4, lw=2.0, label=label)
        late = [(time, row) for time, row in zip(times, rows) if row.get("imerg_late_mm") is not None]
        if late:
            x = [item[0] for item in late]
            error_ax.plot(x, [item[1]["forecast_mm"] - item[1]["imerg_late_mm"] for item in late], color="#d4573b", marker="o", ms=3, label="Raw error")
            error_ax.plot(x, [item[1]["bias_corrected_mm"] - item[1]["imerg_late_mm"] for item in late], color="#315f80", marker="o", ms=3, ls="--", label="Corrected error")
    else:
        value_ax.text(.5, .5, "No realized IMERG intervals are available for this initialization.", transform=value_ax.transAxes, ha="center", va="center", color="#607080")
    value_ax.set_ylabel("Matched interval rainfall (mm)")
    value_ax.grid(alpha=.2)
    value_handles, value_labels = value_ax.get_legend_handles_labels()
    if value_handles:
        value_ax.legend(value_handles, value_labels, loc="upper left", frameon=False, fontsize=8, ncols=2)
    error_ax.axhline(0, color="#71808b", lw=1)
    error_ax.set_ylabel("Error vs IMERG Late (mm)")
    error_ax.set_xlabel("Forecast interval end (UTC)")
    error_ax.grid(alpha=.2)
    error_handles, error_labels = error_ax.get_legend_handles_labels()
    if error_handles:
        error_ax.legend(error_handles, error_labels, loc="upper left", frameon=False, fontsize=8, ncols=2)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    error_ax.xaxis.set_major_locator(locator)
    error_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    init = _naive(initialization)
    fig.suptitle(f"{city_name} · {model_label} precipitation · init {init:%d %b %Y %H:%M UTC}", fontsize=15, fontweight="bold", color="#172b3a")
    interval_hours = sorted({float(row["interval_hours"]) for row in rows})
    cadence = ", ".join(f"{value:g} h" for value in interval_hours) if interval_hours else "native"
    fig.text(.5, .035, f"Exact native forecast intervals ({cadence}); IMERG sums use complete 30-minute cells only · issue-time bias {summary.get('bias_mm', 0):+.2f} mm from {summary.get('training_intervals', 0)} prior intervals", ha="center", fontsize=8, color="#607080")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def common_grid_city_rows(
    prepared_models: dict[str, xr.Dataset],
    ensemble: dict,
    products: dict[str, xr.Dataset],
    cities,
) -> dict[str, list[dict]]:
    """Return like-for-like six-hour city records for interactive validation."""
    from imerg_grid_ensemble import conservative_regrid_precipitation, exact_forecast_windows

    combined = ensemble["dataset"]
    target_lat = np.asarray(combined["lat"].values, dtype=float)
    target_lon = np.asarray(combined["lon"].values, dtype=float)
    observations = {
        "early": conservative_regrid_precipitation(
            aligned_accumulations(products["early"], hours=6), target_lat, target_lon,
        ),
        "late": ensemble["truth_six_hour"],
    }
    observed_times = {
        product: set(pd.to_datetime(values["valid_time"].values).tz_localize(None))
        for product, values in observations.items()
    }
    source_windows = {
        model: exact_forecast_windows(prepared_models[model])
        for model in ensemble["models"] if model in prepared_models
    }
    rows_by_city = {}
    for city in cities:
        combined_point = combined.sel(lat=float(city.lat), lon=float(city.lon), method="nearest")
        source_points = {
            model: values.sel(lat=float(city.lat), lon=float(city.lon), method="nearest")
            for model, values in source_windows.items()
        }
        observation_points = {
            product: values.sel(lat=float(city.lat), lon=float(city.lon), method="nearest")
            for product, values in observations.items()
        }
        rows = []
        for index, valid_value in enumerate(pd.to_datetime(combined["valid_time"].values).tz_localize(None)):
            valid_time = pd.Timestamp(valid_value)
            model_values = {}
            for model, point in source_points.items():
                raw = float(point["precip_interval_mm"].sel(valid_time=np.datetime64(valid_time)).item())
                bias = float(combined_point["bias_mm"].sel(valid_time=np.datetime64(valid_time), model=model).item())
                weight = float(combined_point["model_weight"].sel(valid_time=np.datetime64(valid_time), model=model).item())
                model_values[model] = {
                    "raw_mm": raw,
                    "bias_mm": bias,
                    "bias_corrected_mm": max(0.0, raw + bias),
                    "weight": weight,
                }
            row = {
                "interval_start_utc": utc_text(combined["interval_start"].isel(valid_time=index).values),
                "valid_time_utc": utc_text(valid_time),
                "lead_hours": float(combined["lead_hours"].isel(valid_time=index).item()),
                "interval_hours": 6.0,
                "combined_mm": float(combined_point["precip_interval_mm"].isel(valid_time=index).item()),
                "models": model_values,
            }
            for product, point in observation_points.items():
                row[f"imerg_{product}_mm"] = (
                    float(point.sel(valid_time=np.datetime64(valid_time)).item())
                    if valid_time in observed_times[product] else None
                )
            rows.append(row)
        rows_by_city[city.name] = rows
    return rows_by_city


def build_imerg_products(
    archive: dict,
    cfg,
    india_load,
    open_dataset: Callable,
    stage: Path,
    *,
    model_labels: dict[str, str],
    model_colors: dict[str, str] | None = None,
) -> dict:
    """Build the complete three-day IMERG and recent native-forecast product."""
    products, window = fetch_recent_imerg(open_dataset, cfg.india_bbox, cfg.cache_root)
    observation_manifest = write_imerg_observation_payloads(products, stage)
    recent_runs = archive["runs"][:RECENT_FORECAST_RUNS]
    forecast_runs, prepared_runs = {}, {}
    matches = {city.name: {} for city in cfg.cities}
    observation_lookups = {
        city.name: {
            product: _observation_point_lookup(dataset, city)
            for product, dataset in products.items()
        }
        for city in cfg.cities
    }
    for run in recent_runs:
        init = _naive(run["initialization_utc"])
        model_manifest = {}
        prepared_runs[run["id"]] = {}
        for model_info in run.get("models", []):
            model = model_info["id"]
            try:
                with india_load.load_india_series_cached(
                    model, cfg, init, horizon_days=5, max_members=8,
                ) as opened:
                    prepared = forecast_interval_fields(opened.load(), init, horizon_days=5)
            except Exception as exc:  # noqa: BLE001 - a partial run remains publishable
                print(f"[{run['id']}] native temporal series unavailable for {model}: {exc}", flush=True)
                continue
            prepared_runs[run["id"]][model] = prepared
            model_manifest[model] = {
                "label": model_labels.get(model, model),
                **write_forecast_temporal_payload(prepared, stage, run["id"], model),
            }
            for city in cfg.cities:
                records = forecast_city_records(prepared, city)
                matched = match_city_records(records, observation_lookups[city.name])
                matches[city.name].setdefault(run["id"], {})[model] = {
                    "initialization_utc": utc_text(init),
                    "rows": matched,
                }
        forecast_runs[run["id"]] = {
            "initialization_utc": utc_text(init),
            "models": model_manifest,
        }
    from imerg_grid_ensemble import build_grid_ensembles

    ensembles = build_grid_ensembles(
        cfg.cache_root,
        prepared_runs,
        {run["id"]: run["initialization_utc"] for run in recent_runs},
        aligned_accumulations(products["late"], hours=6),
        forecast_interval_fields,
    )
    ensemble_manifest = {"model_id": IMERG_COMBINED_MODEL_ID, "runs": {}}
    for run in recent_runs:
        learned = ensembles.get(run["id"])
        if not learned:
            continue
        prepared = learned["dataset"][["temperature_c", "precip_interval_mm"]]
        prepared.attrs = dict(learned["dataset"].attrs)
        prepared_runs[run["id"]][IMERG_COMBINED_MODEL_ID] = prepared
        forecast_runs[run["id"]]["models"][IMERG_COMBINED_MODEL_ID] = {
            "label": IMERG_COMBINED_MODEL_LABEL,
            **write_forecast_temporal_payload(prepared, stage, run["id"], IMERG_COMBINED_MODEL_ID),
        }
        ensemble_manifest["runs"][run["id"]] = {
            "initialization_utc": run["initialization_utc"],
            "source_models": learned["models"],
            "history_case_count": learned["history_case_count"],
            "cache_key": learned["cache_key"],
            "times": learned["times"],
            "city_rows": common_grid_city_rows(
                {model: prepared_runs[run["id"]][model] for model in learned["models"]},
                learned, products, cfg.cities,
            ),
        }
    bias_summaries = apply_causal_bias(matches)
    city_manifest = {}
    for city in cfg.cities:
        run_manifest = {}
        for run in recent_runs:
            model_manifest = {}
            for model, item in matches[city.name].get(run["id"], {}).items():
                relative = Path("assets") / "imerg" / "city_validation" / run["id"] / city.name.lower().replace(" ", "-") / f"{model}.png"
                summary = bias_summaries[city.name][model][run["id"]]
                render_city_validation_plot(
                    item["rows"], city_name=city.name, model_label=model_labels.get(model, model),
                    initialization=run["initialization_utc"], summary=summary, out=stage / relative,
                )
                model_manifest[model] = {
                    "label": model_labels.get(model, model),
                    "image": {"path": relative.as_posix(), "alt": f"{city.name} {model_labels.get(model, model)} precipitation compared with IMERG Early and Late"},
                    "summary": summary,
                    "rows": item["rows"],
                }
            run_manifest[run["id"]] = {
                "initialization_utc": run["initialization_utc"],
                "models": model_manifest,
            }
        city_manifest[city.name] = {
            "latitude": float(city.lat), "longitude": float(city.lon), "runs": run_manifest,
        }
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "window": window,
        "native_time_semantics": "IMERG time is interval start: [time, time + 30 minutes). Values are rate × 1,800 seconds.",
        "forecast_time_semantics": "Forecast valid_time is interval end: (previous valid time, valid_time]. No temporal interpolation is used.",
        "matching": "IMERG native half-hours are summed only when they exactly tile the forecast interval; partial intervals are omitted.",
        "bias_correction": "Source grids use cell- and lead-specific additive IMERG-Late corrections fit only from realized prior intervals, with lead/recency weighting and shrinkage. City native-cadence plots retain the scalar diagnostic correction.",
        "grid_ensemble": {
            "label": IMERG_COMBINED_MODEL_LABEL,
            "temporal_resolution_hours": 6,
            "spatial_matching": "IMERG 0.1° cells are conservatively area-averaged onto the common 0.25° forecast grid.",
            "temporal_matching": "Every model and IMERG are summed over identical complete six-hour intervals.",
            "guardrail": "At each cell and lead, a convex inverse-error blend is retained only when its matched historical MSE is no greater than the best corrected source model; otherwise the historical leader is selected. This is a retrospective safeguard, not an out-of-sample guarantee.",
            **ensemble_manifest,
        },
        "encoding": {
            "precipitation": "gzip-compressed little-endian uint16; value / 100 = mm; 65535 = missing",
            "temperature": "gzip-compressed little-endian uint16; (value - 5000) / 100 = °C; 65535 = missing",
        },
        "products": observation_manifest,
        "forecast_runs": forecast_runs,
        "cities": city_manifest,
    }
