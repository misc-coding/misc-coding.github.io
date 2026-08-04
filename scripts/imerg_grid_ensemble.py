#!/usr/bin/env python3
"""Causal grid-cell precipitation calibration and guarded model combination.

Forecast rates are first integrated to exact six-hour windows on the common
0.25-degree model grid.  IMERG half-hour accumulations are summed over the same
windows and conservatively area-averaged from 0.1 degrees to that grid.  Every
published target is trained only from forecast/observation pairs whose valid
time is no later than the target initialization.

The historical guardrail is deliberately retrospective, not a promise about
future weather: a candidate inverse-error blend is retained at a cell/lead only
when its matched historical MSE is no larger than the best corrected source
model's MSE.  Otherwise the historical leader receives unit weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import xarray as xr


COMMON_HOURS = 6
LEAD_BANDWIDTH_HOURS = 12.0
RECENCY_HALF_LIFE_DAYS = 7.0
BIAS_PRIOR_WEIGHT = 4.0
MIN_SPATIAL_COVERAGE = 0.98


def _naive(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or len(centres) < 2 or np.any(np.diff(centres) <= 0):
        raise ValueError("grid centres must be a strictly increasing one-dimensional array")
    middle = (centres[:-1] + centres[1:]) / 2.0
    return np.concatenate([[centres[0] - (middle[0] - centres[0])], middle, [centres[-1] + (centres[-1] - middle[-1])]])


def _overlap_matrix(source_edges: np.ndarray, target_edges: np.ndarray, *, latitude: bool) -> np.ndarray:
    lower = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    upper = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    if latitude:
        overlap = np.maximum(
            0.0,
            np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower)),
        )
    else:
        overlap = np.maximum(0.0, np.deg2rad(upper - lower))
    return overlap.astype(np.float64)


def conservative_regrid_precipitation(
    source: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    min_coverage: float = MIN_SPATIAL_COVERAGE,
) -> xr.DataArray:
    """Area-average a rectilinear precipitation field onto a target grid.

    Leading dimensions are retained. Target edge cells without nearly complete
    source coverage are left missing rather than extrapolated.
    """
    if not {"lat", "lon"}.issubset(source.dims):
        raise ValueError("source must contain lat and lon dimensions")
    source = source.sortby("lat").sortby("lon")
    source_lat = np.asarray(source["lat"].values, dtype=float)
    source_lon = np.asarray(source["lon"].values, dtype=float)
    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)
    lat_weights = _overlap_matrix(_cell_edges(source_lat), _cell_edges(target_lat), latitude=True)
    lon_weights = _overlap_matrix(_cell_edges(source_lon), _cell_edges(target_lon), latitude=False)
    target_lat_area = np.sin(np.deg2rad(_cell_edges(target_lat)[1:])) - np.sin(np.deg2rad(_cell_edges(target_lat)[:-1]))
    target_lon_width = np.deg2rad(np.diff(_cell_edges(target_lon)))
    target_area = target_lat_area[:, None] * target_lon_width[None, :]

    leading_dims = [dim for dim in source.dims if dim not in {"lat", "lon"}]
    ordered = source.transpose(*leading_dims, "lat", "lon")
    values = np.asarray(ordered.values, dtype=np.float64)
    leading_shape = values.shape[:-2]
    flattened = values.reshape((-1, len(source_lat), len(source_lon)))
    finite = np.isfinite(flattened)
    clean = np.where(finite, flattened, 0.0)
    numerator = np.einsum("ai,tij,bj->tab", lat_weights, clean, lon_weights, optimize=True)
    denominator = np.einsum("ai,tij,bj->tab", lat_weights, finite.astype(np.float64), lon_weights, optimize=True)
    coverage = denominator / target_area[None, :, :]
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    result[coverage < min_coverage] = np.nan
    result = result.reshape((*leading_shape, len(target_lat), len(target_lon))).astype(np.float32)
    coords = {dim: ordered[dim] for dim in leading_dims}
    coords.update({"lat": target_lat, "lon": target_lon})
    output = xr.DataArray(result, dims=[*leading_dims, "lat", "lon"], coords=coords, name=source.name)
    for coordinate in source.coords:
        if coordinate not in output.coords and set(source[coordinate].dims).issubset(set(leading_dims)):
            output = output.assign_coords({coordinate: source[coordinate]})
    output.attrs = {**source.attrs, "spatial_matching": "conservative area average to common model grid"}
    return output


def exact_forecast_windows(prepared: xr.Dataset, *, hours: int = COMMON_HOURS) -> xr.Dataset:
    """Aggregate complete native forecast intervals to initialization-relative windows."""
    if hours <= 0:
        raise ValueError("hours must be positive")
    initialization = _naive(prepared.attrs["initialization_utc"])
    valid = pd.to_datetime(prepared["valid_time"].values).tz_localize(None)
    starts = pd.to_datetime(prepared["interval_start"].values).tz_localize(None)
    last = valid.max()
    ends = pd.date_range(initialization + pd.Timedelta(hours=hours), last, freq=f"{hours}h")
    rain_fields, temperature_fields, kept_ends, kept_starts, leads = [], [], [], [], []
    for end in ends:
        start = end - pd.Timedelta(hours=hours)
        chosen = np.flatnonzero((starts >= start) & (valid <= end) & (valid > start))
        if not len(chosen):
            continue
        ordered = chosen[np.argsort(valid[chosen].values)]
        chosen_starts, chosen_ends = starts[ordered], valid[ordered]
        continuous = (
            chosen_starts[0] == start
            and chosen_ends[-1] == end
            and all(chosen_ends[index] == chosen_starts[index + 1] for index in range(len(ordered) - 1))
        )
        exact_temperature = np.flatnonzero(valid == end)
        if not continuous or len(exact_temperature) != 1:
            continue
        rain_fields.append(prepared["precip_interval_mm"].isel(valid_time=ordered).sum("valid_time", skipna=False))
        temperature_fields.append(prepared["temperature_c"].isel(valid_time=int(exact_temperature[0])))
        kept_starts.append(np.datetime64(start, "ns"))
        kept_ends.append(np.datetime64(end, "ns"))
        leads.append(float((end - initialization) / pd.Timedelta(hours=1)))
    if not rain_fields:
        raise ValueError(f"forecast has no complete {hours}-hour windows")
    coordinate = xr.DataArray(np.asarray(kept_ends), dims="valid_time", name="valid_time")
    result = xr.Dataset({
        "precip_interval_mm": xr.concat(rain_fields, dim=coordinate, coords="minimal", compat="override").astype(np.float32),
        "temperature_c": xr.concat(temperature_fields, dim=coordinate, coords="minimal", compat="override").astype(np.float32),
    })
    result = result.assign_coords(
        interval_start=("valid_time", np.asarray(kept_starts)),
        lead_hours=("valid_time", np.asarray(leads, dtype=np.float32)),
    )
    result.attrs = {
        "initialization_utc": initialization.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rain_semantics": f"sum of complete native intervals over exact {hours}-hour windows",
    }
    return result


@dataclass
class HistoricalCase:
    initialization: pd.Timestamp
    valid_time: pd.Timestamp
    lead_hours: float
    truth: np.ndarray
    forecasts: dict[str, np.ndarray]


def _lead_weight(source_lead: float, target_lead: float) -> float:
    distance = abs(float(source_lead) - float(target_lead))
    return float(np.exp(-distance / LEAD_BANDWIDTH_HOURS)) if distance <= 2 * LEAD_BANDWIDTH_HOURS else 0.0


def estimate_bias_field(
    cases: list[HistoricalCase],
    model: str,
    target_lead: float,
    as_of,
    *,
    prior_weight: float = BIAS_PRIOR_WEIGHT,
) -> tuple[np.ndarray, int]:
    """Estimate a shrunken cell bias from cases realized by ``as_of``."""
    cutoff = _naive(as_of)
    eligible = [
        case for case in cases
        if case.initialization < cutoff and case.valid_time <= cutoff and model in case.forecasts
        and _lead_weight(case.lead_hours, target_lead) > 0
    ]
    if not eligible:
        shape = next(iter(cases)).truth.shape if cases else (0, 0)
        return np.zeros(shape, dtype=np.float32), 0
    numerator = np.zeros_like(eligible[0].truth, dtype=np.float64)
    denominator = np.full_like(eligible[0].truth, float(prior_weight), dtype=np.float64)
    for case in eligible:
        age_days = max(0.0, float((cutoff - case.valid_time) / pd.Timedelta(days=1)))
        weight = _lead_weight(case.lead_hours, target_lead) * np.exp(-np.log(2.0) * age_days / RECENCY_HALF_LIFE_DAYS)
        residual = case.truth - case.forecasts[model]
        valid = np.isfinite(residual)
        numerator[valid] += weight * residual[valid]
        denominator[valid] += weight
    bias = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.clip(bias, -50.0, 50.0).astype(np.float32), len(eligible)


def guarded_weights(
    predictions: np.ndarray,
    truth: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit cellwise weights with an in-history best-expert loss guardrail.

    ``predictions`` has shape sample, model, lat, lon and ``truth`` has shape
    sample, lat, lon. The returned weights have shape model, lat, lon.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if predictions.ndim != 4 or truth.shape != (predictions.shape[0], *predictions.shape[2:]):
        raise ValueError("incompatible historical prediction/truth shapes")
    weighted = sample_weights[:, None, None, None]
    errors = predictions - truth[:, None, :, :]
    valid = np.isfinite(errors)
    denominator = np.sum(weighted * valid, axis=0)
    mse = np.divide(
        np.sum(weighted * np.where(valid, errors ** 2, 0.0), axis=0),
        denominator,
        out=np.full(predictions.shape[1:], np.nan),
        where=denominator > 0,
    )
    best_index = np.nanargmin(np.where(np.isfinite(mse), mse, np.inf), axis=0)
    best_mse = np.take_along_axis(mse, best_index[None, :, :], axis=0)[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        scale = np.nanmedian(mse, axis=0)
    scale = np.where(np.isfinite(scale), np.maximum(scale * 0.15, 0.05), 1.0)
    scores = np.exp(-np.divide(mse, scale[None, :, :], out=np.full_like(mse, np.inf), where=scale[None, :, :] > 0))
    scores = np.where(np.isfinite(scores), scores, 0.0)
    score_sum = scores.sum(axis=0)
    candidate = np.divide(scores, score_sum[None, :, :], out=np.zeros_like(scores), where=score_sum[None, :, :] > 0)
    candidate_prediction = np.sum(candidate[None, :, :, :] * np.where(np.isfinite(predictions), predictions, 0.0), axis=1)
    candidate_valid = np.isfinite(truth) & np.all(np.isfinite(predictions), axis=1)
    candidate_denominator = np.sum(sample_weights[:, None, None] * candidate_valid, axis=0)
    candidate_mse = np.divide(
        np.sum(sample_weights[:, None, None] * np.where(candidate_valid, (candidate_prediction - truth) ** 2, 0.0), axis=0),
        candidate_denominator,
        out=np.full_like(best_mse, np.nan),
        where=candidate_denominator > 0,
    )
    leader = np.zeros_like(candidate)
    np.put_along_axis(leader, best_index[None, :, :], 1.0, axis=0)
    use_blend = np.isfinite(candidate_mse) & np.isfinite(best_mse) & (candidate_mse <= best_mse + 1e-8)
    final = np.where(use_blend[None, :, :], candidate, leader)
    no_history = ~np.isfinite(best_mse)
    final[:, no_history] = 1.0 / predictions.shape[1]
    final_prediction = np.sum(final[None, :, :, :] * np.where(np.isfinite(predictions), predictions, 0.0), axis=1)
    final_mse = np.divide(
        np.sum(sample_weights[:, None, None] * np.where(candidate_valid, (final_prediction - truth) ** 2, 0.0), axis=0),
        candidate_denominator,
        out=np.full_like(best_mse, np.nan),
        where=candidate_denominator > 0,
    )
    checked = np.isfinite(final_mse) & np.isfinite(best_mse)
    if np.any(final_mse[checked] > best_mse[checked] + 1e-6):
        raise RuntimeError("historical best-model guardrail was violated")
    return final.astype(np.float32), {
        "best_mse": best_mse.astype(np.float32),
        "combined_mse": final_mse.astype(np.float32),
        "blend_mask": use_blend,
        "history_mask": checked,
    }


def _cached_series(cache_root: Path, models: list[str], truth_times: set[pd.Timestamp], prepare: Callable) -> list[dict]:
    groups: dict[tuple[pd.Timestamp, pd.Timestamp], dict] = {}
    india = Path(cache_root) / "india"
    if not india.is_dir():
        return []
    for run_dir in sorted(india.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith("_00"):
            continue
        try:
            initialization = pd.to_datetime(run_dir.name, format="%Y%m%d_%H")
        except ValueError:
            continue
        for model in models:
            path = run_dir / f"{model}_series.nc"
            if not path.is_file():
                continue
            try:
                with xr.open_dataset(path, decode_timedelta=False) as opened:
                    windows = exact_forecast_windows(prepare(opened.load(), initialization, horizon_days=5))
            except (OSError, ValueError, KeyError):
                continue
            for index, valid_value in enumerate(pd.to_datetime(windows["valid_time"].values).tz_localize(None)):
                valid_time = pd.Timestamp(valid_value)
                if valid_time not in truth_times:
                    continue
                key = (initialization, valid_time)
                item = groups.setdefault(key, {
                    "initialization": initialization,
                    "valid_time": valid_time,
                    "lead_hours": float(windows["lead_hours"].isel(valid_time=index).item()),
                    "forecasts": {},
                })
                item["forecasts"][model] = np.asarray(windows["precip_interval_mm"].isel(valid_time=index).values, dtype=np.float32)
    return [item for item in groups.values() if item["forecasts"]]


def build_grid_ensembles(
    cache_root: Path,
    prepared_runs: dict[str, dict[str, xr.Dataset]],
    initializations: dict[str, object],
    truth_six_hour: xr.DataArray,
    prepare: Callable,
) -> dict[str, dict]:
    """Build guarded six-hour precipitation combinations for target runs."""
    if not prepared_runs:
        return {}
    first = next(dataset for models in prepared_runs.values() for dataset in models.values())
    target_lat = np.asarray(first["lat"].values, dtype=float)
    target_lon = np.asarray(first["lon"].values, dtype=float)
    truth = conservative_regrid_precipitation(truth_six_hour, target_lat, target_lon)
    truth_times = pd.to_datetime(truth["valid_time"].values).tz_localize(None)
    truth_lookup = {
        pd.Timestamp(value): np.asarray(truth.isel(valid_time=index).values, dtype=np.float32)
        for index, value in enumerate(truth_times)
    }
    all_models = sorted(set.intersection(*[
        set(models) for models in prepared_runs.values() if models
    ]))
    raw_history = _cached_series(Path(cache_root), all_models, set(truth_lookup), prepare)
    cases = [
        HistoricalCase(
            initialization=item["initialization"], valid_time=item["valid_time"],
            lead_hours=item["lead_hours"], truth=truth_lookup[item["valid_time"]], forecasts=item["forecasts"],
        )
        for item in raw_history
    ]
    outputs = {}
    for run_id, prepared_models in prepared_runs.items():
        models = [model for model in all_models if model in prepared_models]
        initialization = _naive(initializations[run_id])
        target_windows = {model: exact_forecast_windows(prepared_models[model]) for model in models}
        common_times = sorted(set.intersection(*[
            set(pd.to_datetime(dataset["valid_time"].values).tz_localize(None))
            for dataset in target_windows.values()
        ]))
        rain_fields, temperature_fields, weight_fields, bias_fields, starts, leads, summaries = [], [], [], [], [], [], []
        performance_cases = [
            case for case in cases
            if case.initialization < initialization and case.valid_time <= initialization
            and all(model in case.forecasts for model in models)
        ]
        corrected_history = []
        for case in performance_cases:
            corrected = []
            for model in models:
                historical_bias, _ = estimate_bias_field(cases, model, case.lead_hours, case.initialization)
                corrected.append(np.maximum(0.0, case.forecasts[model] + historical_bias))
            corrected_history.append(np.stack(corrected))
        for valid_time in common_times:
            first_window = target_windows[models[0]].sel(valid_time=np.datetime64(valid_time))
            lead = float(first_window["lead_hours"].item())
            raw_target = np.stack([
                np.asarray(target_windows[model]["precip_interval_mm"].sel(valid_time=np.datetime64(valid_time)).values, dtype=np.float32)
                for model in models
            ])
            temperatures = np.stack([
                np.asarray(target_windows[model]["temperature_c"].sel(valid_time=np.datetime64(valid_time)).values, dtype=np.float32)
                for model in models
            ])
            target_bias, bias_counts = [], []
            for model in models:
                field, count = estimate_bias_field(cases, model, lead, initialization)
                target_bias.append(field)
                bias_counts.append(count)
            target_bias = np.stack(target_bias)
            corrected_target = np.maximum(0.0, raw_target + target_bias)
            selected = [index for index, case in enumerate(performance_cases) if _lead_weight(case.lead_hours, lead) > 0]
            if selected:
                history_predictions = np.stack([corrected_history[index] for index in selected])
                history_truth = np.stack([performance_cases[index].truth for index in selected])
                history_weights = np.asarray([
                    _lead_weight(performance_cases[index].lead_hours, lead)
                    * np.exp(-np.log(2.0) * max(0.0, float((initialization - performance_cases[index].valid_time) / pd.Timedelta(days=1))) / RECENCY_HALF_LIFE_DAYS)
                    for index in selected
                ])
                learned_weights, diagnostics = guarded_weights(history_predictions, history_truth, history_weights)
            else:
                learned_weights = np.full_like(corrected_target, 1.0 / len(models), dtype=np.float32)
                diagnostics = {
                    "best_mse": np.full(corrected_target.shape[1:], np.nan, dtype=np.float32),
                    "combined_mse": np.full(corrected_target.shape[1:], np.nan, dtype=np.float32),
                    "blend_mask": np.zeros(corrected_target.shape[1:], dtype=bool),
                    "history_mask": np.zeros(corrected_target.shape[1:], dtype=bool),
                }
            combined = np.sum(learned_weights * corrected_target, axis=0)
            rain_fields.append(combined.astype(np.float32))
            temperature_fields.append(np.nanmean(temperatures, axis=0).astype(np.float32))
            weight_fields.append(learned_weights)
            bias_fields.append(target_bias.astype(np.float32))
            starts.append(np.datetime64(first_window["interval_start"].values, "ns"))
            leads.append(lead)
            history_mask = diagnostics["history_mask"]
            best_mse = diagnostics["best_mse"]
            combined_mse = diagnostics["combined_mse"]
            summaries.append({
                "interval_start_utc": _naive(first_window["interval_start"].values).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "valid_time_utc": pd.Timestamp(valid_time).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lead_hours": lead,
                "history_cases": len(selected),
                "bias_training_cases": dict(zip(models, bias_counts)),
                "mean_bias_mm": dict(zip(models, [float(np.nanmean(field)) for field in target_bias])),
                "regional_weights": dict(zip(models, [float(np.nanmean(field)) for field in learned_weights])),
                "blend_cell_fraction": float(np.mean(diagnostics["blend_mask"][history_mask])) if np.any(history_mask) else None,
                "historical_combined_rmse_mm": float(np.sqrt(np.nanmean(combined_mse[history_mask]))) if np.any(history_mask) else None,
                "historical_best_model_rmse_mm": float(np.sqrt(np.nanmean(best_mse[history_mask]))) if np.any(history_mask) else None,
                "historical_guardrail_satisfied": bool(
                    np.all(combined_mse[history_mask] <= best_mse[history_mask] + 1e-6)
                ) if np.any(history_mask) else None,
            })
        valid_coordinate = np.asarray(common_times, dtype="datetime64[ns]")
        dataset = xr.Dataset({
            "precip_interval_mm": (("valid_time", "lat", "lon"), np.stack(rain_fields)),
            "temperature_c": (("valid_time", "lat", "lon"), np.stack(temperature_fields)),
            "model_weight": (("valid_time", "model", "lat", "lon"), np.stack(weight_fields)),
            "bias_mm": (("valid_time", "model", "lat", "lon"), np.stack(bias_fields)),
        }, coords={
            "valid_time": valid_coordinate, "model": models, "lat": target_lat, "lon": target_lon,
            "interval_start": ("valid_time", np.asarray(starts, dtype="datetime64[ns]")),
            "lead_hours": ("valid_time", np.asarray(leads, dtype=np.float32)),
        })
        dataset.attrs = {
            "initialization_utc": initialization.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": "cell-and-lead additive bias correction plus retrospective best-expert-guarded convex combination",
            "temporal_matching": "exact complete six-hour forecast and IMERG accumulation windows",
            "spatial_matching": "IMERG conservatively area-averaged from 0.1 degrees to the 0.25-degree forecast grid",
        }
        signature = hashlib.sha256(json.dumps({
            "run": run_id, "models": models,
            "truth_start": str(truth_times.min()), "truth_end": str(truth_times.max()),
            "history_cases": len(cases),
        }, sort_keys=True).encode()).hexdigest()[:16]
        outputs[run_id] = {
            "dataset": dataset,
            "truth_six_hour": truth,
            "models": models,
            "times": summaries,
            "history_case_count": len(cases),
            "cache_key": signature,
        }
    return outputs
