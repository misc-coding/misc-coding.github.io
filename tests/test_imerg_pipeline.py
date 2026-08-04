import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "imerg_pipeline.py"
SPEC = importlib.util.spec_from_file_location("imerg_pipeline", MODULE)
imerg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(imerg)


def native_dataset(*, periods=24, missing_index=None):
    times = pd.date_range("2026-08-01T00:00:00", periods=periods, freq="30min")
    values = np.ones((periods, 2, 2), dtype=np.float32)
    if missing_index is not None:
        times = times.delete(missing_index)
        values = np.delete(values, missing_index, axis=0)
    return xr.Dataset(
        {"precip_mm_30min": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": [10.05, 10.15], "lon": [77.05, 77.15]},
    )


def test_imerg_time_is_the_start_of_a_half_hour_interval():
    start, end = imerg.imerg_interval("2026-08-01T03:30:00Z")
    assert start == pd.Timestamp("2026-08-01T03:30:00")
    assert end == pd.Timestamp("2026-08-01T04:00:00")


def test_rate_conversion_and_aligned_six_hour_accumulation():
    rate = xr.DataArray(
        np.full((24, 1, 1), 1 / 1800, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2026-08-01", periods=24, freq="30min"), "lat": [10], "lon": [77]},
    )
    native = xr.Dataset({"precip_mm_30min": imerg.rate_to_native_accumulation(rate)})
    six_hour = imerg.aligned_accumulations(native, hours=6)
    assert six_hour.shape == (2, 1, 1)
    np.testing.assert_allclose(six_hour.values[:, 0, 0], [12.0, 12.0])
    assert pd.Timestamp(six_hour.interval_start.values[0]) == pd.Timestamp("2026-08-01T00:00:00")
    assert pd.Timestamp(six_hour.valid_time.values[0]) == pd.Timestamp("2026-08-01T06:00:00")


def test_normalized_imerg_cache_strips_non_serializable_source_coordinate_metadata(tmp_path):
    rate = xr.DataArray(
        np.full((2, 2, 2), 1 / 1800, dtype=np.float32),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": pd.date_range("2026-08-01", periods=2, freq="30min"),
            "latitude": [10.15, 10.05], "longitude": [77.05, 77.15],
        },
    )
    rate.latitude.attrs["statistics_approximate"] = {"min": -89.95, "max": 89.95}
    source = xr.Dataset({"precipitation_surface": rate})
    bbox = type("BBox", (), {"lat_min": 10.0, "lat_max": 10.2, "lon_min": 77.0, "lon_max": 77.2})()
    normalized = imerg._select_imerg(source, "2026-08-01T00:00", "2026-08-01T01:00", bbox)
    target = tmp_path / "imerg.nc"
    imerg._write_cache(normalized, target)
    with xr.open_dataset(target) as cached:
        assert cached.precip_mm_30min.shape == (2, 2, 2)
        assert "statistics_approximate" not in cached.lat.attrs


def test_aligned_accumulation_rejects_a_missing_native_half_hour():
    with pytest.raises(ValueError, match="incomplete IMERG 6-hour interval"):
        imerg.aligned_accumulations(native_dataset(missing_index=4), hours=6)


def test_forecast_native_rate_is_integrated_over_exact_previous_to_valid_interval():
    init = pd.Timestamp("2026-08-01T00:00:00")
    series = xr.Dataset(
        {
            "t2m_C": (("valid_time", "lat", "lon"), np.array([[[28]], [[29]], [[30]]], dtype=float)),
            "precip_mmday": (("valid_time", "lat", "lon"), np.full((3, 1, 1), 8.0)),
        },
        coords={"valid_time": pd.date_range(init, periods=3, freq="3h"), "lat": [10.0], "lon": [77.0]},
    )
    result = imerg.forecast_interval_fields(series, init)
    assert list(pd.to_datetime(result.valid_time.values)) == [
        pd.Timestamp("2026-08-01T03:00:00"), pd.Timestamp("2026-08-01T06:00:00"),
    ]
    assert list(pd.to_datetime(result.interval_start.values)) == [
        pd.Timestamp("2026-08-01T00:00:00"), pd.Timestamp("2026-08-01T03:00:00"),
    ]
    np.testing.assert_allclose(result.precip_interval_mm.values[:, 0, 0], [1.0, 1.0])


def test_city_matching_requires_every_exact_half_hour():
    city = type("City", (), {"lat": 10.08, "lon": 77.08})()
    lookup = imerg._observation_point_lookup(native_dataset(periods=12), city)
    records = [{
        "interval_start_utc": "2026-08-01T00:00:00Z",
        "valid_time_utc": "2026-08-01T06:00:00Z",
        "interval_hours": 6.0,
        "temperature_c": 29.0,
        "forecast_mm": 10.0,
    }]
    matched = imerg.match_city_records(records, {"early": lookup, "late": lookup})
    assert matched[0]["imerg_early_mm"] == 12.0
    incomplete = dict(lookup)
    incomplete.pop(pd.Timestamp("2026-08-01T02:00:00"))
    assert imerg.match_city_records(records, {"late": incomplete}) == []


def test_causal_bias_uses_only_truth_realized_by_initialization():
    rows = [
        {"valid_time_utc": "2026-08-01T06:00:00Z", "forecast_mm": 2.0, "imerg_late_mm": 5.0},
        {"valid_time_utc": "2026-08-02T06:00:00Z", "forecast_mm": 1.0, "imerg_late_mm": 30.0},
    ]
    bias, count = imerg.causal_run_bias(rows, "2026-08-02T00:00:00Z", prior_weight=0)
    assert count == 1
    assert bias == pytest.approx(3.0)


def test_gzip_temporal_payload_round_trips_and_preserves_native_shape(tmp_path):
    prepared = xr.Dataset(
        {
            "temperature_c": (("valid_time", "lat", "lon"), np.array([[[30.0, 31.0]]])),
            "precip_interval_mm": (("valid_time", "lat", "lon"), np.array([[[1.25, 2.5]]])),
        },
        coords={
            "valid_time": [np.datetime64("2026-08-01T03:00")],
            "interval_start": ("valid_time", [np.datetime64("2026-08-01T00:00")]),
            "lat": [10.0], "lon": [77.0, 77.25],
        },
    )
    metadata = imerg.write_forecast_temporal_payload(prepared, tmp_path, "20260801_00", "gefs")
    with gzip.open(tmp_path / metadata["path"], "rb") as opened:
        payload = np.frombuffer(opened.read(), dtype="<u2")
    assert metadata["shape"] == [1, 1, 2]
    assert metadata["times"][0]["interval_hours"] == 3.0
    assert payload.tolist() == [8000, 8100, 125, 250]
