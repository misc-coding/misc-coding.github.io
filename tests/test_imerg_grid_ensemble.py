import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "imerg_grid_ensemble.py"
SPEC = importlib.util.spec_from_file_location("imerg_grid_ensemble", MODULE)
grid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid
SPEC.loader.exec_module(grid)


def prepared_hourly(periods=12):
    init = pd.Timestamp("2026-08-01T00:00:00")
    valid = pd.date_range(init + pd.Timedelta(hours=1), periods=periods, freq="h")
    starts = pd.date_range(init, periods=periods, freq="h")
    return xr.Dataset(
        {
            "precip_interval_mm": (("valid_time", "lat", "lon"), np.ones((periods, 2, 2), dtype=np.float32)),
            "temperature_c": (("valid_time", "lat", "lon"), np.arange(periods, dtype=np.float32)[:, None, None] + np.ones((periods, 2, 2))),
        },
        coords={
            "valid_time": valid,
            "interval_start": ("valid_time", starts),
            "lat": [10.0, 10.25], "lon": [77.0, 77.25],
        },
        attrs={"initialization_utc": "2026-08-01T00:00:00Z"},
    )


def test_exact_forecast_windows_sum_complete_native_intervals():
    result = grid.exact_forecast_windows(prepared_hourly())
    assert list(result.lead_hours.values) == [6.0, 12.0]
    np.testing.assert_allclose(result.precip_interval_mm.values, 6.0)
    np.testing.assert_allclose(result.temperature_c.values[0], 6.0)
    assert pd.Timestamp(result.interval_start.values[1]) == pd.Timestamp("2026-08-01T06:00:00")


def test_conservative_regrid_preserves_constant_rainfall():
    source = xr.DataArray(
        np.full((2, 10, 10), 7.5, dtype=np.float32),
        dims=("valid_time", "lat", "lon"),
        coords={
            "valid_time": pd.date_range("2026-08-01T06:00", periods=2, freq="6h"),
            "lat": np.arange(.05, 1.0, .1), "lon": np.arange(70.05, 71.0, .1),
        },
    )
    result = grid.conservative_regrid_precipitation(
        source, np.array([.25, .5, .75]), np.array([70.25, 70.5, 70.75]),
    )
    np.testing.assert_allclose(result.values, 7.5, rtol=1e-6)


def test_bias_field_excludes_forecasts_not_realized_by_cutoff():
    early = grid.HistoricalCase(
        initialization=pd.Timestamp("2026-07-31"), valid_time=pd.Timestamp("2026-08-01T06:00"),
        lead_hours=30, truth=np.full((2, 2), 5.0), forecasts={"a": np.full((2, 2), 2.0)},
    )
    future = grid.HistoricalCase(
        initialization=pd.Timestamp("2026-08-01"), valid_time=pd.Timestamp("2026-08-03T06:00"),
        lead_hours=54, truth=np.full((2, 2), 100.0), forecasts={"a": np.zeros((2, 2))},
    )
    bias, count = grid.estimate_bias_field(
        [early, future], "a", 30, "2026-08-02T00:00", prior_weight=0,
    )
    assert count == 1
    np.testing.assert_allclose(bias, 3.0)


def test_guarded_combination_is_never_worse_than_historical_best_at_each_cell():
    truth = np.array([
        [[1.0, 4.0], [2.0, 8.0]],
        [[2.0, 5.0], [3.0, 9.0]],
        [[3.0, 6.0], [4.0, 10.0]],
    ])
    predictions = np.stack([
        truth + .2,
        truth + np.array([[[1.0, -1.0], [.5, -2.0]]]),
        truth + np.array([[[-.5, .5], [-1.0, 1.0]]]),
    ], axis=1)
    weights, diagnostics = grid.guarded_weights(predictions, truth, np.ones(3))
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)
    mask = diagnostics["history_mask"]
    assert np.all(diagnostics["combined_mse"][mask] <= diagnostics["best_mse"][mask] + 1e-6)


def test_guardrail_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="incompatible"):
        grid.guarded_weights(np.ones((2, 3, 1, 1)), np.ones((3, 1, 1)), np.ones(2))
