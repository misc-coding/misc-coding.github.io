import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "publish_forecast_archive.py"
SPEC = importlib.util.spec_from_file_location("forecast_archive", MODULE)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def _run(day, models=("gfs",)):
    metadata = []
    for model in models:
        metadata.append({
            "id": model,
            "label": model.upper(),
            "provider": "Test source",
            "members_total": 1,
            "members_used": 1,
            "source_url": "https://example.test",
        })
    return {
        "id": f"202607{day:02d}_00",
        "initialization_utc": f"2026-07-{day:02d}T00:00:00Z",
        "models": metadata,
        "available_models": list(models),
        "missing_models": [model for model in archive.ALL_MODEL_IDS if model not in models],
        "status": "complete" if len(models) == len(archive.ALL_MODEL_IDS) else "partial",
        "lead_days": [{"day": day} for day in archive.LEAD_DAYS],
        "artifacts": [{"kind": "grid", "model": model, "path": f"assets/{model}.bin"} for model in models],
    }


def _validation(runs):
    timeseries = {
        run["id"]: {
            "temperature": {"path": f"assets/validation/{run['id']}-temperature.png", "alt": "Delhi temperature"},
            "precipitation": {"path": f"assets/validation/{run['id']}-precipitation.png", "alt": "Delhi precipitation"},
        }
        for run in runs
    }
    return {
        "cities": {
            "Delhi": {
                "latitude": 28.61,
                "longitude": 77.21,
                "images": {
                    "temperature": {"path": "assets/validation/delhi-temperature.png", "alt": "Delhi temperature"},
                    "precipitation": {"path": "assets/validation/delhi-precipitation.png", "alt": "Delhi precipitation"},
                },
                "summary": {"temperature": {"matched_points": 12}, "precipitation": {"matched_points": 12}},
                "timeseries": timeseries,
            },
        },
    }


def test_stamp_uses_the_public_run_identifier():
    assert archive.stamp(pd.Timestamp("2026-07-30T00:00:00")) == "20260730_00"


def test_archive_manifest_is_schema_two_and_orders_runs_newest_first():
    result = archive.archive_manifest([_run(29), _run(30)])
    assert result["schema_version"] == 2
    assert result["latest_initialization_utc"] == "2026-07-30T00:00:00Z"
    assert [run["id"] for run in result["runs"]] == ["20260730_00", "20260729_00"]


def test_model_availability_is_independent_and_filters_non_midnight():
    class Loader:
        @staticmethod
        def available_inits(model, cfg):
            if model == "broken":
                raise ConnectionError("offline")
            return pd.to_datetime(["2026-07-29T00:00:00", "2026-07-30T06:00:00", "2026-07-30T00:00:00"])

    available, errors = archive.model_availability(("ready", "broken"), None, Loader, timeout_seconds=1)
    assert available["ready"] == {pd.Timestamp("2026-07-29"), pd.Timestamp("2026-07-30")}
    assert available["broken"] == set()
    assert "ConnectionError" in errors["broken"]


def test_normal_candidate_selection_never_backfills_an_older_gap():
    existing = [_run(day, ("gfs", "aifs")) for day in range(23, 30)]
    availability = {
        "gfs": {pd.Timestamp("2026-07-30"), pd.Timestamp("2026-07-22")},
        "aifs": {pd.Timestamp("2026-07-29"), pd.Timestamp("2026-07-21")},
    }
    assert archive.candidate_initializations(availability, existing) == [pd.Timestamp("2026-07-30")]


def test_partial_latest_is_revisited_when_a_late_model_arrives():
    existing = [_run(30, ("gfs",)), _run(29, ("gfs", "aifs"))]
    availability = {
        "gfs": {pd.Timestamp("2026-07-30")},
        "aifs": {pd.Timestamp("2026-07-30")},
    }
    assert archive.candidate_initializations(availability, existing) == [pd.Timestamp("2026-07-30")]


def test_partial_latest_is_not_rebuilt_without_newer_or_late_data():
    existing = [_run(30, ("gfs",))]
    availability = {"gfs": {pd.Timestamp("2026-07-30")}, "aifs": {pd.Timestamp("2026-07-29")}}
    assert archive.candidate_initializations(availability, existing) == []


def test_backfill_targets_missing_recent_dates_not_complete_existing_runs():
    existing = [_run(30, ("gfs", "aifs")), _run(29, ("gfs", "aifs"))]
    availability = {
        "gfs": {pd.Timestamp("2026-08-01"), pd.Timestamp("2026-07-31"), pd.Timestamp("2026-07-30")},
        "aifs": {pd.Timestamp("2026-08-01"), pd.Timestamp("2026-07-31"), pd.Timestamp("2026-07-30")},
    }
    assert archive.candidate_initializations(availability, existing, backfill=True) == [
        pd.Timestamp("2026-08-01"), pd.Timestamp("2026-07-31"),
    ]


def test_normalized_weights_renormalize_and_have_uniform_fallback():
    assert archive._normalized_weights({"gfs": 0.2, "aifs": 0.3}, ["gfs", "aifs"]) == {
        "gfs": 0.4, "aifs": 0.6,
    }
    assert archive._normalized_weights({}, ["gfs", "aifs"]) == {"gfs": 0.5, "aifs": 0.5}


def test_daily_city_series_matches_native_step_accumulation():
    init = pd.Timestamp("2026-07-30T00:00:00")
    times = pd.date_range(init + pd.Timedelta(hours=6), periods=20, freq="6h")
    dataset = xr.Dataset(
        {
            "t2m_C": (("valid_time", "lat", "lon"), np.arange(20, dtype=float).reshape(20, 1, 1) + 20),
            "precip_mmday": (("valid_time", "lat", "lon"), np.full((20, 1, 1), 4.0)),
        },
        coords={"valid_time": times, "lat": [28.6], "lon": [77.2]},
    )
    city = type("City", (), {"lat": 28.6, "lon": 77.2})()
    result = archive._daily_city_series(dataset, city, init)
    assert list(result) == [1, 2, 3, 4, 5]
    assert result[1]["low_c"] == 20
    assert result[1]["high_c"] == 23
    assert result[1]["precip_mm"] == 4.0
    assert result[5]["precip_mm"] == 4.0


def test_weather_symbols_use_accumulated_rain_not_probability():
    assert archive._weather_symbol(0)[1] == "Mostly dry"
    assert archive._weather_symbol(6)[1] == "Rain"
    assert archive._weather_symbol(25)[1] == "Heavy rain"


def test_build_html_has_tabs_every_run_and_unique_ids():
    runs = [_run(day) for day in range(24, 31)]
    html = archive.build_html(archive.archive_manifest(runs), object(), _validation(runs))
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    assert 'data-tab="weather"' in html
    assert 'data-tab="maps"' in html
    assert 'data-tab="validation"' in html
    assert 'id="forecast-canvas"' in html
    assert html.count('id="forecast-canvas"') == 1
    assert 'value="20260730_00"' in html
    assert 'id="validation-image"' in html
    assert 'https://scdlds.ashoka.edu.in/' in html
    assert 'assets/scdlds-logo.jpeg' in html


def test_frontend_contract_covers_all_interactive_features():
    javascript = archive.ARCHIVE_JS
    for selector in (
        "[data-tab]", "#init-select", "#city-select", "[data-weather-variable]",
        "[data-map-variable]", "[data-map-day]", "[data-map-model]", "#map-reset",
        "[data-validation-city]", "[data-validation-variable]", "#match-init-select",
        "[data-match-variable]",
    ):
        assert selector in javascript
    assert "pointerdown" in javascript
    assert "pointermove" in javascript
    assert "wheel" in javascript
    assert "fetch(`assets/map_data/${init}/${mapModel}.bin`)" in javascript
