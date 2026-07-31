import importlib.util
from pathlib import Path

import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "publish_forecast_archive.py"
SPEC = importlib.util.spec_from_file_location("forecast_archive", MODULE)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def test_stamp_uses_the_public_run_identifier():
    assert archive.stamp(pd.Timestamp("2026-07-30T00:00:00")) == "20260730_00"


def test_archive_manifest_orders_runs_newest_first():
    runs = [
        {"id": "20260729_00", "initialization_utc": "2026-07-29T00:00:00Z"},
        {"id": "20260730_00", "initialization_utc": "2026-07-30T00:00:00Z"},
    ]
    result = archive.archive_manifest(runs)
    assert result["latest_initialization_utc"] == "2026-07-30T00:00:00Z"
    assert [run["id"] for run in result["runs"]] == ["20260730_00", "20260729_00"]


def test_common_midnight_inits_intersects_every_model_and_filters_non_midnight():
    class Loader:
        @staticmethod
        def available_inits(model, cfg):
            values = {
                "a": ["2026-07-29T00:00:00", "2026-07-30T06:00:00", "2026-07-30T00:00:00"],
                "b": ["2026-07-28T00:00:00", "2026-07-29T00:00:00", "2026-07-30T00:00:00"],
            }
            return pd.to_datetime(values[model])

    result = archive.common_midnight_inits(("a", "b"), None, Loader)
    assert result[:2] == [pd.Timestamp("2026-07-30T00:00:00"), pd.Timestamp("2026-07-29T00:00:00")]


def test_build_html_includes_each_run_in_the_selector():
    class Renderer:
        @staticmethod
        def _view_sections(run):
            return '<section class="forecast-view" data-variable="temperature" data-day="1"></section>'

    run = {
        "models": [{"label": "GFS", "provider": "NOAA", "members_total": 1, "source_url": "https://example.test"}],
    }
    runs = [
        {**run, "id": f"202607{day:02d}_00", "initialization_utc": f"2026-07-{day:02d}T00:00:00Z"}
        for day in range(24, 31)
    ]
    timeseries = {
        f"202607{day:02d}_00": {
            "temperature": {"path": f"assets/validation/timeseries/202607{day:02d}_00/delhi-temperature.png", "alt": "Delhi temperature"},
            "precipitation": {"path": f"assets/validation/timeseries/202607{day:02d}_00/delhi-precipitation.png", "alt": "Delhi precipitation"},
        }
        for day in range(24, 31)
    }
    validation = {
        "cities": {
            "Delhi": {
                "images": {
                    "temperature": {"path": "assets/validation/delhi-temperature.png", "alt": "Delhi temperature"},
                    "precipitation": {"path": "assets/validation/delhi-precipitation.png", "alt": "Delhi precipitation"},
                },
                "summary": {"temperature": {"matched_points": 12}, "precipitation": {"matched_points": 12}},
                "timeseries": timeseries,
            },
        },
    }
    html = archive.build_html(archive.archive_manifest(runs), Renderer, validation)
    assert 'id="run-select"' in html
    assert 'data-init="20260730_00"' in html
    assert 'id="validation-image"' in html
    assert 'id="match-image"' in html
