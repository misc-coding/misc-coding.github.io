import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative):
    return json.loads((ROOT / relative).read_text())


def test_generated_html_has_one_of_every_interactive_surface():
    html = (ROOT / "index.html").read_text()
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    assert html.count('id="forecast-canvas"') == 1
    assert html.count('data-panel="weather"') == 1
    assert html.count('data-panel="maps"') == 1
    assert html.count('data-panel="validation"') == 1
    assert html.count('data-panel="method"') == 1
    assert "assets/scdlds-logo.jpeg" in html
    assert "https://scdlds.ashoka.edu.in/" in html
    assert "precipitation probability" not in html.lower()


def test_archive_is_sorted_retained_and_every_grid_exists():
    archive = load_json("assets/forecast_archive.json")
    assert archive["schema_version"] == 2
    assert len(archive["runs"]) == 7
    times = [run["initialization_utc"] for run in archive["runs"]]
    assert times == sorted(times, reverse=True)
    assert archive["latest_initialization_utc"] == times[0]
    for run in archive["runs"]:
        models = {model["id"] for model in run["models"]}
        artifacts = {item["model"] for item in run["artifacts"]}
        assert models
        assert artifacts == models
        for item in run["artifacts"]:
            path = ROOT / item["path"]
            assert path.is_file()
            assert path.stat().st_size > 50_000


def test_weather_manifest_has_five_daily_values_for_every_city_and_run():
    archive = load_json("assets/forecast_archive.json")
    weather = load_json("assets/weather_forecast.json")
    validation = load_json("assets/validation_manifest.json")
    for run in archive["runs"]:
        product = weather["runs"][run["id"]]
        for city in validation["cities"]:
            city_product = product["cities"][city]
            assert [day["day"] for day in city_product["days"]] == [1, 2, 3, 4, 5]
            assert abs(sum(city_product["temperature_weights"].values()) - 1) < 1e-8
            assert abs(sum(city_product["precipitation_weights"].values()) - 1) < 1e-8
            for day in city_product["days"]:
                assert day["high_c"] >= day["low_c"]
                assert day["precip_mm"] >= 0
                assert "precip_probability" not in day


def test_validation_has_temperature_and_matched_accumulated_rainfall():
    validation = load_json("assets/validation_manifest.json")
    archive = load_json("assets/forecast_archive.json")
    for city in validation["cities"].values():
        assert set(city["images"]) == {"temperature", "precipitation"}
        assert set(city["timeseries"]) == {run["id"] for run in archive["runs"]}
        for image in city["images"].values():
            assert (ROOT / image["path"]).is_file()
        for run in city["timeseries"].values():
            assert set(run) == {"temperature", "precipitation"}
            for image in run.values():
                assert (ROOT / image["path"]).is_file()


def test_daily_automation_is_scheduled_tested_and_bounded():
    timer = (ROOT / "systemd/india-forecast-pages.timer").read_text()
    service = (ROOT / "systemd/india-forecast-pages.service").read_text()
    publisher = (ROOT / "scripts/daily_forecast_publish.sh").read_text()
    assert "14:00:00 Asia/Kolkata" in timer
    assert "Persistent=true" in timer
    assert "TimeoutStartSec=50min" in service
    assert "Restart=on-failure" in service
    assert '"$PYTHON" -m pytest -q' in publisher
    assert "node --check assets/app.js" in publisher
    assert "git push origin main" in publisher


def test_readme_records_build_status_and_counter_decision():
    readme = (ROOT / "README.md").read_text()
    assert "Latest initialization" in readme
    assert "Available models" in readme
    assert "Live visit counting is intentionally disabled" in readme
