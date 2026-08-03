import json
import re
from pathlib import Path

from PIL import Image


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
    assert html.count('id="map-tooltip"') == 1
    assert html.count('id="map-animation"') == 1
    assert html.count('id="map-legend"') == 1
    assert "fixed 0–45 °C scale" in html
    assert "Map rainfall is accumulated only since the previous published endpoint" in html
    assert "assets/scdlds-logo.jpeg" in html
    assert "coastline overlay" in html
    assert "Natural Earth" in html
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
        assert run["variables"]["temperature"]["plot_scale"] == {"minimum": 0.0, "maximum": 45.0}
        assert run["grid_metadata"]["precipitation_accumulation"] == "previous_endpoint_interval"
        assert run["grid_metadata"]["precipitation_windows"] == [
            "init-to-day-1", "day-1-to-day-3", "day-3-to-day-5",
        ]
        assert "previous published endpoint" in run["variables"]["precipitation"]["units"]
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


def test_local_coastline_overlay_covers_the_forecast_domain():
    coastlines = load_json("assets/coastlines.json")
    assert coastlines["source"] == "Natural Earth 1:50m Coastline"
    assert coastlines["license"] == "Public domain"
    assert coastlines["bounding_box"] == {
        "lon_min": 67.0,
        "lat_min": 6.0,
        "lon_max": 99.0,
        "lat_max": 38.0,
    }
    assert len(coastlines["lines"]) >= 20
    points = [point for line in coastlines["lines"] for point in line]
    assert len(points) >= 1_000
    assert all(67.0 <= longitude <= 99.0 and 6.0 <= latitude <= 38.0 for longitude, latitude in points)
    javascript = (ROOT / "assets/app.js").read_text()
    assert 'fetch("assets/coastlines.json")' in javascript
    assert "drawCoastlines(ctx, meta, width, height)" in javascript


def test_every_model_and_historical_run_has_animated_forecast_gifs():
    archive = load_json("assets/forecast_archive.json")
    expected = 0
    for run in archive["runs"]:
        for model in run["models"]:
            for variable in run["grid_metadata"]["variables"]:
                expected += 1
                animation = ROOT / "assets" / "map_animations" / run["id"] / model["id"] / f"{variable}.gif"
                assert animation.is_file()
                assert animation.stat().st_size > 5_000
                with Image.open(animation) as opened:
                    assert opened.format == "GIF"
                    assert opened.n_frames == 3
                    assert opened.size == (520, 489)
    assert expected >= 7 * 4


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
    assert 'shutil.copy2(coastlines, assets / coastlines.name)' in (ROOT / "scripts/publish_forecast_archive.py").read_text()
    assert "render_map_animations(stage, archive" in (ROOT / "scripts/publish_forecast_archive.py").read_text()


def test_readme_records_build_status_and_counter_decision():
    readme = (ROOT / "README.md").read_text()
    assert "Latest initialization" in readme
    assert "Available models" in readme
    assert "Live visit counting is intentionally disabled" in readme
