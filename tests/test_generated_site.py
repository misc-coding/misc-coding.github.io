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
    assert html.count('id="city-grid-map"') == 1
    assert html.count('id="city-grid-time"') == 1
    assert html.count('id="city-grid-models"') == 1
    assert "Forecast valid date and time" in html
    assert "OpenStreetMap contributors" in html or "OpenStreetMap contributors" in (ROOT / "assets/app.js").read_text()
    assert "fixed 0–45 °C scale" in html
    assert "Map rainfall is accumulated only since the previous displayed valid timestamp" in html
    assert "assets/scdlds-logo.jpeg" in html
    assert "coastline overlay" in html
    assert "Natural Earth" in html
    assert 'data-map-model="combined"' in html
    assert "strictly prequential combined model" in html
    assert "https://doi.org/10.1111/rssc.12455" in html
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
    assert weather["schema_version"] == 2
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
                assert day["valid_start_utc"].endswith("Z")
                assert day["valid_end_utc"].endswith("Z")
                assert day["experts"]
                for expert in day["experts"].values():
                    assert -90 <= expert["grid_latitude"] <= 90
                    assert -180 <= expert["grid_longitude"] <= 180
                    assert expert["sample_times_utc"]
                    assert all(value.endswith("Z") for value in expert["sample_times_utc"])
                    assert expert["high_time_utc"] in expert["sample_times_utc"]
                    assert expert["low_time_utc"] in expert["sample_times_utc"]
                    grid = expert["local_grid"]
                    assert 1 <= len(grid["latitudes"]) <= 5
                    assert 1 <= len(grid["longitudes"]) <= 5
                    for variable in ("mean_c", "high_c", "low_c", "precip_mm"):
                        assert len(grid[variable]) == len(grid["latitudes"])
                        assert all(len(row) == len(grid["longitudes"]) for row in grid[variable])


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


def test_combined_model_has_maps_animations_and_causal_recent_error_metadata():
    archive = load_json("assets/forecast_archive.json")
    combination = load_json("assets/combination_manifest.json")
    assert combination["schema_version"] == 2
    assert "only observations" in combination["method"]["causality"]
    assert combination["method"]["fallback"].startswith("Equal weights")
    assert len(combination["method"]["research_sources"]) >= 2
    for run in archive["runs"]:
        item = combination["runs"][run["id"]]
        payload = ROOT / item["map_payload"]
        assert payload.is_file()
        assert payload.stat().st_size > 50_000
        for variable in ("temperature", "precipitation"):
            assert set(item["weights"][variable]) == {"1", "3", "5"}
            for weights in item["weights"][variable].values():
                assert abs(sum(weights.values()) - 1) < 1e-8
        for variable in run["grid_metadata"]["variables"]:
            animation = ROOT / "assets" / "map_animations" / run["id"] / "combined" / f"{variable}.gif"
            assert animation.is_file()
            with Image.open(animation) as opened:
                assert opened.n_frames == 3
        average_payload = ROOT / item["simple_average_map_payload"]
        assert average_payload.is_file()
        assert average_payload.stat().st_size > 50_000
        for variable in run["grid_metadata"]["variables"]:
            animation = ROOT / "assets" / "map_animations" / run["id"] / "simple_average" / f"{variable}.gif"
            assert animation.is_file()
            with Image.open(animation) as opened:
                assert opened.n_frames == 3


def test_validation_has_temperature_and_matched_accumulated_rainfall():
    validation = load_json("assets/validation_manifest.json")
    archive = load_json("assets/forecast_archive.json")
    for city in validation["cities"].values():
        assert set(city["images"]) == {"temperature", "precipitation"}
        assert set(city["timeseries"]) == {run["id"] for run in archive["runs"]}
        for image in city["images"].values():
            assert (ROOT / image["path"]).is_file()
        for variable in ("temperature", "precipitation"):
            assert "combined" in city["summary"][variable]["models"]
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
    assert 'label = f"Valid {valid_time:%d %b %Y %H:%M UTC}"' in (ROOT / "scripts/publish_forecast_archive.py").read_text()
    assert "refreshing observations and online weights" in (ROOT / "scripts/publish_forecast_archive.py").read_text()


def test_readme_records_build_status_and_counter_decision():
    readme = (ROOT / "README.md").read_text()
    assert "Latest initialization" in readme
    assert "Available models" in readme
    assert "Live visit counting is intentionally disabled" in readme
    assert "refreshes observations, validation, and online-combination weights" in readme
