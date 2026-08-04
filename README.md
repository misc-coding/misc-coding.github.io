# India Weather Forecasts

A rolling seven-initialization SCDLDS research dashboard with five-day city forecasts, hoverable India maps, recent-error and simple-average mixtures, animated forecasts, and Open-Meteo validation.

- Last successful build: `2026-08-04T08:30:41.121095Z`
- Latest initialization: `2026-08-03T00:00:00Z`
- Available models: WeatherNext 2 / FGN, GFS, GEFS, AIFS, IFS-ENS
- Models still pending: gencast
- Daily publisher: `india-forecast-pages.timer` at 14:00 Asia/Kolkata

The daily publisher refreshes observations, validation, and online-combination weights even when no newer model initialization is available.

## Tests

```bash
/Datastorage/saptarishi.dhanuka_asp25/conda_envs/realtime_dash/bin/python -m pytest -q
node --check assets/app.js
```

Live visit counting is intentionally disabled because no authenticated analytics backend is configured.

Map coastlines use the public-domain [Natural Earth 1:50m coastline](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-coastline/).

City grid-input maps load visible basemap tiles on demand from [OpenStreetMap](https://www.openstreetmap.org/copyright); attribution remains visible on each map.

Temperature maps use a fixed 0–45 °C yellow-to-red scale. Map rainfall is interval accumulation between the exact published valid timestamps shown on the site, while city and validation rainfall retain their stated matched daily accumulation windows.

The combined field uses a causally selected recent-error exponential weighting scheme with equal weighting as a fallback candidate. Weights are learned separately by variable and valid timestamp from observations available at initialization time, pooled across the four validation cities, and applied uniformly over the map. Historical combined validation is prequential. Full learner metadata and weights are in [`assets/combination_manifest.json`](assets/combination_manifest.json).

The simple-average map is a separate baseline: it takes the arithmetic mean of all available source-model values independently at every grid cell and endpoint.

See [`assets/forecast_archive.json`](assets/forecast_archive.json), [`assets/weather_forecast.json`](assets/weather_forecast.json), and [`assets/validation_manifest.json`](assets/validation_manifest.json) for provenance.
