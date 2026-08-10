# India Weather Forecasts

A rolling seven-initialization SCDLDS research dashboard with five-day city forecasts, native-time India maps, recent-error and simple-average mixtures, and matched Open-Meteo and IMERG validation.

- Last successful build: `2026-08-10T08:42:35.358725Z`
- Latest initialization: `2026-08-10T00:00:00Z`
- Available models: GFS, GEFS, AIFS, IFS-ENS
- Models still pending: weathernext2, gencast
- Daily publisher: `india-forecast-pages.timer` at 14:00 Asia/Kolkata

The daily publisher refreshes Open-Meteo and IMERG observations, native-time forecasts for the latest three initializations, validation, and online-combination weights even when no newer model initialization is available.

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

NASA GPM IMERG V07 [Early](https://dynamical.org/catalog/nasa-imerg-analysis-early/) and [Late](https://dynamical.org/catalog/nasa-imerg-analysis-late/) Run precipitation is published for a rolling six-day window at its native 0.1° and 30-minute resolution, plus exact UTC-aligned six-hour accumulations. IMERG timestamps are interval starts; forecasts are matched only when complete half-hours exactly tile the forecast interval. For the six-hour calibrated combination, IMERG is conservatively area-averaged onto the common 0.25° grid. Each source receives a shrunken cell-and-lead additive correction fit only from IMERG Late errors realized by initialization. A convex inverse-error blend is retained only where its matched historical MSE is no worse than the best corrected source; otherwise the historical leader is used. This retrospective safeguard cannot guarantee future performance. Source NetCDF files are cached on the workstation and decompressed map payloads are cached in memory by the browser.

See [`assets/forecast_archive.json`](assets/forecast_archive.json), [`assets/weather_forecast.json`](assets/weather_forecast.json), and [`assets/validation_manifest.json`](assets/validation_manifest.json), and [`assets/imerg_manifest.json`](assets/imerg_manifest.json) for provenance.
