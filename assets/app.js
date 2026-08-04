(() => {
  "use strict";
  const dataNode = document.querySelector("#site-data");
  if (!dataNode) return;
  const site = JSON.parse(dataNode.textContent);
  const archive = site.archive;
  const validation = site.validation;
  const weather = site.weather;
  const imerg = site.imerg || { products: {}, forecast_runs: {}, cities: {} };
  const spatialCombination = site.combination?.spatial || { runs: {} };
  const runs = archive.runs;
  const params = new URLSearchParams(location.search);
  const q = (selector) => document.querySelector(selector);
  const qa = (selector) => [...document.querySelectorAll(selector)];
  const allowedTabs = new Set(["weather", "maps", "validation", "method"]);
  const allowedVariables = new Set(["temperature", "temperature_high", "temperature_low", "precipitation"]);
  const mapVariableLabels = { temperature: "Temperature", temperature_high: "Daily high", temperature_low: "Daily low", precipitation: "Interval rainfall" };
  const allowedDays = new Set(["1", "3", "5"]);
  const allowedWeatherDays = new Set(["1", "2", "3", "4", "5"]);
  const cityGridColors = { weathernext2: "#2563a6", gencast: "#7c4db3", gfs: "#d4573b",
    gefs: "#be7910", aifs: "#087f73", ifs_ens: "#34495e", combined: "#c51d3b",
    imerg_combined: "#c51d3b" };
  const runIds = new Set(runs.map((run) => run.id));
  const cityNames = Object.keys(validation.cities);
  const sourceModelTotal = site.models.filter((model) => !["combined", "simple_average"].includes(model.id)).length;
  let tab = allowedTabs.has(params.get("tab")) ? params.get("tab") : "weather";
  let init = runIds.has(params.get("init")) ? params.get("init") : runs[0].id;
  let city = cityNames.includes(params.get("city")) ? params.get("city") : cityNames[0];
  let weatherVariable = params.get("weather") === "precipitation" ? "precipitation" : "temperature";
  let weatherDay = allowedWeatherDays.has(params.get("weather_day")) ? params.get("weather_day") : "1";
  let cityGridModel = params.get("grid_model") || null;
  let mapVariable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let mapDay = allowedDays.has(params.get("day")) ? params.get("day") : "1";
  let mapModel = params.get("model") || (spatialCombination.runs?.[runs[0].id]?.map_payload ? "combined" : runs[0].available_models?.[0]) || runs[0].models[0].id;
  let validationVariable = params.get("validation") === "precipitation" ? "precipitation" : "temperature";
  let matchVariable = params.get("match_variable") === "temperature" ? "temperature" : "precipitation";
  let matchInit = runIds.has(params.get("match_init")) ? params.get("match_init") : runs[0].id;
  let withinDayModel = params.get("within_model") || "combined";
  let temporalVariable = params.get("temporal_variable") === "temperature" ? "temperature" : "precipitation";
  let temporalInit = params.get("temporal_init") || Object.keys(imerg.forecast_runs || {})[0] || "";
  let temporalModel = params.get("temporal_model") || "";
  let temporalTimeIndex = Number(params.get("temporal_time") || 0);
  let imergDuration = params.get("imerg_duration") === "6h" ? "6h" : "30min";
  let imergTimeIndex = Number(params.get("imerg_time") || -1);
  let imergValidationInit = params.get("imerg_validation_init") || "";
  let imergValidationInitTouched = Boolean(params.get("imerg_validation_init"));
  let imergValidationMetric = params.get("imerg_metric") === "error" ? "error" : "rainfall";
  let imergValidationForecast = params.get("imerg_forecast") === "raw" ? "raw" : "corrected";
  const validationVisibleModels = new Set();
  const imergVisibleModels = new Set();
  let payload = null;
  let coastlines = [];
  let coastlinePromise = null;
  let mapRequest = 0;
  let view = { scale: 1, x: 0, y: 0 };
  let drag = null;
  let temporalRequest = 0;
  let imergRequest = 0;
  const compressedPayloads = new Map();

  function setUrl() {
    const next = new URL(location.href);
    const values = { tab, init, city, weather: weatherVariable, weather_day: weatherDay, grid_model: cityGridModel,
      variable: mapVariable, day: mapDay,
      model: mapModel, validation: validationVariable, match_init: matchInit, match_variable: matchVariable,
      within_model: withinDayModel, temporal_variable: temporalVariable, temporal_init: temporalInit,
      temporal_model: temporalModel, temporal_time: temporalTimeIndex, imerg_duration: imergDuration,
      imerg_time: imergTimeIndex, imerg_validation_init: imergValidationInit,
      imerg_metric: imergValidationMetric, imerg_forecast: imergValidationForecast };
    Object.entries(values).forEach(([key, value]) => { if (value) next.searchParams.set(key, value); });
    history.replaceState(null, "", next);
  }

  function selectButton(selector, value, key) {
    qa(selector).forEach((button) => button.setAttribute("aria-pressed", String(button.dataset[key] === value)));
  }

  function activateTab(next, update = true) {
    tab = allowedTabs.has(next) ? next : "weather";
    qa("[data-tab]").forEach((button) => {
      const active = button.dataset.tab === tab;
      button.setAttribute("aria-selected", String(active));
      button.classList.toggle("is-active", active);
    });
    qa("[data-panel]").forEach((panel) => { panel.hidden = panel.dataset.panel !== tab; });
    if (tab === "maps") requestAnimationFrame(() => { drawMap(activeRun()); renderTemporalMaps(); });
    if (tab === "validation") requestAnimationFrame(() => { renderImergMaps(); renderImergCityValidation(); });
    if (update) setUrl();
  }

  function activeRun() { return runs.find((run) => run.id === init) || runs[0]; }
  function sourceRunModels(run = activeRun()) { return run.available_models || run.models.map((model) => model.id); }
  function runModels(run = activeRun()) {
    const models = sourceRunModels(run);
    const mixtures = [];
    if (spatialCombination.runs?.[run.id]?.map_payload) mixtures.push("combined");
    if (spatialCombination.runs?.[run.id]?.simple_average_map_payload) mixtures.push("simple_average");
    return [...mixtures, ...models];
  }
  function modelLabel(model) {
    const item = site.models.find((candidate) => candidate.id === model);
    return item?.label || (model === imerg.grid_ensemble?.model_id ? imerg.grid_ensemble.label : model);
  }
  function formatInit(value) {
    return new Date(value).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short",
      year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) + " UTC";
  }

  function formatZoned(value, timeZone, suffix) {
    const formatted = new Intl.DateTimeFormat("en-GB", { timeZone, day: "2-digit", month: "short",
      year: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(value));
    return `${formatted} ${suffix}`;
  }

  function exactTime(value) {
    return `${formatZoned(value, "Asia/Kolkata", "IST")} · ${formatZoned(value, "UTC", "UTC")}`;
  }

  function validTime(run, day = Number(mapDay)) {
    const published = run.lead_days?.find((item) => Number(item.day) === Number(day));
    return published?.valid_time_utc || new Date(new Date(run.initialization_utc).getTime() + Number(day) * 86400000).toISOString();
  }

  function compactValidTime(value) {
    const ist = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short",
      year: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(value));
    const utc = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", hour: "2-digit", minute: "2-digit",
      hourCycle: "h23" }).format(new Date(value));
    return `${ist} IST · ${utc} UTC`;
  }

  function renderRun() {
    const run = activeRun();
    q("#init-select").value = init;
    q("#run-status").textContent = `${formatInit(run.initialization_utc)} · ${sourceRunModels(run).length} of ${sourceModelTotal} source models`;
    q("#availability-note").textContent = run.status === "partial"
      ? `Partial run. Waiting for: ${(run.missing_models || []).map(modelLabel).join(", ")}.`
      : "All configured models are available for this initialization.";
    const available = runModels(run);
    if (!available.includes(mapModel)) mapModel = available[0];
    qa("[data-map-model]").forEach((button) => {
      const enabled = available.includes(button.dataset.mapModel);
      button.disabled = !enabled;
      button.title = enabled ? "" : "Not available for this initialization";
      button.setAttribute("aria-pressed", String(button.dataset.mapModel === mapModel));
    });
    qa("[data-map-day]").forEach((button) => {
      const valid = validTime(run, button.dataset.mapDay);
      button.innerHTML = `<span>${formatZoned(valid, "Asia/Kolkata", "IST")}</span><small>${formatZoned(valid, "UTC", "UTC")}</small>`;
      button.title = `Forecast valid ${exactTime(valid)}`;
      button.setAttribute("aria-pressed", String(button.dataset.mapDay === mapDay));
    });
    renderWeather();
    loadMap();
    renderAnimation();
    setUrl();
  }

  function path(points) {
    if (!points.length) return "";
    return points.map((point, index) => `${index ? "L" : "M"}${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" ");
  }

  function weatherChart(days) {
    const width = 920, height = 270, pad = { l: 45, r: 20, t: 22, b: 58 };
    const value = (day) => weatherVariable === "temperature" ? day.mean_c : day.precip_mm;
    const values = days.map(value);
    if (!values.length) return '<p class="empty-state">No five-day city forecast is available for this run.</p>';
    const low = weatherVariable === "temperature" ? Math.floor(Math.min(...days.map((day) => day.low_c)) - 2) : 0;
    const high = weatherVariable === "temperature" ? Math.ceil(Math.max(...days.map((day) => day.high_c)) + 2) : Math.max(5, Math.ceil(Math.max(...values) * 1.25));
    const x = (index) => pad.l + index * (width - pad.l - pad.r) / Math.max(days.length - 1, 1);
    const y = (number) => pad.t + (high - number) * (height - pad.t - pad.b) / Math.max(high - low, 1);
    const line = path(days.map((day, index) => [x(index), y(value(day))]));
    const area = `${line} L${x(days.length - 1)},${height - pad.b} L${x(0)},${height - pad.b} Z`;
    const gridValues = [low, (low + high) / 2, high];
    const grid = gridValues.map((number) => `<g><line x1="${pad.l}" x2="${width - pad.r}" y1="${y(number)}" y2="${y(number)}"/><text x="${pad.l - 8}" y="${y(number) + 4}" text-anchor="end">${number.toFixed(weatherVariable === "temperature" ? 0 : 1)}</text></g>`).join("");
    const dots = days.map((day, index) => {
      const valid = new Date(day.valid_end_utc);
      const date = valid.toLocaleDateString("en-GB", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", year: "numeric" });
      const ist = valid.toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
      const utc = valid.toLocaleTimeString("en-GB", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
      return `<g><circle cx="${x(index)}" cy="${y(value(day))}" r="4"/><text x="${x(index)}" y="${y(value(day)) - 12}" text-anchor="middle">${value(day).toFixed(1)}${weatherVariable === "temperature" ? "°" : " mm"}</text><text class="date" x="${x(index)}" y="${height - 27}" text-anchor="middle">${date}</text><text class="date time" x="${x(index)}" y="${height - 12}" text-anchor="middle">${ist} IST · ${utc} UTC</text></g>`;
    }).join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Five-day ${weatherVariable} forecast"><g class="chart-grid">${grid}</g><path class="weather-area" d="${area}"/><path class="weather-line" d="${line}"/><g class="weather-points">${dots}</g></svg>`;
  }

  function withinDayChart(rows, model) {
    if (!rows.length) return '<p class="empty-state">Native-time detail is available for the latest three initializations.</p>';
    const width = 980, height = 330, pad = { l: 58, r: 52, t: 24, middle: 205, b: 48 };
    const times = rows.map((row) => new Date(row.valid_time_utc).getTime());
    const x = (time) => pad.l + (time - times[0]) / Math.max(times[times.length - 1] - times[0], 1) * (width - pad.l - pad.r);
    const temperatures = rows.filter((row) => Number.isFinite(row.temperature_c));
    const rainRows = rows.filter((row) => Number.isFinite(row.precip_mm));
    const tempValues = temperatures.map((row) => row.temperature_c);
    const tempLow = tempValues.length ? Math.floor(Math.min(...tempValues) - 1) : 0;
    const tempHigh = tempValues.length ? Math.ceil(Math.max(...tempValues) + 1) : 1;
    const tempY = (value) => pad.t + (tempHigh - value) / Math.max(tempHigh - tempLow, 1) * (pad.middle - pad.t - 18);
    const rainHigh = Math.max(1, ...rainRows.map((row) => row.precip_mm));
    const rainTop = pad.middle + 27, rainBottom = height - pad.b;
    const rainY = (value) => rainBottom - value / rainHigh * (rainBottom - rainTop);
    const tempPath = temperatures.map((row, index) => `${index ? "L" : "M"}${x(new Date(row.valid_time_utc).getTime()).toFixed(1)},${tempY(row.temperature_c).toFixed(1)}`).join(" ");
    const barWidth = Math.max(4, Math.min(30, (width - pad.l - pad.r) / Math.max(rows.length, 1) * .62));
    const bars = rainRows.map((row) => {
      const center = x(new Date(row.valid_time_utc).getTime());
      return `<rect class="within-rain-bar" x="${center - barWidth / 2}" y="${rainY(row.precip_mm)}" width="${barWidth}" height="${Math.max(0, rainBottom - rainY(row.precip_mm))}"><title>${row.precip_mm.toFixed(2)} mm · ${exactTime(row.interval_start_utc)} → ${exactTime(row.valid_time_utc)}</title></rect>`;
    }).join("");
    const dots = temperatures.map((row) => `<circle class="within-temp-dot" cx="${x(new Date(row.valid_time_utc).getTime())}" cy="${tempY(row.temperature_c)}" r="3.5"><title>${row.temperature_c.toFixed(1)} °C · ${exactTime(row.valid_time_utc)}</title></circle>`).join("");
    const labelEvery = Math.max(1, Math.ceil(rows.length / 8));
    const labels = rows.map((row, index) => {
      if (index % labelEvery && index !== rows.length - 1) return "";
      const value = new Date(row.valid_time_utc);
      const ist = value.toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
      const utc = value.toLocaleTimeString("en-GB", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
      return `<text x="${x(value.getTime())}" y="${height - 25}" text-anchor="middle">${ist} IST</text><text x="${x(value.getTime())}" y="${height - 10}" text-anchor="middle">${utc} UTC</text>`;
    }).join("");
    const tempGrid = [tempLow, (tempLow + tempHigh) / 2, tempHigh].map((value) => `<g><line x1="${pad.l}" x2="${width - pad.r}" y1="${tempY(value)}" y2="${tempY(value)}"/><text x="${pad.l - 8}" y="${tempY(value) + 4}" text-anchor="end">${value.toFixed(0)}°</text></g>`).join("");
    const rainGrid = [0, rainHigh].map((value) => `<g><line x1="${pad.l}" x2="${width - pad.r}" y1="${rainY(value)}" y2="${rainY(value)}"/><text x="${pad.l - 8}" y="${rainY(value) + 4}" text-anchor="end">${value.toFixed(1)}</text></g>`).join("");
    return `<svg viewBox="0 0 ${width} ${height}" aria-label="${modelLabel(model)} within-day temperature and interval rainfall"><g class="within-axis">${tempGrid}${rainGrid}${labels}<text x="12" y="18">°C</text><text x="12" y="${rainTop}">mm</text></g><path class="within-temp-line" d="${tempPath}"/>${dots}${bars}</svg>`;
  }

  function renderWithinDay(item, day) {
    const timelines = item?.timelines || {};
    const models = Object.keys(timelines).filter((model) => (timelines[model] || []).some((row) => String(row.day) === weatherDay));
    if (!models.includes(withinDayModel)) withinDayModel = models.includes("combined") ? "combined" : models[0];
    q("#within-day-models").innerHTML = models.map((model) => `<button type="button" data-within-day-model="${model}" aria-pressed="${model === withinDayModel}">${model === "combined" ? "Combined · 6 h" : modelLabel(model)}</button>`).join("");
    qa("[data-within-day-model]").forEach((button) => button.addEventListener("click", () => {
      withinDayModel = button.dataset.withinDayModel;
      renderWithinDay(item, day);
      setUrl();
    }));
    const rows = (timelines[withinDayModel] || []).filter((row) => String(row.day) === weatherDay);
    q("#within-day-chart").innerHTML = withinDayChart(rows, withinDayModel || "combined");
    const cadences = [...new Set(rows.map((row) => row.interval_hours))].sort((a, b) => a - b);
    q("#within-day-note").textContent = rows.length
      ? `${withinDayModel === "combined" ? "Combined forecast" : modelLabel(withinDayModel)} · ${cadences.map((value) => `${Number(value).toFixed(Number(value) % 1 ? 1 : 0)} h`).join(" / ")} exact interval${cadences.length === 1 ? "" : "s"} · ${exactTime(rows[0].interval_start_utc)} → ${exactTime(rows[rows.length - 1].valid_time_utc)}. Bars are interval accumulation, not probability.`
      : "Native-time detail is retained for the latest three initializations.";
  }

  function cityMapWorld(latitude, longitude, zoom) {
    const size = 256 * (2 ** zoom);
    const limitedLatitude = Math.max(-85.0511, Math.min(85.0511, latitude));
    const sine = Math.sin(limitedLatitude * Math.PI / 180);
    return {
      x: (longitude + 180) / 360 * size,
      y: (.5 - Math.log((1 + sine) / (1 - sine)) / (4 * Math.PI)) * size,
    };
  }

  function renderCityGridMap(item, day) {
    const map = q("#city-grid-map");
    const list = q("#grid-input-list");
    if (!item || !day || !Object.keys(day.experts || {}).length) {
      map.innerHTML = '<p class="empty-state">Contributing grid points are unavailable for this selection.</p>';
      list.innerHTML = "";
      q("#city-grid-result").textContent = "No grid inputs available.";
      q("#city-grid-time").textContent = "";
      q("#city-grid-samples").textContent = "";
      return;
    }
    const width = 760, height = 410, zoom = 8, tileSize = 256, tileCount = 2 ** zoom;
    const center = cityMapWorld(item.latitude, item.longitude, zoom);
    const left = center.x - width / 2, top = center.y - height / 2;
    const tileImages = [];
    for (let tileY = Math.floor(top / tileSize); tileY <= Math.floor((top + height) / tileSize); tileY += 1) {
      if (tileY < 0 || tileY >= tileCount) continue;
      for (let tileX = Math.floor(left / tileSize); tileX <= Math.floor((left + width) / tileSize); tileX += 1) {
        const wrappedX = ((tileX % tileCount) + tileCount) % tileCount;
        tileImages.push(`<image class="city-map-tile" href="https://tile.openstreetmap.org/${zoom}/${wrappedX}/${tileY}.png" x="${tileX * tileSize - left}" y="${tileY * tileSize - top}" width="256" height="256"/>`);
      }
    }
    const experts = Object.entries(day.experts);
    if (!day.experts[cityGridModel]) cityGridModel = experts[0][0];
    q("#city-grid-models").innerHTML = experts.map(([model]) => `<button type="button" data-city-grid-model="${model}" aria-pressed="${model === cityGridModel}">${modelLabel(model)}</button>`).join("");
    qa("[data-city-grid-model]").forEach((button) => button.addEventListener("click", () => {
      cityGridModel = button.dataset.cityGridModel;
      renderCityGridMap(item, day);
      setUrl();
    }));
    const valueFor = (expert) => weatherVariable === "temperature" ? expert.mean_c : expert.precip_mm;
    const unit = weatherVariable === "temperature" ? "°C" : "mm";
    const selectedExpert = day.experts[cityGridModel];
    const localGrid = selectedExpert.local_grid;
    const gridKey = weatherVariable === "temperature" ? "mean_c" : "precip_mm";
    const cellColor = (value) => {
      if (weatherVariable === "precipitation") {
        const fraction = Math.max(0, Math.min(1, value / 60));
        return [225 - 185 * fraction, 241 - 80 * fraction, 248 - 25 * fraction];
      }
      const stops = [[255, 255, 204], [254, 217, 118], [253, 141, 60], [240, 59, 32], [189, 0, 38]];
      const scaled = Math.max(0, Math.min(1, value / 45)) * (stops.length - 1);
      const stop = Math.min(stops.length - 2, Math.floor(scaled));
      const fraction = scaled - stop;
      return stops[stop].map((channel, offset) => Math.round(channel + (stops[stop + 1][offset] - channel) * fraction));
    };
    const bounds = (values, index, fallback) => {
      const lower = index > 0 ? (values[index - 1] + values[index]) / 2 : values[index] - (values[1] - values[0] || fallback) / 2;
      const upper = index < values.length - 1 ? (values[index] + values[index + 1]) / 2 : values[index] + (values[index] - values[index - 1] || fallback) / 2;
      return [lower, upper];
    };
    const cells = localGrid.latitudes.flatMap((latitude, latIndex) => localGrid.longitudes.map((longitude, lonIndex) => {
      const value = localGrid[gridKey][latIndex][lonIndex];
      if (value === null) return "";
      const [south, north] = bounds(localGrid.latitudes, latIndex, localGrid.latitude_spacing_degrees || .25);
      const [west, east] = bounds(localGrid.longitudes, lonIndex, localGrid.longitude_spacing_degrees || .25);
      const northwest = cityMapWorld(north, west, zoom);
      const southeast = cityMapWorld(south, east, zoom);
      const x = northwest.x - left, y = northwest.y - top;
      const cellWidth = southeast.x - northwest.x, cellHeight = southeast.y - northwest.y;
      const color = cellColor(value);
      return `<g><rect class="forecast-grid-cell" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${cellWidth.toFixed(1)}" height="${cellHeight.toFixed(1)}" fill="rgb(${color.join(",")})"><title>${modelLabel(cityGridModel)} grid cell centered ${latitude.toFixed(4)}° N, ${longitude.toFixed(4)}° E: ${value.toFixed(1)} ${unit}, valid ${exactTime(day.valid_end_utc)}</title></rect><text class="forecast-grid-value" x="${(x + cellWidth / 2).toFixed(1)}" y="${(y + cellHeight / 2 + 3).toFixed(1)}" text-anchor="middle">${value.toFixed(1)}</text></g>`;
    })).join("");
    const markers = [[cityGridModel, selectedExpert]].map(([model, expert]) => {
      const actual = cityMapWorld(expert.grid_latitude, expert.grid_longitude, zoom);
      const x = actual.x - left, y = actual.y - top;
      const calloutX = Math.max(64, Math.min(width - 64, x + 92));
      const calloutY = Math.max(28, Math.min(height - 28, y - 58));
      const color = cityGridColors[model] || "#41687f";
      const shortLabel = modelLabel(model).replace("WeatherNext 2", "WN2").replace("IFS-ENS", "IFS");
      const title = `${modelLabel(model)}: ${valueFor(expert).toFixed(1)} ${unit} at ${expert.grid_latitude.toFixed(4)}° N, ${expert.grid_longitude.toFixed(4)}° E`;
      return `<g><line class="city-grid-leader" x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${calloutX.toFixed(1)}" y2="${calloutY.toFixed(1)}"/><circle class="city-grid-point" data-grid-model="${model}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${color}"><title>${title}</title></circle><g class="city-grid-callout" transform="translate(${calloutX.toFixed(1)} ${calloutY.toFixed(1)})"><rect x="-48" y="-20" width="96" height="40" rx="5"/><circle cx="-37" cy="-7" r="4" fill="${color}"/><text class="model" x="-28" y="-3">${shortLabel}</text><text class="value" x="0" y="13" text-anchor="middle">${valueFor(expert).toFixed(1)} ${unit}</text></g></g>`;
    }).join("");
    map.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid slice" role="img" aria-label="${modelLabel(cityGridModel)} forecast grid and values over ${city}"><rect width="${width}" height="${height}" class="city-map-fallback"/>${tileImages.join("")}<g class="forecast-grid-mesh">${cells}</g><g class="city-grid-overlay">${markers}<g class="city-location" transform="translate(${width / 2} ${height / 2})"><circle r="9"/><circle r="3"/><text x="13" y="4">${city}</text></g></g></svg><span class="osm-attribution">© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a></span>`;
    q("#city-grid-model-note").textContent = `${modelLabel(cityGridModel)} loaded grid · ${localGrid.latitudes.length} × ${localGrid.longitudes.length} cells shown · ${localGrid.latitude_spacing_degrees?.toFixed(3) || "n/a"}° latitude × ${localGrid.longitude_spacing_degrees?.toFixed(3) || "n/a"}° longitude · valid ${exactTime(day.valid_end_utc)}`;

    const weights = weatherVariable === "temperature" ? item.temperature_weights : item.precipitation_weights;
    list.innerHTML = experts.map(([model, expert]) => {
      const detail = weatherVariable === "temperature"
        ? `Mean ${expert.mean_c.toFixed(1)} °C · high ${expert.high_c.toFixed(1)} °C at ${exactTime(expert.high_time_utc)} · low ${expert.low_c.toFixed(1)} °C at ${exactTime(expert.low_time_utc)}`
        : `${expert.precip_mm.toFixed(1)} mm accumulated over this exact 24-hour window`;
      return `<article class="grid-input"><span class="grid-swatch" style="background:${cityGridColors[model] || "#41687f"}"></span><div><strong>${modelLabel(model)}</strong><small>${expert.grid_latitude.toFixed(4)}° N · ${expert.grid_longitude.toFixed(4)}° E</small><p>${detail}</p></div><b>${((weights[model] || 0) * 100).toFixed(1)}%</b></article>`;
    }).join("");

    const simpleAverage = experts.reduce((sum, [, expert]) => sum + valueFor(expert), 0) / experts.length;
    const combined = weatherVariable === "temperature"
      ? `Recent-error blend: mean ${day.mean_c.toFixed(1)} °C · high ${day.high_c.toFixed(1)} °C · low ${day.low_c.toFixed(1)} °C`
      : `Recent-error blend: ${day.precip_mm.toFixed(1)} mm in 24 h`;
    q("#city-grid-result").textContent = `${combined} · simple average of shown inputs: ${simpleAverage.toFixed(1)} ${unit}`;
    q("#city-grid-time").textContent = `Valid ${exactTime(day.valid_start_utc)} → ${exactTime(day.valid_end_utc)}`;
    const samples = [...new Set(experts.flatMap(([, expert]) => expert.sample_times_utc || []))].sort();
    q("#city-grid-samples").textContent = `Exact native sample times used (${samples.length} unique): ${samples.map(exactTime).join(" · ")}`;
  }

  function renderWeather() {
    q("#city-select").value = city;
    selectButton("[data-weather-variable]", weatherVariable, "weatherVariable");
    const run = weather.runs[init];
    const item = run?.cities?.[city];
    const days = item?.days || [];
    q("#weather-location").textContent = city;
    q("#weather-meta").textContent = item
      ? `Initialized ${exactTime(run.initialization_utc)} · ${item.available_models.length} contributing model${item.available_models.length === 1 ? "" : "s"}`
      : "Forecast unavailable for this selection";
    const first = days[0];
    q("#weather-now").innerHTML = first
      ? `<span aria-hidden="true">${first.symbol}</span><strong>${Math.round(first.mean_c)}°C</strong><small>${first.condition}<br>${first.precip_mm.toFixed(1)} mm in 24 h</small>`
      : "";
    q("#weather-chart").innerHTML = weatherChart(days);
    if (!days.some((day) => String(day.day) === weatherDay) && days.length) weatherDay = String(days[0].day);
    q("#daily-cards").innerHTML = days.map((day) => `<button type="button" class="day-card" data-weather-day="${day.day}" aria-pressed="${String(day.day) === weatherDay}"><strong>${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { weekday: "short" })}</strong><time datetime="${day.valid_end_utc}">${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short" })}<span>ends ${new Date(day.valid_end_utc).toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })} IST</span></time><span class="weather-icon" aria-label="${day.condition}">${day.symbol}</span><p><b>${Math.round(day.high_c)}°</b> <span>${Math.round(day.low_c)}°</span></p><small>${day.precip_mm.toFixed(1)} mm</small></button>`).join("");
    qa("[data-weather-day]").forEach((button) => button.addEventListener("click", () => { weatherDay = button.dataset.weatherDay; renderWeather(); setUrl(); }));
    q("#blend-note").textContent = item
      ? `Temperature: ${item.temperature_method} weights · rainfall: ${item.precipitation_method} weights. Rainfall is a 24-hour accumulation, not a probability.`
      : "";
    renderWithinDay(item, days.find((day) => String(day.day) === weatherDay));
    renderCityGridMap(item, days.find((day) => String(day.day) === weatherDay));
  }

  function mapColor(number) {
    if (mapVariable === "precipitation") {
      const t = Math.max(0, Math.min(1, number / 120));
      return [225 - 185 * t, 241 - 80 * t, 248 - 25 * t];
    }
    const stops = [[255, 255, 204], [254, 217, 118], [253, 141, 60], [240, 59, 32], [189, 0, 38]];
    const scaled = Math.max(0, Math.min(1, number / 45)) * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(scaled));
    const fraction = scaled - index;
    return stops[index].map((channel, offset) => channel + (stops[index + 1][offset] - channel) * fraction);
  }

  function precipitationWindow(day = Number(mapDay), run = activeRun()) {
    const previousDay = day === 1 ? 0 : day === 3 ? 1 : 3;
    const start = previousDay === 0 ? run.initialization_utc : validTime(run, previousDay);
    const end = validTime(run, day);
    return `${compactValidTime(start)} → ${compactValidTime(end)} (${(day - previousDay) * 24} h)`;
  }

  function renderMapLegend() {
    const legend = q("#map-legend");
    const precipitation = mapVariable === "precipitation";
    legend.classList.toggle("is-precipitation", precipitation);
    q("#map-legend-title").textContent = precipitation ? "Interval rainfall (mm)" : "Temperature (°C) · fixed scale";
    q("#map-legend-ticks").innerHTML = (precipitation ? [0, 40, 80, 120] : [0, 15, 30, 45]).map((value) => `<span>${value}</span>`).join("");
    q("#map-legend-note").textContent = precipitation ? precipitationWindow() : "Same 0–45 °C scale for every model, valid time, and temperature layer.";
  }

  function decodeMapValue(encoded) {
    if (encoded === 65535) return null;
    return mapVariable === "precipitation" ? encoded / 10 : (encoded - 5000) / 100;
  }

  function loadCoastlines() {
    if (!coastlinePromise) {
      coastlinePromise = fetch("assets/coastlines.json")
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then((data) => { coastlines = Array.isArray(data.lines) ? data.lines : []; })
        .catch((error) => { console.warn("Coastline overlay unavailable.", error); });
    }
    return coastlinePromise;
  }

  function renderAnimation() {
    const label = mapVariableLabels[mapVariable];
    const model = modelLabel(mapModel);
    const source = `assets/map_animations/${init}/${mapModel}/${mapVariable}.gif`;
    const image = q("#map-animation");
    if (image.getAttribute("src") !== source) image.src = source;
    const run = activeRun();
    const endpoints = run.grid_metadata.lead_days.map((day) => compactValidTime(validTime(run, day)));
    image.alt = `Animated ${label.toLowerCase()} forecast for ${model} at ${endpoints.join(", ")}`;
    q("#animation-title").textContent = `${label} · ${model}`;
    q("#animation-description").textContent = mapVariable === "precipitation"
      ? `Each frame shows interval rainfall for these exact windows: ${run.grid_metadata.lead_days.map((day) => precipitationWindow(day, run)).join(" · ")}.`
      : `Animated forecast valid at ${endpoints.join(" · ")} on a fixed 0–45 °C scale.`;
  }

  async function loadMap() {
    const run = activeRun();
    if (!q("#forecast-canvas") || !run.grid_metadata?.shape || !runModels(run).includes(mapModel)) return;
    const request = ++mapRequest;
    q("#map-readout").textContent = "Loading map…";
    try {
      const [response] = await Promise.all([
        fetch(`assets/map_data/${init}/${mapModel}.bin`),
        loadCoastlines(),
      ]);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const next = new Uint16Array(await response.arrayBuffer());
      if (request !== mapRequest) return;
      payload = next;
      drawMap(run);
    } catch (error) {
      if (request === mapRequest) q("#map-readout").textContent = "Map data unavailable.";
      console.error(error);
    }
  }

  function drawCoastlines(ctx, meta, width, height) {
    const bounds = meta.bounding_box;
    const x = (longitude) => width * (longitude - bounds.lon_min) / (bounds.lon_max - bounds.lon_min);
    const y = (latitude) => height * (bounds.lat_max - latitude) / (bounds.lat_max - bounds.lat_min);
    ctx.strokeStyle = "rgba(19, 44, 57, .9)";
    ctx.lineWidth = 1.35 / view.scale;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    coastlines.forEach((line) => {
      if (line.length < 2) return;
      ctx.beginPath();
      line.forEach(([longitude, latitude], index) => {
        if (index === 0) ctx.moveTo(x(longitude), y(latitude));
        else ctx.lineTo(x(longitude), y(latitude));
      });
      ctx.stroke();
    });
  }

  function mapCoordinates(event, run = activeRun()) {
    const canvas = q("#forecast-canvas");
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const gx = ((event.clientX - rect.left) * ratio - view.x) / view.scale / canvas.width;
    const gy = ((event.clientY - rect.top) * ratio - view.y) / view.scale / canvas.height;
    if (gx < 0 || gx > 1 || gy < 0 || gy > 1) return null;
    const bounds = run.grid_metadata.bounding_box;
    return {
      gx,
      gy,
      cssX: event.clientX - rect.left,
      cssY: event.clientY - rect.top,
      rect,
      longitude: bounds.lon_min + gx * (bounds.lon_max - bounds.lon_min),
      latitude: bounds.lat_max - gy * (bounds.lat_max - bounds.lat_min),
    };
  }

  function mapValueAt(run, point) {
    const meta = run.grid_metadata;
    const [nLead, nLat, nLon] = meta.shape;
    const variableIndex = meta.variables.indexOf(mapVariable);
    const dayIndex = meta.lead_days.indexOf(Number(mapDay));
    if (variableIndex < 0 || dayIndex < 0) return null;
    const xIndex = Math.max(0, Math.min(nLon - 1, Math.round(point.gx * (nLon - 1))));
    const yIndex = Math.max(0, Math.min(nLat - 1, Math.round((1 - point.gy) * (nLat - 1))));
    const count = nLead * nLat * nLon;
    const start = variableIndex * count + dayIndex * nLat * nLon;
    return decodeMapValue(payload[start + yIndex * nLon + xIndex]);
  }

  function hideMapTooltip() {
    q("#map-tooltip").hidden = true;
  }

  function showMapTooltip(event) {
    if (!payload) return;
    const run = activeRun();
    const point = mapCoordinates(event, run);
    const value = point && mapValueAt(run, point);
    if (!point || value === null) { hideMapTooltip(); return; }
    const tooltip = q("#map-tooltip");
    const units = mapVariable === "precipitation" ? "mm" : "°C";
    const valid = mapVariable === "precipitation" ? `${precipitationWindow()} accumulation` : `valid ${compactValidTime(validTime(run))}`;
    tooltip.innerHTML = `<strong>${value.toFixed(1)} ${units}</strong><span>${point.latitude.toFixed(2)}° N · ${point.longitude.toFixed(2)}° E</span><small>${modelLabel(mapModel)} · ${valid}</small>`;
    tooltip.style.left = `${Math.max(8, Math.min(point.rect.width - 185, point.cssX + 12))}px`;
    tooltip.style.top = `${point.cssY > 90 ? point.cssY - 12 : point.cssY + 12}px`;
    tooltip.dataset.side = point.cssY > 90 ? "above" : "below";
    tooltip.hidden = false;
  }

  function drawMap(run = activeRun()) {
    const canvas = q("#forecast-canvas");
    if (!payload || !canvas || !run.grid_metadata?.shape) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(600, Math.round(rect.width * ratio));
    const height = Math.max(520 * ratio, Math.round(rect.width * .86 * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const meta = run.grid_metadata;
    const [nLead, nLat, nLon] = meta.shape;
    const variableIndex = meta.variables.indexOf(mapVariable);
    const dayIndex = meta.lead_days.indexOf(Number(mapDay));
    const count = nLead * nLat * nLon;
    const start = variableIndex * count + dayIndex * nLat * nLon;
    const image = ctx.createImageData(nLon, nLat);
    for (let yIndex = 0; yIndex < nLat; yIndex += 1) for (let xIndex = 0; xIndex < nLon; xIndex += 1) {
      const encoded = payload[start + yIndex * nLon + xIndex];
      const offset = ((nLat - 1 - yIndex) * nLon + xIndex) * 4;
      if (encoded === 65535) { image.data[offset + 3] = 0; continue; }
      const number = decodeMapValue(encoded);
      const rgb = mapColor(number);
      image.data[offset] = rgb[0]; image.data[offset + 1] = rgb[1]; image.data[offset + 2] = rgb[2]; image.data[offset + 3] = 255;
    }
    const raster = document.createElement("canvas");
    raster.width = nLon; raster.height = nLat; raster.getContext("2d").putImageData(image, 0, 0);
    ctx.fillStyle = "#e8f1f5"; ctx.fillRect(0, 0, width, height);
    ctx.save(); ctx.translate(view.x, view.y); ctx.scale(view.scale, view.scale); ctx.imageSmoothingEnabled = true;
    ctx.drawImage(raster, 0, 0, width, height);
    ctx.strokeStyle = "rgba(255,255,255,.38)"; ctx.lineWidth = 1 / view.scale;
    for (let fraction = .2; fraction < 1; fraction += .2) { ctx.beginPath(); ctx.moveTo(width * fraction, 0); ctx.lineTo(width * fraction, height); ctx.moveTo(0, height * fraction); ctx.lineTo(width, height * fraction); ctx.stroke(); }
    drawCoastlines(ctx, meta, width, height);
    Object.entries(validation.cities).forEach(([name, item]) => {
      const x = width * (item.longitude - meta.bounding_box.lon_min) / (meta.bounding_box.lon_max - meta.bounding_box.lon_min);
      const y = height * (meta.bounding_box.lat_max - item.latitude) / (meta.bounding_box.lat_max - meta.bounding_box.lat_min);
      ctx.beginPath(); ctx.arc(x, y, 6 / view.scale, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#c51d3b"; ctx.lineWidth = 2.5 / view.scale; ctx.stroke();
      ctx.fillStyle = "#173f63"; ctx.font = `${12 / view.scale}px system-ui`; ctx.fillText(name, x + 10 / view.scale, y - 8 / view.scale);
    });
    ctx.restore();
    const selectedValidTime = validTime(run);
    q("#map-title").textContent = `${mapVariableLabels[mapVariable]} · ${compactValidTime(selectedValidTime)}`;
    const learnerVariable = mapVariable === "precipitation" ? "precipitation" : "temperature";
    const blend = spatialCombination.runs?.[run.id];
    const candidate = blend?.selected_candidates?.[learnerVariable]?.[mapDay];
    const training = blend?.training_samples?.[learnerVariable]?.[mapDay];
    const blendNote = mapModel === "combined"
      ? ` · ${candidate || "uniform"} from ${training || 0} prior matched samples`
      : mapModel === "simple_average" ? ` · equal weight for each available model at this grid cell` : "";
    q("#map-description").textContent = `${modelLabel(mapModel)} · initialized ${formatInit(run.initialization_utc)}${mapVariable === "precipitation" ? ` · ${precipitationWindow()} accumulation` : ""}${blendNote}`;
    q("#map-readout").textContent = `Valid ${compactValidTime(selectedValidTime)} · drag to pan · scroll to zoom`;
    renderMapLegend();
  }

  function renderMapControls() {
    selectButton("[data-map-variable]", mapVariable, "mapVariable");
    selectButton("[data-map-day]", mapDay, "mapDay");
    selectButton("[data-map-model]", mapModel, "mapModel");
    view = { scale: 1, x: 0, y: 0 };
    loadMap();
    renderAnimation();
    setUrl();
  }

  function attachInteractiveChartTooltip(containerSelector, tooltipSelector) {
    const container = q(containerSelector);
    const tooltip = q(tooltipSelector);
    if (!container || !tooltip) return;
    const show = (event) => {
      const point = event.currentTarget;
      tooltip.textContent = `${point.dataset.label} · ${point.dataset.value} ${point.dataset.units} · ${point.dataset.detail}`;
      const bounds = point.getBoundingClientRect();
      const left = Number.isFinite(event.clientX) && event.clientX ? event.clientX : bounds.left + bounds.width / 2;
      const top = Number.isFinite(event.clientY) && event.clientY ? event.clientY : bounds.top;
      tooltip.style.left = `${Math.max(10, Math.min(window.innerWidth - 260, left + 12))}px`;
      tooltip.style.top = `${Math.max(10, top - 52)}px`;
      tooltip.hidden = false;
    };
    const hide = () => { tooltip.hidden = true; };
    container.querySelectorAll("[data-validation-point]").forEach((point) => {
      point.addEventListener("pointerenter", show);
      point.addEventListener("pointermove", show);
      point.addEventListener("pointerleave", hide);
      point.addEventListener("focus", show);
      point.addEventListener("blur", hide);
    });
  }

  function validationSkillChart(summary, modelIds) {
    const leads = [...new Set(Object.values(summary.models || {}).flatMap((model) => Object.keys(model.mae_by_lead || {}).map(Number)))].sort((a, b) => a - b);
    if (!leads.length || !modelIds.length) return '<p class="empty-state">Select at least one model to draw the validation chart.</p>';
    const width = 1000, height = 430, pad = { l: 70, r: 24, t: 28, b: 62 };
    const unit = validationVariable === "temperature" ? "°C MAE" : "mm MAE";
    const allValues = modelIds.flatMap((id) => Object.values(summary.models[id]?.mae_by_lead || {})).filter(Number.isFinite);
    if (!allValues.length) return '<p class="empty-state">No matched validation scores are available for the selected models.</p>';
    const high = Math.max(validationVariable === "temperature" ? 1 : 5, Math.ceil(Math.max(...allValues) * 1.12 * 10) / 10);
    const x = (lead) => pad.l + leads.indexOf(lead) * (width - pad.l - pad.r) / Math.max(leads.length - 1, 1);
    const y = (value) => pad.t + (high - value) / high * (height - pad.t - pad.b);
    const ticks = [0, .25, .5, .75, 1].map((fraction) => high * fraction);
    const grid = ticks.map((value) => `<g><line x1="${pad.l}" x2="${width - pad.r}" y1="${y(value)}" y2="${y(value)}"/><text x="${pad.l - 10}" y="${y(value) + 4}" text-anchor="end">${value.toFixed(high < 2 ? 2 : 1)}</text></g>`).join("");
    const labels = leads.map((lead) => `<text x="${x(lead)}" y="${height - 30}" text-anchor="middle">+${lead * 24} h</text>`).join("");
    const traces = modelIds.map((id) => {
      const model = summary.models[id];
      const color = cityGridColors[id] || "#64748b";
      const points = leads.map((lead) => ({ lead, value: model?.mae_by_lead?.[String(lead)] })).filter((point) => Number.isFinite(point.value));
      if (!points.length) return "";
      const line = path(points.map((point) => [x(point.lead), y(point.value)]));
      const dots = points.map((point) => `<circle class="validation-chart-point" data-validation-point tabindex="0" cx="${x(point.lead)}" cy="${y(point.value)}" r="4" fill="${color}" data-label="${model.label}" data-value="${point.value.toFixed(2)}" data-units="${unit}" data-detail="forecast horizon +${point.lead * 24} hours" aria-label="${model.label}, ${point.value.toFixed(2)} ${unit}, forecast horizon ${point.lead * 24} hours"></circle>`).join("");
      return `<path class="validation-series" data-validation-series="${id}" d="${line}" fill="none" stroke="${color}" stroke-width="2.5"/>${dots}`;
    }).join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Mean absolute error by forecast horizon for selected models"><g class="interactive-chart-grid">${grid}${labels}<text class="axis-title" x="${(pad.l + width - pad.r) / 2}" y="${height - 7}" text-anchor="middle">Forecast horizon</text><text class="axis-title" transform="translate(17 ${(pad.t + height - pad.b) / 2}) rotate(-90)" text-anchor="middle">${unit}</text></g>${traces}</svg>`;
  }

  function renderValidation() {
    selectButton("[data-validation-city]", city, "validationCity");
    selectButton("[data-validation-variable]", validationVariable, "validationVariable");
    selectButton("[data-match-variable]", matchVariable, "matchVariable");
    q("#match-init-select").value = matchInit;
    const item = validation.cities[city];
    const overview = item.images[validationVariable];
    const matched = item.timeseries[matchInit][matchVariable];
    q("#validation-image").src = overview.path; q("#validation-image").alt = overview.alt;
    q("#match-image").src = matched.path; q("#match-image").alt = matched.alt;
    const summary = item.summary[validationVariable];
    const modelIds = Object.keys(summary.models || {});
    [...validationVisibleModels].forEach((model) => { if (!modelIds.includes(model)) validationVisibleModels.delete(model); });
    if (!validationVisibleModels.size) modelIds.forEach((model) => validationVisibleModels.add(model));
    q("#validation-models").innerHTML = modelIds.map((model) => `<button type="button" class="validation-model-toggle" data-validation-model="${model}" aria-pressed="${validationVisibleModels.has(model)}" style="--model-color:${cityGridColors[model] || "#64748b"}">${summary.models[model].label}</button>`).join("");
    qa("[data-validation-model]").forEach((button) => button.addEventListener("click", () => {
      const model = button.dataset.validationModel;
      if (validationVisibleModels.has(model)) validationVisibleModels.delete(model); else validationVisibleModels.add(model);
      renderValidation();
    }));
    const selected = modelIds.filter((model) => validationVisibleModels.has(model));
    q("#validation-skill-plot").innerHTML = validationSkillChart(summary, selected);
    attachInteractiveChartTooltip("#validation-skill-plot", "#validation-skill-tooltip");
    const points = summary.matched_points;
    const leadErrors = Object.values(summary.models?.combined?.mae_by_lead || {});
    const combinedText = leadErrors.length ? ` · combined mean endpoint MAE ${(leadErrors.reduce((sum, value) => sum + value, 0) / leadErrors.length).toFixed(2)} ${validationVariable === "temperature" ? "°C" : "mm"}` : "";
    q("#validation-summary").textContent = `${city} · ${points} matched points per available model · Open-Meteo observations${combinedText}${selected.length ? "" : " · all model traces hidden"}`;
    setUrl();
  }

  async function loadCompressedUint16(path) {
    if (!compressedPayloads.has(path)) {
      const pending = (async () => {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${path}`);
        const compressed = await response.arrayBuffer();
        if (typeof DecompressionStream !== "function") throw new Error("This browser does not support gzip decompression streams.");
        const stream = new Response(compressed).body.pipeThrough(new DecompressionStream("gzip"));
        const buffer = await new Response(stream).arrayBuffer();
        return new Uint16Array(buffer);
      })();
      compressedPayloads.set(path, pending);
      pending.catch(() => compressedPayloads.delete(path));
    }
    return compressedPayloads.get(path);
  }

  function setBusy(selector, busy, text = "") {
    const element = q(selector);
    element.classList.toggle("is-loading", busy);
    element.setAttribute("aria-busy", String(busy));
    if (text) element.textContent = text;
  }

  function standaloneColor(variable, value) {
    if (variable === "precipitation") {
      const fraction = Math.max(0, Math.min(1, value / 60));
      return [225 - 185 * fraction, 241 - 80 * fraction, 248 - 25 * fraction];
    }
    const stops = [[255, 255, 204], [254, 217, 118], [253, 141, 60], [240, 59, 32], [189, 0, 38]];
    const scaled = Math.max(0, Math.min(1, value / 45)) * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(scaled));
    const fraction = scaled - index;
    return stops[index].map((channel, offset) => channel + (stops[index + 1][offset] - channel) * fraction);
  }

  function decodeStandalone(encoded, variable) {
    if (encoded === 65535) return null;
    return variable === "temperature" ? (encoded - 5000) / 100 : encoded / 100;
  }

  function clearStandaloneMap(canvas, message) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(360, Math.round(rect.width || 420));
    canvas.height = Math.max(340, Math.round(canvas.width * 1.08));
    const context = canvas.getContext("2d");
    context.fillStyle = "#edf2f4"; context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#607080"; context.font = "13px system-ui"; context.fillText(message, 18, 30);
    canvas._standaloneMap = null;
    canvas.classList.add("is-unavailable");
  }

  function drawStandaloneMap(canvas, encoded, grid, variable, label, readoutSelector) {
    if (!canvas || !encoded || !grid) return;
    const [nLat, nLon] = grid.shape;
    if (encoded.length !== nLat * nLon) throw new Error(`${label}: invalid map payload length`);
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(360, Math.round((rect.width || 420) * ratio));
    const height = Math.max(340, Math.round(width * 1.08));
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d");
    const image = context.createImageData(nLon, nLat);
    for (let yIndex = 0; yIndex < nLat; yIndex += 1) for (let xIndex = 0; xIndex < nLon; xIndex += 1) {
      const value = decodeStandalone(encoded[yIndex * nLon + xIndex], variable);
      const offset = ((nLat - 1 - yIndex) * nLon + xIndex) * 4;
      if (value === null) { image.data[offset + 3] = 0; continue; }
      const rgb = standaloneColor(variable, value);
      image.data[offset] = rgb[0]; image.data[offset + 1] = rgb[1]; image.data[offset + 2] = rgb[2]; image.data[offset + 3] = 255;
    }
    const raster = document.createElement("canvas");
    raster.width = nLon; raster.height = nLat; raster.getContext("2d").putImageData(image, 0, 0);
    context.fillStyle = "#e8f1f5"; context.fillRect(0, 0, width, height);
    context.imageSmoothingEnabled = true; context.drawImage(raster, 0, 0, width, height);
    context.strokeStyle = "rgba(19,44,57,.88)"; context.lineWidth = 1.25 * ratio;
    coastlines.forEach((line) => {
      context.beginPath();
      line.forEach(([longitude, latitude], index) => {
        const x = width * (longitude - grid.lon_min) / (grid.lon_max - grid.lon_min);
        const y = height * (grid.lat_max - latitude) / (grid.lat_max - grid.lat_min);
        if (index) context.lineTo(x, y); else context.moveTo(x, y);
      });
      context.stroke();
    });
    canvas._standaloneMap = { encoded, grid, variable, label, readoutSelector };
    canvas.classList.remove("is-unavailable");
    if (!canvas.dataset.hoverBound) {
      canvas.dataset.hoverBound = "true";
      canvas.addEventListener("pointermove", (event) => {
        const state = canvas._standaloneMap;
        if (!state) return;
        const bounds = canvas.getBoundingClientRect();
        const gx = (event.clientX - bounds.left) / Math.max(bounds.width, 1);
        const gy = (event.clientY - bounds.top) / Math.max(bounds.height, 1);
        if (gx < 0 || gx > 1 || gy < 0 || gy > 1) return;
        const [rows, columns] = state.grid.shape;
        const xIndex = Math.max(0, Math.min(columns - 1, Math.round(gx * (columns - 1))));
        const yIndex = Math.max(0, Math.min(rows - 1, Math.round((1 - gy) * (rows - 1))));
        const value = decodeStandalone(state.encoded[yIndex * columns + xIndex], state.variable);
        const latitude = state.grid.lat_min + yIndex / Math.max(rows - 1, 1) * (state.grid.lat_max - state.grid.lat_min);
        const longitude = state.grid.lon_min + xIndex / Math.max(columns - 1, 1) * (state.grid.lon_max - state.grid.lon_min);
        const readout = q(state.readoutSelector);
        if (readout) readout.textContent = value === null
          ? `${state.label} · missing at ${latitude.toFixed(2)}° N, ${longitude.toFixed(2)}° E`
          : `${state.label} · ${value.toFixed(2)} ${state.variable === "temperature" ? "°C" : "mm"} at ${latitude.toFixed(2)}° N, ${longitude.toFixed(2)}° E`;
      });
    }
  }

  function nativeObservationEntries(product) {
    return (imerg.products?.[product]?.native || []).flatMap((asset) => asset.intervals.map((interval, index) => ({ asset, interval, index })));
  }

  function observationEntries(product, duration) {
    if (duration === "6h") {
      const asset = imerg.products?.[product]?.six_hour;
      return asset ? asset.intervals.map((interval, index) => ({ asset, interval, index })) : [];
    }
    return nativeObservationEntries(product);
  }

  async function observationFrame(product, duration, start, end) {
    const entry = observationEntries(product, duration).find((candidate) => candidate.interval.start_utc === start && candidate.interval.end_utc === end);
    if (!entry) return null;
    const payload = await loadCompressedUint16(entry.asset.path);
    const [, nLat, nLon] = entry.asset.shape;
    const count = nLat * nLon;
    return { values: payload.subarray(entry.index * count, (entry.index + 1) * count), grid: imerg.products[product].grid };
  }

  async function summedNativeObservation(product, start, end) {
    const startMs = new Date(start).getTime(), endMs = new Date(end).getTime();
    const expected = (endMs - startMs) / 1_800_000;
    const entries = nativeObservationEntries(product).filter((entry) => {
      const value = new Date(entry.interval.start_utc).getTime();
      return value >= startMs && value < endMs;
    }).sort((a, b) => new Date(a.interval.start_utc) - new Date(b.interval.start_utc));
    if (!Number.isInteger(expected) || entries.length !== expected || !entries.length) return null;
    if (entries[0].interval.start_utc !== start || entries[entries.length - 1].interval.end_utc !== end) return null;
    const grid = imerg.products[product].grid;
    const count = grid.shape[0] * grid.shape[1];
    const totals = new Float64Array(count);
    const valid = new Uint8Array(count); valid.fill(1);
    for (const entry of entries) {
      const payload = await loadCompressedUint16(entry.asset.path);
      const frame = payload.subarray(entry.index * count, (entry.index + 1) * count);
      for (let index = 0; index < count; index += 1) {
        if (frame[index] === 65535) valid[index] = 0;
        else totals[index] += frame[index];
      }
    }
    const encoded = new Uint16Array(count);
    for (let index = 0; index < count; index += 1) encoded[index] = valid[index] ? Math.min(65534, Math.round(totals[index])) : 65535;
    return { values: encoded, grid };
  }

  function populateSelect(select, entries, value, label) {
    select.innerHTML = entries.map((entry) => `<option value="${entry.id}">${label(entry)}</option>`).join("");
    if (entries.some((entry) => entry.id === value)) select.value = value;
    else if (entries.length) select.value = entries[0].id;
    return select.value;
  }

  async function renderTemporalMaps() {
    const runEntries = Object.entries(imerg.forecast_runs || {}).map(([id, value]) => ({ id, ...value }));
    if (!runEntries.length) {
      clearStandaloneMap(q("#temporal-forecast-canvas"), "Native-time forecast data unavailable.");
      clearStandaloneMap(q("#temporal-early-canvas"), "IMERG data unavailable.");
      clearStandaloneMap(q("#temporal-late-canvas"), "IMERG data unavailable.");
      return;
    }
    temporalInit = populateSelect(q("#temporal-init-select"), runEntries, temporalInit, (entry) => formatInit(entry.initialization_utc));
    const active = imerg.forecast_runs[temporalInit];
    const models = Object.entries(active.models || {}).map(([id, value]) => ({ id, ...value }));
    temporalModel = populateSelect(q("#temporal-model-select"), models, temporalModel, (entry) => entry.label);
    const model = active.models[temporalModel];
    if (!model) return;
    temporalTimeIndex = Math.max(0, Math.min(model.times.length - 1, temporalTimeIndex));
    q("#temporal-time-select").innerHTML = model.times.map((time, index) => `<option value="${index}">${compactValidTime(time.valid_time_utc)} · ${time.interval_hours} h interval</option>`).join("");
    q("#temporal-time-select").value = String(temporalTimeIndex);
    selectButton("[data-temporal-variable]", temporalVariable, "temporalVariable");
    const request = ++temporalRequest;
    setBusy("#temporal-map-note", true, "Loading native-time forecast and matched observations…");
    try {
      const [forecastPayload] = await Promise.all([loadCompressedUint16(model.path), loadCoastlines()]);
      if (request !== temporalRequest) return;
      const [nTime, nLat, nLon] = model.shape;
      const count = nLat * nLon;
      const variableIndex = model.variables.indexOf(temporalVariable);
      const start = variableIndex * nTime * count + temporalTimeIndex * count;
      const frame = forecastPayload.subarray(start, start + count);
      const time = model.times[temporalTimeIndex];
      drawStandaloneMap(q("#temporal-forecast-canvas"), frame, model.grid, temporalVariable, `${model.label} forecast`, "#temporal-map-hover");
      q("#temporal-forecast-caption").textContent = `${model.label} · ${temporalVariable === "temperature" ? "valid" : `${time.interval_hours} h accumulation ending`} ${compactValidTime(time.valid_time_utc)}`;
      if (temporalVariable === "precipitation") {
        const [early, late] = await Promise.all([
          summedNativeObservation("early", time.interval_start_utc, time.valid_time_utc),
          summedNativeObservation("late", time.interval_start_utc, time.valid_time_utc),
        ]);
        if (request !== temporalRequest) return;
        if (early) drawStandaloneMap(q("#temporal-early-canvas"), early.values, early.grid, "precipitation", "IMERG Early", "#temporal-map-hover");
        else clearStandaloneMap(q("#temporal-early-canvas"), "IMERG Early not yet available for this exact interval.");
        if (late) drawStandaloneMap(q("#temporal-late-canvas"), late.values, late.grid, "precipitation", "IMERG Late", "#temporal-map-hover");
        else clearStandaloneMap(q("#temporal-late-canvas"), "IMERG Late not yet available for this exact interval.");
        q("#temporal-early-caption").textContent = "IMERG Early · exact matched accumulation";
        q("#temporal-late-caption").textContent = "IMERG Late · exact matched accumulation";
        q("#temporal-map-note").textContent = `${model.label} rainfall ${exactTime(time.interval_start_utc)} → ${exactTime(time.valid_time_utc)} (${time.interval_hours} h). IMERG is summed from complete native half-hours only; unavailable panels are not interpolated.`;
      } else {
        clearStandaloneMap(q("#temporal-early-canvas"), "IMERG is a precipitation-only product.");
        clearStandaloneMap(q("#temporal-late-canvas"), "IMERG is a precipitation-only product.");
        q("#temporal-map-note").textContent = `${model.label} temperature snapshot valid ${exactTime(time.valid_time_utc)}. This is the model's highest available published cadence.`;
      }
      setBusy("#temporal-map-note", false);
      setUrl();
    } catch (error) {
      setBusy("#temporal-map-note", false);
      q("#temporal-map-note").textContent = `Native-time map unavailable: ${error.message}`;
      console.error(error);
    }
  }

  async function renderImergMaps() {
    const entries = observationEntries("early", imergDuration);
    if (!entries.length) {
      clearStandaloneMap(q("#imerg-early-canvas"), "IMERG Early data unavailable.");
      clearStandaloneMap(q("#imerg-late-canvas"), "IMERG Late data unavailable.");
      return;
    }
    selectButton("[data-imerg-duration]", imergDuration, "imergDuration");
    if (imergTimeIndex < 0 || imergTimeIndex >= entries.length) imergTimeIndex = entries.length - 1;
    q("#imerg-time-select").innerHTML = entries.map((entry, index) => `<option value="${index}">${compactValidTime(entry.interval.start_utc)} → ${compactValidTime(entry.interval.end_utc)}</option>`).join("");
    q("#imerg-time-select").value = String(imergTimeIndex);
    const interval = entries[imergTimeIndex].interval;
    const request = ++imergRequest;
    setBusy("#imerg-map-note", true, "Loading native IMERG maps…");
    try {
      const [early, late] = await Promise.all([
        observationFrame("early", imergDuration, interval.start_utc, interval.end_utc),
        observationFrame("late", imergDuration, interval.start_utc, interval.end_utc),
        loadCoastlines(),
      ]);
      if (request !== imergRequest) return;
      if (early) drawStandaloneMap(q("#imerg-early-canvas"), early.values, early.grid, "precipitation", "IMERG Early", "#imerg-map-hover");
      if (late) drawStandaloneMap(q("#imerg-late-canvas"), late.values, late.grid, "precipitation", "IMERG Late", "#imerg-map-hover");
      q("#imerg-map-note").textContent = `${imergDuration === "30min" ? "Native 30-minute" : "UTC-aligned six-hour"} rainfall · ${exactTime(interval.start_utc)} → ${exactTime(interval.end_utc)} · native 0.1° grid · Early and Late use identical valid times.`;
      setBusy("#imerg-map-note", false);
      setUrl();
    } catch (error) {
      setBusy("#imerg-map-note", false);
      q("#imerg-map-note").textContent = `IMERG map unavailable: ${error.message}`;
      console.error(error);
    }
  }

  function imergValidationValue(row, model) {
    if (model !== imerg.grid_ensemble?.model_id) {
      return imergValidationForecast === "raw" ? row.models?.[model]?.raw_mm : row.models?.[model]?.bias_corrected_mm;
    }
    if (imergValidationForecast === "corrected") return row.combined_mm;
    const inputs = Object.values(row.models || {}).filter((item) => Number.isFinite(item.raw_mm) && Number.isFinite(item.weight));
    const totalWeight = inputs.reduce((sum, item) => sum + item.weight, 0);
    return totalWeight > 0 ? inputs.reduce((sum, item) => sum + item.raw_mm * item.weight, 0) / totalWeight : null;
  }

  function imergValidationRmse(rows, model) {
    const errors = rows.map((row) => {
      const forecast = imergValidationValue(row, model);
      return Number.isFinite(forecast) && Number.isFinite(row.imerg_late_mm) ? (forecast - row.imerg_late_mm) ** 2 : null;
    }).filter(Number.isFinite);
    return errors.length ? Math.sqrt(errors.reduce((sum, value) => sum + value, 0) / errors.length) : null;
  }

  function imergValidationChart(rows, modelIds) {
    if (!rows.length) return '<p class="empty-state"><strong>Waiting for observations.</strong><br>IMERG Late has not completed any six-hour intervals for this initialization yet. Choose an earlier initialization to validate realized forecasts.</p>';
    if (!modelIds.length) return '<p class="empty-state">All model traces are hidden. Select a grey model name above to show it.</p>';
    const width = 1000, height = 430, pad = { l: 58, r: 24, t: 25, b: 58 };
    const timestamps = rows.map((row) => new Date(row.valid_time_utc).getTime());
    const x = (time) => pad.l + (time - timestamps[0]) / Math.max(timestamps[timestamps.length - 1] - timestamps[0], 1) * (width - pad.l - pad.r);
    const values = [];
    rows.forEach((row) => {
      if (imergValidationMetric === "rainfall") {
        if (Number.isFinite(row.imerg_early_mm)) values.push(row.imerg_early_mm);
        if (Number.isFinite(row.imerg_late_mm)) values.push(row.imerg_late_mm);
      }
      modelIds.forEach((model) => {
        const forecast = imergValidationValue(row, model);
        const value = imergValidationMetric === "error" && Number.isFinite(row.imerg_late_mm) ? Math.abs(forecast - row.imerg_late_mm) : forecast;
        if (Number.isFinite(value)) values.push(value);
      });
    });
    const high = Math.max(1, Math.ceil(Math.max(...values, 1) * 1.08));
    const y = (value) => pad.t + (high - value) / high * (height - pad.t - pad.b);
    const grid = [0, high / 2, high].map((value) => `<g><line x1="${pad.l}" x2="${width - pad.r}" y1="${y(value)}" y2="${y(value)}"/><text x="${pad.l - 8}" y="${y(value) + 4}" text-anchor="end">${value.toFixed(1)}</text></g>`).join("");
    const labelEvery = Math.max(1, Math.ceil(rows.length / 7));
    const labels = rows.map((row, index) => {
      if (index % labelEvery && index !== rows.length - 1) return "";
      const date = new Date(row.valid_time_utc);
      const day = date.toLocaleDateString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short" });
      const time = date.toLocaleTimeString("en-GB", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
      return `<text x="${x(date.getTime())}" y="${height - 28}" text-anchor="middle">${day}</text><text x="${x(date.getTime())}" y="${height - 12}" text-anchor="middle">${time} UTC</text>`;
    }).join("");
    const observations = imergValidationMetric === "rainfall" ? [
      { id: "imerg_late", label: "IMERG Late", color: "#172b3a", width: 3, value: (row) => row.imerg_late_mm },
      { id: "imerg_early", label: "IMERG Early", color: "#2a9d8f", width: 2, dash: "5 4", value: (row) => row.imerg_early_mm },
    ] : [];
    const series = [
      ...observations,
      ...modelIds.map((model) => ({ id: model, label: modelLabel(model), color: cityGridColors[model] || "#64748b", width: model === imerg.grid_ensemble?.model_id ? 3 : 2, value: (row) => {
        const forecast = imergValidationValue(row, model);
        return imergValidationMetric === "error" && Number.isFinite(forecast) && Number.isFinite(row.imerg_late_mm) ? Math.abs(forecast - row.imerg_late_mm) : forecast;
      } })),
    ];
    const traces = series.map((item) => {
      const points = rows.map((row) => ({ row, value: item.value(row) })).filter((item) => Number.isFinite(item.value));
      if (!points.length) return "";
      const line = path(points.map((item) => [x(new Date(item.row.valid_time_utc).getTime()), y(item.value)]));
      const units = imergValidationMetric === "error" ? "mm absolute error" : "mm / 6 h";
      const dots = points.map((point) => `<circle class="validation-chart-point" data-validation-point tabindex="0" cx="${x(new Date(point.row.valid_time_utc).getTime())}" cy="${y(point.value)}" r="${item.id.startsWith("imerg_") ? 3.8 : 3.2}" fill="${item.color}" data-label="${item.label}" data-value="${point.value.toFixed(2)}" data-units="${units}" data-detail="${exactTime(point.row.interval_start_utc)} → ${exactTime(point.row.valid_time_utc)}" aria-label="${item.label}, ${point.value.toFixed(2)} ${units}"></circle>`).join("");
      return `<path class="validation-series" d="${line}" fill="none" stroke="${item.color}" stroke-width="${item.width}" ${item.dash ? `stroke-dasharray="${item.dash}"` : ""}/>${dots}`;
    }).join("");
    const axisTitle = imergValidationMetric === "error" ? "Absolute error against IMERG Late (mm)" : "Six-hour rainfall (mm)";
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Interactive six-hour model rainfall validation against IMERG"><g class="interactive-chart-grid">${grid}${labels}<text class="axis-title" x="12" y="18">${axisTitle}</text></g>${traces}</svg>`;
  }

  function renderImergCityValidation() {
    const runEntries = Object.entries(imerg.grid_ensemble?.runs || {}).map(([id, value]) => ({ id, ...value }));
    if (!runEntries.length) {
      q("#imerg-validation-summary").textContent = "Common-grid IMERG validation is unavailable for this selection.";
      q("#imerg-validation-chart").innerHTML = "";
      return;
    }
    const scored = runEntries.map((entry) => ({
      entry,
      observed: (entry.city_rows?.[city] || []).filter((row) => Number.isFinite(row.imerg_late_mm)).length,
    })).sort((a, b) => b.observed - a.observed || new Date(b.entry.initialization_utc) - new Date(a.entry.initialization_utc));
    const selectedScore = scored.find((item) => item.entry.id === imergValidationInit);
    if (!selectedScore || (!imergValidationInitTouched && selectedScore.observed === 0)) {
      imergValidationInit = scored[0].entry.id;
    }
    imergValidationInit = populateSelect(q("#imerg-validation-init"), runEntries, imergValidationInit, (entry) => formatInit(entry.initialization_utc));
    const run = imerg.grid_ensemble.runs[imergValidationInit];
    const modelIds = [...run.source_models, imerg.grid_ensemble.model_id];
    [...imergVisibleModels].forEach((model) => { if (!modelIds.includes(model)) imergVisibleModels.delete(model); });
    if (!imergVisibleModels.size) modelIds.forEach((model) => imergVisibleModels.add(model));
    selectButton("[data-imerg-metric]", imergValidationMetric, "imergMetric");
    selectButton("[data-imerg-forecast]", imergValidationForecast, "imergForecast");
    q("#imerg-validation-models").innerHTML = modelIds.map((model) => `<button type="button" class="validation-model-toggle" data-imerg-validation-model="${model}" aria-pressed="${imergVisibleModels.has(model)}" style="--model-color:${cityGridColors[model] || "#64748b"}">${modelLabel(model)}</button>`).join("");
    qa("[data-imerg-validation-model]").forEach((button) => button.addEventListener("click", () => {
      const model = button.dataset.imergValidationModel;
      if (imergVisibleModels.has(model)) imergVisibleModels.delete(model); else imergVisibleModels.add(model);
      renderImergCityValidation();
    }));
    const rows = (run.city_rows?.[city] || []).filter((row) => Number.isFinite(row.imerg_late_mm));
    const selected = modelIds.filter((model) => imergVisibleModels.has(model));
    q("#imerg-validation-plot").innerHTML = imergValidationChart(rows, selected);
    attachInteractiveChartTooltip("#imerg-validation-plot", "#imerg-validation-tooltip");
    const realized = rows.length;
    const metrics = selected.map((model) => {
      const value = imergValidationRmse(rows, model);
      return `${modelLabel(model)} ${value == null ? "pending" : `${value.toFixed(2)} mm RMSE`}`;
    });
    q("#imerg-validation-scores").innerHTML = metrics.length ? metrics.map((metric) => `<span>${metric}</span>`).join("") : "";
    q("#imerg-validation-summary").textContent = realized
      ? `${city} · ${realized} realized common-grid six-hour intervals · ${imergValidationForecast === "raw" ? "raw" : "bias-corrected"} forecasts${metrics.length ? "" : " · all model traces hidden"}. Biases and weights use only observations valid by initialization.`
      : `${city} · no completed IMERG Late intervals for this initialization yet. Choose an earlier initialization to validate realized forecasts.`;
    setUrl();
  }

  qa("[data-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
  q("#init-select").addEventListener("change", (event) => { init = event.target.value; view = { scale: 1, x: 0, y: 0 }; renderRun(); });
  q("#city-select").addEventListener("change", (event) => { city = event.target.value; renderWeather(); renderValidation(); renderImergCityValidation(); setUrl(); });
  qa("[data-weather-variable]").forEach((button) => button.addEventListener("click", () => { weatherVariable = button.dataset.weatherVariable; renderWeather(); setUrl(); }));
  qa("[data-map-variable]").forEach((button) => button.addEventListener("click", () => { mapVariable = button.dataset.mapVariable; renderMapControls(); }));
  qa("[data-map-day]").forEach((button) => button.addEventListener("click", () => { mapDay = button.dataset.mapDay; renderMapControls(); }));
  qa("[data-map-model]").forEach((button) => button.addEventListener("click", () => { mapModel = button.dataset.mapModel; renderMapControls(); }));
  qa("[data-validation-city]").forEach((button) => button.addEventListener("click", () => { city = button.dataset.validationCity; q("#city-select").value = city; renderWeather(); renderValidation(); renderImergCityValidation(); }));
  qa("[data-validation-variable]").forEach((button) => button.addEventListener("click", () => { validationVariable = button.dataset.validationVariable; renderValidation(); }));
  qa("[data-match-variable]").forEach((button) => button.addEventListener("click", () => { matchVariable = button.dataset.matchVariable; renderValidation(); }));
  q("#match-init-select").addEventListener("change", (event) => { matchInit = event.target.value; renderValidation(); });
  q("#temporal-init-select").addEventListener("change", (event) => { temporalInit = event.target.value; temporalModel = ""; temporalTimeIndex = 0; renderTemporalMaps(); });
  q("#temporal-model-select").addEventListener("change", (event) => { temporalModel = event.target.value; temporalTimeIndex = 0; renderTemporalMaps(); });
  q("#temporal-time-select").addEventListener("change", (event) => { temporalTimeIndex = Number(event.target.value); renderTemporalMaps(); });
  qa("[data-temporal-variable]").forEach((button) => button.addEventListener("click", () => { temporalVariable = button.dataset.temporalVariable; renderTemporalMaps(); }));
  qa("[data-imerg-duration]").forEach((button) => button.addEventListener("click", () => { imergDuration = button.dataset.imergDuration; imergTimeIndex = -1; renderImergMaps(); }));
  qa("[data-imerg-metric]").forEach((button) => button.addEventListener("click", () => { imergValidationMetric = button.dataset.imergMetric; renderImergCityValidation(); }));
  qa("[data-imerg-forecast]").forEach((button) => button.addEventListener("click", () => { imergValidationForecast = button.dataset.imergForecast; renderImergCityValidation(); }));
  q("#imerg-time-select").addEventListener("change", (event) => { imergTimeIndex = Number(event.target.value); renderImergMaps(); });
  q("#imerg-validation-init").addEventListener("change", (event) => { imergValidationInit = event.target.value; imergValidationInitTouched = true; imergVisibleModels.clear(); renderImergCityValidation(); });
  q("#map-reset").addEventListener("click", () => { view = { scale: 1, x: 0, y: 0 }; drawMap(); });
  const canvas = q("#forecast-canvas");
  canvas.addEventListener("pointerdown", (event) => { hideMapTooltip(); drag = { x: event.clientX, y: event.clientY, moved: false }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!drag) { showMapTooltip(event); return; } const ratio = window.devicePixelRatio || 1; const dx = (event.clientX - drag.x) * ratio; const dy = (event.clientY - drag.y) * ratio; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true; view.x += dx; view.y += dy; drag.x = event.clientX; drag.y = event.clientY; drawMap(); });
  canvas.addEventListener("pointerleave", () => { if (!drag) hideMapTooltip(); });
  canvas.addEventListener("pointerup", (event) => {
    if (!drag?.moved) {
      const point = mapCoordinates(event);
      const nearest = point && Object.entries(validation.cities).map(([name, item]) => [name, Math.hypot((item.longitude - point.longitude) * .9, item.latitude - point.latitude)]).sort((a, b) => a[1] - b[1])[0];
      if (nearest && nearest[1] < 1.5) { city = nearest[0]; q("#city-select").value = city; renderWeather(); renderValidation(); activateTab("validation"); }
    }
    drag = null;
  });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); hideMapTooltip(); view.scale = Math.max(1, Math.min(4, view.scale * (event.deltaY < 0 ? 1.15 : .87))); drawMap(); }, { passive: false });
  window.addEventListener("resize", () => { if (tab === "maps") drawMap(); });

  activateTab(tab, false);
  renderRun();
  renderValidation();
})();
