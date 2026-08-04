(() => {
  "use strict";
  const dataNode = document.querySelector("#site-data");
  if (!dataNode) return;
  const site = JSON.parse(dataNode.textContent);
  const archive = site.archive;
  const validation = site.validation;
  const weather = site.weather;
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
    gefs: "#be7910", aifs: "#087f73", ifs_ens: "#34495e" };
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
  let payload = null;
  let coastlines = [];
  let coastlinePromise = null;
  let mapRequest = 0;
  let view = { scale: 1, x: 0, y: 0 };
  let drag = null;

  function setUrl() {
    const next = new URL(location.href);
    const values = { tab, init, city, weather: weatherVariable, weather_day: weatherDay, grid_model: cityGridModel,
      variable: mapVariable, day: mapDay,
      model: mapModel, validation: validationVariable, match_init: matchInit, match_variable: matchVariable };
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
    if (tab === "maps") requestAnimationFrame(() => drawMap(activeRun()));
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
    return item?.label || model;
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
    const points = summary.matched_points;
    const leadErrors = Object.values(summary.models?.combined?.mae_by_lead || {});
    const combinedText = leadErrors.length ? ` · combined mean endpoint MAE ${(leadErrors.reduce((sum, value) => sum + value, 0) / leadErrors.length).toFixed(2)} ${validationVariable === "temperature" ? "°C" : "mm"}` : "";
    q("#validation-summary").textContent = `${city} · ${points} matched points per available model · Open-Meteo observations${combinedText}`;
    setUrl();
  }

  qa("[data-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
  q("#init-select").addEventListener("change", (event) => { init = event.target.value; view = { scale: 1, x: 0, y: 0 }; renderRun(); });
  q("#city-select").addEventListener("change", (event) => { city = event.target.value; renderWeather(); renderValidation(); setUrl(); });
  qa("[data-weather-variable]").forEach((button) => button.addEventListener("click", () => { weatherVariable = button.dataset.weatherVariable; renderWeather(); setUrl(); }));
  qa("[data-map-variable]").forEach((button) => button.addEventListener("click", () => { mapVariable = button.dataset.mapVariable; renderMapControls(); }));
  qa("[data-map-day]").forEach((button) => button.addEventListener("click", () => { mapDay = button.dataset.mapDay; renderMapControls(); }));
  qa("[data-map-model]").forEach((button) => button.addEventListener("click", () => { mapModel = button.dataset.mapModel; renderMapControls(); }));
  qa("[data-validation-city]").forEach((button) => button.addEventListener("click", () => { city = button.dataset.validationCity; q("#city-select").value = city; renderWeather(); renderValidation(); }));
  qa("[data-validation-variable]").forEach((button) => button.addEventListener("click", () => { validationVariable = button.dataset.validationVariable; renderValidation(); }));
  qa("[data-match-variable]").forEach((button) => button.addEventListener("click", () => { matchVariable = button.dataset.matchVariable; renderValidation(); }));
  q("#match-init-select").addEventListener("change", (event) => { matchInit = event.target.value; renderValidation(); });
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
