(() => {
  "use strict";
  const dataNode = document.querySelector("#site-data");
  if (!dataNode) return;
  const site = JSON.parse(dataNode.textContent);
  const archive = site.archive;
  const validation = site.validation;
  const weather = site.weather;
  const runs = archive.runs;
  const params = new URLSearchParams(location.search);
  const q = (selector) => document.querySelector(selector);
  const qa = (selector) => [...document.querySelectorAll(selector)];
  const allowedTabs = new Set(["weather", "maps", "validation", "method"]);
  const allowedVariables = new Set(["temperature", "temperature_high", "temperature_low", "precipitation"]);
  const allowedDays = new Set(["1", "3", "5"]);
  const runIds = new Set(runs.map((run) => run.id));
  const cityNames = Object.keys(validation.cities);
  let tab = allowedTabs.has(params.get("tab")) ? params.get("tab") : "weather";
  let init = runIds.has(params.get("init")) ? params.get("init") : runs[0].id;
  let city = cityNames.includes(params.get("city")) ? params.get("city") : cityNames[0];
  let weatherVariable = params.get("weather") === "precipitation" ? "precipitation" : "temperature";
  let mapVariable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let mapDay = allowedDays.has(params.get("day")) ? params.get("day") : "1";
  let mapModel = params.get("model") || runs[0].available_models?.[0] || runs[0].models[0].id;
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
    const values = { tab, init, city, weather: weatherVariable, variable: mapVariable, day: mapDay,
      model: mapModel, validation: validationVariable, match_init: matchInit, match_variable: matchVariable };
    Object.entries(values).forEach(([key, value]) => next.searchParams.set(key, value));
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
  function runModels(run = activeRun()) { return run.available_models || run.models.map((model) => model.id); }
  function modelLabel(model) {
    const item = site.models.find((candidate) => candidate.id === model);
    return item?.label || model;
  }
  function formatInit(value) {
    return new Date(value).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short",
      year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) + " UTC";
  }

  function renderRun() {
    const run = activeRun();
    q("#init-select").value = init;
    q("#run-status").textContent = `${formatInit(run.initialization_utc)} · ${runModels(run).length} of ${site.models.length} models`;
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
    renderWeather();
    loadMap();
    setUrl();
  }

  function path(points) {
    if (!points.length) return "";
    return points.map((point, index) => `${index ? "L" : "M"}${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" ");
  }

  function weatherChart(days) {
    const width = 920, height = 260, pad = { l: 45, r: 20, t: 22, b: 42 };
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
    const dots = days.map((day, index) => `<g><circle cx="${x(index)}" cy="${y(value(day))}" r="4"/><text x="${x(index)}" y="${y(value(day)) - 12}" text-anchor="middle">${value(day).toFixed(1)}${weatherVariable === "temperature" ? "°" : " mm"}</text><text class="date" x="${x(index)}" y="${height - 15}" text-anchor="middle">${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { weekday: "short", day: "numeric" })}</text></g>`).join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Five-day ${weatherVariable} forecast"><g class="chart-grid">${grid}</g><path class="weather-area" d="${area}"/><path class="weather-line" d="${line}"/><g class="weather-points">${dots}</g></svg>`;
  }

  function renderWeather() {
    q("#city-select").value = city;
    selectButton("[data-weather-variable]", weatherVariable, "weatherVariable");
    const run = weather.runs[init];
    const item = run?.cities?.[city];
    const days = item?.days || [];
    q("#weather-location").textContent = city;
    q("#weather-meta").textContent = item
      ? `Initialized ${formatInit(run.initialization_utc)} · ${item.available_models.length} contributing model${item.available_models.length === 1 ? "" : "s"}`
      : "Forecast unavailable for this selection";
    const first = days[0];
    q("#weather-now").innerHTML = first
      ? `<span aria-hidden="true">${first.symbol}</span><strong>${Math.round(first.mean_c)}°C</strong><small>${first.condition}<br>${first.precip_mm.toFixed(1)} mm in 24 h</small>`
      : "";
    q("#weather-chart").innerHTML = weatherChart(days);
    q("#daily-cards").innerHTML = days.map((day, index) => `<article class="day-card${index === 0 ? " is-first" : ""}"><strong>${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { weekday: "short" })}</strong><time>${new Date(day.valid_date + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short" })}</time><span class="weather-icon" aria-label="${day.condition}">${day.symbol}</span><p><b>${Math.round(day.high_c)}°</b> <span>${Math.round(day.low_c)}°</span></p><small>${day.precip_mm.toFixed(1)} mm</small></article>`).join("");
    q("#blend-note").textContent = item
      ? `Temperature: ${item.temperature_method} weights · rainfall: ${item.precipitation_method} weights. Rainfall is a 24-hour accumulation, not a probability.`
      : "";
  }

  function mapColor(number) {
    if (mapVariable === "precipitation") {
      const t = Math.max(0, Math.min(1, number / 120));
      return [225 - 185 * t, 241 - 80 * t, 248 - 25 * t];
    }
    const t = Math.max(0, Math.min(1, (number - 5) / 40));
    return [45 + 205 * t, 110 + 70 * (1 - Math.abs(t - .5) * 2), 190 - 145 * t];
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

  function drawMap(run = activeRun()) {
    const canvas = q("#forecast-canvas");
    if (!payload || !canvas || !run.grid_metadata?.shape) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(600, Math.round(rect.width * ratio));
    const height = Math.max(420, Math.round(Math.max(rect.width * .56, 420) * ratio));
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
      const number = mapVariable === "precipitation" ? encoded / 10 : (encoded - 5000) / 100;
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
    const names = { temperature: "Temperature", temperature_high: "Daily high", temperature_low: "Daily low", precipitation: "Accumulated rainfall" };
    q("#map-title").textContent = `${names[mapVariable]} · Day ${mapDay}`;
    q("#map-description").textContent = `${modelLabel(mapModel)} · initialized ${formatInit(run.initialization_utc)}`;
    q("#map-readout").textContent = `T+${Number(mapDay) * 24} h · drag to pan · scroll to zoom`;
  }

  function renderMapControls() {
    selectButton("[data-map-variable]", mapVariable, "mapVariable");
    selectButton("[data-map-day]", mapDay, "mapDay");
    view = { scale: 1, x: 0, y: 0 };
    loadMap();
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
    const points = item.summary[validationVariable].matched_points;
    q("#validation-summary").textContent = `${city} · ${points} matched points per available model · Open-Meteo observations`;
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
  canvas.addEventListener("pointerdown", (event) => { drag = { x: event.clientX, y: event.clientY, moved: false }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!drag) return; const ratio = window.devicePixelRatio || 1; const dx = (event.clientX - drag.x) * ratio; const dy = (event.clientY - drag.y) * ratio; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true; view.x += dx; view.y += dy; drag.x = event.clientX; drag.y = event.clientY; drawMap(); });
  canvas.addEventListener("pointerup", (event) => {
    if (!drag?.moved) {
      const run = activeRun(), meta = run.grid_metadata, box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
      const gx = ((event.clientX - box.left) * ratio - view.x) / view.scale / canvas.width;
      const gy = ((event.clientY - box.top) * ratio - view.y) / view.scale / canvas.height;
      const lon = meta.bounding_box.lon_min + gx * (meta.bounding_box.lon_max - meta.bounding_box.lon_min);
      const lat = meta.bounding_box.lat_max - gy * (meta.bounding_box.lat_max - meta.bounding_box.lat_min);
      const nearest = Object.entries(validation.cities).map(([name, item]) => [name, Math.hypot((item.longitude - lon) * .9, item.latitude - lat)]).sort((a, b) => a[1] - b[1])[0];
      if (nearest && nearest[1] < 1.5) { city = nearest[0]; q("#city-select").value = city; renderWeather(); renderValidation(); activateTab("validation"); }
    }
    drag = null;
  });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); view.scale = Math.max(1, Math.min(4, view.scale * (event.deltaY < 0 ? 1.15 : .87))); drawMap(); }, { passive: false });
  window.addEventListener("resize", () => { if (tab === "maps") drawMap(); });

  activateTab(tab, false);
  renderRun();
  renderValidation();
})();
