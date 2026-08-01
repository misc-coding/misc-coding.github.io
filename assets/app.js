(() => {
  const variableButtons = [...document.querySelectorAll("[data-variable-button]")];
  const dayButtons = [...document.querySelectorAll("[data-day-button]")];
  const validationCityButtons = [...document.querySelectorAll("[data-validation-city]")];
  const validationVariableButtons = [...document.querySelectorAll("[data-validation-variable]")];
  const validationImage = document.querySelector("#validation-image");
  const validationSummary = document.querySelector("#validation-summary");
  const matchInitSelect = document.querySelector("#match-init-select");
  const matchVariableButtons = [...document.querySelectorAll("[data-match-variable]")];
  const matchImage = document.querySelector("#match-image");
  const runSelect = document.querySelector("#run-select");
  const runSummary = document.querySelector("#run-summary");
  const views = [...document.querySelectorAll(".forecast-view")];
  const siteData = JSON.parse(document.querySelector("#archive-data").textContent);
  const runs = siteData.runs;
  const validation = siteData.validation;
  const mapModelButtons = [...document.querySelectorAll("[data-map-model]")];
  const canvas = document.querySelector("#forecast-canvas");
  const mapTitle = document.querySelector("#map-title");
  const mapDescription = document.querySelector("#map-description");
  const mapReadout = document.querySelector("#map-readout");
  const cityReadout = document.querySelector("#city-readout");
  const combination = siteData.combination || { cities: {} };
  const combinationCityButtons = [...document.querySelectorAll("[data-combination-city]")];
  const combinationTitle = document.querySelector("#combination-title");
  const combinationSummary = document.querySelector("#combination-summary");
  const combinationChart = document.querySelector("#combination-chart");
  const combinationWeights = document.querySelector("#combination-weights");
  const params = new URLSearchParams(window.location.search);
  const allowedVariables = new Set(["temperature", "temperature_high", "temperature_low", "precipitation"]);
  const allowedDays = new Set(["1", "3", "5"]);
  const allowedInits = new Set(runs.map((run) => run.id));
  let variable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let day = allowedDays.has(params.get("day")) ? params.get("day") : "1";
  let init = allowedInits.has(params.get("init")) ? params.get("init") : runs[0].id;
  let validationCity = Object.keys(validation.cities).includes(params.get("city")) ? params.get("city") : Object.keys(validation.cities)[0];
  let validationVariable = allowedVariables.has(params.get("validation")) ? params.get("validation") : "temperature";
  let matchInit = allowedInits.has(params.get("match_init")) ? params.get("match_init") : runs[0].id;
  let matchVariable = allowedVariables.has(params.get("match_variable")) ? params.get("match_variable") : "precipitation";
  let combinationCity = Object.keys(combination.cities).includes(params.get("combination_city")) ? params.get("combination_city") : Object.keys(combination.cities)[0];
  let mapModel = params.get("model") || mapModelButtons[0]?.dataset.mapModel;
  let payload = null;
  let view = { scale: 1, x: 0, y: 0 };
  let drag = null;

  function render(updateUrl = true) {
    const active = runs.find((run) => run.id === init);
    variableButtons.forEach((button) => {
      button.disabled = false;
      button.title = "";
      button.setAttribute("aria-pressed", String(button.dataset.variableButton === variable));
    });
    dayButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.dayButton === day)));
    mapModelButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mapModel === mapModel)));
    runSelect.value = init;
    runSummary.textContent = `Initialized ${new Date(active.initialization_utc).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false })} UTC · 6 experts · 5-day horizon`;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("init", init);
      next.searchParams.set("variable", variable);
      next.searchParams.set("day", day);
      next.searchParams.set("model", mapModel);
      history.replaceState(null, "", next);
    }
    loadMap(active);
  }

  function color(value) {
    if (variable === "precipitation") {
      const t = Math.max(0, Math.min(1, value / 120));
      return `hsl(${205 - t * 150} 78% ${92 - t * 47}%)`;
    }
    const t = Math.max(0, Math.min(1, (value - 5) / 40));
    return `hsl(${235 - t * 235} 82% ${35 + 27 * t}%)`;
  }

  function rgb(value) {
    const hsl = color(value).match(/([\d.]+)/g).map(Number);
    let [h, s, l] = hsl; h /= 360; s /= 100; l /= 100;
    const hue = (n) => { const k = (n + h * 12) % 12; return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return [Math.round(255 * hue(0)), Math.round(255 * hue(8)), Math.round(255 * hue(4))];
  }

  async function loadMap(active) {
    if (!canvas || !active.grid_metadata?.shape) return;
    const url = `assets/map_data/${init}/${mapModel}.bin`;
    mapReadout.textContent = "Loading compact grid…";
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(response.statusText);
      payload = new Uint16Array(await response.arrayBuffer());
      drawMap(active);
    } catch (error) {
      mapReadout.textContent = "Grid unavailable for this selection.";
      console.error(error);
    }
  }

  function drawMap(active) {
    if (!payload || !canvas) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(640, Math.round(rect.width * devicePixelRatio));
    const height = Math.max(420, Math.round(rect.height * devicePixelRatio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const metadata = active.grid_metadata;
    const [nLead, nLat, nLon] = metadata.shape;
    const variableIndex = metadata.variables.indexOf(variable);
    const dayIndex = metadata.lead_days.indexOf(Number(day));
    const count = nLead * nLat * nLon;
    const start = variableIndex * count + dayIndex * nLat * nLon;
    const image = ctx.createImageData(nLon, nLat);
    for (let y = 0; y < nLat; y++) for (let x = 0; x < nLon; x++) {
      const encoded = payload[start + y * nLon + x];
      const offset = ((nLat - 1 - y) * nLon + x) * 4;
      if (encoded === 65535) { image.data[offset + 3] = 0; continue; }
      const value = variable === "precipitation" ? encoded / 10 : (encoded - 5000) / 100;
      const pixel = rgb(value);
      image.data[offset] = pixel[0]; image.data[offset + 1] = pixel[1]; image.data[offset + 2] = pixel[2]; image.data[offset + 3] = 255;
    }
    const raster = document.createElement("canvas"); raster.width = nLon; raster.height = nLat; raster.getContext("2d").putImageData(image, 0, 0);
    ctx.fillStyle = "#071923"; ctx.fillRect(0, 0, width, height);
    const scale = view.scale; const x0 = view.x; const y0 = view.y;
    ctx.save(); ctx.translate(x0, y0); ctx.scale(scale, scale); ctx.imageSmoothingEnabled = true;
    ctx.drawImage(raster, 0, 0, width, height);
    ctx.strokeStyle = "rgba(255,255,255,.22)"; ctx.lineWidth = 1 / scale;
    for (let fraction = .2; fraction < 1; fraction += .2) { ctx.beginPath(); ctx.moveTo(width * fraction, 0); ctx.lineTo(width * fraction, height); ctx.moveTo(0, height * fraction); ctx.lineTo(width, height * fraction); ctx.stroke(); }
    const cities = Object.entries(validation.cities);
    cities.forEach(([name, city]) => {
      const x = width * (city.longitude - metadata.bounding_box.lon_min) / (metadata.bounding_box.lon_max - metadata.bounding_box.lon_min);
      const y = height * (metadata.bounding_box.lat_max - city.latitude) / (metadata.bounding_box.lat_max - metadata.bounding_box.lat_min);
      ctx.beginPath(); ctx.arc(x, y, 7 / scale, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#f2553d"; ctx.lineWidth = 3 / scale; ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = `${13 / scale}px Inter, sans-serif`; ctx.fillText(name, x + 10 / scale, y - 9 / scale);
    });
    ctx.restore();
    const label = variable.replace("temperature_high", "daily high").replace("temperature_low", "daily low").replace("precipitation", "rain accumulation");
    mapTitle.textContent = `${label} · Day ${day}`;
    mapDescription.textContent = `${mapModelButtons.find((button) => button.dataset.mapModel === mapModel)?.textContent || mapModel} · ${metadata.shape[1]} × ${metadata.shape[2]} grid · drag / scroll to navigate`;
    mapReadout.textContent = `T+${Number(day) * 24} h · ${metadata.bounding_box.lat_min}–${metadata.bounding_box.lat_max}°N · ${metadata.bounding_box.lon_min}–${metadata.bounding_box.lon_max}°E`;
  }

  function renderValidation(updateUrl = true) {
    validationCityButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.validationCity === validationCity)));
    validationVariableButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.validationVariable === validationVariable)));
    const active = validation.cities[validationCity];
    const image = active.images[validationVariable];
    const points = active.summary[validationVariable].matched_points;
    validationImage.src = image.path;
    validationImage.alt = image.alt;
    validationSummary.textContent = `${validationCity} · ${points} matched forecast–observation pairs per model · Open-Meteo ground truth`;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("city", validationCity);
      next.searchParams.set("validation", validationVariable);
      history.replaceState(null, "", next);
    }
    renderMatchedTimeseries(updateUrl);
  }

  function renderMatchedTimeseries(updateUrl = true) {
    matchVariableButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.matchVariable === matchVariable)));
    const image = validation.cities[validationCity].timeseries[matchInit][matchVariable];
    matchInitSelect.value = matchInit;
    matchImage.src = image.path;
    matchImage.alt = image.alt;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("match_init", matchInit);
      next.searchParams.set("match_variable", matchVariable);
      history.replaceState(null, "", next);
    }
  }

  function renderCombination(updateUrl = true) {
    if (!combinationCity || !combination.cities[combinationCity]) return;
    const active = combination.cities[combinationCity];
    combinationCityButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.combinationCity === combinationCity)));
    combinationTitle.textContent = `${combinationCity} · 2 m temperature`;
    combinationSummary.textContent = `${active.method.toUpperCase()} learner · ${active.backtest_steps} matched steps · blend RMSE ${active.backtest_rmse_c.toFixed(2)} °C · uniform ${active.uniform_rmse_c.toFixed(2)} °C`;
    const points = active.points;
    if (!points.length) {
      combinationChart.innerHTML = '<p class="tag">No forward points are currently available.</p>';
    } else {
      const width = 900, height = 290, pad = { left: 46, right: 18, top: 20, bottom: 42 };
      const values = points.flatMap((point) => [point.low_c, point.high_c, point.combined_c]);
      const lo = Math.floor((Math.min(...values) - 1) / 2) * 2;
      const hi = Math.ceil((Math.max(...values) + 1) / 2) * 2;
      const x = (i) => pad.left + (i / Math.max(points.length - 1, 1)) * (width - pad.left - pad.right);
      const y = (value) => pad.top + (hi - value) / Math.max(hi - lo, 1) * (height - pad.top - pad.bottom);
      const line = points.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point.combined_c).toFixed(1)}`).join(" ");
      const upper = points.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point.high_c).toFixed(1)}`).join(" ");
      const lower = [...points].reverse().map((point, reverseIndex) => `L${x(points.length - 1 - reverseIndex).toFixed(1)},${y(point.low_c).toFixed(1)}`).join(" ");
      const grid = [lo, (lo + hi) / 2, hi].map((value) => `<g><line x1="${pad.left}" x2="${width - pad.right}" y1="${y(value)}" y2="${y(value)}"/><text x="${pad.left - 8}" y="${y(value) + 4}" text-anchor="end">${value.toFixed(0)}°</text></g>`).join("");
      const labels = points.map((point, index) => `<text x="${x(index)}" y="${height - 16}" text-anchor="middle">${new Date(point.valid_time_utc).toLocaleDateString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short" })}</text>`).join("");
      combinationChart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><g class="combo-grid">${grid}</g><path class="combo-range" d="${upper} ${lower} Z"/><path class="combo-line" d="${line}"/>${labels}</svg><p class="chart-key"><span></span>AdaWeather online blend <i></i>source-model spread</p>`;
    }
    combinationWeights.innerHTML = Object.entries(active.weights).map(([model, weight]) => `<div><span>${model.replace("weathernext2", "WeatherNext 2").replace("ifs_ens", "IFS-ENS").toUpperCase()}</span><strong>${(weight * 100).toFixed(1)}%</strong></div>`).join("");
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("combination_city", combinationCity);
      history.replaceState(null, "", next);
    }
  }

  variableButtons.forEach((button) => button.addEventListener("click", () => { variable = button.dataset.variableButton; render(); }));
  dayButtons.forEach((button) => button.addEventListener("click", () => { day = button.dataset.dayButton; render(); }));
  runSelect.addEventListener("change", () => { init = runSelect.value; render(); });
  validationCityButtons.forEach((button) => button.addEventListener("click", () => { validationCity = button.dataset.validationCity; renderValidation(); }));
  validationVariableButtons.forEach((button) => button.addEventListener("click", () => { validationVariable = button.dataset.validationVariable; renderValidation(); }));
  matchInitSelect.addEventListener("change", () => { matchInit = matchInitSelect.value; renderMatchedTimeseries(); });
  matchVariableButtons.forEach((button) => button.addEventListener("click", () => { matchVariable = button.dataset.matchVariable; renderMatchedTimeseries(); }));
  combinationCityButtons.forEach((button) => button.addEventListener("click", () => { combinationCity = button.dataset.combinationCity; renderCombination(); }));
  mapModelButtons.forEach((button) => button.addEventListener("click", () => { mapModel = button.dataset.mapModel; view = { scale: 1, x: 0, y: 0 }; render(); }));
  document.querySelector("#map-reset")?.addEventListener("click", () => { view = { scale: 1, x: 0, y: 0 }; drawMap(runs.find((run) => run.id === init)); });
  canvas?.addEventListener("pointerdown", (event) => { drag = { x: event.clientX, y: event.clientY, moved: false }; canvas.setPointerCapture(event.pointerId); });
  canvas?.addEventListener("pointermove", (event) => { if (!drag) return; const dx = (event.clientX - drag.x) * devicePixelRatio; const dy = (event.clientY - drag.y) * devicePixelRatio; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true; view.x += dx; view.y += dy; drag.x = event.clientX; drag.y = event.clientY; drawMap(runs.find((run) => run.id === init)); });
  canvas?.addEventListener("pointerup", (event) => {
    if (!drag?.moved) {
      const active = runs.find((run) => run.id === init); const meta = active.grid_metadata; const box = canvas.getBoundingClientRect();
      const px = (event.clientX - box.left) * devicePixelRatio; const py = (event.clientY - box.top) * devicePixelRatio;
      const gx = (px - view.x) / view.scale / canvas.width; const gy = (py - view.y) / view.scale / canvas.height;
      const lon = meta.bounding_box.lon_min + gx * (meta.bounding_box.lon_max - meta.bounding_box.lon_min); const lat = meta.bounding_box.lat_max - gy * (meta.bounding_box.lat_max - meta.bounding_box.lat_min);
      const nearest = Object.entries(validation.cities).map(([name, city]) => [name, Math.hypot((city.longitude - lon) * .9, city.latitude - lat)]).sort((a, b) => a[1] - b[1])[0];
      if (nearest && nearest[1] < 1.5) { validationCity = nearest[0]; combinationCity = nearest[0]; renderValidation(); renderCombination(); cityReadout.innerHTML = `<dt>${nearest[0]}</dt><dd>Matched validation and online-combination panel selected.</dd>`; document.querySelector("#combination").scrollIntoView({ behavior: "smooth", block: "start" }); }
    }
    drag = null;
  });
  canvas?.addEventListener("wheel", (event) => { event.preventDefault(); const factor = event.deltaY < 0 ? 1.15 : .87; view.scale = Math.max(1, Math.min(4, view.scale * factor)); drawMap(runs.find((run) => run.id === init)); }, { passive: false });

  const pinLocations = [
    ["Delhi", "31%", "38%"], ["Mumbai", "23%", "56%"],
    ["Bengaluru", "35%", "70%"], ["Kolkata", "69%", "49%"],
  ];
  document.querySelectorAll("#forecast-views .comparison").forEach((figure) => {
    const map = figure.querySelector(".image-link");
    if (!map) return;
    map.classList.add("map-canvas");
    figure.classList.add("map-figure");
    const controls = document.createElement("div");
    controls.className = "map-tools";
    controls.innerHTML = '<button type="button" data-map-zoom="in" aria-label="Enlarge map">+</button><button type="button" data-map-zoom="out" aria-label="Reduce map">−</button><button type="button" data-map-zoom="reset" aria-label="Reset map zoom">⌂</button>';
    figure.append(controls);
    const pins = document.createElement("div");
    pins.className = "map-pins";
    pinLocations.forEach(([city, left, top]) => {
      const pin = document.createElement("button");
      pin.type = "button"; pin.className = "map-pin"; pin.dataset.mapCity = city;
      pin.style.left = left; pin.style.top = top; pin.textContent = city;
      pins.append(pin);
    });
    figure.append(pins);
    let zoom = 1;
    controls.addEventListener("click", (event) => {
      const action = event.target.closest("button")?.dataset.mapZoom;
      if (!action) return;
      zoom = action === "in" ? Math.min(1.7, zoom + .15) : action === "out" ? Math.max(1, zoom - .15) : 1;
      map.querySelector("img").style.transform = `scale(${zoom})`;
    });
  });
  document.querySelectorAll("[data-map-city]").forEach((pin) => pin.addEventListener("click", () => {
    validationCity = pin.dataset.mapCity;
    renderValidation();
    document.querySelector("#validation").scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  render(false);
  renderValidation(false);
  renderCombination(false);
})();
