import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { JSDOM, VirtualConsole } from "jsdom";


const ROOT = path.resolve(import.meta.dirname, "..");

function canvasContext(stats) {
  return {
    beginPath() {}, arc() {}, fill() {}, stroke() { stats.strokes += 1; }, fillRect() {}, fillText() {},
    moveTo() {}, lineTo() {}, save() {}, restore() {}, translate() {}, scale() {},
    drawImage() {}, putImageData() {},
    createImageData(width, height) {
      return { width, height, data: new Uint8ClampedArray(width * height * 4) };
    },
    set fillStyle(_) {}, set strokeStyle(_) {}, set lineWidth(_) {}, set font(_) {},
    set imageSmoothingEnabled(_) {},
  };
}

async function loadSite() {
  const [html, javascript] = await Promise.all([
    fs.readFile(path.join(ROOT, "index.html"), "utf8"),
    fs.readFile(path.join(ROOT, "assets/app.js"), "utf8"),
  ]);
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (error) => errors.push(error));
  virtualConsole.on("error", (error) => errors.push(error));
  const dom = new JSDOM(html, {
    url: "https://example.test/?tab=weather",
    runScripts: "outside-only",
    pretendToBeVisual: true,
    virtualConsole,
  });
  const { window } = dom;
  const stats = { strokes: 0, fetches: [] };
  window.Response = globalThis.Response;
  window.DecompressionStream = globalThis.DecompressionStream;
  Object.defineProperty(window, "devicePixelRatio", { value: 1 });
  window.HTMLCanvasElement.prototype.getContext = () => canvasContext(stats);
  window.HTMLCanvasElement.prototype.getBoundingClientRect = () => ({
    left: 0, top: 0, right: 900, bottom: 520, width: 900, height: 520,
  });
  window.HTMLCanvasElement.prototype.setPointerCapture = () => {};
  window.fetch = async (url) => {
    const relative = String(url).replace(/^\//, "");
    stats.fetches.push(relative);
    try {
      const buffer = await fs.readFile(path.join(ROOT, relative));
      const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => arrayBuffer,
        json: async () => JSON.parse(buffer.toString("utf8")),
      };
    } catch {
      return {
        ok: false,
        status: 404,
        arrayBuffer: async () => new ArrayBuffer(0),
        json: async () => ({}),
      };
    }
  };
  window.eval(javascript);
  await new Promise((resolve) => setTimeout(resolve, 50));
  return { dom, window, document: window.document, errors, stats };
}

test("all dashboard controls update their panels without runtime errors", async () => {
  const { dom, window, document, errors, stats } = await loadSite();
  assert.equal(document.querySelectorAll("#forecast-canvas").length, 1);
  assert.equal(document.querySelector('[data-panel="weather"]').hidden, false);
  assert.equal(document.querySelectorAll(".day-card").length, 5);
  assert.match(document.querySelector("#blend-note").textContent, /24-hour accumulation/);
  assert.ok(document.querySelectorAll("#city-grid-map .city-grid-point").length > 0);
  assert.match(document.querySelector("#city-grid-time").textContent, /IST.*UTC.*IST.*UTC/);
  assert.match(document.querySelector("#city-grid-result").textContent, /simple average of shown inputs/);
  assert.match(document.querySelector("#city-grid-samples").textContent, /Exact native sample times used/);
  assert.ok(document.querySelectorAll("#city-grid-map .forecast-grid-cell").length >= 9);
  assert.match(document.querySelector("#city-grid-model-note").textContent, /loaded grid.*cells shown.*valid.*IST.*UTC/);
  assert.match(document.querySelector("#weather-chart .date.time").textContent, /IST.*UTC/);
  assert.ok(document.querySelectorAll("#within-day-models button").length > 0);
  assert.ok(document.querySelector("#within-day-chart svg"));
  assert.match(document.querySelector("#within-day-note").textContent, /exact interval/);
  const gfsCityGrid = document.querySelector('[data-city-grid-model="gfs"]');
  if (gfsCityGrid) {
    gfsCityGrid.click();
    assert.equal(document.querySelector('[data-city-grid-model="gfs"]').getAttribute("aria-pressed"), "true");
    assert.match(document.querySelector("#city-grid-model-note").textContent, /GFS loaded grid/);
  }

  document.querySelector('[data-weather-day="3"]').click();
  assert.equal(document.querySelector('[data-weather-day="3"]').getAttribute("aria-pressed"), "true");
  assert.match(window.location.search, /weather_day=3/);
  assert.ok(document.querySelector("#within-day-chart svg"));

  document.querySelector('[data-weather-variable="precipitation"]').click();
  assert.equal(document.querySelector('[data-weather-variable="precipitation"]').getAttribute("aria-pressed"), "true");
  assert.match(document.querySelector("#weather-chart svg").getAttribute("aria-label"), /precipitation/);
  assert.match(document.querySelector("#city-grid-result").textContent, /mm in 24 h/);

  const city = document.querySelector("#city-select");
  city.value = "Mumbai";
  city.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.equal(document.querySelector("#weather-location").textContent, "Mumbai");

  document.querySelector('[data-tab="maps"]').click();
  await new Promise((resolve) => setTimeout(resolve, 300));
  assert.equal(document.querySelector('[data-panel="maps"]').hidden, false);
  assert.ok(stats.fetches.includes("assets/coastlines.json"));
  assert.ok(stats.strokes > 20, "coastline segments should be drawn over the field");
  const canvas = document.querySelector("#forecast-canvas");
  assert.ok(canvas.height >= 774, "the India map should use the taller aspect ratio");
  assert.ok(document.querySelector("#temporal-init-select").options.length === 3);
  assert.ok(document.querySelector("#temporal-model-select").options.length > 0);
  assert.ok(document.querySelector("#temporal-time-select").options.length > 0);
  assert.ok(document.querySelector("#temporal-forecast-canvas")._standaloneMap);
  assert.match(document.querySelector("#temporal-map-note").textContent, /complete native half-hours|highest available/);
  assert.match(document.querySelector("#map-legend-title").textContent, /fixed scale/);
  const simpleAverage = document.querySelector('[data-map-model="simple_average"]');
  assert.equal(simpleAverage.disabled, false);
  simpleAverage.click();
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.match(document.querySelector("#map-description").textContent, /equal weight for each available model at this grid cell/);
  assert.match(document.querySelector("#map-animation").getAttribute("src"), /\/simple_average\/temperature\.gif$/);
  const combined = document.querySelector('[data-map-model="combined"]');
  assert.equal(combined.disabled, false);
  combined.click();
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(combined.getAttribute("aria-pressed"), "true");
  assert.match(document.querySelector("#map-description").textContent, /prior matched samples/);
  assert.match(document.querySelector("#map-animation").getAttribute("src"), /\/combined\/temperature\.gif$/);
  assert.deepEqual([...document.querySelectorAll("#map-legend-ticks span")].map((item) => item.textContent), ["0", "15", "30", "45"]);
  canvas.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 450, clientY: 260, bubbles: true }));
  const tooltip = document.querySelector("#map-tooltip");
  assert.equal(tooltip.hidden, false);
  assert.match(tooltip.textContent, /°C/);
  assert.match(tooltip.textContent, /° N.*° E/);
  canvas.dispatchEvent(new window.MouseEvent("pointerleave", { bubbles: true }));
  assert.equal(tooltip.hidden, true);

  document.querySelector('[data-map-variable="precipitation"]').click();
  document.querySelector('[data-map-day="5"]').click();
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(document.querySelector('[data-map-day="5"]').getAttribute("aria-pressed"), "true");
  assert.match(document.querySelector("#map-title").textContent, /Interval rainfall · .*2026.*IST.*UTC/);
  assert.doesNotMatch(document.querySelector("#map-title").textContent, /Day [135]|T\+/);
  assert.match(document.querySelector("#map-description").textContent, /IST.*UTC → .*IST.*UTC \(48 h\) accumulation/);
  assert.match(document.querySelector("#map-legend-title").textContent, /Interval rainfall/);
  assert.match(document.querySelector("#map-legend-note").textContent, /IST.*UTC → .*IST.*UTC/);
  assert.match(document.querySelector("#map-animation").getAttribute("src"), /precipitation\.gif$/);
  assert.match(document.querySelector("#animation-description").textContent, /exact windows:.*IST.*UTC/);
  canvas.dispatchEvent(new window.MouseEvent("pointermove", { clientX: 450, clientY: 260, bubbles: true }));
  assert.match(tooltip.textContent, /mm/);
  assert.match(tooltip.textContent, /IST.*UTC → .*IST.*UTC.*accumulation/);
  const gfs = document.querySelector('[data-map-model="gfs"]');
  if (!gfs.disabled) {
    gfs.click();
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.equal(gfs.getAttribute("aria-pressed"), "true");
    assert.match(document.querySelector("#map-animation").getAttribute("src"), /\/gfs\/precipitation\.gif$/);
  }
  document.querySelector("#map-reset").click();

  document.querySelector('[data-tab="validation"]').click();
  await new Promise((resolve) => setTimeout(resolve, 300));
  document.querySelector('[data-validation-variable="precipitation"]').click();
  assert.match(document.querySelector("#validation-image").getAttribute("src"), /precipitation\.png/);
  assert.match(document.querySelector("#validation-summary").textContent, /combined mean endpoint MAE/);
  document.querySelector('[data-match-variable="temperature"]').click();
  assert.match(document.querySelector("#match-image").getAttribute("src"), /temperature\.png/);
  assert.equal(document.querySelector("#imerg-time-select").options.length, 144);
  assert.ok(document.querySelector("#imerg-early-canvas")._standaloneMap);
  assert.ok(document.querySelector("#imerg-late-canvas")._standaloneMap);
  assert.match(document.querySelector("#imerg-map-note").textContent, /native 0\.1° grid/i);
  assert.match(document.querySelector("#imerg-validation-image").getAttribute("src"), /imerg\/city_validation/);
  assert.match(document.querySelector("#imerg-validation-summary").textContent, /bias-corrected RMSE/);
  document.querySelector('[data-imerg-duration="6h"]').click();
  await new Promise((resolve) => setTimeout(resolve, 150));
  assert.ok(document.querySelector("#imerg-time-select").options.length >= 10);
  assert.match(document.querySelector("#imerg-map-note").textContent, /six-hour/);

  const init = document.querySelector("#init-select");
  if (init.options.length > 1) {
    init.value = init.options[1].value;
    init.dispatchEvent(new window.Event("change", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.match(document.querySelector("#run-status").textContent, /models/);
  }

  assert.deepEqual(errors, []);
  dom.window.close();
});

test("shareable URL state selects weather, maps, and validation values", async () => {
  const { dom, document } = await loadSite();
  document.querySelector('[data-tab="method"]').click();
  assert.equal(document.querySelector('[data-panel="method"]').hidden, false);
  assert.match(dom.window.location.search, /tab=method/);
  assert.match(dom.window.location.search, /city=/);
  assert.match(dom.window.location.search, /init=/);
  dom.window.close();
});
