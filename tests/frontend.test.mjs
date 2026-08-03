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

  document.querySelector('[data-weather-variable="precipitation"]').click();
  assert.equal(document.querySelector('[data-weather-variable="precipitation"]').getAttribute("aria-pressed"), "true");
  assert.match(document.querySelector("#weather-chart svg").getAttribute("aria-label"), /precipitation/);

  const city = document.querySelector("#city-select");
  city.value = "Mumbai";
  city.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.equal(document.querySelector("#weather-location").textContent, "Mumbai");

  document.querySelector('[data-tab="maps"]').click();
  await new Promise((resolve) => setTimeout(resolve, 30));
  assert.equal(document.querySelector('[data-panel="maps"]').hidden, false);
  assert.ok(stats.fetches.includes("assets/coastlines.json"));
  assert.ok(stats.strokes > 20, "coastline segments should be drawn over the field");
  document.querySelector('[data-map-variable="precipitation"]').click();
  document.querySelector('[data-map-day="5"]').click();
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.match(document.querySelector("#map-title").textContent, /Accumulated rainfall · Day 5/);
  document.querySelector("#map-reset").click();

  document.querySelector('[data-tab="validation"]').click();
  document.querySelector('[data-validation-variable="precipitation"]').click();
  assert.match(document.querySelector("#validation-image").getAttribute("src"), /precipitation\.png/);
  document.querySelector('[data-match-variable="temperature"]').click();
  assert.match(document.querySelector("#match-image").getAttribute("src"), /temperature\.png/);

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
