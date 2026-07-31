#!/usr/bin/env python3
"""Build a rolling, static archive for the India multi-model forecast site.

This command deliberately runs on the workstation that has access to the private
WeatherNext archives.  It creates a complete temporary site and replaces the
published files only after every selected run has passed validation.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REALTIME_ROOT = Path("/home/saptarishi.dhanuka_asp25/weather/real_time")
DEFAULT_PYTHON = Path("/Datastorage/saptarishi.dhanuka_asp25/conda_envs/realtime_dash/bin/python")


def stamp(init: pd.Timestamp) -> str:
    return pd.Timestamp(init).strftime("%Y%m%d_%H")


def utc_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_renderer(realtime_root: Path):
    """Import the maintained model adapters and cartopy renderer without copying them."""
    scripts = realtime_root / "scripts"
    source = realtime_root / "src"
    if not (scripts / "publish_forecast_site.py").is_file() or not source.is_dir():
        raise RuntimeError(f"real-time project is incomplete: {realtime_root}")
    sys.path[:0] = [str(scripts), str(source)]
    import publish_forecast_site as renderer  # type: ignore
    from realtime_dash.config import load_config  # type: ignore
    from realtime_dash.india import load as india_load  # type: ignore
    return renderer, load_config, india_load


def common_midnight_inits(models, cfg, india_load) -> list[pd.Timestamp]:
    """Return available 00 UTC initializations shared by every requested model."""
    sets = []
    for model in models:
        values = {pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)
                  for value in india_load.available_inits(model, cfg)}
        sets.append(values)
    if not sets:
        return []
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return sorted(
        (value for value in set.intersection(*sets) if value.hour == 0 and value <= now),
        reverse=True,
    )


def read_archive(site: Path) -> dict:
    path = site / "assets" / "forecast_archive.json"
    if not path.is_file():
        return {"schema_version": 1, "runs": []}
    archive = json.loads(path.read_text())
    if archive.get("schema_version") != 1 or not isinstance(archive.get("runs"), list):
        raise RuntimeError(f"unsupported archive manifest: {path}")
    return archive


def valid_existing_runs(site: Path, archive: dict, renderer) -> list[dict]:
    """Keep only archive records whose entire asset set is present and valid."""
    runs = []
    for run in archive["runs"]:
        try:
            init = pd.Timestamp(run["initialization_utc"])
            if stamp(init) != run["id"] or len(run["artifacts"]) != 42:
                continue
            for artifact in run["artifacts"]:
                renderer.validate_png(site / artifact["path"])
            runs.append(run)
        except (KeyError, OSError, ValueError):
            continue
    return runs


def render_run(init, models, cfg, renderer, stage: Path, attempts: int) -> dict:
    datasets = {}
    for model in models:
        print(f"[{stamp(init)}] loading {renderer.MODEL_META[model]['label']}", flush=True)
        datasets[model] = renderer.load_with_retries(
            model, cfg, init, max_members=8, attempts=attempts,
        )
    artifacts = renderer.render_all_maps(datasets, models, init, cfg, stage)
    manifest = renderer.build_manifest(datasets, models, init, cfg, artifacts)
    return {
        "id": stamp(init),
        "initialization_utc": manifest["initialization_utc"],
        "generated_at_utc": manifest["generated_at_utc"],
        "lead_days": manifest["lead_days"],
        "models": manifest["models"],
        "variables": manifest["variables"],
        "lead_semantics": manifest["lead_semantics"],
        "bounding_box": manifest["bounding_box"],
        "disclaimer": manifest["disclaimer"],
        "artifacts": artifacts,
    }


def archive_manifest(runs: list[dict]) -> dict:
    runs = sorted(runs, key=lambda run: run["initialization_utc"], reverse=True)
    return {
        "schema_version": 1,
        "title": "India Multi-Model Forecast Archive",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "retention_runs": len(runs),
        "latest_initialization_utc": runs[0]["initialization_utc"],
        "runs": runs,
    }


def run_sections(run: dict, renderer) -> str:
    manifest = dict(run)
    html = renderer._view_sections(manifest)
    return re.sub(
        r'<section class="forecast-view"',
        f'<section class="forecast-view" data-init="{run["id"]}"',
        html,
    )


ARCHIVE_JS = r"""
(() => {
  const variableButtons = [...document.querySelectorAll("[data-variable-button]")];
  const dayButtons = [...document.querySelectorAll("[data-day-button]")];
  const runSelect = document.querySelector("#run-select");
  const runSummary = document.querySelector("#run-summary");
  const views = [...document.querySelectorAll(".forecast-view")];
  const runs = JSON.parse(document.querySelector("#archive-data").textContent).runs;
  const params = new URLSearchParams(window.location.search);
  const allowedVariables = new Set(["temperature", "precipitation"]);
  const allowedDays = new Set(["1", "2", "3"]);
  const allowedInits = new Set(runs.map((run) => run.id));
  let variable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let day = allowedDays.has(params.get("day")) ? params.get("day") : "1";
  let init = allowedInits.has(params.get("init")) ? params.get("init") : runs[0].id;

  function render(updateUrl = true) {
    variableButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.variableButton === variable)));
    dayButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.dayButton === day)));
    views.forEach((view) => { view.hidden = !(view.dataset.init === init && view.dataset.variable === variable && view.dataset.day === day); });
    const active = runs.find((run) => run.id === init);
    runSelect.value = init;
    runSummary.textContent = `Initialized ${new Date(active.initialization_utc).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false })} UTC · 6 models · 3-day forecast`;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("init", init);
      next.searchParams.set("variable", variable);
      next.searchParams.set("day", day);
      history.replaceState(null, "", next);
    }
  }

  variableButtons.forEach((button) => button.addEventListener("click", () => { variable = button.dataset.variableButton; render(); }));
  dayButtons.forEach((button) => button.addEventListener("click", () => { day = button.dataset.dayButton; render(); }));
  runSelect.addEventListener("change", () => { init = runSelect.value; render(); });
  render(false);
})();
"""


def build_html(archive: dict, renderer) -> str:
    latest = archive["runs"][0]
    options = "".join(
        f'<option value="{run["id"]}">{pd.Timestamp(run["initialization_utc"]):%d %b %Y · 00 UTC}</option>'
        for run in archive["runs"]
    )
    sections = "\n".join(run_sections(run, renderer) for run in archive["runs"])
    source_rows = "".join(
        "<tr><th scope=\"row\">{label}</th><td>{provider}</td><td>{members}</td>"
        "<td><a href=\"{url}\">Source details</a></td></tr>".format(
            label=model["label"],
            provider=model["provider"],
            members=(
                "Deterministic"
                if model["members_total"] == 1
                else f"{model['members_used']} / {model['members_total']} members"
            ),
            url=model["source_url"],
        )
        for model in latest["models"]
    )
    data = json.dumps({"runs": [{"id": run["id"], "initialization_utc": run["initialization_utc"]} for run in archive["runs"]]})
    return f'''<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Rolling India-region temperature and precipitation forecasts from six global models.">
  <title>India Multi-Model Forecast Atlas</title><link rel="stylesheet" href="assets/style.css">
  <script defer src="assets/app.js"></script>
</head><body>
  <header class="masthead"><div class="shell nav-shell"><a class="brand" href="#top">FORECAST / INDIA</a><nav aria-label="Primary navigation"><a href="#maps">Maps</a><a href="#method">Method</a><a href="#sources">Sources</a></nav></div></header>
  <main id="top"><section class="hero"><div class="shell hero-grid"><div><p class="eyebrow">Six global models · rolling seven-run archive</p><h1>India forecast atlas</h1><p class="lede">Temperature snapshots and cumulative rainfall from WeatherNext 2, GenCast, GFS, GEFS, AIFS, and IFS-ENS—aligned to the same initialization, forecast leads, map extent, units, and color scales.</p><div class="hero-actions"><a class="primary-action" href="#maps">Explore the maps</a><a class="text-action" href="assets/forecast_archive.json">View archive provenance</a></div></div><dl class="run-card"><div><dt>Initialization</dt><dd id="run-summary">Loading archive…</dd></div><div><dt>Archive run</dt><dd><label class="sr-only" for="run-select">Choose forecast initialization</label><select id="run-select">{options}</select></dd></div><div><dt>Forecast leads</dt><dd>T+24 · T+48 · T+72 hours</dd></div><div><dt>Products</dt><dd>42 maps per initialization</dd></div></dl></div></section>
  <section class="run-strip" aria-label="Forecast summary"><div class="shell stats"><div><strong>6</strong><span>forecast models</span></div><div><strong>7</strong><span>retained runs</span></div><div><strong>3</strong><span>forecast days</span></div><div><strong>294</strong><span>archived PNG products</span></div></div></section>
  <section class="maps shell" id="maps"><div class="intro-row"><div><p class="kicker">Forecast gallery</p><h2>Compare the same atmosphere, six ways.</h2></div><p>Select an initialization, variable, and lead. Every comparison sheet and individual map uses a common scale within its selected variable.</p></div><div class="controls" aria-label="Forecast map controls"><fieldset><legend>Variable</legend><div class="segmented"><button type="button" data-variable-button="temperature" aria-pressed="true">Temperature</button><button type="button" data-variable-button="precipitation" aria-pressed="false">Precipitation</button></div></fieldset><fieldset><legend>Forecast lead</legend><div class="segmented"><button type="button" data-day-button="1" aria-pressed="true">Day 1 · +24h</button><button type="button" data-day-button="2" aria-pressed="false">Day 2 · +48h</button><button type="button" data-day-button="3" aria-pressed="false">Day 3 · +72h</button></div></fieldset></div><div id="forecast-views" aria-live="polite">{sections}</div></section>
  <section class="method-band" id="method"><div class="shell"><div class="intro-row light"><div><p class="kicker">Method</p><h2>A clean, comparable forecast slice.</h2></div><p>Only common 00 UTC cycles with all required models and target leads are published. Missing data stops publication; a previous run is never silently substituted.</p></div><div class="method-grid"><article><span>01</span><h3>Align</h3><p>All sources are cropped to the same India-region bounding box and exact lead endpoints.</p></article><article><span>02</span><h3>Normalize</h3><p>Temperature is converted to °C. Precipitation becomes millimetres accumulated since initialization.</p></article><article><span>03</span><h3>Reduce</h3><p>Ensemble products are shown as means. Private GCS sources use eight evenly spaced members; tiled sources use all members.</p></article><article><span>04</span><h3>Validate</h3><p>Each map and every linked artifact is verified before a run can enter the archive.</p></article></div></div></section>
  <section class="sources shell" id="sources"><div class="intro-row"><div><p class="kicker">Data provenance</p><h2>Source by source.</h2></div><p>WeatherNext products are read from private GCS Zarr archives. NOAA and ECMWF products are read from dynamical.org’s analysis-ready Icechunk archives.</p></div><div class="table-wrap"><table><thead><tr><th>Model</th><th>Archive</th><th>Map reduction</th><th>Documentation</th></tr></thead><tbody>{source_rows}</tbody></table></div><aside class="notice"><strong>Experimental guidance.</strong><p>These maps are for visualization and research. They are not official forecasts, warnings, or public-safety products. Consult the India Meteorological Department and relevant authorities for operational guidance.</p></aside></section></main>
  <footer><div class="shell footer-row"><p>India Multi-Model Forecast Atlas · rolling seven-run archive</p><a href="#top">Back to top ↑</a></div></footer><script id="archive-data" type="application/json">{data}</script></body></html>\n'''


ARCHIVE_CSS = r"""
.run-card select { width: 100%; border: 1px solid rgba(255,255,255,.3); border-radius: 2px; padding: 8px; color: #eef7f5; background: #0d3a48; font: inherit; font-size: .82rem; }
.run-card label { display: block; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
"""


def write_stage(stage: Path, archive: dict, renderer) -> None:
    assets = stage / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "style.css").write_text(renderer.CSS.strip() + "\n" + ARCHIVE_CSS.strip() + "\n")
    (assets / "app.js").write_text(ARCHIVE_JS.strip() + "\n")
    (assets / "forecast_archive.json").write_text(json.dumps(archive, indent=2) + "\n")
    (assets / "forecast_manifest.json").write_text(json.dumps(archive["runs"][0], indent=2) + "\n")
    (stage / "index.html").write_text(build_html(archive, renderer))
    (stage / "README.md").write_text(
        "# India Multi-Model Forecast Atlas\n\n"
        "A rolling seven-initialization static forecast archive. Each retained run has six "
        "models, two variables, three forecast days, and 42 validated PNG products.\n\n"
        "See [`assets/forecast_archive.json`](assets/forecast_archive.json) for provenance.\n"
    )


def validate_stage(stage: Path, archive: dict, renderer) -> None:
    if len(archive["runs"]) != 7:
        raise RuntimeError(f"expected seven retained runs, found {len(archive['runs'])}")
    html = (stage / "index.html").read_text()
    seen = set()
    for run in archive["runs"]:
        if run["id"] in seen or len(run["artifacts"]) != 42:
            raise RuntimeError(f"invalid artifact record for run {run.get('id')}")
        seen.add(run["id"])
        for artifact in run["artifacts"]:
            path = stage / artifact["path"]
            renderer.validate_png(path)
            if artifact["path"] not in html:
                raise RuntimeError(f"unlinked artifact: {artifact['path']}")
    for relative in ("assets/style.css", "assets/app.js", "assets/forecast_archive.json", "assets/forecast_manifest.json"):
        if not (stage / relative).is_file():
            raise RuntimeError(f"missing staged asset: {relative}")


def publish_stage(stage: Path, output_site: Path) -> None:
    if not (output_site / ".git").is_dir():
        raise RuntimeError(f"not a Git Pages repository: {output_site}")
    target_assets = output_site / "assets"
    backup = output_site / ".assets.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if target_assets.exists():
        target_assets.rename(backup)
    try:
        shutil.copytree(stage / "assets", target_assets)
        shutil.copy2(stage / "index.html", output_site / "index.html")
        shutil.copy2(stage / "README.md", output_site / "README.md")
    except Exception:
        if target_assets.exists():
            shutil.rmtree(target_assets)
        if backup.exists():
            backup.rename(target_assets)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-site", type=Path, default=SITE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--history-runs", type=int, default=7)
    parser.add_argument("--backfill", action="store_true", help="fill the archive to the requested retention")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.history_runs != 7:
        raise SystemExit("this public archive is intentionally fixed at seven retained runs")
    renderer, load_config, india_load = load_renderer(args.realtime_root.resolve())
    cfg = load_config()
    models = tuple(renderer.DEFAULT_MODELS)
    existing = valid_existing_runs(args.output_site, read_archive(args.output_site), renderer)
    available = common_midnight_inits(models, cfg, india_load)
    if not available:
        raise RuntimeError("no common 00 UTC initialization is currently available")
    wanted = 7 if args.backfill else 1
    target_ids = {run["id"] for run in existing}
    candidates = [init for init in available if stamp(init) not in target_ids]
    if args.backfill:
        candidates = candidates[: max(21, wanted * 3)]
    else:
        candidates = candidates[:3]

    with tempfile.TemporaryDirectory(prefix="forecast-archive-", dir="/tmp") as tmp:
        stage = Path(tmp)
        stage_forecasts = stage / "assets" / "forecasts"
        stage_forecasts.mkdir(parents=True)
        retained = existing[:7]
        for run in retained:
            source = args.output_site / "assets" / "forecasts" / run["id"]
            shutil.copytree(source, stage_forecasts / run["id"])
        for init in candidates:
            try:
                run = render_run(init, models, cfg, renderer, stage, args.attempts)
            except Exception as exc:  # noqa: BLE001 - keep last good archive intact
                print(f"[{stamp(init)}] rejected: {exc}", file=sys.stderr, flush=True)
                continue
            retained = [entry for entry in retained if entry["id"] != run["id"]] + [run]
            retained = sorted(retained, key=lambda entry: entry["initialization_utc"], reverse=True)[:7]
            if not args.backfill:
                break
            if args.backfill and len(retained) == 7:
                break
        if len(retained) != 7:
            raise RuntimeError(f"could not build a complete seven-run archive (have {len(retained)})")
        archive = archive_manifest(retained)
        write_stage(stage, archive, renderer)
        validate_stage(stage, archive, renderer)
        if args.dry_run:
            print("validated archive build; dry-run leaves the site unchanged")
        else:
            publish_stage(stage, args.output_site)
            print(f"published seven validated forecast runs to {args.output_site}")


if __name__ == "__main__":
    main()
