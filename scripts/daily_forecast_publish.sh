#!/usr/bin/env bash
# Daily 14:00 IST publisher.  Install with the crontab line documented below.
set -euo pipefail

SITE_REPO="/home/saptarishi.dhanuka_asp25/weather/misc-coding.github.io"
PYTHON="/Datastorage/saptarishi.dhanuka_asp25/conda_envs/realtime_dash/bin/python"
LOCK="/tmp/india_forecast_pages.lock"
LOG_DIR="/tmp/india_forecast_pages_logs"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) another forecast publisher is already running" >>"$LOG_DIR/publish_$(date -u +%Y%m%d).log"
  exit 0
fi

export MPLCONFIGDIR="/tmp/india_forecast_pages_mpl"
mkdir -p "$MPLCONFIGDIR"
LOG="$LOG_DIR/publish_$(date -u +%Y%m%d).log"
{
  echo "=== $(date -u +%FT%TZ) publish start ==="
  cd "$SITE_REPO"
  git pull --ff-only origin main
  PUBLISH_ARGS=(--output-site "$SITE_REPO")
  if [[ "${FORECAST_BACKFILL:-0}" == "1" ]]; then
    PUBLISH_ARGS+=(--backfill)
  fi
  "$PYTHON" scripts/publish_forecast_archive.py "${PUBLISH_ARGS[@]}"
  if ! git diff --quiet -- index.html README.md assets scripts; then
    git add index.html README.md assets scripts
    git commit -m "Update India forecast archive"
    git push origin main
  fi
  echo "=== $(date -u +%FT%TZ) publish complete ==="
} >>"$LOG" 2>&1
