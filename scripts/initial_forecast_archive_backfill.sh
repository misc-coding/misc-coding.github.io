#!/usr/bin/env bash
# One-shot wrapper for the initial seven-run archive. It removes its own cron line
# only after the locked publisher has completed successfully.
set -euo pipefail

REPO="/home/saptarishi.dhanuka_asp25/weather/misc-coding.github.io"
FORECAST_BACKFILL=1 "$REPO/scripts/daily_forecast_publish.sh"

TEMP_CRONTAB=$(mktemp)
trap 'rm -f "$TEMP_CRONTAB"' EXIT
(crontab -l 2>/dev/null || true) | grep -v 'initial_forecast_archive_backfill.sh' >"$TEMP_CRONTAB"
crontab "$TEMP_CRONTAB"
