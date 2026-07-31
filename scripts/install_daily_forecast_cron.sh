#!/usr/bin/env bash
# Idempotently install the local 14:00 Asia/Kolkata forecast publisher schedule.
set -euo pipefail

JOB='CRON_TZ=Asia/Kolkata
0 14 * * * /home/saptarishi.dhanuka_asp25/weather/misc-coding.github.io/scripts/daily_forecast_publish.sh'
TMP_FILE=$(mktemp)
trap 'rm -f "$TMP_FILE"' EXIT
(crontab -l 2>/dev/null || true) | grep -v 'daily_forecast_publish.sh' >"$TMP_FILE"
printf '%s\n' "$JOB" >>"$TMP_FILE"
crontab "$TMP_FILE"
