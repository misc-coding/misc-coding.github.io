# India Multi-Model Forecast Atlas

The site publishes a rolling seven-run archive of 00 UTC India-region forecast
maps from WeatherNext 2, GenCast, GFS, GEFS, AIFS, and IFS-ENS.

## Local automation

The publisher must run on the local workstation with private GCS credentials and
the `realtime_dash` conda environment. It validates all 42 maps in a run before
replacing the site and retains the latest seven complete runs.

The preferred scheduler is the checked-in user systemd timer:

```bash
systemctl --user link "$PWD/systemd/india-forecast-pages.service"
systemctl --user link "$PWD/systemd/india-forecast-pages.timer"
systemctl --user daemon-reload
systemctl --user enable --now india-forecast-pages.timer
```

It refreshes daily at 14:00 Asia/Kolkata. Run the initial backfill once with:

```bash
FORECAST_BACKFILL=1 ./scripts/daily_forecast_publish.sh
```

The legacy cron installer remains available where local PAM policy allows
`crontab` access.
