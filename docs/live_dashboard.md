# Live run dashboard

`shield-das-live` serves a small web dashboard for the run that is *currently
being recorded*. It finds the newest run under `results/` with no
`end_time` in its metadata and a recently written `shield_data.csv`, tails
the CSV, and shows three stacked plots on a shared time axis:

- **Upstream pressure** (torr, log scale) — all gauges with
  `gauge_location: "upstream"`, converted with the existing per-gauge
  voltage-to-pressure functions.
- **Downstream pressure** (torr, log scale) — same for `"downstream"`.
- **Temperature** (°C) — thermocouple voltage with cold-junction
  compensation, when the run has a thermocouple.

A header bar shows the run id, a LIVE / WAITING / ENDED status badge, the
elapsed time, the row count, and a fidelity dropdown. When the current run
ends (its metadata gains an `end_time`, or the CSV stops being written), the
badge switches to ENDED and the dashboard automatically attaches to any
newer run that starts.

## Pulling it up when a run is live

```bash
shield-das-live --watch
```

`--watch` polls every 10 seconds until an active run appears, then opens the
dashboard in your browser. Without `--watch`, the command exits with a
message if nothing is live right now.

Other options:

```bash
shield-das-live --results-dir results   # where runs are recorded
shield-das-live --port 8051             # serve port (default 8051)
shield-das-live --host 0.0.0.0          # expose on the LAN (default 127.0.0.1)
shield-das-live --interval 2000         # refresh interval, ms
shield-das-live --max-points 2000       # default plot fidelity
shield-das-live --no-browser            # don't open a browser
shield-das-live --include-test-runs     # also monitor test_run_* dirs
```

`--include-test-runs` is handy for demoing the dashboard without hardware:
start a recorder with `run_type="test_mode"` and the dashboard will pick up
the resulting `test_run_*` directory.

## Keeping the rig armed for remote viewing

The always-on logon task on the rig PC now runs the Supabase publisher, not
this dashboard — remote viewers use the GitHub Pages site instead of
connecting to the rig (see [live_supabase.md](live_supabase.md) for the task
definition and the rest of that setup). If an old
`"SHIELD live dashboard"` logon task is still registered, remove it with
`schtasks /Delete /TN "SHIELD live dashboard" /F`.

`shield-das-live` remains the on-rig console: run it on demand next to the
recorder for a local, offline-capable view. It binds `127.0.0.1` by default;
pass `--host 0.0.0.0` only if you deliberately want it reachable from the
local network.

## Why it stays cheap

Two things keep the dashboard light no matter how long a run gets:

- **Incremental reads** — the CSV is read once at attach time; every later
  refresh seeks to the stored byte offset and reads only newly appended
  lines (a half-written trailing line is buffered until it completes). The
  file is never re-parsed from the top, unlike the main `DataPlotter` UI.
- **Capped point budget** — each trace is decimated by stride to the
  fidelity dropdown's limit (500 / 2000 / 5000 points), always keeping the
  most recent sample. Browser payload per refresh is therefore constant
  regardless of run size.
