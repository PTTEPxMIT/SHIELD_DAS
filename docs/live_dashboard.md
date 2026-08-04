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
shield-das-live --host 127.0.0.1        # local-only (default 0.0.0.0)
shield-das-live --interval 2000         # refresh interval, ms
shield-das-live --max-points 2000       # default plot fidelity
shield-das-live --no-browser            # don't open a browser
shield-das-live --include-test-runs     # also monitor test_run_* dirs
```

`--include-test-runs` is handy for demoing the dashboard without hardware:
start a recorder with `run_type="test_mode"` and the dashboard will pick up
the resulting `test_run_*` directory.

## Keeping it always armed on the rig PC

Register a logon task on the rig PC so the dashboard is waiting whenever a
run starts (no elevation needed for a per-user logon task):

```bat
schtasks /Create /TN "SHIELD live dashboard" /SC ONLOGON ^
    /TR "C:\path\to\python-env\Scripts\shield-das-live.exe --watch --no-browser" /F
```

Point `/TR` at the `shield-das-live` executable inside the Python
environment where SHIELD_DAS is installed, and add
`--results-dir C:\path\to\results` if the task does not start in the
recorder's working directory. Check it with `schtasks /Query /TN
"SHIELD live dashboard"` and remove it with `schtasks /Delete /TN
"SHIELD live dashboard" /F`.

## Remote viewing

By default the server binds `0.0.0.0`, so the dashboard is reachable from
other machines on the LAN or tailnet at `http://<rig-hostname>:8051` — open
the same URL from your laptop or phone via Tailscale or the lab network. On
a slow link, drop the fidelity dropdown to 500 points to shrink each
refresh. Use `--host 127.0.0.1` if you want the dashboard local-only.

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
