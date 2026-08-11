# Uploading completed runs to SHIELD-Data

`shield-das-upload` is an "outbox sweeper" that is run by hand on the rig
computer whenever there are finished runs to upload — after a run ends,
double-click `upload_runs.bat` in the repo root (or run the command in a
terminal). Each sweep it:

1. Scans the local `results/` directory for **completed** runs — runs that are
   not test-mode (`test_run_*`), have either a `run_info.end_time` in
   `run_metadata.json` or a `shield_data.csv` untouched for at least
   `min_age_minutes`, and lasted at least `min_duration_minutes` (first to
   last CSV timestamp).
2. Normalises each run into the SHIELD-Data layout: a staged copy named
   `YY.MM.DD_run_N_HHhMM/` with `shield_data.csv` renamed to
   `pressure_gauge_data.csv`, `run_metadata.json` alongside it, and the
   `backup/` directory excluded. Both old (`MM.DD`) and new (`YY.MM.DD`)
   local date-directory formats are handled; the year for old-style
   directories comes from `run_info.date`.
3. Opens a pull request on
   [PTTEPxMIT/SHIELD-Data](https://github.com/PTTEPxMIT/SHIELD-Data) adding
   the run under `run_data/`, on a branch named `auto/run-<run_key>`.

Runs that are still *in progress* are not the uploader's job — watch those
live with `shield-das-live` instead (see
[live_dashboard.md](live_dashboard.md)); once the run ends, the next sweep
picks it up.

A ledger at `results/.upload_ledger.json` records the sha256 of each uploaded
CSV, so re-running the sweep uploads nothing new. If a run's CSV content ever
changes, the changed hash triggers a re-upload that force-updates the same
branch (a superseding PR).

The uploader is stdlib-only; it needs `git` on the PATH and a GitHub token.

## Creating the GitHub token

Use a **fine-grained personal access token** so the rig can touch only
SHIELD-Data:

1. GitHub → Settings → Developer settings → Fine-grained tokens →
   *Generate new token*.
2. **Resource owner**: `PTTEPxMIT`. **Repository access**: *Only select
   repositories* → `SHIELD-Data`.
3. **Repository permissions**: `Contents` → *Read and write*, and
   `Pull requests` → *Read and write*. Nothing else.
4. Set an expiry (maximum is about one year). When the token expires the
   sweep fails with authentication errors until a new token is configured —
   generate a fresh token the same way and update wherever it is stored.

Provide the token to the uploader either via the environment variable
`SHIELD_UPLOAD_TOKEN` (preferred) or the `token` key in the config file. The
uploader never prints the token and strips it from any logged git output.

## Config file

Default location: `~/.shield_das_uploader.json` (override with `--config`).

```json
{
  "results_dir": "C:/SHIELD/results",
  "repo": "PTTEPxMIT/SHIELD-Data",
  "staging_dir": "C:/SHIELD/results/.upload_staging",
  "min_age_minutes": 30,
  "min_duration_minutes": 5
}
```

All keys are optional; `staging_dir` defaults to
`<results_dir>/.upload_staging`.

## Running it

On the rig PC, double-click **`upload_runs.bat`** in the repo root after a run
finishes. It runs a sweep, prints one PR URL per uploaded run, and keeps the
window open so any errors can be read. The script looks for
`shield-das-upload` in the repo's `.venv` first, then on the PATH.

From a terminal, the equivalent commands are:

```bash
shield-das-upload --dry-run   # show what would be uploaded, no network
shield-das-upload             # sweep and open PRs
shield-das-upload --config C:/SHIELD/uploader.json
```

Set `SHIELD_UPLOAD_TOKEN` as a *user* environment variable (System Properties
→ Environment Variables) so the double-clicked script can see it, or put the
token in the config file.

Running the sweep more often than needed is harmless: it is idempotent, so a
sweep with nothing new to upload does nothing and exits.

## How this interacts with SHIELD-Data CI

Each upload is a normal pull request against SHIELD-Data's protected `main`,
so SHIELD-Data's CI (validation of `run_metadata.json`, CSV checks, database
ingestion) runs on the uploaded run before a human merges it. Nothing lands
in `main` without review. If a run is re-uploaded after its CSV changed, the
same `auto/run-<run_key>` branch is force-updated, so the existing PR simply
re-runs CI on the new content rather than opening a duplicate.
