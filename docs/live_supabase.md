# Live data over Supabase

The rig's remote live view is served through a free [Supabase](https://supabase.com)
project instead of a VPN. The pieces:

```
DataRecorder ──appends──▶ results/<date>/<run>/shield_data.csv      (unchanged)
shield-das-publish ──downsampled, torr/°C──▶ Supabase (Postgres + PostgREST)
GitHub Pages site ──read-only polling, anon key──▶ any browser, no VPN
```

Supabase is a **disposable live mirror**: the system of record remains the CSV
on the rig and the parquet run archive in SHIELD-Data. Anything in the mirror
can be deleted at any time without losing data. The mirror stores physical
units (pressure in torr, temperature in °C), not raw voltages.

`shield-das-live` (the local Dash dashboard) still works on the rig itself; it
just no longer binds the LAN.

## One-time setup

1. **Create the project.** Sign in at [supabase.com](https://supabase.com)
   (use an organisation account if available — free organisations are limited
   to 2 active projects, so check there is a slot). Create a new project on
   the **Free** plan in a region near the rig. Any strong database password
   is fine; nothing in this setup uses it directly.
2. **Record the keys.** In *Project Settings → API*, note:
   - the **Project URL** (`https://<ref>.supabase.co`),
   - the **`anon` public key** — read-only by design, safe to publish (it goes
     into the viewer site),
   - the **`service_role` secret key** — full write access, only ever stored
     on the rig PC (as the `SHIELD_SUPABASE_KEY` environment variable). Never
     commit it, never reuse it for anything else.
3. **Apply the schema.** Open *SQL Editor*, paste the entire contents of
   [`supabase/schema.sql`](../supabase/schema.sql), and run it. The script is
   idempotent — re-running it (e.g. after a schema update in this repo) is
   safe and is the intended upgrade path.
4. **Verify** with the queries in the next section.

## The 500 MB budget

The free tier caps the database at 500 MB. This deployment is designed so the
cap **cannot** be hit, and the guarantee lives in the database itself — not in
the publisher behaving well.

**Hard cap.** The `readings_cap` trigger (see `supabase/schema.sql`) runs in
the same transaction as every insert into `readings` and deletes any row with
`id <= max(id) - 250000`. Inserts are the only way the table grows, and `id`
is a monotone identity column, so the live row count can never exceed
250,000 plus one insert batch (≤ 500) — regardless of publisher bugs, crashes,
restarts, or anything else a client does.

**Size at the cap.** A reading row is `(id bigint, run_id bigint,
ts timestamptz, data jsonb)` with ~6 channels rounded to 6 significant digits:

| Component                                        | Size    |
| ------------------------------------------------ | ------- |
| Row: 28 B tuple overhead + 24 B fixed columns + ≤256 B jsonb | ≤ ~330 B heap |
| Index entries (primary key + `unique(run_id, ts)`) | ~70 B  |
| **All-in per row**                               | **≤ 400 B** |
| 250,000 rows × 400 B                             | ~100 MB |
| × 2 bloat allowance (deleted tuples awaiting autovacuum) | ~200 MB |
| `runs` table + fresh-project Postgres baseline   | ~70 MB  |
| **Worst case total**                             | **~270 MB (54 % of 500 MB)** |

The 256 B jsonb payload is far below the 2 KB TOAST threshold, so no
out-of-line storage appears, and Supabase's reported "Database size" counts
relations, not WAL — so the table above is the whole story. Even a 3× bloat
scenario stays under 400 MB.

**Time coverage.** 250 k rows at the publisher's 5 s cadence is 14.4 days at
full resolution. The publisher additionally thins data older than 48 h to a
60 s cadence (`thin_readings`), which stretches the cap to roughly 5 months of
continuous recording — a multi-week run keeps its entire span visible, with
the older portion at 1-minute resolution (finer than the plots can display
anyway). Thinning and run pruning are quality-of-service; the trigger alone is
the safety proof.

## Verification queries

Run these in the SQL editor after applying the schema.

Check a representative row size (publish something first, or insert a fake):

```sql
insert into public.runs (run_key, started_at)
values ('00.01.01_run_1_00h00', now());

insert into public.readings (run_id, ts, data)
select r.id, now(), '{"WGM701": 1.23456e-06, "CVM211": 0.00123456,
  "Baratron626D_1KT": 123.456, "Baratron626D_1T": 0.00123456,
  "TC1_C": 512.345, "local_C": 25.4}'::jsonb
from public.runs r where r.run_key = '00.01.01_run_1_00h00';

select pg_column_size(data) as jsonb_bytes from public.readings limit 1;
-- expect: well under 256
```

Prove the cap trigger clamps a bulk overrun:

```sql
insert into public.readings (run_id, ts, data)
select r.id, now() + (n || ' seconds')::interval, '{"x": 1}'::jsonb
from public.runs r, generate_series(1, 260000) n
where r.run_key = '00.01.01_run_1_00h00';

select count(*) from public.readings;   -- expect: exactly 250000
```

Confirm the anon role cannot write (should fail with permission denied):

```sql
set role anon;
insert into public.readings (run_id, ts, data) values (1, now(), '{}');
reset role;
```

Clean up the fake run (cascade removes its readings):

```sql
delete from public.runs where run_key = '00.01.01_run_1_00h00';
```

## Free-tier pausing

Free projects **pause after about 7 days without activity** (typically between
campaigns). A paused project refuses all API calls; nothing is lost. Before
each campaign:

1. Open the Supabase dashboard. If the project shows **Paused**, click
   **Restore** and wait for it to come back up (a minute or two).
2. Start recording as usual and confirm the viewer site goes LIVE.

The publisher tolerates a paused project: it logs the failure, backs off, and
keeps buffering recent points — but nobody restores the project for you, so
the pre-campaign check matters.
