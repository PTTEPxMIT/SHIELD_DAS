-- SHIELD live-data mirror: Supabase schema.
--
-- Idempotent setup script: paste the whole file into the Supabase SQL editor
-- and run it (safe to re-run). See docs/live_supabase.md for the full setup
-- guide, the 500 MB budget, and verification queries.
--
-- Design notes:
--   * This database is a DISPOSABLE LIVE MIRROR. The system of record is the
--     rig CSV -> parquet -> SHIELD-Data on GitHub. Anything here can be
--     deleted at any time.
--   * The free tier allows 500 MB. The readings_cap trigger below bounds the
--     readings table at READINGS_CAP rows *inside the database*, so the limit
--     holds no matter what any client does.
--   * Values in readings.data are physical units (pressure in torr,
--     temperature in degC), keyed by channel name. Raw voltages are not
--     mirrored here; they stay in the CSV/parquet record.

create table if not exists public.runs (
    id           bigint generated always as identity primary key,
    run_key      text not null unique,        -- "YY.MM.DD_run_N_HHhMM"
    metadata     jsonb not null default '{}', -- full run_metadata.json
    started_at   timestamptz not null,
    ended_at     timestamptz,
    last_seen_at timestamptz not null default now()
);

create table if not exists public.readings (
    id     bigint generated always as identity primary key,
    run_id bigint not null references public.runs (id) on delete cascade,
    ts     timestamptz not null,
    data   jsonb not null,                    -- {channel: value in torr/degC}
    unique (run_id, ts)                       -- idempotent inserts + query index
);

-- ---------------------------------------------------------------------------
-- Hard cap: the 500 MB guarantee.
--
-- READINGS_CAP = 250000 rows. At <= ~400 bytes/row all-in (heap + indexes)
-- that is ~100 MB live, ~200 MB with a 2x bloat allowance -- see
-- docs/live_supabase.md for the full budget. The trigger runs in the same
-- transaction as every insert, and inserts are the only way the table grows,
-- so the live row count can never exceed the cap plus one batch.
-- ---------------------------------------------------------------------------

create or replace function public.enforce_readings_cap()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
    cap constant bigint := 250000;
    hi bigint;
begin
    select max(id) into hi from public.readings;
    if hi is not null then
        delete from public.readings where id <= hi - cap;
    end if;
    return null;
end
$$;

drop trigger if exists readings_cap on public.readings;
create trigger readings_cap
    after insert on public.readings
    for each statement
    execute function public.enforce_readings_cap();

-- ---------------------------------------------------------------------------
-- Maintenance RPCs, called by the publisher (service-role key only).
-- These are quality-of-service (keep a whole long run visible by thinning old
-- data instead of letting the cap truncate it); the trigger above is the
-- safety mechanism and needs no cooperation from anyone.
-- ---------------------------------------------------------------------------

-- Thin one run's data older than a cutoff to a coarser cadence: within each
-- p_keep_seconds bucket the earliest row is kept, the rest are deleted.
create or replace function public.thin_readings(
    p_run_id bigint,
    p_older_than_hours int,
    p_keep_seconds int
)
returns int
language plpgsql
set search_path = public
as $$
declare
    n int;
begin
    with old as (
        select id,
               row_number() over (
                   partition by floor(extract(epoch from ts) / p_keep_seconds)
                   order by id
               ) as rn
        from public.readings
        where run_id = p_run_id
          and ts < now() - make_interval(hours => p_older_than_hours)
    )
    delete from public.readings r
    using old
    where r.id = old.id
      and old.rn > 1;
    get diagnostics n = row_count;
    return n;
end
$$;

-- Keep only the newest p_keep runs; readings go with them (FK cascade).
create or replace function public.prune_runs(p_keep int)
returns int
language plpgsql
set search_path = public
as $$
declare
    n int;
begin
    delete from public.runs
    where id not in (
        select id from public.runs order by started_at desc limit p_keep
    );
    get diagnostics n = row_count;
    return n;
end
$$;

-- Server-stamped liveness: the viewer site compares last_seen_at with the
-- database clock, so a wrong rig clock cannot fake or break the LIVE badge.
create or replace function public.heartbeat(p_run_id bigint)
returns void
language sql
set search_path = public
as $$
    update public.runs set last_seen_at = now() where id = p_run_id;
$$;

-- ---------------------------------------------------------------------------
-- Read RPC for the viewer site's first paint: stride-decimated rows for one
-- run, at most p_max_points of them, always including the newest row
-- (the SQL twin of run_monitor._decimation_indices). Also sidesteps
-- PostgREST's default 1000-row response limit.
-- ---------------------------------------------------------------------------

create or replace function public.decimated_readings(
    p_run_id bigint,
    p_max_points int
)
returns setof public.readings
language sql
stable
set search_path = public
as $$
    with numbered as (
        select r.*,
               row_number() over (order by r.id desc) - 1 as back_rn,
               count(*) over () as n
        from public.readings r
        where r.run_id = p_run_id
    )
    select id, run_id, ts, data
    from numbered
    where back_rn % greatest(1, ceil(n::numeric / p_max_points)::int) = 0
    order by id;
$$;

-- ---------------------------------------------------------------------------
-- Access control: the anon key (baked into the public site) may only read.
-- All writes and maintenance go through the service-role key on the rig,
-- which bypasses RLS by design.
-- ---------------------------------------------------------------------------

alter table public.runs enable row level security;
alter table public.readings enable row level security;

drop policy if exists runs_read on public.runs;
create policy runs_read on public.runs
    for select using (true);

drop policy if exists readings_read on public.readings;
create policy readings_read on public.readings
    for select using (true);

grant select on public.runs, public.readings to anon, authenticated;
revoke insert, update, delete on public.runs, public.readings
    from anon, authenticated;

-- Functions are executable by everyone unless revoked; only the decimated
-- read RPC stays public.
revoke execute on function
    public.enforce_readings_cap(),
    public.thin_readings(bigint, int, int),
    public.prune_runs(int),
    public.heartbeat(bigint)
from public, anon, authenticated;

grant execute on function public.decimated_readings(bigint, int)
    to anon, authenticated;
