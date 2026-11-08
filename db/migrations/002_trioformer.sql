-- 002_trioformer.sql
-- TrioFormer: hashed inputs, served inferences, per-model latency benchmarks.
--
-- Additive and idempotent: safe to re-run; does not modify anything created by
-- 001_init.sql. Apply AFTER 001_init.sql (Supabase: SQL Editor -> New query).
--
-- Privacy invariant: raw prompt text is NEVER stored. Inputs are identified
-- only by their SHA-256 hex digest.
--
-- Access model: all tables, the materialized view, and the refresh function are
-- reachable ONLY by the API's service_role. anon/authenticated get nothing;
-- Streamlit reads benchmark data through FastAPI.

begin;

-- ---------------------------------------------------------------------------
-- inputs: one row per distinct submitted prompt, keyed by SHA-256 hex digest.
-- ---------------------------------------------------------------------------
create table if not exists public.inputs (
    sha256         text        primary key
                   constraint inputs_sha256_hex check (sha256 ~ '^[0-9a-f]{64}$'),
    first_seen_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- inferences: one row per served generation. latency_ms is measured
-- SERVER-SIDE by the API and written with the service-role key.
-- ---------------------------------------------------------------------------
create table if not exists public.inferences (
    id            bigint generated always as identity primary key,
    input_sha256  text        not null
                  references public.inputs (sha256) on delete cascade,
    model_name    text        not null
                  constraint inferences_model_name_len
                  check (char_length(model_name) between 1 and 100),
    output        text        not null
                  constraint inferences_output_len
                  check (char_length(output) <= 20000),
    latency_ms    integer     not null
                  constraint inferences_latency_range
                  check (latency_ms >= 0 and latency_ms <= 3600000),
    served_at     timestamptz not null default now()
);

create index if not exists idx_inferences_input_sha256
    on public.inferences (input_sha256);
create index if not exists idx_inferences_model_served_at
    on public.inferences (model_name, served_at desc);

-- ---------------------------------------------------------------------------
-- benchmark_refresh_status: single-row metadata recording the ACTUAL last
-- refresh of public.benchmarks. Kept outside the materialized view so the view
-- stays a pure aggregate and the timestamp reflects when a refresh really ran.
-- last_refreshed_at is NULL until refresh_benchmarks() has run successfully.
-- ---------------------------------------------------------------------------
create table if not exists public.benchmark_refresh_status (
    singleton          boolean     primary key default true
                       constraint benchmark_refresh_status_singleton check (singleton),
    last_refreshed_at  timestamptz,
    last_duration_ms   integer,
    last_row_count     integer,
    last_mode          text
                       constraint benchmark_refresh_status_mode
                       check (last_mode in ('initial', 'concurrent'))
);

insert into public.benchmark_refresh_status (singleton)
values (true)
on conflict (singleton) do nothing;

-- ---------------------------------------------------------------------------
-- RLS: deny-by-default (no policies). service_role bypasses RLS.
-- ---------------------------------------------------------------------------
alter table public.inputs                  enable row level security;
alter table public.inferences              enable row level security;
alter table public.benchmark_refresh_status enable row level security;

-- ---------------------------------------------------------------------------
-- benchmarks: latency aggregates per model.
-- NOTE: `create ... if not exists` does NOT update an existing definition. If
-- the query ever changes, ship a new migration that drops and recreates it.
-- ---------------------------------------------------------------------------
create materialized view if not exists public.benchmarks as
select
    model_name,
    count(*)                                                   as n_inferences,
    round(avg(latency_ms)::numeric, 2)                         as mean_latency_ms,
    min(latency_ms)                                            as min_latency_ms,
    percentile_cont(0.5)  within group (order by latency_ms)   as p50_latency_ms,
    percentile_cont(0.95) within group (order by latency_ms)   as p95_latency_ms,
    percentile_cont(0.99) within group (order by latency_ms)   as p99_latency_ms,
    max(latency_ms)                                            as max_latency_ms,
    max(served_at)                                             as last_served_at
from public.inferences
group by model_name;

-- Required by REFRESH ... CONCURRENTLY: a unique, non-partial index on plain
-- columns (model_name is NOT NULL, so every row is covered).
create unique index if not exists idx_benchmarks_model_name
    on public.benchmarks (model_name);

-- ---------------------------------------------------------------------------
-- Refresh procedure.
--
-- * REFRESH MATERIALIZED VIEW CONCURRENTLY is allowed inside a transaction
--   block (unlike CREATE INDEX CONCURRENTLY), so it works when called through
--   PostgREST rpc(), which wraps each call in a transaction.
-- * CONCURRENTLY errors if the view has never been populated (e.g. it was
--   created WITH NO DATA), so we check pg_class.relispopulated and fall back
--   to a plain (briefly blocking) refresh for that one initial case. As written
--   above, the view is created WITH DATA, so this is a defensive fallback.
-- * Two refreshes of the same view cannot overlap, so a transaction-scoped
--   advisory lock makes a second caller return immediately instead of queueing.
-- * SECURITY DEFINER is required: only the view owner may refresh it.
--   search_path is emptied and every reference is schema-qualified.
-- ---------------------------------------------------------------------------
create or replace function public.refresh_benchmarks()
returns jsonb
language plpgsql
security definer
set search_path = ''
set statement_timeout = '30s'   -- function-scoped: restored when the function exits
as $$
declare
    v_start     timestamptz;
    v_populated boolean;
    v_mode      text;
    v_rows      integer;
    v_ms        integer;
begin
    if not pg_catalog.pg_try_advisory_xact_lock(
               pg_catalog.hashtextextended('public.benchmarks', 0)) then
        return pg_catalog.jsonb_build_object(
            'refreshed', false, 'reason', 'refresh_in_progress');
    end if;

    v_start := pg_catalog.clock_timestamp();

    select c.relispopulated
      into v_populated
      from pg_catalog.pg_class c
     where c.oid = 'public.benchmarks'::pg_catalog.regclass;

    if v_populated then
        v_mode := 'concurrent';
        refresh materialized view concurrently public.benchmarks;
    else
        v_mode := 'initial';
        refresh materialized view public.benchmarks;
    end if;

    select pg_catalog.count(*)::integer into v_rows from public.benchmarks;

    v_ms := (extract(epoch from (pg_catalog.clock_timestamp() - v_start)) * 1000)::integer;

    update public.benchmark_refresh_status
       set last_refreshed_at = pg_catalog.clock_timestamp(),
           last_duration_ms  = v_ms,
           last_row_count    = v_rows,
           last_mode         = v_mode
     where singleton;

    return pg_catalog.jsonb_build_object(
        'refreshed', true, 'mode', v_mode, 'rows', v_rows, 'duration_ms', v_ms);
end;
$$;

-- ---------------------------------------------------------------------------
-- Privileges. Supabase auto-grants broad access on new public objects to
-- anon/authenticated, and materialized views bypass RLS, so revoke explicitly.
-- Role grants are guarded so the file also loads on plain PostgreSQL (tests).
-- ---------------------------------------------------------------------------
revoke all on public.inputs                   from public;
revoke all on public.inferences               from public;
revoke all on public.benchmark_refresh_status from public;
revoke all on public.benchmarks               from public;
revoke all on function public.refresh_benchmarks() from public;

do $$
begin
    if exists (select 1 from pg_catalog.pg_roles where rolname = 'anon') then
        revoke all on public.inputs                   from anon;
        revoke all on public.inferences               from anon;
        revoke all on public.benchmark_refresh_status from anon;
        revoke all on public.benchmarks               from anon;
        revoke all on function public.refresh_benchmarks() from anon;
    end if;

    if exists (select 1 from pg_catalog.pg_roles where rolname = 'authenticated') then
        revoke all on public.inputs                   from authenticated;
        revoke all on public.inferences               from authenticated;
        revoke all on public.benchmark_refresh_status from authenticated;
        revoke all on public.benchmarks               from authenticated;
        revoke all on function public.refresh_benchmarks() from authenticated;
    end if;

    if exists (select 1 from pg_catalog.pg_roles where rolname = 'service_role') then
        -- Least privilege: the API appends rows and reads aggregates.
        grant select, insert on public.inputs     to service_role;
        grant select, insert on public.inferences to service_role;
        grant usage, select on sequence public.inferences_id_seq to service_role;
        grant select on public.benchmarks               to service_role;
        grant select on public.benchmark_refresh_status to service_role;
        grant execute on function public.refresh_benchmarks() to service_role;
    end if;
end;
$$;

commit;
