"""Publish the live run to a Supabase mirror (``shield-das-publish``).

Runs as a separate process on the rig PC, next to (never inside) the
recorder. It discovers the run currently being recorded, tails its
``shield_data.csv`` read-only, downsamples new rows to a coarser cadence,
converts raw voltages to physical units (torr, °C), and pushes them to a
Supabase project over PostgREST so the GitHub Pages viewer site can plot
them. See ``docs/live_supabase.md``.

The publisher is stateless across restarts: runs are upserted by their
canonical key and reading inserts are idempotent (server-side unique
``(run_id, ts)`` with duplicates ignored), so on restart it asks the server
what was already published and continues from there. Network failures are
tolerated with a bounded buffer and exponential backoff — the CSV on disk
remains the system of record, the mirror is disposable.

Configuration mirrors the uploader: ``~/.shield_das_publisher.json`` plus the
``SHIELD_SUPABASE_KEY`` environment variable (which takes precedence over the
config file) holding the project's service-role key.
"""

import argparse
import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime

import numpy as np

from .analysis import voltage_to_temperature
from .run_monitor import (
    LOCAL_TEMPERATURE_COLUMN,
    TIMESTAMP_COLUMN,
    IncrementalRunReader,
    _convert_gauge_voltage,
    _read_metadata,
    find_active_run,
)
from .uploader import _redact, run_key

logger = logging.getLogger(__name__)


def _mirror_run_key(run_dir: str) -> str:
    """Build the mirror key for a run directory.

    Uses the canonical SHIELD-Data key (``YY.MM.DD_run_N_HHhMM``) so mirror
    runs and archived runs share names. ``test_run_*`` directories (which the
    uploader deliberately rejects) fall back to ``<date>_<dirname>``.

    Args:
        run_dir: Path to the run directory.

    Returns:
        The key string.
    """
    try:
        return run_key(run_dir)
    except ValueError:
        normalised = os.path.normpath(run_dir)
        date_name = os.path.basename(os.path.dirname(normalised))
        return f"{date_name}_{os.path.basename(normalised)}"


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), ".shield_das_publisher.json"
)
SUPABASE_KEY_ENV_VAR = "SHIELD_SUPABASE_KEY"

# PostgREST rejects overlong requests; 500 rows * ~100 B is comfortably small
INSERT_BATCH_ROWS = 500

# Backoff bounds for failed HTTP calls (seconds)
_BACKOFF_INITIAL_S = 15.0
_BACKOFF_MAX_S = 300.0


@dataclass
class PublisherConfig:
    """Configuration for the live publisher.

    Attributes:
        results_dir: Local directory containing recorded runs
            (``<date>/<run>/`` subdirectories).
        supabase_url: Base URL of the Supabase project
            (``https://<ref>.supabase.co``). Required.
        supabase_key: Service-role key. The ``SHIELD_SUPABASE_KEY``
            environment variable takes precedence.
        sample_period_s: Minimum spacing in seconds between published points
            (the CSV is recorded faster; the mirror only needs plot-rate
            data).
        flush_period_s: How often in seconds buffered points are sent
            (one insert batch plus one heartbeat per flush).
        staleness_seconds: A run whose CSV has not been written for this many
            seconds is no longer considered live.
        include_test_runs: Also publish ``test_run_*`` (recorder test mode)
            directories.
        keep_runs: Number of most recent runs kept in the mirror; older ones
            are pruned (with their readings) when a new run attaches.
        thin_after_hours: Published data older than this many hours is
            thinned server-side...
        thin_to_seconds: ...to one point per this many seconds.
        max_buffer_rows: Maximum unsent points held while the project is
            unreachable; beyond this the oldest points are dropped
            (the CSV keeps everything).
    """

    results_dir: str = "results"
    supabase_url: str = ""
    supabase_key: str | None = None
    sample_period_s: float = 5.0
    flush_period_s: float = 15.0
    staleness_seconds: float = 120.0
    include_test_runs: bool = False
    keep_runs: int = 2
    thin_after_hours: int = 48
    thin_to_seconds: int = 60
    max_buffer_rows: int = 20000

    @classmethod
    def from_file(cls, path: str = DEFAULT_CONFIG_PATH) -> "PublisherConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON config file. If the file does not exist,
                defaults are used.

        Returns:
            The loaded PublisherConfig.
        """
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning("Ignoring unknown config keys in %s: %s", path, unknown)

        return cls(**{k: v for k, v in data.items() if k in known})


def _get_key(config: PublisherConfig) -> str:
    """Resolve the Supabase service-role key from the environment or config.

    Args:
        config: Publisher configuration.

    Returns:
        The key string.

    Raises:
        RuntimeError: If no key is configured.
    """
    key = os.environ.get(SUPABASE_KEY_ENV_VAR) or config.supabase_key
    if not key:
        raise RuntimeError(
            f"No Supabase key: set the {SUPABASE_KEY_ENV_VAR} environment "
            'variable or the "supabase_key" key in the publisher config file'
        )
    return key


def _round_sig(value: float, digits: int = 6) -> float:
    """Round a value to a number of significant digits.

    Bounds the width of numbers stored in the mirror's jsonb payloads, which
    is part of the per-row size budget (see docs/live_supabase.md).

    Args:
        value: Value to round.
        digits: Significant digits to keep.

    Returns:
        The rounded value.
    """
    return float(f"{value:.{digits}g}")


def row_to_channels(metadata: dict, row: dict) -> dict[str, float]:
    """Convert one CSV row of raw readings to physical units.

    Gauge voltages become pressures in torr via the metadata-driven per-gauge
    conversions (unknown gauge types fall back to raw volts under a
    ``<name>_V`` key); thermocouple millivolts become °C via
    ``analysis.voltage_to_temperature`` with the recorded local temperature
    as the cold junction. Non-finite values are dropped (jsonb cannot hold
    NaN), and everything is rounded to 6 significant digits.

    Args:
        metadata: Parsed ``run_metadata.json`` for the run.
        row: Mapping of CSV column name to its raw value for one sample.

    Returns:
        Mapping of channel name to value: ``<gauge>`` in torr, ``<tc>_C``
        and ``local_C`` in °C, ``<gauge>_V`` in volts for unknown gauges.
    """
    channels: dict[str, float] = {}

    def put(name: str, value: float) -> None:
        if math.isfinite(value):
            channels[name] = _round_sig(value)

    for gauge in metadata.get("gauges", []):
        column = f"{gauge.get('name')}_Voltage (V)"
        if column not in row:
            continue
        values, unit = _convert_gauge_voltage(
            gauge, np.asarray([row[column]], dtype=float)
        )
        if unit == "torr":
            put(str(gauge["name"]), float(values[0]))
        else:
            put(f"{gauge['name']}_V", float(values[0]))

    local_temp_c = row.get(LOCAL_TEMPERATURE_COLUMN)
    if local_temp_c is not None:
        put("local_C", float(local_temp_c))
        for thermocouple in metadata.get("thermocouples", []):
            column = f"{thermocouple.get('name')}_Voltage (mV)"
            if column not in row:
                continue
            temperature_c = voltage_to_temperature(
                local_temperature=np.asarray([local_temp_c], dtype=float),
                voltage=np.asarray([row[column]], dtype=float),
            )
            put(f"{thermocouple['name']}_C", float(temperature_c[0]))

    return channels


def _to_utc_iso(timestamp: datetime) -> str:
    """Format a naive rig-local timestamp as an aware ISO-8601 string.

    Args:
        timestamp: Naive datetime in the rig's local timezone (as recorded
            in the CSV).

    Returns:
        ISO-8601 string with a UTC offset attached.
    """
    return timestamp.astimezone().isoformat()


class SupabaseClient:
    """Minimal PostgREST client for the live mirror (urllib only).

    Args:
        url: Base project URL (``https://<ref>.supabase.co``).
        key: Service-role key used for every request.
        timeout_s: Per-request timeout in seconds.
    """

    def __init__(self, url: str, key: str, timeout_s: float = 10.0):
        if not url:
            raise RuntimeError(
                'No Supabase URL: set the "supabase_url" key in the publisher '
                "config file or pass --supabase-url"
            )
        self._base = url.rstrip("/") + "/rest/v1"
        self._key = key
        self._timeout_s = timeout_s

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | list | None = None,
        prefer: str | None = None,
    ) -> dict | list | None:
        """Send one PostgREST request.

        Args:
            method: HTTP method.
            path: Path (with query string) below ``/rest/v1``.
            payload: JSON body, if any.
            prefer: Value for the ``Prefer`` header, if any.

        Returns:
            The decoded JSON response, or None for an empty response.

        Raises:
            RuntimeError: On any HTTP or network error, with the key
                redacted from the message.
        """
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self._base + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = _redact(exc.read().decode("utf-8", errors="replace"), self._key)
            raise RuntimeError(
                f"Supabase error {exc.code} for {method} {path}: {detail}"
            ) from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Supabase unreachable for {method} {path}: "
                f"{_redact(str(exc.reason), self._key)}"
            ) from None
        if not body:
            return None
        return json.loads(body)

    def upsert_run(self, key: str, metadata: dict, started_at: str) -> int:
        """Create or update the mirror row for a run.

        Args:
            key: Canonical run key (``YY.MM.DD_run_N_HHhMM``).
            metadata: Full ``run_metadata.json`` contents.
            started_at: Run start as an ISO-8601 timestamp.

        Returns:
            The run's mirror id.
        """
        rows = self._request(
            "POST",
            "/runs?on_conflict=run_key",
            payload=[{"run_key": key, "metadata": metadata, "started_at": started_at}],
            prefer="resolution=merge-duplicates,return=representation",
        )
        return int(rows[0]["id"])

    def last_reading_ts(self, run_id: int) -> datetime | None:
        """Fetch the newest published reading timestamp for a run.

        Args:
            run_id: Mirror id of the run.

        Returns:
            The timestamp (timezone-aware), or None if nothing is published.
        """
        rows = self._request(
            "GET", f"/readings?run_id=eq.{run_id}&select=ts&order=ts.desc&limit=1"
        )
        if not rows:
            return None
        return datetime.fromisoformat(rows[0]["ts"])

    def insert_readings(self, rows: list[dict]) -> None:
        """Insert reading rows, ignoring any already-published duplicates.

        Args:
            rows: Reading rows (``run_id``, ``ts``, ``data``).
        """
        self._request(
            "POST",
            "/readings?on_conflict=run_id,ts",
            payload=rows,
            prefer="resolution=ignore-duplicates",
        )

    def mark_ended(self, run_id: int, ended_at: str) -> None:
        """Record a run's end time in the mirror.

        Args:
            run_id: Mirror id of the run.
            ended_at: End time as an ISO-8601 timestamp.
        """
        self._request("PATCH", f"/runs?id=eq.{run_id}", payload={"ended_at": ended_at})

    def heartbeat(self, run_id: int) -> None:
        """Bump the run's server-stamped liveness timestamp.

        Args:
            run_id: Mirror id of the run.
        """
        self._request("POST", "/rpc/heartbeat", payload={"p_run_id": run_id})

    def thin_readings(self, run_id: int, older_than_hours: int, keep_seconds: int):
        """Thin a run's old readings to a coarser cadence server-side.

        Args:
            run_id: Mirror id of the run.
            older_than_hours: Only rows older than this many hours are
                thinned.
            keep_seconds: One row per this many seconds is kept.
        """
        self._request(
            "POST",
            "/rpc/thin_readings",
            payload={
                "p_run_id": run_id,
                "p_older_than_hours": older_than_hours,
                "p_keep_seconds": keep_seconds,
            },
        )

    def prune_runs(self, keep: int) -> None:
        """Delete all but the newest ``keep`` runs (readings cascade).

        Args:
            keep: Number of runs to keep.
        """
        self._request("POST", "/rpc/prune_runs", payload={"p_keep": keep})


class DryRunClient:
    """Stand-in client that prints requests instead of sending them."""

    def upsert_run(self, key: str, metadata: dict, started_at: str) -> int:
        print(f"[dry-run] upsert run {key} (started {started_at})")
        return 0

    def last_reading_ts(self, run_id: int) -> None:
        return None

    def insert_readings(self, rows: list[dict]) -> None:
        print(f"[dry-run] insert {len(rows)} readings, e.g. {rows[0]}")

    def mark_ended(self, run_id: int, ended_at: str) -> None:
        print(f"[dry-run] mark run ended at {ended_at}")

    def heartbeat(self, run_id: int) -> None:
        print("[dry-run] heartbeat")

    def thin_readings(self, run_id, older_than_hours, keep_seconds) -> None:
        print(f"[dry-run] thin readings older than {older_than_hours} h")

    def prune_runs(self, keep: int) -> None:
        print(f"[dry-run] prune to newest {keep} runs")


class RunPublisher:
    """Publishes one active run to the mirror.

    Owns an :class:`IncrementalRunReader` on the run's CSV, a bounded buffer
    of not-yet-sent points, and the backoff state for network failures.

    Args:
        run_dir: Path to the run directory.
        config: Publisher configuration.
        client: Supabase client (or :class:`DryRunClient`).
    """

    def __init__(self, run_dir: str, config: PublisherConfig, client):
        self.run_dir = run_dir
        self.config = config
        self.client = client
        self.metadata = _read_metadata(run_dir) or {}
        self.reader = IncrementalRunReader(run_dir)
        self.run_id: int | None = None
        self._pending: deque[dict] = deque(maxlen=config.max_buffer_rows)
        self._consumed_rows = 0
        self._last_sampled: datetime | None = None
        self._skip_before: datetime | None = None
        self._last_flush = 0.0
        self._last_thin = time.monotonic()
        self._backoff_s = 0.0
        self._retry_at = 0.0

    def attach(self) -> None:
        """Register the run in the mirror and resume where it left off.

        Upserts the run row (idempotent by run key), prunes old runs, and
        fetches the newest already-published timestamp so a restarted
        publisher skips history the server already has.
        """
        started = self.metadata.get("run_info", {}).get("start_time")
        started_at = _parse_run_start(started)
        self.run_id = self.client.upsert_run(
            _mirror_run_key(self.run_dir), self.metadata, started_at
        )
        self.client.prune_runs(self.config.keep_runs)
        self._skip_before = self.client.last_reading_ts(self.run_id)
        logger.info(
            "Publishing %s as run %s%s",
            self.run_dir,
            self.run_id,
            f" (resuming after {self._skip_before})" if self._skip_before else "",
        )

    def tick(self) -> None:
        """Read new CSV rows, downsample them, and flush on the cadence."""
        self.reader.poll()
        self._downsample_new_rows()
        now = time.monotonic()
        if now - self._last_flush >= self.config.flush_period_s:
            self._last_flush = now
            self._flush(max_batches=1, with_heartbeat=True)
        if now - self._last_thin >= 3600.0:
            self._last_thin = now
            self._maintain()

    def ended(self) -> bool:
        """Check whether the run has finished or gone stale.

        Returns:
            True if the metadata gained an ``end_time``, the CSV went stale,
            or the run directory disappeared.
        """
        metadata = _read_metadata(self.run_dir)
        if metadata is None:
            return True
        if metadata.get("run_info", {}).get("end_time"):
            self.metadata = metadata
            return True
        try:
            age = time.time() - os.path.getmtime(self.reader.csv_path)
        except OSError:
            return True
        return age > self.config.staleness_seconds

    def finish(self) -> None:
        """Flush everything still buffered and mark the run ended."""
        self._flush(max_batches=None, with_heartbeat=False)
        timestamps = self.reader.data.get(TIMESTAMP_COLUMN, [])
        ended_at = _to_utc_iso(timestamps[-1]) if timestamps else None
        if self.run_id is not None and ended_at is not None:
            try:
                self.client.mark_ended(self.run_id, ended_at)
            except RuntimeError as exc:
                logger.warning("Could not mark run ended: %s", exc)

    def _downsample_new_rows(self) -> None:
        """Move rows the reader gained into the pending buffer, downsampled.

        Keeps at most one point per ``sample_period_s``, skips rows the
        server already has, and converts each kept row to physical units.
        """
        timestamps = self.reader.data.get(TIMESTAMP_COLUMN, [])
        columns = self.reader.columns or []
        for i in range(self._consumed_rows, len(timestamps)):
            timestamp = timestamps[i]
            if (
                self._last_sampled is not None
                and (timestamp - self._last_sampled).total_seconds()
                < self.config.sample_period_s
            ):
                continue
            self._last_sampled = timestamp
            aware = timestamp.astimezone()
            if self._skip_before is not None and aware <= self._skip_before:
                continue
            row = {column: self.reader.data[column][i] for column in columns}
            data = row_to_channels(self.metadata, row)
            if not data:
                continue
            self._pending.append(
                {"run_id": self.run_id, "ts": aware.isoformat(), "data": data}
            )
        self._consumed_rows = len(timestamps)

    def _flush(self, max_batches: int | None, with_heartbeat: bool) -> None:
        """Send pending points (and optionally a heartbeat) to the mirror.

        Respects the failure backoff window; on failure the points stay in
        the bounded buffer for the next attempt.

        Args:
            max_batches: Maximum insert batches to send, or None for all.
            with_heartbeat: Also bump the run's liveness timestamp.
        """
        now = time.monotonic()
        if now < self._retry_at:
            return
        try:
            sent = 0
            while self._pending and (max_batches is None or sent < max_batches):
                batch = [
                    self._pending[i]
                    for i in range(min(INSERT_BATCH_ROWS, len(self._pending)))
                ]
                self.client.insert_readings(batch)
                for _ in batch:
                    self._pending.popleft()
                sent += 1
            if with_heartbeat and self.run_id is not None:
                self.client.heartbeat(self.run_id)
        except RuntimeError as exc:
            self._backoff_s = min(
                _BACKOFF_MAX_S, self._backoff_s * 2 or _BACKOFF_INITIAL_S
            )
            self._retry_at = now + self._backoff_s
            logger.warning(
                "Publish failed (retrying in %.0f s): %s -- if the Supabase "
                "project was paused for inactivity, restore it from the "
                "dashboard",
                self._backoff_s,
                exc,
            )
        else:
            self._backoff_s = 0.0

    def _maintain(self) -> None:
        """Run the hourly server-side thinning; failures only log."""
        if self.run_id is None:
            return
        try:
            self.client.thin_readings(
                self.run_id,
                self.config.thin_after_hours,
                self.config.thin_to_seconds,
            )
        except RuntimeError as exc:
            logger.warning("Thinning failed: %s", exc)


def _parse_run_start(start_time: str | None) -> str:
    """Build an ISO start timestamp from the run metadata.

    Args:
        start_time: ``run_info.start_time`` as written by the recorder
            (``%Y-%m-%d %H:%M:%S``, rig-local), possibly None.

    Returns:
        ISO-8601 timestamp with the local UTC offset; falls back to "now"
        if the metadata field cannot be parsed.
    """
    try:
        naive = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return datetime.now().astimezone().isoformat()
    return _to_utc_iso(naive)


def publish_loop(config: PublisherConfig, client, once: bool = False) -> None:
    """Watch for the active run and publish it until stopped.

    Mirrors the live dashboard's WAITING -> LIVE -> ENDED state machine:
    poll ``results/`` every 10 s until a run is live, publish it until it
    ends, then go back to watching.

    Args:
        config: Publisher configuration.
        client: Supabase client (or :class:`DryRunClient`).
        once: Return after the first published run ends instead of going
            back to watching.
    """
    while True:
        run_dir = find_active_run(
            config.results_dir,
            staleness_seconds=config.staleness_seconds,
            include_test_runs=config.include_test_runs,
        )
        if run_dir is None:
            time.sleep(10)
            continue

        publisher = RunPublisher(run_dir, config, client)
        try:
            publisher.attach()
        except RuntimeError as exc:
            logger.warning(
                "Cannot attach to %s (retrying in 60 s): %s -- if the Supabase "
                "project was paused for inactivity, restore it from the "
                "dashboard",
                run_dir,
                exc,
            )
            time.sleep(60)
            continue

        while not publisher.ended():
            publisher.tick()
            time.sleep(1)
        publisher.tick()  # pick up rows written just before the end
        publisher.finish()
        logger.info("Run ended: %s", run_dir)
        if once:
            return


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the live publisher (``shield-das-publish``).

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="shield-das-publish",
        description=(
            "Publish the run currently being recorded to the Supabase live "
            "mirror. Watches results/ until a run is live, then pushes "
            "downsampled readings in physical units. See "
            "docs/live_supabase.md."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing recorded runs (overrides the config file)",
    )
    parser.add_argument(
        "--supabase-url",
        default=None,
        help="Supabase project URL (overrides the config file)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit when the current run ends instead of watching forever",
    )
    parser.add_argument(
        "--include-test-runs",
        action="store_true",
        help="Also publish test_run_* (recorder test mode) directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be sent without any network access",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    config = PublisherConfig.from_file(args.config)
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.supabase_url is not None:
        config.supabase_url = args.supabase_url
    if args.include_test_runs:
        config.include_test_runs = True

    if args.dry_run:
        client = DryRunClient()
    else:
        client = SupabaseClient(config.supabase_url, _get_key(config))

    try:
        publish_loop(config, client, once=args.once)
    except KeyboardInterrupt:
        logger.info("Stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
