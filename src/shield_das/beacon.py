"""Publish the rig's present vacuum level to the mirror (``shield-das-beacon``).

Answers a different question from the publisher: not "how is this run going?"
but "what is the vacuum right now?", while **no run is being recorded**.

Nothing is written to disk. The beacon samples the LabJack directly, keeps
the last ``history_seconds`` of readings in memory, and overwrites a single
row in the mirror's ``standby`` table. There is no run directory, no CSV, no
``runs`` row and no ``readings`` rows -- so a standby session can never be
mistaken for an experiment, can never be swept up by the uploader, and can
never consume any of the ``readings`` row cap that protects the 500 MB free
tier. See ``docs/live_supabase.md``.

The LabJack is opened, read and closed once per sample (~12 ms, so ~1 % duty
cycle at the default 1 s cadence) rather than held. While a run *is* being
recorded the recorder owns the device continuously, so the beacon goes
dormant for the duration and the viewer site falls back to the run's own live
panels.

Configuration is shared with the publisher (``~/.shield_das_publisher.json``
plus the ``SHIELD_SUPABASE_KEY`` environment variable), so the rig keeps one
config file and one copy of the service-role key.
"""

import argparse
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, fields

import numpy as np

from .publisher import (
    DEFAULT_CONFIG_PATH,
    DryRunClient,
    SupabaseClient,
    _get_key,
    _round_sig,
)
from .publisher import PublisherConfig as _PublisherConfig
from .run_monitor import _convert_gauge_voltage, find_active_run

logger = logging.getLogger(__name__)

# Backoff bounds for failed pushes (seconds), matching the publisher.
_BACKOFF_INITIAL_S = 15.0
_BACKOFF_MAX_S = 300.0

# The rig's gauges, in the same shape as ``run_metadata.json`` gauge entries
# so ``_convert_gauge_voltage`` applies unchanged. Override via the config
# file if the rig is re-plumbed.
DEFAULT_GAUGES = [
    {
        "name": "WGM701",
        "type": "WGM701_Gauge",
        "ain_channel": 10,
        "gauge_location": "downstream",
    },
    {
        "name": "CVM211",
        "type": "CVM211_Gauge",
        "ain_channel": 8,
        "gauge_location": "upstream",
    },
    {
        "name": "Baratron626D_1KT",
        "type": "Baratron626D_Gauge",
        "ain_channel": 6,
        "gauge_location": "upstream",
        "full_scale_torr": 1000,
    },
    {
        "name": "Baratron626D_1T",
        "type": "Baratron626D_Gauge",
        "ain_channel": 4,
        "gauge_location": "downstream",
        "full_scale_torr": 1,
    },
]


@dataclass
class BeaconConfig:
    """Configuration for the vacuum beacon.

    Defaults are read from the publisher's config file so the rig keeps a
    single file and a single copy of the service-role key; keys the publisher
    owns are ignored here and vice versa.

    Attributes:
        supabase_url: Base URL of the Supabase project. Required.
        supabase_key: Service-role key. The ``SHIELD_SUPABASE_KEY``
            environment variable takes precedence.
        results_dir: Local directory containing recorded runs, watched only
            to detect that a run has started (the beacon then goes dormant).
        gauges: Gauge descriptions in ``run_metadata.json`` form.
        primary_channel: Gauge whose history is kept and headlined by the
            viewer site.
        sample_period_s: Seconds between LabJack samples.
        history_seconds: Length of the retained history window in seconds.
        push_period_s: Seconds between mirror updates (each one rewrites the
            whole window into the single ``standby`` row).
        staleness_seconds: Passed to ``find_active_run`` when deciding
            whether a run is currently being recorded.
    """

    supabase_url: str = ""
    supabase_key: str | None = None
    results_dir: str = "results"
    gauges: list[dict] = field(default_factory=lambda: list(DEFAULT_GAUGES))
    primary_channel: str = "WGM701"
    sample_period_s: float = 1.0
    history_seconds: float = 60.0
    push_period_s: float = 5.0
    staleness_seconds: float = 120.0

    def __post_init__(self) -> None:
        """Reject cadences that would divide by zero or spin the CPU.

        Raises:
            ValueError: If a period or the window length is not positive.
        """
        for name in ("sample_period_s", "history_seconds", "push_period_s"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")

    @classmethod
    def from_file(cls, path: str = DEFAULT_CONFIG_PATH) -> "BeaconConfig":
        """Load configuration from the shared JSON config file.

        Args:
            path: Path to the JSON config file. If the file does not exist,
                defaults are used.

        Returns:
            The loaded BeaconConfig.
        """
        data = _PublisherConfig.load_raw(path)

        known = {f.name for f in fields(cls)}
        publisher_only = {f.name for f in fields(_PublisherConfig)} - known
        unknown = set(data) - known - publisher_only
        if unknown:
            logger.warning("Ignoring unknown config keys in %s: %s", path, unknown)

        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def history_points(self) -> int:
        """Number of samples held in the history window (at least 1)."""
        return max(1, round(self.history_seconds / self.sample_period_s))


def read_channels(labjack, gauges: list[dict]) -> dict[str, float]:
    """Read every gauge once and convert the voltages to physical units.

    Uses the same metadata-driven conversion as the publisher
    (``run_monitor._convert_gauge_voltage``), so a beacon reading and a
    recorded reading of the same gauge agree exactly. Unknown gauge types
    fall back to raw volts under a ``<name>_V`` key; non-finite values are
    dropped (jsonb cannot hold NaN).

    Args:
        labjack: An open LabJack U6 handle, or None to read simulated
            voltages (test mode).
        gauges: Gauge descriptions in ``run_metadata.json`` form.

    Returns:
        Mapping of channel name to value: ``<gauge>`` in torr, or
        ``<gauge>_V`` in volts for gauge types with no conversion.
    """
    channels: dict[str, float] = {}
    for gauge in gauges:
        if labjack is None:
            voltage = float(np.random.default_rng().uniform(0, 10))
        else:
            voltage = float(
                labjack.getAIN(
                    positiveChannel=int(gauge["ain_channel"]),
                    resolutionIndex=8,
                    gainIndex=0,
                    settlingFactor=2,
                    differential=False,
                )
            )
        values, unit = _convert_gauge_voltage(gauge, np.asarray([voltage], dtype=float))
        value = float(values[0])
        if not math.isfinite(value):
            continue
        name = str(gauge["name"]) if unit == "torr" else f"{gauge['name']}_V"
        channels[name] = _round_sig(value)
    return channels


class LabJackSampler:
    """Samples the gauges by opening and closing the LabJack each time.

    Holding the handle would block the recorder from starting a run. One
    open/read/close cycle measures ~12 ms, so at the default 1 s cadence the
    device is free ~99 % of the time and a run can claim it whenever it
    wants.

    Args:
        gauges: Gauge descriptions in ``run_metadata.json`` form.
        test_mode: Generate simulated voltages instead of touching hardware.
    """

    def __init__(self, gauges: list[dict], test_mode: bool = False):
        self.gauges = gauges
        self.test_mode = test_mode

    def sample(self) -> dict[str, float]:
        """Take one reading of every gauge.

        Returns:
            Mapping of channel name to value in physical units.

        Raises:
            RuntimeError: If the LabJack cannot be opened or read.
        """
        if self.test_mode:
            return read_channels(None, self.gauges)

        import u6

        labjack = None
        try:
            labjack = u6.U6(firstFound=True)
            labjack.getCalibrationData()
            return read_channels(labjack, self.gauges)
        except Exception as exc:
            raise RuntimeError(f"LabJack read failed: {exc}") from None
        finally:
            if labjack is not None:
                try:
                    labjack.close()
                except Exception:
                    logger.debug("Ignoring LabJack close error", exc_info=True)


class VacuumBeacon:
    """Keeps the recent-history window and pushes it to the mirror.

    Args:
        config: Beacon configuration.
        client: Supabase client (or a dry-run stand-in).
        sampler: Object with a ``sample()`` method returning channel values.
    """

    def __init__(self, config: BeaconConfig, client, sampler):
        self.config = config
        self.client = client
        self.sampler = sampler
        self._history: deque[tuple[float, float]] = deque(maxlen=config.history_points)
        self._latest: dict[str, float] = {}
        # -inf so the first tick samples and pushes immediately: a beacon
        # that started five seconds ago should already show a number.
        self._last_sample = float("-inf")
        self._last_push = float("-inf")
        self._backoff_s = 0.0
        self._retry_at = 0.0

    @property
    def history(self) -> list[tuple[float, float]]:
        """The retained ``(epoch seconds, primary value)`` samples."""
        return list(self._history)

    def dormant(self) -> bool:
        """Check whether a run is being recorded and owns the LabJack.

        Returns:
            True if a run is currently active, in which case the beacon must
            not touch the device.
        """
        return (
            find_active_run(
                self.config.results_dir,
                staleness_seconds=self.config.staleness_seconds,
                include_test_runs=True,
            )
            is not None
        )

    def clear(self) -> None:
        """Drop the retained window (used when waking from dormancy).

        Without this, the first push after a run would show a window whose
        oldest points are hours old, misrepresenting a gap as recent history.
        """
        self._history.clear()
        self._latest = {}

    def sample(self, now: float) -> None:
        """Take one reading and append it to the history window.

        Args:
            now: Current wall-clock time in epoch seconds.

        Raises:
            RuntimeError: If the sampler fails.
        """
        channels = self.sampler.sample()
        self._latest = channels
        primary = channels.get(self.config.primary_channel)
        if primary is not None:
            self._history.append((now, primary))

    def payload(self, now: float) -> dict:
        """Build the jsonb payload for the single ``standby`` row.

        History values are rounded to 4 significant digits (ample for a
        sparkline) to keep the row comfortably inside one 2 KB page, while
        the headline current values keep the publisher's 6 digits.

        Args:
            now: Current wall-clock time in epoch seconds.

        Returns:
            The payload dict.
        """
        cutoff = now - self.config.history_seconds
        return {
            "primary": self.config.primary_channel,
            "channels": dict(self._latest),
            "history_seconds": self.config.history_seconds,
            "history": [
                [round(timestamp, 1), _round_sig(value, 4)]
                for timestamp, value in self._history
                if timestamp >= cutoff
            ],
        }

    def push(self, now: float, monotonic: float) -> None:
        """Send the current window to the mirror, respecting the backoff.

        Args:
            now: Current wall-clock time in epoch seconds.
            monotonic: Current monotonic clock reading, for backoff timing.
        """
        if monotonic < self._retry_at or not self._latest:
            return
        try:
            self.client.standby_update(self.payload(now))
        except RuntimeError as exc:
            self._backoff_s = min(
                _BACKOFF_MAX_S, self._backoff_s * 2 or _BACKOFF_INITIAL_S
            )
            self._retry_at = monotonic + self._backoff_s
            logger.warning(
                "Beacon push failed (retrying in %.0f s): %s -- if the "
                "Supabase project was paused for inactivity, restore it from "
                "the dashboard",
                self._backoff_s,
                exc,
            )
        else:
            self._backoff_s = 0.0

    def tick(self, now: float, monotonic: float) -> None:
        """Advance the beacon by one loop iteration.

        Samples on the sample cadence and pushes on the push cadence; a
        failed sample is logged and skipped rather than ending the loop.

        Args:
            now: Current wall-clock time in epoch seconds.
            monotonic: Current monotonic clock reading.
        """
        if monotonic - self._last_sample >= self.config.sample_period_s:
            self._last_sample = monotonic
            try:
                self.sample(now)
            except RuntimeError as exc:
                logger.warning("Skipping sample: %s", exc)
        if monotonic - self._last_push >= self.config.push_period_s:
            self._last_push = monotonic
            self.push(now, monotonic)


def beacon_loop(config: BeaconConfig, client, sampler, iterations: int | None = None):
    """Sample and publish the present vacuum level until stopped.

    Goes dormant (touching neither the LabJack nor the mirror) whenever a run
    is being recorded, since the recorder owns the device then and the viewer
    site shows the run's own live panels.

    Args:
        config: Beacon configuration.
        client: Supabase client (or a dry-run stand-in).
        sampler: Object with a ``sample()`` method.
        iterations: Stop after this many loop iterations instead of running
            forever (used by the tests).
    """
    beacon = VacuumBeacon(config, client, sampler)
    was_dormant = False
    count = 0

    while iterations is None or count < iterations:
        count += 1
        if beacon.dormant():
            if not was_dormant:
                logger.info("A run is recording: beacon dormant until it ends")
                beacon.clear()
                was_dormant = True
            time.sleep(min(10.0, config.sample_period_s * 10))
            continue

        if was_dormant:
            logger.info("Run ended: beacon resuming")
            was_dormant = False

        beacon.tick(time.time(), time.monotonic())
        time.sleep(min(0.5, config.sample_period_s / 2))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the vacuum beacon (``shield-das-beacon``).

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="shield-das-beacon",
        description=(
            "Publish the rig's present vacuum level to the Supabase mirror "
            "while no run is being recorded. Nothing is written to disk and "
            "no run rows are created. See docs/live_supabase.md."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--supabase-url",
        default=None,
        help="Supabase project URL (overrides the config file)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing recorded runs (overrides the config file)",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=None,
        help="Seconds between LabJack samples (overrides the config file)",
    )
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=None,
        help="Length of the retained history window (overrides the config file)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Generate simulated voltages instead of reading the LabJack",
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

    config = BeaconConfig.from_file(args.config)
    if args.supabase_url is not None:
        config.supabase_url = args.supabase_url
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.sample_period is not None:
        config.sample_period_s = args.sample_period
    if args.history_seconds is not None:
        config.history_seconds = args.history_seconds

    client = (
        DryRunClient()
        if args.dry_run
        else SupabaseClient(config.supabase_url, _get_key(config))
    )
    sampler = LabJackSampler(config.gauges, test_mode=args.test_mode)

    logger.info(
        "Beacon: %s every %.1f s, %.0f s window, pushing every %.1f s",
        config.primary_channel,
        config.sample_period_s,
        config.history_seconds,
        config.push_period_s,
    )
    try:
        beacon_loop(config, client, sampler)
    except KeyboardInterrupt:
        logger.info("Stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
