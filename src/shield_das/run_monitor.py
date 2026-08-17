"""Dash-free helpers for monitoring the run currently being recorded.

Extracted from ``live_dashboard`` so that non-UI consumers (e.g. a publisher
process) can discover the active run, tail its ``shield_data.csv``
incrementally, decimate traces, and convert raw gauge voltages to pressure
without importing Dash or Plotly. ``live_dashboard`` re-exports these names,
so existing imports keep working.
"""

import json
import os
import re
import time
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from .analysis import voltage_to_pressure
from .pressure_gauge import CVM211_Gauge, WGM701_Gauge

DATA_CSV_NAME = "shield_data.csv"
METADATA_NAME = "run_metadata.json"
TIMESTAMP_COLUMN = "RealTimestamp"
LOCAL_TEMPERATURE_COLUMN = "Local_temperature (C)"

# Date directories: old rigs use MM.DD, newer ones YY.MM.DD
_DATE_DIR_RE = re.compile(r"^\d{2}\.\d{2}(\.\d{2})?$")
# Run directories: run_<N>_<HHhMM>, with an optional test_run_ prefix
_RUN_DIR_RE = re.compile(r"^(test_)?run_\d+_\d{1,2}h\d{2}$")


def _parse_timestamp(value: str) -> datetime | None:
    """Parse a CSV RealTimestamp value.

    Args:
        value: Timestamp string ("%Y-%m-%d %H:%M:%S" with optional
            fractional seconds).

    Returns:
        The parsed datetime, or None if the value is not a timestamp.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _read_metadata(run_dir: str) -> dict | None:
    """Read run_metadata.json from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        The parsed metadata dict, or None if missing or corrupt.
    """
    try:
        with open(os.path.join(run_dir, METADATA_NAME)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def find_active_run(
    results_dir: str,
    staleness_seconds: float = 120.0,
    include_test_runs: bool = False,
) -> str | None:
    """Find the run directory that is currently being recorded.

    Scans date directories in both the old ``MM.DD`` and new ``YY.MM.DD``
    naming forms. A run is considered active when its metadata has no
    ``run_info.end_time`` AND its ``shield_data.csv`` was modified within
    the last ``staleness_seconds``. ``test_run_*`` directories (recorder
    test mode) are skipped unless ``include_test_runs`` is True.

    Args:
        results_dir: Directory containing date-named run subdirectories.
        staleness_seconds: Maximum age in seconds of the CSV's last write
            for the run to still count as live.
        include_test_runs: Also consider ``test_run_*`` directories.

    Returns:
        Absolute path of the newest active run directory (by CSV mtime),
        or None if no run is live.
    """
    if not os.path.isdir(results_dir):
        return None

    now = time.time()
    candidates: list[tuple[float, str]] = []

    for date_name in sorted(os.listdir(results_dir)):
        date_dir = os.path.join(results_dir, date_name)
        if not os.path.isdir(date_dir) or not _DATE_DIR_RE.match(date_name):
            continue

        for run_name in sorted(os.listdir(date_dir)):
            run_dir = os.path.join(date_dir, run_name)
            if not os.path.isdir(run_dir) or not _RUN_DIR_RE.match(run_name):
                continue
            if run_name.startswith("test_run_") and not include_test_runs:
                continue

            csv_path = os.path.join(run_dir, DATA_CSV_NAME)
            if not os.path.isfile(csv_path):
                continue

            metadata = _read_metadata(run_dir)
            if metadata is None:
                continue
            if metadata.get("run_info", {}).get("end_time"):
                continue

            mtime = os.path.getmtime(csv_path)
            if now - mtime > staleness_seconds:
                continue

            candidates.append((mtime, os.path.abspath(run_dir)))

    if not candidates:
        return None
    return max(candidates)[1]


class IncrementalRunReader:
    """Incrementally tail a run's ``shield_data.csv`` without re-reading it.

    The first ``poll()`` parses the CSV header and loads all rows written so
    far; every subsequent ``poll()`` seeks to the stored byte offset and reads
    only the newly appended complete lines. A trailing partial line (the
    recorder may be mid-append) is buffered and completed on a later poll.
    A missing CSV or a vanished run directory is tolerated: ``poll()`` simply
    returns 0 new rows.

    Args:
        run_dir: Path to the run directory containing ``shield_data.csv``.

    Attributes:
        run_dir: Path to the run directory.
        csv_path: Path to the tailed CSV file.
        columns: Column names from the CSV header, or None before the header
            has been read.
        data: Mapping of column name to a growing list of values. The
            ``RealTimestamp`` column holds ``datetime`` objects; every other
            column holds floats (raw volts, millivolts, or °C as recorded).
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.csv_path = os.path.join(run_dir, DATA_CSV_NAME)
        self.columns: list[str] | None = None
        self.data: dict[str, list] = {}
        self._offset = 0
        self._partial = ""

    @property
    def row_count(self) -> int:
        """Number of complete data rows read so far."""
        if not self.data:
            return 0
        return len(self.data.get(TIMESTAMP_COLUMN, []))

    @property
    def elapsed_seconds(self) -> float:
        """Seconds between the first and last timestamps read (0 if <2 rows)."""
        timestamps = self.data.get(TIMESTAMP_COLUMN, [])
        if len(timestamps) < 2:
            return 0.0
        return (timestamps[-1] - timestamps[0]).total_seconds()

    def poll(self) -> int:
        """Read any newly appended complete CSV lines.

        Returns:
            Number of new complete data rows appended to ``data``.
        """
        try:
            with open(self.csv_path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read()
        except OSError:
            # File not written yet, or the run directory disappeared
            return 0

        if not chunk:
            return 0
        self._offset += len(chunk)

        text = self._partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        # A chunk not ending in a newline ends with a partial row: buffer it
        self._partial = "" if text.endswith("\n") else lines[-1]
        complete_lines = lines[:-1]

        new_rows = 0
        for line in complete_lines:
            line = line.strip("\r")
            if not line.strip():
                continue
            if self.columns is None:
                self.columns = [name.strip() for name in line.split(",")]
                self.data = {name: [] for name in self.columns}
                continue

            fields = line.split(",")
            if len(fields) != len(self.columns):
                continue  # malformed row

            row_values: dict[str, object] = {}
            valid = True
            for column, raw in zip(self.columns, fields):
                if column == TIMESTAMP_COLUMN:
                    timestamp = _parse_timestamp(raw.strip())
                    if timestamp is None:
                        valid = False
                        break
                    row_values[column] = timestamp
                else:
                    try:
                        row_values[column] = float(raw)
                    except ValueError:
                        row_values[column] = float("nan")
            if not valid:
                continue

            for column, value in row_values.items():
                self.data[column].append(value)
            new_rows += 1

        return new_rows


def _decimation_indices(n_points: int, max_points: int) -> NDArray:
    """Stride-based decimation indices that always keep the last point.

    Args:
        n_points: Number of points available.
        max_points: Maximum number of points to keep.

    Returns:
        Sorted array of at most ``max_points`` indices into the data, always
        including index ``n_points - 1`` (the most recent point).
    """
    if n_points <= max_points:
        return np.arange(n_points)
    stride = -(-n_points // max_points)  # ceil division
    # Walk backwards from the newest point so it is always kept
    return np.arange(n_points - 1, -1, -stride)[::-1]


def _convert_gauge_voltage(gauge: dict, voltage_v: NDArray) -> tuple[NDArray, str]:
    """Convert a gauge voltage trace to pressure using the existing functions.

    Reuses ``WGM701_Gauge.voltage_to_pressure`` and
    ``CVM211_Gauge.voltage_to_pressure`` (log-scale gauges) and
    ``analysis.voltage_to_pressure`` (linear Baratron via full scale), exactly
    as ``Dataset.process_data`` does. Unknown gauge types fall back to raw
    volts.

    Args:
        gauge: Gauge metadata dict with ``type`` and, for Baratrons,
            ``full_scale_torr``.
        voltage_v: Raw gauge voltages in volts.

    Returns:
        Tuple of (converted values, unit string): pressures with unit
        ``"torr"``, or the unchanged voltages with unit ``"V"`` when no
        conversion is available.
    """
    gauge_type = gauge.get("type")
    if gauge_type == "Baratron626D_Gauge" and gauge.get("full_scale_torr"):
        pressure_torr = voltage_to_pressure(
            voltage_v, full_scale_torr=float(gauge["full_scale_torr"])
        )
        return pressure_torr, "torr"
    if gauge_type == "WGM701_Gauge":
        return WGM701_Gauge().voltage_to_pressure(voltage_v), "torr"
    if gauge_type == "CVM211_Gauge":
        return CVM211_Gauge().voltage_to_pressure(voltage_v), "torr"
    return np.asarray(voltage_v, dtype=float), "V"
