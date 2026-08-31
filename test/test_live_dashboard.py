"""Tests for the live run dashboard.

Covers active-run discovery, incremental CSV tailing (offset-based, with
partial-line buffering), stride decimation, metadata-driven trace building,
and the waiting -> live -> ended callback transitions. No hardware, no
network, and no real Dash server are used.
"""

import json
import os
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest

from shield_das.live_dashboard import (
    DashboardState,
    IncrementalRunReader,
    LiveDashboardConfig,
    _decimation_indices,
    _leak_rate_torr_per_s,
    _leak_test_readout,
    build_traces,
    find_active_run,
    update_dashboard,
)

# =============================================================================
# Helpers for building fake run directories
# =============================================================================

GAUGES = [
    {
        "name": "Baratron626D_1KT",
        "type": "Baratron626D_Gauge",
        "gauge_location": "upstream",
        "full_scale_torr": 1000,
    },
    {
        "name": "Baratron626D_1T",
        "type": "Baratron626D_Gauge",
        "gauge_location": "downstream",
        "full_scale_torr": 1,
    },
    {
        "name": "CVM211",
        "type": "CVM211_Gauge",
        "gauge_location": "upstream",
    },
    {
        "name": "WGM701",
        "type": "WGM701_Gauge",
        "gauge_location": "downstream",
    },
]

THERMOCOUPLES = [{"name": "TC1"}]


def csv_header(gauges, thermocouples):
    """Build the CSV header line for the given instruments."""
    columns = ["RealTimestamp"]
    if thermocouples:
        columns.append("Local_temperature (C)")
        columns += [f"{t['name']}_Voltage (mV)" for t in thermocouples]
    columns += [f"{g['name']}_Voltage (V)" for g in gauges]
    return ",".join(columns)


def csv_row(row_index, gauges, thermocouples, voltage_v=5.0):
    """Build one CSV data row with a timestamp 0.5 s apart per row."""
    timestamp = datetime(2026, 8, 4, 10, 0, 0) + timedelta(seconds=0.5 * row_index)
    fields = [timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]]
    if thermocouples:
        fields.append("25.0")  # Local_temperature (C)
        fields += ["1.0" for _ in thermocouples]  # thermocouple mV
    fields += [str(voltage_v) for _ in gauges]
    return ",".join(fields)


def make_run(
    results_dir,
    date_name="26.08.04",
    run_name="run_1_10h00",
    n_rows=5,
    gauges=GAUGES,
    thermocouples=THERMOCOUPLES,
    end_time=None,
    run_info_extra=None,
):
    """Create a fake run directory with metadata and CSV; return its path."""
    run_dir = os.path.join(str(results_dir), date_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    run_info = {"date": "2026-08-04", "start_time": "2026-08-04 10:00:00"}
    if end_time is not None:
        run_info["end_time"] = end_time
    if run_info_extra:
        run_info.update(run_info_extra)
    metadata = {"version": "1.3", "run_info": run_info, "gauges": gauges}
    if thermocouples:
        metadata["thermocouples"] = thermocouples
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f)

    lines = [csv_header(gauges, thermocouples)]
    lines += [csv_row(i, gauges, thermocouples) for i in range(n_rows)]
    with open(os.path.join(run_dir, "shield_data.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return run_dir


def make_stale(run_dir, age_seconds=3600):
    """Backdate the run's CSV mtime by age_seconds."""
    csv_path = os.path.join(run_dir, "shield_data.csv")
    old = os.path.getmtime(csv_path) - age_seconds
    os.utime(csv_path, (old, old))


# =============================================================================
# Tests for find_active_run
# =============================================================================


def test_find_active_run_returns_fresh_run_without_end_time(tmp_path):
    """A run with no end_time and a fresh CSV is found."""
    run_dir = make_run(tmp_path)

    assert find_active_run(str(tmp_path)) == run_dir


def test_find_active_run_excludes_run_with_end_time(tmp_path):
    """A run whose metadata has end_time is not live."""
    make_run(tmp_path, end_time="2026-08-04 11:00:00")

    assert find_active_run(str(tmp_path)) is None


def test_find_active_run_excludes_stale_csv(tmp_path):
    """A run whose CSV mtime is older than staleness_seconds is not live."""
    run_dir = make_run(tmp_path)
    make_stale(run_dir, age_seconds=3600)

    assert find_active_run(str(tmp_path), staleness_seconds=120) is None


def test_find_active_run_supports_old_date_dir_format(tmp_path):
    """Old-style MM.DD date directories are scanned too."""
    run_dir = make_run(tmp_path, date_name="08.04")

    assert find_active_run(str(tmp_path)) == run_dir


def test_find_active_run_skips_test_runs_by_default(tmp_path):
    """test_run_* directories are skipped unless include_test_runs is set."""
    run_dir = make_run(tmp_path, run_name="test_run_1_10h00")

    assert find_active_run(str(tmp_path)) is None
    assert find_active_run(str(tmp_path), include_test_runs=True) == run_dir


def test_find_active_run_returns_newest_of_multiple(tmp_path):
    """With two live runs, the one with the newest CSV wins."""
    older = make_run(tmp_path, run_name="run_1_09h00")
    newer = make_run(tmp_path, run_name="run_2_10h00")
    make_stale(older, age_seconds=60)  # still fresh, but older than run_2

    assert find_active_run(str(tmp_path)) == newer


def test_find_active_run_missing_results_dir_returns_none(tmp_path):
    """A nonexistent results directory yields None, not an error."""
    assert find_active_run(str(tmp_path / "nope")) is None


def test_find_active_run_ignores_non_run_directories(tmp_path):
    """Directories not matching the date/run naming are ignored."""
    os.makedirs(tmp_path / "not_a_date" / "run_1_10h00")
    os.makedirs(tmp_path / "26.08.04" / "not_a_run")

    assert find_active_run(str(tmp_path)) is None


# =============================================================================
# Tests for IncrementalRunReader
# =============================================================================


def test_reader_initial_load_parses_header_and_rows(tmp_path):
    """First poll reads the header and all existing rows."""
    run_dir = make_run(tmp_path, n_rows=4)
    reader = IncrementalRunReader(run_dir)

    assert reader.poll() == 4
    assert reader.row_count == 4
    assert "RealTimestamp" in reader.columns
    assert "Baratron626D_1KT_Voltage (V)" in reader.columns
    assert reader.data["Baratron626D_1KT_Voltage (V)"] == [5.0, 5.0, 5.0, 5.0]
    assert isinstance(reader.data["RealTimestamp"][0], datetime)
    # 4 rows spaced 0.5 s apart -> 1.5 s elapsed
    assert reader.elapsed_seconds == pytest.approx(1.5)


def test_reader_incremental_poll_reads_only_new_bytes(tmp_path):
    """After the initial load, poll only consumes newly appended bytes."""
    run_dir = make_run(tmp_path, n_rows=3)
    csv_path = os.path.join(run_dir, "shield_data.csv")
    reader = IncrementalRunReader(run_dir)
    reader.poll()

    size_after_first = os.path.getsize(csv_path)
    assert reader._offset == size_after_first  # everything consumed

    with open(csv_path, "a") as f:
        f.write(csv_row(3, GAUGES, THERMOCOUPLES) + "\n")

    assert reader.poll() == 1  # exactly one new row
    # Offset advanced by exactly the appended bytes: nothing was re-read
    assert reader._offset == os.path.getsize(csv_path)
    assert reader._offset > size_after_first
    assert reader.row_count == 4

    # A poll with no new data reads nothing
    assert reader.poll() == 0
    assert reader._offset == os.path.getsize(csv_path)


def test_reader_buffers_partial_line(tmp_path):
    """A half-written row is buffered and completed by a later poll."""
    run_dir = make_run(tmp_path, n_rows=2)
    csv_path = os.path.join(run_dir, "shield_data.csv")
    reader = IncrementalRunReader(run_dir)
    reader.poll()

    full_row = csv_row(2, GAUGES, THERMOCOUPLES)
    half = len(full_row) // 2
    with open(csv_path, "a") as f:
        f.write(full_row[:half])

    assert reader.poll() == 0  # incomplete row is not consumed
    assert reader.row_count == 2

    with open(csv_path, "a") as f:
        f.write(full_row[half:] + "\n")

    assert reader.poll() == 1
    assert reader.row_count == 3
    # The stitched row parsed correctly
    assert reader.data["WGM701_Voltage (V)"][-1] == 5.0


def test_reader_handles_missing_csv(tmp_path):
    """Polling before the CSV exists returns 0 rows without raising."""
    run_dir = str(tmp_path / "26.08.04" / "run_1_10h00")
    os.makedirs(run_dir)
    reader = IncrementalRunReader(run_dir)

    assert reader.poll() == 0
    assert reader.row_count == 0


def test_reader_handles_run_dir_disappearing(tmp_path):
    """Polling after the run directory is deleted returns 0 rows."""
    run_dir = make_run(tmp_path, n_rows=2)
    reader = IncrementalRunReader(run_dir)
    reader.poll()

    shutil.rmtree(run_dir)

    assert reader.poll() == 0
    assert reader.row_count == 2  # previously read data is retained


def test_reader_skips_malformed_rows(tmp_path):
    """Rows with the wrong number of fields are skipped."""
    run_dir = make_run(tmp_path, n_rows=1)
    csv_path = os.path.join(run_dir, "shield_data.csv")
    reader = IncrementalRunReader(run_dir)
    reader.poll()

    with open(csv_path, "a") as f:
        f.write("garbage,line\n")
        f.write(csv_row(1, GAUGES, THERMOCOUPLES) + "\n")

    assert reader.poll() == 1
    assert reader.row_count == 2


# =============================================================================
# Tests for _decimation_indices
# =============================================================================


def test_decimation_keeps_all_points_when_under_cap():
    """No decimation happens when n <= max_points."""
    indices = _decimation_indices(100, 500)

    assert len(indices) == 100
    assert list(indices) == list(range(100))


@pytest.mark.parametrize("n_points", [501, 1000, 1234, 9999, 100000])
def test_decimation_respects_cap(n_points):
    """Decimated traces never exceed max_points."""
    indices = _decimation_indices(n_points, 500)

    assert len(indices) <= 500


@pytest.mark.parametrize("n_points", [1, 499, 500, 501, 1234, 100000])
def test_decimation_always_keeps_last_point(n_points):
    """The most recent point is always retained."""
    indices = _decimation_indices(n_points, 500)

    assert indices[-1] == n_points - 1
    assert list(indices) == sorted(set(indices))  # strictly increasing


# =============================================================================
# Tests for build_traces
# =============================================================================


def loaded_reader(tmp_path, **kwargs):
    """Create a run, attach a reader and load it."""
    run_dir = make_run(tmp_path, **kwargs)
    reader = IncrementalRunReader(run_dir)
    reader.poll()
    metadata = json.load(open(os.path.join(run_dir, "run_metadata.json")))
    return reader, metadata


def trace_names_by_row(fig):
    """Map subplot row (via yaxis) to trace names."""
    mapping = {}
    for trace in fig.data:
        mapping.setdefault(trace.yaxis, []).append(trace.name)
    return mapping


def test_build_traces_splits_gauges_by_location(tmp_path):
    """Upstream gauges land in panel 1, downstream in panel 2."""
    reader, metadata = loaded_reader(tmp_path)
    fig = build_traces(reader, metadata, max_points=500)

    by_row = trace_names_by_row(fig)
    assert sorted(by_row["y"]) == ["Baratron626D_1KT", "CVM211"]
    assert sorted(by_row["y2"]) == ["Baratron626D_1T", "WGM701"]
    assert by_row["y3"] == ["TC1"]


def test_build_traces_converts_voltages_per_gauge_type(tmp_path):
    """Each gauge type uses its own voltage-to-pressure conversion."""
    reader, metadata = loaded_reader(tmp_path)
    fig = build_traces(reader, metadata, max_points=500)

    traces = {t.name: t for t in fig.data}
    # Baratron (linear): 5 V on a 1000 torr full scale -> 500 torr
    assert traces["Baratron626D_1KT"].y[0] == pytest.approx(500.0)
    # Baratron 1 torr full scale (linear): 5 V * (1/10) = 0.5 torr
    assert traces["Baratron626D_1T"].y[0] == pytest.approx(0.5)
    # CVM211 (log): 10^(5 - 5) = 1 torr
    assert traces["CVM211"].y[0] == pytest.approx(1.0)
    # WGM701 (log): 10^((5 - 5.5)/0.5) = 0.1 torr
    assert traces["WGM701"].y[0] == pytest.approx(0.1)


def test_build_traces_pressure_axes_are_log_torr(tmp_path):
    """Pressure panels use log axes labelled in torr."""
    reader, metadata = loaded_reader(tmp_path)
    fig = build_traces(reader, metadata, max_points=500)

    assert fig.layout.yaxis.type == "log"
    assert fig.layout.yaxis.title.text == "Pressure (torr)"
    assert fig.layout.yaxis2.type == "log"
    assert fig.layout.yaxis2.title.text == "Pressure (torr)"
    assert fig.layout.yaxis3.title.text == "Temperature (°C)"


def test_build_traces_temperature_uses_cold_junction_conversion(tmp_path):
    """The temperature trace comes from voltage_to_temperature."""
    from shield_das.analysis import voltage_to_temperature

    reader, metadata = loaded_reader(tmp_path)
    fig = build_traces(reader, metadata, max_points=500)

    expected = voltage_to_temperature(np.array([25.0]), np.array([1.0]))[0]
    traces = {t.name: t for t in fig.data}
    assert traces["TC1"].y[0] == pytest.approx(expected)


def test_build_traces_no_thermocouple_annotation(tmp_path):
    """A run without thermocouples gets an annotation, not a trace."""
    reader, metadata = loaded_reader(tmp_path, thermocouples=[])
    fig = build_traces(reader, metadata, max_points=500)

    assert "y3" not in trace_names_by_row(fig)
    annotations = [a.text for a in fig.layout.annotations]
    assert "no thermocouple in this run" in annotations


def test_build_traces_unknown_gauge_type_falls_back_to_raw_volts(tmp_path):
    """An unknown gauge type is plotted as raw volts with a labelled axis."""
    gauges = [{"name": "Mystery", "type": "FutureGauge", "gauge_location": "upstream"}]
    reader, metadata = loaded_reader(tmp_path, gauges=gauges, thermocouples=[])
    fig = build_traces(reader, metadata, max_points=500)

    traces = {t.name: t for t in fig.data}
    assert traces["Mystery (raw V)"].y[0] == pytest.approx(5.0)
    assert fig.layout.yaxis.title.text == "Voltage (V)"
    assert fig.layout.yaxis.type != "log"


def test_build_traces_decimates_to_max_points(tmp_path):
    """Traces are decimated to the cap and keep the newest sample."""
    reader, metadata = loaded_reader(tmp_path, n_rows=250)
    last_timestamp = reader.data["RealTimestamp"][-1]
    fig = build_traces(reader, metadata, max_points=100)

    for trace in fig.data:
        assert len(trace.y) <= 100
        assert trace.x[-1] == last_timestamp


# =============================================================================
# Tests for the interval callback logic (update_dashboard called directly)
# =============================================================================


def test_update_dashboard_waiting_when_no_run(tmp_path):
    """With no active run the dashboard reports WAITING."""
    state = DashboardState()
    config = LiveDashboardConfig(results_dir=str(tmp_path))

    _figure, label, badge, colour, _elapsed, _rows = update_dashboard(
        state, config, max_points=500
    )

    assert badge == "WAITING"
    assert colour == "secondary"
    assert state.reader is None
    assert "Waiting" in label


def test_update_dashboard_attaches_to_live_run(tmp_path):
    """A live run is discovered and reported as LIVE with its data loaded."""
    run_dir = make_run(tmp_path, n_rows=6)
    state = DashboardState()
    config = LiveDashboardConfig(results_dir=str(tmp_path))

    figure, label, badge, colour, _elapsed, rows = update_dashboard(
        state, config, max_points=500
    )

    assert badge == "LIVE"
    assert colour == "success"
    assert state.run_dir == run_dir
    assert label == "26.08.04/run_1_10h00"
    assert rows == "6 rows"
    assert len(figure.data) > 0


def test_update_dashboard_marks_run_ended_on_end_time(tmp_path):
    """When the run gains an end_time the badge switches to ENDED."""
    run_dir = make_run(tmp_path)
    state = DashboardState()
    config = LiveDashboardConfig(results_dir=str(tmp_path))
    update_dashboard(state, config, max_points=500)

    metadata_path = os.path.join(run_dir, "run_metadata.json")
    metadata = json.load(open(metadata_path))
    metadata["run_info"]["end_time"] = "2026-08-04 11:00:00"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    _figure, label, badge, colour, _elapsed, _rows = update_dashboard(
        state, config, max_points=500
    )

    assert badge == "ENDED"
    assert colour == "dark"
    assert label == "26.08.04/run_1_10h00"  # last run still shown


def test_update_dashboard_marks_run_ended_when_stale(tmp_path):
    """A run whose CSV stops updating goes ENDED after staleness_seconds."""
    run_dir = make_run(tmp_path)
    state = DashboardState()
    config = LiveDashboardConfig(results_dir=str(tmp_path), staleness_seconds=120)
    update_dashboard(state, config, max_points=500)

    make_stale(run_dir, age_seconds=3600)

    _figure, _label, badge, colour, _elapsed, _rows = update_dashboard(
        state, config, max_points=500
    )

    assert badge == "ENDED"
    assert colour == "dark"


def test_update_dashboard_switches_to_newer_run_after_end(tmp_path):
    """After the current run ends, a newer active run is picked up."""
    make_run(tmp_path, run_name="run_1_09h00", end_time="2026-08-04 09:30:00")
    first = make_run(tmp_path, run_name="run_2_10h00")
    state = DashboardState()
    config = LiveDashboardConfig(results_dir=str(tmp_path))
    update_dashboard(state, config, max_points=500)
    assert state.run_dir == first

    # End the current run and start a newer one
    metadata_path = os.path.join(first, "run_metadata.json")
    metadata = json.load(open(metadata_path))
    metadata["run_info"]["end_time"] = "2026-08-04 10:30:00"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    newer = make_run(tmp_path, run_name="run_3_11h00", n_rows=2)

    _figure, label, badge, _colour, _elapsed, rows = update_dashboard(
        state, config, max_points=500
    )

    assert state.run_dir == newer
    assert badge == "LIVE"
    assert label == "26.08.04/run_3_11h00"
    assert rows == "2 rows"


def test_update_dashboard_polls_new_rows_between_ticks(tmp_path):
    """New CSV rows written between ticks appear in the next refresh."""
    run_dir = make_run(tmp_path, n_rows=3)
    state = DashboardState()
    config = LiveDashboardConfig(results_dir=str(tmp_path))
    update_dashboard(state, config, max_points=500)

    with open(os.path.join(run_dir, "shield_data.csv"), "a") as f:
        f.write(csv_row(3, GAUGES, THERMOCOUPLES) + "\n")

    *_first, rows = update_dashboard(state, config, max_points=500)

    assert rows == "4 rows"


# =============================================================================
# Tests for the leak-test header readout
# =============================================================================


def make_leak_run(results_dir, voltages, run_info_extra=None, run_name="run_1_10h00"):
    """Create a leak-test run whose gauge voltages vary per row.

    Args:
        results_dir: Root results directory.
        voltages: Per-row gauge voltage in volts (applied to every gauge).
        run_info_extra: Extra run_info keys merged over the defaults.
        run_name: Name of the run directory.

    Returns:
        Path to the created run directory.
    """
    extra = {"run_type": "leak_test", "sample_id": "SAMPLE-001"}
    extra.update(run_info_extra or {})
    run_dir = make_run(results_dir, run_name=run_name, run_info_extra=extra)

    lines = [csv_header(GAUGES, THERMOCOUPLES)]
    lines += [
        csv_row(i, GAUGES, THERMOCOUPLES, voltage_v=v) for i, v in enumerate(voltages)
    ]
    with open(os.path.join(run_dir, "shield_data.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return run_dir


def attached_state(results_dir):
    """Attach a DashboardState to whatever run is live in results_dir."""
    state = DashboardState()
    config = LiveDashboardConfig(results_dir=str(results_dir))
    update_dashboard(state, config, max_points=500)
    return state


def test_leak_readout_hidden_for_normal_run(tmp_path):
    """A non-leak-test run hides the leak badge and shows no rate."""
    make_run(tmp_path)
    state = attached_state(tmp_path)

    badge_text, badge_style, rate_text = _leak_test_readout(state)

    assert badge_text == ""
    assert badge_style == {"display": "none"}
    assert rate_text == ""


def test_leak_readout_hidden_when_no_run(tmp_path):
    """With nothing attached the leak badge stays hidden."""
    state = DashboardState()

    assert _leak_test_readout(state) == ("", {"display": "none"}, "")


def test_leak_readout_badge_shows_sample_id(tmp_path):
    """A leak-test run shows a visible badge with its sample_id."""
    make_leak_run(tmp_path, voltages=[1.0 + 0.1 * i for i in range(10)])
    state = attached_state(tmp_path)

    badge_text, badge_style, rate_text = _leak_test_readout(state)

    assert badge_text == "LEAK TEST — SAMPLE-001"
    assert badge_style == {}
    assert rate_text.startswith("Leak rate: ")
    assert "Torr/s" in rate_text and "Torr/h" in rate_text


def test_leak_rate_fits_downstream_pressure_slope(tmp_path):
    """The fitted rate matches the known slope of the 1-torr Baratron trace.

    Rows are 0.5 s apart and the voltage rises 0.1 V per row; on the 1-torr
    full-scale head that is 0.01 torr per row, i.e. 0.02 torr/s.
    """
    make_leak_run(tmp_path, voltages=[1.0 + 0.1 * i for i in range(10)])
    state = attached_state(tmp_path)

    rate = _leak_rate_torr_per_s(state.reader, state.metadata)

    assert rate == pytest.approx(0.02, rel=1e-6)


def test_leak_rate_window_starts_at_downstream_isolated_time(tmp_path):
    """Samples before downstream_isolated_time are excluded from the fit.

    The first five rows are flat; the rate must come from the rising rows
    after the isolation event only (0.02 torr per 0.5 s row = 0.04 torr/s).
    """
    voltages = [2.0] * 5 + [2.0 + 0.2 * i for i in range(5)]
    isolated = datetime(2026, 8, 4, 10, 0, 0) + timedelta(seconds=2.5)
    make_leak_run(
        tmp_path,
        voltages=voltages,
        run_info_extra={
            "downstream_isolated_time": isolated.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        },
    )
    state = attached_state(tmp_path)

    rate = _leak_rate_torr_per_s(state.reader, state.metadata)

    assert rate == pytest.approx(0.04, rel=1e-6)


def test_leak_rate_none_with_too_few_samples(tmp_path):
    """Fewer than 2 usable samples yields no rate and an em-dash readout."""
    make_leak_run(tmp_path, voltages=[1.0])
    state = attached_state(tmp_path)

    assert _leak_rate_torr_per_s(state.reader, state.metadata) is None
    _badge_text, _badge_style, rate_text = _leak_test_readout(state)
    assert rate_text == "Leak rate: —"


def test_leak_rate_none_without_downstream_baratron(tmp_path):
    """A run with no downstream Baratron gauge yields no rate."""
    make_leak_run(tmp_path, voltages=[1.0 + 0.1 * i for i in range(5)])
    state = attached_state(tmp_path)
    metadata = dict(state.metadata)
    metadata["gauges"] = [g for g in GAUGES if g["name"] != "Baratron626D_1T"]

    assert _leak_rate_torr_per_s(state.reader, metadata) is None
