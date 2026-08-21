"""Live run dashboard for the SHIELD rig.

Serves a small Dash app (``shield-das-live``) that automatically finds the run
currently being recorded under ``results/``, tails its ``shield_data.csv``
incrementally (never re-reading the whole file), and shows three stacked plots
sharing a time axis: upstream pressure (torr, log scale), downstream pressure
(torr, log scale), and temperature (°C). Voltage-to-pressure conversion reuses
the existing per-gauge functions; browser payload is capped by stride-based
decimation so remote viewing over a slow link stays cheap regardless of run
length. See ``docs/live_dashboard.md``.
"""

import argparse
import os
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from .analysis import voltage_to_temperature

# Re-exported for backwards compatibility: these previously lived here and
# are imported from this module by tests and __init__.
from .run_monitor import (  # noqa: F401
    DATA_CSV_NAME,
    LOCAL_TEMPERATURE_COLUMN,
    METADATA_NAME,
    TIMESTAMP_COLUMN,
    IncrementalRunReader,
    _convert_gauge_voltage,
    _decimation_indices,
    _parse_timestamp,
    _read_metadata,
    find_active_run,
)

# Fidelity options offered by the max-points dropdown
MAX_POINTS_OPTIONS = [500, 2000, 5000]

# Fallback fit window (seconds) for the leak-rate readout when a leak test
# has no downstream_isolated_time event in its metadata yet
LEAK_RATE_WINDOW_SECONDS = 600.0


def build_traces(
    reader: IncrementalRunReader, metadata: dict, max_points: int
) -> go.Figure:
    """Build the three-panel live figure from a reader's current data.

    Panels (shared time axis): upstream pressure (torr, log y), downstream
    pressure (torr, log y), and temperature (°C, via
    ``analysis.voltage_to_temperature`` with cold-junction compensation).
    Gauge columns are mapped through the run metadata, never hardcoded. Each
    trace is decimated to at most ``max_points`` points (most recent point
    always kept) so the browser payload stays constant as the run grows.
    Gauges whose type has no known conversion are plotted as raw volts and
    labelled accordingly; if a run has no thermocouple the temperature panel
    shows an annotation instead.

    Args:
        reader: Reader holding the run's data read so far.
        metadata: Parsed ``run_metadata.json`` for the run.
        max_points: Maximum plotted points per trace.

    Returns:
        Plotly figure with three stacked subplots sharing the x-axis.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Upstream pressure",
            "Downstream pressure",
            "Temperature",
        ),
    )

    timestamps = reader.data.get(TIMESTAMP_COLUMN, [])
    indices = _decimation_indices(len(timestamps), max_points)
    x_values = [timestamps[i] for i in indices]

    location_rows = {"upstream": 1, "downstream": 2}
    row_units: dict[int, set[str]] = {1: set(), 2: set()}

    for gauge in metadata.get("gauges", []):
        row = location_rows.get(gauge.get("gauge_location"))
        column = f"{gauge.get('name')}_Voltage (V)"
        if row is None or column not in reader.data:
            continue
        voltage_v = np.asarray(reader.data[column], dtype=float)[indices]
        values, unit = _convert_gauge_voltage(gauge, voltage_v)
        row_units[row].add(unit)
        trace_name = gauge["name"] if unit == "torr" else f"{gauge['name']} (raw V)"
        fig.add_trace(
            go.Scattergl(x=x_values, y=values, mode="lines", name=trace_name),
            row=row,
            col=1,
        )

    for row in (1, 2):
        if row_units[row] == {"V"}:
            # Only unconverted gauges in this panel: label as raw volts
            fig.update_yaxes(title_text="Voltage (V)", row=row, col=1)
        else:
            fig.update_yaxes(title_text="Pressure (torr)", type="log", row=row, col=1)

    thermocouples = metadata.get("thermocouples", [])
    local_temp_c = reader.data.get(LOCAL_TEMPERATURE_COLUMN)
    temperature_plotted = False
    if local_temp_c is not None:
        local_array = np.asarray(local_temp_c, dtype=float)[indices]
        for thermocouple in thermocouples:
            column = f"{thermocouple.get('name')}_Voltage (mV)"
            if column not in reader.data:
                continue
            tc_voltage_mv = np.asarray(reader.data[column], dtype=float)[indices]
            temperature_c = voltage_to_temperature(
                local_temperature=local_array, voltage=tc_voltage_mv
            )
            fig.add_trace(
                go.Scattergl(
                    x=x_values,
                    y=temperature_c,
                    mode="lines",
                    name=thermocouple.get("name", "thermocouple"),
                ),
                row=3,
                col=1,
            )
            temperature_plotted = True

    fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
    if not temperature_plotted:
        fig.add_annotation(
            text="no thermocouple in this run",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"color": "#888888"},
            row=3,
            col=1,
        )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=850,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": 1.06},
        uirevision=reader.run_dir,  # keep zoom/pan across interval refreshes
    )
    return fig


@dataclass
class LiveDashboardConfig:
    """Configuration for the live run dashboard.

    Attributes:
        results_dir: Directory containing recorded runs
            (``<date>/<run>/`` subdirectories).
        interval_ms: Refresh interval of the Dash app in milliseconds.
        max_points: Default maximum plotted points per trace (the fidelity
            dropdown can override it in the browser).
        staleness_seconds: A run whose CSV has not been written for this many
            seconds is no longer considered live.
        include_test_runs: Also monitor ``test_run_*`` (recorder test mode)
            directories.
    """

    results_dir: str = "results"
    interval_ms: int = 2000
    max_points: int = 2000
    staleness_seconds: float = 120.0
    include_test_runs: bool = False


@dataclass
class DashboardState:
    """Mutable server-side state of the live dashboard.

    Attributes:
        run_dir: Path of the run currently displayed, or None.
        reader: Incremental reader attached to ``run_dir``, or None.
        metadata: Parsed metadata of the current run.
        status: One of ``"waiting"``, ``"live"`` or ``"ended"``.
    """

    run_dir: str | None = None
    reader: IncrementalRunReader | None = None
    metadata: dict = field(default_factory=dict)
    status: str = "waiting"


def _run_label(run_dir: str) -> str:
    """Short display label for a run directory (``<date>/<run>``).

    Args:
        run_dir: Path to the run directory.

    Returns:
        The last two path components joined with a slash.
    """
    normalised = os.path.normpath(run_dir)
    date_part = os.path.basename(os.path.dirname(normalised))
    return f"{date_part}/{os.path.basename(normalised)}"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as H:MM:SS.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        The formatted duration string.
    """
    total = int(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _waiting_figure(message: str) -> go.Figure:
    """Placeholder figure shown while no run is live.

    Args:
        message: Text displayed in the centre of the empty figure.

    Returns:
        An empty figure with a centred annotation.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 18, "color": "#888888"},
    )
    fig.update_layout(
        template="plotly_white",
        height=850,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _current_run_has_ended(state: DashboardState, config: LiveDashboardConfig) -> bool:
    """Check whether the currently displayed run has finished or gone stale.

    Args:
        state: Current dashboard state.
        config: Dashboard configuration.

    Returns:
        True if the run gained an ``end_time``, its CSV went stale, or the
        run directory disappeared.
    """
    if state.run_dir is None:
        return False
    metadata = _read_metadata(state.run_dir)
    if metadata is None:
        return True  # run directory disappeared
    if metadata.get("run_info", {}).get("end_time"):
        return True
    csv_path = os.path.join(state.run_dir, DATA_CSV_NAME)
    try:
        age = time.time() - os.path.getmtime(csv_path)
    except OSError:
        return True
    return age > config.staleness_seconds


def _attach_run(state: DashboardState, run_dir: str) -> None:
    """Point the dashboard state at a new run.

    Args:
        state: Dashboard state to update in place.
        run_dir: Path of the run to attach.
    """
    state.run_dir = run_dir
    state.reader = IncrementalRunReader(run_dir)
    state.metadata = _read_metadata(run_dir) or {}
    state.status = "live"


def update_dashboard(
    state: DashboardState, config: LiveDashboardConfig, max_points: int
) -> tuple[go.Figure, str, str, str, str, str]:
    """Perform one dashboard refresh tick.

    This is the logic behind the interval callback (kept as a module-level
    function so tests can call it directly). If no run is attached, it looks
    for one; if the attached run gains an ``end_time`` or goes stale it is
    marked ENDED and any newer active run is switched to automatically.

    Args:
        state: Mutable dashboard state, updated in place.
        config: Dashboard configuration.
        max_points: Maximum plotted points per trace for this refresh.

    Returns:
        Tuple of (figure, run label, status text, badge colour, elapsed time
        text, row count text) matching the callback outputs.
    """
    if state.run_dir is not None and _current_run_has_ended(state, config):
        state.status = "ended"

    if state.reader is None or state.status == "ended":
        candidate = find_active_run(
            config.results_dir,
            staleness_seconds=config.staleness_seconds,
            include_test_runs=config.include_test_runs,
        )
        if candidate is not None and candidate != state.run_dir:
            _attach_run(state, candidate)

    if state.reader is None:
        return (
            _waiting_figure("Waiting for run…"),
            "Waiting for run…",
            "WAITING",
            "secondary",
            "-",
            "-",
        )

    state.reader.poll()
    figure = build_traces(state.reader, state.metadata, max_points)

    if state.status == "ended":
        badge_text, badge_colour = "ENDED", "dark"
    else:
        badge_text, badge_colour = "LIVE", "success"

    return (
        figure,
        _run_label(state.run_dir),
        badge_text,
        badge_colour,
        _format_elapsed(state.reader.elapsed_seconds),
        f"{state.reader.row_count} rows",
    )


def _find_downstream_baratron(metadata: dict) -> dict | None:
    """Pick the downstream Baratron gauge for the leak-rate readout.

    Prefers the 1-torr full-scale head (the range leak-test setpoints sit
    in); falls back to any other downstream Baratron626D with a known full
    scale.

    Args:
        metadata: Parsed ``run_metadata.json`` for the run.

    Returns:
        The gauge metadata dict, or None if no convertible downstream
        Baratron is present.
    """
    candidates = [
        gauge
        for gauge in metadata.get("gauges", [])
        if gauge.get("type") == "Baratron626D_Gauge"
        and gauge.get("gauge_location") == "downstream"
        and gauge.get("full_scale_torr")
    ]
    for gauge in candidates:
        if float(gauge["full_scale_torr"]) == 1.0:
            return gauge
    return candidates[0] if candidates else None


def _leak_rate_torr_per_s(reader: IncrementalRunReader, metadata: dict) -> float | None:
    """Fit the downstream leak rate of a leak-test run in torr/s.

    Linear fit (``np.polyfit`` degree 1) of the downstream Baratron pressure
    (torr, converted from raw volts exactly as ``build_traces`` does) against
    time (s). The fit window is every sample after
    ``run_info.downstream_isolated_time`` when that event is in the
    metadata, otherwise the last ``LEAK_RATE_WINDOW_SECONDS`` of data.

    Args:
        reader: Reader holding the run's data read so far.
        metadata: Parsed ``run_metadata.json`` for the run.

    Returns:
        The fitted slope in torr/s, or None when there is no convertible
        downstream Baratron, fewer than 2 usable samples, or no time span
        to fit over.
    """
    gauge = _find_downstream_baratron(metadata)
    if gauge is None:
        return None

    timestamps = reader.data.get(TIMESTAMP_COLUMN, [])
    voltages = reader.data.get(f"{gauge.get('name')}_Voltage (V)", [])
    n_rows = min(len(timestamps), len(voltages))
    if n_rows < 2:
        return None

    isolated_raw = metadata.get("run_info", {}).get("downstream_isolated_time")
    window_start = _parse_timestamp(isolated_raw) if isolated_raw else None
    if window_start is None:
        window_start = timestamps[n_rows - 1] - timedelta(
            seconds=LEAK_RATE_WINDOW_SECONDS
        )

    selected = [i for i in range(n_rows) if timestamps[i] >= window_start]
    if len(selected) < 2:
        return None

    voltage_v = np.asarray([voltages[i] for i in selected], dtype=float)
    pressure_torr, unit = _convert_gauge_voltage(gauge, voltage_v)
    if unit != "torr":
        return None

    first = timestamps[selected[0]]
    time_s = np.array([(timestamps[i] - first).total_seconds() for i in selected])
    if time_s[-1] <= 0:
        return None  # no time span to fit over

    slope_torr_per_s, _intercept = np.polyfit(time_s, pressure_torr, 1)
    return float(slope_torr_per_s)


def _leak_test_readout(state: DashboardState) -> tuple[str, dict, str]:
    """Header badge and live leak-rate text for a leak-test run.

    Args:
        state: Current dashboard state.

    Returns:
        Tuple of (badge text, badge style, leak-rate text). For anything
        other than an attached leak-test run the badge is hidden and both
        texts are empty; a leak test with too little usable data shows
        "Leak rate: —".
    """
    run_info = (state.metadata or {}).get("run_info", {})
    if state.reader is None or run_info.get("run_type") != "leak_test":
        return "", {"display": "none"}, ""

    sample_id = run_info.get("sample_id")
    badge_text = f"LEAK TEST — {sample_id}" if sample_id else "LEAK TEST"

    rate = _leak_rate_torr_per_s(state.reader, state.metadata)
    if rate is None:
        rate_text = "Leak rate: —"
    else:
        rate_text = f"Leak rate: {rate:.2e} Torr/s ({rate * 3600:.2f} Torr/h)"
    return badge_text, {}, rate_text


def _create_layout(config: LiveDashboardConfig) -> dbc.Container:
    """Build the Dash layout for the live dashboard.

    Args:
        config: Dashboard configuration (sets the refresh interval and the
            default fidelity).

    Returns:
        The top-level Bootstrap container.
    """
    max_points_options = sorted({*MAX_POINTS_OPTIONS, config.max_points})
    header = dbc.Row(
        [
            dbc.Col(html.H4("SHIELD live run", className="mb-0"), width="auto"),
            dbc.Col(html.Span(id="live-run-id", className="text-muted"), width="auto"),
            dbc.Col(
                dbc.Badge("WAITING", id="live-status-badge", color="secondary"),
                width="auto",
            ),
            dbc.Col(
                dbc.Badge(
                    "",
                    id="live-leak-badge",
                    color="warning",
                    style={"display": "none"},
                ),
                width="auto",
            ),
            dbc.Col(html.Span(id="live-leak-rate"), width="auto"),
            dbc.Col(html.Span(id="live-elapsed"), width="auto"),
            dbc.Col(html.Span(id="live-row-count"), width="auto"),
            dbc.Col(
                dcc.Dropdown(
                    id="live-max-points",
                    options=[
                        {"label": f"{n} points", "value": n} for n in max_points_options
                    ],
                    value=config.max_points,
                    clearable=False,
                    style={"width": "150px"},
                ),
                width="auto",
                className="ms-auto",
            ),
        ],
        align="center",
        className="g-3 py-2 border-bottom",
    )
    return dbc.Container(
        [
            header,
            dcc.Graph(id="live-graph", figure=_waiting_figure("Waiting for run…")),
            dcc.Interval(id="live-interval", interval=config.interval_ms),
        ],
        fluid=True,
    )


def register_live_dashboard_callbacks(
    app: dash.Dash, config: LiveDashboardConfig, state: DashboardState
) -> None:
    """Register the interval refresh callback on the app.

    Args:
        app: Dash application to register the callback on.
        config: Dashboard configuration.
        state: Shared server-side dashboard state.
    """

    @app.callback(
        [
            Output("live-graph", "figure"),
            Output("live-run-id", "children"),
            Output("live-status-badge", "children"),
            Output("live-status-badge", "color"),
            Output("live-elapsed", "children"),
            Output("live-row-count", "children"),
            Output("live-leak-badge", "children"),
            Output("live-leak-badge", "style"),
            Output("live-leak-rate", "children"),
        ],
        [
            Input("live-interval", "n_intervals"),
            Input("live-max-points", "value"),
        ],
    )
    def refresh_dashboard(n_intervals, max_points):
        """Interval-driven refresh of the whole dashboard.

        Args:
            n_intervals: Number of interval ticks so far (unused trigger).
            max_points: Fidelity dropdown value (max plotted points/trace).

        Returns:
            tuple: (figure, run label, status text, badge colour, elapsed
            time text, row count text, leak badge text, leak badge style,
            leak rate text)
        """
        outputs = update_dashboard(state, config, int(max_points or config.max_points))
        return (*outputs, *_leak_test_readout(state))


def create_app(config: LiveDashboardConfig) -> dash.Dash:
    """Create the live dashboard Dash application.

    Args:
        config: Dashboard configuration.

    Returns:
        The configured Dash app; its shared state is exposed as
        ``app.live_state`` for inspection.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "SHIELD live run"
    state = DashboardState()
    app.layout = _create_layout(config)
    register_live_dashboard_callbacks(app, config, state)
    app.live_state = state
    return app


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the live run dashboard (``shield-das-live``).

    Serves the dashboard on ``--host``/``--port``. The default host
    ``127.0.0.1`` keeps it local to the rig PC — remote viewing goes through
    the Supabase live mirror instead (``shield-das-publish`` and the GitHub
    Pages site; see ``docs/live_supabase.md``). Pass ``--host 0.0.0.0`` to
    deliberately expose it to the local network. The browser is always
    opened toward ``http://127.0.0.1:<port>`` regardless of the bind host.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Process exit code (0 on success, 1 if no run is live and ``--watch``
        was not given).
    """
    parser = argparse.ArgumentParser(
        prog="shield-das-live",
        description="Live dashboard for the run currently being recorded.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing recorded runs (default: results)",
    )
    parser.add_argument(
        "--port", type=int, default=8051, help="Port to serve on (default: 8051)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind (default: 127.0.0.1, local-only; remote "
            "viewing goes through the Supabase live mirror instead)"
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=2000,
        help="Refresh interval in milliseconds (default: 2000)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Default maximum plotted points per trace (default: 2000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a web browser",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Poll every 10 s until an active run appears instead of exiting "
            "when nothing is live"
        ),
    )
    parser.add_argument(
        "--include-test-runs",
        action="store_true",
        help="Also monitor test_run_* (recorder test mode) directories",
    )
    args = parser.parse_args(argv)

    config = LiveDashboardConfig(
        results_dir=args.results_dir,
        interval_ms=args.interval,
        max_points=args.max_points,
        include_test_runs=args.include_test_runs,
    )

    active_run = find_active_run(
        config.results_dir,
        staleness_seconds=config.staleness_seconds,
        include_test_runs=config.include_test_runs,
    )
    if active_run is None:
        if not args.watch:
            print(
                f"No live run found under {config.results_dir!r}. Start the "
                "recorder first, or use --watch to wait for one."
            )
            return 1
        print(f"Watching {config.results_dir!r} for a live run (every 10 s)...")
        while active_run is None:
            time.sleep(10)
            active_run = find_active_run(
                config.results_dir,
                staleness_seconds=config.staleness_seconds,
                include_test_runs=config.include_test_runs,
            )

    print(f"Live run: {active_run}")
    print(f"Serving dashboard on http://{args.host}:{args.port}")

    app = create_app(config)

    if not args.no_browser:
        # Always point the local browser at the loopback address, whatever
        # interface the server binds
        threading.Timer(
            0.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")
        ).start()

    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
