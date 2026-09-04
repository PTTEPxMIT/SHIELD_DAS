// SHIELD live viewer: polls the Supabase mirror with the read-only anon key.
//
// Two modes, picked automatically:
//   * a run is recording -> the same three stacked panels as the on-rig
//     dashboard (upstream torr log-y, downstream torr linear-y fixed to
//     0-1 torr with the WGM701 hidden, temperature degC), filled by the
//     publisher (shield-das-publish);
//   * no run is recording -> the standby card: the rig's present vacuum level
//     and the last minute of it, filled by the beacon (shield-das-beacon).
//
// Plain fetch() against PostgREST -- no client library, no realtime quota.
// See docs/live_supabase.md.

"use strict";

const POLL_MS = 10000;
const STANDBY_POLL_MS = 5000; // matches the beacon's push cadence
const BACKFILL_POINTS = 4000;
const REDECIMATE_ABOVE = 8000;
const LIVE_WINDOW_MS = 120000;
// The beacon pushes every 5 s; allow a wide margin before calling it stale.
const STANDBY_FRESH_MS = 60000;
// Downstream panel: linear axis pinned to the 1-torr Baratron's range; the
// wide-range WGM701 is omitted there (twin of live_dashboard.py's constants).
const DOWNSTREAM_RANGE_TORR = [0, 1];
const DOWNSTREAM_HIDDEN_GAUGE_TYPES = new Set(["WGM701_Gauge"]);

// -- pure helpers (kept dependency-free for easy eyeballing/testing) ---------

// Stride decimation that always keeps the newest point (twin of
// run_monitor._decimation_indices).
function decimate(rows, maxPoints) {
  if (rows.length <= maxPoints) return rows;
  const stride = Math.ceil(rows.length / maxPoints);
  const kept = [];
  for (let i = rows.length - 1; i >= 0; i -= stride) kept.push(rows[i]);
  return kept.reverse();
}

// Run status from the server-stamped liveness fields.
function statusFor(run, nowMs) {
  if (!run) return "waiting";
  if (run.ended_at) return "ended";
  const seen = Date.parse(run.last_seen_at);
  return nowMs - seen <= LIVE_WINDOW_MS ? "live" : "stale";
}

// Channel plan from run metadata: which mirror channels go on which panel.
// Mirrors build_traces: panel by gauge_location, never hardcoded names.
function channelPlan(metadata) {
  const panels = { upstream: 1, downstream: 2 };
  const plan = [];
  for (const gauge of metadata.gauges || []) {
    const panel = panels[gauge.gauge_location];
    if (!panel) continue;
    if (panel === 2 && DOWNSTREAM_HIDDEN_GAUGE_TYPES.has(gauge.type)) continue;
    // The publisher stores converted pressure under the gauge name, or raw
    // volts under <name>_V for gauge types it cannot convert.
    plan.push({ key: gauge.name, fallbackKey: `${gauge.name}_V`, panel });
  }
  for (const tc of metadata.thermocouples || []) {
    plan.push({ key: `${tc.name}_C`, fallbackKey: null, panel: 3 });
  }
  return plan;
}

// Pressure in torr as a readable magnitude: "1.93 x 10^-4 torr" for the
// decades a vacuum gauge lives in, plain digits near atmosphere.
const SUPERSCRIPTS = {
  "-": "⁻",
  0: "⁰",
  1: "¹",
  2: "²",
  3: "³",
  4: "⁴",
  5: "⁵",
  6: "⁶",
  7: "⁷",
  8: "⁸",
  9: "⁹",
};

function formatTorr(value) {
  if (typeof value !== "number" || !isFinite(value)) return "—";
  if (value === 0) return "0 torr";
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  if (exponent >= -2 && exponent < 4) return `${value.toPrecision(3)} torr`;
  const mantissa = value / 10 ** exponent;
  const superscript = String(exponent)
    .split("")
    .map((character) => SUPERSCRIPTS[character])
    .join("");
  return `${mantissa.toFixed(2)} × 10${superscript} torr`;
}

// How long ago the beacon last reported, in words.
function formatAge(milliseconds) {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  if (seconds < 60) return `${seconds} s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  return `${Math.round(minutes / 60)} h ago`;
}

function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${Math.floor(total / 3600)}:${m}:${s}`;
}

function sampleLine(metadata) {
  const info = (metadata && metadata.run_info) || {};
  const parts = [];
  if (info.sample_substrate) parts.push(`substrate: ${info.sample_substrate}`);
  if (info.sample_coating) parts.push(`coating: ${info.sample_coating}`);
  if (info.furnace_setpoint) parts.push(`furnace: ${info.furnace_setpoint} K`);
  return parts.join(" · ");
}

// -- theme -------------------------------------------------------------------

function themeTokens() {
  const styles = getComputedStyle(document.documentElement);
  const token = (name) => styles.getPropertyValue(name).trim();
  return {
    surface: token("--surface-1"),
    ink: token("--text-primary"),
    muted: token("--text-muted"),
    grid: token("--gridline"),
    baseline: token("--baseline"),
    series: [1, 2, 3, 4, 5, 6].map((n) => token(`--series-${n}`)),
  };
}

// -- PostgREST access --------------------------------------------------------

const config = window.SHIELD_LIVE_CONFIG || {};

async function api(path, options = {}) {
  const response = await fetch(`${config.supabaseUrl}/rest/v1${path}`, {
    ...options,
    headers: {
      apikey: config.supabaseAnonKey,
      Authorization: `Bearer ${config.supabaseAnonKey}`,
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
  return response.json();
}

const fetchNewestRun = async () =>
  (await api("/runs?order=started_at.desc&limit=1"))[0] || null;

// The beacon's single row: present values plus the recent history window.
const fetchStandby = async () =>
  (await api("/standby?id=eq.1&select=updated_at,data"))[0] || null;

const fetchBackfill = (runId) =>
  api("/rpc/decimated_readings", {
    method: "POST",
    body: JSON.stringify({ p_run_id: runId, p_max_points: BACKFILL_POINTS }),
  });

const fetchSince = (runId, lastTs) =>
  api(
    `/readings?run_id=eq.${runId}&ts=gt.${encodeURIComponent(lastTs)}` +
      "&order=ts.asc&select=id,run_id,ts,data",
  );

// -- rendering ---------------------------------------------------------------

const el = (id) => document.getElementById(id);
const state = {
  run: null,
  rows: [],
  plotted: false,
  standby: null,
  showingStandby: false,
};

function buildFigure(run, rows, tokens) {
  const plan = channelPlan(run.metadata || {});
  const x = rows.map((row) => row.ts);
  const traces = [];
  const panelHasData = { 1: false, 2: false, 3: false };
  const panelIsRawVolts = { 1: true, 2: true };

  plan.forEach((entry, index) => {
    const usesFallback =
      rows.length > 0 &&
      !(entry.key in rows[rows.length - 1].data) &&
      entry.fallbackKey &&
      entry.fallbackKey in rows[rows.length - 1].data;
    const key = usesFallback ? entry.fallbackKey : entry.key;
    const y = rows.map((row) => (key in row.data ? row.data[key] : null));
    if (!y.some((value) => value !== null)) return;

    panelHasData[entry.panel] = true;
    if (entry.panel !== 3 && !usesFallback) panelIsRawVolts[entry.panel] = false;
    const axis = entry.panel === 1 ? "y" : `y${entry.panel}`;
    traces.push({
      type: "scattergl",
      mode: "lines",
      name: usesFallback ? `${entry.key} (raw V)` : key,
      x,
      y,
      yaxis: axis,
      line: { width: 2, color: tokens.series[index % tokens.series.length] },
      connectgaps: false,
    });
  });

  const axisBase = {
    gridcolor: tokens.grid,
    linecolor: tokens.baseline,
    tickcolor: tokens.baseline,
    tickfont: { color: tokens.muted, size: 11 },
    zeroline: false,
  };
  const pressureAxis = (panel) => {
    const inTorr = panelHasData[panel] && !panelIsRawVolts[panel];
    const axis = {
      ...axisBase,
      type: inTorr && panel === 1 ? "log" : "linear",
      title: {
        text:
          panelHasData[panel] && panelIsRawVolts[panel]
            ? "Voltage (V)"
            : "Pressure (torr)",
        font: { color: tokens.muted, size: 12 },
      },
    };
    if (inTorr && panel === 2) axis.range = DOWNSTREAM_RANGE_TORR;
    return axis;
  };

  const layout = {
    uirevision: run.run_key, // keep zoom/pan across refreshes
    paper_bgcolor: tokens.surface,
    plot_bgcolor: tokens.surface,
    font: { color: tokens.ink, family: "system-ui, sans-serif" },
    margin: { l: 65, r: 20, t: 30, b: 45 },
    hovermode: "x unified",
    showlegend: true,
    legend: { orientation: "h", y: 1.05, font: { color: tokens.ink } },
    annotations: [
      panelTitle("Upstream pressure", 0.995, tokens),
      panelTitle("Downstream pressure", 0.64, tokens),
      panelTitle("Temperature", 0.285, tokens),
    ],
  };
  // Three stacked panels sharing one time axis (anchored to the bottom panel)
  layout.xaxis = {
    ...axisBase,
    anchor: "y3",
    title: { text: "Time", font: { color: tokens.muted, size: 12 } },
  };
  layout.yaxis = { ...pressureAxis(1), domain: [0.72, 0.98] };
  layout.yaxis2 = { ...pressureAxis(2), domain: [0.37, 0.63] };
  layout.yaxis3 = {
    ...axisBase,
    domain: [0.02, 0.28],
    title: { text: "Temperature (°C)", font: { color: tokens.muted, size: 12 } },
  };

  if (!panelHasData[3]) {
    layout.annotations.push({
      text: "no thermocouple in this run",
      xref: "paper",
      yref: "paper",
      x: 0.5,
      y: 0.14,
      showarrow: false,
      font: { color: tokens.muted },
    });
  }
  return { traces, layout };
}

function panelTitle(text, y, tokens) {
  return {
    text,
    xref: "paper",
    yref: "paper",
    x: 0,
    y,
    xanchor: "left",
    yanchor: "bottom",
    showarrow: false,
    font: { color: tokens.ink, size: 13 },
  };
}

// The last minute of the primary gauge, as a bare sparkline. Log y unless a
// reading has hit the gauge's zero floor, which log cannot draw.
function renderSparkline(tokens) {
  const history = state.standby.data.history || [];
  const chart = el("standby-chart");
  if (history.length < 2) {
    chart.hidden = true;
    return;
  }
  chart.hidden = false;

  const latest = history[history.length - 1][0];
  const values = history.map((point) => point[1]);
  const positive = values.every((value) => value > 0);

  Plotly.react(
    chart,
    [
      {
        type: "scatter",
        mode: "lines",
        x: history.map((point) => point[0] - latest),
        y: values,
        line: { width: 2, color: tokens.series[0] },
        hovertemplate: "%{y:.3e} torr, %{x:.0f} s<extra></extra>",
      },
    ],
    {
      paper_bgcolor: tokens.surface,
      plot_bgcolor: tokens.surface,
      font: { color: tokens.ink, family: "system-ui, sans-serif" },
      margin: { l: 62, r: 10, t: 6, b: 34 },
      showlegend: false,
      xaxis: {
        gridcolor: tokens.grid,
        linecolor: tokens.baseline,
        tickcolor: tokens.baseline,
        tickfont: { color: tokens.muted, size: 11 },
        zeroline: false,
        title: {
          text: "seconds ago",
          font: { color: tokens.muted, size: 11 },
        },
      },
      yaxis: {
        type: positive ? "log" : "linear",
        gridcolor: tokens.grid,
        linecolor: tokens.baseline,
        tickcolor: tokens.baseline,
        tickfont: { color: tokens.muted, size: 11 },
        zeroline: false,
        title: { text: "torr", font: { color: tokens.muted, size: 11 } },
      },
    },
    { responsive: true, displaylogo: false, displayModeBar: false },
  );
}

// The rig is idle: show what the vacuum is doing right now instead of an
// empty "waiting for a run" page.
function renderStandby(tokens, nowMs) {
  const { data, updated_at: updatedAt } = state.standby;
  const age = nowMs - Date.parse(updatedAt);
  const fresh = age <= STANDBY_FRESH_MS;
  const primary = data.primary || "WGM701";
  const channels = data.channels || {};

  const badge = el("status-badge");
  badge.textContent = fresh ? "STANDBY" : "STALE";
  badge.className = fresh ? "standby" : "stale";

  el("run-id").textContent = "no run recording";
  el("elapsed").textContent = "";
  el("row-count").textContent = "";
  el("sample-info").textContent = "";

  el("standby-primary").textContent = primary;
  el("standby-value").textContent = formatTorr(channels[primary]);
  el("standby-age").textContent = fresh
    ? `updated ${formatAge(age)}`
    : `no reading for ${formatAge(age)} — is the beacon running on the rig?`;

  el("standby-others").innerHTML = Object.entries(channels)
    .filter(([name]) => name !== primary)
    .map(
      ([name, value]) =>
        `<span><b>${name}</b> ${formatTorr(value)}</span>`,
    )
    .join("");

  el("message").hidden = true;
  el("chart").hidden = true;
  el("standby").hidden = false;
  renderSparkline(tokens);
}

function render() {
  const tokens = themeTokens();
  const nowMs = Date.now();
  const status = statusFor(state.run, nowMs);

  // A live run owns the page; anything else defers to the beacon when it has
  // something to say (a stale or ended run still leaves the rig pumping).
  state.showingStandby = status !== "live" && state.standby !== null;
  if (state.showingStandby) {
    renderStandby(tokens, nowMs);
    return;
  }
  el("standby").hidden = true;

  const badge = el("status-badge");
  badge.textContent = status.toUpperCase();
  badge.className = status;

  if (!state.run) {
    el("run-id").textContent = "—";
    el("message").textContent = "Waiting for a run to start…";
    el("message").hidden = false;
    el("chart").hidden = true;
    return;
  }

  el("run-id").textContent = state.run.run_key;
  el("sample-info").textContent = sampleLine(state.run.metadata);
  el("row-count").textContent = `${state.rows.length} points`;
  if (state.rows.length >= 2) {
    const first = Date.parse(state.rows[0].ts);
    const last = Date.parse(state.rows[state.rows.length - 1].ts);
    el("elapsed").textContent = formatElapsed((last - first) / 1000);
  }

  if (state.rows.length === 0) {
    el("message").textContent = "Run registered — waiting for data…";
    el("message").hidden = false;
    el("chart").hidden = true;
    return;
  }

  el("message").hidden = true;
  el("chart").hidden = false;
  const { traces, layout } = buildFigure(state.run, state.rows, tokens);
  Plotly.react("chart", traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  });
  state.plotted = true;
}

// -- polling loop ------------------------------------------------------------

async function refresh() {
  const [run, standby] = await Promise.all([
    fetchNewestRun(),
    fetchStandby().catch(() => null), // an un-migrated project has no standby table
  ]);
  state.standby = standby;

  if (!run) {
    state.run = null;
    state.rows = [];
    render();
    return;
  }

  if (!state.run || state.run.id !== run.id) {
    state.run = run;
    state.rows = await fetchBackfill(run.id);
  } else {
    state.run = run;
    if (state.rows.length > 0) {
      const lastTs = state.rows[state.rows.length - 1].ts;
      state.rows.push(...(await fetchSince(run.id, lastTs)));
    } else {
      state.rows = await fetchBackfill(run.id);
    }
    if (state.rows.length > REDECIMATE_ABOVE) {
      state.rows = decimate(state.rows, BACKFILL_POINTS);
    }
  }
  render();
}

async function tick() {
  try {
    await refresh();
  } catch (error) {
    console.error("[shield-live]", error);
    el("message").textContent =
      `Cannot reach the live mirror (${error.message}). ` +
      "If no campaign is running the Supabase project may be paused.";
    el("message").hidden = false;
  }
}

function start() {
  if (!config.supabaseUrl || !config.supabaseAnonKey) {
    el("message").textContent =
      "Not configured: fill in site/config.js with the Supabase project URL " +
      "and anon key (docs/live_supabase.md).";
    return;
  }
  const loop = async () => {
    await tick();
    // The standby card shows a 60 s window, so it earns a faster poll than
    // the run view, whose points arrive every 5 s anyway.
    setTimeout(loop, state.showingStandby ? STANDBY_POLL_MS : POLL_MS);
  };
  loop();
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener(
      "change",
      () => (state.plotted || state.showingStandby) && render(),
    );
}

start();
