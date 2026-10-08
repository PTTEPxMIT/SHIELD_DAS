// SHIELD live viewer: polls the Supabase mirror with the read-only anon key.
//
// Two modes, picked automatically:
//   * a run is recording -> the on-rig dashboard's three panels plus one:
//     downstream torr (linear-y fixed to 0-1 torr, WGM701 hidden) full-width
//     on top; upstream torr (log-y), temperature degC and the downstream's
//     residual about its steady-state line (Pa) side by side beneath it at a
//     quarter of the height; filled by the publisher (shield-das-publish);
//   * no run is recording -> the standby card: the rig's present vacuum level
//     and the last minute of it, filled by the beacon (shield-das-beacon).
//
// Plain fetch() against PostgREST -- no client library, no realtime quota.
// See docs/live_supabase.md.

"use strict";

const POLL_MS = 10000;
const STANDBY_POLL_MS = 5000; // matches the beacon's push cadence
// The run is fetched as time-bucketed series from the run_series RPC (one
// JSON document, immune to PostgREST's 1000-row cap): SERIES_POINTS buckets
// for the first paint, then only the rows since the last one on each poll.
// Once the accumulated points pass RESERIES_ABOVE the whole series is
// re-fetched so the trace stays at plot resolution however long the run.
const SERIES_POINTS = 1200;
const POLL_POINTS = 500;
const RESERIES_ABOVE = 3000;
const LIVE_WINDOW_MS = 120000;
// The beacon pushes every 5 s; allow a wide margin before calling it stale.
const STANDBY_FRESH_MS = 60000;
// Downstream panel: linear axis pinned to the 1-torr Baratron's range; the
// wide-range WGM701 is omitted there (twin of live_dashboard.py's constants).
const DOWNSTREAM_RANGE_TORR = [0, 1];
const DOWNSTREAM_HIDDEN_GAUGE_TYPES = new Set(["WGM701_Gauge"]);
// Panel geometry (paper fractions): downstream owns the top; upstream,
// temperature and the steady-state residual share the strip beneath it at a
// quarter of its height.
const TOP_DOMAIN = [0.3, 0.98];
const BOTTOM_DOMAIN = [0.03, 0.2];
const BOTTOM_LEFT_X = [0, 0.28];
const BOTTOM_MIDDLE_X = [0.36, 0.64];
const BOTTOM_RIGHT_X = [0.72, 1];
// Trace y-axes stay y/y2/y3/y4 for panels 1/2/3/4; the x-axes are laid out
// so the downstream panel (2) owns the primary one.
const PANEL_X_AXIS = { 1: "x2", 2: "x", 3: "x3", 4: "x4" };
// Residual panel: the toolbox's steady-state fit (shield_toolbox
// analysis.time_lag.fit_steady_state) without the background subtraction,
// which is a straight line and so leaves the residuals unchanged. Constants
// twin the toolbox defaults.
const TORR_TO_PA = 133.322368;
const PRESSURISED_TORR = 10.0;
const DOWNSTREAM_MAX_TORR = 0.95;
const ANALYSIS_HOURS = 30;
const PRE_STEP_WINDOW_S = 60;
const SS_START_TAUS = 3.0;
const SS_MAX_ITER = 20;

// -- pure helpers (kept dependency-free for easy eyeballing/testing) ---------

// Append one run_series result onto another, in place. Channels absent from
// either side are padded with nulls so every array stays as long as ts.
function appendSeries(base, extra) {
  const before = base.ts.length;
  base.ts.push(...extra.ts);
  const keys = new Set([...Object.keys(base.channels), ...Object.keys(extra.channels)]);
  for (const key of keys) {
    const existing = base.channels[key] || new Array(before).fill(null);
    const added = extra.channels[key] || new Array(extra.ts.length).fill(null);
    base.channels[key] = existing.concat(added);
  }
  if (extra.last_ts) base.last_ts = extra.last_ts;
  return base;
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
// The publisher stores converted pressure under the gauge name, or raw volts
// under <name>_V for gauge types it cannot convert.
function channelPlan(metadata) {
  const panels = { upstream: 1, downstream: 2 };
  const plan = [];
  for (const gauge of metadata.gauges || []) {
    const panel = panels[gauge.gauge_location];
    if (!panel) continue;
    if (panel === 2 && DOWNSTREAM_HIDDEN_GAUGE_TYPES.has(gauge.type)) continue;
    plan.push({ key: gauge.name, fallbackKey: `${gauge.name}_V`, panel });
  }
  for (const tc of metadata.thermocouples || []) {
    plan.push({ key: `${tc.name}_C`, fallbackKey: null, panel: 3 });
  }
  return plan;
}

// -- steady-state residual ----------------------------------------------------

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Least-squares line y = slope*x + intercept.
function lineFit(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}

// Downstream minus the steady-state line, as the toolbox's residual plot.
//
// t_init is the first sample above half the upstream plateau (the median of
// the readings above PRESSURISED_TORR). The line is fitted to the downstream
// from SS_START_TAUS·τ_L to the last usable sample, τ_L being where the line
// crosses the pre-step downstream level; τ_L and the window start are
// iterated from a quarter of the span until τ_L moves by less than 1 s.
//
// Args: timesS (s), upstreamTorr and downstreamTorr (torr, null for gaps),
// all the same length.
// Returns null before there is a pressure step and four usable samples,
// else {residualPa (null outside the usable span), tInitS, timeLagS,
// windowStartS (absolute s), windowResidualPa, converged}.
function steadyStateResidual(timesS, upstreamTorr, downstreamTorr) {
  const n = timesS.length;
  const valid = (i) =>
    typeof upstreamTorr[i] === "number" && typeof downstreamTorr[i] === "number";
  const pressurised = upstreamTorr.filter(
    (v) => typeof v === "number" && v > PRESSURISED_TORR,
  );
  if (pressurised.length === 0) return null;
  const half = 0.5 * median(pressurised);
  const initIndex = upstreamTorr.findIndex((v) => typeof v === "number" && v > half);
  const tInit = timesS[initIndex];

  // Level the downstream starts from: the last minute before the step, or
  // the first reading at it when the recording starts pressurised.
  const before = [];
  for (let i = 0; i < initIndex; i++) {
    if (valid(i) && timesS[i] >= tInit - PRE_STEP_WINDOW_S) before.push(downstreamTorr[i]);
  }
  const startLevel = before.length ? median(before) : downstreamTorr[initIndex];
  if (typeof startLevel !== "number") return null;

  const tRel = timesS.map((t) => t - tInit);
  const usable = [];
  for (let i = 0; i < n; i++) {
    if (
      valid(i) &&
      tRel[i] > 0 &&
      tRel[i] <= ANALYSIS_HOURS * 3600 &&
      upstreamTorr[i] > PRESSURISED_TORR &&
      downstreamTorr[i] < DOWNSTREAM_MAX_TORR
    ) {
      usable.push(i);
    }
  }
  if (usable.length < 4) return null;
  const endS = tRel[usable[usable.length - 1]];

  const windowFrom = (startS) => usable.filter((i) => tRel[i] >= startS);
  const fit = (window) => {
    const { slope, intercept } = lineFit(
      window.map((i) => tRel[i]),
      window.map((i) => downstreamTorr[i]),
    );
    return { slope, intercept, tau: (startLevel - intercept) / slope };
  };

  let tau = 0.25 * endS; // first guess
  let window = windowFrom(SS_START_TAUS * tau);
  if (window.length < 4) {
    tau = (0.5 * endS) / SS_START_TAUS;
    window = windowFrom(SS_START_TAUS * tau);
  }
  let line = fit(window);
  let converged = false;
  for (let iter = 1; iter < SS_MAX_ITER; iter++) {
    if (Math.abs(line.tau - tau) < 1) {
      converged = true;
      break;
    }
    tau = Math.max(line.tau, 60);
    const next = windowFrom(SS_START_TAUS * tau);
    if (next.length < 4) break; // next start lies past the data
    window = next;
    line = fit(window);
  }
  if (!converged) converged = Math.abs(line.tau - tau) < 1;

  const residualPa = new Array(n).fill(null);
  for (const i of usable) {
    residualPa[i] =
      (downstreamTorr[i] - (line.intercept + line.slope * tRel[i])) * TORR_TO_PA;
  }
  return {
    residualPa,
    tInitS: tInit,
    timeLagS: line.tau,
    windowStartS: tInit + tRel[window[0]],
    windowResidualPa: window.map((i) => residualPa[i]),
    converged,
  };
}

function formatHours(seconds) {
  return seconds < 3600
    ? `${(seconds / 60).toFixed(0)} min`
    : `${(seconds / 3600).toFixed(1)} h`;
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

// A standby channel in its own units: thermocouples arrive as <name>_C in
// degrees Celsius, raw-volt fallbacks as <name>_V, everything else in torr.
function formatChannel(name, value) {
  if (typeof value !== "number" || !isFinite(value)) return "—";
  if (name.endsWith("_C")) return `${value.toFixed(1)} °C`;
  if (name.endsWith("_V")) return `${value.toPrecision(3)} V`;
  return formatTorr(value);
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

// Bucketed series for one run: {ts: [...], channels: {name: [...]},
// last_ts, points, rows}. With `since`, only rows after that timestamp.
const fetchSeries = (runId, maxPoints, since = null) =>
  api("/rpc/run_series", {
    method: "POST",
    body: JSON.stringify({
      p_run_id: runId,
      p_max_points: maxPoints,
      p_since: since,
    }),
  });

// -- rendering ---------------------------------------------------------------

const el = (id) => document.getElementById(id);
const emptySeries = () => ({ ts: [], channels: {}, last_ts: null });
const state = {
  run: null,
  series: emptySeries(),
  plotted: false,
  standby: null,
  showingStandby: false,
};

function buildFigure(run, series, tokens) {
  const plan = channelPlan(run.metadata || {});
  const x = series.ts;
  const traces = [];
  const panelHasData = { 1: false, 2: false, 3: false };
  const panelIsRawVolts = { 1: true, 2: true };

  plan.forEach((entry, index) => {
    const usesFallback =
      !(entry.key in series.channels) &&
      entry.fallbackKey &&
      entry.fallbackKey in series.channels;
    const key = usesFallback ? entry.fallbackKey : entry.key;
    const y = series.channels[key];
    if (!y || !y.some((value) => value !== null)) return;

    panelHasData[entry.panel] = true;
    if (entry.panel !== 3 && !usesFallback) panelIsRawVolts[entry.panel] = false;
    // Panel 2 (downstream) rides the primary x-axis; the two bottom panels
    // each have their own, matched to it.
    const axis = entry.panel === 1 ? "y" : `y${entry.panel}`;
    const xAxis = PANEL_X_AXIS[entry.panel];
    traces.push({
      type: "scattergl",
      mode: "lines",
      name: usesFallback ? `${entry.key} (raw V)` : key,
      x,
      y,
      xaxis: xAxis,
      yaxis: axis,
      line: { width: 2, color: tokens.series[index % tokens.series.length] },
      connectgaps: false,
    });
  });

  // Residual: the first upstream and downstream gauges reading in torr.
  const torrChannel = (panel) => {
    const entry = plan.find((e) => e.panel === panel && e.key in series.channels);
    return entry ? series.channels[entry.key] : null;
  };
  const upstream = torrChannel(1);
  const downstream = torrChannel(2);
  const residual =
    upstream && downstream
      ? steadyStateResidual(
          x.map((t) => Date.parse(t) / 1000),
          upstream,
          downstream,
        )
      : null;
  if (residual) {
    traces.push({
      type: "scattergl",
      mode: "lines",
      name: "residual",
      showlegend: false,
      x,
      y: residual.residualPa,
      xaxis: "x4",
      yaxis: "y4",
      line: { width: 1.5, color: tokens.ink },
      hovertemplate: "%{y:.3g} Pa<extra>residual</extra>",
      connectgaps: false,
    });
  }

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
      panelTitle("Downstream pressure", TOP_DOMAIN[1] + 0.005, 0, tokens),
      panelTitle("Upstream pressure", BOTTOM_DOMAIN[1] + 0.005, 0, tokens),
      panelTitle("Temperature", BOTTOM_DOMAIN[1] + 0.005, BOTTOM_MIDDLE_X[0], tokens),
      panelTitle("Steady-state residual", BOTTOM_DOMAIN[1] + 0.005, BOTTOM_RIGHT_X[0], tokens),
    ],
    shapes: [],
  };
  const timeTitle = { text: "Time", font: { color: tokens.muted, size: 12 } };
  // Downstream: full-width top panel on the primary time axis
  layout.xaxis = { ...axisBase, anchor: "y2", domain: [0, 1] };
  layout.yaxis2 = { ...pressureAxis(2), anchor: "x", domain: TOP_DOMAIN };
  // Upstream and temperature: side by side beneath, time axes matched to
  // the top one so zooming any panel zooms all three.
  layout.xaxis2 = {
    ...axisBase,
    anchor: "y",
    domain: BOTTOM_LEFT_X,
    matches: "x",
    title: timeTitle,
  };
  layout.yaxis = { ...pressureAxis(1), anchor: "x2", domain: BOTTOM_DOMAIN };
  layout.xaxis3 = {
    ...axisBase,
    anchor: "y3",
    domain: BOTTOM_MIDDLE_X,
    matches: "x",
    title: timeTitle,
  };
  layout.yaxis3 = {
    ...axisBase,
    anchor: "x3",
    domain: BOTTOM_DOMAIN,
    title: { text: "Temperature (°C)", font: { color: tokens.muted, size: 12 } },
  };
  layout.xaxis4 = {
    ...axisBase,
    anchor: "y4",
    domain: BOTTOM_RIGHT_X,
    matches: "x",
    title: timeTitle,
  };
  layout.yaxis4 = {
    ...axisBase,
    anchor: "x4",
    domain: BOTTOM_DOMAIN,
    title: { text: "Residual (Pa)", font: { color: tokens.muted, size: 12 } },
  };

  if (!panelHasData[3]) {
    layout.annotations.push(
      panelNote("no thermocouple in this run", BOTTOM_MIDDLE_X, tokens),
    );
  }
  if (residual) {
    addResidualDecor(layout, residual, x, tokens);
  } else {
    layout.annotations.push(
      panelNote(
        upstream && downstream
          ? "waiting for the upstream step"
          : "needs upstream and downstream in torr",
        BOTTOM_RIGHT_X,
        tokens,
      ),
    );
  }
  return { traces, layout };
}

// Zero line, shaded steady-state window and the τ_L readout; the y-range
// fits the window's residuals so the pre-steady-state rise, often orders of
// magnitude larger, runs off the top instead of flattening the window.
function addResidualDecor(layout, residual, x, tokens) {
  const windowStart = new Date(residual.windowStartS * 1000).toISOString();
  const lastX = x[x.length - 1];
  layout.shapes.push(
    {
      type: "rect",
      xref: "x4",
      yref: "y4 domain",
      x0: windowStart,
      x1: lastX,
      y0: 0,
      y1: 1,
      fillcolor: tokens.muted,
      opacity: 0.12,
      line: { width: 0 },
      layer: "below",
    },
    {
      type: "line",
      xref: "x4 domain",
      yref: "y4",
      x0: 0,
      x1: 1,
      y0: 0,
      y1: 0,
      line: { color: tokens.baseline, width: 1 },
    },
  );
  const extent = Math.max(...residual.windowResidualPa.map(Math.abs));
  if (extent > 0) layout.yaxis4.range = [-2 * extent, 2 * extent];

  const tau = residual.timeLagS;
  const readout =
    (tau > 0 ? `τ<sub>L</sub> ${formatHours(tau)}` : "τ<sub>L</sub> —") +
    ` · window from ${formatHours(residual.windowStartS - residual.tInitS)}` +
    (residual.converged ? "" : " · not converged");
  layout.annotations.push({
    text: readout,
    xref: "paper",
    yref: "paper",
    x: BOTTOM_RIGHT_X[1],
    y: BOTTOM_DOMAIN[1] + 0.005,
    xanchor: "right",
    yanchor: "bottom",
    showarrow: false,
    font: { color: tokens.muted, size: 11 },
  });
}

function panelNote(text, xDomain, tokens) {
  return {
    text,
    xref: "paper",
    yref: "paper",
    x: (xDomain[0] + xDomain[1]) / 2,
    y: (BOTTOM_DOMAIN[0] + BOTTOM_DOMAIN[1]) / 2,
    showarrow: false,
    font: { color: tokens.muted },
  };
}

function panelTitle(text, y, x, tokens) {
  return {
    text,
    xref: "paper",
    yref: "paper",
    x,
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
        `<span><b>${name}</b> ${formatChannel(name, value)}</span>`,
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

  const ts = state.series.ts;
  el("run-id").textContent = state.run.run_key;
  el("sample-info").textContent = sampleLine(state.run.metadata);
  el("row-count").textContent = `${ts.length} points`;
  if (ts.length >= 2) {
    const first = Date.parse(ts[0]);
    const last = Date.parse(ts[ts.length - 1]);
    el("elapsed").textContent = formatElapsed((last - first) / 1000);
  }

  if (ts.length === 0) {
    el("message").textContent = "Run registered — waiting for data…";
    el("message").hidden = false;
    el("chart").hidden = true;
    return;
  }

  el("message").hidden = true;
  el("chart").hidden = false;
  const { traces, layout } = buildFigure(state.run, state.series, tokens);
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
    state.series = emptySeries();
    render();
    return;
  }

  const newRun = !state.run || state.run.id !== run.id;
  state.run = run;
  if (newRun || !state.series.last_ts) {
    state.series = await fetchSeries(run.id, SERIES_POINTS);
  } else {
    // Only what arrived since the last poll: a handful of rows, bucketed to
    // at most POLL_POINTS if the page was asleep for a while.
    const extra = await fetchSeries(run.id, POLL_POINTS, state.series.last_ts);
    appendSeries(state.series, extra);
    if (state.series.ts.length > RESERIES_ABOVE) {
      state.series = await fetchSeries(run.id, SERIES_POINTS);
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
