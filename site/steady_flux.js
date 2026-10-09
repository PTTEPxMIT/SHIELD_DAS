// SHIELD live viewer: is the downstream pressure rise at steady state yet?
//
// Pure functions, no DOM: loaded by index.html before app.js, and required by
// node in test/test_live_steady_flux.py.
//
// The downstream rise is cut into chunks spanning at least
// STEADY_CHUNK_TORR of pressure and at least STEADY_CHUNK_MIN_POINTS samples,
// and each chunk's flux (dp/dt, torr/s) is a least-squares slope. Chunks are
// sized in torr rather than time because the scatter of a slope depends on
// how much pressure it spans, not how long it took, so the same tolerance
// means the same thing on a 30 min fill and on a 4-day run; the sample floor
// keeps fast fills (one live point per 5 s) from fitting slopes to a handful
// of points.
//
// The steady span is the trailing stretch of pressure over which every chunk
// is within STEADY_TOLERANCE of the current flux (the mean of the newest
// STEADY_REF_CHUNKS chunks). The rise is steady once that span reaches
// STEADY_HOLD_TORR, or STEADY_SLOW_HOLD_TORR held for STEADY_SLOW_HOLD_S on
// slow runs that would take days to rise further. Replayed on 85 recorded
// runs, and on the September-October 2026 runs compressed to 30 and 15 min
// fills at the live 5 s cadence, it never called a run steady while its flux
// was still rising by more than 5 %.
//
// Once called, a run stays steady while the current flux is within
// STEADY_DRIFT_TOLERANCE of the flux it was called at, so a noisy chunk near
// the top of the range does not undo it; a flux that moves further than that
// (a sample that peaks and then declines) is reported as drifted.

"use strict";

const STEADY_PRESSURISED_TORR = 10.0; // upstream above this = hydrogen is in
const STEADY_RANGE_TORR = 0.95; // downstream Baratron's usable top (1 torr FS)
const STEADY_CHUNK_TORR = 0.05;
const STEADY_CHUNK_MIN_POINTS = 12;
const STEADY_MAX_CHUNKS = 80;
const STEADY_TOLERANCE = 0.03;
const STEADY_REF_CHUNKS = 2;
const STEADY_HOLD_TORR = 0.25;
const STEADY_SLOW_HOLD_TORR = 0.1;
const STEADY_SLOW_HOLD_S = 8 * 3600;
const STEADY_MIN_CHUNKS = STEADY_REF_CHUNKS; // fewer than this and there is nothing to judge
const STEADY_DRIFT_TOLERANCE = 0.05; // steady -> drifted beyond this change
const STEADY_TREND_TORR = 0.1;
const STEADY_SCAN_TORR = 0.01; // grid for finding when a run was first steady // "flux +x % over the last 0.1 torr"

function steadyMedian(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Least-squares slope of ys against xs over indices [from, to).
function steadySlope(xs, ys, from, to) {
  const n = to - from;
  let mx = 0;
  let my = 0;
  for (let i = from; i < to; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  for (let i = from; i < to; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  return sxy / sxx;
}

// Trailing chunks, newest first: {flux (torr/s), start, end} with start/end
// indices into t/p (end exclusive). Neighbouring chunks share one sample.
function fluxChunks(t, p) {
  const chunks = [];
  let end = p.length;
  while (end > STEADY_CHUNK_MIN_POINTS && chunks.length < STEADY_MAX_CHUNKS) {
    const target = p[end - 1] - STEADY_CHUNK_TORR;
    let k = end - 1;
    while (k >= 0 && p[k] > target) k--;
    k = Math.min(k, end - STEADY_CHUNK_MIN_POINTS);
    if (k < 0) break;
    chunks.push({ flux: steadySlope(t, p, k, end), start: k, end });
    end = k + 1;
  }
  return chunks;
}

// The criterion on one stretch of usable rise (t, p): the chunks, the current
// flux, the steady span and whether it is held. No range or state wording.
function judgeRise(t, p) {
  const chunks = fluxChunks(t, p);
  const judged = { chunks, ref: null, n: 0, spanTorr: 0, spanS: 0, spanStart: null, held: false };
  if (chunks.length < STEADY_MIN_CHUNKS) return judged;
  let ref = 0;
  for (let i = 0; i < STEADY_REF_CHUNKS; i++) ref += chunks[i].flux;
  ref /= STEADY_REF_CHUNKS;
  let n = 0;
  while (n < chunks.length && Math.abs(chunks[n].flux / ref - 1) < STEADY_TOLERANCE) n++;
  judged.ref = ref;
  if (n >= STEADY_REF_CHUNKS) {
    const start = chunks[n - 1].start;
    judged.n = n;
    judged.spanStart = start;
    judged.spanTorr = p[p.length - 1] - p[start];
    judged.spanS = t[t.length - 1] - t[start];
  }
  judged.held =
    judged.spanTorr >= STEADY_HOLD_TORR ||
    (judged.spanTorr >= STEADY_SLOW_HOLD_TORR && judged.spanS >= STEADY_SLOW_HOLD_S);
  return judged;
}

// The first point where the rise was judged steady, scanning prefixes every
// STEADY_SCAN_TORR of rise (a fixed pressure grid, so the answer does not
// move as data arrives): {ref, spanStart} or null.
function firstSteady(t, p) {
  let next = p[0] + STEADY_HOLD_TORR;
  for (let end = STEADY_CHUNK_MIN_POINTS; end < p.length; end++) {
    if (p[end - 1] < next) continue;
    next = p[end - 1] + STEADY_SCAN_TORR;
    const judged = judgeRise(t.slice(0, end), p.slice(0, end));
    if (judged.held) return { ref: judged.ref, spanStart: judged.spanStart };
  }
  return null;
}

// Is the downstream rise at steady state?
//
// Args: timesS (s), upstreamTorr and downstreamTorr (torr, null for gaps),
// all the same length.
// Returns null before the upstream step, else
//   {state: "waiting" | "rising" | "settling" | "steady" | "drifted" |
//      "cannot-confirm",
//    tInitS, pressureTorr, fluxTorrPerS (current; null while waiting),
//    spanTorr, spanS, spanStartS (absolute s; null without a span),
//    steadyFluxTorrPerS (flux when first called steady; steady/drifted),
//    driftChange (current vs that flux; drifted only),
//    rangeLeftTorr, rangeLeftS, toGoS (settling only), recentChange
//    (fractional flux change over the last ~0.1 torr; rising only),
//    chunks: [{tMidS (absolute s), flux, fraction (of current), inSpan}]
//    oldest first}.
function steadyFlux(timesS, upstreamTorr, downstreamTorr) {
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const pressurised = upstreamTorr.filter(
    (v) => isNum(v) && v > STEADY_PRESSURISED_TORR,
  );
  if (pressurised.length === 0) return null;
  const half = 0.5 * steadyMedian(pressurised);
  const initIndex = upstreamTorr.findIndex((v) => isNum(v) && v > half);
  const tInit = timesS[initIndex];

  // The usable rise: pressurised, from the step until the downstream first
  // reaches the top of the gauge's range.
  const t = [];
  const p = [];
  let saturated = false;
  for (let i = initIndex; i < timesS.length; i++) {
    if (!isNum(upstreamTorr[i]) || !isNum(downstreamTorr[i])) continue;
    if (upstreamTorr[i] <= STEADY_PRESSURISED_TORR) continue;
    if (downstreamTorr[i] >= STEADY_RANGE_TORR) {
      saturated = true;
      break;
    }
    t.push(timesS[i]);
    p.push(downstreamTorr[i]);
  }
  const last = p.length ? p[p.length - 1] : null;
  const pressureTorr = saturated ? STEADY_RANGE_TORR : last;
  const result = {
    state: "waiting",
    tInitS: tInit,
    pressureTorr,
    fluxTorrPerS: null,
    spanTorr: 0,
    spanS: 0,
    spanStartS: null,
    steadyFluxTorrPerS: null,
    driftChange: null,
    rangeLeftTorr: Math.max(0, STEADY_RANGE_TORR - (pressureTorr ?? 0)),
    rangeLeftS: null,
    toGoS: null,
    recentChange: null,
    chunks: [],
  };
  const judged = judgeRise(t, p);
  const { chunks, ref } = judged;
  if (ref === null) {
    if (saturated) result.state = "cannot-confirm";
    return result;
  }
  result.fluxTorrPerS = ref;
  result.rangeLeftS = ref > 0 ? result.rangeLeftTorr / ref : null;

  // Steady now, or steady earlier and still within tolerance of that flux.
  let spanStart = judged.n >= STEADY_REF_CHUNKS ? judged.spanStart : null;
  const earlier = judged.held ? null : firstSteady(t, p);
  if (judged.held) {
    result.state = "steady";
    result.steadyFluxTorrPerS = ref;
  } else if (earlier) {
    result.steadyFluxTorrPerS = earlier.ref;
    const change = ref / earlier.ref - 1;
    if (Math.abs(change) < STEADY_DRIFT_TOLERANCE) {
      result.state = "steady";
      spanStart = earlier.spanStart;
    } else {
      result.state = "drifted";
      result.driftChange = change;
    }
  }
  if (spanStart !== null) {
    result.spanTorr = last - p[spanStart];
    result.spanS = t[t.length - 1] - t[spanStart];
    result.spanStartS = t[spanStart];
  }
  result.chunks = chunks
    .map((c) => ({
      tMidS: t[(c.start + c.end - 1) >> 1],
      flux: c.flux,
      fraction: c.flux / ref,
      inSpan: spanStart !== null && c.start >= spanStart,
    }))
    .reverse();
  if (result.state !== "waiting") return result;

  // Not steady yet: can the hold still complete before the gauge tops out?
  const span = result.spanTorr;
  const left = result.rangeLeftTorr;
  const byTorr = span + left >= STEADY_HOLD_TORR;
  const bySlowHold =
    span + left >= STEADY_SLOW_HOLD_TORR &&
    result.rangeLeftS !== null &&
    result.spanS + result.rangeLeftS >= STEADY_SLOW_HOLD_S;
  if (saturated || (!byTorr && !bySlowHold)) {
    result.state = "cannot-confirm";
  } else if (span >= STEADY_SLOW_HOLD_TORR) {
    result.state = "settling";
    result.toGoS = ref > 0 ? (STEADY_HOLD_TORR - span) / ref : null;
  } else {
    result.state = "rising";
    const back = Math.min(
      chunks.length - 1,
      Math.round(STEADY_TREND_TORR / STEADY_CHUNK_TORR),
    );
    result.recentChange = chunks[0].flux / chunks[back].flux - 1;
  }
  return result;
}

if (typeof module !== "undefined") {
  module.exports = { steadyFlux, fluxChunks, STEADY_HOLD_TORR, STEADY_TOLERANCE };
}
