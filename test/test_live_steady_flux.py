"""Tests for the live viewer's steady-state flux check (site/steady_flux.js).

The JavaScript is run under node on synthetic downstream rises: the analytic
time-lag solution for diffusion through a membrane, where the true approach
to steady flux is known exactly, plus a ramping flux, a slow run and a drift.
Skipped when node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

STEADY_FLUX_JS = Path(__file__).resolve().parents[1] / "site" / "steady_flux.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

# Evaluate steadyFlux on growing prefixes of one series, like successive polls.
_RUNNER = """
const { steadyFlux } = require(process.argv[1]);
const d = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = d.ends.map((n) => {
  const r = steadyFlux(d.t.slice(0, n), d.up.slice(0, n), d.dn.slice(0, n));
  return r && { state: r.state, flux: r.fluxTorrPerS, span: r.spanTorr,
                pressure: r.pressureTorr, left: r.rangeLeftTorr,
                drift: r.driftChange, chunks: r.chunks.length };
});
console.log(JSON.stringify(out));
"""

PRE_STEP_S = 120.0
UPSTREAM_TORR = 520.0
NOISE_TORR = 2e-4


def run_steady_flux(t, up, dn, ends):
    """steadyFlux results for the prefixes t[:end] for each end in ends."""
    payload = {
        "t": [float(v) for v in t],
        "up": [float(v) for v in up],
        "dn": [float(v) for v in dn],
        "ends": [int(e) for e in ends],
    }
    completed = subprocess.run(
        [NODE, "-e", _RUNNER, str(STEADY_FLUX_JS)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def flux_fraction(t_s, time_lag_s):
    """Downstream flux as a fraction of steady, for diffusion with lag τ_L."""
    n = np.arange(1, 60)[:, None]
    t = np.maximum(np.asarray(t_s, float), 1e-9)[None, :]
    terms = (-1.0) ** n * np.exp(-(n**2) * np.pi**2 * t / (6 * time_lag_s))
    return 1 + 2 * terms.sum(axis=0)


def time_lag_rise(t_s, time_lag_s, flux_torr_per_s):
    """Analytic time-lag downstream rise (torr) since the upstream step."""
    n = np.arange(1, 60)[:, None]
    t = np.maximum(np.asarray(t_s, float), 0.0)[None, :]
    series = (
        (-1.0) ** n / n**2 * np.exp(-(n**2) * np.pi**2 * t / (6 * time_lag_s))
    ).sum(axis=0)
    rise = flux_torr_per_s * (t[0] - time_lag_s - 12 * time_lag_s / np.pi**2 * series)
    return np.where(t[0] > 0, rise, 0.0)


def rig(duration_s, period_s, rise, seed=0):
    """Pre-step vacuum then an upstream step at t=0, sampled every period_s."""
    rng = np.random.default_rng(seed)
    t = np.arange(-PRE_STEP_S, duration_s, period_s)
    up = np.where(t < 0, 0.01, UPSTREAM_TORR)
    dn = 0.03 + rise(t) + rng.normal(0, NOISE_TORR, t.size)
    return t, up, dn


def every_poll(t, period_s=10.0):
    """Prefix ends for a poll every period_s seconds after the upstream step."""
    times = np.arange(period_s, t[-1] + period_s, period_s)
    return np.searchsorted(t, times, side="right")


def test_returns_null_before_the_upstream_step():
    t = np.arange(0, 600, 5.0)
    up = np.full(t.size, 0.01)
    dn = np.full(t.size, 0.03)
    assert run_steady_flux(t, up, dn, [t.size]) == [None]


def test_waits_until_there_is_enough_rise_to_judge():
    t, up, dn = rig(300, 5.0, lambda t: time_lag_rise(t, 30.0, 1e-4))
    (result,) = run_steady_flux(t, up, dn, [t.size])
    assert result["state"] == "waiting"
    assert result["flux"] is None


def test_fast_fill_is_called_steady_only_once_the_flux_has_settled():
    """A 30 min fill at the live 5 s cadence, τ_L = 3 min."""
    time_lag_s, flux = 180.0, 0.95 / (30 * 60)
    t, up, dn = rig(30 * 60, 5.0, lambda t: time_lag_rise(t, time_lag_s, flux))
    ends = every_poll(t)
    results = run_steady_flux(t, up, dn, ends)

    steady_at = [t[end - 1] for end, r in zip(ends, results) if r["state"] == "steady"]
    assert steady_at, "a settled 30 min fill must be called steady"
    # Never called while the true flux was more than 5 % short of steady.
    assert flux_fraction(steady_at, time_lag_s).min() > 0.95
    # Called with range to spare, and it stays steady once called.
    first = next(r for r in results if r["state"] == "steady")
    assert first["pressure"] < 0.7
    assert all(r["state"] == "steady" for r in results[results.index(first) :])
    assert results[-1]["flux"] == pytest.approx(flux, rel=0.02)


def test_states_progress_in_order_through_a_fill():
    time_lag_s, flux = 180.0, 0.95 / (30 * 60)
    t, up, dn = rig(30 * 60, 5.0, lambda t: time_lag_rise(t, time_lag_s, flux))
    results = run_steady_flux(t, up, dn, every_poll(t))
    seen = [r["state"] for r in results]
    order = ["waiting", "rising", "settling", "steady"]
    firsts = [seen.index(s) for s in order]
    assert firsts == sorted(firsts)


def test_ramping_flux_is_never_steady_and_cannot_be_confirmed():
    """Flux rising linearly in time all the way to the top of the gauge."""
    t, up, dn = rig(30 * 60, 5.0, lambda t: 0.95 * (np.maximum(t, 0) / 1800) ** 2)
    ends = every_poll(t)
    results = run_steady_flux(t, up, dn, ends)
    assert not any(r["state"] == "steady" for r in results)
    assert results[-1]["state"] == "cannot-confirm"
    assert results[-1]["left"] == 0


def test_slow_run_is_steady_after_a_long_hold_on_little_rise():
    """1.6 h time lag, 0.01 torr/h: under 0.25 torr in 20 h, steady for hours."""
    time_lag_s, flux = 1.6 * 3600, 0.01 / 3600
    t, up, dn = rig(20 * 3600, 60.0, lambda t: time_lag_rise(t, time_lag_s, flux))
    (result,) = run_steady_flux(t, up, dn, [t.size])
    assert result["span"] < 0.25
    assert result["state"] == "steady"


def test_flux_that_declines_after_steady_is_reported_as_drifted():
    """Steady by 12 min, then the flux sinks steadily, 15 % lower by 28 min."""
    flux, time_lag_s, end_s, knee_s = 0.95 / (30 * 60), 60.0, 28 * 60, 12 * 60

    def rise(t):
        decline = 1 - 0.15 * np.clip((t - knee_s) / (end_s - knee_s), 0, 1)
        rate = flux * flux_fraction(t, time_lag_s) * decline * (t > 0)
        return np.concatenate([[0.0], np.cumsum(rate[1:] * np.diff(t))])

    t, up, dn = rig(end_s, 5.0, rise)
    results = run_steady_flux(t, up, dn, every_poll(t))
    states = [r["state"] for r in results]
    assert "steady" in states
    assert states[-1] == "drifted"
    assert results[-1]["drift"] < -0.05
