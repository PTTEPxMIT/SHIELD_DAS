"""Tests for the present-vacuum beacon.

Covers config loading from the shared publisher file, voltage to torr
conversion of a direct gauge read, the rolling history window (bounding,
expiry, clearing), payload shape and size, push backoff, dormancy while a run
is recording, and the CLI's dry-run wiring. No hardware and no network: the
LabJack is replaced by a fake sampler and ``urllib.request.urlopen`` is
mocked.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

from shield_das import beacon as beacon_module
from shield_das.beacon import (
    DEFAULT_GAUGES,
    BeaconConfig,
    LabJackSampler,
    VacuumBeacon,
    beacon_loop,
    main,
    read_channels,
)
from shield_das.publisher import DryRunClient

# =============================================================================
# Helpers
# =============================================================================


class FakeLabJack:
    """Returns a fixed voltage per AIN channel and counts reads."""

    def __init__(self, voltages: dict[int, float]):
        self.voltages = voltages
        self.reads = 0

    def getAIN(self, positiveChannel, **kwargs):
        self.reads += 1
        return self.voltages[positiveChannel]


class FakeSampler:
    """Sampler stand-in yielding scripted channel readings."""

    def __init__(self, values, fail_times: int = 0):
        self.values = values
        self.fail_times = fail_times
        self.calls = 0

    def sample(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("LabJack read failed: device busy")
        value = self.values[min(self.calls - 1, len(self.values) - 1)]
        return {"WGM701": value, "CVM211": 1.0e-3}


class RecordingClient:
    """Captures standby_update payloads, optionally failing first."""

    def __init__(self, fail_times: int = 0):
        self.payloads = []
        self.fail_times = fail_times
        self.calls = 0

    def standby_update(self, data):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("Supabase unreachable for POST /rpc/standby_update")
        self.payloads.append(data)


def make_config(**overrides) -> BeaconConfig:
    defaults = {
        "supabase_url": "https://example.supabase.co",
        "supabase_key": "service-key",
        "sample_period_s": 1.0,
        "history_seconds": 60.0,
        "push_period_s": 5.0,
    }
    defaults.update(overrides)
    return BeaconConfig(**defaults)


def write_active_run(tmp_path, name="run_1_09h00"):
    """Create a run directory that ``find_active_run`` treats as live."""
    run_dir = tmp_path / "26.08.27" / name
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run_info": {"start_time": "2026-08-27 09:00:00"}})
    )
    (run_dir / "shield_data.csv").write_text("RealTimestamp\n2026-08-27 09:00:00.000\n")
    return run_dir


# =============================================================================
# Configuration
# =============================================================================


def test_config_defaults_target_the_downstream_gauge():
    config = BeaconConfig()
    assert config.primary_channel == "WGM701"
    assert config.history_seconds == 60.0
    assert config.history_points == 60


def test_history_points_tracks_window_and_cadence():
    assert make_config(history_seconds=60, sample_period_s=2).history_points == 30
    assert make_config(history_seconds=60, sample_period_s=0.5).history_points == 120
    # Never zero, however coarse the cadence.
    assert make_config(history_seconds=10, sample_period_s=600).history_points == 1


@pytest.mark.parametrize(
    "field", ["sample_period_s", "history_seconds", "push_period_s"]
)
def test_non_positive_cadences_are_rejected(field):
    """A zero period would divide by zero and spin the rig's CPU."""
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        make_config(**{field: 0})


def test_config_shares_the_publisher_file_and_ignores_its_keys(tmp_path, caplog):
    path = tmp_path / ".shield_das_publisher.json"
    path.write_text(
        json.dumps(
            {
                "supabase_url": "https://x.supabase.co",
                "supabase_key": "k",
                "results_dir": "/rig/results",
                "flush_period_s": 15.0,  # publisher-only
                "keep_runs": 2,  # publisher-only
                "primary_channel": "CVM211",  # beacon-only
            }
        )
    )
    with caplog.at_level("WARNING"):
        config = BeaconConfig.from_file(str(path))

    assert config.supabase_url == "https://x.supabase.co"
    assert config.results_dir == "/rig/results"
    assert config.primary_channel == "CVM211"
    # Publisher-owned keys are expected here, so they must not warn.
    assert "Ignoring unknown config keys" not in caplog.text


def test_config_warns_on_genuinely_unknown_keys(tmp_path, caplog):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"nonsense_key": 1}))
    with caplog.at_level("WARNING"):
        BeaconConfig.from_file(str(path))
    assert "nonsense_key" in caplog.text


def test_config_missing_file_uses_defaults(tmp_path):
    config = BeaconConfig.from_file(str(tmp_path / "absent.json"))
    assert config.gauges == DEFAULT_GAUGES


# =============================================================================
# Reading and converting
# =============================================================================


def test_read_channels_converts_each_gauge_type():
    labjack = FakeLabJack({10: 3.5, 8: 2.0, 6: 5.0, 4: 10.0})
    channels = read_channels(labjack, DEFAULT_GAUGES)

    # WGM701: 10 ** ((V - 5.5) / 0.5)
    assert channels["WGM701"] == pytest.approx(10 ** ((3.5 - 5.5) / 0.5), rel=1e-6)
    # Baratrons are linear on full scale.
    assert channels["Baratron626D_1KT"] == pytest.approx(500.0, rel=1e-6)
    assert channels["Baratron626D_1T"] == pytest.approx(1.0, rel=1e-6)
    assert labjack.reads == len(DEFAULT_GAUGES)


def test_read_channels_matches_the_publishers_conversion():
    """A beacon reading and a recorded reading of one gauge must agree."""
    from shield_das.publisher import row_to_channels

    gauge = DEFAULT_GAUGES[0]
    labjack = FakeLabJack({10: 4.25})
    direct = read_channels(labjack, [gauge])
    via_csv = row_to_channels({"gauges": [gauge]}, {"WGM701_Voltage (V)": 4.25})
    assert direct["WGM701"] == via_csv["WGM701"]


def test_read_channels_falls_back_to_volts_for_unknown_gauges():
    unknown = [{"name": "Mystery", "type": "Nonesuch_Gauge", "ain_channel": 2}]
    channels = read_channels(FakeLabJack({2: 7.5}), unknown)
    assert channels == {"Mystery_V": 7.5}


def test_sampler_test_mode_needs_no_hardware():
    channels = LabJackSampler(DEFAULT_GAUGES, test_mode=True).sample()
    assert set(channels) == {gauge["name"] for gauge in DEFAULT_GAUGES}


# =============================================================================
# The rolling window
# =============================================================================


def test_history_is_bounded_by_the_window():
    config = make_config(history_seconds=60, sample_period_s=1)
    beacon = VacuumBeacon(config, RecordingClient(), FakeSampler([1e-6]))
    for i in range(500):
        beacon.sample(1000.0 + i)
    assert len(beacon.history) == 60


def test_payload_drops_samples_older_than_the_window():
    config = make_config(history_seconds=60, sample_period_s=1)
    beacon = VacuumBeacon(config, RecordingClient(), FakeSampler([1e-6]))
    beacon.sample(1000.0)  # 100 s before "now" -> outside the window
    beacon.sample(1075.0)
    beacon.sample(1090.0)

    payload = beacon.payload(1100.0)
    timestamps = [point[0] for point in payload["history"]]
    assert timestamps == [1075.0, 1090.0]


def test_clear_drops_the_window_so_a_gap_is_not_shown_as_history():
    beacon = VacuumBeacon(make_config(), RecordingClient(), FakeSampler([1e-6]))
    beacon.sample(1000.0)
    beacon.clear()
    assert beacon.history == []
    assert beacon.payload(1000.0)["channels"] == {}


def test_payload_shape_and_size():
    config = make_config()
    values = [1.0e-6 * (1 + i / 100) for i in range(60)]
    beacon = VacuumBeacon(config, RecordingClient(), FakeSampler(values))
    for i in range(60):
        beacon.sample(1_800_000_000.0 + i)

    payload = beacon.payload(1_800_000_060.0)
    assert payload["primary"] == "WGM701"
    assert payload["history_seconds"] == 60.0
    assert len(payload["history"]) == 60
    assert set(payload["channels"]) == {"WGM701", "CVM211"}
    # One full window must stay well inside a single 2 KB heap page, so the
    # row never TOASTs (see docs/live_supabase.md).
    assert len(json.dumps(payload)) < 2048


def test_sample_failure_is_skipped_not_fatal():
    sampler = FakeSampler([1e-6], fail_times=1)
    beacon = VacuumBeacon(make_config(), RecordingClient(), sampler)
    beacon.tick(1000.0, 100.0)  # first sample raises
    assert beacon.history == []
    beacon.tick(1001.0, 200.0)
    assert len(beacon.history) == 1


# =============================================================================
# Pushing
# =============================================================================


def test_push_sends_the_window():
    client = RecordingClient()
    beacon = VacuumBeacon(make_config(), client, FakeSampler([2.5e-6]))
    beacon.sample(1000.0)
    beacon.push(1000.0, 10.0)
    assert client.payloads[0]["channels"]["WGM701"] == 2.5e-6


def test_push_before_any_sample_sends_nothing():
    client = RecordingClient()
    beacon = VacuumBeacon(make_config(), client, FakeSampler([1e-6]))
    beacon.push(1000.0, 10.0)
    assert client.calls == 0


def test_push_failure_backs_off_and_recovers():
    client = RecordingClient(fail_times=1)
    beacon = VacuumBeacon(make_config(), client, FakeSampler([1e-6]))
    beacon.sample(1000.0)

    beacon.push(1000.0, 100.0)  # fails, arms a 15 s backoff
    assert client.calls == 1
    beacon.push(1001.0, 105.0)  # inside the backoff window: not attempted
    assert client.calls == 1
    beacon.push(1016.0, 116.0)  # past it: retried and succeeds
    assert client.calls == 2
    assert len(client.payloads) == 1


def test_tick_respects_both_cadences():
    client = RecordingClient()
    sampler = FakeSampler([1e-6])
    config = make_config(sample_period_s=1.0, push_period_s=5.0)
    beacon = VacuumBeacon(config, client, sampler)

    for i in range(11):  # 0.0 .. 5.0 monotonic seconds, half-second steps
        beacon.tick(1000.0 + i * 0.5, i * 0.5)

    assert sampler.calls == 6  # t = 0, 1, 2, 3, 4, 5
    assert client.calls == 2  # t = 0 and t = 5


def test_first_tick_samples_and_pushes_immediately():
    """A beacon started moments ago should already show a number."""
    client = RecordingClient()
    sampler = FakeSampler([7.5e-7])
    beacon = VacuumBeacon(make_config(push_period_s=300.0), client, sampler)

    beacon.tick(1000.0, 0.0)

    assert sampler.calls == 1
    assert client.payloads[0]["channels"]["WGM701"] == 7.5e-7


# =============================================================================
# Dormancy while a run is recording
# =============================================================================


def test_dormant_while_a_run_is_recording(tmp_path):
    write_active_run(tmp_path)
    beacon = VacuumBeacon(
        make_config(results_dir=str(tmp_path)), RecordingClient(), FakeSampler([1e-6])
    )
    assert beacon.dormant() is True


def test_not_dormant_when_no_run_is_recording(tmp_path):
    beacon = VacuumBeacon(
        make_config(results_dir=str(tmp_path)), RecordingClient(), FakeSampler([1e-6])
    )
    assert beacon.dormant() is False


def test_loop_never_touches_the_labjack_while_a_run_records(tmp_path):
    write_active_run(tmp_path)
    client = RecordingClient()
    sampler = FakeSampler([1e-6])
    config = make_config(results_dir=str(tmp_path), sample_period_s=0.001)

    with patch("shield_das.beacon.time.sleep"):
        beacon_loop(config, client, sampler, iterations=5)

    assert sampler.calls == 0
    assert client.calls == 0


def test_loop_samples_and_pushes_when_idle(tmp_path):
    client = RecordingClient()
    sampler = FakeSampler([1e-6])
    config = make_config(
        results_dir=str(tmp_path), sample_period_s=1.0, push_period_s=1.0
    )
    # One simulated second per iteration, so both cadences fire every time.
    clock = iter([float(i) for i in range(100)])

    with (
        patch("shield_das.beacon.time.sleep"),
        patch("shield_das.beacon.time.monotonic", lambda: next(clock)),
    ):
        beacon_loop(config, client, sampler, iterations=3)

    assert sampler.calls == 3
    assert len(client.payloads) == 3


# =============================================================================
# CLI
# =============================================================================


def test_main_dry_run_touches_no_network(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"supabase_url": "https://x.supabase.co"}))

    with (
        patch("shield_das.beacon.beacon_loop") as loop,
        patch("urllib.request.urlopen", side_effect=AssertionError("network!")),
    ):
        exit_code = main(
            [
                "--config",
                str(config_path),
                "--results-dir",
                str(tmp_path),
                "--dry-run",
                "--test-mode",
                "--history-seconds",
                "30",
                "--sample-period",
                "2",
            ]
        )

    assert exit_code == 0
    config, client, sampler = loop.call_args[0]
    assert isinstance(client, DryRunClient)
    assert sampler.test_mode is True
    assert config.history_seconds == 30
    assert config.sample_period_s == 2
    assert config.history_points == 15


def test_main_requires_a_key_when_publishing_for_real(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"supabase_url": "https://x.supabase.co"}))
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(RuntimeError, match="No Supabase key"),
    ):
        main(["--config", str(config_path)])


def test_dry_run_client_prints_a_readable_line(capsys):
    DryRunClient().standby_update(
        {
            "primary": "WGM701",
            "channels": {"WGM701": 4.2e-6},
            "history": [[1.0, 4.2e-6]],
        }
    )
    out = capsys.readouterr().out
    assert "WGM701" in out
    assert "1 points in window" in out


# =============================================================================
# LabJack handling (hardware mocked out entirely)
# =============================================================================


class FakeU6Module:
    """Stand-in for the ``u6`` module, recording opens and closes."""

    def __init__(self, voltages=None, open_error=None, close_error=None):
        # `or` would treat an intentionally empty mapping as "use defaults".
        default = {10: 3.5, 8: 2.0, 6: 5.0, 4: 10.0}
        self.voltages = default if voltages is None else voltages
        self.open_error = open_error
        self.close_error = close_error
        self.opens = 0
        self.closes = 0
        module = self

        class _U6:
            def __init__(self, firstFound=True):
                module.opens += 1
                if module.open_error:
                    raise module.open_error

            def getCalibrationData(self):
                pass

            def getAIN(self, positiveChannel, **kwargs):
                return module.voltages[positiveChannel]

            def close(self):
                module.closes += 1
                if module.close_error:
                    raise module.close_error

        self.U6 = _U6


def test_sampler_opens_and_closes_the_device_each_time():
    """Holding the handle would stop a run from claiming the LabJack."""
    fake = FakeU6Module()
    sampler = LabJackSampler(DEFAULT_GAUGES)

    with patch.dict(sys.modules, {"u6": fake}):
        first = sampler.sample()
        second = sampler.sample()

    assert first["WGM701"] == second["WGM701"]
    assert fake.opens == 2
    assert fake.closes == 2


def test_sampler_reports_a_failed_open_as_runtime_error():
    fake = FakeU6Module(open_error=OSError("device busy"))
    with patch.dict(sys.modules, {"u6": fake}):
        with pytest.raises(RuntimeError, match="LabJack read failed"):
            LabJackSampler(DEFAULT_GAUGES).sample()


def test_sampler_still_closes_after_a_read_failure():
    """A leaked handle would lock the device out for the next run."""
    fake = FakeU6Module(voltages={})  # KeyError on the first getAIN
    with patch.dict(sys.modules, {"u6": fake}):
        with pytest.raises(RuntimeError):
            LabJackSampler(DEFAULT_GAUGES).sample()
    assert fake.closes == 1


def test_sampler_tolerates_a_failing_close():
    fake = FakeU6Module(close_error=OSError("already gone"))
    with patch.dict(sys.modules, {"u6": fake}):
        channels = LabJackSampler(DEFAULT_GAUGES).sample()
    assert "WGM701" in channels


def test_read_channels_drops_non_finite_values():
    """jsonb cannot hold NaN, so it must never reach the payload."""
    unknown = [{"name": "Mystery", "type": "Nonesuch_Gauge", "ain_channel": 2}]
    channels = read_channels(FakeLabJack({2: float("nan")}), unknown)
    assert channels == {}


# =============================================================================
# Waking back up after a run
# =============================================================================


def test_loop_resumes_after_the_run_ends(tmp_path):
    """The window is cleared on the way in, so no stale gap is published."""
    run_dir = write_active_run(tmp_path)
    client = RecordingClient()
    sampler = FakeSampler([1e-6])
    config = make_config(
        results_dir=str(tmp_path), sample_period_s=1.0, push_period_s=1.0
    )
    clock = iter([float(i) for i in range(100)])

    calls = {"n": 0}
    real_find = beacon_module.find_active_run

    def find_until_third_call(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 3:  # the run "ends" partway through
            return None
        return real_find(*args, **kwargs)

    with (
        patch("shield_das.beacon.time.sleep"),
        patch("shield_das.beacon.time.monotonic", lambda: next(clock)),
        patch("shield_das.beacon.find_active_run", find_until_third_call),
    ):
        beacon_loop(config, client, sampler, iterations=4)

    assert run_dir.exists()
    assert sampler.calls == 2  # dormant for the first two iterations
    assert len(client.payloads) == 2


# =============================================================================
# CLI overrides
# =============================================================================


def test_main_applies_url_and_results_dir_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"supabase_url": "https://old.supabase.co"}))

    with patch("shield_das.beacon.beacon_loop") as loop:
        main(
            [
                "--config",
                str(config_path),
                "--supabase-url",
                "https://new.supabase.co",
                "--results-dir",
                str(tmp_path),
                "--dry-run",
                "--test-mode",
            ]
        )

    config = loop.call_args[0][0]
    assert config.supabase_url == "https://new.supabase.co"
    assert config.results_dir == str(tmp_path)


def test_main_exits_cleanly_on_ctrl_c(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"supabase_url": "https://x.supabase.co"}))

    with patch("shield_das.beacon.beacon_loop", side_effect=KeyboardInterrupt):
        exit_code = main(["--config", str(config_path), "--dry-run", "--test-mode"])

    assert exit_code == 0
