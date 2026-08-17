"""Tests for the Supabase live publisher.

Covers config loading (env-var precedence, unknown-key warnings), voltage to
physical-unit conversion of CSV rows, downsampling, resume-after-restart
skipping, the HTTP client's headers and key redaction, offline buffering with
backoff, run-end handling, and the dry-run mode. No hardware and no network:
``urllib.request.urlopen`` is always mocked, runs are fake directories on
tmp_path (same helpers as test_live_dashboard).
"""

import io
import json
import os
import urllib.error
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from shield_das.analysis import voltage_to_temperature
from shield_das.publisher import (
    SUPABASE_KEY_ENV_VAR,
    DryRunClient,
    PublisherConfig,
    RunPublisher,
    SupabaseClient,
    _get_key,
    _parse_run_start,
    main,
    row_to_channels,
)

# =============================================================================
# Helpers for building fake run directories (mirrors test_live_dashboard)
# =============================================================================

GAUGES = [
    {
        "name": "Baratron626D_1KT",
        "type": "Baratron626D_Gauge",
        "gauge_location": "upstream",
        "full_scale_torr": 1000,
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


def csv_row(row_index, gauges, thermocouples, voltage_v=5.0, period_s=0.5):
    """Build one CSV data row with timestamps period_s apart."""
    timestamp = datetime(2026, 8, 4, 10, 0, 0) + timedelta(seconds=period_s * row_index)
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
    period_s=0.5,
):
    """Create a fake run directory with metadata and CSV; return its path."""
    run_dir = os.path.join(str(results_dir), date_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    run_info = {"date": "2026-08-04", "start_time": "2026-08-04 10:00:00"}
    if end_time is not None:
        run_info["end_time"] = end_time
    metadata = {"version": "1.4", "run_info": run_info, "gauges": gauges}
    if thermocouples:
        metadata["thermocouples"] = thermocouples
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f)

    lines = [csv_header(gauges, thermocouples)]
    lines += [
        csv_row(i, gauges, thermocouples, period_s=period_s) for i in range(n_rows)
    ]
    with open(os.path.join(run_dir, "shield_data.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return run_dir


class FakeClient:
    """In-memory client recording every call the publisher makes."""

    def __init__(self, last_ts=None, fail_inserts=False):
        self.inserted = []
        self.batches = []
        self.heartbeats = 0
        self.ended_at = None
        self.pruned_to = None
        self.thinned = []
        self.upserted = None
        self._last_ts = last_ts
        self.fail_inserts = fail_inserts

    def upsert_run(self, key, metadata, started_at):
        self.upserted = (key, started_at)
        return 42

    def last_reading_ts(self, run_id):
        return self._last_ts

    def insert_readings(self, rows):
        if self.fail_inserts:
            raise RuntimeError("boom")
        self.batches.append(list(rows))
        self.inserted.extend(rows)

    def mark_ended(self, run_id, ended_at):
        self.ended_at = ended_at

    def heartbeat(self, run_id):
        self.heartbeats += 1

    def thin_readings(self, run_id, older_than_hours, keep_seconds):
        self.thinned.append((older_than_hours, keep_seconds))

    def prune_runs(self, keep):
        self.pruned_to = keep


def make_publisher(tmp_path, config=None, client=None, **run_kwargs):
    """Build an attached RunPublisher over a fake run."""
    run_dir = make_run(tmp_path, **run_kwargs)
    config = config or PublisherConfig(results_dir=str(tmp_path))
    client = client or FakeClient()
    publisher = RunPublisher(run_dir, config, client)
    publisher.attach()
    return publisher, client


# =============================================================================
# Config and key resolution
# =============================================================================


def test_config_from_missing_file_uses_defaults(tmp_path):
    """A missing config file yields the documented defaults."""
    config = PublisherConfig.from_file(str(tmp_path / "nope.json"))

    assert config.sample_period_s == 5.0
    assert config.keep_runs == 2
    assert config.max_buffer_rows == 20000


def test_config_warns_on_unknown_keys(tmp_path, caplog):
    """Unknown config keys warn and are ignored, like the uploader."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"supabase_url": "https://x.supabase.co", "nope": 1}))

    with caplog.at_level("WARNING"):
        config = PublisherConfig.from_file(str(path))

    assert config.supabase_url == "https://x.supabase.co"
    assert "nope" in caplog.text


def test_env_var_takes_precedence_over_config(monkeypatch):
    """SHIELD_SUPABASE_KEY wins over the config file key."""
    monkeypatch.setenv(SUPABASE_KEY_ENV_VAR, "env-key")

    assert _get_key(PublisherConfig(supabase_key="file-key")) == "env-key"


def test_missing_key_raises(monkeypatch):
    """With no env var and no config key, resolution fails clearly."""
    monkeypatch.delenv(SUPABASE_KEY_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=SUPABASE_KEY_ENV_VAR):
        _get_key(PublisherConfig())


# =============================================================================
# row_to_channels conversion
# =============================================================================


def test_row_to_channels_converts_known_gauges_to_torr():
    """Known gauge types land under their name, converted to torr."""
    metadata = {"gauges": GAUGES, "thermocouples": []}
    row = {"Baratron626D_1KT_Voltage (V)": 5.0, "WGM701_Voltage (V)": 5.5}

    channels = row_to_channels(metadata, row)

    # Baratron: linear, 5 V of 10 V full scale over 1000 torr
    assert channels["Baratron626D_1KT"] == pytest.approx(500.0)
    # WGM701: 10**((5.5 - 5.5) / 0.5) = 1 torr
    assert channels["WGM701"] == pytest.approx(1.0)


def test_row_to_channels_unknown_gauge_falls_back_to_volts():
    """An unknown gauge type is published as raw volts under <name>_V."""
    metadata = {"gauges": [{"name": "mystery", "type": "Unobtainium_Gauge"}]}

    channels = row_to_channels(metadata, {"mystery_Voltage (V)": 3.25})

    assert channels == {"mystery_V": 3.25}


def test_row_to_channels_thermocouple_and_local_temperature():
    """Thermocouple mV becomes °C (cold-junction compensated) plus local_C."""
    metadata = {"gauges": [], "thermocouples": THERMOCOUPLES}
    row = {"Local_temperature (C)": 25.0, "TC1_Voltage (mV)": 1.0}

    channels = row_to_channels(metadata, row)

    expected = voltage_to_temperature(
        local_temperature=np.asarray([25.0]), voltage=np.asarray([1.0])
    )[0]
    assert channels["TC1_C"] == pytest.approx(expected, rel=1e-5)
    assert channels["local_C"] == 25.0


def test_row_to_channels_drops_non_finite_values():
    """NaN readings are dropped: jsonb cannot represent them."""
    metadata = {"gauges": GAUGES, "thermocouples": []}
    row = {
        "Baratron626D_1KT_Voltage (V)": float("nan"),
        "WGM701_Voltage (V)": 5.5,
    }

    channels = row_to_channels(metadata, row)

    assert "Baratron626D_1KT" not in channels
    assert "WGM701" in channels


def test_row_to_channels_rounds_to_six_significant_digits():
    """Values are rounded to bound the jsonb payload width."""
    metadata = {"gauges": [{"name": "mystery", "type": "Unobtainium_Gauge"}]}

    channels = row_to_channels(metadata, {"mystery_Voltage (V)": 1.23456789})

    assert channels["mystery_V"] == 1.23457


def test_parse_run_start_falls_back_to_now_on_garbage():
    """Unparseable metadata start times do not crash the attach."""
    assert _parse_run_start("2026-08-04 10:00:00").startswith("2026-08-04T10:00:00")
    assert _parse_run_start(None)  # falls back to now, still an ISO string
    assert _parse_run_start("not a date")


# =============================================================================
# RunPublisher: attach, downsample, resume, flush, end
# =============================================================================


def test_attach_upserts_run_and_prunes(tmp_path):
    """Attaching registers the run by key and prunes old mirror runs."""
    publisher, client = make_publisher(tmp_path)

    assert publisher.run_id == 42
    assert client.upserted[0] == "26.08.04_run_1_10h00"
    assert client.upserted[1].startswith("2026-08-04T10:00:00")
    assert client.pruned_to == 2


def test_downsampling_keeps_one_point_per_sample_period(tmp_path):
    """20 rows at 0.5 s spacing with a 5 s period publish ~2 points."""
    publisher, client = make_publisher(tmp_path, n_rows=20)

    publisher.tick()

    sent_ts = [row["ts"] for row in client.inserted]
    assert len(sent_ts) == 2  # t=0 and t=5.0 (t=9.5 is within the period)
    assert all(row["run_id"] == 42 for row in client.inserted)


def test_published_data_is_in_physical_units(tmp_path):
    """The published payload holds torr/°C channel values, not volts."""
    publisher, client = make_publisher(tmp_path)

    publisher.tick()

    data = client.inserted[0]["data"]
    assert data["Baratron626D_1KT"] == pytest.approx(500.0)
    assert "local_C" in data and "TC1_C" in data


def test_resume_skips_rows_the_server_already_has(tmp_path):
    """After a restart, rows at or before the server's newest ts are skipped."""
    already = datetime(2026, 8, 4, 10, 0, 0).astimezone()  # server already has t=0
    publisher, client = make_publisher(
        tmp_path, client=FakeClient(last_ts=already), n_rows=20
    )

    publisher.tick()

    sent_ts = [row["ts"] for row in client.inserted]
    assert len(sent_ts) == 1  # only t=5.0; t=0 was already published


def test_flush_sends_heartbeat_with_batch(tmp_path):
    """Each periodic flush also bumps the liveness heartbeat."""
    publisher, client = make_publisher(tmp_path)

    publisher.tick()

    assert client.heartbeats == 1


def test_failed_insert_buffers_and_backs_off(tmp_path):
    """A network failure keeps points buffered and sets a backoff window."""
    client = FakeClient(fail_inserts=True)
    publisher, _ = make_publisher(tmp_path, client=client, n_rows=20)

    publisher.tick()

    assert len(publisher._pending) == 2
    assert publisher._retry_at > 0

    # Within the backoff window nothing is retried
    client.fail_inserts = False
    publisher._flush(max_batches=None, with_heartbeat=False)
    assert client.inserted == []

    # After the window the buffered points go through
    publisher._retry_at = 0.0
    publisher._flush(max_batches=None, with_heartbeat=False)
    assert len(client.inserted) == 2
    assert len(publisher._pending) == 0


def test_buffer_is_bounded(tmp_path):
    """The offline buffer drops oldest points beyond max_buffer_rows."""
    config = PublisherConfig(results_dir=str(tmp_path), max_buffer_rows=3)
    client = FakeClient(fail_inserts=True)
    publisher, _ = make_publisher(tmp_path, config=config, client=client, n_rows=100)

    publisher.tick()

    assert len(publisher._pending) == 3


def test_large_backlogs_are_split_into_batches(tmp_path):
    """A backlog larger than the batch size goes out in several inserts."""
    config = PublisherConfig(results_dir=str(tmp_path), sample_period_s=0.5)
    publisher, client = make_publisher(
        tmp_path, config=config, n_rows=1200, period_s=0.5
    )

    publisher.tick()
    publisher._flush(max_batches=None, with_heartbeat=False)

    assert sum(len(batch) for batch in client.batches) == 1200
    assert all(len(batch) <= 500 for batch in client.batches)
    assert len(client.batches) >= 3


def test_ended_detects_end_time_and_finish_reports_it(tmp_path):
    """A run gaining end_time is detected; finish flushes and marks ended."""
    publisher, client = make_publisher(tmp_path)
    publisher.tick()
    assert not publisher.ended()

    metadata_path = os.path.join(publisher.run_dir, "run_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["run_info"]["end_time"] = "2026-08-04 10:30:00"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    assert publisher.ended()
    publisher.finish()
    assert client.ended_at is not None


def test_ended_when_run_directory_disappears(tmp_path):
    """A vanished run directory counts as ended, not a crash."""
    publisher, _ = make_publisher(tmp_path)
    os.remove(os.path.join(publisher.run_dir, "run_metadata.json"))

    assert publisher.ended()


# =============================================================================
# SupabaseClient HTTP behaviour (urllib mocked)
# =============================================================================


def _http_error(code=500, body=b"secret-key-leaked"):
    return urllib.error.HTTPError(
        url="https://x.supabase.co",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def test_client_sends_key_headers_and_prefer():
    """Requests carry apikey + bearer headers and the Prefer resolution."""
    client = SupabaseClient("https://x.supabase.co", "sk-123")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return io.BytesIO(b"")

    with patch("urllib.request.urlopen", fake_urlopen):
        client.insert_readings([{"run_id": 1, "ts": "t", "data": {}}])

    request = captured["request"]
    assert request.get_header("Apikey") == "sk-123"
    assert request.get_header("Authorization") == "Bearer sk-123"
    assert request.get_header("Prefer") == "resolution=ignore-duplicates"
    assert "/rest/v1/readings?on_conflict=run_id,ts" in request.full_url


def test_client_redacts_key_from_errors():
    """The service key never appears in raised error messages."""
    client = SupabaseClient("https://x.supabase.co", "secret-key")

    with patch("urllib.request.urlopen", side_effect=_http_error()):
        with pytest.raises(RuntimeError) as excinfo:
            client.heartbeat(1)

    assert "secret-key" not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_client_requires_url():
    """A missing supabase_url fails with a clear message."""
    with pytest.raises(RuntimeError, match="supabase_url"):
        SupabaseClient("", "key")


def test_upsert_run_parses_returned_id():
    """upsert_run returns the id from the representation response."""
    client = SupabaseClient("https://x.supabase.co", "k")

    with patch(
        "urllib.request.urlopen",
        return_value=io.BytesIO(json.dumps([{"id": 7}]).encode()),
    ):
        assert client.upsert_run("key", {}, "2026-01-01T00:00:00") == 7


def test_last_reading_ts_parses_timestamp():
    """last_reading_ts returns an aware datetime, or None when empty."""
    client = SupabaseClient("https://x.supabase.co", "k")

    with patch(
        "urllib.request.urlopen",
        return_value=io.BytesIO(b'[{"ts": "2026-08-04T10:00:00+00:00"}]'),
    ):
        ts = client.last_reading_ts(1)
    assert ts == datetime.fromisoformat("2026-08-04T10:00:00+00:00")

    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"[]")):
        assert client.last_reading_ts(1) is None


# =============================================================================
# CLI
# =============================================================================


def test_dry_run_makes_no_network_calls(tmp_path, capsys):
    """--dry-run with --once publishes a whole run with zero urlopen calls."""
    make_run(tmp_path, run_name="test_run_1_10h00", end_time=None, n_rows=3)
    # End the run immediately so --once returns after one pass
    with patch("urllib.request.urlopen") as urlopen:
        with patch("time.sleep"):
            with patch.object(RunPublisher, "ended", side_effect=[False, True]):
                exit_code = main(
                    [
                        "--config",
                        str(tmp_path / "none.json"),
                        "--results-dir",
                        str(tmp_path),
                        "--include-test-runs",
                        "--dry-run",
                        "--once",
                    ]
                )

    assert exit_code == 0
    assert urlopen.call_count == 0
    assert "[dry-run]" in capsys.readouterr().out


def test_main_without_key_fails_before_touching_network(tmp_path, monkeypatch):
    """Without a key, main fails fast (no urlopen) when not in dry-run."""
    monkeypatch.delenv(SUPABASE_KEY_ENV_VAR, raising=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"supabase_url": "https://x.supabase.co"}))

    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RuntimeError, match=SUPABASE_KEY_ENV_VAR):
            main(["--config", str(config), "--results-dir", str(tmp_path)])

    assert urlopen.call_count == 0


def test_dry_run_client_reports_payloads(capsys):
    """DryRunClient prints every operation it would perform."""
    client = DryRunClient()
    assert client.upsert_run("key", {}, "t") == 0
    assert client.last_reading_ts(0) is None
    client.insert_readings([{"run_id": 0, "ts": "t", "data": {"x": 1}}])
    client.heartbeat(0)
    client.mark_ended(0, "t")
    client.thin_readings(0, 48, 60)
    client.prune_runs(2)

    out = capsys.readouterr().out
    assert out.count("[dry-run]") == 6
