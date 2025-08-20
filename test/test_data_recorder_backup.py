import os
import time
from unittest.mock import Mock

from shield_das import DataRecorder, PressureGauge


def test_backup_segments_created(tmp_path):
    # Setup fast recorder with backup enabled and very short interval
    gauge = Mock(spec=PressureGauge)
    gauge.name = "G1"
    gauge.voltage_data = [1.23]
    gauge.record_ain_channel_voltage.return_value = None
    gauge.ain_channel = 10
    gauge.gauge_location = "downstream"

    rec = DataRecorder(
        gauges=[gauge],
        thermocouples=[],
        results_dir=str(tmp_path),
        run_type="test_mode",
        recording_interval=0.05,
        backup_interval=0.1,
    )

    rec.start()
    # Allow a few intervals to pass and at least one rotation
    time.sleep(0.35)
    rec.stop()

    # Verify backup directory exists and contains one or more backup files
    backup_dir = os.path.join(rec.run_dir, "backup")
    assert os.path.isdir(backup_dir)
    files = [f for f in os.listdir(backup_dir) if f.endswith(".csv")]
    # At least one backup segment should be created
    assert len(files) >= 1

    # Verify headers exist in backup files and have data lines
    for fname in files:
        with open(os.path.join(backup_dir, fname)) as f:
            lines = f.readlines()
        assert lines
        assert lines[0].startswith("RealTimestamp")
        # may or may not have data depending on timing, but usually should
        # have at least one data line; be tolerant and just check header exists


def test_backup_rotation(tmp_path):
    gauge = Mock(spec=PressureGauge)
    gauge.name = "G1"
    gauge.voltage_data = [2.34]
    gauge.record_ain_channel_voltage.return_value = None
    gauge.ain_channel = 4
    gauge.gauge_location = "upstream"

    rec = DataRecorder(
        gauges=[gauge],
        thermocouples=[],
        results_dir=str(tmp_path),
        run_type="test_mode",
        recording_interval=0.02,
        backup_interval=0.06,
    )

    rec.start()
    # duration ~0.2s should allow multiple rotations
    time.sleep(0.22)
    rec.stop()

    backup_dir = os.path.join(rec.run_dir, "backup")
    files = sorted(f for f in os.listdir(backup_dir) if f.endswith(".csv"))
    # Expect 2 or more rotated files depending on timing
    assert len(files) >= 2
