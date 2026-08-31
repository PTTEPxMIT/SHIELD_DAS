"""Record a leak test (e.g. a downstream rate-of-rise baseline).

Same hardware as a permeation run (rig_config.py), but tagged
run_type="leak_test" in run_metadata.json and with the furnace off.

Procedure: start this script FIRST so the base pressure is captured,
then isolate the volume under test and let the pressure rise.
Stop with Ctrl+C when done.
"""

from rig_config import (
    build_gauges,
    build_thermocouples,
    start_vacuum_beacon,
    stop_vacuum_beacon,
)
from shield_das import DataRecorder

my_recorder = DataRecorder(
    gauges=build_gauges(),
    thermocouples=build_thermocouples(),
    recording_interval=1,
    backup_interval=1000,
    furnace_setpoint=0,
    run_type="leak_test",
    # No sample is being characterized in a leak test; recorded as null in metadata.
    sample_thickness=None,
    sample_material=None,
)

was_running = stop_vacuum_beacon()
try:
    my_recorder.run()
finally:
    if was_running:
        start_vacuum_beacon()
