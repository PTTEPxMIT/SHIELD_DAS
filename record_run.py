"""Record a run on the SHIELD permeation rig.

Open this file in VS Code and press the Run button (or `python record_run.py`).
Edit the settings in the "EDIT ME" block below before each run — nothing else
in this file should need touching.

Stop recording with Ctrl+C (data is saved continuously, so nothing is lost).
"""

from pathlib import Path

from shield_das import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    DataRecorder,
    Thermocouple,
    WGM701_Gauge,
)

# ============================================================================
# EDIT ME — settings for this run
# ============================================================================

# Furnace setpoint temperature (°C)
FURNACE_SETPOINT = 500

# Sample under test: material must be "316" or "AISI 1018", thickness in metres
SAMPLE_MATERIAL = "316"
SAMPLE_THICKNESS = 0.001

# Run type: "permeation_exp", "leak_test", or
# "test_mode" (no hardware needed — for testing the DAS itself)
RUN_TYPE = "permeation_exp"

# Timing (seconds): how often to sample, and how often to rotate backup CSVs
RECORDING_INTERVAL = 0.5
BACKUP_INTERVAL = 5.0

# ============================================================================
# Hardware configuration — only change if the rig wiring changes
# ============================================================================

gauges = [
    WGM701_Gauge(gauge_location="downstream", ain_channel=10),
    CVM211_Gauge(gauge_location="upstream", ain_channel=8),
    Baratron626D_Gauge(
        name="Baratron626D_1KT",
        gauge_location="upstream",
        full_scale_Torr=1000,
        ain_channel=6,
    ),
    Baratron626D_Gauge(
        name="Baratron626D_1T",
        gauge_location="downstream",
        full_scale_Torr=1,
        ain_channel=4,
    ),
]

thermocouples = [Thermocouple()]

# Results always land in SHIELD_DAS/results/ regardless of where this is run from
RESULTS_DIR = str(Path(__file__).parent / "results")

# ============================================================================
# Recording — no need to edit below this line
# ============================================================================

if __name__ == "__main__":
    recorder = DataRecorder(
        gauges=gauges,
        thermocouples=thermocouples,
        furnace_setpoint=FURNACE_SETPOINT,
        sample_material=SAMPLE_MATERIAL,
        sample_thickness=SAMPLE_THICKNESS,
        run_type=RUN_TYPE,
        recording_interval=RECORDING_INTERVAL,
        backup_interval=BACKUP_INTERVAL,
        results_dir=RESULTS_DIR,
    )
    recorder.run()
