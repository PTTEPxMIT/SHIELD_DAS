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

# Sample under test. Substrate and coating materials are spelled out in full
# (e.g. "carbon steel", "316L steel"; "tungsten", not "W"). Thickness in
# metres. SAMPLE_ID identifies the physical specimen (e.g. "S07") — it pairs
# leak tests with the permeation runs on the same sample, so keep it
# identical for every run of one mounted sample.
SAMPLE_SUBSTRATE = "carbon steel"
SAMPLE_COATING = "uncoated"  # e.g. "800nm tungsten", or "uncoated"
SAMPLE_COATING_LAYERS = []  # e.g. [{"material": "tungsten", "thickness_nm": 800}]
SAMPLE_THICKNESS = 0.001
SAMPLE_ID = None  # e.g. "S07"; REQUIRED when RUN_TYPE = "leak_test"

# Run type: "permeation_exp", "leak_test" (background leak-rate measurement,
# needs SAMPLE_ID), or "test_mode" (no hardware needed — for testing the DAS)
RUN_TYPE = "permeation_exp"

# Leak test only: the downstream isolation setpoint (torr, 0.0025-1);
# recorded in the metadata. Ignored for other run types when None.
DOWNSTREAM_SETPOINT_TORR = None

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
        sample_substrate=SAMPLE_SUBSTRATE,
        sample_coating=SAMPLE_COATING,
        sample_coating_layers=SAMPLE_COATING_LAYERS,
        sample_thickness=SAMPLE_THICKNESS,
        sample_id=SAMPLE_ID,
        downstream_setpoint_torr=DOWNSTREAM_SETPOINT_TORR,
        run_type=RUN_TYPE,
        recording_interval=RECORDING_INTERVAL,
        backup_interval=BACKUP_INTERVAL,
        results_dir=RESULTS_DIR,
    )
    recorder.run()
