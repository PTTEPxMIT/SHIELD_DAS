"""Record a run on the SHIELD permeation rig.

Edit the "EDIT PER RUN" section below to describe the sample and furnace
setpoint, then run this script on the rig computer:

    python record_run.py

During recording, press the spacebar at each valve event, in order:
V4 close, V5 close, V6 close, V3 open. Results are written to a new
timestamped directory under ``results/``.
"""

from shield_das import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    DataRecorder,
    Thermocouple,
    WGM701_Gauge,
)

# ----------------------------- EDIT PER RUN -----------------------------

furnace_setpoint = 500  # °C

sample_substrate = "carbon steel"  # spelled out in full, e.g. "316L steel"
sample_coating = [
    # Layers ordered as named on the sample; [] for an uncoated sample.
    # Materials spelled out in full ("tungsten", not "W").
    {"material": "tungsten", "thickness_nm": 800},
]
sample_thickness = 0.00065  # m

# "permeation_exp" for a real run; "leak_test" for a leak test;
# "test_mode" runs without the LabJack attached.
run_type = "permeation_exp"

# ------------------------------------------------------------------------

# Standard rig gauge setup — only change if the wiring changes
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

recorder = DataRecorder(
    gauges=gauges,
    thermocouples=thermocouples,
    furnace_setpoint=furnace_setpoint,
    sample_substrate=sample_substrate,
    sample_coating=sample_coating,
    sample_thickness=sample_thickness,
    run_type=run_type,
    recording_interval=0.5,  # s between readings
    backup_interval=5.0,  # s between backup CSV rotations
)

recorder.run()
