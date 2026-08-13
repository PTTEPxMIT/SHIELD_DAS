# SHIELD permeation rig Data Acquisition System
[![CI](https://github.com/PTTEPxMIT/SHIELD-DAS/actions/workflows/ci_conda.yml/badge.svg)](https://github.com/PTTEPxMIT/SHIELD-DAS/actions/workflows/ci_conda.yml)
[![codecov](https://codecov.io/gh/PTTEPxMIT/SHIELD-DAS/graph/badge.svg?token=mDUOcHgDN5)](https://codecov.io/gh/PTTEPxMIT/SHIELD-DAS)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/981399539.svg)](https://doi.org/10.5281/zenodo.17544899)

This is a tool to be used with the SHIELD hydrogen permeation rig, providing a way to both record data from the rig and have a live UI displaying plots of the pressure values in the gauges connected to the rig and the temperature of the connected thermocouple.

<img width="1901" height="900" alt="Image" src="https://github.com/user-attachments/assets/4cbdcaeb-0226-4381-a8f3-61f411e6f0aa" />

## The SHIELD software stack

This repo is one of three that make up the SHIELD software stack:

| Repo | Package | Role in the data flow |
|------|---------|-----------------------|
| **SHIELD_DAS** (this repo) | `shield_das` | **Records** rig data (LabJack + live Dash UI) |
| [`SHIELD-Data`](https://github.com/PTTEPxMIT/SHIELD-Data) | `shield_data` | **Stores & serves** uploaded runs |
| [`SHIELD-toolbox`](https://github.com/PTTEPxMIT/SHIELD-toolbox) | `shield_toolbox` | **Processes** the served data (analysis package + notebooks) |

**Data flow:** DAS records → Data stores/serves → toolbox processes.

This README covers recording a run, watching it live, and uploading it. For
how to *process* recorded data — time-lag analysis extracting permeability,
diffusivity, and solubility — see the
[SHIELD-toolbox](https://github.com/PTTEPxMIT/SHIELD-toolbox) README.

## Installation

The shield DAS package can be downloaded with `pip`

```python
pip install SHIELD-DAS
```

However, in order to interact with the Labjack, additional drivers are required from the [manufacturers site](https://support.labjack.com/docs/windows-setup-basic-driver-only).


## Running an experiment on the rig

The repo root contains **`record_run.py`**, a ready-made recording script
wired for the rig's standard gauge setup — no code needs to be written to
start a run. A full run looks like:

1. **Describe the run** — open `record_run.py` and edit the "EDIT PER RUN"
   block: furnace setpoint, sample substrate, coating layers, sample
   thickness (see [Describing the sample](#describing-the-sample)) and the
   run type. Leave the gauge setup below that block alone unless the rig
   wiring has changed.
2. **Start recording** — double-click **`start_run.bat`** (or run
   `python record_run.py` from the repo root). A new timestamped run
   directory is created under `results/` and readings start immediately.
3. **Watch it live (optional)** — run `shield-das-live --watch` in another
   terminal to serve a live dashboard of the run
   (see [Live run dashboard](#live-run-dashboard)).
4. **Mark the valve events** — press SPACEBAR at each valve event, in
   order: V4 close, V5 close, V6 close, V3 open. The script prompts for the
   next event after each press. Note the spacebar hotkey is *global* — it
   fires even when another window has focus, so don't type in other windows
   with the spacebar between valve events.
5. **Stop the run** — press Ctrl+C in the recording window. This is what
   writes `end_time` into `run_metadata.json`, so always stop with Ctrl+C
   rather than closing the window. (If Windows then asks
   "Terminate batch job (Y/N)?", answer N to keep the output readable.)
6. **Upload the run** — double-click **`upload_runs.bat`** to open a pull
   request on SHIELD-Data with the run's data
   (see [docs/auto_upload.md](docs/auto_upload.md)).

To rehearse this loop without the rig, set `run_type = "test_mode"` in
`record_run.py`: the recorder then runs without a LabJack attached,
generating dummy readings. Test runs are written to `test_run_*` directories
and are never uploaded.

## Example data recording script

This is an example of a script that can be used to activate the DAS.

```python
from shield_das import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    DataRecorder,
    Thermocouple,
    WGM701_Gauge,
)

# Define gauges
gauge_1 = WGM701_Gauge(
    gauge_location="downstream",
    ain_channel=10,
)
gauge_2 = CVM211_Gauge(
    gauge_location="upstream",
    ain_channel=8,
)
gauge_3 = Baratron626D_Gauge(
    name="Baratron626D_1KT",
    gauge_location="upstream",
    full_scale_Torr=1000,
    ain_channel=6,
)
gauge_4 = Baratron626D_Gauge(
    name="Baratron626D_1T",
    gauge_location="downstream",
    full_scale_Torr=1,
    ain_channel=4,
)

# Define thermocouple
thermocouple_1 = Thermocouple()

# Create recorder
my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2, gauge_3, gauge_4],
    thermocouples=[thermocouple_1],
    run_type="test_mode",
    recording_interval=0.5,
    backup_interval=5,
    furnace_setpoint=500,
    sample_substrate="carbon steel",
    sample_coating=[{"material": "tungsten", "thickness_nm": 800}],
    sample_thickness=0.00065,
)

# Start recording
my_recorder.run()
```

### Describing the sample

Every run must record what was mounted on the rig:

- `sample_substrate` — the substrate material, spelled out in full, e.g.
  `"carbon steel"`, `"316L steel"`.
- `sample_coating` — the coating as a list of layers, ordered as named on the
  sample. Each layer is `{"material": ..., "thickness_nm": ...}` with the
  material spelled out in full (`"tungsten"`, `"silicon carbide"`,
  `"chromium"`, `"alumina"` — not `"W"`, `"SiC"`). Pass an empty list `[]`
  for an uncoated sample. A two-layer stack looks like:

  ```python
  sample_coating=[
      {"material": "tungsten", "thickness_nm": 200},
      {"material": "chromium", "thickness_nm": 50},
  ]
  ```

- `sample_thickness` — the substrate thickness in metres.

The recorder writes these to `run_metadata.json` (schema version 1.4) as
`sample_substrate`, `sample_coating_layers` (the list above), and
`sample_coating` — a derived human-readable summary such as
`"800nm tungsten"`, `"200nm tungsten + 50nm chromium"`, or `"none"` for an
uncoated sample. Versions ≤ 1.3 recorded a single `sample_material` field
instead; readers treat that as the substrate.

## Example data visualisation script

```python
from shield_das import DataPlotter

data_500C_run1 = "results/25.08.12/run_2_11h45/"
data_500C_run2 = "results/25.08.18/run_2_09h47/"
data_500C_run3 = "results/25.08.19/run_2_09h21/"
data_500C_run4 = "results/25.08.25/run_1_09h07/"

my_plotter = DataPlotter(
    dataset_paths=[data_500C_run1, data_500C_run2, data_500C_run3, data_500C_run4],
    dataset_names=["500C_run1", "500C_run2", "500C_run3", "500C_run4"],
)
my_plotter.start()
```

## Standalone Analysis Functions

> **Note:** full processing of recorded runs (time-lag analysis for
> permeability, diffusivity, and solubility) lives in
> [SHIELD-toolbox](https://github.com/PTTEPxMIT/SHIELD-toolbox). The functions
> below are the DAS-side conversion and quick-look helpers used by the plotter.

SHIELD_DAS provides **analysis functions** that can be used independently without running the full plotter application. This is useful when you want to:
- Convert raw voltage data to pressure/temperature values
- Perform custom analysis on experimental data
- Use the conversion functions in your own scripts
- Create your own plots with the converted data

### Quick Examples

**Convert voltage to pressure:**
```python
from shield_das import voltage_to_pressure
import numpy as np

voltage = np.array([5.0, 7.5, 10.0])  # 0-10V gauge readings
pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
print(pressure)  # [500.0, 750.0, 1000.0] torr
```

**Convert thermocouple voltage to temperature:**
```python
from shield_das import voltage_to_temperature
import numpy as np

tc_voltage_mv = np.array([10.0, 20.0, 30.0])  # millivolts
local_temp_c = np.array([25.0, 25.0, 25.0])  # cold junction temp

temperature = voltage_to_temperature(local_temp_c, tc_voltage_mv)
print(temperature)  # [270.7, 508.3, 744.9] °C
```

**Calculate flux and permeability:**
```python
from shield_das import calculate_flux_from_sample, calculate_error_on_pressure_reading
import numpy as np

# Your experimental data
time = np.linspace(0, 1000, 5000)
pressure = 0.1 + 0.001 * time  # Downstream pressure rise

# Calculate hydrogen flux
flux = calculate_flux_from_sample(time, pressure)
print(f"Flux: {flux:.6e} torr/s")

# Calculate measurement uncertainties
pressure_error = calculate_error_on_pressure_reading(pressure)
```

### Available Functions

**Data Conversion:**
- `voltage_to_pressure(voltage, full_scale_torr)` - Convert gauge voltage to pressure
- `voltage_to_temperature(local_temp_c, voltage_mv)` - Convert thermocouple voltage
- `calculate_error_on_pressure_reading(pressure)` - Calculate measurement uncertainties

**Analysis:**
- `calculate_flux_from_sample(time, pressure)` - Calculate hydrogen flux
- `calculate_permeability_from_flux(...)` - Calculate permeability using Takaishi-Sensui method
- `fit_permeability_data(temps, perms)` - Fit Arrhenius equation
- `average_pressure_after_increase(time, pressure)` - Detect stable pressure after transient
- `evaluate_permeability_values(datasets)` - Extract permeability from multiple datasets

For complete documentation, see the docstrings in
[`src/shield_das/analysis.py`](src/shield_das/analysis.py).

## Live run dashboard

While a run is being recorded, `shield-das-live --watch` serves a lightweight
dashboard for it: upstream/downstream pressure (torr, log scale) and
temperature on a shared time axis, refreshed by tailing the run's CSV
incrementally. It binds the LAN by default so the same URL can be viewed
remotely (e.g. over Tailscale). See
**[docs/live_dashboard.md](docs/live_dashboard.md)**.

## Uploading completed runs

After a run finishes, double-click **`upload_runs.bat`** on the rig computer
to upload it to the
[SHIELD-Data](https://github.com/PTTEPxMIT/SHIELD-Data) repository as a pull
request. It wraps the `shield-das-upload` command (an idempotent "outbox
sweeper" — already-uploaded runs are skipped, so it is safe to run any time).
See **[docs/auto_upload.md](docs/auto_upload.md)** for the one-time setup:
GitHub token and config file.

## Processing recorded data

Once a run is uploaded, processing happens downstream of this repo:
[SHIELD-Data](https://github.com/PTTEPxMIT/SHIELD-Data) stores and serves the
run, and [SHIELD-toolbox](https://github.com/PTTEPxMIT/SHIELD-toolbox)
processes it with the time-lag method to extract **permeability, diffusivity,
and solubility** of the sample and its coatings. See the SHIELD-toolbox README
for setup and usage.
