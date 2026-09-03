import json
import os
from datetime import datetime

import numpy as np
import numpy.typing as npt

from .analysis import (
    calculate_error_on_pressure_reading,
    voltage_to_pressure,
    voltage_to_temperature,
)

# run_metadata.json versions this reader understands. 1.3 and 1.4 share the
# same file layout; 1.4 adds the structured sample description and sample_id.
SUPPORTED_METADATA_VERSIONS = ("1.3", "1.4")


class Dataset:
    """
    Represents a single dataset with pressure and temperature measurements and addtional
    information from the metadata file for a given experimental run.

    Handles loading, processing, and storing of data.

    Args:
        path: Path to dataset folder
        name: Display name for the dataset

    Attributes:
        path: Path to dataset folder
        name: Display name for the dataset
        colour: Color for plotting this dataset
        time_data: Array of time values (seconds from start)
        upstream_pressure: Array of upstream pressure values
        upstream_error: Array of upstream pressure errors
        downstream_pressure: Array of downstream pressure values
        downstream_error: Array of downstream pressure errors
        valve_times: Dictionary of valve event times
        colour: Color for plotting this dataset
        temperature: Furnace set point temperature (K)
        sample_material: Substrate name (v1.4 sample_substrate, else the v1.3
            sample_material), "Unknown" if neither is recorded
        sample_substrate: Substrate material as recorded, or None
        sample_coating: Human-readable coating summary, or None
        sample_coating_layers: Coating layers as recorded (empty if uncoated)
        sample_id: Identifier of the physical sample, or None
        sample_thickness: Sample thickness (m)
        furnace_set_point: Furnace set point temperature (K)
        local_temperature_data: Local temperature readings (optional)
        thermocouple_data: Thermocouple temperature readings (optional)
        thermocouple_name: Name of thermocouple (optional)
    """

    path: str
    name: str

    time_data: npt.NDArray
    upstream_pressure: npt.NDArray
    upstream_error: npt.NDArray
    downstream_pressure: npt.NDArray
    downstream_error: npt.NDArray
    colour: str
    valve_times: dict[str, float]

    sample_material: str
    sample_thickness: float

    furnace_setpoint: float
    local_temperature_data: npt.NDArray | None = None
    thermocouple_data: npt.NDArray | None = None
    thermocouple_name: str | None = None

    def __init__(
        self,
        path: str,
        name: str,
    ):
        self.path = path
        self.name = name

        self.time_data = None
        self.upstream_pressure = None
        self.upstream_error = None
        self.downstream_pressure = None
        self.downstream_error = None
        self.valve_times = None
        self.colour = None
        self.sample_material = None
        self.sample_substrate = None
        self.sample_coating = None
        self.sample_coating_layers = []
        self.sample_id = None
        self.sample_thickness = None
        self.furnace_setpoint = None
        self.local_temperature_data = None
        self.thermocouple_data = None
        self.thermocouple_name = None

        self.live_data = False

    @property
    def live_data(self):
        return self._live_data

    @live_data.setter
    def live_data(self, value: bool):
        self._live_data = value

    @property
    def dataset_file(self):
        csv_path = os.path.join(self.path, "shield_data.csv")
        return csv_path

    @property
    def metadata(self):
        metadata_path = os.path.join(self.path, "run_metadata.json")

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Validate version: 1.3 and 1.4 share the same file layout (1.4 adds
        # the structured sample description and sample_id to run_info).
        version = metadata.get("version")
        if version not in SUPPORTED_METADATA_VERSIONS:
            raise ValueError(
                f"Unsupported metadata version: {version}. "
                f"Supported versions: {', '.join(SUPPORTED_METADATA_VERSIONS)}. "
                f"Please regenerate your data with the latest version of the recorder."
            )

        return metadata

    def read_csv_file(self, csv_path: str) -> npt.NDArray:
        """Read CSV data file and return structured numpy array."""

        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        return data

    def process_data(self):
        data = self.read_csv_file(self.dataset_file)

        # Convert timestamps to relative time
        dt_objects = [
            datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in data["RealTimestamp"]
        ]
        self.time_data = np.array(
            [(dt - dt_objects[0]).total_seconds() for dt in dt_objects]
        )

        # Process gauge data
        for gauge in self.metadata.get("gauges", []):
            if gauge.get("type") == "Baratron626D_Gauge":
                col_name = f"{gauge['name']}_Voltage_V"
                volt_vals = np.array(data[col_name], dtype=float)
                pressure_vals = voltage_to_pressure(
                    volt_vals, full_scale_torr=float(gauge["full_scale_torr"])
                )
                if gauge.get("gauge_location") == "upstream":
                    self.upstream_pressure = pressure_vals
                else:
                    self.downstream_pressure = pressure_vals

        self.upstream_error = calculate_error_on_pressure_reading(
            self.upstream_pressure
        )
        self.downstream_error = calculate_error_on_pressure_reading(
            self.downstream_pressure
        )

        # Process temperature data if present
        try:
            local_temperature_data = np.array(data["Local_temperature_C"], dtype=float)
            self.local_temperature_data = local_temperature_data
            tname = self.metadata["thermocouples"][0]["name"]
            self.thermocouple_name = tname
            volt_vals = np.array(data[f"{tname}_Voltage_mV"], dtype=float)
            self.thermocouple_data = voltage_to_temperature(
                local_temperature=local_temperature_data, voltage=volt_vals
            )
        except (KeyError, ValueError, IndexError) as e:
            # Temperature data not available or invalid
            print(f"Warning: Could not load temperature data for {self.name}: {e}")
            self.local_temperature_data = None
            self.thermocouple_data = None
            self.thermocouple_name = None

        # Extract valve times
        self.valve_times = {}
        start_time_str = self.metadata["run_info"]["start_time"]
        if start_time_str:
            try:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

            # Every other "*_time" key in run_info is an operator event recorded
            # by the recorder's spacebar sequence (valve operations, pressure
            # set); their names are not assumed here.
            for key, value in self.metadata["run_info"].items():
                if key.endswith("_time") and key not in ("start_time", "end_time"):
                    valve_dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
                    self.valve_times[key] = (valve_dt - start_time).total_seconds()

        # Extract metadata properties
        self.furnace_setpoint = (
            self.metadata["run_info"].get("furnace_setpoint", 25.0) + 273.15
        )
        # Sample description: v1.4 fields, falling back to the v1.3
        # sample_material (which SHIELD-Data treats as the substrate).
        run_info = self.metadata["run_info"]
        self.sample_substrate = run_info.get("sample_substrate") or run_info.get(
            "sample_material"
        )
        self.sample_material = (
            self.sample_substrate if self.sample_substrate is not None else "Unknown"
        )
        self.sample_coating = run_info.get("sample_coating")
        self.sample_coating_layers = run_info.get("sample_coating_layers") or []
        self.sample_id = run_info.get("sample_id")
        self.sample_thickness = self.metadata["run_info"].get(
            "sample_thickness", 0.00088
        )
