import os
from datetime import datetime

import numpy as np
import u6


class PressureGauge:
    """
    Base class for all pressure gauges.

    Arguments:
        name: Name of the gauge
        ain_channel: The AIN channel of the gauge
        export_filename: The filename to export the data to
        gauge_location: Location of the gauge, either "upstream" or "downstream"

    Attributes:
        name: Name of the gauge
        ain_channel: The AIN channel of the gauge
        export_filename: The filename to export the data to
        gauge_location: Location of the gauge, either "upstream" or "downstream"
        timestamp_data: List to store timestamps of readings in seconds
        real_timestamp_data: List to store real timestamps of readings in seconds
        pressure_data: List to store pressure readings in Torr
        voltage_data: List to store voltage readings in volts
        backup_dir: Directory for backups
        backup_counter: Counter for backup files
        measurements_since_backup: Counter for measurements since last backup
        backup_interval: Interval for creating backups
    """

    name: str
    ain_channel: int
    export_filename: str
    gauge_location: str
    timestamp_data: list[float]
    pressure_data: list[float]
    voltage_data: list[float]
    backup_dir: str
    backup_counter: int
    measurements_since_backup: int
    backup_interval: int

    def __init__(
        self,
        name: str,
        ain_channel: int,
        export_filename: str,
        gauge_location: str,
    ):
        self.name = name
        self.export_filename = export_filename
        self.ain_channel = ain_channel
        self.gauge_location = gauge_location

        # Data storage
        self.timestamp_data = []
        self.real_timestamp_data = []
        self.pressure_data = []
        self.voltage_data = []

        # Backup settings
        self.backup_dir = None
        self.backup_counter = 0
        self.measurements_since_backup = 0
        self.backup_interval = 10  # Save backup every 10 measurements

    @property
    def gauge_location(self):
        return self._gauge_location

    @gauge_location.setter
    def gauge_location(self, value):
        if value not in ["upstream", "downstream"]:
            raise ValueError("gauge_location must be 'upstream' or 'downstream'")
        self._gauge_location = value

    def get_ain_channel_voltage(
        self,
        labjack: u6.U6,
        resolution_index: int | None = 0,
        gain_index: int | None = 0,
        settling_factor: int | None = 0,
    ) -> float:
        """
        Obtains the voltage reading from a channel of the LabJack u6 hub.

        Args:
            labjack: The LabJack device
            resolution_index: Resolution index for the reading
            gain_index: Gain index for the reading (x1 which is +/-10V range)
            settling_factor: Settling factor for the reading

        returns:
            float: The voltage reading from the channel
        """

        # Get a single-ended reading from AIN0 using the getAIN convenience method.
        # getAIN will get the binary voltage and convert it to a decimal value.

        ain_channel_voltage = labjack.getAIN(
            positiveChannel=self.ain_channel,
            resolutionIndex=resolution_index,
            gainIndex=gain_index,
            settlingFactor=settling_factor,
            differential=False,
        )

        return ain_channel_voltage

    def voltage_to_pressure(self, voltage):
        pass

    def get_data(self, labjack: u6.U6, timestamp: float):
        """
        Gets the data from the gauge and appends it to the lists.

        Args:
            labjack: The LabJack device
            timestamp: The relative time of the reading (seconds since start)
        """
        real_timestamp = datetime.now()

        if labjack is None:
            rng = np.random.default_rng()
            pressure = rng.uniform(1, 50)
            self.timestamp_data.append(timestamp)
            self.real_timestamp_data.append(real_timestamp)
            self.voltage_data.append("test_mode")
            self.pressure_data.append(pressure)
            return

        voltage = self.get_ain_channel_voltage(labjack=labjack)
        pressure = self.voltage_to_pressure(voltage)

        # Append the data to the lists
        self.timestamp_data.append(timestamp)
        self.real_timestamp_data.append(real_timestamp)
        self.voltage_data.append(voltage)
        self.pressure_data.append(pressure)

    def initialise_export(self):
        """Initialize the main export file."""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.export_filename), exist_ok=True)

        # Create and write the header to the file
        with open(self.export_filename, "w") as f:
            f.write("RealTimestamp,RelativeTime,Pressure (Torr),Voltage (V)\n")

    def export_write(self):
        """Write the latest data point to the main export file."""
        if len(self.timestamp_data) > 0:
            # Get the latest data point
            idx = len(self.timestamp_data) - 1
            rel_timestamp = self.timestamp_data[idx]
            real_timestamp = self.real_timestamp_data[idx].strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            pressure = self.pressure_data[idx]
            voltage = self.voltage_data[idx] if idx < len(self.voltage_data) else 0

            # Write to the main export file
            with open(self.export_filename, "a") as f:
                f.write(f"{real_timestamp},{rel_timestamp},{pressure},{voltage}\n")

            # Increment the backup counter and check if we need to create a backup
            self.measurements_since_backup += 1
            if self.measurements_since_backup >= self.backup_interval:
                self.create_backup()
                self.measurements_since_backup = 0

    def create_backup(self):
        """Create a backup file with all current data."""
        if self.backup_dir is None:
            return  # Backup not initialized

        # Create a new backup filename with incrementing counter
        backup_filename = os.path.join(
            self.backup_dir, f"{self.name}_backup_{self.backup_counter:05d}.csv"
        )

        # Write all current data to the backup file
        with open(backup_filename, "w") as f:
            f.write("RealTimestamp,RelativeTime,Pressure (Torr),Voltage (V)\n")
            for i in range(len(self.timestamp_data)):
                real_ts = self.real_timestamp_data[i].strftime("%Y-%m-%d %H:%M:%S.%f")[
                    :-3
                ]
                rel_ts = self.timestamp_data[i]
                voltage = self.voltage_data[i] if i < len(self.voltage_data) else 0
                f.write(f"{real_ts},{rel_ts},{self.pressure_data[i]},{voltage}\n")

        print(f"Created backup file: {backup_filename}")
        self.backup_counter += 1


class WGM701_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name: str = "WGM701",
        ain_channel: int = 10,
        export_filename: str = "WGM701_pressure_data.csv",
        gauge_location: str = "downstream",
    ):
        super().__init__(name, ain_channel, export_filename, gauge_location)

    def voltage_to_pressure(self, voltage: float) -> float:
        """
        Converts the voltage reading from a Instrutech WGM701 pressure gauge
        to pressure in Torr.

        Args:
            voltage: The voltage reading from the gauge

        Returns:
            float: The pressure in Torr
        """
        # Convert voltage to pressure in Torr
        pressure = 10 ** ((voltage - 5.5) / 0.5)

        # Ensure pressure is within the valid range
        if pressure > 760:
            pressure = 760
        elif pressure < 7.6e-10:
            pressure = 7.6e-10

        return pressure

    def calculate_error(self, pressure_value: float) -> float:
        """
        Calculate the error in the pressure reading.

        Args:
            pressure_value: The pressure reading in Torr

        Returns:
            float: The error in the pressure reading
        """

        if 7.6e-09 < pressure_value < 7.6e-03:
            error = pressure_value * 0.3
        elif 7.6e-03 < pressure_value < 75:
            error = pressure_value * 0.15
        else:
            error = pressure_value * 0.5

        return error


class CVM211_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name: str = "CVM211",
        ain_channel: int = 8,
        export_filename: str = "CVM211_pressure_data.csv",
        gauge_location: str = "upstream",
    ):
        super().__init__(name, ain_channel, export_filename, gauge_location)

    def voltage_to_pressure(self, voltage: float) -> float:
        """
        Converts the voltage reading from a Instrutech WGM701 pressure gauge
        to pressure in Torr.

        Args:
            voltage: The voltage reading from the gauge

        Returns:
            float: The pressure in Torr
        """
        # Convert voltage to pressure in Torr
        pressure = 10 ** (voltage - 5)

        # Ensure pressure is within the valid range
        if pressure > 1000:
            pressure = 1000
        elif pressure < 1e-04:
            pressure = 1e-04

        return pressure

    def calculate_error(self, pressure_value: float) -> float:
        """
        Calculate the error in the pressure reading.

        Args:`
            pressure_value: The pressure reading in Torr

        Returns:
            float: The error in the pressure reading
        """

        if 1e-04 < pressure_value < 1e-03:
            error = 0.1e-03
        elif 1e-03 < pressure_value < 400:
            error = pressure_value * 0.1
        else:
            error = pressure_value * 0.025

        return error


class Baratron626D_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name: str = "Baratron626D",
        ain_channel: int = 6,
        export_filename: str = "Baratron626D_pressure_data.csv",
        gauge_location: str = "downstream",
        full_scale_Torr: float | None = None,
    ):
        super().__init__(name, ain_channel, export_filename, gauge_location)

        self.full_scale_Torr = full_scale_Torr

    @property
    def full_scale_Torr(self) -> float:
        if self._full_scale_Torr is None:
            raise ValueError("full_scale_Torr must be set for Baratron626D_Gauge")
        if float(self._full_scale_Torr) not in (1.0, 1000.0):
            raise ValueError(
                "full_scale_Torr must be either 1 or 1000 for Baratron626D_Gauge"
            )
        return float(self._full_scale_Torr)

    @full_scale_Torr.setter
    def full_scale_Torr(self, value):
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "full_scale_Torr must be a number (1 or 1000) for Baratron626D_Gauge"
            )
        if val not in (1.0, 1000.0):
            raise ValueError(
                "full_scale_Torr must be either 1 or 1000 for Baratron626D_Gauge"
            )
        self._full_scale_Torr = val

    def voltage_to_pressure(self, voltage: float) -> float:
        """
        Converts the voltage reading from a Instrutech WGM701 pressure gauge
        to pressure in Torr.

        Args:
            voltage: The voltage reading from the gauge

        Returns:
            float: The pressure in Torr
        """
        # Convert voltage to pressure in Torr
        pressure = 0.01 * 10 ** (2 * voltage)

        # Ensure pressure is within the valid range
        if self.full_scale_Torr == 1000:
            if pressure > 1000:
                pressure = 1000
            elif pressure < 0.5:
                pressure = 0.5

        elif self.full_scale_Torr == 1:
            if pressure > 1:
                pressure = 1
            elif pressure < 0.0005:
                pressure = 0.0005

        return pressure

    def calculate_error(self, pressure_value: float) -> float:
        """
        Calculate the error in the pressure reading.

        Args:
            pressure_value: The pressure reading in Torr

        Returns:
            float: The error in the pressure reading
        """

        if 1 < pressure_value:
            error = pressure_value * 0.0025
        else:
            error = pressure_value * 0.005

        return error
