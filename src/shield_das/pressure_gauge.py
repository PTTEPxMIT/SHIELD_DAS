import numpy as np
from typing import Optional
import os
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
        resolution_index: Optional[int] = 0,
        gain_index: Optional[int] = 0,
        settling_factor: Optional[int] = 0,
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
            timestamp: The time of the reading
        """
        if labjack is None:
            pressure = np.random.uniform(1, 50)
            self.timestamp_data.append(timestamp)
            self.voltage_data.append("test_mode")
            self.pressure_data.append(pressure)
            return

        voltage = self.get_ain_channel_voltage(labjack=labjack)
        pressure = self.voltage_to_pressure(voltage)

        # Append the data to the lists
        self.timestamp_data.append(timestamp)
        self.voltage_data.append(voltage)
        self.pressure_data.append(pressure)

    def initialise_export(self):
        """Initialize the main export file."""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.export_filename), exist_ok=True)
        
        # Create and write the header to the file
        with open(self.export_filename, "w") as f:
            f.write("Timestamp,Pressure (Torr),Voltage (V)\n")
    
    def initialise_backup(self, backup_root_dir: str):
        """Initialize the backup directory for this gauge."""
        # Create a backup directory for this specific gauge
        self.backup_dir = os.path.join(backup_root_dir, f"{self.name}_backup")
        os.makedirs(self.backup_dir, exist_ok=True)
        print(f"Initialized backup directory: {self.backup_dir}")
    
    def export_write(self):
        """Write the latest data point to the main export file."""
        if len(self.timestamp_data) > 0:
            # Get the latest data point
            idx = len(self.timestamp_data) - 1
            timestamp = self.timestamp_data[idx]
            pressure = self.pressure_data[idx]
            voltage = self.voltage_data[idx] if idx < len(self.voltage_data) else 0
            
            # Write to the main export file
            with open(self.export_filename, "a") as f:
                f.write(f"{timestamp},{pressure},{voltage}\n")
            
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
            self.backup_dir, 
            f"{self.name}_backup_{self.backup_counter:05d}.csv"
        )
        
        # Write all current data to the backup file
        with open(backup_filename, "w") as f:
            f.write("Timestamp,Pressure (Torr),Voltage (V)\n")
            for i in range(len(self.timestamp_data)):
                voltage = self.voltage_data[i] if i < len(self.voltage_data) else 0
                f.write(f"{self.timestamp_data[i]},{self.pressure_data[i]},{voltage}\n")
        
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
        elif 75 < pressure_value < 760:
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

    def calculate_error(self, pressure_value:float) -> float:
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
        elif 400 < pressure_value < 1000:
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
        pressure = 0.01 * 10 ** (2 * voltage)

        # Ensure pressure is within the valid range
        if self.name == "Baratron626D_1KT":
            if pressure > 1000:
                pressure = 1000
            elif pressure < 0.5:
                pressure = 0.5
            
        elif self.name == "Baratron626D_1T":
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

        if 1 < pressure_value < 1000:
            error = pressure_value * 0.0025
        elif pressure_value < 1:
            error = pressure_value * 0.005

        return error
