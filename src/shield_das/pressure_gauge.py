import numpy as np
from typing import Optional


class PressureGauge:

    def __init__(
        self,
        name: str,
        ain_channel: int,
        export_filename: str,
        gauge_location: str,
        test_mode: Optional[bool] = False,
    ):
        self.name = name
        self.export_filename = export_filename
        self.ain_channel = ain_channel
        self.gauge_location = gauge_location
        self.test_mode = test_mode

        self.export_header = [
            "timestamp",
            f"voltage {self.name} (V)",
            f"pressure {self.name} (Torr)",
        ]
        self.timestamp_data = []
        self.voltage_data = []
        self.pressure_data = []

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
        labjack,
        resolution_index: float = 0,
        gain_index: float = 0,
        settling_factor: float = 0,
    ) -> float:
        """
        Obtains the voltage reading from a channel of the LabJack u6 hub.

        Args:
            resolution_index (float): Resolution index for the reading
            gain_index (float): Gain index for the reading (x1 which is +/-10V range)
            settling_factor (float): Settling factor for the reading

        returns:
            float: The voltage reading from the channel
        """

        if self.test_mode:
            # Simulate a voltage reading for testing purposes
            return round(np.random.uniform(6.8, 6.9), 4)

        # Common analog input settings.
        resolution_index = 0  # Resolution Index = 0 (default)
        gain_index = 0  # Gain Index = 0 (x1 which is +/-10V range)
        settling_factor = 0  # Settling Factor = 0 (auto)

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

    def get_data(self, labjack, timestamp):
        """
        Gets the data from the gauge and appends it to the lists.

        Args:
            labjack: The LabJack device
            timestamp (str): The timestamp of the reading
        """
        if self.test_mode:
            pressure = np.random.uniform(400, 500)
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
        """
        Initializes the export of data to a CSV file.

        Args:
            output_filename (str): The name of the output file
        """
        # Save the header to the CSV file
        with open(self.export_filename, "w") as f:
            f.write(",".join(self.export_header) + "\n")

    def export_write(self):
        """
        Exports the timestamp, voltage and pressure data to a CSV file.
        """
        row = [self.timestamp_data[-1], self.voltage_data[-1], self.pressure_data[-1]]
        with open(self.export_filename, "a") as f:
            np.savetxt(f, [row], delimiter=",", fmt="%s")


class WGM701_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name="WGM701",
        ain_channel=10,
        export_filename="WGM701_pressure_data.csv",
        gauge_location="downstream",
        test_mode=False,
    ):
        super().__init__(name, ain_channel, export_filename, gauge_location, test_mode)

    def voltage_to_pressure(self, voltage):
        """
        Converts the voltage reading from a Instrutech WGM701 pressure gauge
        to pressure in Torr.

        Args:
            voltage (float): The voltage reading from the gauge

        Returns:
            float: The pressure in Torr
        """
        # Convert voltage to pressure in Torr
        pressure = 10 ** ((voltage - 5.5) / 0.5)

        return pressure


class CVM211_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name="CVM211",
        ain_channel=8,
        export_filename="CVM211_pressure_data.csv",
        gauge_location="upstream",
        test_mode=False,
    ):
        super().__init__(name, ain_channel, export_filename, gauge_location, test_mode)

    def voltage_to_pressure(self, voltage):
        """
        Converts the voltage reading from a Instrutech WGM701 pressure gauge
        to pressure in Torr.

        Args:
            voltage (float): The voltage reading from the gauge

        Returns:
            float: The pressure in Torr
        """
        # Convert voltage to pressure in Torr
        pressure = 10 ** (voltage - 5)

        return pressure


class Baratron626D_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name="Baratron626D",
        ain_channel=6,
        export_filename="Baratron626D_pressure_data.csv",
        gauge_location="downstream",
        test_mode=False,
    ):
        super().__init__(name, ain_channel, export_filename, gauge_location, test_mode)

    def voltage_to_pressure(self, voltage):
        """
        Converts the voltage reading from a Instrutech WGM701 pressure gauge
        to pressure in Torr.

        Args:
            voltage (float): The voltage reading from the gauge

        Returns:
            float: The pressure in Torr
        """
        # Convert voltage to pressure in Torr
        pressure = 0.01 * 10 ** (2 * voltage)

        return pressure
