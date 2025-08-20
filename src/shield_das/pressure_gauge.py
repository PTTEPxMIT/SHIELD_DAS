import numpy as np
import numpy.typing as npt


class PressureGauge:
    """
    Base class for all pressure gauges.

    Arguments:
        name: Name of the gauge
        ain_channel: The AIN channel of the gauge
        gauge_location: Location of the gauge, either "upstream" or "downstream"

    Attributes:
        name: Name of the gauge
        ain_channel: The AIN channel of the gauge
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
    gauge_location: str

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
        gauge_location: str,
    ):
        self.name = name
        self.ain_channel = ain_channel
        self.gauge_location = gauge_location

        # Data storage
        self.voltage_data = []

    @property
    def gauge_location(self):
        return self._gauge_location

    @gauge_location.setter
    def gauge_location(self, value):
        if value not in ["upstream", "downstream"]:
            raise ValueError("gauge_location must be 'upstream' or 'downstream'")
        self._gauge_location = value

    def record_ain_channel_voltage(
        self,
        labjack=None,  # Remove type hint to avoid import issues
        resolution_index: int | None = 8,
        gain_index: int | None = 0,
        settling_factor: int | None = 2,
    ):
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

        if labjack is None:
            # Generate random voltage for test mode
            rng = np.random.default_rng()
            ain_channel_voltage = rng.uniform(0, 10)

        else:
            ain_channel_voltage = labjack.getAIN(
                positiveChannel=self.ain_channel,
                resolutionIndex=resolution_index,
                gainIndex=gain_index,
                settlingFactor=settling_factor,
                differential=False,
            )

        self.voltage_data.append(ain_channel_voltage)


class WGM701_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name: str = "WGM701",
        ain_channel: int = 10,
        gauge_location: str = "downstream",
    ):
        super().__init__(name, ain_channel, gauge_location)

    def voltage_to_pressure(self, voltage: npt.NDArray) -> npt.NDArray:
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

        # Apply valid range: set very small values to 0, and cap at 760 Torr
        pressure = np.where(pressure < 7.6e-10, 0, pressure)
        pressure = np.clip(pressure, 0, 760)

        return pressure

    def calculate_error(self, pressure_value: float) -> float:
        """
        Calculate the error in the pressure reading.

        Args:
            pressure_value: The pressure reading in Torr

        Returns:
            float: The error in the pressure reading
        """

        p = np.asarray(pressure_value, dtype=float)

        # Initialise with default error (0.5 * pressure)
        error = p * 0.5

        # Apply conditions with np.where
        error = np.where((7.6e-09 < p) & (p < 7.6e-03), p * 0.3, error)
        error = np.where((7.6e-03 < p) & (p < 75), p * 0.15, error)

        return error


class CVM211_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.
    """

    def __init__(
        self,
        name: str = "CVM211",
        ain_channel: int = 8,
        gauge_location: str = "upstream",
    ):
        super().__init__(name, ain_channel, gauge_location)

    def voltage_to_pressure(self, voltage: npt.NDArray) -> npt.NDArray:
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

        # Apply valid range: set very small values to 0, and cap at 1000 Torr
        pressure = np.where(pressure < 1e-04, 0, pressure)
        pressure = np.clip(pressure, 0, 1000)

        return pressure

    def calculate_error(self, pressure_value: float) -> float:
        """
        Calculate the error in the pressure reading.

        Args:
            pressure_value: The pressure reading in Torr

        Returns:
            float: The error in the pressure reading
        """

        p = np.asarray(pressure_value, dtype=float)

        # Initialize with default error (2.5% of pressure)
        error = p * 0.025

        # Apply conditions with np.where
        error = np.where((1e-04 < p) & (p < 1e-03), 0.1e-03, error)
        error = np.where((1e-03 < p) & (p < 400), p * 0.1, error)

        return error


class Baratron626D_Gauge(PressureGauge):
    """
    Class for the WGM701 pressure gauge.

    Upstream AIN channel = 6
    Downstream AIN channel = 4
    """

    def __init__(
        self,
        ain_channel: int,
        name: str = "Baratron626D",
        gauge_location: str = "downstream",
        full_scale_Torr: float | None = None,
    ):
        super().__init__(name, ain_channel, gauge_location)

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

    def voltage_to_pressure(self, voltage: npt.NDArray) -> npt.NDArray:
        """
        Converts the voltage reading from a Instrutech WGM701 pressure gauge
        to pressure in Torr.

        Args:
            voltage: The voltage reading from the gauge

        Returns:
            float: The pressure in Torr
        """
        # Convert voltage to pressure in Torr
        pressure = voltage * (self.full_scale_Torr / 10.0)

        # Apply valid range based on full scale
        if self.full_scale_Torr == 1000:
            pressure = np.where(pressure < 0.5, 0, pressure)
            pressure = np.clip(pressure, 0, 1000)
        elif self.full_scale_Torr == 1:
            pressure = np.where(pressure < 0.0005, 0, pressure)
            pressure = np.clip(pressure, 0, 1)

        return pressure

    def calculate_error(self, pressure_value: float) -> float:
        """
        Calculate the error in the pressure reading.

        Args:
            pressure_value: The pressure reading in Torr

        Returns:
            float: The error in the pressure reading
        """

        p = np.asarray(pressure_value, dtype=float)

        # Initialize with default error (0.5% of pressure)
        error = p * 0.005

        # Apply conditions with np.where
        error = np.where(p > 1, p * 0.0025, error)

        return error
