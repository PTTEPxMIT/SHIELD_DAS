import os
from datetime import datetime

import numpy as np


class Thermocouple:
    """
    Class to handle Type K thermocouple data acquisition and conversion.

    This class reads the thermocouple voltage and converts it to temperature
    using NIST ITS-90 polynomial coefficients.

    Attributes:
        labjack: LabJack U6 device instance for reading thermocouple data.
        cjc_mv: Cold junction compensation voltage in millivolts.
    """

    def __init__(
        self,
        name: str = "type K thermocouple",
        export_filename: str = "temperature_data.csv",
    ):
        self.name = name
        self.export_filename = export_filename

        # Data storage
        self.timestamp_data = []
        self.real_timestamp_data = []
        self.local_temperature_data = []
        self.measured_temperature_data = []

        # Backup settings
        self.backup_dir = None
        self.backup_counter = 0
        self.measurements_since_backup = 0
        self.backup_interval = 10  # Save backup every 10 measurements

    def get_temperature(
        self,
        labjack,  # Remove type hint to avoid import issues
        timestamp: float,
        ain_channel: int = 0,
        gain_index: int = 3,
    ) -> float:
        """
        Read temperature from a Type K thermocouple connected to a LabJack U6 using
        differential input mode.

        This function reads the cold junction temperature from the device's internal
        sensor, reads the differential voltage from the thermocouple input channels,
        applies cold junction compensation, and converts the resulting voltage to
        temperature.

        args:
            labjack: An instance of the LabJack U6 device.
            pos_channel: The positive analog input channel number connected to the
                thermocouple positive lead (default 0).
            gain_index: The LabJack gain setting index to set input voltage range and
                resolution (default 3, ±0.1 V range).

        returns:
            float: The calculated temperature in degrees Celsius.
        """
        real_timestamp = datetime.now()

        if labjack is None:
            rng = np.random.default_rng()
            temp_l = rng.uniform(25, 30)
            temp_m = rng.uniform(25, 30)
            self.timestamp_data.append(timestamp)
            self.real_timestamp_data.append(real_timestamp)
            self.local_temperature_data.append(temp_l)
            self.measured_temperature_data.append(temp_m)
            return

        # Read cold junction temperature in Celsius (LabJack returns Kelvin)
        local_temperature = labjack.getTemperature() - 273.15 + 2.5

        # Read differential thermocouple voltage (volts)
        tc_v = labjack.getAIN(
            ain_channel, resolutionIndex=8, gainIndex=gain_index, differential=True
        )

        # Convert thermocouple voltage to millivolts
        tc_mv = tc_v * 1000

        # Calculate cold junction compensation voltage (mV)
        cjc_mv = temp_c_to_mv(local_temperature)

        # Total thermocouple voltage including cold junction compensation
        total_mv = tc_mv + cjc_mv

        # Convert total voltage to temperature in Celsius
        measured_temperature = mv_to_temp_c(total_mv)

        # Append the data to the lists
        self.timestamp_data.append(timestamp)
        self.real_timestamp_data.append(real_timestamp)
        self.local_temperature_data.append(local_temperature)
        self.measured_temperature_data.append(measured_temperature)

    def initialise_export(self):
        """Initialize the main export file."""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.export_filename), exist_ok=True)

        # Create and write the header to the file
        with open(self.export_filename, "w") as f:
            f.write("RealTimestamp,RelativeTime,LocalTemp (C),MeasuredTemp (C)\n")

    def export_write(self):
        """Write the latest data point to the main export file."""
        if len(self.timestamp_data) > 0:
            # Get the latest data point
            idx = len(self.timestamp_data) - 1
            rel_timestamp = self.timestamp_data[idx]
            real_timestamp = self.real_timestamp_data[idx].strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            local_temp = self.local_temperature_data[idx]
            measured_temp = (
                self.measured_temperature_data[idx]
                if idx < len(self.measured_temperature_data)
                else 0
            )

            # Write to the main export file
            with open(self.export_filename, "a") as f:
                f.write(
                    f"{real_timestamp},{rel_timestamp},{local_temp},{measured_temp}\n"
                )

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
            f.write("RealTimestamp,RelativeTime,LocalTemp (C),MeasuredTemp (C)\n")
            for i in range(len(self.timestamp_data)):
                real_ts = self.real_timestamp_data[i].strftime("%Y-%m-%d %H:%M:%S.%f")[
                    :-3
                ]
                rel_ts = self.timestamp_data[i]
                measured_temperature = (
                    self.measured_temperature_data[i]
                    if i < len(self.measured_temperature_data)
                    else 0
                )
                f.write(
                    f"{real_ts},{rel_ts},{self.local_temperature_data[i]},{measured_temperature}\n"
                )

        print(f"Created backup file: {backup_filename}")
        self.backup_counter += 1


def evaluate_poly(coeffs: list[float] | tuple[float], x: float) -> float:
    """ "
    Evaluate a polynomial at x given the list of coefficients.

    The polynomial is:
        P(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    where coeffs = [a0, a1, ..., an]

    args:
        coeffs:Polynomial coefficients ordered by ascending power.
        x: The value at which to evaluate the polynomial.

    returns;
        float: The evaluated polynomial result.
    """
    return sum(a * x**i for i, a in enumerate(coeffs))


def volts_to_temp_constants(mv: float) -> tuple[float, ...]:
    """
    Select the appropriate NIST ITS-90 polynomial coefficients for converting
    Type K thermocouple voltage (in millivolts) to temperature (°C).

    The valid voltage range is -5.891 mV to 54.886 mV.

    args:
        mv: Thermocouple voltage in millivolts.

    returns:
        tuple of float: Polynomial coefficients for the voltage-to-temperature conversion.

    raises:
        ValueError: If the input voltage is out of the valid range.
    """
    # Use a small tolerance for floating-point comparison
    if mv < -5.892 or mv > 54.887:
        raise ValueError("Voltage out of valid Type K range (-5.891 to 54.886 mV).")
    if mv < 0:
        # Range: -5.891 mV to 0 mV
        return (
            0.0e0,
            2.5173462e1,
            -1.1662878e0,
            -1.0833638e0,
            -8.977354e-1,
            -3.7342377e-1,
            -8.6632643e-2,
            -1.0450598e-2,
            -5.1920577e-4,
        )
    elif mv < 20.644:
        # Range: 0 mV to 20.644 mV
        return (
            0.0e0,
            2.508355e1,
            7.860106e-2,
            -2.503131e-1,
            8.31527e-2,
            -1.228034e-2,
            9.804036e-4,
            -4.41303e-5,
            1.057734e-6,
            -1.052755e-8,
        )
    else:
        # Range: 20.644 mV to 54.886 mV
        return (
            -1.318058e2,
            4.830222e1,
            -1.646031e0,
            5.464731e-2,
            -9.650715e-4,
            8.802193e-6,
            -3.11081e-8,
        )


def temp_to_volts_constants(
    temp_c: float,
) -> tuple[tuple[float, ...], tuple[float, float, float] | None]:
    """
    Select the appropriate NIST ITS-90 polynomial coefficients for converting
    temperature (°C) to Type K thermocouple voltage (in millivolts).

    Valid temperature range is -270°C to 1372°C.

    args:
        temp_c: Temperature in degrees Celsius.

    returns:
        Tuple containing:
            - tuple of float: Polynomial coefficients for temperature-to-voltage conversion.
            - tuple of three floats or None: Extended exponential term coefficients for temp >= 0°C, else None.

    raises:
        ValueError: If the input temperature is out of the valid range.
    """
    if temp_c < -270 or temp_c > 1372:
        raise ValueError("Temperature out of valid Type K range (-270 to 1372 C).")
    if temp_c < 0:
        # Range: -270 °C to 0 °C
        return (
            0.0e0,
            0.39450128e-1,
            0.236223736e-4,
            -0.328589068e-6,
            -0.499048288e-8,
            -0.675090592e-10,
            -0.574103274e-12,
            -0.310888729e-14,
            -0.104516094e-16,
            -0.198892669e-19,
            -0.163226975e-22,
        ), None
    else:
        # Range: 0 °C to 1372 °C, with extended exponential term
        return (
            -0.176004137e-1,
            0.38921205e-1,
            0.1855877e-4,
            -0.994575929e-7,
            0.318409457e-9,
            -0.560728449e-12,
            0.560750591e-15,
            -0.3202072e-18,
            0.971511472e-22,
            -0.121047213e-25,
        ), (0.1185976e0, -0.1183432e-3, 0.1269686e3)


def temp_c_to_mv(temp_c: float) -> float:
    """
    Convert temperature (°C) to Type K thermocouple voltage (mV) using
    NIST ITS-90 polynomial approximations and an exponential correction for
    temperatures ≥ 0 °C.

    args:
        temp_c: Temperature in degrees Celsius.

    returns:
        float: Thermocouple voltage in millivolts.
    """
    coeffs, extended = temp_to_volts_constants(temp_c)
    mv = evaluate_poly(coeffs, temp_c)
    if extended:
        a0, a1, a2 = extended
        mv += a0 * np.exp(a1 * (temp_c - a2) ** 2)
    return mv


def mv_to_temp_c(mv: float) -> float:
    """
    Convert Type K thermocouple voltage (mV) to temperature (°C) using
    NIST ITS-90 polynomial approximations.

    args:
        mv: Thermocouple voltage in millivolts.

    returns:
        float: Temperature in degrees Celsius.
    """
    coeffs = volts_to_temp_constants(mv)
    return evaluate_poly(coeffs, mv)
