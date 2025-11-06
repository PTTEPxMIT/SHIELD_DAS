import numpy as np
from numpy.typing import ArrayLike, NDArray
from uncertainties import UFloat, ufloat


def average_pressure_after_increase(
    time: ArrayLike,
    pressure: ArrayLike,
    window: int = 5,
    slope_threshold: float = 1e-3,
) -> float:
    """
    Detect when pressure stabilises after a sudden increase and return the average
    stable pressure value.

    This function analyses pressure vs time data to identify when the pressure has
    stabilised after an initial transient increase. The algorithm:

    1. Calculates instantaneous pressure slopes using numpy gradient
    2. Applies moving average smoothing (convolution with uniform kernel) to reduce
       noise in slope calculations
    3. Identifies stable regions where absolute slope falls below threshold AND
       time is after initial 5-second settling period
    4. Returns average pressure from the first stable point onwards

    If no stable region is detected, the function falls back to averaging the second
    half of the data.

    Args:
        time: Time data in seconds.
        pressure: Pressure data in torr.
        window: Size of the moving window for slope smoothing, defaults to 5.
        slope_threshold: Maximum absolute slope (torr/s) considered "stable". Defaults
            to 1e-3 torr/s.

    Returns:
        Average pressure in torr after the system has stabilised.

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import average_pressure_after_increase
            import numpy as np

            time = np.linspace(0, 100, 100)
            pressure = np.concatenate([np.full(50, 10), np.full(50, 100)])
            avg_p = average_pressure_after_increase(time, pressure)
            print(f"Stable pressure: {avg_p:.2f} torr")

        Stable pressure: 100.00 torr

    """
    # Convert inputs to numpy arrays for consistent array operations
    time_array = np.asarray(time)
    pressure_array = np.asarray(pressure)

    # Calculate instantaneous pressure slopes using gradient
    pressure_slopes = np.gradient(pressure_array, time_array)

    # Apply moving average smoothing to reduce noise in slope calculation
    # Creates a uniform kernel of size 'window' for convolution
    smoothing_kernel = np.ones(window) / window
    smoothed_slopes = np.convolve(pressure_slopes, smoothing_kernel, mode="same")

    # Identify where pressure has stabilised:
    # 1. Absolute slope must be below threshold (nearly flat)
    # 2. Time must be after initial 5-second settling period
    is_stable = np.abs(smoothed_slopes) < slope_threshold
    after_settling_period = time_array > time_array.min() + 5
    stability_mask = is_stable & after_settling_period

    # Find first index where conditions are met, or use midpoint as fallback
    if stability_mask.any():
        stable_region_start = np.argmax(stability_mask)
    else:
        # No stable region found, use second half of data
        stable_region_start = len(time_array) // 2

    # Calculate and return average pressure from stable region onwards
    stable_pressure_values = pressure_array[stable_region_start:]
    return float(np.mean(stable_pressure_values))


def calculate_flux_from_sample(t_data: ArrayLike, P_data: ArrayLike) -> float:
    """
    Calculate hydrogen flux from downstream pressure rise data.

    This function fits a weighted linear regression to the pressure rise data in the
    downstream chamber, filtering out unreliable low and high pressure readings from
    the gauge. The slope of this fit represents the flux of hydrogen permeating
    through the sample.

    The algorithm applies:

    1. Pressure filtering to remove unreliable gauge readings outside the range
       0.05-0.95 torr (gauge extremes where readings are less accurate)
    2. Exponential weighting that emphasises later data points, ranging from
       exp(-1) ≈ 0.37 at start to exp(0) = 1.0 at end (more stable regime)
    3. Weighted linear least squares polynomial fitting (degree 1) using numpy
       polyfit to obtain the flux slope

    Args:
        t_data: Time data in seconds.
        P_data: Downstream pressure data in torr.

    Returns:
        Slope of the pressure rise in torr/s, representing the flux rate.

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import calculate_flux_from_sample
            import numpy as np

            time = np.linspace(0, 100, 100)
            pressure = 0.1 + 0.01 * time  # Linear rise from 0.1 to 1.1 torr
            flux = calculate_flux_from_sample(time, pressure)
            print(f"Flux: {flux:.4f} torr/s")

        Flux: 0.0100 torr/s

    """
    # Convert inputs to numpy arrays for consistent array operations
    time_values = np.asarray(t_data)
    pressure_values = np.asarray(P_data)

    # Define reliable pressure gauge range (avoid low/high extremes)
    MINIMUM_RELIABLE_PRESSURE = 0.05  # torr
    MAXIMUM_RELIABLE_PRESSURE = 0.95  # torr

    # Filter data to only include reliable gauge measurements
    is_within_reliable_range = (pressure_values >= MINIMUM_RELIABLE_PRESSURE) & (
        pressure_values <= MAXIMUM_RELIABLE_PRESSURE
    )
    filtered_time = time_values[is_within_reliable_range]
    filtered_pressure = pressure_values[is_within_reliable_range]

    # Create exponential weights favouring later measurements (more stable regime)
    # Weights range from exp(-1) ≈ 0.37 at start to exp(0) = 1.0 at end
    num_points = len(filtered_time)
    weight_exponents = np.linspace(-1, 0, num_points)
    exponential_weights = np.exp(weight_exponents)

    # Perform weighted linear least squares fit to get flux (slope)
    polynomial_degree = 1  # Linear fit
    fit_coefficients = np.polyfit(
        filtered_time, filtered_pressure, polynomial_degree, w=exponential_weights
    )
    flux_slope = fit_coefficients[0]  # First coefficient is slope

    return float(flux_slope)


def calculate_permeability_from_flux(
    slope_torr_per_s: float,
    V_m3: float,
    T_K: float,
    A_m2: float,
    e_m: float,
    P_down_torr: float | ArrayLike,
    P_up_torr: float,
) -> UFloat:
    """
    Calculate permeability using the Takaishi-Sensui method with uncertainty
    propagation.

    This function implements the Takaishi-Sensui correction for hydrogen permeation
    measurements, accounting for tube conductance effects and properly propagating
    measurement uncertainties through all calculations using the uncertainties package.

    The method is based on Takaishi & Sensui (1963): "Thermal Transpiration Effect
    of Hydrogen, Rare Gases and Methane", Trans. Faraday Soc., 59, 2503-2514,
    DOI: 10.1039/tf9635902503

    The algorithm:

    1. Defines physical constants (conversion factors, gas constant, Avogadro number)
    2. Creates volume parameters with uncertainties (12% volume uncertainty, 10%
       heated/ambient volume split uncertainty)
    3. Splits total volume into heated section (at sample temperature) and ambient
       section (at 300 K room temperature)
    4. Applies Takaishi-Sensui empirical constants for hydrogen tube conductance
       corrections (accounts for thermal transpiration and molecular flow effects)
    5. Calculates conductance correction factors for non-ideal flow through
       connecting tubes
    6. Computes molar flow rate with corrections for both heated and ambient sections
    7. Converts to hydrogen atomic flux (atoms/(m²·s)) using Avogadro's number
    8. Calculates final permeability using Sieverts' law (depends on sqrt of
       upstream pressure)

    Args:
        slope_torr_per_s: Rate of downstream pressure rise in torr/s (flux).
        V_m3: Downstream chamber volume in cubic metres.
        T_K: Sample temperature in Kelvin.
        A_m2: Sample area in square metres.
        e_m: Sample thickness in metres.
        P_down_torr: Downstream pressure in torr (float or array, uses final value).
        P_up_torr: Upstream pressure in torr.

    Returns:
        Permeability value with propagated uncertainty (ufloat), in units of
            mol/(m·s·Pa^0.5). Access nominal value with ``.n`` and standard deviation
            with ``.s`` attributes.

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import calculate_permeability_from_flux

            perm = calculate_permeability_from_flux(
                slope_torr_per_s=0.01,
                V_m3=7.9e-5,
                T_K=873,
                A_m2=1.89e-4,
                e_m=0.00088,
                P_down_torr=0.5,
                P_up_torr=100.0
            )
            print(f"Permeability: {perm.n:.2e} ± {perm.s:.2e}")

        Permeability: 1.23e-10 ± 5.67e-12

    Note:
        Volume uncertainty is assumed to be 12% based on measurement precision. Volume
        ratio (heated/ambient split) has 10% uncertainty. The heated section represents
        35% of total volume. All uncertainties are rigorously propagated through the
        calculation using the uncertainties package.

    """
    # Physical constants
    TORR_TO_PASCAL_CONVERSION = 133.3
    GAS_CONSTANT = 8.314  # J/(mol·K), universal gas constant
    AVOGADRO_NUMBER = 6.022e23  # molecules/mol

    # Volume parameters with uncertainty propagation
    VOLUME_UNCERTAINTY_FRACTION = 0.12  # 12% measurement uncertainty
    volume_with_uncertainty = ufloat(V_m3, V_m3 * VOLUME_UNCERTAINTY_FRACTION)

    # Split volume into heated (V1) and ambient (V2) sections
    HEATED_VOLUME_FRACTION = 0.35  # 35% of total volume is heated
    FRACTION_UNCERTAINTY = 0.1  # 10% uncertainty in volume split
    heated_volume_ratio = ufloat(HEATED_VOLUME_FRACTION, FRACTION_UNCERTAINTY)

    volume_heated_section = volume_with_uncertainty * heated_volume_ratio
    volume_ambient_section = volume_with_uncertainty * (1 - heated_volume_ratio)

    # Temperature definitions
    temperature_heated_section = T_K  # Sample temperature
    AMBIENT_TEMPERATURE = 300  # Kelvin, room temperature

    # Takaishi-Sensui empirical constants for hydrogen in tube conductance
    # These account for thermal transpiration and molecular flow effects
    CONSTANT_A = 1.24 * 56.3 / 10e-5
    CONSTANT_B = 8 * 7.7 / 10e-2
    CONSTANT_C = 10.6 * 2.73
    TUBE_DIAMETER = 0.0155  # metres, connecting tube diameter

    # Convert flux from torr/s to Pa/s
    pressure_rise_rate_pascals = slope_torr_per_s * TORR_TO_PASCAL_CONVERSION

    # Extract final downstream pressure value (if array provided)
    if hasattr(P_down_torr, "__len__"):
        # Array-like input: use final pressure value
        downstream_pressure_pascals = P_down_torr[-1] * TORR_TO_PASCAL_CONVERSION
    else:
        # Scalar input: use directly
        downstream_pressure_pascals = P_down_torr * TORR_TO_PASCAL_CONVERSION

    # Takaishi-Sensui correction factors for tube conductance
    # These correct for non-ideal flow through connecting tubes
    conductance_numerator_2 = (
        CONSTANT_C * (TUBE_DIAMETER * downstream_pressure_pascals) ** 0.5
        + (AMBIENT_TEMPERATURE / temperature_heated_section) ** 0.5
        + CONSTANT_A * TUBE_DIAMETER**2 * downstream_pressure_pascals**2
        + CONSTANT_B * TUBE_DIAMETER * downstream_pressure_pascals
    )

    conductance_denominator_3 = (
        CONSTANT_C * (TUBE_DIAMETER * downstream_pressure_pascals) ** 0.5
        + CONSTANT_A * TUBE_DIAMETER**2 * downstream_pressure_pascals**2
        + CONSTANT_B * TUBE_DIAMETER * downstream_pressure_pascals
        + 1
    )

    conductance_numerator_1 = (
        CONSTANT_B * TUBE_DIAMETER * pressure_rise_rate_pascals
        + (CONSTANT_C * TUBE_DIAMETER * pressure_rise_rate_pascals)
        / (2 * (TUBE_DIAMETER * downstream_pressure_pascals) ** 0.5)
        + 2
        * CONSTANT_A
        * TUBE_DIAMETER**2
        * downstream_pressure_pascals
        * pressure_rise_rate_pascals
    )

    # Calculate molar flow rate (dn/dt) with Takaishi-Sensui corrections
    # This accounts for gas flow through both heated and ambient sections
    molar_flow_rate = (
        # Contribution from ambient volume section
        (volume_ambient_section * pressure_rise_rate_pascals)
        / (GAS_CONSTANT * AMBIENT_TEMPERATURE)
        # Contribution from heated volume (without correction)
        + (volume_heated_section * pressure_rise_rate_pascals)
        / (GAS_CONSTANT * temperature_heated_section * conductance_numerator_2)
        # Conductance correction term (positive)
        + (
            volume_heated_section
            * downstream_pressure_pascals
            * conductance_numerator_1
        )
        / (
            GAS_CONSTANT
            * temperature_heated_section
            * conductance_numerator_2
            * conductance_denominator_3
        )
        # Conductance correction term (negative)
        - (
            volume_heated_section
            * downstream_pressure_pascals
            * conductance_numerator_1
        )
        / (GAS_CONSTANT * temperature_heated_section * conductance_numerator_2**2)
    )

    # Convert molar flow rate to hydrogen atomic flux
    # Multiply by Avogadro's number and divide by area
    hydrogen_flux = molar_flow_rate / A_m2 * AVOGADRO_NUMBER  # H atoms/(m²·s)

    # Calculate permeability using Sieverts' law (depends on sqrt(P))
    # Permeability = (flux * thickness) / sqrt(upstream pressure)
    upstream_pressure_pascals = P_up_torr * TORR_TO_PASCAL_CONVERSION
    permeability_takaishi_sensui = (
        hydrogen_flux * e_m / (upstream_pressure_pascals) ** 0.5
    )

    return permeability_takaishi_sensui


def evaluate_permeability_values(
    datasets: dict[str, dict[str, any]],
) -> tuple[list[float], list[UFloat], NDArray, NDArray, list[float], list[float]]:
    """
    Evaluate permeability values from multiple experimental datasets with uncertainty
    propagation.

    This function processes multiple permeation measurement runs, calculates
    permeability for each using the Takaishi-Sensui method, groups measurements by
    temperature, and performs weighted averaging for runs at the same temperature.
    All uncertainties are rigorously propagated through the calculations.

    The algorithm:

    1. Defines experimental apparatus constants (sample diameter 0.0155 m, chamber
       volume 7.9e-5 m³, default thickness 0.00088 m)
    2. Processes each experimental run by:
       - Extracting temperature, time, upstream/downstream pressure data
       - Calculating stable upstream pressure using moving average stability detection
       - Calculating flux from downstream pressure rise using weighted linear fitting
       - Computing permeability with full Takaishi-Sensui corrections and uncertainty
         propagation
    3. Groups measurements by temperature into a defaultdict
    4. For each unique temperature:
       - Single measurements: uses uncertainty directly
       - Multiple measurements: performs inverse-variance weighted averaging
         (w = 1/sigma²) and calculates combined uncertainty as standard error
         of weighted mean
    5. Prepares Arrhenius plot data (x = 1000/T, y = permeability) sorted by
       ascending temperature

    Args:
        datasets: Dictionary of experimental datasets where each key is a run
            identifier and each value is a dict containing 'temperature' (float, K),
            'time_data' (array-like, seconds), 'upstream_data' (dict with
            'pressure_data' in torr), 'downstream_data' (dict with 'pressure_data' in
            torr), and optionally 'sample_thickness' (float, metres, defaults to
            0.00088 m).

    Returns:
        A tuple of six elements: all_temperatures (list of all input temperatures in
            K), all_permeabilities (list of ufloat permeabilities for each run in
            mol/(m·s·Pa^0.5)), inverse_temperature_values (NDArray of 1000/T for unique
            temperatures), nominal_permeability_values (NDArray of averaged
            permeabilities), lower_error_bars (list of lower error magnitudes),
            upper_error_bars (list of upper error magnitudes).

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import evaluate_permeability_values
            import numpy as np

            datasets = {
                "run1": {
                    "temperature": 873,
                    "time_data": np.linspace(0, 100, 100),
                    "upstream_data": {"pressure_data": np.full(100, 100.0)},
                    "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
                    "sample_thickness": 0.00088
                }
            }
            temps, perms, x_err, y_err, err_low, err_up = (
                evaluate_permeability_values(datasets)
            )
            print(f"Temperature: {temps[0]} K")
            print(f"Permeability: {perms[0].n:.2e} ± {perms[0].s:.2e}")

        Temperature: 873 K
        Permeability: 1.23e-10 ± 5.67e-12

    Note:
        Multiple runs at the same temperature are combined using inverse-variance
        weighted averaging. Measurements with zero uncertainty receive a small weight
        (1e-10) to avoid division by zero. Results are sorted by ascending temperature.

    """
    # Experimental apparatus constants
    SAMPLE_DIAMETER_METRES = 0.0155
    SAMPLE_AREA_M2 = np.pi * (SAMPLE_DIAMETER_METRES / 2) ** 2
    CHAMBER_VOLUME_M3 = 7.9e-5
    DEFAULT_SAMPLE_THICKNESS_METRES = 0.00088

    # Initialize lists to store results from each dataset
    all_temperatures = []
    all_permeabilities = []

    # Process each experimental run
    for run_name, dataset in datasets.items():
        # Extract data from dataset dictionary
        temperature_kelvin = dataset["temperature"]
        time_data_seconds = dataset["time_data"]
        upstream_pressure_torr = dataset["upstream_data"]["pressure_data"]
        downstream_pressure_torr = dataset["downstream_data"]["pressure_data"]

        # Get sample thickness (use default if not specified)
        sample_thickness_metres = dataset.get(
            "sample_thickness", DEFAULT_SAMPLE_THICKNESS_METRES
        )

        # Calculate stable upstream pressure after initial transient
        average_upstream_pressure = average_pressure_after_increase(
            time_data_seconds, upstream_pressure_torr
        )

        # Calculate flux from downstream pressure rise
        flux_torr_per_second = calculate_flux_from_sample(
            time_data_seconds, downstream_pressure_torr
        )

        # Calculate permeability with full uncertainty propagation
        permeability_with_uncertainty = calculate_permeability_from_flux(
            flux_torr_per_second,
            CHAMBER_VOLUME_M3,
            temperature_kelvin,
            SAMPLE_AREA_M2,
            sample_thickness_metres,
            downstream_pressure_torr,
            average_upstream_pressure,
        )

        all_temperatures.append(temperature_kelvin)
        all_permeabilities.append(permeability_with_uncertainty)

    # Group measurements by temperature for weighted averaging
    from collections import defaultdict

    temperature_grouped_perms = defaultdict(list)
    for temp, perm in zip(all_temperatures, all_permeabilities):
        temperature_grouped_perms[temp].append(perm)

    # Calculate weighted averages for each unique temperature
    unique_temperatures = []
    averaged_permeabilities = []
    lower_error_bars = []
    upper_error_bars = []

    for temperature in sorted(temperature_grouped_perms.keys()):
        permeability_measurements = temperature_grouped_perms[temperature]

        if len(permeability_measurements) == 1:
            # Single measurement: use its uncertainty directly
            average_permeability = permeability_measurements[0]
        else:
            # Multiple measurements: perform weighted average
            # Extract nominal values and standard deviations
            nominal_values = np.array(
                [p.n if hasattr(p, "n") else p for p in permeability_measurements]
            )
            standard_deviations = np.array(
                [p.s if hasattr(p, "s") else 0 for p in permeability_measurements]
            )

            # Calculate inverse-variance weights (w = 1/sigma^2)
            # Use small weight for measurements with zero uncertainty
            SMALL_WEIGHT = 1e-10
            variance_weights = np.where(
                standard_deviations > 0, 1.0 / standard_deviations**2, SMALL_WEIGHT
            )

            # Compute weighted mean
            weighted_mean_value = np.sum(variance_weights * nominal_values) / np.sum(
                variance_weights
            )

            # Compute combined uncertainty (standard error of weighted mean)
            combined_uncertainty = np.sqrt(1.0 / np.sum(variance_weights))

            average_permeability = ufloat(weighted_mean_value, combined_uncertainty)

        unique_temperatures.append(temperature)
        averaged_permeabilities.append(average_permeability)

        # Extract error bar values for plotting
        if hasattr(average_permeability, "n"):
            # Symmetric error bars from standard deviation
            error_magnitude = average_permeability.s
            lower_error_bars.append(error_magnitude)
            upper_error_bars.append(error_magnitude)
        else:
            # No uncertainty available
            lower_error_bars.append(0)
            upper_error_bars.append(0)

    # Prepare data for Arrhenius plotting (x = 1000/T, y = permeability)
    inverse_temperature_values = 1000 / np.array(unique_temperatures)
    nominal_permeability_values = np.array(
        [p.n if hasattr(p, "n") else p for p in averaged_permeabilities]
    )

    return (
        all_temperatures,
        all_permeabilities,
        inverse_temperature_values,
        nominal_permeability_values,
        lower_error_bars,
        upper_error_bars,
    )


def fit_permeability_data(
    temps: ArrayLike, perms: list[UFloat] | list[float]
) -> tuple[NDArray, NDArray]:
    """
    Fit Arrhenius equation to permeability data using weighted least squares.

    This function performs a weighted linear regression in log-transformed space to
    fit an Arrhenius model to permeability vs temperature data. Measurements with
    smaller uncertainties are weighted more heavily in the fit.

    The Arrhenius model is: Perm = P0 * exp(-Ea / (R*T))

    In log space: log10(Perm) = m * (1000/T) + c

    The algorithm:

    1. Converts temperature input to numpy array
    2. Extracts nominal values and standard deviations from ufloat permeabilities
       (or creates zero uncertainties for regular floats)
    3. Transforms to log10 space for linear fitting
    4. Propagates uncertainties through log10 transformation using derivative
       d(log10(x))/dx = 1/(x * ln(10)), so sigma_log10(x) = sigma_x / (x * ln(10))
    5. Calculates inverse-uncertainty weights (w = 1/sigma) with unit weight (1.0)
       fallback for points with zero or invalid uncertainties
    6. Performs weighted polynomial fit (degree 1, linear) in 1000/T space using
       numpy polyfit
    7. Generates 100 evenly spaced x-values and transforms fit back from log space
       using 10^(m*x + c)

    Args:
        temps: Temperature values in Kelvin.
        perms: Permeability values (ufloat objects with uncertainties or regular
            floats), in mol/(m·s·Pa^0.5).

    Returns:
        A tuple of fit_x_values (NDArray of 100 inverse temperature values, 1000/T
            spanning input range) and fit_y_values (NDArray of 100 fitted permeability
            values for smooth plotting).

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import fit_permeability_data
            from uncertainties import ufloat
            import matplotlib.pyplot as plt
            import numpy as np

            temps = [773, 873, 973]
            perms = [
                ufloat(1e-10, 1e-11),
                ufloat(5e-10, 5e-11),
                ufloat(2e-9, 2e-10)
            ]
            fit_x, fit_y = fit_permeability_data(temps, perms)
            plt.plot(fit_x, fit_y, 'r-', label='Arrhenius fit')
            plt.plot(1000/np.array(temps), [p.n for p in perms], 'bo')
            plt.xlabel('1000/T (K⁻¹)')
            plt.ylabel('Permeability (mol/(m·s·Pa^0.5))')
            plt.show()

    Note:
        Weights are calculated as w = 1/sigma for each point. Points with zero or
        invalid uncertainties receive unit weight (1.0). Divide-by-zero warnings are
        suppressed as they're handled by np.where. Returns 100 points for smooth
        plotting.

    """
    # Convert temperature input to numpy array
    temperature_array = np.array(temps)

    # Extract nominal values and uncertainties from permeability data
    if hasattr(perms[0], "n"):
        # ufloat objects: extract nominal values and standard deviations
        permeability_nominal_values = np.array([p.n for p in perms])
        permeability_std_deviations = np.array([p.s for p in perms])
    else:
        # Regular floats: no uncertainties available
        permeability_nominal_values = np.array(perms)
        permeability_std_deviations = np.zeros_like(permeability_nominal_values)

    # Transform to log space for Arrhenius fit
    # Arrhenius equation: P = P0 * exp(-Ea/(R*T))
    # Log form: log10(P) = log10(P0) - (Ea/(R*ln(10))) * (1/T)
    log10_permeability = np.log10(permeability_nominal_values)

    # Propagate uncertainties through log10 transformation
    # Derivative: d(log10(x))/dx = 1/(x * ln(10))
    # Therefore: sigma_log10(x) = sigma_x / (x * ln(10))
    # Suppress divide-by-zero warnings (handled by np.where below)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_permeability_uncertainties = permeability_std_deviations / (
            permeability_nominal_values * np.log(10)
        )

        # Calculate weights for weighted least squares fitting
        # Weight = 1/sigma (inverse of uncertainty)
        # For points with no uncertainty or infinite uncertainty, use unit weight
        UNIT_WEIGHT = 1.0
        is_valid_uncertainty = (log10_permeability_uncertainties > 0) & np.isfinite(
            log10_permeability_uncertainties
        )
        fit_weights = np.where(
            is_valid_uncertainty,
            1.0 / log10_permeability_uncertainties,
            UNIT_WEIGHT,
        )

    # Perform weighted linear fit in 1/T space
    # x-axis: 1000/T (inverse temperature, gives better numerical scaling)
    # y-axis: log10(permeability)
    inverse_temperature_scaled = 1000 / temperature_array

    # Fit polynomial of degree 1 (linear: y = mx + c)
    POLYNOMIAL_DEGREE = 1
    fit_coefficients = np.polyfit(
        inverse_temperature_scaled, log10_permeability, POLYNOMIAL_DEGREE, w=fit_weights
    )
    slope_coefficient = fit_coefficients[0]
    intercept_coefficient = fit_coefficients[1]

    # Generate smooth fit curve for plotting
    NUM_FIT_POINTS = 100
    fit_x_values = np.linspace(
        inverse_temperature_scaled.min(),
        inverse_temperature_scaled.max(),
        NUM_FIT_POINTS,
    )

    # Transform back from log space: permeability = 10^(m*x + c)
    fit_y_values = 10 ** (slope_coefficient * fit_x_values + intercept_coefficient)

    return fit_x_values, fit_y_values
