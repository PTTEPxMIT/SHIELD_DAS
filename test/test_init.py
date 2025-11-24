"""Tests for the shield_das __init__ module.

Tests versioning and public API imports.
"""


def test_version_attribute_exists():
    """Test that __version__ attribute exists."""
    import shield_das

    assert hasattr(shield_das, "__version__")
    assert isinstance(shield_das.__version__, str)


def test_version_format():
    """Test that __version__ has expected format or is 'unknown'."""
    import shield_das

    # Version should be either "unknown" or follow semantic versioning pattern
    assert (
        shield_das.__version__ == "unknown"
        or len(shield_das.__version__.split(".")) >= 2
    )


def test_all_attribute_exists():
    """Test that __all__ attribute exists and is a list."""
    import shield_das

    assert hasattr(shield_das, "__all__")
    assert isinstance(shield_das.__all__, list)
    assert len(shield_das.__all__) > 0


def test_all_exports_are_strings():
    """Test that all items in __all__ are strings."""
    import shield_das

    for item in shield_das.__all__:
        assert isinstance(item, str), f"Item {item} in __all__ is not a string"


def test_all_exports_are_importable():
    """Test that all items in __all__ can be imported."""
    import shield_das

    for item in shield_das.__all__:
        assert hasattr(shield_das, item), f"Item '{item}' in __all__ is not importable"


def test_analysis_functions_exported():
    """Test that key analysis functions are exported."""
    import shield_das

    expected_functions = [
        "voltage_to_pressure",
        "voltage_to_temperature",
        "calculate_error_on_pressure_reading",
        "calculate_flux_from_sample",
        "calculate_permeability_from_flux",
        "fit_permeability_data",
        "average_pressure_after_increase",
        "evaluate_permeability_values",
    ]

    for func in expected_functions:
        assert func in shield_das.__all__, f"Function '{func}' not in __all__"
        assert hasattr(shield_das, func), f"Function '{func}' not importable"


def test_core_classes_exported():
    """Test that core DAS classes are exported."""
    import shield_das

    expected_classes = [
        "DataPlotter",
        "DataRecorder",
        "Dataset",
        "PressureGauge",
        "Thermocouple",
        "Baratron626D_Gauge",
        "CVM211_Gauge",
        "WGM701_Gauge",
    ]

    for cls in expected_classes:
        assert cls in shield_das.__all__, f"Class '{cls}' not in __all__"
        assert hasattr(shield_das, cls), f"Class '{cls}' not importable"


def test_plotting_functions_not_exported():
    """Test that plotting functions are NOT exported in __all__."""
    import shield_das

    plotting_functions = [
        "plot_pressure_from_voltage",
        "plot_temperature_from_voltage",
        "plot_dual_pressure",
        "quick_pressure_plot",
    ]

    for func in plotting_functions:
        assert func not in shield_das.__all__, (
            f"Plotting function '{func}' should not be in __all__"
        )


def test_analysis_functions_callable():
    """Test that analysis functions are callable."""
    import shield_das

    analysis_functions = [
        shield_das.voltage_to_pressure,
        shield_das.voltage_to_temperature,
        shield_das.calculate_error_on_pressure_reading,
        shield_das.calculate_flux_from_sample,
    ]

    for func in analysis_functions:
        assert callable(func), f"Function {func.__name__} is not callable"


def test_core_classes_are_classes():
    """Test that core classes are actually classes."""
    import inspect

    import shield_das

    classes = [
        shield_das.DataPlotter,
        shield_das.DataRecorder,
        shield_das.Dataset,
        shield_das.PressureGauge,
        shield_das.Thermocouple,
    ]

    for cls in classes:
        assert inspect.isclass(cls), f"{cls.__name__} is not a class"


def test_direct_import_of_functions():
    """Test that analysis functions can be imported directly."""
    from shield_das import (
        calculate_error_on_pressure_reading,
        calculate_flux_from_sample,
        voltage_to_pressure,
        voltage_to_temperature,
    )

    assert callable(voltage_to_pressure)
    assert callable(voltage_to_temperature)
    assert callable(calculate_error_on_pressure_reading)
    assert callable(calculate_flux_from_sample)


def test_direct_import_of_classes():
    """Test that core classes can be imported directly."""
    import inspect

    from shield_das import (
        Baratron626D_Gauge,
        DataPlotter,
        DataRecorder,
        Dataset,
    )

    assert inspect.isclass(DataPlotter)
    assert inspect.isclass(DataRecorder)
    assert inspect.isclass(Dataset)
    assert inspect.isclass(Baratron626D_Gauge)


def test_all_list_sorted():
    """Test that __all__ list is sorted alphabetically."""
    import shield_das

    sorted_all = sorted(shield_das.__all__)
    assert shield_das.__all__ == sorted_all, (
        "__all__ list should be sorted alphabetically"
    )


def test_no_duplicate_exports():
    """Test that __all__ contains no duplicate entries."""
    import shield_das

    assert len(shield_das.__all__) == len(set(shield_das.__all__)), (
        "__all__ contains duplicate entries"
    )


def test_expected_export_count():
    """Test that we have the expected number of exports."""
    import shield_das

    # 8 analysis functions + 8 core classes/gauges = 16 total
    expected_count = 16
    actual_count = len(shield_das.__all__)

    assert actual_count == expected_count, (
        f"Expected {expected_count} exports, got {actual_count}"
    )


def test_module_docstring_exists():
    """Test that the module has a docstring (optional but good practice)."""
    import shield_das

    # This is optional, but if present should be a string
    if shield_das.__doc__ is not None:
        assert isinstance(shield_das.__doc__, str)


def test_version_fallback_to_unknown():
    """Test that version falls back to 'unknown' when metadata is unavailable."""
    import sys
    from importlib import metadata
    from unittest.mock import patch

    # Mock metadata.version to raise PackageNotFoundError
    with patch.object(metadata, "version", side_effect=metadata.PackageNotFoundError):
        # Remove the module from cache if it exists
        if "shield_das" in sys.modules:
            del sys.modules["shield_das"]

        # Import will now hit the exception path
        import shield_das

        assert shield_das.__version__ == "unknown"

    # Clean up: reload the module normally
    if "shield_das" in sys.modules:
        del sys.modules["shield_das"]
