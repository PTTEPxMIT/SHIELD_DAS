import base64

import pandas as pd
import pytest

from shield_das import DataPlotter


class TestDataPlotter:
    """Test class for DataPlotter pure Python functionality."""

    def setup_method(self):
        """Set up a DataPlotter instance for each test."""
        self.plotter = DataPlotter()

    # Tests for get_next_color
    def test_get_next_color_first_color(self):
        """Test that index 0 returns black."""
        color = self.plotter.get_next_color(0)
        assert color == "#000000"

    def test_get_next_color_second_color(self):
        """Test that index 1 returns magenta."""
        color = self.plotter.get_next_color(1)
        assert color == "#DF1AD2"

    def test_get_next_color_all_colors(self):
        """Test all 8 defined colors are returned correctly."""
        expected_colors = [
            "#000000",  # Black
            "#DF1AD2",  # Magenta
            "#779BE7",  # Light Blue
            "#49B6FF",  # Blue
            "#254E70",  # Dark Blue
            "#0CCA4A",  # Green
            "#929487",  # Gray
            "#A1B0AB",  # Light Gray
        ]

        for i, expected in enumerate(expected_colors):
            assert self.plotter.get_next_color(i) == expected

    def test_get_next_color_cycles_after_eight(self):
        """Test that colors cycle back to beginning after 8."""
        # Index 8 should return the same as index 0
        assert self.plotter.get_next_color(8) == self.plotter.get_next_color(0)
        assert self.plotter.get_next_color(9) == self.plotter.get_next_color(1)

    def test_get_next_color_large_index(self):
        """Test color cycling with large indices."""
        # Index 16 should be same as 0 (16 % 8 = 0)
        assert self.plotter.get_next_color(16) == "#000000"
        # Index 25 should be same as 1 (25 % 8 = 1)
        assert self.plotter.get_next_color(25) == "#DF1AD2"

    # Tests for is_valid_color
    def test_is_valid_color_valid_hex_6_digit(self):
        """Test valid 6-digit hex colors."""
        assert self.plotter.is_valid_color("#000000") is True
        assert self.plotter.is_valid_color("#FFFFFF") is True
        assert self.plotter.is_valid_color("#ff6600") is True
        assert self.plotter.is_valid_color("#ABC123") is True

    def test_is_valid_color_valid_hex_3_digit(self):
        """Test valid 3-digit hex colors."""
        assert self.plotter.is_valid_color("#000") is True
        assert self.plotter.is_valid_color("#FFF") is True
        assert self.plotter.is_valid_color("#f60") is True
        assert self.plotter.is_valid_color("#A1B") is True

    def test_is_valid_color_valid_rgb(self):
        """Test valid RGB color format."""
        assert self.plotter.is_valid_color("rgb(0,0,0)") is True
        assert self.plotter.is_valid_color("rgb(255,255,255)") is True
        assert self.plotter.is_valid_color("rgb(128, 64, 192)") is True
        assert (
            self.plotter.is_valid_color("RGB(100, 200, 50)") is True
        )  # Case insensitive

    def test_is_valid_color_invalid_hex(self):
        """Test invalid hex color formats."""
        assert self.plotter.is_valid_color("#GGG") is False  # Invalid hex chars
        assert self.plotter.is_valid_color("#12345") is False  # Wrong length
        assert self.plotter.is_valid_color("#1234567") is False  # Too long
        assert self.plotter.is_valid_color("123456") is False  # Missing #
        assert self.plotter.is_valid_color("#XY1234") is False  # Invalid chars

    def test_is_valid_color_invalid_rgb(self):
        """Test invalid RGB color formats."""
        assert self.plotter.is_valid_color("rgb(256,0,0)") is False  # Out of range
        assert self.plotter.is_valid_color("rgb(-1,0,0)") is False  # Negative
        assert self.plotter.is_valid_color("rgb(1,2)") is False  # Missing value
        assert self.plotter.is_valid_color("rgb(1,2,3,4)") is False  # Too many values
        assert self.plotter.is_valid_color("rgba(1,2,3,0.5)") is False  # Not supported

    def test_is_valid_color_empty_and_none(self):
        """Test empty, None, and whitespace inputs."""
        assert self.plotter.is_valid_color("") is False
        assert self.plotter.is_valid_color(None) is False
        assert self.plotter.is_valid_color("   ") is False

    def test_is_valid_color_with_whitespace(self):
        """Test color validation with whitespace (should be stripped)."""
        assert self.plotter.is_valid_color("  #000000  ") is True
        assert self.plotter.is_valid_color(" rgb(0,0,0) ") is True

    # Tests for convert_timestamps_to_seconds
    def test_convert_timestamps_empty_list(self):
        """Test conversion with empty input."""
        result = self.plotter.convert_timestamps_to_seconds([])
        assert result == []

    def test_convert_timestamps_single_value(self):
        """Test conversion with single timestamp."""
        result = self.plotter.convert_timestamps_to_seconds(["123.45"])
        assert result == [123.45]

    def test_convert_timestamps_multiple_values(self):
        """Test conversion with multiple timestamps."""
        input_timestamps = ["0", "1.5", "3.0", "4.25"]
        result = self.plotter.convert_timestamps_to_seconds(input_timestamps)
        expected = [0.0, 1.5, 3.0, 4.25]
        assert result == expected

    def test_convert_timestamps_integer_strings(self):
        """Test conversion with integer string timestamps."""
        input_timestamps = ["0", "1", "2", "3"]
        result = self.plotter.convert_timestamps_to_seconds(input_timestamps)
        expected = [0.0, 1.0, 2.0, 3.0]
        assert result == expected

    def test_convert_timestamps_float_strings(self):
        """Test conversion with float string timestamps."""
        input_timestamps = ["0.0", "1.234", "2.567", "3.999"]
        result = self.plotter.convert_timestamps_to_seconds(input_timestamps)
        expected = [0.0, 1.234, 2.567, 3.999]
        assert result == expected

    def test_convert_timestamps_none_input(self):
        """Test conversion with None input."""
        result = self.plotter.convert_timestamps_to_seconds(None)
        assert result == []


# Additional edge case tests
class TestDataPlotterEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up a DataPlotter instance for each test."""
        self.plotter = DataPlotter()

    def test_get_next_color_negative_index(self):
        """Test get_next_color with negative index."""
        # Python's modulo should handle this gracefully
        color = self.plotter.get_next_color(-1)
        assert color == "#A1B0AB"  # -1 % 8 = 7, so last color

    def test_convert_timestamps_invalid_strings(self):
        """Test convert_timestamps with non-numeric strings."""
        # This should raise ValueError since we're calling float()
        with pytest.raises(ValueError):
            self.plotter.convert_timestamps_to_seconds(["not_a_number"])

    def test_is_valid_color_edge_rgb_values(self):
        """Test RGB validation at boundary values."""
        assert self.plotter.is_valid_color("rgb(0,0,0)") is True
        assert self.plotter.is_valid_color("rgb(255,255,255)") is True
        assert self.plotter.is_valid_color("rgb(0,255,0)") is True
        assert self.plotter.is_valid_color("rgb(256,0,0)") is False  # Just over limit
        assert self.plotter.is_valid_color("rgb(-1,0,0)") is False  # Just under limit
