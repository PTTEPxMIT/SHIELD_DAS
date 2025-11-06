import numpy as np

from shield_das.helpers import import_htm_data


class TestImportHtmData:
    """Tests for import_htm_data function"""

    def test_returns_correct_structure_for_316l_steel(self):
        """Test that function returns three lists for 316l_steel"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        assert isinstance(x_values, list)
        assert isinstance(y_values, list)
        assert isinstance(labels, list)
        assert len(x_values) == len(y_values) == len(labels)

    def test_returns_non_empty_data_for_316l_steel(self):
        """Test that 316l_steel returns non-empty data"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        assert len(x_values) > 0
        assert len(y_values) > 0
        assert len(labels) > 0

    def test_x_values_are_numpy_arrays(self):
        """Test that x_values are numpy arrays"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            assert isinstance(x, np.ndarray)
            assert len(x) > 0

    def test_y_values_are_numpy_arrays(self):
        """Test that y_values are numpy arrays"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for y in y_values:
            assert isinstance(y, np.ndarray)
            assert len(y) > 0

    def test_labels_are_strings(self):
        """Test that labels are strings"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for label in labels:
            assert isinstance(label, str)
            assert len(label) > 0

    def test_x_and_y_arrays_have_same_length(self):
        """Test that each x and y array pair has the same length"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x, y in zip(x_values, y_values):
            assert len(x) == len(y)

    def test_x_values_are_positive_temperatures(self):
        """Test that x_values (temperatures) are positive"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            assert np.all(x > 0)
            # Temperature should be in reasonable range (Kelvin)
            assert np.all(x >= 200)  # Above absolute zero
            assert np.all(x <= 2000)  # Below melting point

    def test_y_values_are_positive_permeabilities(self):
        """Test that y_values (permeabilities) are positive"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for y in y_values:
            assert np.all(y > 0)

    def test_labels_contain_author_and_year(self):
        """Test that labels contain author and year information"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for label in labels:
            # Should contain parentheses with year
            assert "(" in label
            assert ")" in label

    def test_permeability_increases_with_temperature(self):
        """Test that permeability generally increases with temperature"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x, y in zip(x_values, y_values):
            # For Arrhenius behavior, permeability increases with temperature
            # Check that last value is greater than first value
            assert y[-1] > y[0]

    def test_temperature_values_are_sorted(self):
        """Test that temperature values are in ascending order"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            # Check if sorted
            assert np.all(x[:-1] <= x[1:])

    def test_function_works_with_316l_steel_input(self):
        """Test that function successfully executes with 316l_steel"""
        # This is the main test - ensure it doesn't raise any exceptions
        try:
            x_values, y_values, labels = import_htm_data("316l_steel")
            success = True
        except Exception:
            success = False

        assert success

    def test_x_values_have_100_points(self):
        """Test that x_values arrays have 100 points as specified"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            assert len(x) == 100

    def test_y_values_have_100_points(self):
        """Test that y_values arrays have 100 points as specified"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for y in y_values:
            assert len(y) == 100

    def test_labels_are_capitalized(self):
        """Test that labels are capitalized"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for label in labels:
            # First character should be uppercase
            assert label[0].isupper()
