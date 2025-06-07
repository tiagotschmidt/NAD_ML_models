import pytest
from eppnad.utils.framework_parameters import (
    FrameworkParameterType,
    PercentageRangeParameter,
    RangeMode,
    RangeParameter,
)

# A dummy FrameworkParameterType for testing purposes
DUMMY_TYPE = FrameworkParameterType.Epochs


class TestRangeParameter:
    """Groups all tests for the RangeParameter class."""

    def test_direct_initialization(self):
        """Tests that the __init__ method correctly stores a list."""
        my_list = [1, 5, 10]
        rp = RangeParameter(my_list)
        assert list(rp) == my_list, "Direct initialization should preserve the list."

    @pytest.mark.parametrize(
        "start, end, stride, expected",
        [
            # Test Case 1: Standard case
            (10, 50, 10, [10, 20, 30, 40, 50]),
            # Test Case 2: Stride doesn't align perfectly with end
            (5, 25, 7, [5, 12, 19]),
            # Edge Case: start == end
            (5, 5, 1, [5]),
            # Edge Case: start > end (should produce an empty list)
            (10, 5, 2, []),
        ],
    )
    def test_additive_mode(self, start, end, stride, expected):
        """Tests various scenarios for the Additive range mode."""
        rp = RangeParameter.from_range(
            start, end, stride, DUMMY_TYPE, RangeMode.Additive
        )
        assert list(rp) == expected

    def test_additive_mode_zero_stride_raises_error(self):
        """Tests that a stride of 0 in Additive mode raises a ValueError."""
        with pytest.raises(ValueError, match=r"range\(\) arg 3 must not be zero"):
            # The error is raised inside the list() constructor which consumes the range generator
            list(RangeParameter.from_range(1, 10, 0, DUMMY_TYPE, RangeMode.Additive))

    @pytest.mark.parametrize(
        "start, end, stride, expected",
        [
            # Test Case 1: Standard case
            (2, 30, 2, [2, 4, 8, 16]),
            # Edge Case: start == end
            (8, 8, 2, [8]),
            # Edge Case: start > end (should produce an empty list)
            (10, 5, 2, []),
            # Edge Case: Stride of 1 (should not get stuck in an infinite loop)
            (1, 10, 1, [1]),
            # Edge Case: Start value of 1
            (1, 10, 3, [1, 3, 9]),
        ],
    )
    def test_multiplicative_mode(self, start, end, stride, expected):
        """Tests various scenarios for the Multiplicative range mode."""
        rp = RangeParameter.from_range(
            start, end, stride, DUMMY_TYPE, RangeMode.Multiplicative
        )
        assert list(rp) == expected

    def test_multiplicative_mode_zero_start_raises_error(self):
        """
        Tests that a start value of 0 in Multiplicative mode raises an error
        to prevent an infinite loop.
        """
        with pytest.raises(ValueError, match="Multiplicative mode start cannot be 0"):
            RangeParameter.from_range(0, 10, 2, DUMMY_TYPE, RangeMode.Multiplicative)

    def test_multiplicative_mode_zero_stride_raises_error(self):
        """Tests that a stride of 0 in Multiplicative mode raises an error."""
        with pytest.raises(ValueError, match="Multiplicative mode stride cannot be 0"):
            RangeParameter.from_range(1, 10, 0, DUMMY_TYPE, RangeMode.Multiplicative)


class TestPercentageRangeParameter:
    """Groups all tests for the PercentageRangeParameter class."""

    def test_initialization_with_valid_percentages(self):
        """
        Tests that the class can be initialized with a list of valid percentages.
        """
        valid_list = [0.0, 0.25, 0.5, 0.75, 1.0]
        try:
            pr = PercentageRangeParameter(valid_list)
            assert list(pr) == valid_list, "Valid list should be stored correctly."
        except ValueError:
            pytest.fail("ValueError was raised unexpectedly for a valid list.")

    def test_initialization_with_value_too_high_raises_error(self):
        """
        Tests that a ValueError is raised if a percentage is greater than 1.0.
        """
        invalid_list = [0.1, 0.5, 1.01]  # Contains an invalid value
        with pytest.raises(ValueError, match="All items must be between 0.0 and 1.0"):
            PercentageRangeParameter(invalid_list)

    def test_initialization_with_value_too_low_raises_error(self):
        """
        Tests that a ValueError is raised if a percentage is less than 0.0.
        """
        invalid_list = [-0.01, 0.5, 0.9]  # Contains an invalid value
        with pytest.raises(ValueError, match="All items must be between 0.0 and 1.0"):
            PercentageRangeParameter(invalid_list)

    def test_from_steps_factory_valid(self):
        """
        Tests that the from_steps factory method generates a correct list.
        """
        # 11 steps should generate [0.0, 0.1, 0.2, ..., 1.0]
        pr = PercentageRangeParameter.from_steps(num_steps=11)
        result_list = list(pr)
        assert len(result_list) == 11
        assert result_list[0] == 0.0
        assert result_list[-1] == 1.0
        # Using pytest.approx for safe float comparison
        assert result_list[1] == pytest.approx(0.1)

    @pytest.mark.parametrize("invalid_steps", [0, 1, -5])
    def test_from_steps_factory_invalid_steps_raises_error(self, invalid_steps):
        """
        Tests that from_steps raises a ValueError for an invalid number of steps.
        """
        with pytest.raises(ValueError, match="num_steps must be 2 or greater"):
            PercentageRangeParameter.from_steps(num_steps=invalid_steps)
