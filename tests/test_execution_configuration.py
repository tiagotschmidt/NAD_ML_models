"""
This module contains the unit tests for the ExecutionConfiguration class.
"""

import pytest

from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.framework_parameters import Lifecycle, Platform


class TestExecutionConfiguration:
    """Groups all tests for the ExecutionConfiguration class."""

    @pytest.mark.parametrize(
        "layers, units, epochs, features, sampling_rate, platform, cycle",
        [
            # Test Case 1: Typical CPU training run
            (4, 128, 50, 20, 1.0, Platform.CPU, Lifecycle.TRAIN),
            # Test Case 2: GPU testing run with different values
            (8, 256, 1, 30, 0.8, Platform.GPU, Lifecycle.TEST),
            # Edge Case: Minimum values
            (1, 1, 1, 1, 0.1, Platform.CPU, Lifecycle.TRAIN),
        ],
    )
    def test_initialization_and_attributes(
        self, layers, units, epochs, features, sampling_rate, platform, cycle
    ):
        """
        Tests that the __init__ method correctly assigns all attributes for
        various valid configurations.
        """
        # 1. Create an instance of the class with parameterized data
        config = ExecutionConfiguration(
            layers=layers,
            units=units,
            number_of_epochs=epochs,
            features=features,
            sampling_rate=sampling_rate,
            platform=platform,
            cycle=cycle,
        )

        # 2. Assert that all attributes were stored correctly
        assert config.number_of_layers == layers
        assert config.number_of_units == units
        assert config.number_of_epochs == epochs
        assert config.number_of_features == features
        assert config.sampling_rate == sampling_rate
        assert config.platform == platform
        assert config.cycle == cycle

    def test_str_representation(self):
        """
        Tests that the __str__ method produces a correctly formatted and
        accurate string.
        """
        # 1. Create a specific configuration
        config = ExecutionConfiguration(
            layers=5,
            units=100,
            number_of_epochs=25,
            features=15,
            sampling_rate=0.9,
            platform=Platform.GPU,
            cycle=Lifecycle.TRAIN,
        )

        # 2. Generate the expected string
        # Note: We access the .name attribute of the enums for the string representation
        expected_str = (
            "ExecutionConfiguration("
            "layers=5, "
            "units=100, "
            "epochs=25, "
            "features=15, "
            "sampling_rate=0.9, "
            "platform=GPU, "
            "cycle=TRAIN"
            ")"
        )

        # 3. Assert that the __str__ output matches the expected format and content
        assert str(config) == expected_str
