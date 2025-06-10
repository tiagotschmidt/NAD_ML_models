"""
This module contains the unit tests for the Manager module, with a specific
focus on the configuration generation logic.
"""

import pytest

# Assuming the function is in 'manager.py' and utilities are in 'eppnad.utils'
from eppnad import manager
from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.framework_parameters import (
    Lifecycle,
    Platform,
    ProfileMode,
    RangeParameter,
    PercentageRangeParameter,
)


class TestGenerateConfigurationsList:
    """
    Groups all tests for the private `_generate_configurations_list` function
    from the manager module.
    """

    @pytest.fixture
    def sample_parameters(self):
        """
        Provides a standard set of hyperparameter ranges for testing.
        Each range contains two values to make testing combinations predictable.
        """
        params = {
            "layers": RangeParameter([2, 4]),
            "units": RangeParameter([64, 128]),
            "epochs": RangeParameter([10]),
            "features": RangeParameter([20]),
            "sampling_rates": PercentageRangeParameter([0.8, 1.0]),
        }
        return params

    def test_generate_configurations_only_train(self, sample_parameters):
        """
        Tests that the function correctly generates configurations for a
        `ONLY_TRAIN` Lifecycle.

        It should produce a list where all configurations have the `Lifecycle.Train`
        and `Platform.CPU` settings.
        """
        # 1. Define the ProfileMode for this test case
        profile_mode = ProfileMode(cycle=Lifecycle.TRAIN, train_platform=Platform.CPU)

        # 2. Call the function under test
        configurations = manager._generate_configurations_list(
            **sample_parameters, profile_mode=profile_mode
        )

        # 3. Assert the results
        # Expected length = 2 (layers) * 2 (units) * 1 (epoch) * 1 (feature) * 2 (sampling) = 8
        assert (
            len(configurations) == 8
        ), "Should generate the correct number of configurations"

        assert all(
            isinstance(c, ExecutionConfiguration) for c in configurations
        ), "All items in the list should be ExecutionConfiguration objects"

        assert all(
            c.cycle == Lifecycle.TRAIN for c in configurations
        ), "All configurations should have the Train Lifecycle"

        assert all(
            c.platform == Platform.CPU for c in configurations
        ), "All configurations should be set to the CPU platform"

    def test_generate_configurations_only_test(self, sample_parameters):
        """
        Tests that the function correctly generates configurations for an
        `ONLY_TEST` Lifecycle.

        It should produce a list where all configurations have the `Lifecycle.Test`
        and `Platform.GPU` settings.
        """
        # 1. Define the ProfileMode for this test case
        profile_mode = ProfileMode(cycle=Lifecycle.TEST, test_platform=Platform.GPU)

        # 2. Call the function under test
        configurations = manager._generate_configurations_list(
            **sample_parameters, profile_mode=profile_mode
        )

        # 3. Assert the results
        # Expected length = 2 * 2 * 1 * 1 * 2 = 8
        assert (
            len(configurations) == 8
        ), "Should generate the correct number of configurations"

        assert all(
            c.cycle == Lifecycle.TEST for c in configurations
        ), "All configurations should have the Test Lifecycle"

        assert all(
            c.platform == Platform.GPU for c in configurations
        ), "All configurations should be set to the GPU platform"

    def test_generate_configurations_train_and_test(self, sample_parameters):
        """
        Tests that the function correctly generates configurations for a
        `TRAIN_AND_TEST` Lifecycle.

        It should produce a list containing configurations for both training
        on CPU and testing on GPU, doubling the total number of configurations.
        """
        # 1. Define the ProfileMode for this test case
        profile_mode = ProfileMode(
            cycle=Lifecycle.TRAIN_AND_TEST,
            train_platform=Platform.CPU,
            test_platform=Platform.GPU,
        )

        # 2. Call the function under test
        configurations = manager._generate_configurations_list(
            **sample_parameters, profile_mode=profile_mode
        )

        # 3. Assert the results
        # Expected length = 8 (for train) + 8 (for test) = 16
        assert (
            len(configurations) == 16
        ), "Should generate configurations for both train and test Lifecycles"

        # Check the train configurations
        train_configs = [c for c in configurations if c.cycle == Lifecycle.TRAIN]
        assert len(train_configs) == 8, "There should be 8 training configurations"
        assert all(
            c.platform == Platform.CPU for c in train_configs
        ), "All training configurations should be set to the CPU platform"

        # Check the test configurations
        test_configs = [c for c in configurations if c.cycle == Lifecycle.TEST]
        assert len(test_configs) == 8, "There should be 8 testing configurations"
        assert all(
            c.platform == Platform.GPU for c in test_configs
        ), "All testing configurations should be set to the GPU platform"

    def test_generate_configurations_with_empty_parameter_range(self):
        """
        Tests that if any hyperparameter range is empty, the function
        returns an empty list of configurations.
        """
        # 1. Define parameters with one empty list
        params = {
            "layers": RangeParameter([2, 4]),
            "units": RangeParameter([]),  # Empty list
            "epochs": RangeParameter([10]),
            "features": RangeParameter([20]),
            "sampling_rates": PercentageRangeParameter([1.0]),
        }
        profile_mode = ProfileMode(cycle=Lifecycle.TRAIN, train_platform=Platform.CPU)

        # 2. Call the function under test
        configurations = manager._generate_configurations_list(
            **params, profile_mode=profile_mode
        )

        # 3. Assert the result is an empty list
        assert (
            len(configurations) == 0
        ), "Should return an empty list if any parameter range is empty"
