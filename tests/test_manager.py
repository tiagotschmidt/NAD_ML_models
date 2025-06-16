"""
This module contains the unit tests for the EPPNAD manager module.

It focuses on two key areas:
1.  The logic for generating hyperparameter configurations.
2.  The orchestration logic of the main `profile` and `intermittent_profile`
    functions, ensuring they correctly manage `RuntimeSnapshot` and trigger
    the execution pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

# Assuming the function is in 'eppnad.manager' and utilities are in 'eppnad.utils'
from eppnad.core import manager
from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.framework_parameters import (
    Lifecycle,
    Platform,
    ProfileMode,
    RangeParameter,
    PercentageRangeParameter,
)
from eppnad.utils.runtime_snapshot import RuntimeSnapshot


class TestConfigurationGeneration:
    """
    Groups all unit tests for the private `_generate_configurations_list`
    function from the manager module.
    """

    @pytest.fixture
    def sample_hyperparameters(self):
        """
        Provides a standard set of hyperparameter ranges for testing.
        Each range contains two values to make testing combinations predictable.
        """
        return {
            "layers": RangeParameter([2, 4]),
            "units": RangeParameter([64, 128]),
            "epochs": RangeParameter([10]),
            "features": RangeParameter([20]),
            "sampling_rates": PercentageRangeParameter([0.8, 1.0]),
        }

    def test_generates_configurations_for_train_only_lifecycle(
        self, sample_hyperparameters
    ):
        """
        Tests that configurations are correctly generated for a `TRAIN` only
        lifecycle, assigning the correct platform.
        """
        # 1. Define a TRAIN_ONLY profile mode
        profile_mode = ProfileMode(cycle=Lifecycle.TRAIN, train_platform=Platform.CPU)

        # 2. Call the function under test
        configurations = manager._generate_configurations_list(
            **sample_hyperparameters, profile_mode=profile_mode
        )

        # 3. Assert the results
        # (2 layers * 2 units * 1 epoch * 1 feature * 2 rates) = 8 configs
        assert len(configurations) == 8, "Should generate 8 total configurations"
        assert all(
            isinstance(c, ExecutionConfiguration) for c in configurations
        ), "All items should be ExecutionConfiguration instances"
        assert all(
            c.cycle == Lifecycle.TRAIN for c in configurations
        ), "All configurations should have the TRAIN lifecycle"
        assert all(
            c.platform == Platform.CPU for c in configurations
        ), "All configurations should be set to the CPU platform"

    def test_generates_configurations_for_test_only_lifecycle(
        self, sample_hyperparameters
    ):
        """
        Tests that configurations are correctly generated for a `TEST` only
        lifecycle, assigning the correct platform.
        """
        # 1. Define a TEST_ONLY profile mode
        profile_mode = ProfileMode(cycle=Lifecycle.TEST, test_platform=Platform.GPU)

        # 2. Call the function under test
        configurations = manager._generate_configurations_list(
            **sample_hyperparameters, profile_mode=profile_mode
        )

        # 3. Assert the results
        assert len(configurations) == 8, "Should generate 8 total configurations"
        assert all(
            c.cycle == Lifecycle.TEST for c in configurations
        ), "All configurations should have the TEST lifecycle"
        assert all(
            c.platform == Platform.GPU for c in configurations
        ), "All configurations should be set to the GPU platform"

    def test_generates_configurations_for_train_and_test_lifecycle(
        self, sample_hyperparameters
    ):
        """
        Tests that configurations are correctly generated for a combined
        `TRAIN_AND_TEST` lifecycle, creating distinct sets for each phase.
        """
        # 1. Define a TRAIN_AND_TEST profile mode
        profile_mode = ProfileMode(
            cycle=Lifecycle.TRAIN_AND_TEST,
            train_platform=Platform.CPU,
            test_platform=Platform.GPU,
        )

        # 2. Call the function under test
        configurations = manager._generate_configurations_list(
            **sample_hyperparameters, profile_mode=profile_mode
        )

        # 3. Assert the results
        assert (
            len(configurations) == 16
        ), "Should generate 16 total configurations (8 train + 8 test)"

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

    def test_returns_empty_list_for_empty_parameter_range(self):
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


class TestProfilingFunctions:
    """
    Groups all tests for the main `profile` and `intermittent_profile`
    functions. These tests use mocking to isolate the functions from their
    dependencies (e.g., file system, multiprocessing).
    """

    @pytest.fixture
    def profile_base_args(self):
        """Provides a dictionary of common, valid arguments for profiling functions."""
        return {
            "user_model_function": MagicMock(),
            "user_model_name": "test_model",
            "repeated_custom_layer_code": MagicMock(),
            "first_custom_layer_code": MagicMock(),
            "final_custom_layer_code": MagicMock(),
            "statistical_samples": 5,
        }

    @patch("eppnad.core.manager.RuntimeSnapshot")
    @patch("eppnad.core.manager.Path")
    @patch("eppnad.core.manager._execute_profiling_run")
    @patch("eppnad.core.manager.logging.FileHandler")
    def test_profile_creates_new_snapshot_and_runs(
        self,
        mock_file_handler,
        mock_execute_run,
        mock_path,
        mock_snapshot_class,
        profile_base_args,
    ):
        """
        Verifies that `profile` correctly initializes a new session by:
        1. Creating a directory.
        2. Instantiating a new `RuntimeSnapshot`.
        3. Calling the execution pipeline with the new snapshot.
        """
        mock_file_handler.return_value.level = 0

        manager.profile(**profile_base_args)

        mock_path.assert_called_with("./test_model/")
        mock_path.return_value.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )

        # Check that a new snapshot was created with index 0
        mock_snapshot_class.assert_called_once()
        snapshot_args, snapshot_kwargs = mock_snapshot_class.call_args
        assert snapshot_kwargs["last_profiled_index"] == 0
        new_snapshot_instance = mock_snapshot_class.return_value

        # Check that the execution pipeline was called with the new snapshot
        mock_execute_run.assert_called_once_with(
            profile_base_args["user_model_function"],
            "./test_model/",
            new_snapshot_instance,
            profile_base_args["statistical_samples"],
        )

    @patch("eppnad.core.manager.RuntimeSnapshot")
    @patch("eppnad.core.manager.Path")
    @patch("eppnad.core.manager._execute_profiling_run")
    @patch("eppnad.core.manager.logging.FileHandler")
    def test_intermittent_profile_resumes_from_existing_snapshot(
        self,
        mock_file_handler,
        mock_execute_run,
        mock_path,
        mock_snapshot_class,
        profile_base_args,
    ):
        """
        Verifies that `intermittent_profile` resumes a session by:
        1. Loading an existing `RuntimeSnapshot`.
        2. NOT creating a new snapshot.
        3. Calling the execution pipeline with the loaded snapshot.
        """
        # 1. Setup: Mock `load_latest` to return a mock snapshot
        mock_loaded_snapshot = MagicMock(spec=RuntimeSnapshot)
        mock_snapshot_class.load_latest.return_value = mock_loaded_snapshot

        # Instantiate a mock for the constructor to check it's NOT called
        mock_constructor = MagicMock()
        mock_snapshot_class.side_effect = mock_constructor

        mock_file_handler.return_value.level = 0

        # 2. Call the function under test
        manager.intermittent_profile(**profile_base_args)

        # 3. Assertions
        mock_path.assert_called_with("./test_model/")
        mock_snapshot_class.load_latest.assert_called_once_with("./test_model/")

        # Ensure a NEW snapshot was NOT created
        mock_constructor.assert_not_called()

        # Ensure the pipeline runs with the LOADED snapshot
        mock_execute_run.assert_called_once_with(
            profile_base_args["user_model_function"],
            "./test_model/",
            mock_loaded_snapshot,
            profile_base_args["statistical_samples"],
        )

    @patch("eppnad.core.manager.RuntimeSnapshot")
    @patch("eppnad.core.manager.Path")
    @patch("eppnad.core.manager._execute_profiling_run")
    @patch("eppnad.core.manager.logging.FileHandler")
    def test_intermittent_profile_starts_new_run_if_no_snapshot(
        self,
        mock_file_handler,
        mock_execute_run,
        mock_path,
        mock_snapshot_class,
        profile_base_args,
    ):
        """
        Verifies that `intermittent_profile` starts a new session if `load_latest`
        returns `None` by:
        1. Attempting to load a snapshot.
        2. Falling back to creating a new `RuntimeSnapshot`.
        3. Calling the execution pipeline with the new snapshot.
        """
        # 1. Setup: Mock `load_latest` to find nothing
        mock_snapshot_class.load_latest.return_value = None
        new_snapshot_instance = mock_snapshot_class.return_value

        mock_file_handler.return_value.level = 0

        # 2. Call the function under test
        manager.intermittent_profile(**profile_base_args)

        # 3. Assertions
        mock_snapshot_class.load_latest.assert_called_once_with("./test_model/")

        # Check that it fell back to creating a new snapshot
        mock_snapshot_class.assert_called_once()
        snapshot_args, snapshot_kwargs = mock_snapshot_class.call_args
        assert snapshot_kwargs["last_profiled_index"] == 0

        # Check that the pipeline runs with the NEWLY CREATED snapshot
        mock_execute_run.assert_called_once_with(
            profile_base_args["user_model_function"],
            "./test_model/",
            new_snapshot_instance,
            profile_base_args["statistical_samples"],
        )
