"""
This module contains the unit tests for the ModelExecutionConfig class.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from eppnad.utils.model_execution_config import ModelExecutionConfig


class TestModelExecutionConfig:
    """Groups all tests for the ModelExecutionConfig class."""

    def test_initialization_and_attribute_assignment(self, mocker: MagicMock):
        """
        Tests that the __init__ method correctly assigns all provided
        attributes to the instance.
        """
        # 1. Create mock objects for complex parameters
        mock_callable = mocker.MagicMock()
        mock_dataset = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        # Mocking multiprocessing.Connection
        mock_pipe_a = mocker.MagicMock()
        mock_pipe_b = mocker.MagicMock()
        mock_pipe_c = mocker.MagicMock()
        mock_pipe_d = mocker.MagicMock()

        # 2. Define test data
        test_config_data = {
            "first_custom_layer_code": mock_callable,
            "repeated_custom_layer_code": mock_callable,
            "final_custom_layer_code": mock_callable,
            "statistical_samples": 10,
            "batch_size": 32,
            "performance_metrics_list": ["accuracy", "precision"],
            "dataset": mock_dataset,
            "dataset_target_label": "target",
            "loss_metric_str": "binary_crossentropy",
            "optimizer": "adam",
        }

        # 3. Create an instance of the class
        config_instance = ModelExecutionConfig(**test_config_data)

        # 4. Assert that all attributes were stored correctly
        assert config_instance.first_custom_layer_code is mock_callable
        assert config_instance.repeated_custom_layer_code is mock_callable
        assert config_instance.final_custom_layer_code is mock_callable
        assert config_instance.number_of_samples == 10
        assert config_instance.batch_size == 32
        assert config_instance.performance_metrics_list == ["accuracy", "precision"]
        assert config_instance.dataset.equals(mock_dataset)
        assert config_instance.dataset_target_label == "target"
        assert config_instance.loss_metric_str == "binary_crossentropy"
        assert config_instance.optimizer == "adam"

    def test_initialization_with_different_values(self, mocker: MagicMock):
        """
        Tests initialization with a different set of valid parameters to ensure
        flexibility.
        """
        # Create different mock objects to ensure no test cross-contamination
        mock_callable_2 = mocker.MagicMock()
        mock_dataset_2 = pd.DataFrame({"data": [5, 6, 7], "label": [1, 0, 1]})
        mock_pipe_e = mocker.MagicMock()

        # Create an instance with different values
        config_instance = ModelExecutionConfig(
            first_custom_layer_code=mock_callable_2,
            repeated_custom_layer_code=mock_callable_2,
            final_custom_layer_code=mock_callable_2,
            statistical_samples=5,
            batch_size=64,
            performance_metrics_list=["mse"],
            dataset=mock_dataset_2,
            dataset_target_label="label",
            loss_metric_str="mae",
            optimizer="sgd",
        )

        # Assert a subset of the attributes to confirm they are set correctly
        assert config_instance.number_of_samples == 5
        assert config_instance.batch_size == 64
        assert config_instance.performance_metrics_list == ["mse"]
        assert config_instance.dataset_target_label == "label"
        assert config_instance.optimizer == "sgd"
