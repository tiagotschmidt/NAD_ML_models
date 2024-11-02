import pandas as pd
import pytest
from unittest.mock import Mock, patch
import numpy as np
import tensorflow as tf
from multiprocessing import Pipe
from framework_parameters import (
    ExecutionConfiguration,
    EnvironmentConfiguration,
    Platform,
    Lifecycle,
    ProcessSignal,
)
from keras.models import Sequential

from ..execution_engine import ExecutionEngine


@pytest.fixture
def mock_environment():
    env = Mock(spec=EnvironmentConfiguration)
    env.loss_metric_str = "mse"
    env.optimizer = "adam"
    env.performance_metrics_list = ["accuracy"]
    env.batch_size = 32
    env.number_of_samples = 10
    env.repeated_custom_layer_code = Mock()
    env.final_custom_layer_code = Mock()
    env.results_pipe, _ = Pipe()
    return env


@pytest.fixture
def mock_execution_config():
    config = Mock(spec=ExecutionConfiguration)
    config.platform = Platform.CPU
    config.number_of_layers = 3
    config.number_of_units = 64
    config.number_of_epochs = 5
    config.number_of_features = 10
    config.cycle = Lifecycle.Train
    return config


@pytest.fixture
def mock_internal_logger():
    return Mock()


@pytest.fixture
def mock_model():
    model = Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(10,)))
    model.add(tf.keras.layers.Dense(1))
    return model


@pytest.fixture
def execution_engine(
    mock_environment, mock_execution_config, mock_model, mock_internal_logger
):
    config_list = [mock_execution_config]
    return ExecutionEngine(
        config_list, mock_model, "test_model", mock_environment, mock_internal_logger
    )


def test_initialization(
    execution_engine, mock_model, mock_execution_config, mock_environment
):
    assert execution_engine.underlying_model == mock_model
    assert execution_engine.model_name == "test_model"
    assert execution_engine.configuration_list == [mock_execution_config]
    assert execution_engine.environment == mock_environment


def test_process_results(execution_engine):
    current_results = [
        {"accuracy": 0.9, "loss": 0.1},
        {"accuracy": 0.85, "loss": 0.15},
        {"accuracy": 0.8, "loss": 0.2},
    ]
    processed = execution_engine._ExecutionEngine__process_results(current_results)
    assert processed["accuracy"]["mean"] == pytest.approx(0.85, 0.01)
    assert processed["loss"]["mean"] == pytest.approx(0.15, 0.01)
    assert processed["accuracy"]["total_samples"] == 3


def test_is_train_config(execution_engine, mock_execution_config):
    mock_execution_config.cycle = Lifecycle.Train
    assert (
        execution_engine._ExecutionEngine__is_train_config(mock_execution_config)
        == True
    )
    mock_execution_config.cycle = Lifecycle.Test
    assert (
        execution_engine._ExecutionEngine__is_train_config(mock_execution_config)
        == False
    )


def test_try_load_model_from_storage(execution_engine, mock_execution_config):
    result = execution_engine._ExecutionEngine__try_load_model_from_storage(
        mock_execution_config
    )
    assert result == False  ## no save


def test_assemble_model_for_config(execution_engine, mock_execution_config):
    x_train_shape = 10
    execution_engine._ExecutionEngine__assemble_model_for_config(
        mock_execution_config, x_train_shape
    )
    assert (
        execution_engine.environment.repeated_custom_layer_code.call_count
        == mock_execution_config.number_of_layers
    )
    execution_engine.environment.final_custom_layer_code.assert_called_once_with(
        execution_engine.underlying_model
    )


def test_get_x_and_y(execution_engine):
    execution_engine.environment = Mock()
    execution_engine.environment.dataset = pd.DataFrame(np.random.rand(120, 11))
    execution_engine.environment.dataset_target_label = 10

    X_train, y_train, X_test, y_test = execution_engine._ExecutionEngine__get_x_and_y(
        10
    )
    assert X_train.shape == (96, 10)
    assert len(y_train) == 96
    assert X_test.shape == (24, 10)
    assert len(y_test) == 24


def test_get_full_dataset(execution_engine):
    execution_engine.environment = Mock()
    execution_engine.environment.dataset = pd.DataFrame(np.random.rand(120, 11))
    execution_engine.environment.dataset_target_label = 10

    _, _, X_test, Y_test = execution_engine._ExecutionEngine__get_full_dataset(10)
    assert X_test.shape == (120, 10)
    assert len(Y_test) == 120
