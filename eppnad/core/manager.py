# eppnad/manager.py
"""
Core profiling manager for EPPNAD.

This module provides the main entry points for running energy and performance
profiling sessions. It orchestrates the setup of configurations, the initialization
of the execution and monitoring engines, and the final plotting of results.

Functions:
    profile: Starts a new, complete profiling session.
    intermittent_profile: Resumes a profiling session from the last saved state.
"""

import multiprocessing
import logging
import datetime
from pathlib import Path
from typing import Dict, List

import keras
import pandas as pd

from eppnad.core.energy_monitor import EnergyMonitor
from eppnad.core.execution_engine import ExecutionEngine
from eppnad.core.plotter import plot
from eppnad.utils.execution_configuration import (
    ExecutionConfiguration,
    Lifecycle,
    Platform,
)
from eppnad.utils.framework_parameters import (
    PercentageRangeParameter,
    ProfileMode,
    RangeParameter,
)
from eppnad.utils.model_execution_config import (
    ModelExecutionConfig,
    ModelFinalLayerLambda,
    ModelLayerLambda,
)
from eppnad.utils.plot_list_collection import PlotCollection
from eppnad.utils.runtime_snapshot import RuntimeSnapshot

logger = logging.getLogger("eppnad")
logger.setLevel(logging.DEBUG)

logger.propagate = False


def _generate_configurations_list(
    layers: RangeParameter,
    units: RangeParameter,
    epochs: RangeParameter,
    features: RangeParameter,
    sampling_rates: PercentageRangeParameter,
    profile_mode: ProfileMode,
) -> List[ExecutionConfiguration]:
    """
    Generates a list of all possible hyperparameter configurations.

    This private helper function creates the Cartesian product of all provided
    hyperparameter ranges, creating an ExecutionConfiguration object for each
    unique combination based on the specified profile mode.

    Args:
        layers: The range of layer counts to test.
        units: The range of unit counts to test.
        epochs: The range of epoch counts to test.
        features: The range of feature counts to test.
        sampling_rates: The range of sampling rates to test.
        profile_mode: The ProfileMode specifying which lifecycles and platforms to run.

    Returns:
        A list of ExecutionConfiguration objects for the profiling session.
    """
    config_list = []
    if Lifecycle.TRAIN in profile_mode.cycle:
        _flatten_configurations(
            layers,
            units,
            epochs,
            features,
            sampling_rates,
            config_list,
            Lifecycle.TRAIN,
            profile_mode.train_platform,  # type: ignore
        )

    if Lifecycle.TEST in profile_mode.cycle:
        _flatten_configurations(
            layers,
            units,
            epochs,
            features,
            sampling_rates,
            config_list,
            Lifecycle.TEST,
            profile_mode.test_platform,  # type: ignore
        )

    return config_list


def _flatten_configurations(
    layers: RangeParameter,
    units: RangeParameter,
    epochs: RangeParameter,
    features: RangeParameter,
    sampling_rates: PercentageRangeParameter,
    config_list: list,
    lifecycle: Lifecycle,
    platform: Platform,
):
    """
    A nested loop helper to flatten the parameter space into a list.

    This function iterates through all combinations of the provided parameter
    ranges and appends a new ExecutionConfiguration object to the config_list
    for each one.

    Args:
        layers: The range of layer counts.
        units: The range of unit counts.
        epochs: The range of epoch counts.
        features: The range of feature counts.
        sampling_rates: The range of sampling rates.
        config_list: The list to which new configurations will be appended.
        lifecycle: The lifecycle phase (Train or Test) for this batch.
        platform: The platform (CPU or GPU) for this batch.
    """
    for layer_count in layers:
        for unit_count in units:
            for epoch_count in epochs:
                for feature_count in features:
                    for rate in sampling_rates:
                        config_list.append(
                            ExecutionConfiguration(
                                layer_count,
                                unit_count,
                                epoch_count,
                                feature_count,
                                rate,
                                platform,
                                lifecycle,
                            )
                        )


def _execute_profiling_run(
    user_model_function: keras.models.Model,
    profile_execution_dir: str,
    runtime_snapshot: RuntimeSnapshot,
    statistical_samples: int,
) -> Dict[str, Dict[str, PlotCollection]]:
    """
    Initializes and runs the core execution and monitoring processes.

    Args:
        user_model_function: The user-defined Keras model function.
        profile_execution_dir: Directory to save profiling artifacts.
        runtime_snapshot: The snapshot containing the execution state.
        statistical_samples: The number of times each experiment is repeated.

    Returns:
        A dictionary containing the plotted results of the profiling.
    """
    # --- Setup multiprocessing pipes for communication ---
    start_pipe_engine_side, start_engine_pipe_manager_side = multiprocessing.Pipe()
    log_side_signal_pipe, engine_side_signal_pipe = multiprocessing.Pipe()
    engine_side_result_pipe, log_side_result_pipe = multiprocessing.Pipe()

    # --- Initialize core components ---
    engine = ExecutionEngine(
        user_model_function,
        profile_execution_dir,
        runtime_snapshot,
        logger,
        start_pipe_engine_side,
        engine_side_signal_pipe,
        engine_side_result_pipe,
    )

    energy_monitor = EnergyMonitor(
        log_side_signal_pipe,
        log_side_result_pipe,
        logger,
    )

    # --- Start processes ---
    engine.start()
    energy_monitor.start()

    # Signal the engine to begin processing configurations
    start_engine_pipe_manager_side.send(True)

    # --- Wait for processes to complete ---
    engine.join()
    energy_monitor.join()

    logger.info(f"Joined ExecutionEngine and EnergyMonitor.")

    # --- Generate and return final plots ---
    final_results = plot(profile_execution_dir, statistical_samples, logger)
    return final_results


def profile(
    user_model_function,
    user_model_name: str,
    repeated_custom_layer_code: ModelLayerLambda,
    first_custom_layer_code: ModelLayerLambda,
    final_custom_layer_code: ModelFinalLayerLambda,
    layers: RangeParameter = RangeParameter([100, 200, 300]),
    units: RangeParameter = RangeParameter([10, 50, 100]),
    epochs: RangeParameter = RangeParameter([50, 100, 150]),
    features: RangeParameter = RangeParameter([13, 53, 93]),
    sampling_rates: PercentageRangeParameter = PercentageRangeParameter([0.1, 0.5, 1]),
    profile_mode: ProfileMode = ProfileMode(
        Lifecycle.TRAIN_AND_TEST, Platform.GPU, Platform.CPU
    ),
    statistical_samples: int = 10,
    batch_size: int = 2048,
    performance_metrics: List[str] = ["accuracy", "f1_score", "recall"],
    dataset: pd.DataFrame = pd.read_csv("data/NSL-KDD/preprocessed_binary_dataset.csv"),
    target_label: str = "intrusion",
    loss_function: str = "binary_crossentropy",
    optimizer: str = "adam",
) -> Dict[str, Dict[str, PlotCollection]]:
    """
    Starts a new EPPNAD profiling session from scratch.

    This function orchestrates a full profiling run, generating all
    hyperparameter configurations and executing them. It does not load any
    previous state.

    Args:
        user_model_function: A function that returns a compiled Keras model.
        user_model_name: The name for the model, used for directory creation.
        repeated_custom_layer_code: Lambda for repeated model layers.
        first_custom_layer_code: Lambda for the first model layer.
        final_custom_layer_code: Lambda for the final model layer.
        layers: Range of layer counts to profile.
        units: Range of neuron counts per layer to profile.
        epochs: Range of training epochs to profile.
        features: Range of input features to profile.
        sampling_rates: Range of dataset sampling rates to profile.
        profile_mode: Defines the lifecycle and platform for the session.
        statistical_samples: Number of times to repeat each experiment.
        batch_size: The batch size for training and testing.
        performance_metrics: List of Keras metrics to evaluate.
        dataset: The preprocessed pandas DataFrame for the session.
        target_label: The name of the target column in the dataset.
        loss_function: The Keras loss function to use.
        optimizer: The Keras optimizer to use.

    Returns:
        A dictionary containing the plotted results of the profiling.
    """
    logger.handlers.clear()

    profile_execution_dir = f"./{user_model_name}/"
    Path(profile_execution_dir).mkdir(parents=True, exist_ok=True)

    log_path = Path(profile_execution_dir) / "eppnad.log"
    file_handler = logging.FileHandler(log_path)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info(
        f"Starting new EPPNAD profiling session for '{user_model_name}'. Log file at: {log_path}"
    )

    configurations_list = _generate_configurations_list(
        layers, units, epochs, features, sampling_rates, profile_mode
    )

    model_exec_config = ModelExecutionConfig(
        first_custom_layer_code,
        repeated_custom_layer_code,
        final_custom_layer_code,
        statistical_samples,
        batch_size,
        performance_metrics,
        dataset,
        target_label,
        loss_function,
        optimizer,
    )

    runtime_snapshot = RuntimeSnapshot(
        user_model_name,
        profile_execution_dir,
        configurations_list,
        model_exec_config,
        last_profiled_index=-1,
    )

    return _execute_profiling_run(
        user_model_function,
        profile_execution_dir,
        runtime_snapshot,
        statistical_samples,
    )


def intermittent_profile(
    user_model_function,
    user_model_name: str,
    repeated_custom_layer_code: ModelLayerLambda,
    first_custom_layer_code: ModelLayerLambda,
    final_custom_layer_code: ModelFinalLayerLambda,
    layers: RangeParameter = RangeParameter([100, 200, 300]),
    units: RangeParameter = RangeParameter([10, 50, 100]),
    epochs: RangeParameter = RangeParameter([50, 100, 150]),
    features: RangeParameter = RangeParameter([13, 53, 93]),
    sampling_rates: PercentageRangeParameter = PercentageRangeParameter([0.1, 0.5, 1]),
    profile_mode: ProfileMode = ProfileMode(
        Lifecycle.TRAIN_AND_TEST, Platform.GPU, Platform.CPU
    ),
    statistical_samples: int = 10,
    batch_size: int = 2048,
    performance_metrics: List[str] = ["accuracy", "f1_score", "recall"],
    dataset: pd.DataFrame = pd.read_csv("data/NSL-KDD/preprocessed_binary_dataset.csv"),
    target_label: str = "intrusion",
    loss_function: str = "binary_crossentropy",
    optimizer: str = "adam",
) -> Dict[str, Dict[str, PlotCollection]]:
    """
    Resumes a previously started EPPNAD profiling session.

    This function attempts to load the latest `RuntimeSnapshot` to continue
    an interrupted session. If no snapshot is found, it initializes a new one.

    Args:
        user_model_function: A function that returns a compiled Keras model.
        user_model_name: The name for the model, used for directory creation.
        repeated_custom_layer_code: Lambda for repeated model layers.
        first_custom_layer_code: Lambda for the first model layer.
        final_custom_layer_code: Lambda for the final model layer.
        layers: Range of layer counts to profile.
        units: Range of neuron counts per layer to profile.
        epochs: Range of training epochs to profile.
        features: Range of input features to profile.
        sampling_rates: Range of dataset sampling rates to profile.
        profile_mode: Defines the lifecycle and platform for the session.
        statistical_samples: Number of times to repeat each experiment.
        batch_size: The batch size for training and testing.
        performance_metrics: List of Keras metrics to evaluate.
        dataset: The preprocessed pandas DataFrame for the session.
        target_label: The name of the target column in the dataset.
        loss_function: The Keras loss function to use.
        optimizer: The Keras optimizer to use.

    Returns:
        A dictionary containing the plotted results of the profiling.
    """
    logger.handlers.clear()

    profile_execution_dir = f"./{user_model_name}/"
    Path(profile_execution_dir).mkdir(parents=True, exist_ok=True)

    log_path = Path(profile_execution_dir) / "eppnad.log"
    file_handler = logging.FileHandler(log_path)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info(
        f"Starting new EPPNAD profiling session for '{user_model_name}'. Log file at: {log_path}"
    )

    runtime_snapshot = RuntimeSnapshot.load_latest(profile_execution_dir, logger)

    if runtime_snapshot is None:
        logging.warning("No previous snapshot found. Starting a new session.")
        configurations_list = _generate_configurations_list(
            layers, units, epochs, features, sampling_rates, profile_mode
        )
        model_exec_config = ModelExecutionConfig(
            first_custom_layer_code,
            repeated_custom_layer_code,
            final_custom_layer_code,
            statistical_samples,
            batch_size,
            performance_metrics,
            dataset,
            target_label,
            loss_function,
            optimizer,
        )
        runtime_snapshot = RuntimeSnapshot(
            user_model_name,
            profile_execution_dir,
            configurations_list,
            model_exec_config,
            last_profiled_index=-1,
        )

    return _execute_profiling_run(
        user_model_function,
        profile_execution_dir,
        runtime_snapshot,
        statistical_samples,
    )
