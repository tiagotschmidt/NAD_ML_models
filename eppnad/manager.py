import multiprocessing
from typing import List
from typing_extensions import runtime
import keras
import pandas as pd
import logging
import datetime
from pathlib import Path

from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.framework_parameters import (
    FrameworkParameterType,
    Lifecycle,
    ProfileMode,
    PercentageRangeParameter,
    Platform,
    RangeMode,
    RangeParameter,
)
from eppnad.utils.model_execution_config import (
    ModelExecutionConfig,
    ModelFinalLayerLambda,
    ModelLayerLambda,
)
from eppnad.utils.plot_list_collection import PlotListCollection
from eppnad.utils.runtime_snapshot import RuntimeSnapshot
from .plotter import plot

logger = logging.getLogger(__name__)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
internal_log_filename = f"epp_nad_{timestamp}.log"
logging.basicConfig(
    filename=internal_log_filename,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

from .execution_engine import ExecutionEngine
from .core.energy_monitor import EnergyMonitor

DOT_DIR = "./"


def profile(
    user_model_function: keras.models.Model,
    user_model_name: str,
    repeated_custom_layer_code: ModelLayerLambda,
    first_custom_layer_code: ModelLayerLambda,
    final_custom_layer_code: ModelFinalLayerLambda,
    numbers_of_layers: RangeParameter = RangeParameter.from_range(
        100, 300, 100, FrameworkParameterType.Layers, RangeMode.Additive
    ),
    numbers_of_units: RangeParameter = RangeParameter.from_range(
        10, 90, 40, FrameworkParameterType.Neurons, RangeMode.Additive
    ),
    numbers_of_epochs: RangeParameter = RangeParameter.from_range(
        50, 150, 50, FrameworkParameterType.Epochs, RangeMode.Additive
    ),
    numbers_of_features: RangeParameter = RangeParameter.from_range(
        70, 10, 90, FrameworkParameterType.Features, RangeMode.Additive
    ),
    sampling_rates: PercentageRangeParameter = PercentageRangeParameter([1]),
    profile_mode: ProfileMode = ProfileMode(
        Lifecycle.TRAIN_AND_TEST, Platform.GPU, Platform.CPU
    ),
    statistical_samples: int = 30,
    batch_size: int = 32,
    performance_metrics_list: List[str] = [
        "accuracy",
        "f1_score",
        "recall",
    ],
    preprocessed_dataset: pd.DataFrame = pd.read_csv(
        "data/NSL-KDD/preprocessed_binary_dataset.csv"
    ),
    dataset_target_label: str = "intrusion",
    loss_metric_str: str = "binary_crossentropy",
    optimizer: str = "adam",
) -> PlotListCollection:
    logging.info("Starting EPPNAD profiling.")
    configurations_list = _generate_configurations_list(
        numbers_of_layers,
        numbers_of_units,
        numbers_of_epochs,
        numbers_of_features,
        sampling_rates,
        profile_mode,
    )

    start_pipe_engine_side, start_engine_pipe_manager_side = multiprocessing.Pipe()
    log_side_signal_pipe, engine_side_signal_pipe = multiprocessing.Pipe()
    engine_side_result_pipe, log_side_result_pipe = multiprocessing.Pipe()
    results_pipe_manager_side, results_pipe_engine_side = multiprocessing.Pipe()

    modelExecutionConfig = ModelExecutionConfig(
        first_custom_layer_code,
        repeated_custom_layer_code,
        final_custom_layer_code,
        statistical_samples,
        batch_size,
        performance_metrics_list,
        preprocessed_dataset,
        dataset_target_label,
        loss_metric_str,
        optimizer,
    )

    profile_execution_dir = DOT_DIR + user_model_name + "/"
    Path(profile_execution_dir).mkdir()

    runtime_snapshot = RuntimeSnapshot(
        user_model_name,
        profile_execution_dir,
        configurations_list,
        modelExecutionConfig,
        0,
    )

    engine = ExecutionEngine(
        user_model_function,
        runtime_snapshot,
        logger,  # type: ignore
        start_pipe_engine_side,
        results_pipe_engine_side,
        engine_side_signal_pipe,
        engine_side_result_pipe,
    )

    energy_monitor = EnergyMonitor(
        log_side_signal_pipe,
        log_side_result_pipe,
        logger,  # type: ignore
    )

    engine.start()
    energy_monitor.start()

    start_engine_pipe_manager_side.send(True)

    list_results = results_pipe_manager_side.recv()

    final_results = plot(list_results, performance_metrics_list, statistical_samples)

    engine.join()
    energy_monitor.join()

    return final_results


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
        A list of ExecutionConfiguration objects, one for each experiment to be run.
    """
    return_list = []
    if Lifecycle.TRAIN in profile_mode.cycle:
        lifecycle = Lifecycle.TRAIN
        platform = profile_mode.train_platform
        flatten_configurations(
            layers,
            units,
            epochs,
            features,
            sampling_rates,
            return_list,
            lifecycle,
            platform,
        )

    if Lifecycle.TEST in profile_mode.cycle:
        lifecycle = Lifecycle.TEST
        platform = profile_mode.test_platform
        flatten_configurations(
            layers,
            units,
            epochs,
            features,
            sampling_rates,
            return_list,
            lifecycle,
            platform,
        )

    return return_list


def flatten_configurations(
    layers, units, epochs, features, sampling_rates, return_list, lifecycle, platform
):
    """
    A nested loop helper to flatten the parameter space into a list.

    This function iterates through all combinations of the provided parameter
    ranges and appends a new ExecutionConfiguration object to the return_list
    for each one.

    Args:
        layers: The range of layer counts.
        units: The range of unit counts.
        epochs: The range of epoch counts.
        features: The range of feature counts.
        sampling_rates: The range of sampling rates.
        return_list: The list to which new configurations will be appended.
        lifecycle: The lifecycle phase (Train or Test) for this batch.
        platform: The platform (CPU or GPU) for this batch.
    """
    for number_of_layers in layers:
        for number_of_neurons in units:
            for number_of_epochs in epochs:
                for number_of_features in features:
                    for sampling_rate in sampling_rates:
                        return_list.append(
                            ExecutionConfiguration(
                                number_of_layers,
                                number_of_neurons,
                                number_of_epochs,
                                number_of_features,
                                sampling_rate,
                                platform,
                                lifecycle,
                            )
                        )
