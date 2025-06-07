import multiprocessing
from typing import List
import keras
import pandas as pd
import logging
import datetime

from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.framework_parameters import (
    FrameworkParameterType,
    LifeCycle,
    LifecycleSelected,
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
from .plotter import plot

internal_logger = logging.getLogger(__name__)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
internal_log_filename = f"epp_nad_{timestamp}.log"
logging.basicConfig(
    filename=internal_log_filename,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

from .execution_engine import ExecutionEngine
from .logger import Logger


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
        LifecycleSelected.TrainAndTest, Platform.GPU, Platform.CPU
    ),
    statistical_samples: int = 30,
    batch_size: int = 32,
    performance_metrics_list: List[str] = [
        "accuracy",
        "f1_score",
        "recall",
    ],
    preprocessed_dataset: pd.DataFrame = pd.read_csv(
        "dataset/preprocessed_binary_dataset.csv"
    ),
    dataset_target_label: str = "intrusion",
    loss_metric_str: str = "binary_crossentropy",
    optimizer: str = "adam",
) -> PlotListCollection:
    logging.info("Starting EPPNAD profiling.")
    configurations_list = __generate_configurations_list(
        numbers_of_layers,
        numbers_of_units,
        numbers_of_epochs,
        numbers_of_features,
        sampling_rates,
        profile_mode,
    )

    # sample_size = int(len(preprocessed_dataset) * sampling_rate)
    # preprocessed_dataset = preprocessed_dataset.sample(sample_size)

    start_pipe_engine_side, start_engine_pipe_manager_side = multiprocessing.Pipe()
    start_pipe_logger_side, start_logger_pipe_manager_side = multiprocessing.Pipe()
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
        start_pipe_engine_side,
        results_pipe_engine_side,
        engine_side_signal_pipe,
        engine_side_result_pipe,
    )

    engine = ExecutionEngine(
        configurations_list,
        user_model_function,
        user_model_name,
        modelExecutionConfig,
        internal_logger,
    )

    logger = Logger(
        start_pipe_logger_side,
        internal_logger,
        log_side_signal_pipe,
        log_side_result_pipe,
    )

    engine.start()
    logger.start()

    start_engine_pipe_manager_side.send(True)
    start_logger_pipe_manager_side.send(True)

    list_results = results_pipe_manager_side.recv()

    final_results = plot(list_results, performance_metrics_list, statistical_samples)

    engine.join()
    logger.join()

    return final_results


def __generate_configurations_list(
    layers: RangeParameter,
    units: RangeParameter,
    epochs: RangeParameter,
    features: RangeParameter,
    sampling_rates: PercentageRangeParameter,
    profile_mode: ProfileMode,
) -> List[ExecutionConfiguration]:
    return_list = []
    if LifecycleSelected.ONLY_TRAIN in profile_mode.cycle:
        lifecycle = LifeCycle.Train
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

    if LifecycleSelected.ONLY_TEST in profile_mode.cycle:
        lifecycle = LifeCycle.Test
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
