import multiprocessing
from typing import Callable, List
import keras
import pandas as pd
import logging
import datetime
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

from .framework_parameters import (
    EnvironmentConfiguration,
    ExecutionConfiguration,
    FrameworkParameterType,
    Lifecycle,
    LifecycleSelected,
    MLMode,
    Platform,
    PlotListCollection,
    RangeMode,
    RangeParameter,
)
from .logger import Logger


def profile(
    user_model_func,
    user_model_name: str,
    repeated_custom_layer_code: Callable[[keras.models.Model, int, int], None],
    final_custom_layer_code: Callable[[keras.models.Model], None],
    numbers_of_layers: RangeParameter = RangeParameter(
        100, 300, 100, FrameworkParameterType.NumberOfLayers, RangeMode.Additive
    ),
    numbers_of_neurons: RangeParameter = RangeParameter(
        10, 90, 40, FrameworkParameterType.NumberOfNeurons, RangeMode.Additive
    ),
    numbers_of_epochs: RangeParameter = RangeParameter(
        50, 150, 50, FrameworkParameterType.NumberOfEpochs, RangeMode.Additive
    ),
    numbers_of_features: RangeParameter = RangeParameter(
        70, 10, 90, FrameworkParameterType.NumberOfFeatures, RangeMode.Additive
    ),
    profile_mode: MLMode = MLMode(
        LifecycleSelected.TrainAndTest, Platform.GPU, Platform.GPU
    ),
    number_of_samples: int = 30,
    batch_size: int = 32,
    performance_metrics_list: List[str] = [
        "accuracy",
        "f1_score",
        "recall",
    ],
    sampling_rate: float = 1,
    preprocessed_dataset: pd.DataFrame = pd.read_csv(
        "dataset/preprocessed_binary_dataset.csv"
    ),
    dataset_target_label: str = "intrusion",
    loss_metric_str: str = "binary_crossentropy",
    optimizer: str = "adam",
) -> PlotListCollection:
    configurations_list = __generate_configurations_list(
        numbers_of_layers,
        numbers_of_neurons,
        numbers_of_epochs,
        numbers_of_features,
        profile_mode,
    )

    logging.info("Starting EPPNAD.")

    if not 0 <= sampling_rate <= 1:
        sampling_rate = 1
        logging.info("Sampling rate must be between 0 and 1.")

    sample_size = int(len(preprocessed_dataset) * sampling_rate)
    preprocessed_dataset = preprocessed_dataset.sample(sample_size)

    start_pipe_engine_side, start_engine_pipe_manager_side = multiprocessing.Pipe()
    start_pipe_logger_side, start_logger_pipe_manager_side = multiprocessing.Pipe()
    log_side_signal_pipe, engine_side_signal_pipe = multiprocessing.Pipe()
    engine_side_result_pipe, log_side_result_pipe = multiprocessing.Pipe()
    results_pipe_manager_side, results_pipe_engine_side = multiprocessing.Pipe()

    environment = EnvironmentConfiguration(
        repeated_custom_layer_code,
        final_custom_layer_code,
        number_of_samples,
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
        user_model_func,
        user_model_name,
        environment,
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

    final_results = plot(list_results, performance_metrics_list)

    engine.join()
    logger.join()

    return final_results


def __generate_configurations_list(
    numbers_of_layers: RangeParameter,
    numbers_of_neurons: RangeParameter,
    numbers_of_epochs: RangeParameter,
    numbers_of_features: RangeParameter,
    profile_mode: MLMode,
) -> List[ExecutionConfiguration]:
    return_list = []
    if profile_mode.cycle.value == LifecycleSelected.OnlyTest.value:
        for number_of_layers in numbers_of_layers:
            for number_of_neurons in numbers_of_neurons:
                for number_of_epochs in numbers_of_epochs:
                    for number_of_features in numbers_of_features:
                        return_list.append(
                            ExecutionConfiguration(
                                number_of_layers,
                                number_of_neurons,
                                number_of_epochs,
                                number_of_features,
                                profile_mode.test_platform,
                                Lifecycle.Test,
                            )
                        )
    elif profile_mode.cycle.value == LifecycleSelected.OnlyTrain.value:
        for number_of_layers in numbers_of_layers:
            for number_of_neurons in numbers_of_neurons:
                for number_of_epochs in numbers_of_epochs:
                    for number_of_features in numbers_of_features:
                        return_list.append(
                            ExecutionConfiguration(
                                number_of_layers,
                                number_of_neurons,
                                number_of_epochs,
                                number_of_features,
                                profile_mode.train_platform,
                                Lifecycle.Train,
                            )
                        )
    elif profile_mode.cycle.value == LifecycleSelected.TrainAndTest.value:
        for number_of_layers in numbers_of_layers:
            for number_of_neurons in numbers_of_neurons:
                for number_of_epochs in numbers_of_epochs:
                    for number_of_features in numbers_of_features:
                        return_list.append(
                            ExecutionConfiguration(
                                number_of_layers,
                                number_of_neurons,
                                number_of_epochs,
                                number_of_features,
                                profile_mode.train_platform,
                                Lifecycle.Train,
                            )
                        )
        for number_of_layers in numbers_of_layers:
            for number_of_neurons in numbers_of_neurons:
                for number_of_epochs in numbers_of_epochs:
                    for number_of_features in numbers_of_features:
                        return_list.append(
                            ExecutionConfiguration(
                                number_of_layers,
                                number_of_neurons,
                                number_of_epochs,
                                number_of_features,
                                profile_mode.test_platform,
                                Lifecycle.Test,
                            )
                        )

    return return_list
