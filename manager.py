from keras.models import Sequential
import multiprocessing
from typing import List
import keras
import pandas as pd

from execution_engine import ExecutionEngine

from framework_parameters import (
    ExecutionConfiguration,
    FrameworkParameterType,
    Lifecycle,
    LifecycleSelected,
    MLMode,
    Platform,
    RangeParameter,
)
from logger import Logger


def profile(
    user_model: keras.models.Model,
    numbers_of_layers: RangeParameter = RangeParameter(
        100, 300, 100, FrameworkParameterType.NumberOfLayers
    ),
    numbers_of_neurons: RangeParameter = RangeParameter(
        10, 90, 40, FrameworkParameterType.NumberOfNeurons
    ),
    numbers_of_epochs: RangeParameter = RangeParameter(
        50, 150, 50, FrameworkParameterType.NumberOfEpochs
    ),
    numbers_of_features: RangeParameter = RangeParameter(
        70, 10, 90, FrameworkParameterType.NumberOfFeatures
    ),
    profile_mode: MLMode = MLMode(
        LifecycleSelected.TrainAndTest, Platform.GPU, Platform.GPU
    ),
    number_of_samples: int = 30,
    batch_size: int = 32,
    performance_metrics_list: List[str] = [
        "accuracy",
        "f1_score",
        "precision",
        "recall",
    ],
    preprocessed_dataset: pd.DataFrame = pd.read_csv(
        "dataset/preprocessed_binary_dataset.csv"
    ),
    dataset_target_label: str = "intrusion",
):
    if not isinstance(user_model, keras.models.Model):
        raise TypeError("Input model must be a Keras model.")

    configurations_list = __generate_configurations_list(
        numbers_of_layers,
        numbers_of_neurons,
        numbers_of_epochs,
        numbers_of_features,
        profile_mode,
    )

    start_pipe_engine_side, start_engine_pipe_manager_side = multiprocessing.Pipe()
    start_pipe_logger_side, start_logger_pipe_manager_side = multiprocessing.Pipe()
    log_pipe_logger_side, log_pipe_engine_side = multiprocessing.Pipe()

    engine = ExecutionEngine(
        user_model,
        configurations_list,
        number_of_samples,
        batch_size,
        performance_metrics_list,
        preprocessed_dataset,
        dataset_target_label,
        start_pipe_engine_side,
        log_pipe_engine_side,
    )

    logger = Logger(start_pipe_logger_side, log_pipe_logger_side)

    engine.start()
    logger.start()

    start_engine_pipe_manager_side.send(True)
    start_logger_pipe_manager_side.send(True)

    engine.join()
    engine.join()


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
                                profile_mode.test_platform,
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
                                profile_mode.train_platform,
                                Lifecycle.Test,
                            )
                        )

    return return_list


# print("eae")
# cnn = Sequential()
# profile(cnn)
