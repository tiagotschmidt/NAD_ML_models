import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from keras.models import Sequential
from keras.layers import Dense
import multiprocessing
from typing import Callable, List
import keras
import pandas as pd
import logging
import datetime

internal_logger = logging.getLogger(__name__)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
internal_log_filename = f"epp_nad_{timestamp}.log"
logging.basicConfig(
    filename=internal_log_filename,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

from execution_engine import ExecutionEngine

from framework_parameters import (
    EnvironmentConfiguration,
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
    user_model_name: str,
    repeated_custom_layer_code: Callable[[keras.models.Model, int, int], None],
    final_custom_layer_code: Callable[[keras.models.Model], None],
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
    loss_metric_str: str = "binary_crossentropy",
    optimizer: str = "adam",
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

    logging.info("Starting EPPNAD.")
    print("Starting EPPNAD")

    start_pipe_engine_side, start_engine_pipe_manager_side = multiprocessing.Pipe()
    start_pipe_logger_side, start_logger_pipe_manager_side = multiprocessing.Pipe()
    log_engine_queue = multiprocessing.Queue()
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
        log_engine_queue,
        results_pipe_engine_side,
    )

    engine = ExecutionEngine(
        configurations_list, user_model, user_model_name, environment, internal_logger
    )

    logger = Logger(start_pipe_logger_side, log_engine_queue, internal_logger)

    engine.start()
    logger.start()

    start_engine_pipe_manager_side.send(True)
    start_logger_pipe_manager_side.send(True)

    list_results = results_pipe_manager_side.recv()
    (config, result) = list_results[0]
    print(config)
    print(result)

    engine.join()
    logger.join()


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


model = Sequential()


def repeated_custom_layer(model, number_of_units, input_shape):
    model.add(Dense(units=number_of_units, input_dim=input_shape, activation="relu"))


def final_custom_layer(model):
    model.add(Dense(units=1, activation="sigmoid"))


numbers_of_layers = RangeParameter(100, 100, 100, FrameworkParameterType.NumberOfLayers)
numbers_of_neurons = RangeParameter(10, 10, 40, FrameworkParameterType.NumberOfNeurons)
numbers_of_epochs = RangeParameter(1, 1, 1, FrameworkParameterType.NumberOfEpochs)
numbers_of_features = RangeParameter(
    70, 70, 10, FrameworkParameterType.NumberOfFeatures
)

# Define profile mode
profile_mode = MLMode(LifecycleSelected.TrainAndTest, Platform.GPU, Platform.GPU)

# Define other parameters
number_of_samples = 2
batch_size = 40960
performance_metrics_list = ["accuracy", "f1_score", "precision", "recall"]
preprocessed_dataset = pd.read_csv("dataset/preprocessed_binary_dataset.csv")
dataset_target_label = "intrusion"
loss_metric_str = "binary_crossentropy"
optimizer = "adam"

profile(
    model,
    "Test_Model",
    repeated_custom_layer_code=repeated_custom_layer,
    final_custom_layer_code=final_custom_layer,
    numbers_of_layers=numbers_of_layers,
    numbers_of_neurons=numbers_of_neurons,
    numbers_of_epochs=numbers_of_epochs,
    numbers_of_features=numbers_of_features,
    profile_mode=profile_mode,
    number_of_samples=number_of_samples,
    batch_size=batch_size,
    performance_metrics_list=performance_metrics_list,
    preprocessed_dataset=preprocessed_dataset,
    dataset_target_label=dataset_target_label,
    loss_metric_str=loss_metric_str,
    optimizer=optimizer,
)
