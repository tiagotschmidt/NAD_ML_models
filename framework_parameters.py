from enum import Enum
from multiprocessing.connection import Connection
from typing import Callable, List

import keras
import pandas as pd


class FrameworkParameterType(Enum):
    NumberOfLayers = 0
    NumberOfNeurons = 1
    NumberOfEpochs = 2
    NumberOfFeatures = 4


class ProcessSignal(Enum):
    Start = 0
    Stop = 1


class RangeParameter:
    def __init__(self, start: int, end: int, stride: int, type: FrameworkParameterType):
        self.start = start
        self.end = end
        self.stride = stride
        self.type = type

    def __iter__(self):
        for i in range(self.start, self.end + 1, self.stride):
            yield i


class Platform(Enum):
    CPU = 0
    GPU = 1


class Lifecycle(Enum):
    Train = 0
    Test = 1


class LifecycleSelected(Enum):
    OnlyTrain = 0
    OnlyTest = 1
    TrainAndTest = 2


class MLMode:
    def __init__(
        self,
        cycle: LifecycleSelected,
        train_platform: Platform = None,
        test_platform: Platform = None,
    ):
        self.cycle = cycle
        self.test_platform = test_platform
        self.train_platform = train_platform


class ExecutionConfiguration:
    def __init__(
        self,
        number_of_layers: int,
        number_of_units: int,
        number_of_epochs: int,
        number_of_features: int,
        platform: Platform,
        cycle: Lifecycle,
    ):
        self.number_of_layers = number_of_layers
        self.number_of_units = number_of_units
        self.number_of_epochs = number_of_epochs
        self.number_of_features = number_of_features
        self.platform = platform
        self.cycle = cycle


class EnvironmentConfiguration:
    def __init__(
        self,
        repeated_custom_layer_code: Callable[[keras.models.Model, int, int], None],
        final_custom_layer_code: Callable[[keras.models.Model], None],
        number_of_samples: int,
        batch_size: int,
        performance_metrics_list: List[str],
        dataset: pd.DataFrame,
        dataset_target_label: str,
        loss_metric_str: str,
        optimizer: str,
        start_pipe: Connection,
        log_pipe: Connection,
        results_pipe: Connection,
    ):
        self.repeated_custom_layer_code = (repeated_custom_layer_code,)
        self.final_custom_layer_code = final_custom_layer_code
        self.number_of_samples = number_of_samples
        self.batch_size = batch_size
        self.performance_metrics_list = performance_metrics_list
        self.dataset = dataset
        self.dataset_target_label = dataset_target_label
        self.loss_metric_str = loss_metric_str
        self.optimizer = optimizer
        self.start_pipe = start_pipe
        self.log_pipe = log_pipe
        self.results_pipe = log_pipe
