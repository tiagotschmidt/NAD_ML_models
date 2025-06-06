from enum import Enum
from multiprocessing.connection import Connection
from typing import Callable, List, Tuple

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
    FinalStop = 2


class RangeMode(Enum):
    Additive = 0
    Multiplicative = 1


class RangeParameter:
    def __init__(self, item_list: List):  # type: ignore
        self._iterable_list = item_list

    @classmethod
    def from_range(
        cls,
        start: int,
        end: int,
        stride: int,
        type: FrameworkParameterType,
        mode: RangeMode,
    ):
        _iterable_list = []
        # if stride == 0:
        #     return RangeParameter(_iterable_list)
        if mode.value == RangeMode.Additive.value:
            _iterable_list = list(range(start, end + 1, stride))
        elif mode.value == RangeMode.Multiplicative.value:
            value = start
            while value <= end:
                _iterable_list.append(value)
                value *= stride

        return RangeParameter(_iterable_list)

    def __iter__(self):
        for item in self._iterable_list:
            yield item


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
        train_platform: Platform = None,  # type: ignore
        test_platform: Platform = None,  # type: ignore
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

    def __str__(self):
        return (
            f"ExecutionConfiguration("
            f"layers={self.number_of_layers}, "
            f"units={self.number_of_units}, "
            f"epochs={self.number_of_epochs}, "
            f"features={self.number_of_features}, "
            f"platform={self.platform}, "
            f"cycle={self.cycle}"
            f")"
        )


class EnvironmentConfiguration:
    def __init__(
        self,
        first_custom_layer_code: Callable[[keras.models.Model, int, int], None],
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
        results_pipe: Connection,
        log_signal_pipe: Connection,
        log_result_pipe: Connection,
    ):
        self.first_custom_layer_code = first_custom_layer_code
        self.repeated_custom_layer_code = repeated_custom_layer_code
        self.final_custom_layer_code = final_custom_layer_code
        self.number_of_samples = number_of_samples
        self.batch_size = batch_size
        self.performance_metrics_list = performance_metrics_list
        self.dataset = dataset
        self.dataset_target_label = dataset_target_label
        self.loss_metric_str = loss_metric_str
        self.optimizer = optimizer
        self.start_pipe = start_pipe
        self.results_pipe = results_pipe
        self.log_signal_pipe = log_signal_pipe
        self.log_result_pipe = log_result_pipe


SnapshotResultsList = List[Tuple[ExecutionConfiguration, List[int], List[dict]]]  # type: ignore


class PlotListCollection:
    def __init__(
        self,
        test_epoch_lists: SnapshotResultsList,
        test_units_lists: SnapshotResultsList,
        test_features_lists: SnapshotResultsList,
        test_layers_lists: SnapshotResultsList,
        train_epoch_lists: SnapshotResultsList,
        train_units_lists: SnapshotResultsList,
        train_features_lists: SnapshotResultsList,
        train_layers_lists: SnapshotResultsList,
    ):
        self.test_epoch_lists = test_epoch_lists
        self.test_units_lists = test_units_lists
        self.test_features_lists = test_features_lists
        self.test_layers_lists = test_layers_lists
        self.train_epoch_lists = train_epoch_lists
        self.train_units_lists = train_units_lists
        self.train_features_lists = train_features_lists
        self.train_layers_lists = train_layers_lists

    def __str__(self):
        return (
            f"PlotListCollection(\n"
            f"  test_epoch_lists={self.test_epoch_lists},\n"
            f"  test_units_lists={self.test_units_lists},\n"
            f"  test_features_lists={self.test_features_lists},\n"
            f"  test_layers_lists={self.test_layers_lists},\n"
            f"  train_epoch_lists={self.train_epoch_lists},\n"
            f"  train_units_lists={self.train_units_lists},\n"
            f"  train_features_lists={self.train_features_lists},\n"
            f"  train_layers_lists={self.train_layers_lists}\n"
            f")"
        )
