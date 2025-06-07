from multiprocessing.connection import Connection
from typing import Callable

from eppnad.utils.framework_parameters import Lifecycle, Platform
import keras

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
