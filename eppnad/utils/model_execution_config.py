"""
This module defines the configuration for a model execution.
"""

from multiprocessing.connection import Connection
from typing import Callable, List
import keras
import pandas as pd


# Type hint for the lambda functions that define model layers.
# The callable should accept the Keras model, the number of neurons, and the input dimension.
ModelLayerLambda = Callable[[keras.models.Model, int, int], None]
# The callable should accept the Keras model.
ModelFinalLayerLambda = Callable[[keras.models.Model], None]


class ModelExecutionConfig:
    """
    A data class that encapsulates all the necessary parameters, assets, and
    communication pipes for a single, self-contained model execution profiler.

    This class centralizes the configuration, making it easier to pass the
    entire execution context between different components of the framework.
    """

    def __init__(
        self,
        first_custom_layer_code: ModelLayerLambda,
        repeated_custom_layer_code: ModelLayerLambda,
        final_custom_layer_code: ModelFinalLayerLambda,
        statistical_samples: int,
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
        """
        Initializes the ModelExecutionConfig with all execution-related settings.

        Args:
            first_custom_layer_code: A callable that adds the first layer(s) to the Keras model.
            repeated_custom_layer_code: A callable that adds the repeating intermediate layers
                to the Keras model.
            final_custom_layer_code: A callable that adds the final layer(s) to the Keras model.
            statistical_samples: The number of times to repeat the experiment for statistical significance.
            batch_size: The batch size to use for training the model.
            performance_metrics_list: A list of Keras performance metric strings (e.g., ['accuracy']).
            dataset: The pandas DataFrame containing the entire dataset.
            dataset_target_label: The column name of the target variable in the dataset.
            loss_metric_str: The Keras loss function string (e.g., 'binary_crossentropy').
            optimizer: The Keras optimizer string (e.g., 'adam').
            start_pipe: The multiprocessing connection pipe for starting signals.
            results_pipe: The multiprocessing connection pipe for sending back results.
            log_signal_pipe: The multiprocessing connection pipe for logging signals.
            log_result_pipe: The multiprocessing connection pipe for sending back log data.
        """
        self.first_custom_layer_code = first_custom_layer_code
        self.repeated_custom_layer_code = repeated_custom_layer_code
        self.final_custom_layer_code = final_custom_layer_code
        self.number_of_samples = statistical_samples
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
