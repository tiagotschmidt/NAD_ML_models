"""
This module defines the ExecutionEngine, a dedicated process responsible for
running model training and testing based on a list of configurations.

It handles platform-specific execution (CPU/GPU), data preparation, model
assembly, performance and energy measurement, and result aggregation.
"""

import gc
import multiprocessing
import statistics
from logging import Logger
from multiprocessing.connection import Connection
from os import path
from time import time
from tracemalloc import start
from turtle import st
from typing import Dict, List, Tuple

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import model_from_json

from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.execution_result import ExecutionResult
from eppnad.utils.framework_parameters import Lifecycle, Platform, ProcessSignal
from eppnad.utils.runtime_snapshot import RuntimeSnapshot


class ExecutionEngine(multiprocessing.Process):
    """
    A separate process that executes a series of model profiling tasks.

    This class iterates through a list of `ExecutionConfiguration` objects,
    manages the ML model's lifecycle (training, testing, saving, loading),
    and collects performance and energy consumption metrics. It communicates
    with the main process via pipes.
    """

    def __init__(
        self,
        user_model_function: keras.models.Model,
        runtime_snapshot: RuntimeSnapshot,
        logger: Logger,
        start_pipe: Connection,
        results_pipe: Connection,
        log_signal_pipe: Connection,
        log_result_pipe: Connection,
    ):
        """
        Initializes the ExecutionEngine process.

        Args:
            runtime_snapshot: A snapshot of the runtime state, including configs.
            logger: A shared logger instance for logging messages.
            start_pipe: Pipe for receiving the initial start signal.
            results_pipe: Pipe for sending the final aggregated results back.
            log_signal_pipe: Pipe for sending start/stop signals to the energy logger.
            log_result_pipe: Pipe for receiving energy measurements from the logger.
        """
        super().__init__()
        self.user_model_function = user_model_function
        self.runtime_snapshot = runtime_snapshot
        self.logger = logger
        self.start_pipe = start_pipe
        self.results_pipe = results_pipe
        self.log_signal_pipe = log_signal_pipe
        self.log_result_pipe = log_result_pipe
        self.results_list: List[ExecutionResult] = []

    def run(self):
        """
        The main entry point for the process.

        Waits for a start signal, then iterates through all configurations,
        executing them on the appropriate hardware platform. Finally, it sends
        the collected results back to the main process.
        """
        self.start_pipe.recv()  # Wait for the start signal from the manager
        self.logger.info(
            "[EXECUTION-ENGINE] Start signal received. Beginning profiling."
        )

        start_index = self.runtime_snapshot.last_profiled_index + 1

        for index, config in enumerate(
            self.runtime_snapshot.configuration_list[start_index:]
        ):
            self._execute_on_platform(index, config)

        self.results_pipe.send(self.results_list)

        self.logger.info(
            "[EXECUTION-ENGINE] All configurations processed. Sending results."
        )

        self.log_signal_pipe.send(ProcessSignal.FinalStop)
        self.logger.info("[EXECUTION-ENGINE] Sent final stop signal to logger.")

    # region Platform Execution
    def _execute_on_platform(self, index: int, config: ExecutionConfiguration):
        """Routes the execution to the correct TensorFlow device context."""
        device = "/gpu:0" if config.platform == Platform.GPU else "/cpu:0"
        self.logger.info(
            f"[EXECUTION-ENGINE] Executing on {device} for config: {config}"
        )
        with tf.device(device):
            self._run_single_configuration(config, index)

    def _run_single_configuration(self, config: ExecutionConfiguration, index: int):
        """
        Orchestrates the entire process for a single hyperparameter configuration.
        """
        self.logger.info(
            f"[EXECUTION-ENGINE] Starting execution for model "
            f"{self.runtime_snapshot.model_name}"
        )

        # 1. Prepare datasets (X_train, y_train, etc.)
        X_train, y_train, X_test, y_test = self._prepare_datasets_from_config(config)

        # 2. Determine model input shape
        input_shape = (
            X_train.shape[1] if self.is_config_train(config) else X_test.shape[1]
        )

        # 3. Assemble or load the Keras model
        model = self._get_model_for_config(config, input_shape, index)

        # 4. Compile the model
        model.compile(
            jit_compile=False,  # type: ignore
            loss=self.runtime_snapshot.model_execution_configuration.loss_metric_str,
            optimizer=self.runtime_snapshot.model_execution_configuration.optimizer,
            metrics=self.runtime_snapshot.model_execution_configuration.performance_metrics_list,
        )

        model.summary(print_fn=self.logger.info)

        # 5. Run the measurement routine
        processed_results = self._measurement_routine(
            model, config, X_train, y_train, X_test, y_test
        )

        # 7. Store results and perform cleanup
        self.results_list.append((config, processed_results))
        self.runtime_snapshot.last_profiled_index = index
        self.runtime_snapshot.save()
        collected = gc.collect()
        self.logger.info(f"Garbage collector: collected {collected} objects.")

    def is_config_train(self, config):
        return Lifecycle.TRAIN in config.cycle

    # endregion

    # region Measurement and Results Processing
    def _measurement_routine(
        self,
        model: keras.Model,
        config: ExecutionConfiguration,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Handles the core measurement loop for performance and energy.

        This function signals the logger, runs the model fitting or evaluation
        for a number of statistical samples, and calculates the final averaged
        metrics for time, energy, and performance.
        """
        sample_results = []

        # Signal logger to start recording energy
        self.log_signal_pipe.send((ProcessSignal.Start, config.platform))
        self.logger.info("[EXECUTION-ENGINE] Signaled logger to START.")

        total_time = 0.0

        for i in range(
            self.runtime_snapshot.model_execution_configuration.number_of_samples
        ):
            start_time = time()
            results = self._execute_sample(
                model, config, X_train, y_train, X_test, y_test
            )
            end_time = time()
            total_time = total_time + end_time - start_time

            # Save the trained model if required
            if self.is_config_train(config):
                self._save_model_to_storage(model, config, i)

            sample_results.append(results)

        # Signal logger to stop and send back data
        self.log_signal_pipe.send((ProcessSignal.Stop, config.platform))
        self.logger.info("[EXECUTION-ENGINE] Signaled logger to STOP.")

        energy_data = self.log_result_pipe.recv()

        # Process results
        total_elapsed_time_s = total_time
        processed_metrics = self._process_statistical_results(sample_results)
        processed_metrics["total_elapsed_time_s"] = total_elapsed_time_s

        avg_energy = self._calculate_average_energy(
            total_elapsed_time_s, energy_data, config.platform
        )
        processed_metrics["average_energy_consumption_joules"] = avg_energy

        self.logger.info(
            f"[EXECUTION-ENGINE] Total Elapsed time (s): {total_elapsed_time_s}"
        )
        self.logger.info(
            f"[EXECUTION-ENGINE] Average energy consumption (Joules): {avg_energy}"
        )

        return processed_metrics

    def _execute_sample(
        self,
        model: keras.Model,
        config: ExecutionConfiguration,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Executes a single sample run (either train or test)."""
        if self.is_config_train(config):
            model.fit(
                X_train,
                y_train,
                epochs=config.number_of_epochs,
                batch_size=self.runtime_snapshot.model_execution_configuration.batch_size,
                validation_split=0,
                verbose=1,  # type: ignore
            )

        # Always evaluate on the test set to get performance metrics
        return model.evaluate(X_test, y_test, verbose=0, return_dict=True)  # type: ignore

    def _process_statistical_results(self, sample_results: List[Dict]) -> Dict:
        """
        Calculates descriptive statistics for all collected metrics.

        For each metric (e.g., 'loss', 'accuracy'), this function computes
        the mean, median, standard deviation, min/max, and quartiles, providing
        a rich data set for generating box plots or other detailed visualizations.

        Args:
            sample_results: A list of dictionaries, where each dictionary is
                            a result from a single sample run.

        Returns:
            A dictionary where keys are metric names and values are another
            dictionary containing the calculated statistics for that metric.
        """
        if not sample_results:
            return {}

        # Gather all unique metric keys from all sample runs
        all_metrics = set()
        for result in sample_results:
            all_metrics.update(result.keys())

        processed_results = {}
        for metric in all_metrics:
            # Collect all valid values for the current metric
            metric_values = [
                res[metric]
                for res in sample_results
                if metric in res and res[metric] is not None
            ]

            if not metric_values:
                continue

            # Use numpy for efficient and robust statistical calculations
            values_np = np.array(metric_values)

            # For a single sample, stdev is 0 and all values are the same.
            if len(values_np) > 1:
                std_dev = np.std(values_np)
            else:
                std_dev = 0

            # Store a rich set of statistics for each metric
            processed_results[metric] = {
                "mean": np.mean(values_np),
                "std_dev": std_dev,
                "median": np.median(values_np),
                "min": np.min(values_np),
                "max": np.max(values_np),
                "p25": np.percentile(values_np, 25),  # 25th percentile (Q1)
                "p75": np.percentile(values_np, 75),  # 75th percentile (Q3)
            }

        return processed_results

    def _calculate_average_energy(
        self, total_time_s: float, energy_data, platform: Platform
    ) -> float:
        """Calculates the average energy consumption in Joules."""
        total_energy_joules = 0
        if platform == Platform.GPU and energy_data:
            # For GPU, energy_data is a list of power readings in Watts
            avg_power = statistics.mean(energy_data)
            total_energy_joules = avg_power * total_time_s
        elif platform == Platform.CPU and energy_data:
            # For CPU, energy_data is total energy in micro-Joules
            total_energy_joules = energy_data / 1_000_000

        if not total_energy_joules:
            return 0.0

        num_samples = (
            self.runtime_snapshot.model_execution_configuration.number_of_samples
        )
        return total_energy_joules / num_samples

    # endregion

    # region Data Preparation
    def _prepare_datasets_from_config(self, config: ExecutionConfiguration):
        """Selects features and splits data based on the configuration."""
        model_execution_configuration = (
            self.runtime_snapshot.model_execution_configuration
        )
        full_dataset = model_execution_configuration.dataset

        full_dataset = full_dataset.sample(frac=config.sampling_rate)

        # Select the number of features
        num_features = min(config.number_of_features, full_dataset.shape[1] - 1)
        features = full_dataset.iloc[:, 0:num_features].values
        target = full_dataset[
            [model_execution_configuration.dataset_target_label]
        ].values

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # If we are only testing, we don't need the training data
        if self.is_config_test(config):
            X_train, y_train = np.array([]), np.array([])
            X_test, y_test = features, target

        self.logger.info(
            f"[EXECUTION-ENGINE] Using dataset with {X_test.shape[1]} features."
        )

        return (
            np.asarray(X_train).astype(np.float32),
            np.asarray(y_train).astype(np.float32),
            np.asarray(X_test).astype(np.float32),
            np.asarray(y_test).astype(np.float32),
        )

    def is_config_test(self, config):
        return Lifecycle.TEST in config.cycle

    # endregion

    # region Model Assembly and Handling
    def _get_model_for_config(
        self, config: ExecutionConfiguration, input_shape: int, index: int
    ) -> keras.Model:
        """
        Provides a Keras model for the given configuration.

        It will first attempt to load a pre-trained model from storage if in
        a 'Test' lifecycle. If not found or in a 'Train' lifecycle, it builds
        a new model from scratch.
        """
        if self.is_config_test(config):
            loaded_model = self._try_load_model_from_storage(config, index)
            if loaded_model:
                self.logger.info(
                    "[EXECUTION-ENGINE] Loaded pre-trained model from storage."
                )
                return loaded_model
            self.logger.info(
                "[EXECUTION-ENGINE] Pre-trained model not found. Building new model."
            )

        return self._build_new_model(config, input_shape)

    def _build_new_model(
        self, config: ExecutionConfiguration, input_shape: int
    ) -> keras.Model:
        """Assembles a new Keras model using the provided layer lambdas."""
        model_config = self.runtime_snapshot.model_execution_configuration
        model = self.user_model_function  # Start with the base user model

        model_config.first_custom_layer_code(model, config.number_of_units, input_shape)
        for _ in range(config.number_of_layers - 1):
            model_config.repeated_custom_layer_code(
                model, config.number_of_units, input_shape
            )
        model_config.final_custom_layer_code(model)

        return model

    def _get_model_filepaths(
        self, config: ExecutionConfiguration, index: int
    ) -> Tuple[str, str]:
        """Generates the filepaths for a model's JSON and weights."""
        base_name = (
            f"{self.runtime_snapshot.model_name}_"
            f"{config.number_of_layers}_{config.number_of_units}_"
            f"{config.number_of_epochs}_{config.number_of_features}_{config.sampling_rate}_{index}"
        )
        json_path = f"./models/json_models/{base_name}.json"
        weights_path = f"./models/models_weights/{base_name}.weights.h5"
        return json_path, weights_path

    def _try_load_model_from_storage(
        self, config: ExecutionConfiguration, index: int
    ) -> keras.Model | None:
        """
        Tries to load a model's architecture and weights from disk.

        Returns:
            The loaded Keras model if successful, otherwise None.
        """
        json_path, weights_path = self._get_model_filepaths(config, index)
        if not path.exists(json_path) or not path.exists(weights_path):
            return None

        try:
            with open(json_path, "r") as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(weights_path)
            return model
        except Exception as e:
            self.logger.error(f"[EXECUTION-ENGINE] Error loading model: {e}")
            return None

    def _save_model_to_storage(
        self, model: keras.Model, config: ExecutionConfiguration, index: int
    ):
        """Saves a model's architecture and weights to disk."""
        json_path, weights_path = self._get_model_filepaths(config, index)

        try:
            # Save architecture
            model_json = model.to_json()
            with open(json_path, "w") as json_file:
                json_file.write(model_json)

            # Save weights
            model.save_weights(weights_path)
            self.logger.info(f"[EXECUTION-ENGINE] Saved model to {json_path}")
        except Exception as e:
            self.logger.error(f"[EXECUTION-ENGINE] Error saving model: {e}")

    # endregion
