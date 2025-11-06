"""
This module defines the ExecutionEngine, a dedicated process responsible for
running model training and testing based on a list of configurations.

It handles platform-specific execution (CPU/GPU), data preparation, model
assembly, performance and energy measurement, and result aggregation.
"""

import gc
import multiprocessing
import os
import statistics
from logging import Logger
from multiprocessing.connection import Connection
from os import path
from time import time
from typing import Dict, List, Tuple
from pathlib import Path

import keras
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import model_from_json

from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.execution_result import (
    EnergyExecutionResult,
    ExecutionResult,
    Metrics,
    ResultsWriter,
    StatisticalValues,
    TimeAndEnergy,
)
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
        user_model_function,
        profile_execution_directory: str,
        runtime_snapshot: RuntimeSnapshot,
        logger: Logger,
        start_pipe: Connection,
        log_signal_pipe: Connection,
        energy_monitor_pipe: Connection,
        execution_timeout_seconds: int | None = None,
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
            execution_timeout_seconds: Optional timer in seconds to limit total execution time.
        """
        super().__init__()
        self.user_model_function = user_model_function
        self.runtime_snapshot = runtime_snapshot
        self.logger = logger
        self.start_pipe = start_pipe
        self.signal_energy_pipe = log_signal_pipe
        self.energy_monitor_pipe = energy_monitor_pipe
        self.results_writer = ResultsWriter(logger, profile_execution_directory)
        self.model_directory = profile_execution_directory + "models/"
        self.execution_timeout_seconds = execution_timeout_seconds
        self.execution_start_time = 0.0

    def run(self):
        """
        The main entry point for the process.

        Waits for a start signal, then iterates through all configurations,
        executing them on the appropriate hardware platform. If a timeout is
        set, it will stop execution when the time is up. Finally, it sends
        the collected results back to the main process.
        """
        self.start_pipe.recv()  # Wait for the start signal from the manager
        self.logger.info(
            "[EXECUTION-ENGINE] Start signal received. Beginning profiling."
        )
        self.execution_start_time = time()

        start_index = self.runtime_snapshot.last_profiled_index + 1
        total_configs = len(self.runtime_snapshot.configuration_list)

        # for config in self.runtime_snapshot.configuration_list:
        #     print(config)

        for index in range(start_index, len(self.runtime_snapshot.configuration_list)):
            config = self.runtime_snapshot.configuration_list[index]

            # --- Progress Bar Start ---
            progress = (index + 1) / total_configs
            bar_length = 40
            block = int(round(bar_length * progress))
            text = f"\rProgress: [{'#' * block}{'-' * (bar_length - block)}] {index + 1}/{total_configs} ({progress * 100:.2f}%)"
            print(text)
            # --- Progress Bar End ---

            if not self._execute_on_platform(index, config):
                self.logger.info(
                    f"[EXECUTION-ENGINE] Execution timeout of {self.execution_timeout_seconds}s reached. "
                    "Stopping profiling. State has been saved to resume later."
                )
                # Exit the loop early
                break

        self.signal_energy_pipe.send((ProcessSignal.FinalStop, None))
        self.logger.info("[EXECUTION-ENGINE] Sent final stop signal to logger.")

    # region Platform Execution
    def _execute_on_platform(self, index: int, config: ExecutionConfiguration) -> bool:
        """
        Routes the execution to the correct TensorFlow device context.
        Returns True to continue, or False if the execution timeout is reached.
        """
        device = "/gpu:0" if config.platform == Platform.GPU else "/cpu:0"
        self.logger.info(
            f"[EXECUTION-ENGINE] Executing on {device} for config: {config}"
        )
        with tf.device(device):
            self._run_single_configuration(config, index)

        # Check if the execution timer has been reached
        if (
            self.execution_timeout_seconds is not None
            and self.execution_timeout_seconds > 0
        ):
            elapsed_time = time() - self.execution_start_time
            if elapsed_time >= self.execution_timeout_seconds:
                return False  # Signal to stop

        return True  # Signal to continue

    def _run_single_configuration(self, config: ExecutionConfiguration, index: int):
        """
        Orchestrates the entire process for a single hyperparameter configuration.
        """
        self.logger.info(
            f"[EXECUTION-ENGINE] Starting execution for model "
            f"{self.runtime_snapshot.model_name}"
        )

        X_train, y_train, X_test, y_test = self._prepare_datasets_from_config(config)

        processed_results, average_elapsed_time_s, average_energy = (
            self._measurement_routine(config, X_train, y_train, X_test, y_test)
        )

        self.results_writer.append_execution_result(
            ExecutionResult((config, processed_results))
        )
        self.results_writer.append_energy_result(
            EnergyExecutionResult(
                (config, TimeAndEnergy(average_elapsed_time_s, average_energy))
            )
        )

        self.runtime_snapshot.last_profiled_index = index
        self.runtime_snapshot.save()
        collected = gc.collect()
        self.logger.info(
            f"[EXECUTION-ENGINE] Garbage collector: collected {collected} objects."
        )

    def is_config_train(self, config):
        return Lifecycle.TRAIN in config.cycle

    # endregion

    # region Measurement and Results Processing
    def _measurement_routine(
        self,
        config: ExecutionConfiguration,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[Metrics, float, float]:
        """
        Handles the core measurement loop for performance and energy.

        This function orchestrates a full measurement cycle. It signals the
        energy profiler, runs the model training or evaluation for a number of
        statistical samples, and calculates the final averaged metrics.

        It includes a retry mechanism: if the energy measurement from the
        profiler is invalid (resulting in 0 Joules), it will automatically
        repeat the entire measurement process up to a defined number of times.
        """
        retry_count = 0
        measurement_successful = False

        processed_metrics: Metrics = {}
        total_elapsed_time_s = 0.0
        avg_energy = 0.0

        while not measurement_successful:
            sample_results = []
            total_time_this_attempt = 0.0

            self.logger.info(
                f"[EXECUTION-ENGINE] Starting measurement attempt {retry_count + 1}..."
            )

            self.signal_energy_pipe.send((ProcessSignal.Start, config.platform))
            self.logger.info("[EXECUTION-ENGINE] Signaled profiler to START.")

            for i in range(
                self.runtime_snapshot.model_execution_configuration.number_of_samples
            ):
                input_shape = (
                    X_train.shape[1]
                    if self.is_config_train(config)
                    else X_test.shape[1]
                )

                model = self._get_model_for_config(config, input_shape, i)

                model.compile(
                    jit_compile=False,  # type: ignore
                    loss=self.runtime_snapshot.model_execution_configuration.loss_metric_str,
                    optimizer=self.runtime_snapshot.model_execution_configuration.optimizer,
                    metrics=self.runtime_snapshot.model_execution_configuration.performance_metrics_list,
                )

                model.summary(print_fn=self.logger.info)

                start_time = time()
                results = self._execute_sample(
                    model, config, X_train, y_train, X_test, y_test
                )
                end_time = time()
                total_time_this_attempt += end_time - start_time

                if self.is_config_train(config):
                    self._save_model_to_storage(model, config, i)
                sample_results.append(results)

            self.signal_energy_pipe.send((ProcessSignal.Stop, config.platform))
            self.logger.info("[EXECUTION-ENGINE] Signaled profiler to STOP.")
            energy_data = self.energy_monitor_pipe.recv()

            current_avg_energy = self._calculate_average_energy(
                total_time_this_attempt, energy_data, config.platform
            )

            if current_avg_energy > 0:
                self.logger.info(
                    "[EXECUTION-ENGINE] Successfully obtained valid energy measurement."
                )
                measurement_successful = True
                avg_energy = current_avg_energy
                total_elapsed_time_s = total_time_this_attempt
                processed_metrics = self._process_statistical_results(sample_results)
            else:
                retry_count += 1
                self.logger.warning(
                    f"[EXECUTION-ENGINE] Attempt {retry_count}: "
                    "Invalid energy data received (0 Joules). Retrying measurement."
                )

        self.logger.info(
            f"[EXECUTION-ENGINE] Total Elapsed time (s): {total_elapsed_time_s}"
        )
        self.logger.info(
            f"[EXECUTION-ENGINE] Final Average energy consumption (Joules): {avg_energy}"
        )

        average_execution_time_s = (
            total_elapsed_time_s
            / self.runtime_snapshot.model_execution_configuration.number_of_samples
        )

        return processed_metrics, average_execution_time_s, avg_energy

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
                epochs=config.epochs,
                batch_size=self.runtime_snapshot.model_execution_configuration.batch_size,
                validation_split=0,
                verbose=0,  # type: ignore
            )

        return model.evaluate(X_test, y_test, verbose=0, return_dict=True)  # type: ignore

    def _process_statistical_results(self, sample_results: List[Dict]) -> Metrics:
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

        all_metrics = set()
        for result in sample_results:
            all_metrics.update(result.keys())

        processed_results = {}
        for metric in all_metrics:
            metric_values = [
                res[metric]
                for res in sample_results
                if metric in res and res[metric] is not None
            ]

            if not metric_values:
                continue

            values_np = np.array(metric_values)

            if len(values_np) > 1:
                std_dev = np.std(values_np)
            else:
                std_dev = 0

            processed_results[metric] = StatisticalValues(
                mean=np.mean(values_np),  # type: ignore
                std_dev=std_dev,  # type: ignore
                median=np.median(values_np),  # type: ignore
                min=np.min(values_np),
                max=np.max(values_np),
                p25=np.percentile(values_np, 25),  # type: ignore
                p75=np.percentile(values_np, 75),  # type: ignore
            )

        return processed_results

    def _calculate_average_energy(
        self, total_time_s: float, energy_data, platform: Platform
    ) -> float:
        """Calculates the average energy consumption in Joules."""
        total_energy_joules = 0
        if platform == Platform.GPU and energy_data:
            avg_power = statistics.mean(energy_data)
            total_energy_joules = avg_power * total_time_s
        elif platform == Platform.CPU and energy_data:
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

        num_features = min(config.features, full_dataset.shape[1] - 1)
        features = full_dataset.iloc[:, 0:num_features].values
        target = full_dataset[
            [model_execution_configuration.dataset_target_label]
        ].values

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

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
                    f"[EXECUTION-ENGINE] Loaded pre-trained model from storage. Index {index}."
                )
                return loaded_model
            self.logger.info(
                f"[EXECUTION-ENGINE] Pre-trained model not found. Building new model. Index: {index}."
            )

        self.logger.info(f"[EXECUTION-ENGINE] Building new model. Index: {index}")
        return self._build_new_model(config, input_shape)

    def _build_new_model(
        self, config: ExecutionConfiguration, input_shape: int
    ) -> keras.models.Model:
        """Assembles a new Keras model using the provided layer lambdas."""
        model_config = self.runtime_snapshot.model_execution_configuration
        model = self.user_model_function()

        model_config.first_custom_layer_code(model, config.units, input_shape)
        for _ in range(config.layers):
            model_config.repeated_custom_layer_code(model, config.units, input_shape)
        model_config.final_custom_layer_code(model, config.units)

        return model

    def _get_model_filepaths(
        self, config: ExecutionConfiguration, index: int
    ) -> Tuple[str, str]:
        """Generates the filepaths for a model's JSON and weights."""
        base_name = (
            f"{self.runtime_snapshot.model_name}_"
            f"{config.layers}_{config.units}_"
            f"{config.epochs}_{config.features}_{config.sampling_rate}_{index}"
        )

        try:
            os.makedirs(self.model_directory, exist_ok=True)
        except IOError as e:
            self.logger.error(
                f"Error creating results directory {self.model_directory}: {e}"
            )
            raise

        try:
            os.makedirs(self.model_directory + f"json_models/", exist_ok=True)
        except IOError as e:
            self.logger.error(
                f"Error creating results directory {self.model_directory}json_models/: {e}"
            )
            raise

        try:
            os.makedirs(self.model_directory + f"models_weights/", exist_ok=True)
        except IOError as e:
            self.logger.error(
                f"Error creating results directory {self.model_directory}models_weights/: {e}"
            )
            raise

        json_path = self.model_directory + f"json_models/{base_name}.json"
        weights_path = self.model_directory + f"./models_weights/{base_name}.weights.h5"
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
            model_json = model.to_json()
            with open(json_path, "w") as json_file:
                json_file.write(model_json)

            model.save_weights(weights_path)
            self.logger.info(f"[EXECUTION-ENGINE] Saved model to {json_path}")
        except Exception as e:
            self.logger.error(f"[EXECUTION-ENGINE] Error saving model: {e}")

    # endregion
