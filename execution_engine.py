import multiprocessing
from multiprocessing.connection import Connection
from os import path
import os
import statistics
from time import sleep, time
from typing import List
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from framework_parameters import (
    EnvironmentConfiguration,
    ExecutionConfiguration,
    Lifecycle,
    Platform,
    ProcessSignal,
)
from keras._tf_keras.keras.models import (
    model_from_json,
)  # saving and loading trained model


class ExecutionEngine(multiprocessing.Process):
    def __init__(
        self,
        configuration_list: List[ExecutionConfiguration],
        model_func,
        model_name: str,
        environment: EnvironmentConfiguration,
        internal_logger,
    ):
        super(ExecutionEngine, self).__init__()
        self.underlying_model_func = model_func
        self.model_name = model_name
        self.configuration_list = configuration_list
        self.environment = environment
        self.internal_logger = internal_logger

    def run(self):
        _ = self.environment.start_pipe.recv()  ### Blocking recv call to wait star
        self.results_list = []

        for configuration in self.configuration_list:
            if configuration.platform.value == Platform.CPU.value:
                with tf.device("/cpu:0"):
                    self.internal_logger.info("Starting execution on CPU.")
                    self.execute_configuration(configuration)
            else:
                with tf.device("/gpu:0"):
                    self.internal_logger.info("Starting execution on GPU.")
                    self.execute_configuration(configuration)

        self.environment.results_pipe.send(self.results_list)
        self.environment.log_signal_pipe.send(
            (ProcessSignal.FinalStop, configuration.platform)  # type: ignore
        )

    def execute_configuration(self, configuration):
        self.internal_logger.info(
            f"Starting execution for model {self.model_name} config: {configuration}"
        )

        underlying_model = self.underlying_model_func()
        ## TODO: implement sampling rate via test and training proportion and via proportion of the dataset
        X_train, y_train, X_test, y_test = self.__select_x_and_y_from_config(
            configuration
        )
        input_shape = (
            X_train.shape[1]
            if configuration.cycle.value == Lifecycle.Train.value
            else X_test.shape[1]
        )

        self.__assemble_model_for_config(
            underlying_model,
            configuration,
            input_shape,
        )

        underlying_model.compile(
            jit_compile=False,
            loss=self.environment.loss_metric_str,
            optimizer=self.environment.optimizer,
            metrics=self.environment.performance_metrics_list,
        )
        underlying_model.summary()

        processed_results = {}

        processed_results = self.__execution_routine(
            underlying_model, configuration, X_train, y_train, X_test, y_test
        )

        if self.__is_train_config(configuration):
            self.internal_logger.info("Saving model to storage.")
            self.__save_model_to_storage(underlying_model, configuration)

        self.results_list.append((configuration, processed_results))

    def __execution_routine(
        self,
        underlying_model: keras.models.Model,
        configuration: ExecutionConfiguration,
        X_train: np.ndarray[np.floating],  # type: ignore
        y_train: np.ndarray[np.floating],  # type: ignore
        X_test: np.ndarray[np.floating],  # type: ignore
        y_test: np.ndarray[np.floating],  # type: ignore
    ) -> dict:
        current_results = []
        self.environment.log_signal_pipe.send(
            (ProcessSignal.Start, configuration.platform)
        )

        start_time = time()  ### Measurement cycle BEGIN

        for i in range(0, self.environment.number_of_samples):
            test_results = self.__execute_sample(
                underlying_model, configuration, X_train, y_train, X_test, y_test
            )
            current_results.append(test_results)

        end_time = time()  ### Measuremente cycle END

        self.environment.log_signal_pipe.send(
            (ProcessSignal.Stop, configuration.platform)
        )

        gpu_sampled_power_list = []  ### Energy consumption log gathering
        cpu_total_energy_micro_joules = None
        if configuration.platform.value == Platform.GPU.value:
            gpu_sampled_power_list = self.environment.log_result_pipe.recv()
        elif configuration.platform.value == Platform.CPU.value:
            cpu_total_energy_micro_joules = self.environment.log_result_pipe.recv()

        elapsed_time_ms = int((end_time - start_time) * 1000)

        processed_results = self.__process_results(
            current_results,
        )

        processed_results["average_elapsed_time_ms"] = (
            elapsed_time_ms / self.environment.number_of_samples
        )

        total_energy_joules = 0

        if cpu_total_energy_micro_joules == None and len(gpu_sampled_power_list) != 0:
            average_power_consumption = statistics.mean(gpu_sampled_power_list)
            total_energy_joules = average_power_consumption * elapsed_time_ms / 1000
        if cpu_total_energy_micro_joules != None and len(gpu_sampled_power_list) == 0:
            total_energy_joules = cpu_total_energy_micro_joules / 1000000

        processed_results["average_energy_consumption_joules"] = (
            total_energy_joules / self.environment.number_of_samples
        )

        return processed_results

    def __process_results(
        self,
        current_results: List[dict],
    ) -> dict:
        all_metrics = set()
        for result in current_results:
            all_metrics.update(result.keys())

        processed_results = {}
        total_samples = 0

        for metric in all_metrics:
            metric_values = []
            for result in current_results:
                if metric in result:
                    metric_values.append(result[metric])

            total_samples = len(metric_values)
            processed_results[metric] = {
                "mean": statistics.mean(metric_values),
                "median": statistics.median(metric_values),
                "std": statistics.stdev(metric_values),
                "error": statistics.stdev(metric_values) / len(metric_values) ** 0.5,
                "total_samples": total_samples,
            }

        return processed_results

    def __is_train_config(self, configuration) -> bool:
        return configuration.cycle.value == Lifecycle.Train.value

    def __execute_sample(
        self,
        underlying_model: keras.models.Model,
        configuration: ExecutionConfiguration,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> dict:  # type: ignore
        test_results = {}
        if configuration.cycle.value == Lifecycle.Train.value:
            _ = underlying_model.fit(
                X_train,
                y_train,
                epochs=configuration.number_of_epochs,
                batch_size=self.environment.batch_size,
                validation_split=0.2,
            )
            test_results = underlying_model.evaluate(
                X_test, y_test, verbose=1, return_dict=True  # type: ignore
            )
            return test_results
        elif configuration.cycle.value == Lifecycle.Test.value:
            test_results = underlying_model.evaluate(
                X_test, y_test, verbose=1, return_dict=True  # type: ignore
            )
            return test_results

    def __select_x_and_y_from_config(
        self, configuration: ExecutionConfiguration
    ) -> tuple[
        np.ndarray[np.floating],  # type: ignore
        np.ndarray[np.floating],  # type: ignore
        np.ndarray[np.floating],  # type: ignore
        np.ndarray[np.floating],  # type: ignore
    ]:
        (X_train, y_train, X_test, y_test) = (
            self.__get_x_and_y(configuration.number_of_features)
            if configuration.cycle.value == Lifecycle.Train.value
            else self.__get_full_dataset(configuration.number_of_features)
        )

        return X_train, y_train, X_test, y_test

    def __assemble_model_for_config(
        self,
        underlying_model: keras.models.Model,
        configuration: ExecutionConfiguration,
        x_train_shape: int,
    ):
        self.internal_logger.info(f"Current x_train_shape: {x_train_shape}")
        if configuration.cycle.value == Lifecycle.Test.value:
            (load_sucess, loaded_model) = self.__try_load_model_from_storage(
                configuration
            )
            if load_sucess:
                self.internal_logger.info("Subject model was loaded from storage.")
                underlying_model = loaded_model
                return
            else:
                self.internal_logger.info(
                    "Subject model was not avaible in storage. Assembling model."
                )
        for i in range(0, configuration.number_of_layers):
            self.environment.repeated_custom_layer_code(
                underlying_model, configuration.number_of_units, x_train_shape
            )
        self.environment.final_custom_layer_code(underlying_model)

    def __get_x_and_y(self, number_of_features: int) -> tuple[
        np.ndarray[np.floating],  # type: ignore
        np.ndarray[np.floating],  # type: ignore
        np.ndarray[np.floating],  # type: ignore
        np.ndarray[np.floating],  # type: ignore
    ]:
        preprocessed_data = self.environment.dataset

        original_number_of_features = preprocessed_data.shape[1] - 1
        current_number_of_features = (
            original_number_of_features
            if number_of_features > original_number_of_features
            else number_of_features
        )

        features = preprocessed_data.iloc[:, 0:current_number_of_features].values
        target = preprocessed_data[[self.environment.dataset_target_label]].values

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        X_train = np.asarray(X_train).astype(np.float32)
        self.internal_logger.info(f"Using dataset with {X_train.shape[1]} features")
        X_test = np.asarray(X_test).astype(np.float32)
        return (X_train, y_train, X_test, y_test)

    def __get_full_dataset(self, number_of_features: int) -> tuple:
        preprocessed_data = self.environment.dataset

        original_number_of_features = preprocessed_data.shape[1] - 1
        current_number_of_features = (
            original_number_of_features
            if number_of_features > original_number_of_features
            else number_of_features
        )

        features = preprocessed_data.iloc[:, 0:current_number_of_features].values
        target = preprocessed_data[[self.environment.dataset_target_label]].values

        X_test = np.asarray(features).astype(np.float32)
        Y_test = np.asarray(target).astype(np.float32)
        return ([], [], X_test, Y_test)

    def __try_load_model_from_storage(
        self,
        configuration: ExecutionConfiguration,
    ) -> tuple[bool, keras.models.Model]:
        filepath = f"./models/json_models/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}_{configuration.platform}.json"
        weightspath = f"./models/models_weights/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}_{configuration.platform}.weights.h5"

        return_model = None
        try:
            with open(filepath, "r") as json_file:
                loaded_model_json = json_file.read()
                return_model = model_from_json(loaded_model_json)
                json_file.close()
                return_model.load_weights(weightspath)
                return (True, return_model)
        except (FileNotFoundError, IOError, ValueError) as e:
            return (False, None)  # type: ignore

    def __save_model_to_storage(
        self,
        underlying_model: keras.models.Model,
        configuration: ExecutionConfiguration,
    ):
        filepath = f"./models/json_models/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}_{configuration.platform}.json"
        weightspath = f"./models/models_weights/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}_{configuration.platform}.weights.h5"
        if not path.isfile(filepath):
            model_json = underlying_model.to_json()
            with open(filepath, "w") as json_file:
                json_file.write(model_json)

            underlying_model.save_weights(weightspath)
