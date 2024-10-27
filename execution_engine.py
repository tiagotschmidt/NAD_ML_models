import multiprocessing
from multiprocessing.connection import Connection
from os import path
import statistics
from time import time
from typing import Callable, List
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
from keras.models import model_from_json  # saving and loading trained model


class ExecutionEngine(multiprocessing.Process):
    def __init__(
        self,
        configuration_list: List[ExecutionConfiguration],
        model: keras.models.Model,
        model_name: str,
        environment: EnvironmentConfiguration,
    ):
        super(ExecutionEngine, self).__init__()
        self.underlying_model = model
        self.model_name = model_name
        self.configuration_list = configuration_list
        self.environment = environment

    def run(self):
        _ = self.start_pipe.recv()  ### Blocking recv call to wait star
        self.results_list = []

        for configuration in self.configuration_list:
            ## TODO: implement sampling rate via test and training proportion and via proportion of the dataset
            X_train, y_train, X_test, y_test = self.__select_x_and_y_from_config(
                configuration
            )

            self.__assemble_model_for_config(
                configuration,
                self.environment.repeated_custom_layer_code,
                self.environment.final_custom_layer_code,
                X_train.shape[1],
            )

            self.underlying_model.compile(
                loss=self.environment.loss_metric_str,
                optimizer=self.environment.optimizer,
                metrics=self.environment.performance_metrics_list,
            )

            processed_results = {}

            if configuration.platform.value == Platform.CPU:
                with tf.device("/cpu:0"):
                    processed_results = self.__execution_routine(
                        configuration, X_train, y_train, X_test, y_test
                    )
            else:
                with tf.device("/gpu:0"):
                    processed_results = self.__execution_routine(
                        configuration, X_train, y_train, X_test, y_test
                    )

            if self.__is_first_config(configuration) and self.__is_train_config(
                configuration
            ):
                self.__save_model_to_storage(self.model_name)

            self.results_list.append((configuration, processed_results))

        self.environment.results_pipe.send(self.results_list)

    def __execution_routine(
        self,
        configuration: ExecutionConfiguration,
        X_train: np.ndarray[np.floating],
        y_train: np.ndarray[np.floating],
        X_test: np.ndarray[np.floating],
        y_test: np.ndarray[np.floating],
    ) -> dict:
        current_results = []
        self.environment.log_pipe.send(ProcessSignal.Start)
        start_time = time()

        for i in range(0, self.environment.number_of_samples):
            test_results = self.__execute_config(
                configuration, X_train, y_train, X_test, y_test
            )
            current_results.append(test_results)

        end_time = time()
        self.environment.log_pipe.send(ProcessSignal.Stop)

        elapsed_time_ms = int((end_time - start_time) * 1000)
        processed_results = self.__process_results(current_results)
        processed_results["elapsed_time_ms"] = elapsed_time_ms

        return processed_results

    def __process_results(self, current_results: List[dict]) -> dict:
        all_metrics = set()
        for result in current_results:
            all_metrics.update(result.keys())

        processed_results = {}

        for metric in all_metrics:
            metric_values = []
            for result in current_results:
                if metric in result:
                    metric_values.append(result[metric])

            processed_results[metric] = {
                "mean": statistics.mean(metric_values),
                "median": statistics.median(metric_values),
                "std": statistics.stdev(metric_values),
                "error": statistics.stdev(metric_values) / len(metric_values) ** 0.5,
                "total_samples": len(metric_values),
            }

        return processed_results

    def __is_train_config(self, configuration) -> bool:
        return configuration.cycle.value == Lifecycle.Train.value

    def __is_first_config(self, configuration) -> bool:
        return self.configuration_list.index(configuration) == 0

    def __execute_config(
        self, configuration: ExecutionConfiguration, X_train, y_train, X_test, y_test
    ) -> dict:
        if configuration.cycle.value == Lifecycle.Train:
            _ = self.underlying_model.fit(
                X_train,
                y_train,
                epochs=configuration.number_of_epochs,
                batch_size=self.environment.batch_size,
                validation_split=0.2,
            )
            test_results = self.underlying_model.evaluate(
                X_test, y_test, verbose=1, return_dict=True
            )
        elif configuration.cycle.value == Lifecycle.Test:
            test_results = self.underlying_model.evaluate(
                X_train, y_train, verbose=1, return_dict=True
            )

        return test_results

    def __select_x_and_y_from_config(
        self, configuration: ExecutionConfiguration
    ) -> tuple[
        np.ndarray[np.floating],
        np.ndarray[np.floating],
        np.ndarray[np.floating],
        np.ndarray[np.floating],
    ]:
        (X_train, y_train, X_test, y_test) = (
            self.__get_x_and_y(
                self.environment.dataset_target_label, configuration.number_of_features
            )
            if configuration.cycle.value == Lifecycle.Train
            else self.__get_full_dataset(configuration.number_of_features)
        )

        return X_train, y_train, X_test, y_test

    def __assemble_model_for_config(
        self,
        configuration: ExecutionConfiguration,
        x_train_shape: int,
    ):
        if configuration.cycle.value == Lifecycle.Test:
            sucess, returned_model = self.__try_load_model_from_storage(self.model_name)
            if sucess:
                self.underlying_model = returned_model
                return
        for i in range(0, configuration.number_of_layers):
            self.environment.repeated_custom_layer_code(
                self.underlying_model, configuration.number_of_units, x_train_shape
            )
        self.environment.final_custom_layer_code(self.underlying_model)

    def __get_x_and_y(self, number_of_features: int) -> tuple[
        np.ndarray[np.floating],
        np.ndarray[np.floating],
        np.ndarray[np.floating],
        np.ndarray[np.floating],
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
        X_test = np.asarray(X_test).astype(np.float32)
        return (X_train, y_train, X_test, y_test)

    def __get_full_dataset(self, number_of_features: int) -> tuple[
        np.ndarray[np.floating],
        np.ndarray[np.floating],
        np.ndarray[np.floating],
        np.ndarray[np.floating],
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

        X_test = np.asarray(features).astype(np.float32)
        Y_test = np.asarray(target).astype(np.float32)
        return ([], [], X_test, Y_test)

    def __try_load_model_from_storage(
        self,
        configuration: ExecutionConfiguration,
    ) -> bool:
        filepath = f"./models/json_models/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.json"
        weightspath = f"./models/models_weights/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.weights.h5"

        try:
            with open(filepath, "r") as json_file:
                loaded_model_json = json_file.read()
                self.underlying_model = model_from_json(loaded_model_json)
                self.underlying_model.load_weights(weightspath)
                return True
        except (FileNotFoundError, IOError, ValueError) as e:
            return False

    def __save_model_to_storage(
        self,
        configuration: ExecutionConfiguration,
    ):
        filepath = f"./models/json_models/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.json"
        weightspath = f"./models/models_weights/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.weights.h5"
        if not path.isfile(filepath):
            model_json = self.underlying_model.to_json()
            with open(filepath, "w") as json_file:
                json_file.write(model_json)

            self.underlying_model.save_weights(weightspath)
