import gc
import multiprocessing
from os import path
import statistics
from time import time
from typing import List
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from .framework_parameters import (
    EnvironmentConfiguration,
    ExecutionConfiguration,
    Lifecycle,
    Platform,
    ProcessSignal,
)
from keras._tf_keras.keras.models import (
    model_from_json,
)


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

        for (
            configuration
        ) in self.configuration_list:  ### Iterate over configuration list
            if configuration.platform.value == Platform.CPU.value:
                with tf.device("/cpu:0"):
                    self.internal_logger.info(
                        "[EXECUTION-ENGINE] Starting execution on CPU."
                    )
                    self.execute_configuration(configuration)
            else:
                with tf.device("/gpu:0"):
                    self.internal_logger.info(
                        "[EXECUTION-ENGINE] Starting execution on GPU."
                    )
                    self.execute_configuration(configuration)

        self.environment.results_pipe.send(self.results_list)
        self.internal_logger.info("[EXECUTION-ENGINE] Sending results list to manager.")
        self.environment.log_signal_pipe.send(
            (ProcessSignal.FinalStop, configuration.platform)  # type: ignore
        )
        self.internal_logger.info(
            "[EXECUTION-ENGINE] Sending logger final stop signal."
        )

    def execute_configuration(self, configuration):
        self.internal_logger.info(
            f"[EXECUTION-ENGINE] Starting execution for model {self.model_name} config: {configuration}"
        )

        underlying_model = self.underlying_model_func()  ### Start base user model.

        X_train, y_train, X_test, y_test = (
            self.__select_x_and_y_from_config(  ### Get dataset x,y; train,test.
                configuration
            )
        )

        input_shape = (  ### Get input shape.
            X_train.shape[1]
            if configuration.cycle.value == Lifecycle.Train.value
            else X_test.shape[1]
        )

        underlying_model = (
            self.__assemble_model_for_config(  ### Get from storage or assemble ML model
                underlying_model,
                configuration,
                input_shape,
            )
        )

        underlying_model.compile(  ### Compile model
            jit_compile=False,  # type: ignore
            loss=self.environment.loss_metric_str,
            optimizer=self.environment.optimizer,
            metrics=self.environment.performance_metrics_list,
        )

        underlying_model.summary()

        processed_results = self.__execution_routine(
            underlying_model, configuration, X_train, y_train, X_test, y_test
        )

        if self.__is_train_config(configuration):
            self.internal_logger.info("[EXECUTION-ENGINE] Saving model to storage.")
            self.__save_model_to_storage(underlying_model, configuration)

        self.results_list.append((configuration, processed_results))
        collected = gc.collect()

        self.internal_logger.info(f"Garbage collector: collected {collected} objects.")

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
        self.internal_logger.info("[EXECUTION-ENGINE] Sending logger start signal.")

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
        self.internal_logger.info("[EXECUTION-ENGINE] Sending logger stop signal.")

        gpu_sampled_power_list = []  ### Energy consumption log gathering
        cpu_total_energy_micro_joules = None
        if configuration.platform.value == Platform.GPU.value:
            gpu_sampled_power_list = self.environment.log_result_pipe.recv()
            self.internal_logger.info(
                "[EXECUTION-ENGINE] Received GPU sample power list."
            )
        elif configuration.platform.value == Platform.CPU.value:
            cpu_total_energy_micro_joules = self.environment.log_result_pipe.recv()
            self.internal_logger.info("[EXECUTION-ENGINE] Received CPU total energy")

        total_elapsed_time_s = end_time - start_time

        processed_results = self.__process_results(
            current_results,
        )

        self.internal_logger.info(
            "[EXECUTION-ENGINE] Total Elapsed time (s):" + str(total_elapsed_time_s)
        )
        processed_results["total_elapsed_time_s"] = total_elapsed_time_s

        total_energy_joules = 0

        if cpu_total_energy_micro_joules == None and len(gpu_sampled_power_list) != 0:
            average_power_consumption = statistics.mean(gpu_sampled_power_list)
            total_energy_joules = average_power_consumption * total_elapsed_time_s
        if cpu_total_energy_micro_joules != None and len(gpu_sampled_power_list) == 0:
            total_energy_joules = cpu_total_energy_micro_joules / 1000000

        average_energy = total_energy_joules / self.environment.number_of_samples
        processed_results["average_energy_consumption_joules"] = average_energy

        self.internal_logger.info(
            "[EXECUTION-ENGINE] Average energy consumption:" + str(average_energy)
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

        for metric in all_metrics:
            metric_values = []
            for result in current_results:
                if metric in result:
                    metric_values.append(result[metric])

            processed_results[metric] = {
                "mean": statistics.mean(metric_values),
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
                validation_split=0,
                verbose=1,  # type: ignore
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
    ) -> keras.models.Model:
        if configuration.cycle.value == Lifecycle.Test.value:
            (load_sucess, candidate_model) = self.__try_load_model_from_storage(
                underlying_model, configuration
            )
            if load_sucess:
                self.internal_logger.info(
                    "[EXECUTION-ENGINE] Subject model was loaded from storage."
                )
                return candidate_model
            else:
                self.internal_logger.info(
                    "[EXECUTION-ENGINE] Subject model was not avaible in storage. Assembling model."
                )
        self.environment.first_custom_layer_code(
            underlying_model, configuration.number_of_units, x_train_shape
        )
        for i in range(0, configuration.number_of_layers - 1):
            self.environment.repeated_custom_layer_code(
                underlying_model, configuration.number_of_units, x_train_shape
            )
        self.environment.final_custom_layer_code(underlying_model)
        return underlying_model

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
        X_test = np.asarray(X_test).astype(np.float32)

        self.internal_logger.info(
            f"[EXECUTION-ENGINE] Using dataset with {X_train.shape[1]} features"
        )

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
        underlying_model: keras.models.Model,
        configuration: ExecutionConfiguration,
    ) -> tuple[bool, keras.models.Model]:
        filepath = f"./models/json_models/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.json"
        weightspath = f"./models/models_weights/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.weights.h5"

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
        filepath = f"./models/json_models/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.json"
        weightspath = f"./models/models_weights/{self.model_name}_{configuration.number_of_layers}_{configuration.number_of_units}_{configuration.number_of_epochs}_{configuration.number_of_features}.weights.h5"
        if not path.isfile(filepath):
            model_json = underlying_model.to_json()
            with open(filepath, "w") as json_file:
                json_file.write(model_json)

            underlying_model.save_weights(weightspath)
