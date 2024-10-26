import multiprocessing
from multiprocessing.connection import Connection
from os import path
from time import sleep
from typing import List
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from framework_parameters import (
    ExecutionConfiguration,
    Lifecycle,
    Platform,
    ProcessSignal,
)
from keras.models import model_from_json  # saving and loading trained model


class ExecutionEngine(multiprocessing.Process):
    def __init__(
        self,
        model: keras.models.Model,
        model_name: str,
        repeat_layers_set: List[keras.layers.Layer],
        configuration_list: List[ExecutionConfiguration],
        number_of_samples: int,
        batch_size: int,
        performance_metrics_list: List[str],
        dataset: pd.DataFrame,
        dataset_target_label: str,
        loss_metric_str: str,
        optimizer: str,
        start_pipe: Connection,
        log_pipe: Connection,
    ):
        super(ExecutionEngine, self).__init__()
        self.underlying_model = model
        self.model_name = model_name
        self.repeat_layers = repeat_layers_set
        self.configuration_list = configuration_list
        self.number_of_samples = number_of_samples
        self.batch_size = batch_size
        self.performance_metrics_list = performance_metrics_list
        self.dataset = dataset
        self.dataset_target_label = dataset_target_label
        self.loss_metric_str = loss_metric_str
        self.optimizer = optimizer
        self.start_pipe = start_pipe
        self.log_pipe = log_pipe

    def run(self):
        _ = (
            self.start_pipe.recv()
        )  ### Blocking recv call to wait start signal from manager.
        stop = False
        self.test_results_list = []

        first_configuration = self.configuration_list[0]
        current_platform = first_configuration.platform

        for configuration in self.configuration_list:
            self.__load_model(configuration, X_train.shape[1])

            ## TODO: implement sampling rate via test and training proportion and via proportion of the dataset
            X_train, y_train, X_test, y_test = self.__get_x_and_y(first_configuration)

            self.underlying_model.compile(
                loss=self.loss_metric_str,
                optimizer=self.optimizer_str,
                metrics=self.performance_metrics_list,
            )

            if configuration.platform.value == Platform.CPU:
                with tf.device("/cpu:0"):
                    self.log_pipe.send(ProcessSignal.Start)

                    for i in range(0, self.number_of_samples - 1):
                        test_results = self.__execute_configuration(
                            configuration, X_train, y_train, X_test, y_test
                        )
                        self.test_results_list.append(test_results)

                    self.log_pipe.send(ProcessSignal.Stop)
            else:
                with tf.device("/gpu:0"):
                    self.log_pipe.send(ProcessSignal.Start)

                    for i in range(0, self.number_of_samples - 1):
                        test_results = self.__execute_configuration(
                            configuration, X_train, y_train, X_test, y_test
                        )
                        self.test_results_list.append(test_results)

            self.log_pipe.send(ProcessSignal.Stop)

            if (
                self.configuration_list.index(configuration) == 0
                and configuration.cycle.value == Lifecycle.Train
            ):
                self.__save_model(self.underlying_model, self.model_name)

    def __execute_configuration(self, configuration, X_train, y_train, X_test, y_test):
        if configuration.cycle.value == Lifecycle.Train:
            _ = self.underlying_model.fit(
                X_train,
                y_train,
                epochs=configuration.number_of_epochs,
                batch_size=self.batch_size,
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

    def __get_x_and_y(self, configuration: ExecutionConfiguration):
        (X_train, y_train, X_test, y_test) = (
            self.__convert_dataset_into_x_and_y(
                self.dataset_target_label, configuration.number_of_features
            )
            if configuration.cycle.value == Lifecycle.Train
            else self.__get_full_dataset(
                self.dataset_target_label, configuration.number_of_features
            )
        )

        return X_train, y_train, X_test, y_test

    def __load_model(self, configuration: ExecutionConfiguration, x_train_shape: int):
        if configuration.cycle.value == Lifecycle.Train:
            for i in range(0, configuration.number_of_layers):
                for layer in self.repeat_layers:
                    self.underlying_model.add(
                        layer(
                            units=configuration.number_of_neurons,
                            input_shape=(x_train_shape, 1),
                        )
                    )
        elif configuration.cycle.value == Lifecycle.Test:
            self.underlying_model = self.__load_model(self.model_name)

    def __convert_dataset_into_x_and_y(
        self, target_label: str, number_of_features: int
    ):
        preprocessed_data = self.dataset

        original_number_of_features = preprocessed_data.shape[1] - 1
        current_number_of_features = (
            original_number_of_features
            if number_of_features > original_number_of_features
            else number_of_features
        )

        features = preprocessed_data.iloc[:, 0:current_number_of_features].values
        target = preprocessed_data[[target_label]].values

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        X_train = np.asarray(X_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        return (X_train, y_train, X_test, y_test)

    def __get_full_dataset(self, target_label: str, number_of_features: int):
        preprocessed_data = self.dataset

        original_number_of_features = preprocessed_data.shape[1] - 1
        current_number_of_features = (
            original_number_of_features
            if number_of_features > original_number_of_features
            else number_of_features
        )

        features = preprocessed_data.iloc[:, 0:current_number_of_features].values
        target = preprocessed_data[[target_label]].values

        X_test = np.asarray(features).astype(np.float32)
        Y_test = np.asarray(target).astype(np.float32)
        return ([], [], X_test, Y_test)

    def __load_model(self, model_name: str):
        filepath = f"./models/json_models/{model_name}_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.json"
        weightspath = f"./models/models_weights/{model_name}_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.weights.h5"
        json_file = open(filepath, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(weightspath)
        return model

    def __save_model(self, model, model_name):
        filepath = f"./models/json_models/{model_name}_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.json"
        weightspath = f"./models/models_weights/{model_name}_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.weights.h5"
        if not path.isfile(filepath):
            mlp_json = model.to_json()
            with open(filepath, "w") as json_file:
                json_file.write(mlp_json)

            model.save_weights(weightspath)
