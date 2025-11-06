import os

from keras.layers import LSTM, Dropout, Flatten
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np

from eppnad.core.manager import profile
from eppnad.utils.framework_parameters import (
    Lifecycle,
    PercentageRangeParameter,
    Platform,
    ProfileMode,
    RangeParameter,
)


def preprocess_ids_2018(dataset_list):
    preprocessed_dataset = pd.concat(dataset_list, ignore_index=True, axis=0)
    del dataset_list
    dataset = preprocessed_dataset.drop(
        columns=["Timestamp", "Dst Port"]
    )  # 78 features
    # print(preprocessed_dataset["Label"].unique())
    # print("Before")
    value = "Label"

    indices_with_value = preprocessed_dataset.index[
        preprocessed_dataset["Label"] == value
    ]  # locate the rows with class 'Label'
    # print(indices_with_value)
    preprocessed_dataset = preprocessed_dataset.drop(indices_with_value)
    # print(preprocessed_dataset["Label"].unique())
    dataset["Protocol"].unique()
    dataset = dataset.astype({"Protocol": str})  # Change the column data type to string
    dataset["Protocol"].unique()
    dataset = pd.get_dummies(dataset, columns=["Protocol"], drop_first=True)
    num_rows_with_nan = (
        dataset.isna().sum().sum()
    )  # Sum total number of nan values present

    # print("Number of rows with NaN values:", num_rows_with_nan)
    dataset.columns.to_series()[dataset.isna().any()]
    dataset = dataset.dropna()
    dataset = dataset.replace(
        np.inf, np.nan
    )  # replace them with nan because its simpler
    # print("Number of rows with infinite values:", dataset.isna().sum().sum())
    dataset.columns.to_series()[dataset.isna().any()]
    # print(f'Max of Flow Bytes/s: { max(dataset["Flow Byts/s"]) }')
    # print(f'Max of Flow Pkts/s: { max(dataset["Flow Pkts/s"]) }')
    # Replace 'nan' with the maximum value in its column
    dataset["Flow Byts/s"] = dataset["Flow Byts/s"].replace(
        np.nan, max(dataset["Flow Byts/s"])
    )
    dataset["Flow Pkts/s"] = dataset["Flow Pkts/s"].replace(
        np.nan, max(dataset["Flow Pkts/s"])
    )
    # print(dataset.isna().sum().sum())  # Check if any null values left
    le = LabelEncoder()
    le.fit(dataset["Label"])

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))  # type: ignore
    dataset["Label"] = le.transform(dataset["Label"])

    # print(mapping)

    string_columns = []

    for column in dataset.columns:
        if dataset[column].dtype == "object":
            string_columns.append(column)

    # Print columns containing string values
    # print("Columns containing string values:")
    # print(string_columns)
    dataset[string_columns] = dataset[string_columns].astype(
        float
    )  # Change data types of string columns to float
    dataset = dataset[dataset.ge(0).all(axis=1)]
    dataset = dataset.drop_duplicates(ignore_index=True)
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(dataset[dataset.columns[:78]])
    preprocessed_dataset = pd.DataFrame(
        normalized_features, columns=dataset.columns[:78]
    )
    preprocessed_dataset["Label"] = dataset["Label"]
    return preprocessed_dataset


def first_layer(model: keras.models.Model, number_of_units, input_shape):
    model.add(keras.Input(shape=(input_shape, 1)))


def repeated_layer(model: keras.models.Model, number_of_units, input_shape):
    model.add(LSTM(number_of_units, return_sequences=True))
    model.add(Dropout(0.5))


def final_layer(model: keras.models.Model, number_of_units):
    model.add(LSTM(number_of_units, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_units, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))


numbers_of_layers = RangeParameter([1, 10])
numbers_of_units = RangeParameter([1])
numbers_of_features = RangeParameter([13])

numbers_of_epochs = RangeParameter([10])
sampling_rates = PercentageRangeParameter([0.1])

# Define profile mode
profile_mode = ProfileMode(
    Lifecycle.TRAIN_AND_TEST,
    train_platform=Platform.GPU,
    test_platform=Platform.CPU,
)

# Define other parameters
number_of_samples = 2
batch_size = 8192 * 2
performance_metrics_list = ["precision", "f1_score", "recall"]

# from tensorflow.python.client import device_lib


# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]


# print(get_available_devices())

dataset_list = [
    pd.read_csv("data/CSE-CIC-IDS2018/02-14-2018.csv"),
    # pd.read_csv("data/CSE-CIC-IDS2018/02-23-2018.csv"),  # TODO: Add remaining days
]

preprocessed_dataset = preprocess_ids_2018(dataset_list)

dataset_target_label = "Label"
loss_metric_str = "binary_crossentropy"
optimizer = "adam"

profile(
    Sequential,
    "Nasser_LSTM",
    first_custom_layer_code=first_layer,
    repeated_custom_layer_code=repeated_layer,
    final_custom_layer_code=final_layer,
    layers=numbers_of_layers,
    units=numbers_of_units,
    epochs=numbers_of_epochs,
    features=numbers_of_features,
    profile_mode=profile_mode,
    statistical_samples=number_of_samples,
    batch_size=batch_size,
    sampling_rates=sampling_rates,
    performance_metrics=performance_metrics_list,
    dataset=preprocessed_dataset,
    target_label=dataset_target_label,
    loss_function=loss_metric_str,
    optimizer=optimizer,
)
