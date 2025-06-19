import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
import pandas as pd

from eppnad.core.manager import intermittent_profile, profile
from eppnad.utils.framework_parameters import (
    Lifecycle,
    PercentageRangeParameter,
    Platform,
    ProfileMode,
    RangeParameter,
)


def first_layer(model: keras.models.Model, number_of_units, input_shape):
    model.add(Dense(units=number_of_units, input_dim=input_shape, activation="relu"))


def repeated_layer(model: keras.models.Model, number_of_units, input_shape):
    model.add(Dense(units=number_of_units, input_dim=input_shape, activation="relu"))


def final_layer(model: keras.models.Model):
    model.add(Dense(units=1, activation="sigmoid"))


numbers_of_layers = RangeParameter([1])
numbers_of_units = RangeParameter([1, 2])
numbers_of_epochs = RangeParameter([10])
numbers_of_features = RangeParameter([13])
sampling_rates = PercentageRangeParameter([0.1])

# Define profile mode
profile_mode = ProfileMode(
    Lifecycle.TRAIN_AND_TEST,
    train_platform=Platform.GPU,
    test_platform=Platform.CPU,
)

# Define other parameters
number_of_samples = 10
batch_size = 8192 * 2
performance_metrics_list = ["precision", "f1_score", "recall"]
preprocessed_dataset = pd.read_csv("data/NSL-KDD/preprocessed_binary_dataset.csv")
dataset_target_label = "intrusion"
loss_metric_str = "binary_crossentropy"
optimizer = "adam"


intermittent_profile(
    Sequential,
    "MLP_neurons_intermittent2",
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
    execution_timeout_seconds=60,
)
