import os
import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
import pandas as pd
from eppnad.framework_parameters import (
    FrameworkParameterType,
    LifecycleSelected,
    MLMode,
    Platform,
    RangeMode,
    RangeParameter,
)
from eppnad.manager import profile

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


def repeated_custom_layer(model: keras.models.Model, number_of_units, input_shape):
    model.add(Dense(units=number_of_units, input_dim=input_shape, activation="relu"))


def final_custom_layer(model: keras.models.Model):
    model.add(Dense(units=1, activation="sigmoid"))


numbers_of_layers = RangeParameter(
    10, 320, 2, FrameworkParameterType.NumberOfLayers, RangeMode.Multiplicative
)
numbers_of_neurons = RangeParameter(
    50, 50, 2, FrameworkParameterType.NumberOfNeurons, RangeMode.Multiplicative
)
numbers_of_epochs = RangeParameter(
    40, 40, 40, FrameworkParameterType.NumberOfEpochs, RangeMode.Additive
)
numbers_of_features = RangeParameter(
    93, 93, 15, FrameworkParameterType.NumberOfFeatures, RangeMode.Additive
)

# Define profile mode
profile_mode = MLMode(
    LifecycleSelected.TrainAndTest,
    train_platform=Platform.GPU,
    test_platform=Platform.CPU,
)

# Define other parameters
number_of_samples = 30
batch_size = 32768
performance_metrics_list = ["accuracy", "f1_score", "recall"]
preprocessed_dataset = pd.read_csv("dataset/preprocessed_binary_dataset.csv")
dataset_target_label = "intrusion"
loss_metric_str = "binary_crossentropy"
optimizer = "adam"

profile(
    Sequential,
    "MLP_layers",
    repeated_custom_layer_code=repeated_custom_layer,
    final_custom_layer_code=final_custom_layer,
    numbers_of_layers=numbers_of_layers,
    numbers_of_neurons=numbers_of_neurons,
    numbers_of_epochs=numbers_of_epochs,
    numbers_of_features=numbers_of_features,
    profile_mode=profile_mode,
    number_of_samples=number_of_samples,
    batch_size=batch_size,
    sampling_rate=1,
    performance_metrics_list=performance_metrics_list,
    preprocessed_dataset=preprocessed_dataset,
    dataset_target_label=dataset_target_label,
    loss_metric_str=loss_metric_str,
    optimizer=optimizer,
)
