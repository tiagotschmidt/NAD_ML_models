import os
from manager import profile
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

from framework_parameters import (
    FrameworkParameterType,
    LifecycleSelected,
    MLMode,
    Platform,
    RangeParameter,
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

model = Sequential()


def repeated_custom_layer(model, number_of_units, input_shape):
    model.add(Dense(units=number_of_units, input_dim=input_shape, activation="relu"))


def final_custom_layer(model):
    model.add(Dense(units=1, activation="sigmoid"))


numbers_of_layers = RangeParameter(100, 100, 100, FrameworkParameterType.NumberOfLayers)
numbers_of_neurons = RangeParameter(10, 10, 40, FrameworkParameterType.NumberOfNeurons)
numbers_of_epochs = RangeParameter(1, 1, 1, FrameworkParameterType.NumberOfEpochs)
numbers_of_features = RangeParameter(
    70, 70, 10, FrameworkParameterType.NumberOfFeatures
)

# Define profile mode
profile_mode = MLMode(
    LifecycleSelected.TrainAndTest,
    train_platform=Platform.GPU,
    test_platform=Platform.CPU,
)

# Define other parameters
number_of_samples = 2
###batch_size = 40960
batch_size = 32
performance_metrics_list = ["accuracy", "f1_score", "precision", "recall"]
preprocessed_dataset = pd.read_csv("dataset/preprocessed_binary_dataset.csv")
dataset_target_label = "intrusion"
loss_metric_str = "binary_crossentropy"
optimizer = "adam"

profile(
    model,
    "Test_Model",
    repeated_custom_layer_code=repeated_custom_layer,
    final_custom_layer_code=final_custom_layer,
    numbers_of_layers=numbers_of_layers,
    numbers_of_neurons=numbers_of_neurons,
    numbers_of_epochs=numbers_of_epochs,
    numbers_of_features=numbers_of_features,
    profile_mode=profile_mode,
    number_of_samples=number_of_samples,
    batch_size=batch_size,
    performance_metrics_list=performance_metrics_list,
    preprocessed_dataset=preprocessed_dataset,
    dataset_target_label=dataset_target_label,
    loss_metric_str=loss_metric_str,
    optimizer=optimizer,
)
