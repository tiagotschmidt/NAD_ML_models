import datetime
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from framework_parameters import ExecutionConfiguration, Lifecycle, PlotListCollection


def extract_plot_list_collection(
    results_list: List[tuple[ExecutionConfiguration, dict]]
) -> PlotListCollection:
    ### MODO TREINO
    ### EP0CH

    train_epoch_lists = []
    train_features_lists = []
    train_units_lists = []
    train_layer_lists = []

    test_epoch_lists = []
    test_features_lists = []
    test_units_lists = []
    test_layer_lists = []

    for configuration, results in results_list:
        current_lifecycle = Lifecycle.Train
        separate_hyperparameters_lists(
            train_epoch_lists,
            train_features_lists,
            train_units_lists,
            train_layer_lists,
            configuration,
            results,
            current_lifecycle,
        )

        current_lifecycle = Lifecycle.Test
        separate_hyperparameters_lists(
            test_epoch_lists,
            test_features_lists,
            test_units_lists,
            test_layer_lists,
            configuration,
            results,
            current_lifecycle,
        )

    return PlotListCollection(
        test_epoch_lists,
        test_units_lists,
        test_features_lists,
        test_layer_lists,
        train_epoch_lists,
        train_units_lists,
        train_features_lists,
        train_layer_lists,
    )


def plot(plot_list_collection: PlotListCollection, metrics_str_list: List[str]):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # type: ignore

    plot_results_dir = "plot_results"
    os.makedirs(plot_results_dir, exist_ok=True)

    timestamped_dir = os.path.join(plot_results_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)

    for metric in metrics_str_list:
        for snapshot_result_list in plot_list_collection.test_epoch_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Epochs", timestamped_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_features_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Features", timestamped_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_layers_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Layers", timestamped_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_units_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Units", timestamped_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_epoch_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Epocs", timestamped_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_features_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Features", timestamped_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_layers_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Layers", timestamped_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_units_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Units", timestamped_dir)  # type: ignore


def plot_metric_and_energy(
    config_and_results_list: tuple[ExecutionConfiguration, List[dict]],
    metric_name: str,
    hyperparameter: str,
    timestamped_dir: str,
):
    (config, results_list) = config_and_results_list
    x_values = range(len(results_list))
    metric_values = []
    energy_values = []
    metric_errors = []
    energy_errors = []

    for result in results_list:
        metric_values.append(result[metric_name]["mean"])
        energy_values.append(result["average_energy_consumption_joules"])
        metric_errors.append(result[metric_name]["error"])

    print("Testes")
    print(metric_values)
    print(energy_values)
    print(metric_errors)

    # Create the plot
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        x_values, metric_values, yerr=metric_errors, fmt="o-", label=f"{metric_name}"
    )

    plt.xlabel(hyperparameter)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} and Energy Consumption")
    plt.legend()
    plt.grid(True)

    custom_name = f"{metric_name}_layers_{config.number_of_layers}_units_{config.number_of_units}_epochs_{config.number_of_epochs}_features_{config.number_of_features}_{config.platform.value}_{config.cycle.value}.png"
    filename = os.path.join(timestamped_dir, custom_name + ".png")
    plt.savefig(filename)
    plt.close()


def separate_hyperparameters_lists(
    epoch_lists,
    features_lists,
    units_lists,
    layer_lists,
    configuration,
    results,
    current_lifecycle,
):
    if configuration.cycle.value == current_lifecycle.value:
        has_found_epoch = False
        for saved_configuration, result_list in epoch_lists:
            if is_same_epoch_snapshot(configuration, saved_configuration):
                has_found_epoch = True
                result_list.append(results)
        if not has_found_epoch:
            epoch_lists.append((configuration, [results]))

        has_found_features = False
        for saved_configuration, result_list in features_lists:
            if is_same_features_snapshot(configuration, saved_configuration):
                has_found_features = True
                result_list.append(results)
        if not has_found_features:
            features_lists.append((configuration, [results]))

        has_found_units = False
        for saved_configuration, result_list in units_lists:
            if is_same_units_snapshot(configuration, saved_configuration):
                has_found_units = True
                result_list.append(results)
        if not has_found_units:
            units_lists.append((configuration, [results]))

        has_found_layers = False
        for saved_configuration, result_list in layer_lists:
            if is_same_layers_snapshot(configuration, saved_configuration):
                has_found_layers = True
                result_list.append(results)
        if not has_found_layers:
            layer_lists.append((configuration, [results]))


def is_same_epoch_snapshot(
    configuration: ExecutionConfiguration, saved_configuration: ExecutionConfiguration
):
    return (
        configuration.number_of_features == saved_configuration.number_of_features
        and configuration.number_of_layers == saved_configuration.number_of_layers
        and configuration.number_of_units == saved_configuration.number_of_units
    )


def is_same_features_snapshot(
    configuration: ExecutionConfiguration, saved_configuration: ExecutionConfiguration
):
    return (
        configuration.number_of_epochs == saved_configuration.number_of_epochs
        and configuration.number_of_layers == saved_configuration.number_of_layers
        and configuration.number_of_units == saved_configuration.number_of_units
    )


def is_same_units_snapshot(
    configuration: ExecutionConfiguration, saved_configuration: ExecutionConfiguration
):
    return (
        configuration.number_of_epochs == saved_configuration.number_of_epochs
        and configuration.number_of_layers == saved_configuration.number_of_layers
        and configuration.number_of_features == saved_configuration.number_of_features
    )


def is_same_layers_snapshot(
    configuration: ExecutionConfiguration, saved_configuration: ExecutionConfiguration
):
    return (
        configuration.number_of_epochs == saved_configuration.number_of_epochs
        and configuration.number_of_units == saved_configuration.number_of_units
        and configuration.number_of_features == saved_configuration.number_of_features
    )
