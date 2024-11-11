import datetime
import os
from time import sleep
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot
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
        metric_dir = os.path.join(timestamped_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        for snapshot_result_list in plot_list_collection.test_epoch_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Epochs", metric_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_features_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Features", metric_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_layers_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Layers", metric_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_units_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Units", metric_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_epoch_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Epochs", metric_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_features_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Features", metric_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_layers_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Layers", metric_dir)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_units_lists:
            plot_metric_and_energy(snapshot_result_list, metric, "Units", metric_dir)  # type: ignore


def plot_metric_and_energy(
    config_and_results_list: tuple[ExecutionConfiguration, List[int], List[dict]],
    metric_name: str,
    hyperparameter: str,
    metric_dir: str,
):
    hyperparameter_dir = os.path.join(metric_dir, hyperparameter)
    os.makedirs(hyperparameter_dir, exist_ok=True)

    if config_and_results_list == None:
        return
    (config, hyperparameter_list, results_list) = config_and_results_list

    if len(results_list) == 0:
        return

    custom_name = ""
    textstr = ""

    if hyperparameter == "Epochs":
        custom_name = f"layers_{config.number_of_layers}_units_{config.number_of_units}_features_{config.number_of_features}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"Layers: {config.number_of_layers}\nUnits: {config.number_of_units}\nFeatures: {config.number_of_features}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"
    elif hyperparameter == "Features":
        custom_name = f"layers_{config.number_of_layers}_units_{config.number_of_units}_epochs_{config.number_of_epochs}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"Layers: {config.number_of_layers}\nUnits: {config.number_of_units}\nEpochs: {config.number_of_epochs}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"
    elif hyperparameter == "Layers":
        custom_name = f"units_{config.number_of_units}_epochs_{config.number_of_epochs}_features_{config.number_of_features}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"\nUnits: {config.number_of_units}\nEpochs: {config.number_of_epochs}\nFeatures: {config.number_of_features}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"
    elif hyperparameter == "Units":
        custom_name = f"layers_{config.number_of_layers}_epochs_{config.number_of_epochs}_features_{config.number_of_features}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"Layers: {config.number_of_layers}\nEpochs: {config.number_of_epochs}\nFeatures: {config.number_of_features}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"

    x_labels = hyperparameter_list
    metric_values = []
    energy_values = []
    upperbound_metric_error = []
    lowerbound_metric_error = []

    for result in results_list:
        metric_values.append(result[metric_name]["mean"])
        energy_values.append(result["average_energy_consumption_joules"])
        upperbound_metric_error.append(
            result[metric_name]["mean"] + result[metric_name]["error"]
        )
        lowerbound_metric_error.append(
            result[metric_name]["mean"] - result[metric_name]["error"]
        )

    plt.figure(figsize=(8, 6))

    fig, ax1 = plt.subplots()

    ax1.errorbar(
        x_labels,
        metric_values,
        yerr=[lowerbound_metric_error, upperbound_metric_error],
        fmt="o-",
        label=f"{metric_name.capitalize()}",
    )

    ax1.set_ylabel(metric_name.capitalize(), labelpad=5)
    ax1.set_xlabel(hyperparameter.capitalize(), labelpad=5)

    plt.xticks(x_labels)
    plt.xlim(min(x_labels), max(x_labels))

    ax2 = ax1.twinx()

    color = "tab:red"
    ax2.plot(
        x_labels, energy_values, color=color, label="Average Energy Consumption (J)"
    )
    ax2.set_ylabel("Average Energy Consumption (J)", color=color, labelpad=5)
    ax2.tick_params(axis="y", labelcolor=color)

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.title(f"{metric_name.capitalize()} and Energy Consumption")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(hyperparameter_dir, custom_name)
    plt.savefig(filename)
    matplotlib.pyplot.close()
    plt.close(fig)
    plt.close()
    plt.close("all")
    plt.clf()


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
        for saved_configuration, hyperparameter_list, result_list in epoch_lists:
            if is_same_epoch_snapshot(configuration, saved_configuration):
                has_found_epoch = True
                hyperparameter_list.append(configuration.number_of_epochs)
                result_list.append(results)
        if not has_found_epoch:
            epoch_lists.append(
                (configuration, [configuration.number_of_epochs], [results])
            )

        has_found_features = False
        for saved_configuration, hyperparameter_list, result_list in features_lists:
            if is_same_features_snapshot(configuration, saved_configuration):
                has_found_features = True
                hyperparameter_list.append(configuration.number_of_features)
                result_list.append(results)
        if not has_found_features:
            features_lists.append(
                (configuration, [configuration.number_of_features], [results])
            )

        has_found_units = False
        for saved_configuration, hyperparameter_list, result_list in units_lists:
            if is_same_units_snapshot(configuration, saved_configuration):
                has_found_units = True
                hyperparameter_list.append(configuration.number_of_units)
                result_list.append(results)
        if not has_found_units:
            units_lists.append(
                (configuration, [configuration.number_of_units], [results])
            )

        has_found_layers = False
        for saved_configuration, hyperparameter_list, result_list in layer_lists:
            if is_same_layers_snapshot(configuration, saved_configuration):
                has_found_layers = True
                hyperparameter_list.append(configuration.number_of_layers)
                result_list.append(results)
        if not has_found_layers:
            layer_lists.append(
                (configuration, [configuration.number_of_layers], [results])
            )


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
