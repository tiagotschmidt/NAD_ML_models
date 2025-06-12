import datetime
import os
from time import sleep
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot
import numpy as np

from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.framework_parameters import Lifecycle
from eppnad.utils.plot_list_collection import PlotListCollection


def plot(
    results_list: List[tuple[ExecutionConfiguration, dict]],
    metrics_str_list: List[str],
    number_of_samples: int,
) -> PlotListCollection:
    plot_list_collection = _get_plot_list_collection(results_list)
    _create_output_directories_and_plot(
        plot_list_collection, metrics_str_list, number_of_samples
    )
    return plot_list_collection


def _get_plot_list_collection(
    results_list: List[tuple[ExecutionConfiguration, dict]],
) -> PlotListCollection:
    train_epoch_lists = []
    train_features_lists = []
    train_units_lists = []
    train_layer_lists = []

    test_epoch_lists = []
    test_features_lists = []
    test_units_lists = []
    test_layer_lists = []

    for configuration, results in results_list:
        current_Lifecycle = Lifecycle.TRAIN
        _populate_lists_with_results(
            train_epoch_lists,
            train_features_lists,
            train_units_lists,
            train_layer_lists,
            configuration,
            results,
            current_Lifecycle,
        )

        current_Lifecycle = Lifecycle.TEST
        _populate_lists_with_results(
            test_epoch_lists,
            test_features_lists,
            test_units_lists,
            test_layer_lists,
            configuration,
            results,
            current_Lifecycle,
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


def _create_output_directories_and_plot(
    plot_list_collection: PlotListCollection,
    metrics_str_list: List[str],
    number_of_samples: int,
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # type: ignore

    plot_results_dir = "plot_results"
    os.makedirs(plot_results_dir, exist_ok=True)

    timestamped_dir = os.path.join(plot_results_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)

    for metric in metrics_str_list:
        metric_dir = os.path.join(timestamped_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        for snapshot_result_list in plot_list_collection.test_epoch_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Epochs", metric_dir, number_of_samples)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_features_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Features", metric_dir, number_of_samples)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_layers_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Layers", metric_dir, number_of_samples)  # type: ignore
        for snapshot_result_list in plot_list_collection.test_units_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Units", metric_dir, number_of_samples)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_epoch_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Epochs", metric_dir, number_of_samples)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_features_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Features", metric_dir, number_of_samples)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_layers_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Layers", metric_dir, number_of_samples)  # type: ignore
        for snapshot_result_list in plot_list_collection.train_units_lists:
            _generate_metric_and_energy_line_plot(snapshot_result_list, metric, "Units", metric_dir, number_of_samples)  # type: ignore


def _generate_metric_and_energy_line_plot(
    config_and_results_list: tuple[ExecutionConfiguration, List[int], List[dict]],
    metric_name: str,
    hyperparameter_name: str,
    base_metric_dir: str,
    number_of_samples: int,
):
    hyperparameter_dir = os.path.join(base_metric_dir, hyperparameter_name)
    os.makedirs(hyperparameter_dir, exist_ok=True)

    if config_and_results_list == None:
        return
    (config, hyperparameter_list, results_list) = config_and_results_list

    if len(results_list) == 0:
        return

    custom_name = ""
    textstr = ""

    if hyperparameter_name == "Epochs":
        custom_name = f"layers_{config.layers}_units_{config.units}_features_{config.features}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"Samples: {number_of_samples}\nLayers: {config.layers}\nUnits: {config.units}\nFeatures: {config.features}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"
    elif hyperparameter_name == "Features":
        custom_name = f"layers_{config.layers}_units_{config.units}_epochs_{config.epochs}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"Samples: {number_of_samples}\nLayers: {config.layers}\nUnits: {config.units}\nEpochs: {config.epochs}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"
    elif hyperparameter_name == "Layers":
        custom_name = f"units_{config.units}_epochs_{config.epochs}_features_{config.features}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"Samples: {number_of_samples}\nUnits: {config.units}\nEpochs: {config.epochs}\nFeatures: {config.features}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"
    elif hyperparameter_name == "Units":
        custom_name = f"layers_{config.layers}_epochs_{config.epochs}_features_{config.features}_{config.platform.name}_{config.cycle.name}.pdf"
        textstr = f"Samples: {number_of_samples}\nLayers: {config.layers}\nEpochs: {config.epochs}\nFeatures: {config.features}\nPlatform: {config.platform.name}\nCycle: {config.cycle.name}"

    x_labels = hyperparameter_list
    metric_values = []
    energy_values = []
    metric_errors = []

    for result in results_list:
        metric_values.append(result[metric_name]["mean"])
        energy_values.append(result["average_energy_consumption_joules"])
        metric_errors.append(result[metric_name]["error"])

    plt.figure(figsize=(8, 6))

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    lims = (0, x_labels[-1] * 1.1)

    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment="right")

    ax1.set_ylabel(metric_name.capitalize(), labelpad=5)
    ax1.set_xlabel(hyperparameter_name.capitalize(), labelpad=5)
    ax1.set_xlim(lims)

    plt.xticks(x_labels)
    plt.xlim(min(x_labels), max(x_labels))

    ax2 = ax1.twinx()

    color = "tab:red"
    energy_consumption_label = ""
    if Lifecycle.TRAIN in config.cycle:
        energy_consumption_label = "Training Epoch Average Energy Consumption (J)"
    else:
        energy_consumption_label = "Test Evaluation Average Energy Consumption (J)"

    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment="right")

    ax2.set_xlim(lims)
    ax2.set_ylabel(energy_consumption_label, color=color, labelpad=5)
    ax2.tick_params(axis="y", labelcolor=color)

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.01,
        0.01,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
    )

    ax1.errorbar(
        x=x_labels,
        y=metric_values,
        yerr=metric_errors,
        fmt="o-",
        label=f"{metric_name.capitalize()}",
        capsize=5,
    )
    ax1.legend(loc=1)

    ax2.plot(
        x_labels,
        energy_values,
        color=color,
        label="Energy Consumption (J)",
    )
    ax2.legend(loc=4)

    plt.title(f"{metric_name.capitalize()} and Energy Consumption")
    plt.grid(True)

    filename = os.path.join(hyperparameter_dir, custom_name)
    plt.savefig(filename)
    matplotlib.pyplot.close()
    plt.close(fig)
    plt.close()
    plt.close("all")
    plt.clf()


def _populate_lists_with_results(
    epoch_lists,
    features_lists,
    units_lists,
    layer_lists,
    configuration,
    results,
    current_Lifecycle,
):
    if configuration.cycle.value == current_Lifecycle.value:
        has_found_epoch = False
        for saved_configuration, hyperparameter_list, result_list in epoch_lists:
            if _is_same_epoch_snapshot(configuration, saved_configuration):
                has_found_epoch = True
                hyperparameter_list.append(configuration.number_of_epochs)
                result_list.append(results)
        if not has_found_epoch:
            epoch_lists.append(
                (configuration, [configuration.number_of_epochs], [results])
            )

        has_found_features = False
        for saved_configuration, hyperparameter_list, result_list in features_lists:
            if _is_same_features_snapshot(configuration, saved_configuration):
                has_found_features = True
                hyperparameter_list.append(configuration.number_of_features)
                result_list.append(results)
        if not has_found_features:
            features_lists.append(
                (configuration, [configuration.number_of_features], [results])
            )

        has_found_units = False
        for saved_configuration, hyperparameter_list, result_list in units_lists:
            if _is_same_units_snapshot(configuration, saved_configuration):
                has_found_units = True
                hyperparameter_list.append(configuration.number_of_units)
                result_list.append(results)
        if not has_found_units:
            units_lists.append(
                (configuration, [configuration.number_of_units], [results])
            )

        has_found_layers = False
        for saved_configuration, hyperparameter_list, result_list in layer_lists:
            if _is_same_layers_snapshot(configuration, saved_configuration):
                has_found_layers = True
                hyperparameter_list.append(configuration.layers)
                result_list.append(results)
        if not has_found_layers:
            layer_lists.append((configuration, [configuration.layers], [results]))


def _is_same_epoch_snapshot(
    configuration: ExecutionConfiguration,
    saved_configuration: ExecutionConfiguration,
):
    return (
        configuration.features == saved_configuration.features
        and configuration.layers == saved_configuration.layers
        and configuration.units == saved_configuration.units
    )


def _is_same_features_snapshot(
    configuration: ExecutionConfiguration,
    saved_configuration: ExecutionConfiguration,
):
    return (
        configuration.epochs == saved_configuration.epochs
        and configuration.layers == saved_configuration.layers
        and configuration.units == saved_configuration.units
    )


def _is_same_units_snapshot(
    configuration: ExecutionConfiguration,
    saved_configuration: ExecutionConfiguration,
):
    return (
        configuration.epochs == saved_configuration.epochs
        and configuration.layers == saved_configuration.layers
        and configuration.features == saved_configuration.features
    )


def _is_same_layers_snapshot(
    configuration: ExecutionConfiguration,
    saved_configuration: ExecutionConfiguration,
):
    return (
        configuration.epochs == saved_configuration.epochs
        and configuration.units == saved_configuration.units
        and configuration.features == saved_configuration.features
    )
