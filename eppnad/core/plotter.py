import datetime
import os
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.execution_result import (
    ResultsReader,
    StatisticalValues,
    TimeAndEnergy,
    WrittenResults,
)
from eppnad.utils.plot_list_collection import PlotCollection, PlotPoint


def plot(
    profile_execution_directory: str,
    number_of_samples: int,
    logger,
    expected_results: int,
) -> Dict[str, Dict[str, PlotCollection]]:
    """
    Reads experiment results, processes them into plottable collections,
    and generates and saves plots.

    This is the main entry point for the plotting functionality.

    Args:
        profile_execution_directory: The root directory where results are stored.
        number_of_samples: The number of samples used in the experiments,
                           for annotation on the plots.

    Returns:
        The fully processed, nested dictionary of results grouped for plotting.
    """
    read_results = ResultsReader(
        logger, profile_execution_directory
    ).read_results_from_directory()

    results_list = read_results["energy"]

    if len(results_list) != expected_results:
        return {}

    collections_for_plotting = _group_results_for_plotting(read_results)

    _prepare_and_save_plots(
        collections_for_plotting, profile_execution_directory, number_of_samples
    )
    return collections_for_plotting


def _group_results_for_plotting(
    results_by_metric: WrittenResults,
) -> Dict[str, Dict[str, PlotCollection]]:
    """
    Organizes raw experiment results into collections ready for plotting.

    This function processes a dictionary of experiment results and groups them
    to isolate the impact of a single, varying hyperparameter. It works for
    any metric, including performance (StatisticalValues) and energy
    (TimeAndEnergy).

    Args:
        results_by_metric: A dictionary mapping each metric's name to a list
                           of its results, as read from the CSV files.

    Returns:
        A nested dictionary structured for easy plotting, keyed by metric name,
        then by the name of the varying hyperparameter.
    """
    metric_collections: Dict[str, Dict[str, PlotCollection]] = {}
    hyperparameter_names = [f.name for f in fields(ExecutionConfiguration)]

    for metric_name, results_for_metric in results_by_metric.items():
        hyperparameter_collections: Dict[str, PlotCollection] = {
            name: {} for name in hyperparameter_names
        }

        for config_and_result in results_for_metric:
            original_config = config_and_result.configuration
            result_object = config_and_result.result

            for varying_hp_name in hyperparameter_names:
                field = ExecutionConfiguration.__dataclass_fields__[varying_hp_name]
                placeholder: Any
                if isinstance(field.type, type) and issubclass(field.type, Enum):
                    placeholder = list(field.type)[0]
                else:
                    placeholder = 0

                template_config = replace(
                    original_config, **{varying_hp_name: placeholder}
                )
                x_value = getattr(original_config, varying_hp_name)
                plot_point = PlotPoint(x_value=x_value, y_values=result_object)

                group = hyperparameter_collections[varying_hp_name].setdefault(
                    template_config, []
                )
                group.append(plot_point)

        metric_collections[metric_name] = hyperparameter_collections

    return metric_collections


def _generate_plot_metadata(
    template_config: ExecutionConfiguration,
    varying_hyperparameter: str,
    number_of_samples: int,
) -> Tuple[str, str]:
    """
    Dynamically generates a filename and a textbox string for a plot based on
    the fixed hyperparameters of an experiment.

    Args:
        template_config: The template configuration for the plot group.
        varying_hyperparameter: The name of the hyperparameter being varied.
        number_of_samples: The number of samples used in the experiment.

    Returns:
        A tuple containing the generated filename suffix and the textbox string.
    """
    filename_parts = []
    textbox_parts = [f"Samples: {number_of_samples}"]

    for field_name, field_value in template_config.__dict__.items():
        if field_name != varying_hyperparameter:
            value_str = (
                field_value.name if isinstance(field_value, Enum) else field_value
            )
            filename_parts.append(f"{field_name}_{value_str}")
            textbox_parts.append(f"{field_name.capitalize()}: {value_str}")

    filename = "_".join(filename_parts) + ".pdf"
    textbox_str = "\n".join(textbox_parts)

    return filename, textbox_str


def _prepare_and_save_plots(
    all_collections: Dict[str, Dict[str, PlotCollection]],
    output_directory: str,
    number_of_samples: int,
):
    """
    Iterates through processed collections to prepare data and save plots.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plots_root_dir = os.path.join(output_directory, f"plots_{timestamp}")
    os.makedirs(plots_root_dir, exist_ok=True)

    energy_collections = all_collections.get("energy", {})

    for metric_name, collections_by_hp in all_collections.items():
        if metric_name == "energy":
            continue

        for varying_hp_name, groups_by_template in collections_by_hp.items():
            for template_config, plot_points in groups_by_template.items():
                if len(plot_points) > 1:
                    plot_points.sort(key=lambda p: p.x_value)

                    x_values = [p.x_value for p in plot_points]

                    boxplot_stats = [
                        {
                            "med": p.y_values.median,
                            "q1": p.y_values.p25,
                            "q3": p.y_values.p75,
                            "whislo": p.y_values.min,
                            "whishi": p.y_values.max,
                            "label": p.x_value,
                        }
                        for p in plot_points
                        if isinstance(p.y_values, StatisticalValues)
                    ]

                    aligned_energy_joules = []
                    if energy_collections:
                        corresponding_energy_points = energy_collections.get(
                            varying_hp_name, {}
                        ).get(template_config, [])
                        energy_lookup = {
                            p.x_value: p.y_values for p in corresponding_energy_points
                        }

                        for x in x_values:
                            energy_result = energy_lookup.get(x)
                            if energy_result and isinstance(
                                energy_result, TimeAndEnergy
                            ):
                                aligned_energy_joules.append(
                                    energy_result.average_joules
                                )

                    chart_filename, textbox_str = _generate_plot_metadata(
                        template_config, varying_hp_name, number_of_samples
                    )

                    save_dir = os.path.join(
                        plots_root_dir, metric_name, varying_hp_name
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, chart_filename)

                    _create_box_and_line_chart(
                        save_path=save_path,
                        boxplot_stats=boxplot_stats,
                        energy_y_values=aligned_energy_joules,
                        metric_name=metric_name,
                        varying_hp_name=varying_hp_name,
                        textbox_str=textbox_str,
                    )


def _create_box_and_line_chart(
    save_path: str,
    boxplot_stats: List[Dict],
    energy_y_values: List[float],
    metric_name: str,
    varying_hp_name: str,
    textbox_str: str,
):
    """
    Generates and saves a single chart with a primary box plot and a
    secondary line plot.

    Args:
        save_path: The full path where the plot image will be saved.
        boxplot_stats: A list of statistics for each box plot.
        energy_y_values: The y-values (in Joules) for the energy line plot.
        metric_name: The name of the primary performance metric.
        varying_hp_name: The name of the varying hyperparameter.
        textbox_str: The formatted string with metadata for the text box.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    x_values = [stat["label"] for stat in boxplot_stats]
    positions = np.arange(len(x_values))

    ax1.set_xlabel(varying_hp_name.replace("_", " ").capitalize())
    ax1.set_ylabel(metric_name.replace("_", " ").capitalize())

    box_plot = ax1.bxp(
        boxplot_stats,
        positions=positions,
        showfliers=False,
        patch_artist=True,
        widths=0.2,
    )

    for patch in box_plot["boxes"]:
        patch.set_facecolor("lightblue")
    for median in box_plot["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    for i, stat in enumerate(boxplot_stats):
        ax1.text(
            positions[i],
            stat["med"],
            f'{stat["med"]:.2f}',
            va="center",
            ha="center",
            fontweight="bold",
            color="navy",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.8),
        )

    ax1.set_xticks(positions)
    ax1.set_xticklabels(x_values)
    ax1.tick_params(axis="x", rotation=30)

    ax2 = ax1.twinx()
    ax2.plot(positions, energy_y_values, "o-", color="tab:red", label="Energy")
    ax2.set_ylabel("Average Energy (Joules)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    min_whisker = min(s["whislo"] for s in boxplot_stats)
    max_whisker = max(s["whishi"] for s in boxplot_stats)
    y_margin = (max_whisker - min_whisker) * 0.1  # 10% margin
    ax1.set_ylim(min_whisker - y_margin, max_whisker + y_margin)

    if energy_y_values:
        min_energy, max_energy = min(energy_y_values), max(energy_y_values)
        energy_margin = (max_energy - min_energy) * 0.1
        ax2.set_ylim(min_energy - energy_margin, max_energy + energy_margin)

    title = f"{metric_name.capitalize()} vs. {varying_hp_name.capitalize()}"
    plt.title(title, fontsize=16, fontweight="bold")
    ax1.grid(True, linestyle="--", axis="y", which="both")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax1.text(
        0.02,
        0.98,
        textbox_str,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.savefig(save_path)
    plt.close(fig)
