import os
from unittest.mock import call, patch

import pytest
from eppnad.core.plotter import (
    _generate_plot_metadata,
    _group_results_for_plotting,
    _prepare_and_save_plots,
)
from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.execution_result import (
    ConfigAndResult,
    StatisticalValues,
    TimeAndEnergy,
    WrittenResults,
)
from eppnad.utils.framework_parameters import Lifecycle, Platform
from eppnad.utils.plot_list_collection import PlotPoint

# --- Test Fixtures for Mock Data ---


@pytest.fixture
def mock_config_1() -> ExecutionConfiguration:
    """A sample execution configuration."""
    return ExecutionConfiguration(
        layers=2,
        units=32,
        epochs=10,
        features=5,
        sampling_rate=1.0,
        platform=Platform.CPU,
        cycle=Lifecycle.TRAIN,
    )


@pytest.fixture
def mock_config_2() -> ExecutionConfiguration:
    """A second sample execution configuration, varying one hyperparameter."""
    return ExecutionConfiguration(
        layers=4,
        units=32,
        epochs=10,
        features=5,
        sampling_rate=1.0,
        platform=Platform.CPU,
        cycle=Lifecycle.TRAIN,
    )


@pytest.fixture
def sample_written_results():
    """
    Provides a sample results dictionary containing multiple metrics ('energy', 'accuracy')
    with multiple data points each, allowing for testing of grouping and plotting functions.
    The 'layers' hyperparameter is varied to enable comparative plotting.
    """
    return {
        "energy": [
            ConfigAndResult(
                configuration=ExecutionConfiguration(
                    layers=2,
                    units=32,
                    epochs=10,
                    features=5,
                    sampling_rate=1.0,
                    platform=Platform.CPU,
                    cycle=Lifecycle.TRAIN,
                ),
                result=TimeAndEnergy(average_seconds=150, average_joules=3500),
            ),
            ConfigAndResult(
                configuration=ExecutionConfiguration(
                    layers=4,
                    units=32,
                    epochs=10,
                    features=5,
                    sampling_rate=1.0,
                    platform=Platform.CPU,
                    cycle=Lifecycle.TRAIN,
                ),
                result=TimeAndEnergy(average_seconds=250, average_joules=4500),
            ),
        ],
        "accuracy": [
            ConfigAndResult(
                configuration=ExecutionConfiguration(
                    layers=2,
                    units=32,
                    epochs=10,
                    features=5,
                    sampling_rate=1.0,
                    platform=Platform.CPU,
                    cycle=Lifecycle.TRAIN,
                ),
                result=TimeAndEnergy(average_seconds=150, average_joules=0.85),
            ),
            ConfigAndResult(
                configuration=ExecutionConfiguration(
                    layers=4,
                    units=32,
                    epochs=10,
                    features=5,
                    sampling_rate=1.0,
                    platform=Platform.CPU,
                    cycle=Lifecycle.TRAIN,
                ),
                result=TimeAndEnergy(average_seconds=250, average_joules=0.92),
            ),
        ],
    }


# --- Unit Test Functions ---


def test_group_results_for_plotting(sample_written_results):
    """
    Tests that raw results are correctly grouped and transformed into PlotPoint collections.
    """
    # Action
    collections = _group_results_for_plotting(sample_written_results)

    # Assertions
    assert "energy" in collections
    assert "layers" in collections["energy"]  # Check for hyperparameter key

    # Check a specific group: accuracy vs layers, with units=32, epochs=10, etc. fixed.
    varying_hp_collection = collections["energy"]["layers"]
    assert len(varying_hp_collection) == 1  # Should be one group of plots

    template_config = list(varying_hp_collection.keys())[0]
    plot_points = list(varying_hp_collection.values())[0]

    assert template_config.units == 32
    assert template_config.layers == 0  # Placeholder value
    assert len(plot_points) == 2

    # Verify the transformation to PlotPoint is correct
    point_1 = plot_points[0]
    assert isinstance(point_1, PlotPoint)
    assert point_1.x_value == 2  # The actual value of the varying hyperparameter
    assert point_1.y_values.average_joules == 3500  # type: ignore


def test_generate_plot_metadata(mock_config_1):
    """
    Tests the dynamic generation of filenames and text box strings.
    """
    # Action
    filename, textbox_str = _generate_plot_metadata(
        template_config=mock_config_1,
        varying_hyperparameter="layers",
        number_of_samples=1000,
    )

    # Assertions
    assert "layers" not in filename
    assert "units_32" in filename
    assert "platform_CPU" in filename
    assert filename.endswith(".pdf")

    assert "Samples: 1000" in textbox_str
    assert "Layers" not in textbox_str
    assert "Units: 32" in textbox_str
    assert "Platform: CPU" in textbox_str


@patch("eppnad.core.plotter._create_box_and_line_chart")
def test_prepare_and_save_plots(mock_create_chart, sample_written_results, tmp_path):
    """
    Tests the main plotting loop, ensuring data is prepared correctly and
    the chart creation function is called with the right arguments.
    This test does NOT create actual plot files.
    """
    # Setup
    output_dir = str(tmp_path)
    collections = _group_results_for_plotting(sample_written_results)

    # Action
    _prepare_and_save_plots(collections, output_dir, 1000)

    # Assertions
    # With the new condition, we now correctly expect only one plot:
    # the one for accuracy vs. layers, as it's the only group with > 1 point.
    assert mock_create_chart.call_count == 1

    # Get the arguments from the single call to the mocked function
    call_args, call_kwargs = mock_create_chart.call_args

    # Check the keyword arguments passed to the chart creation function
    assert "save_path" in call_kwargs
    assert call_kwargs["metric_name"] == "accuracy"
    assert call_kwargs["varying_hp_name"] == "layers"

    # Verify that the boxplot statistics were prepared correctly
    boxplot_stats = call_kwargs["boxplot_stats"]
    # assert len(boxplot_stats) == 2
    # assert boxplot_stats[0]["med"] == 0.91
    # assert boxplot_stats[1]["med"] == 0.86

    # Verify that the energy data was correctly aligned
    energy_y_values = call_kwargs["energy_y_values"]
    assert len(energy_y_values) == 2
    assert energy_y_values == [3500, 4500]

    # Check that the directory structure was created
    expected_plot_path = os.path.join(
        output_dir, mock_create_chart.call_args[1]["save_path"]
    )
    assert os.path.exists(os.path.dirname(expected_plot_path))
