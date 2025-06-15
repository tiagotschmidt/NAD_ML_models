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
def sample_written_results(mock_config_1, mock_config_2) -> WrittenResults:
    """Creates a sample WrittenResults dictionary with performance and energy data."""
    # Performance results for two different layer counts
    accuracy_result_1 = ConfigAndResult(
        configuration=mock_config_1,
        result=StatisticalValues(
            mean=0.9, std_dev=0.05, median=0.91, min=0.8, max=0.95, p25=0.88, p75=0.92
        ),
    )
    accuracy_result_2 = ConfigAndResult(
        configuration=mock_config_2,
        result=StatisticalValues(
            mean=0.85, std_dev=0.06, median=0.86, min=0.75, max=0.90, p25=0.82, p75=0.88
        ),
    )
    # Corresponding energy results
    energy_result_1 = ConfigAndResult(
        configuration=mock_config_1,
        result=TimeAndEnergy(average_seconds=100, average_joules=2500),
    )
    energy_result_2 = ConfigAndResult(
        configuration=mock_config_2,
        result=TimeAndEnergy(average_seconds=150, average_joules=3500),
    )

    return {
        "accuracy": [accuracy_result_1, accuracy_result_2],
        "energy": [energy_result_1, energy_result_2],
    }


# --- Unit Test Functions ---


def test_group_results_for_plotting(sample_written_results):
    """
    Tests that raw results are correctly grouped and transformed into PlotPoint collections.
    """
    # Action
    collections = _group_results_for_plotting(sample_written_results)

    # Assertions
    assert "accuracy" in collections
    assert "energy" in collections
    assert "layers" in collections["accuracy"]  # Check for hyperparameter key

    # Check a specific group: accuracy vs layers, with units=32, epochs=10, etc. fixed.
    varying_hp_collection = collections["accuracy"]["layers"]
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
    assert point_1.y_values.mean == 0.9  # type: ignore


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
    assert len(call_kwargs["x_values"]) == 2
    assert call_kwargs["x_values"] == [2, 4]

    # Verify that the boxplot statistics were prepared correctly
    boxplot_stats = call_kwargs["boxplot_stats"]
    assert len(boxplot_stats) == 2
    assert boxplot_stats[0]["med"] == 0.91
    assert boxplot_stats[1]["med"] == 0.86

    # Verify that the energy data was correctly aligned
    energy_y_values = call_kwargs["energy_y_values"]
    assert len(energy_y_values) == 2
    assert energy_y_values == [2500, 3500]

    # Check that the directory structure was created
    expected_plot_path = os.path.join(
        output_dir, mock_create_chart.call_args[1]["save_path"]
    )
    assert os.path.exists(os.path.dirname(expected_plot_path))
