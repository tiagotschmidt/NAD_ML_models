# eppnad/tests/test_results_writer.py

import os
import csv
import pytest
from dataclasses import dataclass

# Mock the real classes with simple test versions
from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.execution_result import (
    ResultsReader,
    ResultsWriter,
    StatisticalValues,
    TimeAndEnergy,
)
from eppnad.utils.framework_parameters import Platform, Lifecycle

# Import the classes to be tested and the real data structures for integration tests


@dataclass
class MockStatisticalValues:
    mean: float
    median: float


@dataclass
class MockTimeAndEnergy:
    average_seconds: float
    average_joules: float


# --- Test Fixtures ---
@pytest.fixture
def temp_output_dir(tmp_path):
    """Provides a temporary output directory path."""
    return str(tmp_path)


@pytest.fixture
def mock_logger(mocker):
    """Creates a mock logger."""
    return mocker.MagicMock()


@pytest.fixture
def results_writer(temp_output_dir, mock_logger):
    """Provides a ResultsWriter instance initialized with a temp directory."""
    return ResultsWriter(logger=mock_logger, output_dir=temp_output_dir)


@pytest.fixture
def results_reader(temp_output_dir, mock_logger):
    """Provides a ResultsReader instance initialized with a temp directory."""
    return ResultsReader(logger=mock_logger, output_dir=temp_output_dir)


# --- Test Cases for ResultsWriter ---
class TestResultsWriter:
    """Tests the functionality of the ResultsWriter class."""

    def test_initialization(self, results_writer, temp_output_dir):
        """Tests that the results directory path is correctly constructed."""
        expected_path = os.path.join(temp_output_dir, "results")
        assert results_writer.results_dir == expected_path

    def test_append_first_result_creates_file_with_header(self, results_writer):
        """
        Tests that appending the first result creates a new CSV file with a
        correct header and data row.
        """
        config = ExecutionConfiguration(
            layers=2,
            units=64,
            epochs=1,
            sampling_rate=1,
            features=1,
            platform=Platform.CPU,
            cycle=Lifecycle.TRAIN,
        )
        metrics = {"accuracy": MockStatisticalValues(mean=0.95, median=0.96)}
        results_writer.append_execution_result((config, metrics))

        expected_file = os.path.join(results_writer.results_dir, "accuracy.csv")
        assert os.path.isfile(expected_file)
        with open(expected_file, "r") as f:
            header = next(csv.reader(f))
        assert header == [
            "layers",
            "units",
            "epochs",
            "features",
            "sampling_rate",
            "platform",
            "cycle",
            "mean",
            "median",
        ]

    def test_append_energy_result_creates_file(self, results_writer):
        """
        Tests that appending an energy result creates 'energy.csv' with
        the correct header and data row.
        """
        config = ExecutionConfiguration(
            layers=4,
            units=128,
            epochs=2,
            sampling_rate=2,
            features=2,
            platform=Platform.GPU,
            cycle=Lifecycle.TEST,
        )
        energy_data = MockTimeAndEnergy(average_seconds=15.5, average_joules=250.75)
        results_writer.append_energy_result((config, energy_data))

        expected_file = os.path.join(results_writer.results_dir, "energy.csv")
        assert os.path.isfile(expected_file)

    def test_append_second_result_adds_row_without_header(self, results_writer):
        """
        Tests that appending a second result adds a new row to the existing
        CSV file without adding a second header.
        """
        config1 = ExecutionConfiguration(
            layers=2,
            units=64,
            epochs=2,
            sampling_rate=2,
            features=2,
            platform=Platform.CPU,
            cycle=Lifecycle.TRAIN,
        )
        metrics1 = {"loss": MockStatisticalValues(mean=0.1, median=0.09)}
        results_writer.append_execution_result((config1, metrics1))

        config2 = ExecutionConfiguration(
            layers=4,
            units=128,
            epochs=3,
            sampling_rate=3,
            features=3,
            platform=Platform.GPU,
            cycle=Lifecycle.TEST,
        )
        metrics2 = {"loss": MockStatisticalValues(mean=0.05, median=0.04)}
        results_writer.append_execution_result((config2, metrics2))

        expected_file = os.path.join(results_writer.results_dir, "loss.csv")
        with open(expected_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_multiple_metrics_create_multiple_files(self, results_writer):
        """
        Tests that a single result with multiple metrics creates a separate
        CSV file for each metric.
        """
        config = ExecutionConfiguration(
            layers=1,
            units=1,
            epochs=4,
            sampling_rate=4,
            features=4,
            platform=Platform.CPU,
            cycle=Lifecycle.TRAIN,
        )
        metrics = {
            "loss": MockStatisticalValues(mean=0.5, median=0.5),
            "accuracy": MockStatisticalValues(mean=0.8, median=0.8),
        }
        results_writer.append_execution_result((config, metrics))
        assert os.path.isfile(os.path.join(results_writer.results_dir, "loss.csv"))
        assert os.path.isfile(os.path.join(results_writer.results_dir, "accuracy.csv"))


# --- New Test Class for Writer-Reader Integration ---
class TestWriterReaderIntegration:
    """
    Tests the end-to-end functionality of writing results with ResultsWriter
    and reading them back with ResultsReader.
    """

    def test_write_and_read_single_performance_result(
        self, results_writer, results_reader
    ):
        """
        Tests that a single performance result can be written and then read back accurately.
        """
        config = ExecutionConfiguration(
            layers=2,
            units=64,
            epochs=10,
            features=12,
            sampling_rate=1.0,
            platform=Platform.CPU,
            cycle=Lifecycle.TRAIN,
        )
        stats = StatisticalValues(
            mean=0.95, std_dev=0.01, median=0.96, min=0.92, max=0.98, p25=0.94, p75=0.97
        )
        results_writer.append_execution_result((config, {"accuracy": stats}))
        read_results = results_reader.read_results_from_directory()

        assert "accuracy" in read_results
        assert len(read_results["accuracy"]) == 1
        read_config_and_result = read_results["accuracy"][0]
        # The reader should reconstruct the configuration and result objects exactly
        assert read_config_and_result.configuration == config
        assert read_config_and_result.result == stats

    def test_write_and_read_energy_result(self, results_writer, results_reader):
        """
        Tests that an energy result can be written and read back accurately.
        """
        config = ExecutionConfiguration(
            layers=4,
            units=128,
            epochs=20,
            features=15,
            sampling_rate=2.0,
            platform=Platform.GPU,
            cycle=Lifecycle.TEST,
        )
        energy_data = TimeAndEnergy(average_seconds=120.5, average_joules=3500.75)
        results_writer.append_energy_result((config, energy_data))
        read_results = results_reader.read_results_from_directory()

        assert "energy" in read_results
        assert len(read_results["energy"]) == 1
        read_config_and_result = read_results["energy"][0]
        assert read_config_and_result.configuration == config
        assert read_config_and_result.result == energy_data

    def test_write_and_read_multiple_runs_and_metrics(
        self, results_writer, results_reader
    ):
        """
        Tests writing multiple experiment runs and reading them all back correctly.
        """
        config1 = ExecutionConfiguration(
            layers=2,
            units=32,
            epochs=5,
            features=10,
            sampling_rate=1.0,
            platform=Platform.CPU,
            cycle=Lifecycle.TRAIN,
        )
        metrics1 = {
            "loss": StatisticalValues(
                mean=0.5, std_dev=0.1, median=0.48, min=0.3, max=0.7, p25=0.4, p75=0.6
            ),
            "accuracy": StatisticalValues(
                mean=0.8,
                std_dev=0.05,
                median=0.81,
                min=0.7,
                max=0.9,
                p25=0.78,
                p75=0.82,
            ),
        }
        config2 = ExecutionConfiguration(
            layers=4,
            units=64,
            epochs=5,
            features=10,
            sampling_rate=1.0,
            platform=Platform.GPU,
            cycle=Lifecycle.TEST,
        )
        metrics2 = {
            "loss": StatisticalValues(
                mean=0.3,
                std_dev=0.08,
                median=0.29,
                min=0.2,
                max=0.4,
                p25=0.25,
                p75=0.35,
            ),
            "accuracy": StatisticalValues(
                mean=0.9,
                std_dev=0.03,
                median=0.91,
                min=0.85,
                max=0.95,
                p25=0.88,
                p75=0.92,
            ),
        }

        results_writer.append_execution_result((config1, metrics1))
        results_writer.append_execution_result((config2, metrics2))
        read_results = results_reader.read_results_from_directory()

        assert set(read_results.keys()) == {"loss", "accuracy"}
        assert len(read_results["loss"]) == 2
        assert len(read_results["accuracy"]) == 2

        # Find and verify a specific result, since read order is not guaranteed
        accuracy_result_2 = next(
            r for r in read_results["accuracy"] if r.configuration.layers == 4
        )
        assert accuracy_result_2.configuration == config2
        assert accuracy_result_2.result == metrics2["accuracy"]

    def test_read_from_empty_directory(self, results_reader):
        """
        Tests that reading from an empty directory returns an empty dict.
        """
        read_results = results_reader.read_results_from_directory()
        assert read_results == {}
