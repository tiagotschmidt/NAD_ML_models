# eppnad/tests/test_results_writer.py

import os
import csv
import pytest
from dataclasses import dataclass, field

# Mock the real classes with simple test versions
from eppnad.utils.framework_parameters import Platform, Lifecycle


# Use a simple dataclass for testing instead of the real one
@dataclass
class MockExecutionConfiguration:
    layers: int
    units: int
    epochs: int
    features: int
    sampling_rate: float
    platform: Platform
    cycle: Lifecycle


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
def results_writer(temp_output_dir):
    """Provides a ResultsWriter instance initialized with a temp directory."""
    from eppnad.utils.execution_result import ResultsWriter

    return ResultsWriter(output_dir=temp_output_dir)


# --- Test Cases ---
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
        # 1. Setup Data
        config = MockExecutionConfiguration(
            layers=2,
            units=64,
            epochs=1,
            sampling_rate=1,
            features=1,
            platform=Platform.CPU,
            cycle=Lifecycle.TRAIN,
        )
        metrics = {"accuracy": MockStatisticalValues(mean=0.95, median=0.96)}
        result = (config, metrics)

        # 2. Action
        results_writer.append_execution_result(result)

        # 3. Assertions
        expected_file = os.path.join(results_writer.results_dir, "accuracy.csv")
        assert os.path.isfile(expected_file), "CSV file was not created."

        with open(expected_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)

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
        # Note: Enum values are written as their names ("CPU", "Train")
        assert first_row == ["2", "64", "1", "1", "1", "CPU", "TRAIN", "0.95", "0.96"]

    def test_append_energy_result_creates_file(self, results_writer):
        """
        Tests that appending an energy result creates 'energy.csv' with
        the correct header and data row.
        """
        config = MockExecutionConfiguration(
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
        assert os.path.isfile(expected_file), "energy.csv file was not created."

        with open(expected_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)

        assert header == [
            "layers",
            "units",
            "epochs",
            "features",
            "sampling_rate",
            "platform",
            "cycle",
            "average_seconds",
            "average_joules",
        ]
        assert first_row == ["4", "128", "2", "2", "2", "GPU", "TEST", "15.5", "250.75"]

    def test_append_second_result_adds_row_without_header(self, results_writer):
        """
        Tests that appending a second result adds a new row to the existing
        CSV file without adding a second header.
        """
        # 1. First result
        config1 = MockExecutionConfiguration(
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

        # 2. Second result
        config2 = MockExecutionConfiguration(
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

        # 3. Assertions
        expected_file = os.path.join(results_writer.results_dir, "loss.csv")
        with open(expected_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 3, "File should contain one header and two data rows."
        assert "layers,units" in lines[0]  # Check header is there
        assert "4,128,3,3,3,GPU,TEST,0.05,0.04\n" in lines[2]  # Check second row data

    def test_multiple_metrics_create_multiple_files(self, results_writer):
        """
        Tests that a single result with multiple metrics creates a separate
        CSV file for each metric.
        """
        config = MockExecutionConfiguration(
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

        loss_file = os.path.join(results_writer.results_dir, "loss.csv")
        accuracy_file = os.path.join(results_writer.results_dir, "accuracy.csv")

        assert os.path.isfile(loss_file)
        assert os.path.isfile(accuracy_file)
