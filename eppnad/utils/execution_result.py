import csv
import logging
import os
from dataclasses import dataclass, fields, is_dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Union

from eppnad.utils.execution_configuration import ExecutionConfiguration


@dataclass
class StatisticalValues:
    mean: float
    std_dev: float
    median: float
    min: float
    max: float
    p25: float
    p75: float


@dataclass
class TimeAndEnergy:
    average_seconds: float
    average_joules: float


@dataclass
class ConfigAndResult:
    """Represents a single result row, linking a configuration to its outcome."""

    configuration: ExecutionConfiguration
    result: Union[StatisticalValues, TimeAndEnergy]


Metrics = dict[str, StatisticalValues]
WrittenResults = dict[str, List[ConfigAndResult]]


ExecutionResult = tuple[ExecutionConfiguration, Metrics]
EnergyExecutionResult = tuple[ExecutionConfiguration, TimeAndEnergy]


class ResultsWriter:
    """
    Handles writing execution results to CSV files.

    For each metric (e.g., 'loss', 'accuracy'), this class creates and appends
    to a corresponding CSV file. Each row in the CSV contains the full
    hyperparameter configuration and the statistical results for that metric.
    """

    _logger = logging.getLogger(__name__)

    def __init__(self, output_dir: str = "./profiler_output/"):
        """
        Initializes the ResultsWriter.

        Args:
            output_dir: The root directory where results will be stored.
        """
        # Create a dedicated results directory for the specific model
        self.results_dir = os.path.join(output_dir, "results")

    def append_execution_result(self, result: ExecutionResult):
        """
        Appends a performance result to the appropriate CSV files.

        This method iterates through each metric in the result (e.g., 'loss',
        'accuracy') and writes a row to a corresponding CSV file.

        Args:
            result: A tuple containing the ExecutionConfiguration and a
                    dictionary of the statistical performance metrics.
        """
        self._ensure_results_directory_exists()

        config, metrics = result  # type: ignore
        for metric_name, stats_values in metrics.items():
            filepath = os.path.join(self.results_dir, f"{metric_name}.csv")
            self._write_row(filepath, config, stats_values)

    def append_energy_result(self, result: EnergyExecutionResult):
        """
        Appends a single time and energy result to the 'energy.csv' file.

        Args:
            result: A tuple containing the ExecutionConfiguration and the
                    TimeAndEnergy data.
        """
        self._ensure_results_directory_exists()

        config, time_and_energy_data = result
        filepath = os.path.join(self.results_dir, "energy.csv")
        self._write_row(filepath, config, time_and_energy_data)

    def _write_row(self, filepath: str, config_obj: Any, data_obj: Any):
        """
        A generic helper to write a row to a specified CSV file.

        It dynamically generates headers from the provided dataclass objects
        and handles file creation, header writing (if new), and appending
        the data row.

        Args:
            filepath: The full path to the target CSV file.
            config_obj: The dataclass instance for the configuration part of the row.
            data_obj: The dataclass instance for the data part of the row.
        """
        if not is_dataclass(config_obj) or not is_dataclass(data_obj):
            raise TypeError("Configuration and data objects must be dataclasses.")

        file_exists = os.path.isfile(filepath)

        # Dynamically get field names from the dataclasses for the header
        config_fields = [field.name for field in fields(config_obj)]
        data_fields = [field.name for field in fields(data_obj)]
        header = config_fields + data_fields

        # Prepare the row data as a dictionary for the DictWriter
        row_dict = {}
        for field_name in config_fields:
            value = getattr(config_obj, field_name)
            # Write the name of enums for readability, not their raw value
            row_dict[field_name] = value.name if isinstance(value, Enum) else value

        row_dict.update(asdict(data_obj))  # type: ignore # Add data fields

        try:
            with open(filepath, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_dict)
        except IOError as e:
            self._logger.error(f"Error writing to CSV file {filepath}: {e}")
            raise

    def _ensure_results_directory_exists(self):
        """Creates the results directory if it does not already exist."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except IOError as e:
            self._logger.error(
                f"Error creating results directory {self.results_dir}: {e}"
            )
            raise


class ResultsReader:
    _logger = logging.getLogger(__name__)

    def __init__(self, output_dir: str = "./profiler_output/"):
        """
        Initializes the ResultsReader.

        Args:
            output_dir: The root directory where results will be stored.
        """
        # Create a dedicated results directory for the specific model
        self.results_dir = os.path.join(output_dir, "results")

    def read_results_from_directory(self) -> WrittenResults:
        """
        Args:
            result: A tuple containing the ExecutionConfiguration and a
                    dictionary of the statistical performance metrics.
        """
        if not os.path.isdir(self.results_dir):
            self._logger.warning(f"Results directory not found: {self.results_dir}")
            return {}

        all_results: WrittenResults = {}
        config_fields_map = {f.name: f for f in fields(ExecutionConfiguration)}  # type: ignore
        stats_fields_map = {f.name: f for f in fields(StatisticalValues)}
        energy_fields_map = {f.name: f for f in fields(TimeAndEnergy)}

        try:
            for filename in os.listdir(self.results_dir):
                if not filename.endswith(".csv"):
                    continue

                metric_name = os.path.splitext(filename)[0]
                filepath = os.path.join(self.results_dir, filename)
                metric_results: List[ConfigAndResult] = []

                result_fields_map, result_dataclass = (
                    (energy_fields_map, TimeAndEnergy)
                    if metric_name == "energy"
                    else (stats_fields_map, StatisticalValues)
                )

                try:
                    with open(filepath, "r", newline="") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            config_data = self._parse_row_part(row, config_fields_map)
                            result_data = self._parse_row_part(row, result_fields_map)

                            config_obj = ExecutionConfiguration(**config_data)
                            result_obj = result_dataclass(**result_data)

                            metric_results.append(
                                ConfigAndResult(
                                    configuration=config_obj, result=result_obj
                                )
                            )
                except FileNotFoundError:
                    self._logger.error(f"Could not open file: {filepath}")
                    continue
                except Exception as e:
                    self._logger.error(f"Error parsing file {filepath}: {e}")
                    continue

                all_results[metric_name] = metric_results

        except IOError as e:
            self._logger.error(
                f"Error accessing results directory {self.results_dir}: {e}"
            )
            raise

        return all_results

    def _parse_row_part(
        self, row: dict[str, str], fields_map: dict[str, Any]
    ) -> dict[str, Any]:
        """Helper to parse parts of a CSV row into a dictionary with correct types."""
        data = {}
        for key, field_info in fields_map.items():
            if key in row:
                value_str = row[key]
                field_type = field_info.type
                try:
                    if issubclass(field_type, Enum):
                        data[key] = field_type[value_str]
                    else:
                        data[key] = field_type(value_str)
                except (KeyError, ValueError) as e:
                    self._logger.error(
                        f"Cannot parse value '{value_str}' for field '{key}': {e}"
                    )
                    # Assign a default or skip, depending on desired robustness
                    data[key] = None
        return data

    def _ensure_results_directory_exists(self):
        """Creates the results directory if it does not already exist."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except IOError as e:
            self._logger.error(
                f"Error creating results directory {self.results_dir}: {e}"
            )
            raise
