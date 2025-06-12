import csv
import logging
import os
from dataclasses import dataclass, fields, is_dataclass, asdict
from enum import Enum
from typing import Tuple, List, Dict

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


Metrics = dict[str, StatisticalValues]


ExecutionResult = Tuple[ExecutionConfiguration, Metrics]


class ResultsWriter:
    """
    Handles writing execution results to CSV files.

    For each metric (e.g., 'loss', 'accuracy'), this class creates and appends
    to a corresponding CSV file. Each row in the CSV contains the full
    hyperparameter configuration and the statistical results for that metric.
    """

    _logger = logging.getLogger(__name__)

    def __init__(self, model_name: str, output_dir: str = "./profiler_output"):
        """
        Initializes the ResultsWriter.

        Args:
            model_name: The name of the model being profiled. This will be
                        used to create a subdirectory for the results.
            output_dir: The root directory where results will be stored.
        """
        self.model_name = model_name
        # Create a dedicated results directory for the specific model
        self.results_dir = os.path.join(output_dir, "results")

    def append_execution_result(self, result: ExecutionResult):
        """
        Appends a single execution result to the appropriate CSV files.

        This method iterates through each metric in the result and writes a
        row to the corresponding CSV file (e.g., results for 'loss' go to
        'loss.csv').

        Args:
            result: A tuple containing the ExecutionConfiguration and a
                    dictionary of the statistical results (Metrics).
        """
        self._ensure_results_directory_exists()

        config, metrics = result

        for metric_name, stats_values in metrics.items():
            if not is_dataclass(stats_values):
                raise TypeError("Statistics object must be a dataclass.")

            filepath = os.path.join(self.results_dir, f"{metric_name}.csv")
            self._write_metric_row(filepath, config, stats_values)

    def _write_metric_row(
        self, filepath: str, config: ExecutionConfiguration, stats: StatisticalValues
    ):
        """Handles writing a single row to a specific metric CSV file."""
        file_exists = os.path.isfile(filepath)

        # Get field names dynamically from the dataclasses
        config_fields = [
            "layers",
            "units",
            "epochs",
            "features",
            "sampling_rate",
            "platform",
            "cycle",
        ]
        stats_fields = [field.name for field in fields(stats)]
        header = config_fields + stats_fields

        # Prepare the row data as a dictionary
        row_dict = {}
        for field_name in config_fields:
            value = getattr(config, field_name)
            # Write the name of enums for readability, not their raw value
            row_dict[field_name] = value.name if isinstance(value, Enum) else value

        # asdict is a convenient way to convert a dataclass instance to a dict
        row_dict.update(asdict(stats))

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
