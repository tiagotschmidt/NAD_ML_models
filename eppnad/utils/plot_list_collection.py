from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.execution_result import StatisticalValues, TimeAndEnergy


@dataclass(frozen=True)
class PlotPoint:
    """
    Represents a single, ready-to-plot data point, containing the
    value for the x-axis and the complete result object for the y-axis.
    """

    x_value: Any  # The value of the varying hyperparameter (e.g., 2, 4, 8)
    # The y_values can now be either performance stats or energy results.
    y_values: Union[StatisticalValues, TimeAndEnergy]


# A type alias for a collection of plot points for a single graph
PlotCollection = Dict[ExecutionConfiguration, List[PlotPoint]]

SnapshotResultsList = List[Tuple[ExecutionConfiguration, List[int], List[dict]]]  # type: ignore
PlotListCollection = Dict[str, Dict[str, PlotCollection]]
