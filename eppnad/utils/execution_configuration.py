"""
This module defines the data class for a single execution's configuration.
"""

from dataclasses import dataclass
from eppnad.utils.framework_parameters import Lifecycle, Platform


@dataclass(frozen=True)
class ExecutionConfiguration:
    """
    A data class to hold the specific hyperparameter configuration for a single
    profiling execution (e.g., a single train or test run).

    This object is typically created for each unique combination of hyperparameters
    to be tested by the execution engine.
    """

    layers: int
    units: int
    epochs: int
    features: int
    sampling_rate: float
    platform: Platform
    cycle: Lifecycle

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the configuration.

        This is useful for logging and debugging purposes, providing a clear
        summary of the hyperparameters used in a specific execution.

        Returns:
            A string detailing the configuration parameters.
        """
        return (
            f"ExecutionConfiguration("
            f"layers={self.layers}, "
            f"units={self.units}, "
            f"epochs={self.epochs}, "
            f"features={self.features}, "
            f"sampling_rate={self.sampling_rate}, "
            f"platform={self.platform.name}, "
            f"cycle={self.cycle.name}"
            f")"
        )
