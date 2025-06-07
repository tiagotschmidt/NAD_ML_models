"""
This module defines the data class for a single execution's configuration.
"""

from eppnad.utils.framework_parameters import LifeCycle, Platform


class ExecutionConfiguration:
    """
    A data class to hold the specific hyperparameter configuration for a single
    profiling execution (e.g., a single train or test run).

    This object is typically created for each unique combination of hyperparameters
    to be tested by the execution engine.
    """

    def __init__(
        self,
        layers: int,
        units: int,
        number_of_epochs: int,
        features: int,
        sampling_rate: float,
        platform: Platform,
        cycle: LifeCycle,
    ):
        """
        Initializes the ExecutionConfiguration with a specific set of hyperparameters.

        Args:
            layers: The number of layers in the model for this run.
            units: The number of units (neurons) in each layer for this run.
            number_of_epochs: The number of epochs to train the model for.
            features: The number of input features selected for this run.
            sampling_rate: The sampling rate of the dataset for this run.
            platform: The execution platform (CPU or GPU) for this run.
            cycle: The lifecycle phase (Train or Test) for this run.
        """
        self.number_of_layers = layers
        self.number_of_units = units
        self.number_of_epochs = number_of_epochs
        self.number_of_features = features
        self.sampling_rate = sampling_rate
        self.platform = platform
        self.cycle = cycle

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
            f"layers={self.number_of_layers}, "
            f"units={self.number_of_units}, "
            f"epochs={self.number_of_epochs}, "
            f"features={self.number_of_features}, "
            f"sampling_rate={self.sampling_rate}, "
            f"platform={self.platform.name}, "
            f"cycle={self.cycle.name}"
            f")"
        )
