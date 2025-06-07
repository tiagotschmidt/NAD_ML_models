"""
This module contains the definitions for framework parameters.
"""

from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import pandas as pd


class FrameworkParameterType(Enum):
    """Enumeration for the different types of hyperparameters supported by the framework."""
    NumberOfLayers = 0
    NumberOfNeurons = 1
    NumberOfEpochs = 2
    NumberOfFeatures = 4


class ProcessSignal(Enum):
    Start = 0
    Stop = 1
    FinalStop = 2


class RangeMode(Enum):
    """Enumeration for the mode of range generation (additive or multiplicative)."""
    Additive = 0
    Multiplicative = 1


class RangeParameter:
    """A container for a list of parameter values to be iterated over.

    This class can be initialized directly with a list of values or through
    the `from_range` classmethod factory, which generates the list based on
    a specified start, end, and stride.
    """
    def __init__(self, item_list: List[Any]):
        """Initializes the RangeParameter with a pre-defined list of items."""
        self._iterable_list = item_list

    @classmethod
    def from_range(
        cls,
        start: int,
        end: int,
        stride: int,
        # 'type' is unused in this method but kept for API consistency.
        type: FrameworkParameterType,
        mode: RangeMode,
    ) -> 'RangeParameter':
        """Constructs a RangeParameter by generating a list of values.

        Args:
            start: The starting value of the range.
            end: The ending value of the range (inclusive).
            stride: The step to take between values.
            type: The type of framework parameter (for API consistency).
            mode: The mode of range generation (Additive or Multiplicative).

        Returns:
            An instance of RangeParameter containing the generated list.

        Raises:
            ValueError: If parameters for multiplicative mode are invalid
                        (e.g., start=0, stride=0, or stride=1 causing a loop).
        """
        _iterable_list: List[int] = []
        if mode == RangeMode.Additive:
            # range() handles stride=0 by raising a ValueError automatically.
            _iterable_list = list(range(start, end + 1, stride))
        elif mode == RangeMode.Multiplicative:
            # Manually check for invalid arguments to prevent infinite loops.
            if stride <= 0:
                raise ValueError("Multiplicative mode stride cannot be 0")
            if start <= 0:
                raise ValueError("Multiplicative mode start cannot be 0")
            if stride == 1:
                # If stride is 1, it will loop infinitely if start <= end.
                # We return a list with only the start value in this case.
                _iterable_list = [start]
            else:
                value = start
                while value <= end:
                    _iterable_list.append(value)
                    value *= stride

        return cls(_iterable_list)

    def __iter__(self):
        """Allows iteration over the internal list of parameter values."""
        for item in self._iterable_list:
            yield item


class Platform(Enum):
    """Enumeration for the execution platform (CPU or GPU)."""
    CPU = 0
    GPU = 1


class Lifecycle(Enum):
    """Enumeration for the execution lifecycle phase (Train or Test)."""
    Train = 0
    Test = 1


class LifecycleSelected(Enum):
    """Enumeration for the selected lifecycle combination."""
    OnlyTrain = 0
    OnlyTest = 1
    TrainAndTest = 2


class MLMode:
    """Data class to hold the selected ML lifecycle and platform configuration."""
    def __init__(
        self,
        cycle: LifecycleSelected,
        train_platform: Optional[Platform] = None,
        test_platform: Optional[Platform] = None,
    ):
        """Initializes the MLMode configuration.

        Args:
            cycle: The selected lifecycle mode.
            train_platform: The platform for the training phase, if applicable.
            test_platform: The platform for the testing phase, if applicable.
        """
        self.cycle = cycle
        self.test_platform = test_platform
        self.train_platform = train_platform