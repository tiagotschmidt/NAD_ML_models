from enum import Enum


class FrameworkParameterType(Enum):
    NumberOfLayers = 0
    NumberOfNeurons = 1
    NumberOfEpochs = 2
    NumberOfFeatures = 4


class ProcessSignal(Enum):
    Start = 0
    Stop = 1


class RangeParameter:
    def __init__(self, start: int, end: int, stride: int, type: FrameworkParameterType):
        self.start = start
        self.end = end
        self.stride = stride
        self.type = type

    def __iter__(self):
        for i in range(self.start, self.end + 1, self.stride):
            yield i


class Platform(Enum):
    CPU = 0
    GPU = 1


class LifecycleSelected(Enum):
    OnlyTrain = 0
    OnlyTest = 1
    TrainAndTest = 2


class MLMode:
    def __init__(
        self,
        cycle: LifecycleSelected,
        train_platform: Platform = None,
        test_platform: Platform = None,
    ):
        self.cycle = cycle
        self.test_platform = test_platform
        self.train_platform = train_platform


class ExecutionConfiguration:
    def __init__(
        self,
        number_of_layers: int,
        number_of_neurons: int,
        number_of_epochs: int,
        number_of_features: int,
        platform: Platform,
        cycle: LifecycleSelected,
    ):
        self.number_of_layers = number_of_layers
        self.number_of_neurons = number_of_neurons
        self.number_of_epochs = number_of_epochs
        self.number_of_features = number_of_features
        self.platform = platform
        self.cycle = cycle
