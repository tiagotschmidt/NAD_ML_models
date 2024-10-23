from enum import Enum

class FrameworkParameterType(Enum):
    NumberOfLayers = 0
    NumberOfNeurons = 1
    NumberOfEpochs = 2
    NumberOfFeatures = 4

class RangeParameter:
    def __init__(self, start:int, end:int, stride:int, type:FrameworkParameterType):
        self.start=start
        self.end=end
        self.stride=stride
        self.type=type
        pass

    def __iter__(self):
        for i in range(self.start,self.end,self.stride):
            yield i

class Platform(Enum):
    CPU=0
    GPU=1

class LifecycleSelected(Enum):
    OnlyTrain = 0
    OnlyTest = 1
    TrainAndTest = 2

class Mode:
    def __init__(self,cycle:LifecycleSelected, platform: Platform):
        self.cycle=cycle
        self.platform = platform


a = Mode(LifecycleSelected.TrainAndTest,Platform.CPU)
        

