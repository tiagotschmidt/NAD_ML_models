from ...data_structures.framework_parameters import (
    FrameworkParameterType,
    LifecycleSelected,
    MLMode,
    Platform,
    RangeParameter,
)
from ..manager import __generate_configurations_list


def test_generate_configurations_list():
    # Test for OnlyTest mode
    profile_mode = MLMode(LifecycleSelected.OnlyTest, test_platform=Platform.CPU)
    configurations = __generate_configurations_list(
        numbers_of_layers=RangeParameter(
            1, 3, 1, FrameworkParameterType.NumberOfLayers
        ),
        numbers_of_neurons=RangeParameter(
            2, 4, 1, FrameworkParameterType.NumberOfNeurons
        ),
        numbers_of_epochs=RangeParameter(
            3, 5, 1, FrameworkParameterType.NumberOfEpochs
        ),
        numbers_of_features=RangeParameter(
            4, 6, 1, FrameworkParameterType.NumberOfFeatures
        ),
        profile_mode=profile_mode,
    )
    assert len(configurations) == 81
    for config in configurations:
        assert config.platform == Platform.CPU

    # Test for OnlyTrain mode
    profile_mode = MLMode(LifecycleSelected.OnlyTrain, train_platform=Platform.GPU)
    configurations = __generate_configurations_list(
        numbers_of_layers=RangeParameter(
            1, 2, 1, FrameworkParameterType.NumberOfLayers
        ),
        numbers_of_neurons=RangeParameter(
            2, 3, 1, FrameworkParameterType.NumberOfNeurons
        ),
        numbers_of_epochs=RangeParameter(
            3, 4, 1, FrameworkParameterType.NumberOfEpochs
        ),
        numbers_of_features=RangeParameter(
            4, 5, 1, FrameworkParameterType.NumberOfFeatures
        ),
        profile_mode=profile_mode,
    )
    assert len(configurations) == 16
    for config in configurations:
        assert config.platform == Platform.GPU

    # Test for TrainAndTest mode
    profile_mode = MLMode(LifecycleSelected.TrainAndTest, Platform.CPU, Platform.GPU)
    configurations = __generate_configurations_list(
        numbers_of_layers=RangeParameter(
            1, 2, 1, FrameworkParameterType.NumberOfLayers
        ),
        numbers_of_neurons=RangeParameter(
            2, 3, 1, FrameworkParameterType.NumberOfNeurons
        ),
        numbers_of_epochs=RangeParameter(
            3, 4, 1, FrameworkParameterType.NumberOfEpochs
        ),
        numbers_of_features=RangeParameter(
            4, 5, 1, FrameworkParameterType.NumberOfFeatures
        ),
        profile_mode=profile_mode,
    )
    assert len(configurations) == 32
