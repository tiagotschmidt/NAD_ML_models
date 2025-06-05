from eppnad.framework_parameters import (
    FrameworkParameterType,
    RangeMode,
    RangeParameter,
)


def test_range_parameter_iterator():
    """
    Tests the iterator of the RangeParameter class for both
    Additive and Multiplicative modes.
    """
    # Test Case 1: Additive Mode
    additive_range = RangeParameter(
        start=10,
        end=50,
        stride=10,
        type=FrameworkParameterType.NumberOfEpochs,
        mode=RangeMode.Additive,
    )
    expected_additive_list = [10, 20, 30, 40, 50]
    actual_additive_list = list(additive_range)
    assert (
        actual_additive_list == expected_additive_list
    ), f"Additive mode failed: expected {expected_additive_list}, got {actual_additive_list}"

    # Test Case 2: Multiplicative Mode
    multiplicative_range = RangeParameter(
        start=2,
        end=30,
        stride=2,
        type=FrameworkParameterType.NumberOfNeurons,
        mode=RangeMode.Multiplicative,
    )
    expected_multiplicative_list = [2, 4, 8, 16]
    actual_multiplicative_list = list(multiplicative_range)
    assert (
        actual_multiplicative_list == expected_multiplicative_list
    ), f"Multiplicative mode failed: expected {expected_multiplicative_list}, got {actual_multiplicative_list}"

    # Test Case 3: Additive Mode with a stride that doesn't perfectly align with the end
    additive_range_imperfect_stride = RangeParameter(
        start=5,
        end=25,
        stride=7,
        type=FrameworkParameterType.NumberOfLayers,
        mode=RangeMode.Additive,
    )
    expected_additive_imperfect = [5, 12, 19]
    actual_additive_imperfect = list(additive_range_imperfect_stride)
    assert (
        actual_additive_imperfect == expected_additive_imperfect
    ), f"Additive mode (imperfect stride) failed: expected {expected_additive_imperfect}, got {actual_additive_imperfect}"
