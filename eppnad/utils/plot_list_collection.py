from typing import List, Tuple
from eppnad.utils.execution_configuration import ExecutionConfiguration

SnapshotResultsList = List[Tuple[ExecutionConfiguration, List[int], List[dict]]]  # type: ignore


class PlotListCollection:
    def __init__(
        self,
        test_epoch_lists: SnapshotResultsList,
        test_units_lists: SnapshotResultsList,
        test_features_lists: SnapshotResultsList,
        test_layers_lists: SnapshotResultsList,
        train_epoch_lists: SnapshotResultsList,
        train_units_lists: SnapshotResultsList,
        train_features_lists: SnapshotResultsList,
        train_layers_lists: SnapshotResultsList,
    ):
        self.test_epoch_lists = test_epoch_lists
        self.test_units_lists = test_units_lists
        self.test_features_lists = test_features_lists
        self.test_layers_lists = test_layers_lists
        self.train_epoch_lists = train_epoch_lists
        self.train_units_lists = train_units_lists
        self.train_features_lists = train_features_lists
        self.train_layers_lists = train_layers_lists

    def __str__(self):
        return (
            f"PlotListCollection(\n"
            f"  test_epoch_lists={self.test_epoch_lists},\n"
            f"  test_units_lists={self.test_units_lists},\n"
            f"  test_features_lists={self.test_features_lists},\n"
            f"  test_layers_lists={self.test_layers_lists},\n"
            f"  train_epoch_lists={self.train_epoch_lists},\n"
            f"  train_units_lists={self.train_units_lists},\n"
            f"  train_features_lists={self.train_features_lists},\n"
            f"  train_layers_lists={self.train_layers_lists}\n"
            f")"
        )
