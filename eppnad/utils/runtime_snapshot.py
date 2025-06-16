"""
This module defines the RuntimeSnapshot class, used for saving and loading
the state of a profiling session to enable intermittent execution.
"""

import pickle
import os
import time
import logging
from typing import List, Optional

# Assuming these classes are defined in your project structure
from eppnad.utils.execution_configuration import ExecutionConfiguration
from eppnad.utils.model_execution_config import ModelExecutionConfig


class RuntimeSnapshot:
    """
    Represents a serializable snapshot of the profiler's state.

    This class handles its own file I/O, saving to and loading from a
    predefined directory. It is designed to be saved to a file, allowing a
    profiling session to be paused and resumed later.
    """

    def __init__(
        self,
        model_name: str,
        profile_execution_directory: str,
        configuration_list: List[ExecutionConfiguration],
        model_execution_config: ModelExecutionConfig,
        last_profiled_index: int = -1,
        filepath: Optional[str] = None,
    ):
        """
        Initializes the RuntimeSnapshot.

        Args:
            model_name: The name of the model being profiled (e.g., "MLP").
            configuration_list: The complete list of all hyperparameter configs.
            model_execution_config: The general model execution configuration.
            last_profiled_index: The index of the last completed config.
            results_so_far: A list to accumulate results from completed configs.
            filepath: The path to the file this snapshot is associated with.
                      This is managed internally by the save/load methods.
        """
        self.model_name = model_name
        self.runtime_snapshot_dir = profile_execution_directory + "runtime_snapshot/"
        self.configuration_list = configuration_list
        self.model_execution_configuration = model_execution_config
        self.last_profiled_index = last_profiled_index
        self.filepath = filepath  # The path to this snapshot's file

    def save(self):
        """
        Serializes and saves the snapshot to its file.

        If the snapshot was loaded from a file or previously saved, it will
        overwrite that same file. If this is a new, unsaved snapshot, it will
        create a new file with a unique timestamp-based name.

        The method automatically creates the destination directory if needed.

        Returns:
            The filepath where the snapshot was saved.
        """
        # If the snapshot doesn't have a file yet, create one.
        if self.filepath is None:
            timestamp = int(time.time())
            filename = f"{self.model_name}_{timestamp}.snapshot"
            self.filepath = os.path.join(self.runtime_snapshot_dir, filename)

        # self._logger.info(f"Saving runtime snapshot to {self.filepath}...")
        try:
            # Ensure the directory exists
            directory = os.path.dirname(self.filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(self.filepath, "wb") as filepath:
                pickle.dump(self, filepath)
            # self._logger.info("Snapshot saved successfully.")
            return self.filepath
        except IOError as e:
            # self._logger.error(f"Error saving snapshot to {self.filepath}: {e}")
            raise

    @classmethod
    def load_latest(cls, runtime_snapshot_dir) -> Optional["RuntimeSnapshot"]:
        """
        Finds and loads the most recent snapshot from the predefined directory.

        It determines the latest snapshot by finding the file with the most
        recent timestamp in its name.

        Returns:
            An instance of the RuntimeSnapshot class with the loaded state,
            or None if no snapshots are found.
        """
        # cls._logger.info(f"Searching for latest snapshot in {runtime_snapshot_dir}...")
        if not os.path.isdir(runtime_snapshot_dir):
            # cls._logger.warning("Snapshot directory not found. Starting a new run.")
            return None

        try:
            snapshot_files = [
                file
                for file in os.listdir(runtime_snapshot_dir)
                if file.endswith(".snapshot")
            ]

            if not snapshot_files:
                # cls._logger.warning(
                #     "No snapshot files found in directory. Starting a new run."
                # )
                return None

            # Find the latest file based on the timestamp in the name
            latest_file = max(snapshot_files)
            latest_filepath = os.path.join(runtime_snapshot_dir, latest_file)

            # cls._logger.info(f"Loading latest snapshot: {latest_filepath}")
            with open(latest_filepath, "rb") as f:
                snapshot = pickle.load(f)

            if not isinstance(snapshot, cls):
                raise TypeError(
                    f"File {latest_filepath} is not a valid RuntimeSnapshot."
                )

            # Assign the filepath to the loaded object
            snapshot.filepath = latest_filepath
            # cls._logger.info("Snapshot loaded successfully.")
            return snapshot

        except (IOError, pickle.UnpicklingError, TypeError) as e:
            # cls._logger.error(f"Failed to load or validate latest snapshot: {e}")
            return None

    def __str__(self) -> str:
        """Provides a human-readable summary of the snapshot's state."""
        total_configs = len(self.configuration_list)
        completed_configs = self.last_profiled_index + 1
        return (
            f"<RuntimeSnapshot for '{self.model_name}': "
            f"{completed_configs}/{total_configs} configurations completed>"
        )
