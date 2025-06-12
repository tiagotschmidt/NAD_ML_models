# tests/test_runtime_snapshot.py

import os
import time
import pickle
import pytest

from eppnad.utils.runtime_snapshot import RuntimeSnapshot

# --- Test Fixtures ---


@pytest.fixture
def temp_snapshot_dir(tmp_path, monkeypatch):
    """
    Creates a temporary directory for snapshots and configures the class
    to use it for the duration of a test.
    """
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    model_dir = snapshot_dir / "test_model"
    model_dir.mkdir()
    old_model_dir = snapshot_dir / "old_model"
    old_model_dir.mkdir()
    return str(snapshot_dir)


@pytest.fixture
def snapshot_instance(temp_snapshot_dir):
    """
    Provides a fresh, picklable RuntimeSnapshot instance for each test.

    THIS FIXTURE FIXES: PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>
    by passing `None` instead of a mock object for the config.
    """
    return RuntimeSnapshot(
        model_name="test_model",
        root_directory=temp_snapshot_dir + "test_model/",
        configuration_list=[1, 2, 3],  # type: ignore
        model_execution_config=None,  # Use None to make it picklable # type: ignore
        last_profiled_index=-1,
    )


# --- Test Cases ---


class TestRuntimeSnapshot:
    """Groups all tests for the RuntimeSnapshot class."""

    def test_initialization(self, snapshot_instance):
        """Tests that the constructor correctly assigns all attributes."""
        assert snapshot_instance.model_name == "test_model"
        assert snapshot_instance.last_profiled_index == -1
        assert snapshot_instance.filepath is None

    def test_string_representation(self, snapshot_instance):
        """Tests the __str__ method for a clear, human-readable output."""
        expected_str = (
            "<RuntimeSnapshot for 'test_model': 0/3 configurations completed>"
        )
        assert str(snapshot_instance) == expected_str
        snapshot_instance.last_profiled_index = 0
        expected_str_progress = (
            "<RuntimeSnapshot for 'test_model': 1/3 configurations completed>"
        )
        assert str(snapshot_instance) == expected_str_progress


class TestSnapshotSaving:
    """Tests the file saving functionality."""

    def test_save_creates_new_timestamped_file(
        self, snapshot_instance, temp_snapshot_dir
    ):
        """
        Tests that save() creates a new file with a timestamp when called on
        a new snapshot instance.
        """
        saved_path = snapshot_instance.save()
        assert os.path.exists(saved_path)
        assert snapshot_instance.filepath == saved_path
        with open(saved_path, "rb") as f:
            loaded_snapshot = pickle.load(f)
        assert loaded_snapshot.model_name == "test_model"

    def test_save_overwrites_existing_file(self, snapshot_instance, temp_snapshot_dir):
        """
        Tests that calling save() multiple times on the same instance
        overwrites the same file instead of creating new ones.
        """
        first_path = snapshot_instance.save()
        assert len(os.listdir(temp_snapshot_dir + "test_model/")) == 1
        snapshot_instance.last_profiled_index = 5
        second_path = snapshot_instance.save()
        assert len(os.listdir(temp_snapshot_dir + "test_model/")) == 1
        assert first_path == second_path
        with open(second_path, "rb") as f:
            reloaded_snapshot = pickle.load(f)
        assert reloaded_snapshot.last_profiled_index == 5


class TestSnapshotLoading:
    """Tests the file loading functionality."""

    def test_load_latest_returns_none_if_dir_not_exists(self, monkeypatch):
        """
        Tests that load_latest returns None if the snapshot directory
        doesn't exist at all.
        """
        assert RuntimeSnapshot.load_latest("./non_existent_dir/") is None

    def test_load_latest_returns_none_for_empty_dir(self, temp_snapshot_dir):
        """
        Tests that load_latest returns None if the directory exists but is empty.
        """
        assert os.listdir(temp_snapshot_dir + "/" + "test_model/") == []
        assert RuntimeSnapshot.load_latest(temp_snapshot_dir + "test_model/") is None

    def test_load_latest_finds_the_most_recent_file(self, temp_snapshot_dir):
        """
        Tests that load_latest correctly identifies and loads the file with
        the highest (most recent) timestamp in its name.
        """
        base_time = int(time.time())
        old_snapshot = RuntimeSnapshot("old_model", temp_snapshot_dir + "/" + "old_model/", [], None)  # type: ignore
        old_filepath = os.path.join(
            temp_snapshot_dir + "/" + "old_model/",
            f"old_model_{base_time - 100}.snapshot",
        )
        with open(old_filepath, "wb") as f:
            pickle.dump(old_snapshot, f)

        latest_snapshot = RuntimeSnapshot(
            "old_model", temp_snapshot_dir + "/" + "old_model/", [1], None, last_profiled_index=99  # type: ignore
        )
        latest_filepath = os.path.join(
            temp_snapshot_dir + "/" + "old_model/", f"old_model_{base_time}.snapshot"
        )
        with open(latest_filepath, "wb") as f:
            pickle.dump(latest_snapshot, f)

        loaded = RuntimeSnapshot.load_latest(temp_snapshot_dir + "/" + "old_model/")
        assert loaded is not None
        assert loaded.model_name == "old_model"
        assert loaded.last_profiled_index == 99
        assert loaded.filepath == latest_filepath
