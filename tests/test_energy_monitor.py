# eppnad/tests/test_energy_profiler.py

import subprocess
from unittest.mock import MagicMock, call
import pytest

from eppnad.core.energy_monitor import EnergyMonitor
from eppnad.utils.framework_parameters import Platform, ProcessSignal

# --- Test Fixtures ---


@pytest.fixture
def mock_pipes(mocker):
    """Creates a dictionary of mock pipes for testing."""
    return {
        "signal_pipe": mocker.MagicMock(),
        "result_pipe": mocker.MagicMock(),
    }


@pytest.fixture
def mock_logger(mocker):
    """Creates a mock logger."""
    return mocker.MagicMock()


# --- Test Cases ---


class TestEnergyProfiler:
    """Tests the core logic of the EnergyProfiler class."""

    def test_CPU_profiling_workflow(self, mock_pipes, mock_logger, mocker):
        """
        Tests that the profiler correctly handles a start/stop CPU cycle
        and sends a valid result.
        """
        # 1. Setup Mocks
        # Configure the signal pipe to deliver signals in sequence
        mock_pipes["signal_pipe"].recv.side_effect = [
            (ProcessSignal.Start, Platform.CPU),
            (ProcessSignal.Stop, Platform.CPU),
            (ProcessSignal.FinalStop, None),
        ]

        # Mock pyRAPL
        mock_pyrapl = mocker.patch("eppnad.core.energy_monitor.pyRAPL")
        mock_measurement = MagicMock()
        mock_measurement.result.pkg = [123456]  # Fake energy reading
        mock_pyrapl.Measurement.return_value = mock_measurement

        # 2. Initialize and run the profiler
        profiler = EnergyMonitor(**mock_pipes, logger=mock_logger)
        profiler.run()

        # 3. Assertions
        mock_pyrapl.setup.assert_called_once()
        mock_measurement.begin.assert_called_once()
        mock_measurement.end.assert_called_once()
        mock_pipes["result_pipe"].send.assert_called_once_with(123456)

    def test_GPU_profiling_workflow(self, mock_pipes, mock_logger, mocker):
        """
        Tests that the profiler correctly samples GPU power until a stop
        signal is received.
        """
        # 1. Setup Mocks
        # First recv is the start signal, second is the final stop
        mock_pipes["signal_pipe"].recv.side_effect = [
            (ProcessSignal.Start, Platform.GPU),
            (ProcessSignal.Stop, Platform.GPU),
            (ProcessSignal.FinalStop, None),  # To terminate the main loop
        ]
        # poll() controls the inner GPU loop. Return False twice, then True.
        mock_pipes["signal_pipe"].poll.side_effect = [False, False, True]

        # Mock subprocess.run to simulate nvidia-smi output
        mock_subprocess = mocker.patch("eppnad.core.energy_monitor.subprocess.run")
        mock_subprocess.return_value.stdout = "45.12\n"

        # 2. Initialize and run
        profiler = EnergyMonitor(**mock_pipes, logger=mock_logger)
        profiler.run()

        # 3. Assertions
        # It should have called nvidia-smi twice before poll() returned True
        assert mock_subprocess.call_count == 2

        # The result sent back should be a list of the parsed floats
        mock_pipes["result_pipe"].send.assert_called_once_with([45.12, 45.12])

    def test_final_stop_signal_terminates_run_loop(self, mock_pipes, mock_logger):
        """
        Tests that receiving a FinalStop signal cleanly exits the main run loop.
        """
        # 1. Setup Mocks
        # The first and only signal received is FinalStop
        mock_pipes["signal_pipe"].recv.return_value = (ProcessSignal.FinalStop, None)

        # 2. Initialize and run
        profiler = EnergyMonitor(**mock_pipes, logger=mock_logger)
        profiler.run()

        # 3. Assertions
        # recv should be called only once
        mock_pipes["signal_pipe"].recv.assert_called_once()
        # result_pipe.send should never be called
        mock_pipes["result_pipe"].send.assert_not_called()
        mock_logger.info.assert_any_call(
            "[PROFILER] Received FINAL_STOP signal. Shutting down."
        )
