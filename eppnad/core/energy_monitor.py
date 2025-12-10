import logging
import multiprocessing
from pathlib import Path
import subprocess
import time
from logging import Logger
from multiprocessing.connection import Connection

import pyRAPL

from eppnad.utils.framework_parameters import Platform, ProcessSignal


class EnergyMonitor(multiprocessing.Process):
    """
    A dedicated process for monitoring energy consumption of CPU and GPU.

    This process listens for signals to start and stop monitoring on a given
    hardware platform. It uses pyRAPL for CPU energy measurements and the
    nvidia-smi command-line tool for GPU power sampling.
    """

    def __init__(
        self,
        signal_pipe: Connection,
        result_pipe: Connection,
        logger: Logger,
        log_dir: str,  # <--- ADD THIS ARGUMENT
    ):
        """
        Initializes the EnergyProfiler process.

        Args:
            signal_pipe: The pipe to receive start/stop signals from the engine.
            result_pipe: The pipe to send measurement results back to the engine.
            logger: A shared logger instance for logging messages.
        """
        super().__init__()
        self.signal_pipe = signal_pipe
        self.result_pipe = result_pipe
        self.logger = logger
        self.log_dir = log_dir

    def run(self):
        """
        The main loop for the profiler process.

        It waits for a signal and dispatches to the appropriate monitoring
        function. The loop terminates upon receiving a `FinalStop` signal.
        """
        if not self.logger.handlers:
            log_path = Path(self.log_dir) / "eppnad.log"

            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(fh)
            self.logger.setLevel(logging.DEBUG)

        self.logger.info(
            "[PROFILER] EnergyProfiler process started and waiting for signals."
        )
        pyRAPL.setup()

        while True:
            try:
                signal, platform = self.signal_pipe.recv()
            except EOFError:
                self.logger.warning(
                    "[PROFILER] Signal pipe closed unexpectedly. Shutting down."
                )
                break

            if signal == ProcessSignal.Start:
                self.logger.info(
                    f"[PROFILER] Received START signal for {platform.name}."
                )
                if platform == Platform.CPU:
                    self._profile_cpu()
                elif platform == Platform.GPU:
                    self._profile_gpu()

            elif signal == ProcessSignal.FinalStop:
                self.logger.info(
                    "[PROFILER] Received FINAL_STOP signal. Shutting down."
                )
                break

    def _profile_cpu(self):
        """
        Measures CPU energy consumption for a defined period.

        This method begins a pyRAPL measurement and waits for a 'Stop' signal
        to end it. It then safely extracts and sends the energy data.
        """
        measurement = pyRAPL.Measurement("rapl_measurement")
        measurement.begin()

        # Wait for the stop signal
        self.signal_pipe.recv()
        measurement.end()
        self.logger.info("[PROFILER] Stopped CPU profiling.")

        energy_microjoules = 0
        try:
            # Safely access the result to prevent crashes
            if measurement.result and measurement.result.pkg:
                energy_microjoules = measurement.result.pkg[0]
            else:
                self.logger.warning(
                    "[PROFILER] pyRAPL result was empty. Reporting 0 energy."
                )
        except (AttributeError, IndexError) as e:
            self.logger.error(
                f"[PROFILER] Could not extract pyRAPL data: {e}. Reporting 0."
            )

        self.result_pipe.send(energy_microjoules)
        self.logger.info(f"[PROFILER] Sent CPU energy result: {energy_microjoules} uJ.")

    def _profile_gpu(self):
        """
        Samples GPU power usage at a high frequency.

        This method repeatedly calls `nvidia-smi` to get power readings until
        a 'Stop' signal is received. It then sends the list of all collected
        power samples.
        """
        power_samples = []
        while not self.signal_pipe.poll():  # Continue as long as no stop signal
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,  # Raise exception on non-zero exit codes
                )
                power_usage = float(result.stdout.strip())
                power_samples.append(power_usage)
                time.sleep(0.01)  # Small sleep to prevent overwhelming the CPU
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
                self.logger.error(
                    f"[PROFILER] nvidia-smi query failed: {e}. Skipping sample."
                )

        # Consume the stop signal
        self.signal_pipe.recv()
        self.logger.info("[PROFILER] Stopped GPU profiling.")

        self.result_pipe.send(power_samples)
        self.logger.info(f"[PROFILER] Sent {len(power_samples)} GPU power samples.")
