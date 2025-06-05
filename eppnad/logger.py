import gc
import subprocess
import pyRAPL
import multiprocessing
from multiprocessing.connection import Connection
from .framework_parameters import Platform, ProcessSignal


class Logger(multiprocessing.Process):
    def __init__(
        self,
        start_pipe: Connection,
        internal_logger,
        signal_pipe: Connection,
        result_pipe: Connection,
    ):
        super(Logger, self).__init__()
        self.start_pipe = start_pipe
        self.internal_logger = internal_logger
        self.signal_pipe = signal_pipe
        self.result_pipe = result_pipe

    def run(self):
        start_trigger = self.start_pipe.recv()
        stop = False

        pyRAPL.setup()
        cpu_power_meter = pyRAPL.Measurement("measurement")
        gpu_power_results = []

        while not stop:
            gpu_power_results = []
            (signal, platform) = self.signal_pipe.recv()
            self.internal_logger.info(
                "[LOGGER] Starting logging for:" + str(signal) + ";" + str(platform)
            )
            if signal.value == ProcessSignal.Start.value:
                if platform.value == Platform.CPU.value:
                    cpu_power_meter.begin()

                    (signal, platform) = self.signal_pipe.recv()
                    self.internal_logger.info(
                        "[LOGGER] Stop logging for:" + str(signal) + ";" + str(platform)
                    )
                    cpu_power_meter.end()
                    total_energy_microjoules = 0
                    if cpu_power_meter != None and cpu_power_meter.result != None and cpu_power_meter.result.pkg != None and cpu_power_meter.result.pkg[0] != None:  # type: ignore
                        total_energy_microjoules = cpu_power_meter.result.pkg[0]  # type: ignore
                    else:
                        self.internal_logger.warning(
                            "[LOGGER] Registering poisonous energy information."
                        )
                        total_energy_microjoules = 0
                        cpu_power_meter = pyRAPL.Measurement("measurement")
                    self.result_pipe.send(total_energy_microjoules)  # type: ignore
                    self.internal_logger.info("[LOGGER] Sending cpu total energy.")
                else:
                    while signal.value != ProcessSignal.Stop.value:
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv"],
                            capture_output=True,
                            text=True,
                        )
                        output = result.stdout.strip()
                        power_usage_str = output.split("\n")[1]
                        power_usage = float(power_usage_str[:-2])
                        gpu_power_results.append(power_usage)

                        if self.signal_pipe.poll(timeout=0.005):
                            (signal, platform) = self.signal_pipe.recv()
                            self.internal_logger.info(
                                "[LOGGER] Stop logging for:"
                                + str(signal)
                                + ";"
                                + str(platform)
                            )

                    self.result_pipe.send(gpu_power_results)
                    self.internal_logger.info(
                        "[LOGGER] Sending GPU power samples. Total:"
                        + str(len(gpu_power_results))
                    )
            if signal.value == ProcessSignal.FinalStop.value:
                stop = True
