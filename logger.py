import subprocess
import pyRAPL
import multiprocessing
from multiprocessing.connection import Connection
from time import sleep

from framework_parameters import Platform, ProcessSignal


class Logger(multiprocessing.Process):
    def __init__(
        self,
        start_pipe: Connection,
        log_queue: multiprocessing.Queue,
        internal_logger,
    ):
        super(Logger, self).__init__()
        self.start_pipe = start_pipe
        self.log_queue = log_queue
        self.internal_logger = internal_logger

    def run(self):
        start_trigger = self.start_pipe.recv()
        stop = False

        pyRAPL.setup()
        cpu_power_meter = pyRAPL.Measurement("measurement")
        gpu_power_results = []

        while not stop:
            (signal, platform) = self.log_queue.get()
            if signal.value == ProcessSignal.Start.value:
                if platform.value == Platform.CPU.value:
                    cpu_power_meter.begin()

                    ((signal, platform)) = self.log_queue.get()
                    cpu_power_meter.end()
                    print(cpu_power_meter.result)
                    self.log_queue.put(cpu_power_meter.result)
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

                        try:
                            (item, item2) = self.log_queue.get(block=False)
                            if isinstance(item, ProcessSignal):
                                signal = item
                        except:
                            sleep(0.005)
                            pass
                    self.log_queue.put(gpu_power_results)
                    sleep(1)
            if signal.value == ProcessSignal.FinalStop.value:
                stop = True
