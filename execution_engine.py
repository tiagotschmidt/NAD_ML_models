import multiprocessing
from multiprocessing.connection import Connection
from time import sleep
from typing import List
import pandas as pd
from framework_parameters import ExecutionConfiguration, ProcessSignal


class ExecutionEngine(multiprocessing.Process):
    def __init__(
        self,
        configuration_list: List[ExecutionConfiguration],
        number_of_samples: int,
        batch_size: int,
        performance_metrics_list: List[str],
        dataset: pd.DataFrame,
        start_pipe: Connection,
        configurations_queue,
        log_pipe: Connection,
    ):
        super(ExecutionEngine, self).__init__()
        self.configuration_list = configuration_list
        self.number_of_samples = number_of_samples
        self.batch_size = batch_size
        self.performance_metrics_list = performance_metrics_list
        self.dataset = dataset
        self.start_pipe = start_pipe
        self.configuration_queue = configurations_queue
        self.log_pipe = log_pipe

    def run(self):
        start_trigger = self.start_pipe.recv()
        stop = False

        while not stop:
            current_configuration = self.configuration_queue.get()
            if current_configuration is None:
                break
            self.log_pipe.send(ProcessSignal.Start)

            self.log_pipe.send(ProcessSignal.Stop)
