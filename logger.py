import multiprocessing
from multiprocessing.connection import Connection
from time import sleep

from framework_parameters import ProcessSignal


class Logger(multiprocessing.Process):
    def __init__(
        self,
        start_pipe: Connection,
        log_pipe: Connection,
    ):
        super(Logger, self).__init__()
        self.start_pipe = start_pipe
        self.log_pipe = log_pipe

    def run(self):
        start_trigger = self.start_pipe.recv()
        stop = False

        while not stop:
            signal = self.log_pipe.recv()
            if signal == ProcessSignal.Start:
                print("Comecei")

            signal = self.log_pipe.recv()
            if signal == ProcessSignal.Stop:
                print("Parei")
