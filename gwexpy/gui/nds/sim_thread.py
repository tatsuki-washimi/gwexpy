from __future__ import annotations

import logging
import time

import numpy as np
from qtpy import QtCore

logger = logging.getLogger(__name__)


class SimulationThread(QtCore.QThread):  # type: ignore[name-defined]
    dataReceived = QtCore.Signal(object, object, bool)

    def __init__(self, channels, fs=16384, chunk_sec=1.0):
        super().__init__()
        self.channels = channels
        self.fs = fs
        self.chunk_sec = chunk_sec
        self.chunk_samples = int(fs * chunk_sec)
        self._stop = False
        self.t_offset = time.time()
        logger.info(f"SimulationThread initialized for channels: {self.channels}")

    def stop(self):
        self._stop = True

    def run(self):
        logger.info("SimulationThread run() started")
        while not self._stop:
            start_t = time.time()
            payload = {}

            t = np.linspace(
                self.t_offset,
                self.t_offset + self.chunk_sec,
                self.chunk_samples,
                endpoint=False,
            )
            dt = 1.0 / self.fs

            for ch in self.channels:
                if "white_noise" in ch:
                    data = np.random.normal(0, 1.0, self.chunk_samples)
                elif "sine" in ch:
                    data = np.sin(2 * np.pi * 10.0 * t)
                else:
                    data = np.random.normal(0, 0.1, self.chunk_samples)

                payload[ch] = {"data": data, "gps_start": self.t_offset, "step": dt}

            logger.info(
                f"SimulationThread emitting payload with {len(payload)} channels"
            )
            self.dataReceived.emit(payload, False, True)
            self.t_offset += self.chunk_sec

            elapsed = time.time() - start_t
            sleep_time = max(0.01, self.chunk_sec - elapsed)
            time.sleep(sleep_time)
        logger.info("SimulationThread run() finished")
