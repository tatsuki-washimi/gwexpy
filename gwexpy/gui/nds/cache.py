"""
NDS Data Cache management.
Adapts NDSThread and DataBufferDict.
"""

from __future__ import annotations

import os
import logging
from typing import Any, ClassVar, Iterable

from qtpy import QtCore
from .nds_thread import NDSThread
from .audio_thread import AudioThread
from .sim_thread import SimulationThread
from .buffer import DataBufferDict
from .util import parse_server_string

logger = logging.getLogger(__name__)

class ChannelListCache:
    _instance: ClassVar["ChannelListCache" | None] = None
    cache: dict[str, list[str] | None]
    is_fetching: dict[str, bool]

    def __new__(cls) -> "ChannelListCache":
        if cls._instance is None:
            cls._instance = super(ChannelListCache, cls).__new__(cls)
            cls._instance.cache = {}
            cls._instance.is_fetching = {}
        return cls._instance

    def get_channels(self, server_str: str) -> list[str] | None:
        return self.cache.get(server_str)

    def set_channels(self, server_str: str, channels: list[str] | None) -> None:
        self.cache[server_str] = channels
        self.is_fetching[server_str] = False

    def has_channels(self, server_str: str) -> bool:
        return server_str in self.cache and self.cache[server_str] is not None

class NDSDataCache(QtCore.QObject):
    signal_data = QtCore.Signal(object)
    signal_payload = QtCore.Signal(object)
    signal_error = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.channels: list[str] = []
        self.server: str = os.getenv("NDSSERVER", "localhost:31200")
        self.lookback: float = 30.0
        self.buffers: DataBufferDict = DataBufferDict(self.lookback)
        self.thread: NDSThread | None = None
        self.sim_thread: SimulationThread | None = None
        self.audio_threads: dict[str, AudioThread] = {}

    def set_channels(self, channels: Iterable[str]) -> None:
        self.channels = list(set([c for c in channels if c]))

    def set_server(self, server_env: str) -> None:
        self.server = server_env

    def online_start(self, lookback: float = 30.0) -> None:
        self.lookback = lookback
        self.buffers.lookback = lookback
        if not self.channels:
            logger.warning("NDSDataCache: No channels to fetch.")
            return

        nds_chans = [c for c in self.channels if not c.startswith("PC:")]
        audio_chans = [c for c in self.channels if c.startswith("PC:")]
        if nds_chans:
            if self.thread and self.thread.isRunning():
                logger.info("NDSDataCache: NDS Thread already running.")
            else:
                # Ensure old thread is fully dead and disconnected before starting new
                self.online_stop() 
                
                host, port = parse_server_string(self.server)
                logger.info(f"Starting NDSThread for {nds_chans} on {host}:{port}")
                self.thread = NDSThread(nds_chans, host, port)
                self.thread.dataReceived.connect(self._on_data_received)
                if hasattr(self.thread, "signal_error"):
                    self.thread.signal_error.connect(self.signal_error.emit)
                self.thread.start()
        if audio_chans:
            # ... audio logic ...
            pass

    def sim_start(self, lookback: float = 30.0, fs: float = 16384) -> None:
        self.lookback = lookback
        self.buffers.lookback = lookback
        if self.sim_thread and self.sim_thread.isRunning():
            logger.info("NDSDataCache: Simulation Thread already running.")
            return
        
        self.online_stop()
        
        logger.info(f"Starting SimulationThread for {self.channels}")
        self.sim_thread = SimulationThread(self.channels, fs=fs)
        self.sim_thread.dataReceived.connect(self._on_data_received)
        self.sim_thread.start()

    def online_stop(self) -> None:
        # Disconnect signals first to avoid callbacks during shutdown
        if self.thread:
            try:
                self.thread.dataReceived.disconnect(self._on_data_received)
            except (TypeError, RuntimeError) as exc:
                logger.debug("NDSThread signal already disconnected: %s", exc)
            except Exception as exc:
                logger.warning("Unexpected error disconnecting NDSThread: %s", exc)
            self.thread.stop()
            if not self.thread.wait(3000):
                logger.warning("NDSThread did not stop in time, terminating.")
                self.thread.terminate()
            self.thread = None
            
        if self.sim_thread:
            try:
                self.sim_thread.dataReceived.disconnect(self._on_data_received)
            except (TypeError, RuntimeError) as exc:
                logger.debug("SimulationThread signal already disconnected: %s", exc)
            except Exception as exc:
                logger.warning("Unexpected error disconnecting SimulationThread: %s", exc)
            self.sim_thread.stop()
            if not self.sim_thread.wait(3000):
                self.sim_thread.terminate()
            self.sim_thread = None
            
        for ath in self.audio_threads.values():
            ath.stop()
            ath.wait(2000)
        self.audio_threads = {}

    def reset(self) -> None:
        self.online_stop()
        self.buffers.reset()

    def _on_data_received(self, payload: Any, trend: Any, is_online: bool) -> None:
        logger.info("Cache received data payload")
        self.buffers.update_buffers(payload)
        self.signal_data.emit(self.buffers)
        self.signal_payload.emit(payload)
