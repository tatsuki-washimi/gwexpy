"""
NDS Data Cache management.
Adapts NDSThread and DataBufferDict.
"""

import os
from qtpy import QtCore
from .nds_thread import NDSThread
from .audio_thread import AudioThread
from .buffer import DataBufferDict
from .util import parse_server_string




class ChannelListCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChannelListCache, cls).__new__(cls)
            cls._instance.cache = {}  # server_str -> list of (name, rate, type)
            cls._instance.is_fetching = {}  # server_str -> bool
        return cls._instance

    def get_channels(self, server_str):
        return self.cache.get(server_str)

    def set_channels(self, server_str, channels):
        """
        channels: list of (name, rate, type) tuples
        """
        self.cache[server_str] = channels
        self.is_fetching[server_str] = False

    def has_channels(self, server_str):
        return server_str in self.cache and self.cache[server_str] is not None


class NDSDataCache(QtCore.QObject):
    signal_data = QtCore.Signal(object)  # emit(DataBufferDict)
    signal_payload = QtCore.Signal(object) # emit(payload) - incremental data

    def __init__(self):
        super().__init__()
        self.channels = []
        self.server = os.getenv("NDSSERVER", "localhost:31200")
        self.lookback = 30.0
        self.buffers = DataBufferDict(self.lookback)
        self.thread = None
        self.audio_threads = {}  # Dict of device_index -> AudioThread
        self.active_tid = None

    def set_channels(self, channels):
        self.channels = list(set([c for c in channels if c]))  # Unique, non-empty

    def set_server(self, server_env):
        self.server = server_env

    def online_start(self, lookback=30.0):
        self.lookback = lookback
        self.buffers.lookback = lookback

        if not self.channels:
            print("NDSDataCache: No channels to fetch.")
            return

        # Split channels
        nds_chans = [c for c in self.channels if not c.startswith("PC:")]
        audio_chans = [c for c in self.channels if c.startswith("PC:")]

        # Start NDS Thread if needed
        if nds_chans:
            if self.thread and self.thread.isRunning():
                print("NDSDataCache: NDS Thread already running.")
            else:
                host, port = parse_server_string(self.server)
                print(f"DEBUG: Starting NDSThread for {nds_chans} on {host}:{port}")
                self.thread = NDSThread(nds_chans, host, port)
                self.thread.dataReceived.connect(self._on_data_received)
                self.thread.start()

        # Start Audio Threads if needed
        if audio_chans:
            # Group by device index
            # PC:MIC:[device]-CH[channel] or PC:MIC-CH[channel]
            by_device = {}
            for c in audio_chans:
                dev_idx = None
                if c.startswith("PC:MIC:") or c.startswith("PC:SPEAKER:"):
                    try:
                        dev_str = c.split(":")[2].split("-")[0]
                        dev_idx = int(dev_str)
                    except Exception:
                        pass

                if dev_idx not in by_device:
                    by_device[dev_idx] = []
                by_device[dev_idx].append(c)

            for dev_idx, chans in by_device.items():
                if (
                    dev_idx in self.audio_threads
                    and self.audio_threads[dev_idx].isRunning()
                ):
                    # Update channels if already running?
                    # For simplicity, currently we assume start is called when all channels are set.
                    print(
                        f"NDSDataCache: Audio Thread for device {dev_idx} already running."
                    )
                else:
                    print(
                        f"DEBUG: Starting AudioThread for {chans} on device {dev_idx}"
                    )
                    ath = AudioThread(chans, device_index=dev_idx)
                    ath.dataReceived.connect(self._on_data_received)
                    ath.start()
                    self.audio_threads[dev_idx] = ath

    def online_stop(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait(2000)
            if self.thread.isRunning():
                self.thread.terminate()
            self.thread = None
            print("NDS Online stopped.")

        for dev_idx, ath in list(self.audio_threads.items()):
            ath.stop()
            ath.wait(2000)
            if ath.isRunning():
                ath.terminate()
            print(f"Audio Online stopped for device {dev_idx}.")
        self.audio_threads = {}

    def reset(self):
        self.online_stop()
        self.buffers.reset()
        print("NDS Cache reset.")

    def _on_data_received(self, payload, trend, is_online):
        # Update internal buffers
        print("DEBUG: Cache received data payload")
        self.buffers.update_buffers(payload)
        # Emit updated buffers to GUI
        self.signal_data.emit(self.buffers)
        # Emit incremental payload for streaming analysis
        self.signal_payload.emit(payload)
