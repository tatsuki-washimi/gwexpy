"""
NDS Data Cache management.
Adapts NDSThread and DataBufferDict.
"""
import os
from qtpy import QtCore
from .nds_thread import NDSThread
from .buffer import DataBufferDict
from .util import parse_server_string

class NDSDataCache(QtCore.QObject):
    signal_data = QtCore.Signal(object) # emit(DataBufferDict)

    def __init__(self):
        super().__init__()
        self.channels = []
        self.server = os.getenv('NDSSERVER', 'localhost:31200')
        self.lookback = 30.0
        self.buffers = DataBufferDict(self.lookback)
        self.thread = None
        self.active_tid = None # To prevent multiple threads

    def set_channels(self, channels):
        self.channels = list(set([c for c in channels if c])) # Unique, non-empty

    def set_server(self, server_env):
        self.server = server_env

    def online_start(self, lookback=30.0):
        self.lookback = lookback
        self.buffers.lookback = lookback
        
        if not self.channels:
            print("NDSDataCache: No channels to fetch.")
            return

        if self.thread and self.thread.isRunning():
             print("NDSDataCache: Thread already running.")
             return

        host, port = parse_server_string(self.server)
        
        print(f"DEBUG: Starting NDSThread for {self.channels} on {host}:{port}")
        self.thread = NDSThread(self.channels, host, port)
        self.thread.dataReceived.connect(self._on_data_received)
        self.thread.start()
        print(f"NDS Online started for {self.channels} on {host}:{port}")

    def online_stop(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait(2000) # Wait up to 2s
            if self.thread.isRunning():
                self.thread.terminate() # Force kill if necessary
            self.thread = None
            print("NDS Online stopped.")

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
