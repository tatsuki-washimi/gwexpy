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

class NDSDataCache(QtCore.QObject):
    signal_data = QtCore.Signal(object) # emit(DataBufferDict)

    def __init__(self):
        super().__init__()
        self.channels = []
        self.server = os.getenv('NDSSERVER', 'localhost:31200')
        self.lookback = 30.0
        self.buffers = DataBufferDict(self.lookback)
        self.thread = None
        self.audio_thread = None
        self.active_tid = None 

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
        
        # Start Audio Thread if needed
        if audio_chans:
            if self.audio_thread and self.audio_thread.isRunning():
                print("NDSDataCache: Audio Thread already running.")
            else:
                print(f"DEBUG: Starting AudioThread for {audio_chans}")
                self.audio_thread = AudioThread(audio_chans)
                self.audio_thread.dataReceived.connect(self._on_data_received)
                self.audio_thread.start()

    def online_stop(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait(2000)
            if self.thread.isRunning(): self.thread.terminate()
            self.thread = None
            print("NDS Online stopped.")
        
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait(2000)
            if self.audio_thread.isRunning(): self.audio_thread.terminate()
            self.audio_thread = None
            print("Audio Online stopped.")

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
