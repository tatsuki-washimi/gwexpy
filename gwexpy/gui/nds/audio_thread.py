"""
Thread for capturing PC Audio using sounddevice.
"""
import time
import numpy as np
try:
    import sounddevice as sd
except ImportError:
    sd = None

from qtpy import QtCore

class AudioThread(QtCore.QThread):
    # Signal to emit received data: (data_dict, trend_type, is_online)
    dataReceived = QtCore.Signal(object, str, bool)
    finished = QtCore.Signal()

    def __init__(self, channels, sample_rate=44100, block_size=4096):
        super().__init__()
        self.channels = channels # e.g. ["PC:MIC-CH0", "PC:MIC-CH1"]
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.running = True
        
    def run(self):
        if sd is None:
            print("AudioThread: sounddevice not found.")
            self.finished.emit()
            return

        print(f"DEBUG: Starting AudioThread for {self.channels} at {self.sample_rate}Hz")
        
        # We only support input (Microphone) for now as "PC:MIC-..."
        # Loopback is harder and platform dependent.
        
        mic_channels = [c for c in self.channels if "PC:MIC" in c]
        spk_channels = [c for c in self.channels if "PC:SPEAKER" in c]
        
        if not mic_channels and not spk_channels:
            print("AudioThread: No PC Audio channels requested.")
            self.finished.emit()
            return

        # Simple implementation: capture from default input device
        # We map PC:MIC-CH0 to device input index 0, PC:MIC-CH1 to 1, etc.
        try:
            # Determine how many channels we need to capture
            max_ch = 0
            for c in mic_channels:
                try:
                    ch_idx = int(c.split("-CH")[-1])
                    max_ch = max(max_ch, ch_idx + 1)
                except:
                    pass
            
            if max_ch == 0 and mic_channels: max_ch = 1

            def callback(indata, frames, time_info, status):
                if status:
                    print(f"AudioThread Status: {status}")
                
                # GPS Epoch: 315964800 is 1980-01-06 (Unix). 
                # Leap seconds since 1980 is 18 (as of now, but let's keep it simple)
                gps_start = time.time() + 315964800 - 18
                
                payload = {}
                for c in self.channels:
                    try:
                        if "PC:MIC" in c:
                            idx = int(c.split("-CH")[-1])
                            if idx < indata.shape[1]:
                                payload[c] = {
                                    'data': indata[:, idx].copy(),
                                    'gps_start': gps_start,
                                    'step': 1.0 / self.sample_rate
                                }
                        elif "PC:SPEAKER" in c:
                            payload[c] = {
                                'data': np.zeros(frames),
                                'gps_start': gps_start,
                                'step': 1.0 / self.sample_rate
                            }
                    except:
                        pass
                
                if payload:
                    self.dataReceived.emit(payload, 'raw', True)

            print(f"DEBUG: Attempting to open InputStream for {max_ch} channels")
            with sd.InputStream(samplerate=self.sample_rate, 
                                blocksize=self.block_size,
                                channels=max_ch, 
                                callback=callback):
                while self.running:
                    sd.sleep(100)
                    
        except Exception as e:
            print(f"AudioThread Error during capture: {e}")
            # If failed (e.g. no devices), just wait and keep thread alive until stopped
            while self.running:
                time.sleep(0.5)
        finally:
            self.finished.emit()

    def stop(self):
        self.running = False
