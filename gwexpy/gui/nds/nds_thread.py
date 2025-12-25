"""
NDS Thread for asynchronous data fetching.
Adapted from reference_ndscope.
"""
import nds2
from qtpy import QtCore
from .util import parse_server_string

class NDSThread(QtCore.QThread):
    # Signal to emit received data: (data_dict, trend_type, is_online)
    dataReceived = QtCore.Signal(object, str, bool)
    finished = QtCore.Signal()

    def __init__(self, channels, server, port=31200):
        super().__init__()
        self.channels = channels
        self.server = server
        self.port = port
        self.running = True
        self.conn = None

    def run(self):
        try:
            self.conn = nds2.connection(self.server, self.port)
            # Phase 1: RAW Online
            for bufs in self.conn.iterate(self.channels):
                if not self.running:
                    break
                
                payload = {}
                for b in bufs:
                    # NDS2 buffer to dict
                    payload[b.channel.name] = {
                        'data': b.data,
                        'gps_start': b.gps_seconds + b.gps_nanoseconds * 1e-9,
                        'step': 1.0 / b.channel.sample_rate
                    }
                
                # Emit data
                print(f"DEBUG: NDSThread received {len(payload)} channels") 
                self.dataReceived.emit(payload, 'raw', True)
                
        except Exception as e:
            print(f"NDSThread Error: {e}")
        finally:
             if self.conn:
                 try:
                     self.conn.close()
                 except:
                     pass
             self.finished.emit()

    def stop(self):
        self.running = False
        # nds2.iterate is blocking, so we might need to rely on the iterator checking 
        # or closing connection from outside if possible, but standard iteration usually checks flag.
        # If it blocks indefinitely waiting for data, we might hang here.
        # For Phase 1, we rely on the loop checking self.running after each buffer.
