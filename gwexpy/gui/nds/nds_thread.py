"""
NDS Thread for asynchronous data fetching.
Adapted from reference_ndscope.
"""

import nds2
from qtpy import QtCore


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
                        "data": b.data,
                        "gps_start": b.gps_seconds + b.gps_nanoseconds * 1e-9,
                        "step": 1.0 / b.channel.sample_rate,
                    }

                # Emit data
                print(f"DEBUG: NDSThread received {len(payload)} channels")
                self.dataReceived.emit(payload, "raw", True)

        except Exception as e:
            print(f"NDSThread Error: {e}")
        finally:
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
            self.finished.emit()

    def stop(self):
        self.running = False
        # nds2.iterate is blocking, so we might need to rely on the iterator checking
        # or closing connection from outside if possible, but standard iteration usually checks flag.
        # If it blocks indefinitely waiting for data, we might hang here.
        # For Phase 1, we rely on the loop checking self.running after each buffer.


class ChannelListWorker(QtCore.QThread):
    finished = QtCore.Signal(list, str)  # results (list of tuples), error

    def __init__(self, server, port, pattern="*"):
        super().__init__()
        self.server = server
        self.port = port
        self.pattern = pattern

    def run(self):
        try:
            conn = nds2.connection(self.server, self.port)
            # Find channels with details
            # nds2.find_channels returns list of channel objects.
            # We want name, rate, type.
            channels = conn.find_channels(self.pattern)

            # Convert to list of tuples: (name, rate, type)
            results = []
            for c in channels:
                # Filter trends (simple heuristic from existing code if needed,
                # but typically we want raw channels mostly)
                name = c.name
                if "," in name and "-trend" in name:
                    continue

                # c.channel_type is an integer enum usually.
                # c.sample_rate is float.
                results.append((c.name, c.sample_rate, c.channel_type))

            # Sort by name
            results.sort(key=lambda x: x[0])

            self.finished.emit(results, "")

        except Exception as e:
            self.finished.emit([], str(e))

