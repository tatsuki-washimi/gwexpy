"""
NDS Thread for asynchronous data fetching.
Adapted from reference_ndscope.
"""

try:
    import nds2
    _NDS2_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional dependency
    nds2 = None
    _NDS2_IMPORT_ERROR = exc
from qtpy import QtCore


def _nds2_missing_message() -> str:
    base = "nds2 is required for NDS connections. Install nds2-client via conda-forge."
    if _NDS2_IMPORT_ERROR is None:
        return base
    return f"{base} ({_NDS2_IMPORT_ERROR})"


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
        if nds2 is None:
            print(f"NDSThread Error: {_nds2_missing_message()}")
            return
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

                # Emit data only if still running to avoid shutdown crashes
                if self.running:
                    self.dataReceived.emit(payload, "raw", True)

        except Exception as e:
            if self.running: # Only log if not intentional stop
                print(f"NDSThread Error: {e}")
        finally:
            if self.conn:
                try:
                    c = self.conn
                    self.conn = None # Avoid double close
                    c.close()
                except Exception:
                    pass
            self.finished.emit()

    def stop(self):
        self.running = False
        # nds2.iterate is blocking. Closing the connection from another thread
        # is the standard way to unblock it in nds2-client.
        if self.conn:
            try:
                # Assign to local to avoid race condition where self.conn is set to None by run()
                c = self.conn
                if c:
                    c.close()
            except Exception:
                pass


class ChannelListWorker(QtCore.QThread):
    finished = QtCore.Signal(list, str)  # results (list of tuples), error

    def __init__(self, server, port, pattern="*"):
        super().__init__()
        self.server = server
        self.port = port
        self.pattern = pattern

    def run(self):
        if nds2 is None:
            self.finished.emit([], _nds2_missing_message())
            return
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
