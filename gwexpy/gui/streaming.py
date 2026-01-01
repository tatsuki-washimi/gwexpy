import logging
import numpy as np
from gwpy.timeseries import TimeSeries
from collections import deque

logger = logging.getLogger(__name__)

class SpectralAccumulator:
    """
    Accumulates spectral data from streaming TimeSeries chunks.
    Supports Fixed, Infinite, and Exponential averaging.
    """

    def __init__(self):
        self.params = {}
        self.reset()
        
    def reset(self):
        """Clear all accumulated state."""
        self.state = {
            "count": 0,
            "sum_psd_a": None,
            "sum_psd_b": None,
            "sum_csd_ab": None,
            # For Exponential averaging
            "avg_psd_a": None,
            "avg_psd_b": None,
            "avg_csd_ab": None,
        }
        # Buffers for handling overlaps. 
        # Key: channel name, Value: TimeSeries (or deque of chunks)
        # We need to buffer enough data to form one FFT length.
        self.buffers = {} 
        self.is_done = False

    def configure(self, params, active_traces):
        """
        params: dict containing:
          - 'bw': Resolution Bandwidth
          - 'averages': Number of averages (for Fixed)
          - 'avg_type': 'Fixed', 'Infinite', 'Exponential'
          - 'overlap': 0.0 to 1.0 (fraction)
          - 'window': Window function name
        active_traces: list of dicts [{'ch_a': ..., 'ch_b': ...}, ...]
        """
        self.params = params
        self.active_traces = active_traces
        self.reset()
        
        # Pre-calculate stride and FFT length
        bw = params.get("bw", 1.0)
        if bw <= 0: bw = 1.0
        self.fftlength = 1.0 / bw
        
        overlap_frac = params.get("overlap", 0.5)
        self.overlap_sec = self.fftlength * overlap_frac
        self.stride = self.fftlength - self.overlap_sec
        
        # Ensure stride is positive
        if self.stride <= 0:
            self.stride = self.fftlength
            self.overlap_sec = 0

        self.last_segments = {} # Cache for Time Series display

        logger.info(f"SpectralAccumulator configured. FFT={self.fftlength}s, Stride={self.stride}s")
        # for i, t in enumerate(self.active_traces):
        #     print(f"DEBUG: Trace {i} Config: Active={t.get('active')}, ChA='{t.get('ch_a')}', ChB='{t.get('ch_b')}'")

    def add_chunk(self, data_dict):
        """
        Ingest new data.
        data_dict: { channel_name: { 'data': np.array, 'step': dt, 'gps_start': t0 } }
        """
        if self.is_done and self.params.get("avg_type") == "Fixed":
            return

        for ch, packet in data_dict.items():
            if ch not in self.buffers:
                self.buffers[ch] = {
                    "data": [],
                    "dt": packet["step"],
                    "t0": packet["gps_start"],
                    "current_len": 0
                }
            
            buf = self.buffers[ch]
            buf["data"].append(packet["data"])
            buf["current_len"] += len(packet["data"])
            
            # Debug incoming data stats
            d = packet["data"]
            if len(d) > 0:
                print(f"DEBUG: add_chunk {ch}: len={len(d)}, min={d.min()}, max={d.max()}, mean={d.mean()}")

        self._process_buffers()

    def _process_buffers(self):
        if not self.buffers:
            return

        # Assuming all channels synchronized for now.
        any_ch = list(self.buffers.keys())[0]
        buf = self.buffers[any_ch]
        dt = buf["dt"]
        
        required_samples = int(self.fftlength / dt)
        stride_samples = int(self.stride / dt)
        
        total_strides = 0
        
        while buf["current_len"] >= required_samples:
            # We can compute at least one FFT
            
            # Extract segment for all channels
            segment_map = {}
            # Need to be robust if some channels are missing data (lagging)
            ready = True
            for trace in self.active_traces:
                if not trace.get("active", True):
                    continue
                for key in ["ch_a", "ch_b"]:
                    ch = trace.get(key)
                    if ch and (ch not in self.buffers or self.buffers[ch]["current_len"] < required_samples):
                        # print(f"DEBUG: Trace {i} waiting for {ch}. Buf len: {self.buffers.get(ch, {}).get('current_len', 'N/A')}")
                        ready = False
                        break
                if not ready: break
            
            if not ready:
                # Wait for more data
                break

            for ch, b in self.buffers.items():
                # Concatenate current buffer list to array if needed
                # Ideally we only do this once or optimize queue
                if len(b["data"]) > 1:
                    full_arr = np.concatenate(b["data"])
                    b["data"] = [full_arr]
                else:
                    full_arr = b["data"][0]
                
                # Extract segment
                seg_data = full_arr[:required_samples]
                
                # Create TimeSeries for gwpy methods
                # NOTE: t0 is not critical for ASD averaging but good for metadata
                ts = TimeSeries(seg_data, dt=b["dt"], t0=b["t0"])
                segment_map[ch] = ts
                
                # Update buffer: shift by stride
                remaining = full_arr[stride_samples:]
                b["data"] = [remaining]
                b["current_len"] = len(remaining)
                b["t0"] += stride_samples * b["dt"] # Update for next chunk time
            
            # Compute Spectral update
            self._update_spectra(segment_map)
            self.state["count"] += 1
            total_strides += 1
            
            # Check stop condition
            if self.params.get("avg_type") == "Fixed":
                if self.state["count"] >= self.params.get("averages", 10):
                    self.is_done = True
                    return

    def _update_spectra(self, segment_map):
        fft_kwargs = {"fftlength": self.fftlength, "window": self.params.get("window", "hann")}
        
        # Iterate over active traces and compute
        count = self.state["count"] + 1 # 1-based for averaging
        avg_type = self.params.get("avg_type", "Infinite") # Fixed treated same as Infinite while running
        alpha = 0.1 # Default for Exponential? usually 2/(N+1) or specific param
        
        # Cache unique calculations per step to avoid re-computing same PSD for different traces
        
        # Cache unique calculations per step to avoid re-computing same PSD for different traces
        step_cache = {} 
        
        # Cache segments for Time Series display
        for ch, ts in segment_map.items():
            self.last_segments[ch] = ts
            # Debug segment stats
            if len(ts) > 0:
                print(f"DEBUG: last_segment {ch}: min={ts.min()}, max={ts.max()}")

        
        # print(f"DEBUG: _update_spectra. Count={count}. SegMap keys={list(segment_map.keys())}")

        for i, trace in enumerate(self.active_traces):
            if not trace.get("active", True):
                continue
            
            ch_a = trace.get("ch_a")
            ch_b = trace.get("ch_b")
            
            if not ch_a or ch_a not in segment_map:
                # if ch_a: print(f"DEBUG: Trace {i} ch_a '{ch_a}' NOT in segment_map")
                continue

            # Keys for state storage
            key_a = f"{ch_a}"
            key_ab = f"{ch_a}_{ch_b}" if ch_b else None

            # 1. Compute PSD A
            if key_a not in step_cache:
                try:
                    p = segment_map[ch_a].psd(**fft_kwargs)
                    step_cache[key_a] = p
                    print(f"DEBUG: PSD {ch_a}: min={p.min().value}, max={p.max().value}")
                except Exception as e:
                    logger.error(f"PSD Error {ch_a}: {e}")
                    continue
            psd_a_new = step_cache.get(key_a)
            
            if psd_a_new is not None:
                self._accumulate(key_a, psd_a_new, count, avg_type)

            # 2. Compute PSD B if needed
            if ch_b and ch_b in segment_map:
                key_b = f"{ch_b}"
                if key_b not in step_cache:
                    try:
                         step_cache[key_b] = segment_map[ch_b].psd(**fft_kwargs)
                    except: pass
                psd_b_new = step_cache.get(key_b)
                if psd_b_new is not None:
                    self._accumulate(key_b, psd_b_new, count, avg_type)
                
                # 3. Compute CSD AB
                key_csd = f"csd_{key_ab}"
                # gwpy CSD
                try:
                    csd_new = segment_map[ch_a].csd(segment_map[ch_b], **fft_kwargs)
                    self._accumulate(key_csd, csd_new, count, avg_type)
                except: pass

    def _accumulate(self, key, new_val, count, avg_type):
        current_val = self.state.get(key)
        
        # Debug accumulation (checking for NaNs or Zeros)
        # if new_val is not None and key.startswith("K1:"):
        #    v = new_val.value
        #    if v.max() == 0:
        #        print(f"DEBUG: Accumulate {key} - Incoming ZERO PSD")
        
        if current_val is None:
            self.state[key] = new_val
        else:
            if avg_type == "Exponential":
                # params should ideally have 'alpha' or 'averages' to derive alpha
                N = self.params.get("averages", 10)
                alpha = 2.0 / (N + 1.0) # Standard EMA? or 1/N
                # DTT "Exponential" often means "Average" setting implies decay.
                self.state[key] = (1.0 - alpha) * current_val + alpha * new_val
            else:
                # Infinite / Fixed (Cumulative Moving Average)
                # avg_new = avg_old + (new - avg_old) / count
                self.state[key] = current_val + (new_val - current_val) / count

    def get_results(self):
        """
        Return list of result tuples/dicts matching the active_traces structure.
        Compatible with Engine.compute output format.
        """
        results = []
        # print("DEBUG: get_results state keys:", list(self.state.keys()))
        for i, trace in enumerate(self.active_traces):
            if not trace.get("active", True):
                results.append(None)
                continue
            
            ch_a = trace.get("ch_a")
            ch_b = trace.get("ch_b")
            graph_type = trace.get("graph_type", self.params.get("graph_type", "Amplitude Spectral Density"))
            
            res = None
            key_a = f"{ch_a}"
            # key_b = f"{ch_b}"
            # key_csd = f"csd_{ch_a}_{ch_b}"
            
            # Special handling for "Time Series" - do not require PSD
            if graph_type == "Time Series":
                res = self.last_segments.get(key_a)
                if res is not None:
                    results.append((res.times.value, res.value))
                else:
                    results.append(None)
                continue
            
            psd_a = self.state.get(key_a)
            # psd_b = self.state.get(key_b) if ch_b else None
            # csd_ab = self.state.get(key_csd) if ch_b else None
            
            if psd_a is None:
                # print(f"DEBUG: Res {i}: psd_a is None for '{key_a}'")
                results.append(None)
                continue
            
            # ... rest is same
            # To save tokens, I will just uncomment critical prints in next steps or include logic here.
            # Wait, I am replacing the function. I must include the logic.
            
            key_b = f"{ch_b}"
            key_csd = f"csd_{ch_a}_{ch_b}"
            psd_b = self.state.get(key_b) if ch_b else None
            csd_ab = self.state.get(key_csd) if ch_b else None

            try:
                # print(f"DEBUG: Trace {i} GraphType: '{graph_type}' KeyA: '{key_a}'")
                if "Amplitude Spectral Density" in graph_type or "Power Spectral Density" in graph_type:
                    # ASD = sqrt(PSD)
                    res = psd_a ** 0.5 if "Amplitude" in graph_type else psd_a
                
                elif graph_type == "Coherence":
                    if psd_a is not None and psd_b is not None and csd_ab is not None:
                        res = (csd_ab.abs() ** 2) / (psd_a * psd_b)
                        
                elif graph_type == "Transfer Function":
                    if psd_a is not None and csd_ab is not None:
                         # H = P_ba / P_aa = conj(P_ab) / P_aa
                        res = csd_ab.conjugate() / psd_a
                        
                elif graph_type == "Cross Spectral Density":
                    res = csd_ab
                
                if res is not None:
                    # Crop frequency
                    start_f = self.params.get("start_freq", 0)
                    stop_f = self.params.get("stop_freq", 1000)
                    res = res.crop(start_f, stop_f)
                    results.append((res.frequencies.value, res.value))
                else:
                    # print(f"DEBUG: Res {i} is None after calculation. GraphType: '{graph_type}'")
                    results.append(None)

            except Exception as e:
                logger.error(f"Error computing result for {ch_a}: {e}")
                results.append(None)
                
        return results
