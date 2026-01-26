import logging
from collections import deque

import numpy as np
from gwpy.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class SpectralAccumulator:
    """
    Accumulates spectral data from streaming TimeSeries chunks.
    Supports Fixed, Infinite, and Exponential averaging.
    """

    def __init__(self):
        self.params = {}
        self.buffers = {}
        self.common_t0 = None  # Reference start time for synchronization
        self.display_history = {}  # {ch: deque(TimeSeries, maxlen=...)}
        self.spectrogram_history = {}  # {ch: deque(row, maxlen=...)}
        self.active_traces = []
        self.state = {}
        self.is_done = False
        logger.info("SpectralAccumulator state reset.")
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

    def configure(self, params, active_traces, available_channels=None):
        """
        params: dict containing:
          - 'bw': Resolution Bandwidth
          - 'averages': Number of averages (for Fixed)
          - 'avg_type': 'Fixed', 'Infinite', 'Exponential'
          - 'overlap': 0.0 to 1.0 (fraction)
          - 'window': Window function name
        active_traces: list of dicts [{'ch_a': ..., 'ch_b': ...}, ...]
        available_channels: list/set of channels that are expected to be streamed.
        """
        # Normalize window name
        window = params.get("window", "hann").lower()
        if window in ["hanning", "hann"]:
            window = "hann"
        elif window in ["uniform", "boxcar"]:
            window = "boxcar"
        params["window"] = window
        self.params = params
        self.active_traces = active_traces
        self.available_channels = (
            set(available_channels) if available_channels is not None else None
        )
        self.reset()

        # Pre-calculate stride and FFT length
        bw = params.get("bw", 1.0)
        if bw <= 0:
            bw = 1.0
        self.fftlength = 1.0 / bw

        overlap_frac = params.get("overlap", 0.5)
        self.overlap_sec = self.fftlength * overlap_frac
        self.stride = self.fftlength - self.overlap_sec

        # Ensure stride is positive
        if self.stride <= 0:
            self.stride = self.fftlength
            self.overlap_sec = 0

        self.last_segments = {}  # Cache for Time Series display

        logger.info(
            f"SpectralAccumulator configured. FFT={self.fftlength}s, Stride={self.stride}s"
        )
        # for i, t in enumerate(self.active_traces):
        #     print(f"DEBUG: Trace {i} Config: Active={t.get('active')}, ChA='{t.get('ch_a')}', ChB='{t.get('ch_b')}'")

    def add_chunk(self, data_dict):
        """
        Ingest new data.
        data_dict: { channel_name: { 'data': np.array, 'step': dt, 'gps_start': t0 } }
        """
        try:
            if self.is_done and self.params.get("avg_type") == "Fixed":
                return

            if not data_dict:
                return

            # Initialize common_t0 from the first packet we see
            if self.common_t0 is None:
                first_ch = list(data_dict.keys())[0]
                self.common_t0 = data_dict[first_ch]["gps_start"]
                logger.info(
                    f"SpectralAccumulator starting alignment at t0={self.common_t0}"
                )

            for ch, packet in data_dict.items():
                t0 = packet["gps_start"]
                dt = packet["step"]
                data = packet["data"]

                # 1. Time Alignment
                # If packet is ahead of common_t0, drop samples
                if t0 < self.common_t0:
                    diff_sec = self.common_t0 - t0
                    drop_samples = int(round(diff_sec / dt))
                    if drop_samples >= len(data):
                        continue
                    data = data[drop_samples:]
                    t0 = self.common_t0
                # (If t0 > common_t0, the buffer just correctly reflects the delay)

                # 2. Update Processing Buffer
                if ch not in self.buffers:
                    self.buffers[ch] = {
                        "data": [],
                        "dt": dt,
                        "t0": t0,
                        "current_len": 0,
                    }

                buf = self.buffers[ch]
                buf["data"].append(data)
                buf["current_len"] += len(data)

            self._process_buffers()

        except (KeyError, TypeError, ValueError):
            logger.exception("Error in add_chunk")

    def _process_buffers(self):
        """
        Process buffers synchronously.
        Only proceed if ALL available channels have enough data for a segment.
        """
        try:
            if not self.buffers:
                return

            # Which channels do we need?
            targets = (
                self.available_channels
                if self.available_channels
                else set(self.buffers.keys())
            )

            # If some target channel has not even appeared in buffers yet, we must wait.
            for ch in targets:
                if ch not in self.buffers:
                    return

            # Get dt from any available buffer
            any_ch = list(targets)[0]
            dt = self.buffers[any_ch]["dt"]

            # Guard against invalid dt
            if dt <= 0:
                logger.error(f"Invalid dt={dt} detected in buffer for {any_ch}")
                return

            required_samples = int(self.fftlength / dt)
            stride_samples = int(self.stride / dt)

            if required_samples <= 0 or stride_samples <= 0:
                logger.warning(
                    f"Insufficient samples for FFT: req={required_samples}, stride={stride_samples}. Check bw/fftlength."
                )
                return

            # Process each segment step by step for ALL channels
            processed_any = True
            while processed_any:
                processed_any = False

                # Check if ALL targets are ready for the next segment
                all_ready = True
                for ch in targets:
                    if self.buffers[ch]["current_len"] < required_samples:
                        all_ready = False
                        break

                if not all_ready:
                    break

                segment_map = {}

                # Extract segments for all targets
                for ch in targets:
                    b = self.buffers[ch]

                    # Concatenate buffer data if needed
                    if len(b["data"]) > 1:
                        full_arr = np.concatenate(b["data"])
                        b["data"] = [full_arr]
                    else:
                        full_arr = b["data"][0]

                    # Extract segment
                    seg_data = full_arr[:required_samples]

                    # Create TimeSeries
                    ts = TimeSeries(seg_data, dt=b["dt"], t0=b["t0"])
                    segment_map[ch] = ts

                    # Update Display History (Deque) - Synchronized
                    if ch not in self.display_history:
                        # Default to 30s history if nds_win not in params
                        win_sec = self.params.get("nds_win", 30.0)
                        maxlen = int(win_sec / b["dt"])
                        self.display_history[ch] = deque(maxlen=maxlen)

                    self.display_history[ch].extend(seg_data)

                    # Update buffer: shift by stride
                    remaining = full_arr[stride_samples:]
                    b["data"] = [remaining]
                    b["current_len"] = len(remaining)
                    b["t0"] += stride_samples * b["dt"]

                # If we extracted segments, update spectra
                if segment_map:
                    self._update_spectra(segment_map)
                    self.state["count"] += 1
                    processed_any = True

                    # Check stop condition for Fixed averaging
                    if self.params.get("avg_type") == "Fixed":
                        if self.state["count"] >= self.params.get("averages", 10):
                            self.is_done = True
                            return

        except (KeyError, TypeError, ValueError, RuntimeError):
            logger.exception("Error in _process_buffers")

    def _update_spectra(self, segment_map):
        fft_kwargs = {
            "fftlength": self.fftlength,
            "window": self.params.get("window", "hann"),
        }

        # Iterate over active traces and compute
        count = self.state["count"] + 1  # 1-based for averaging
        avg_type = self.params.get(
            "avg_type", "Infinite"
        )  # Fixed treated same as Infinite while running

        # Cache unique calculations per step to avoid re-computing same PSD for different traces

        # Cache unique calculations per step to avoid re-computing same PSD for different traces
        step_cache = {}

        # Cache segments for display/calculation
        for ch, ts in segment_map.items():
            self.last_segments[ch] = ts

        for i, trace in enumerate(self.active_traces):
            if not trace.get("active", True):
                continue

            ch_a = trace.get("ch_a")
            ch_b = trace.get("ch_b")

            if not ch_a or ch_a not in segment_map:
                # if ch_a: print(f"DEBUG: Trace {i} ch_a '{ch_a}' NOT in segment_map")
                continue

            # Keys for state storage (unique per channel)
            # We accumulate RAW PSD/CSD (gain=1.0) to allow shared computation,
            # and apply gain during get_results for flexibility.
            key_a = f"{ch_a}"
            key_ab = f"{ch_a}_{ch_b}" if ch_b else None

            # 1. Compute PSD A
            if key_a not in step_cache:
                try:
                    ts_a = segment_map[ch_a]
                    p = ts_a.psd(**fft_kwargs)
                    step_cache[key_a] = p
                except (RuntimeError, ValueError) as e:
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
                        ts_b = segment_map[ch_b]
                        step_cache[key_b] = ts_b.psd(**fft_kwargs)
                    except (RuntimeError, ValueError):
                        pass
                psd_b_new = step_cache.get(key_b)
                if psd_b_new is not None:
                    self._accumulate(key_b, psd_b_new, count, avg_type)

                # 3. Compute CSD AB
                key_csd = f"csd_{key_ab}"
                try:
                    ts_a = segment_map[ch_a]
                    ts_b = segment_map[ch_b]
                    csd_new = ts_a.csd(ts_b, **fft_kwargs)
                    self._accumulate(key_csd, csd_new, count, avg_type)
                except (RuntimeError, ValueError):
                    pass

            # 4. Handle Spectrogram Updates (Per Segment)
            # Check if this trace requires Spectrogram
            graph_type = trace.get(
                "graph_type",
                self.params.get("graph_type", "Amplitude Spectral Density"),
            )
            if graph_type == "Spectrogram":
                # Only if we have new data for ch_a
                if ch_a in segment_map:
                    try:
                        # Compute single-segment ASD (Magnitude)
                        # Use cached PSD if available? No, PSD is average.
                        # We need instantaneous ASD for this segment.
                        # But wait, step_cache['key_a'] is `ts_a.psd()`. `psd` is periodogram of this segment.
                        # So sqrt(psd) is ASD of this segment.
                        # This avoids re-computing FFT.

                        # However, psd() returns Power. Spectrogram usually displays Amplitude or Power.
                        # We store Amplitude for consistency with existing logic (asd function).
                        # p = |FFT|^2 / const.  asd = sqrt(p).

                        current_psd = step_cache.get(key_a)
                        if current_psd is None:
                            # Should have been computed above
                            ts_a = segment_map[ch_a]
                            current_psd = ts_a.psd(**fft_kwargs)
                            step_cache[key_a] = current_psd

                        # Convert to ASD (Magnitude)
                        # We use gwpy's build-in or just sqrt.
                        # `current_psd` is a FrequencySeries.
                        spec_obj = current_psd**0.5
                        # Correct unit? psd is unit^2/Hz. sqrt is unit/sqrt(Hz).
                        # This matches standard ASD.

                        # Crop frequency range (store only relevant part to save memory?)
                        start_f = self.params.get("start_freq", 0)
                        stop_f = self.params.get("stop_freq", 1000)
                        spec_obj = spec_obj.crop(start_f, stop_f)

                        # Update History
                        if ch_a not in self.spectrogram_history:
                            self.spectrogram_history[ch_a] = deque(maxlen=200)

                        # Use the segment's central time or start time?
                        # `ts_a.t0` is start.
                        t_val = ts_a.t0.value + ts_a.duration.value / 2.0

                        self.spectrogram_history[ch_a].append(
                            {
                                "t": t_val,
                                "v": spec_obj.value,
                                "f": spec_obj.frequencies.value,
                            }
                        )
                    except (AttributeError, RuntimeError, ValueError) as e:
                        logger.error(f"Spectrogram Update Error {ch_a}: {e}")

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
                alpha = 2.0 / (N + 1.0)  # Standard EMA? or 1/N
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
            graph_type = trace.get(
                "graph_type",
                self.params.get("graph_type", "Amplitude Spectral Density"),
            )

            res = None
            key_a = f"{ch_a}"
            # key_b = f"{ch_b}"
            # key_csd = f"csd_{ch_a}_{ch_b}"

            # Special handling for "Time Series" - return history
            if graph_type == "Time Series":
                history = self.display_history.get(ch_a)
                if history:
                    data = np.array(history)
                    # Calculate timestamps based on common_t0 or first sample in history
                    # For simplicity, we assume dt is constant.
                    # Actually, better to use TimeSeries for consistency.
                    # If common_t0 is 100 and we have 1000 samples at 1000Hz,
                    # times go from 100 to 101.
                    # But if we have been running for a long time, we need to know
                    # where the CURRENT deque starts.
                    # We can track 'samples_pushed' or just use the buffer's t0
                    # (which shifts in _process_buffers)

                    # Safer approach: use the latest buffer's t0 and count back?
                    # No, deque popleft() makes it hard.
                    # Let's just return what we have as a tuple (times, values).
                    buf_ref = self.buffers[ch_a]
                    t_end = buf_ref["t0"] + buf_ref["current_len"] * buf_ref["dt"]
                    t_start = t_end - len(data) * buf_ref["dt"]
                    times = np.linspace(t_start, t_end, len(data), endpoint=False)
                    results.append((times, data))
                else:
                    results.append(None)
                continue

            # Special handling for "Spectrogram" - return history
            if graph_type == "Spectrogram":
                if (
                    ch_a in self.spectrogram_history
                    and len(self.spectrogram_history[ch_a]) > 0
                ):
                    hist = list(self.spectrogram_history[ch_a])
                    times = np.array([h["t"] for h in hist])
                    # mask checks that hist is not empty done above
                    values = np.stack([h["v"] for h in hist])  # (Time, Freq)
                    freqs = hist[0]["f"]

                    results.append(
                        {
                            "type": "spectrogram",
                            "value": values,
                            "times": times,
                            "freqs": freqs,
                        }
                    )
                else:
                    results.append(None)
                continue

            psd_a = self.state.get(key_a)
            # Apply Gain (power) to PSD/CSD
            gain = trace.get("gain", 1.0)
            if psd_a is not None and gain != 1.0:
                psd_a = psd_a * (gain**2)

            if psd_a is None:
                # print(f"DEBUG: Res {i}: psd_a is None for '{key_a}'")
                results.append(None)
                continue

            key_b = f"{ch_b}"
            key_csd = f"csd_{ch_a}_{ch_b}"
            psd_b = self.state.get(key_b) if ch_b else None
            csd_ab = self.state.get(key_csd) if ch_b else None

            if gain != 1.0:
                if psd_b is not None:
                    psd_b = psd_b * (gain**2)
                if csd_ab is not None:
                    csd_ab = csd_ab * (gain**2)

            try:
                # print(f"DEBUG: Trace {i} GraphType: '{graph_type}' KeyA: '{key_a}'")
                res = None
                if (
                    "Amplitude Spectral Density" in graph_type
                    or "Power Spectral Density" in graph_type
                ):
                    # ASD = sqrt(PSD)
                    res = psd_a**0.5 if "Amplitude" in graph_type else psd_a

                elif graph_type == "Coherence":
                    if psd_a is not None and psd_b is not None and csd_ab is not None:
                        # Linear Coherence = sqrt(MSC)
                        # Gain cancels out in Coherence
                        res = ((csd_ab.abs() ** 2) / (psd_a * psd_b)) ** 0.5

                elif graph_type == "Squared Coherence":
                    if psd_a is not None and psd_b is not None and csd_ab is not None:
                        # Magnitude Squared Coherence (MSC)
                        # Gain cancels out in MSC
                        res = (csd_ab.abs() ** 2) / (psd_a * psd_b)

                elif graph_type == "Transfer Function":
                    if psd_a is not None and csd_ab is not None:
                        # H = P_ba / P_aa = conj(P_ab) / P_aa
                        # If both A and B are scaled by G, TF is unchanged.
                        # In gwexpy GUI, Gain is trace-specific. We assume it's applied to the signal.
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

            except (RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Error computing result for {ch_a}: {e}")
                results.append(None)

        return results
