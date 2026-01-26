import logging

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self):
        self.params = {}
        # Stores accumulation buffers or state if needed

    def reset(self):
        """Reset the engine state and parameters."""
        self.params = {}
        self.state = {}
        logger.info("Engine state reset.")

    def configure(self, params):
        """
        Update local parameters from the GUI.
        params expected keys:
          - 'start_freq'
          - 'stop_freq'
          - 'bw' (Resolution Bandwidth)
          - 'averages'
          - 'window'
          - 'overlap'
          - 'avg_type'
        """
        self.params = params

    def _get_fft_kwargs(self, sample_rate):
        """
        Calculate gwpy-compatible FFT arguments from DTT parameters.
        DTT defines Resolution BW = fs / N_fft (roughly, window dependent).
        gwpy .asd() takes 'fftlength' in seconds.
        fftlength = 1 / bw
        """
        bw = self.params.get("bw", 1.0)
        window = self.params.get("window", "hanning")
        overlap = self.params.get("overlap", 0.5)  # Fractional

        if window == "uniform":
            window = "boxcar"
        if window == "hanning":
            window = "hann"

        if bw <= 0:
            bw = 1.0
        fftlength = 1.0 / bw

        return {
            "fftlength": fftlength,
            "overlap": overlap * fftlength,  # gwpy takes overlap in seconds
            "window": window,
        }

    def compute(self, data_map, graph_type, active_traces):
        """
        Perform spectral calculations.

        data_map: { channel_name: TimeSeries }
        graph_type: str (e.g., "Amplitude Spectral Density", "Coherence")
        active_traces: list of dicts [{'active': bool, 'ch_a': str, 'ch_b': str}, ...]
        """
        results = []

        if not data_map:
            return results

        # Assuming all TimeSeries have same sample_rate for now
        sample_rate = list(data_map.values())[0].sample_rate.value
        fft_kwargs = self._get_fft_kwargs(sample_rate)

        fftlength_sec = fft_kwargs.get("fftlength", 1.0)
        min_samples = int(fftlength_sec * sample_rate)

        start_f = self.params.get("start_freq", 0)
        stop_f = self.params.get("stop_freq", 1000)

        for i, trace in enumerate(active_traces):
            if not trace["active"]:
                results.append(None)
                continue

            ch_a = trace["ch_a"]
            ch_b = trace["ch_b"]

            ts_a = data_map.get(ch_a)
            ts_b = data_map.get(ch_b)

            if ts_a is None:
                results.append(None)
                continue

            # Check length for spectrum calculations
            if graph_type in [
                "Amplitude Spectral Density",
                "Power Spectral Density",
                "Coherence",
                "Squared Coherence",
                "Transfer Function",
                "Cross Spectral Density",
                "Spectrogram",
            ]:
                if len(ts_a) < min_samples:
                    # Not enough data yet
                    results.append(None)
                    continue
                if ts_b is not None and len(ts_b) < min_samples:
                    results.append(None)
                    continue

            # Apply Gain (Calibration)
            gain = trace.get("gain", 1.0)
            if gain != 1.0:
                ts_a = ts_a * gain
                if ts_b is not None:
                    ts_b = ts_b * gain

            try:
                # Calculation logic
                # NOTE: DTT's "Power Spectrum" is actually ASD.
                if (
                    graph_type == "Amplitude Spectral Density"
                    or graph_type == "Power Spectral Density"
                ):
                    # We compute ASD regardless of the label, as requested.
                    spec = ts_a.asd(**fft_kwargs)

                elif graph_type == "Coherence":
                    if ts_b is None:
                        spec = None
                    else:
                        # gwpy coherence() returns MSC. Linear coherence is sqrt(MSC).
                        spec = ts_a.coherence(ts_b, **fft_kwargs) ** 0.5

                elif graph_type == "Squared Coherence":
                    if ts_b is None:
                        spec = None
                    else:
                        # Magnitude Squared Coherence (MSC)
                        spec = ts_a.coherence(ts_b, **fft_kwargs)

                elif graph_type == "Transfer Function":
                    if ts_b is None:
                        spec = None
                    else:
                        spec = ts_b.transfer_function(
                            ts_a, **fft_kwargs
                        )  # TF from A to B usually

                elif graph_type == "Cross Spectral Density":
                    if ts_b is None:
                        spec = None
                    else:
                        spec = ts_a.csd(ts_b, **fft_kwargs)

                # Time Series
                elif graph_type == "Time Series":
                    # Just return time vs amplitude
                    results.append((ts_a.times.value, ts_a.value))
                    continue

                elif graph_type == "Spectrogram":
                    if ts_a is None:
                        spec = None
                    else:
                        # stride: segment length in seconds.
                        # Use fftlength for segment length, and overlap for overlapping.
                        # stride in gwpy.spectrogram is strict separation between start of consecutive FFTs?
                        # gwpy spectrogram(stride, fftlength=..., overlap=...)
                        # If overlap is given, stride is calculated? No...
                        # "stride" argument: "step size between FFTs in seconds".
                        # If we want overlap: step = length - overlap_sec

                        length = fft_kwargs["fftlength"]
                        ovlap = fft_kwargs.pop(
                            "overlap", 0
                        )  # Remove overlap from kwargs to avoid conflict/confusion

                        # Use spectrogram2 which supports overlap correctly.
                        # spectrogram2(fftlength, overlap=..., window=...)
                        # Note: spectrogram2 usually returns a Spectrogram object similar to the others.

                        length = fft_kwargs["fftlength"]
                        # overlap was removed from fft_kwargs in _get_fft_kwargs but stored in ovlap variable above?
                        # Wait, in compute() we popped overlap into ovlap.

                        # Re-construct overlap argument if needed, or pass directly.
                        # spectrogram2 takes 'overlap' in seconds (default 0).

                        # fft_kwargs has 'window', 'fftlength'.
                        # We need to ensure we call it correctly.

                        # Remove 'fftlength' from kwargs if we pass it as positional args?
                        # spectrogram2 signature: spectrogram2(fftlength, overlap=None, window=None, ...)

                        kw = fft_kwargs.copy()
                        if "fftlength" in kw:
                            del kw["fftlength"]

                        # ovlap is in seconds (calculated in _get_fft_kwargs as overlap * fftlength)
                        spec = ts_a.spectrogram2(length, overlap=ovlap, **kw)

                        # Crop Frequency (Y-axis)
                        # Spectrogram.crop() acts on Time (X-axis).
                        # To crop Frequency, we interpret it as array slicing.
                        try:
                            # Try crop_frequencies if available (some versions)
                            if hasattr(spec, "crop_frequencies"):
                                spec = spec.crop_frequencies(start_f, stop_f)
                            else:
                                # Manual slicing
                                freqs = spec.frequencies.value
                                mask = (freqs >= start_f) & (freqs <= stop_f)
                                spec = spec[:, mask]
                        except (AttributeError, IndexError, TypeError, ValueError) as e:
                            logger.error(f"Spectrogram crop failed: {e}")
                            # Fallback: return full

                        # Return results
                        results.append(
                            {
                                "type": "spectrogram",
                                "times": spec.times.value,
                                "freqs": spec.frequencies.value,
                                "value": spec.value,
                            }
                        )
                    continue

                # Crop Frequency
                if spec is not None:
                    spec = spec.crop(start_f, stop_f)
                    results.append((spec.frequencies.value, spec.value))
                else:
                    results.append(None)

            except (RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Calculation Error Trace {i}: {e}")
                results.append(None)

        return results
