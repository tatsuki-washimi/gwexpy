from typing import Optional, Union

import numpy as np
import scipy.signal  # noqa: F401 - availability check

from .params import GeneratorParams


class SignalGenerator:
    """
    Generates simulation waveforms based on GeneratorParams.
    Read-Only: Does not interact with hardware.
    """

    def __init__(self):
        # Filter stats cache: {target_channel: {'zi': zi, 'last_t': t, 'sos': sos}}
        self.filter_states = {}

    def generate(self, times: np.ndarray, params: GeneratorParams) -> np.ndarray:
        if not params.enabled:
            return np.zeros_like(times)

        # Waveform generation dispatch
        wtype = params.waveform_type.lower()

        # Common Phase Calculation (t=0 aligned to times[0] or absolute?)
        # For simulation, usually we want continuity?
        # But simple implementation: f(t) where t is the array provided.
        # This means if times slides, phase slides naturally.

        # Adjust time to be relative to start if necessary,
        # but usually 'times' contains absolute or relative timestamps.
        # If 'times' are large GPS, sin(constant * large) loses precision.
        # Ideally we shift t to be 0-based for the math if freq is static,
        # but that resets phase every frame.
        # BETTER: Use (times - times[0]) + internal_offset (if we tracked phase).
        # FOR NOW: Use raw times.
        # Caution: np.sin(large_float) precision issues.
        # Re-center time for calculation: t_local = times - times[0]
        # BUT this resets phase every buffer update in NDS mode.
        # So we MUST use the actual time values to keep phase continuity across visual updates.
        # To avoid precision loss, maybe user subtracts a large epoch, but let's try raw first.

        t = times

        if "none" in wtype or wtype == "none":
            return np.zeros_like(t)

        elif "sine" in wtype:
            # Standard Sine
            phi = np.deg2rad(params.phase)
            return params.offset + params.amplitude * np.sin(
                2 * np.pi * params.frequency * t + phi
            )

        elif "square" in wtype:
            # Phase shift for square?
            phi = np.deg2rad(params.phase)
            return params.offset + params.amplitude * scipy.signal.square(
                2 * np.pi * params.frequency * t + phi
            )

        elif "ramp" in wtype:
            # Sawtooth is essentially a ramp
            phi = np.deg2rad(params.phase)
            return params.offset + params.amplitude * scipy.signal.sawtooth(
                2 * np.pi * params.frequency * t + phi
            )

        elif "triangle" in wtype:
            # Triangle is sawtooth with width=0.5
            phi = np.deg2rad(params.phase)
            return params.offset + params.amplitude * scipy.signal.sawtooth(
                2 * np.pi * params.frequency * t + phi, width=0.5
            )

        elif "impulse" in wtype:
            # Periodic impulse? That's just a pulse train.
            # Or single impulse at t=0?
            # Given it's a test signal in time domain, likely a pulse train if continuous, or single shot.
            # Let's do a repetitive pulse train at Frequency.
            # Using square wave with very small duty cycle?
            phi = np.deg2rad(params.phase)
            # Create a localized pulse if t is near a multiple of 1/f
            period = 1.0 / params.frequency if params.frequency > 0 else 1.0
            (t + phi / (2 * np.pi) * period) % period
            # Impulse width: 1 sample? or fixed percentage? Let's say 1% duty cycle.
            # Impulse = 1 if t within narrow window.
            # Simpler: use scipy.signal.gausspulse or unit impulse?
            # Let's map it to a narrow square wave for visibility in plot since true impulse is 1-sample.
            return (
                params.offset
                + params.amplitude
                * (
                    scipy.signal.square(
                        2 * np.pi * params.frequency * t + phi, duty=0.01
                    )
                    + 1
                )
                / 2
                * 2
            )  # Rescale 0/1 to -1/1? Impulse is usually 0 to Amp.
            # "Impulse" usually means 0 everywhere except peak.
            # Adjusted: 0 to Amp.
            duty = 0.05
            sq = scipy.signal.square(
                2 * np.pi * params.frequency * t + phi, duty=duty
            )  # returns +1 and -1.
            # We want +1 for duty time, 0 otherwise? Or -1?
            # Let's assume bipolar impulse? Or Unipolar?
            # Standard: Unipolar pulse train.
            return params.offset + params.amplitude * (
                (sq > (1 - 2 * duty)).astype(float)
            )

        elif "offset" in wtype:
            return np.full_like(
                t, params.offset + params.amplitude
            )  # Maybe Amp is the offset value here? Or Amp + Offset? usually just Offset. Let's use Amp as the value.

        elif "noise (gauss)" in wtype or "noise (uniform)" in wtype:
            # Base Noise
            if "gauss" in wtype:
                raw_noise = np.random.normal(0, 1, size=len(t))
            else:
                raw_noise = np.random.uniform(-1, 1, size=len(t))

            # Filtering Logic
            # Calculate sample rate
            if len(t) > 1:
                dt = t[1] - t[0]
                fs = 1.0 / dt
            else:
                fs = 16384.0  # Fallback

            nyq = fs / 2.0
            f_low = params.start_freq  # 'Frequency' input
            f_high = params.stop_freq  # 'Freq. Range' input

            # Determine if filtering is required
            # Logic: If f_low > 0 or f_high < nyq (and f_high > f_low), apply filtering.
            # If f_high == 0 or f_high is default (small), maybe user didn't set Range?
            # Convention: If Range (Stop) <= Start, maybe treat as HighPass(Start)?
            # Or if Range == 0, treat as Broadband (if Start==0) or HighPass(Start)?
            # Diaggui: If Range is invalid, usually ignored?
            # Let's assume:
            # Case 1: Start > 0, Stop >= Start -> Bandpass
            # Case 2: Start > 0, Stop <= 0 (or < Start) -> Highpass (Start)
            # Case 3: Start <= 0, Stop > 0 -> Lowpass (Stop)
            # Case 4: Start <= 0, Stop <= 0 -> Broadband (No filter)

            apply_filter = False
            btype = "band"
            Wn: Optional[Union[float, list[float]]] = None

            if f_low > 0:
                if f_high > f_low and f_high < nyq:
                    apply_filter = True
                    btype = "band"
                    Wn = [f_low, f_high]
                elif f_high >= nyq:  # High end open
                    apply_filter = True
                    btype = "high"
                    Wn = f_low
                elif (
                    f_high <= f_low
                ):  # Invalid range or intended HP? Treat as HP > Start
                    apply_filter = True
                    btype = "high"
                    Wn = f_low
            elif f_high > 0 and f_high < nyq:
                # Lowpass
                apply_filter = True
                btype = "low"
                Wn = f_high

            if apply_filter:
                # Generate or Retrieve Filter
                # Key state by channel name
                key = params.target_channel
                if not key:
                    key = "default"

                # Check for continuity to reset state
                if key not in self.filter_states:
                    self.filter_states[key] = {
                        "zi": None,
                        "last_t": -np.inf,
                        "cfg": None,
                    }

                state = self.filter_states[key]

                # Check if config changed or time gap
                current_cfg = (btype, Wn, fs)
                # Reset if time gap > 1.5*dt (approx) or config changed
                time_gap = t[0] - state["last_t"]
                is_continuous = (
                    (abs(time_gap - dt) < dt * 0.5) if len(t) > 1 else False
                )  # simple check

                if state["cfg"] != current_cfg or not is_continuous:
                    # Re-design filter
                    sos = scipy.signal.butter(4, Wn, btype=btype, fs=fs, output="sos")
                    # Initial state
                    zi = scipy.signal.sosfilt_zi(sos)
                    state["sos"] = sos
                    state["zi"] = zi
                    state["cfg"] = current_cfg

                # Apply Filter
                filtered_noise, new_zi = scipy.signal.sosfilt(
                    state["sos"], raw_noise, zi=state["zi"]
                )
                state["zi"] = new_zi
                state["last_t"] = t[-1]

                return params.offset + params.amplitude * filtered_noise

            return params.offset + params.amplitude * raw_noise

        elif "sweep (linear)" in wtype:
            period = 10.0
            t_loc = t % period
            return params.offset + params.amplitude * scipy.signal.chirp(
                t_loc,
                f0=params.start_freq,
                t1=period,
                f1=params.stop_freq,
                method="linear",
            )

        elif "sweep (log)" in wtype:
            period = 10.0
            t_loc = t % period
            # Avoid f0=0 for log chirp
            f0 = max(1e-3, params.start_freq)
            f1 = max(1e-3, params.stop_freq)
            return params.offset + params.amplitude * scipy.signal.chirp(
                t_loc, f0=f0, t1=period, f1=f1, method="logarithmic"
            )

        else:
            return np.zeros_like(t)
