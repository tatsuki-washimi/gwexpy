from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneratorParams:
    """
    UI Parameters for Signal Generator.
    Acts as a bridge between Qt UI and SignalGenerator logic.
    """

    enabled: bool = False
    waveform_type: str = "Sine"  # Sine, Chirp, White Noise, etc.
    amplitude: float = 1.0
    frequency: float = 100.0
    offset: float = 0.0
    phase: float = 0.0  # Degrees

    # Advanced / Specific params
    start_freq: float = 0.0
    stop_freq: float = 100.0

    # Routing
    output_mode: str = "Overlay"  # Overlay or Sum
    target_channel: str = ""  # Name of the channel to sum with (if Sum mode)
    target_channel_b: str = ""  # Not mostly used, but for completeness if needed
