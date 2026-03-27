"""gwexpy.noise.line_mask - Line noise masking tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..spectrogram import Spectrogram


# Default KAGRA lines (approximate ranges for masking)
KAGRA_LINES = [
    (59.9, 60.1),    # Power line
    (119.8, 120.2),  # 2nd harmonic
    (179.7, 180.3),  # 3rd harmonic
    (239.6, 240.4),  # 4th harmonic
    (136.0, 137.5),  # Calibration/Control lines
    (400.0, 401.0),  # Example high-freq line
]


def create_line_mask(
    frequencies: np.ndarray,
    detector: str = "KAGRA",
    custom_lines: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Create a boolean mask for line noise exclusion.
    True = clean frequency, False = masked frequency.
    """
    mask = np.ones(len(frequencies), dtype=bool)
    
    lines = custom_lines if custom_lines is not None else []
    if detector == "KAGRA":
        lines.extend(KAGRA_LINES)
        
    for fmin, fmax in lines:
        in_range = (frequencies >= fmin) & (frequencies <= fmax)
        mask[in_range] = False
        
    return mask


def apply_line_mask(
    spectrogram: Spectrogram,
    detector: str = "KAGRA",
    custom_lines: list[tuple[float, float]] | None = None,
    fill_value: float = np.nan,
) -> Spectrogram:
    """
    Apply a line mask to a spectrogram.
    Masked frequencies are replaced with fill_value.
    """
    mask = create_line_mask(spectrogram.frequencies.value, detector, custom_lines)
    
    new_value = spectrogram.value.copy()
    new_value[:, ~mask] = fill_value
    
    return spectrogram.__class__(
        new_value,
        times=spectrogram.times,
        frequencies=spectrogram.frequencies,
        unit=spectrogram.unit,
        name=f"{spectrogram.name}_masked" if spectrogram.name else "masked",
    )
