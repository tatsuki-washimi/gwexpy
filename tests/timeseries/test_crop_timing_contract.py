"""Bit-exact timing contracts for cropped time-domain objects (#617)."""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix


def _float64_bits(value: float) -> np.uint64:
    """Return the IEEE-754 payload so a one-ulp regression cannot hide."""
    return np.float64(value).view(np.uint64)


@pytest.mark.parametrize("kind", ["series", "matrix"])
def test_crop_preserves_sample_spacing_bit_exactly_after_xindex_materialization(
    kind: str,
) -> None:
    """Crop selects samples without re-deriving the regular time spacing."""
    values = np.arange(128, dtype=np.float64)
    if kind == "series":
        source = TimeSeries(values, t0=0.0, dt=0.1)
        expected_values = values[2:8]
    else:
        source = TimeSeriesMatrix(values.reshape(1, 1, -1), t0=0.0, dt=0.1)
        expected_values = values[2:8].reshape(1, 1, -1)

    original_dt = float(source.dt.value)
    _ = source.xindex
    cropped = source.crop(0.2, 0.8)

    assert _float64_bits(float(cropped.dt.value)) == _float64_bits(original_dt)
    np.testing.assert_array_equal(cropped.value, expected_values)
    assert float(cropped.t0.value) == 0.2
