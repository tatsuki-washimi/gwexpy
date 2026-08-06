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


@pytest.mark.parametrize("kind", ["series", "matrix"])
@pytest.mark.parametrize("materialize_xindex", [False, True])
def test_crop_large_gps_exact_grid_matches_positional_slice(
    kind: str, materialize_xindex: bool
) -> None:
    """Exact binary64 grid bounds select the same samples as ``[100:600]``.

    This deliberately uses a large GPS epoch and a non-binary spacing.  The
    oracle is positional slicing, not another time-selection method.
    """
    t0 = 1_234_567_890.1234567
    dt = 1 / 30
    values = np.arange(768, dtype=np.float64)
    start = float(t0 + 100 * dt)
    end = float(t0 + 600 * dt)

    if kind == "series":
        source = TimeSeries(values, t0=t0, dt=dt)
        expected = values[100:600]
    else:
        source = TimeSeriesMatrix(values.reshape(1, 1, -1), t0=t0, dt=dt)
        expected = values[100:600].reshape(1, 1, -1)
    if materialize_xindex:
        _ = source.xindex

    cropped = source.crop(start, end)

    np.testing.assert_array_equal(cropped.value, expected)
    assert _float64_bits(float(cropped.t0.value)) == _float64_bits(start)
    assert _float64_bits(float(cropped.dt.value)) == _float64_bits(float(source.dt.value))


@pytest.mark.parametrize("kind", ["series", "matrix"])
def test_crop_floor_rule_clamps_off_grid_and_outside_bounds(kind: str) -> None:
    """Off-grid crop bounds floor to samples and never wrap negative indices."""
    t0 = 1_234_567_890.1234567
    dt = 1 / 30
    values = np.arange(128, dtype=np.float64)
    if kind == "series":
        source = TimeSeries(values, t0=t0, dt=dt)
        expected_shape = values.shape
    else:
        source = TimeSeriesMatrix(values.reshape(1, 1, -1), t0=t0, dt=dt)
        expected_shape = (1, 1, values.size)

    half_sample = source.crop(t0 + 100.5 * dt, t0 + 101.5 * dt)
    np.testing.assert_array_equal(
        half_sample.value, values[100:101].reshape(expected_shape[:-1] + (1,))
    )

    before = source.crop(t0 - 20 * dt, t0 + 1.5 * dt)
    np.testing.assert_array_equal(
        before.value, values[:1].reshape(expected_shape[:-1] + (1,))
    )

    after = source.crop(t0 + 200 * dt, t0 + 220 * dt)
    assert after.shape[-1] == 0
