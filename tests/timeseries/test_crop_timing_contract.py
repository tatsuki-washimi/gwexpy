"""Bit-exact timing contracts for cropped time-domain objects (#617)."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

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
        reference = GwpyTimeSeries(values, t0=0.0, dt=0.1)
        _ = reference.xindex
        expected = reference.crop(0.2, 0.8)
    else:
        source = TimeSeriesMatrix(values.reshape(1, 1, -1), t0=0.0, dt=0.1)
        expected = source[..., 2:8]

    expected_dt = (
        float(expected.dt.value) if kind == "series" else float(source.dt.value)
    )
    _ = source.xindex
    cropped = source.crop(0.2, 0.8)

    assert _float64_bits(float(cropped.dt.value)) == _float64_bits(expected_dt)
    np.testing.assert_array_equal(cropped.value, expected.value)
    assert float(cropped.t0.value) == float(expected.t0.value)


@pytest.mark.parametrize("kind", ["series", "matrix"])
@pytest.mark.parametrize("materialize_xindex", [False, True])
def test_crop_large_gps_exact_grid_matches_positional_slice(
    kind: str, materialize_xindex: bool
) -> None:
    """Exact binary64 grid bounds select the same samples as ``[100:600]``.

    This deliberately uses a large GPS epoch and a non-binary spacing.  GWpy
    is the compatibility oracle for the public ``TimeSeries.crop`` contract.
    """
    t0 = 1_234_567_890.1234567
    dt = 1 / 30
    values = np.arange(768, dtype=np.float64)
    start = float(t0 + 100 * dt)
    end = float(t0 + 600 * dt)

    if kind == "series":
        source = TimeSeries(values, t0=t0, dt=dt)
        reference = GwpyTimeSeries(values, t0=t0, dt=dt)
        if materialize_xindex:
            _ = reference.xindex
        expected = reference.crop(start, end)
    else:
        source = TimeSeriesMatrix(values.reshape(1, 1, -1), t0=t0, dt=dt)
        expected = source[..., 100:600]
    if materialize_xindex:
        _ = source.xindex

    cropped = source.crop(start, end)

    np.testing.assert_array_equal(cropped.value, expected.value)
    expected_t0 = float(expected.t0.value) if kind == "series" else start
    assert _float64_bits(float(cropped.t0.value)) == _float64_bits(expected_t0)
    expected_dt = expected.dt if kind == "series" else source.dt
    assert _float64_bits(float(cropped.dt.value)) == _float64_bits(
        float(expected_dt.value)
    )


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


@pytest.mark.parametrize("kind", ("series", "matrix"))
def test_crop_one_ulp_below_an_exact_boundary_uses_floor(kind: str) -> None:
    """Only the exact floating grid point may snap to its sample index."""
    t0 = 1_234_567_890.1234567
    dt = 1.0 / 30.0
    values = np.arange(256, dtype=np.float64)
    source = (
        TimeSeries(values, t0=t0, dt=dt)
        if kind == "series"
        else TimeSeriesMatrix(values.reshape(1, 1, -1), t0=t0, dt=dt)
    )
    exact = t0 + 100 * dt
    below = np.nextafter(exact, -np.inf)

    cropped = source.crop(below, exact + dt)

    if kind == "series":
        reference = GwpyTimeSeries(values, t0=t0, dt=dt)
        expected = reference.crop(below, exact + dt)
        np.testing.assert_array_equal(cropped.value, expected.value)
        assert float(cropped.t0.value) == float(expected.t0.value)
    else:
        np.testing.assert_array_equal(cropped.value, source.value[..., 99:101])


def test_crop_psd_matches_gwpy_at_a_large_gps_epoch() -> None:
    """The timing selection and resulting PSD must match GWpy."""
    rng = np.random.default_rng(617)
    t0 = 1_234_567_890.1234567
    dt = 1.0 / 30.0
    source = TimeSeries(rng.standard_normal(768), t0=t0, dt=dt)
    reference = GwpyTimeSeries(source.value, t0=t0, dt=dt)
    start = t0 + 100 * dt
    end = t0 + 600 * dt

    cropped = source.crop(start, end)
    expected = reference.crop(start, end)
    cropped_psd = cropped.psd(fftlength=3.0, overlap=1.0, window="hann")
    expected_psd = expected.psd(fftlength=3.0, overlap=1.0, window="hann")

    np.testing.assert_array_equal(cropped.value, expected.value)
    assert float(cropped.t0.value) == float(expected.t0.value)
    np.testing.assert_array_equal(
        cropped_psd.frequencies.value, expected_psd.frequencies.value
    )
    np.testing.assert_allclose(cropped_psd.value, expected_psd.value, rtol=0, atol=0)
