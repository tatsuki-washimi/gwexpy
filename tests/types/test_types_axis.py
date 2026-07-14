import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.units import Quantity
from gwpy.types.index import Index

from gwexpy.types.axis import AxisDescriptor


def test_axis_descriptor_irregular():
    idx = Quantity([0, 1, 1.5, 3], "Hz")
    desc = AxisDescriptor("freq", idx)

    assert desc.name == "freq"
    assert desc.size == 4
    assert desc.unit == u.Hz
    assert not desc.regular
    assert desc.delta is None


def test_axis_descriptor_regular():
    idx = Index(np.arange(101, dtype=float) * 0.125, unit="s")
    # depending on Index implementation, it might have .regular=True or we calculate it
    desc = AxisDescriptor("time", idx)

    assert desc.regular
    assert np.isclose(desc.delta.value, 0.125)
    assert desc.delta.unit == u.s


def test_axis_descriptor_small_scale_irregular_axis_rejects_coordinate_step():
    """Small-scale unequal intervals must not be hidden by an absolute tolerance."""
    desc = AxisDescriptor("time", Quantity([0.0, 1e-12, 3e-12], "s"))

    assert not desc.regular
    assert desc.delta is None
    with pytest.raises(ValueError, match="irregular axis"):
        desc.iloc_slice(slice(None, None, 1e-12 * u.s))


@pytest.mark.parametrize(
    ("index", "expected_regular"),
    [
        (Quantity([0.0, 1e-12, 2e-12], "s"), True),
        (Quantity([0.0, 1.0, 2.0], "ps"), True),
        (Quantity([0.0, 1e-12, 3e-12], "s"), False),
        (Quantity([0.0, 1.0, 3.0], "ps"), False),
    ],
)
def test_axis_descriptor_regular_is_invariant_under_equivalent_units(
    index, expected_regular
):
    """Regularity must agree for equivalent seconds and picoseconds axes."""
    assert AxisDescriptor("time", index).regular is expected_regular


def test_axis_descriptor_large_offset_regular_axis_is_regular():
    """A represented uniform large-offset axis is regular."""
    desc = AxisDescriptor("time", (1e9 + np.arange(4) * 0.125) * u.s)

    assert desc.regular
    assert np.isclose(desc.delta.to_value(u.s), 0.125)


def test_axis_descriptor_large_offset_irregular_axis_is_not_regular():
    values = 1e9 + np.array([0.0, 0.1, 0.2, 0.30001])

    assert not AxisDescriptor("time", values * u.s).regular


def test_axis_descriptor_float32_large_offset_regular_axis_is_regular():
    values = np.float32(1e9) + np.arange(4, dtype=np.float32) * np.float32(128)

    assert AxisDescriptor("time", values * u.s).regular


def test_axis_descriptor_float32_large_offset_irregular_axis_is_not_regular():
    values = np.float32(1e9) + np.array([0, 128, 640], dtype=np.float32)

    assert not AxisDescriptor("time", values * u.s).regular


def test_axis_descriptor_float32_large_offset_quantized_axis_is_not_regular():
    values = np.float32(1e9) + np.arange(4, dtype=np.float32) * np.float32(100)

    assert not AxisDescriptor("time", values * u.s).regular


def test_axis_descriptor_complex64_large_offset_regular_axis_is_regular():
    values = np.complex64(1e9) + np.arange(4, dtype=np.float32) * np.complex64(128)

    assert AxisDescriptor("complex", values * u.s).regular


@pytest.mark.parametrize(
    "index",
    [
        np.float32(1e9) + np.arange(4, dtype=np.float32) * np.float32(128),
        np.complex64(1e9) + np.arange(4, dtype=np.float32) * np.complex64(128),
    ],
)
def test_axis_descriptor_regular_delta_matches_all_represented_intervals(index):
    axis = AxisDescriptor("coordinate", Quantity(index, "s"))
    diffs = np.diff(axis.index.to_value(axis.unit))
    interval_scale = np.max(np.abs(diffs))
    atol = abs(np.spacing(interval_scale))

    assert axis.regular
    assert axis.delta is not None
    assert np.allclose(diffs, axis.delta.to_value(axis.unit), rtol=0, atol=atol)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([10**18, 10**18 + 1, 10**18 + 2], True),
        ([10**18, 10**18 + 1, 10**18 + 3], False),
    ],
)
def test_axis_descriptor_large_integer_offset_regularity(values, expected):
    """Integer coordinates must retain exact interval information."""
    axis = AxisDescriptor("time", Quantity(values, "ns", dtype=np.int64))

    assert axis.regular is expected


def test_axis_descriptor_complex_axis_uses_imaginary_coordinate_values():
    """Regularity must not discard the imaginary components of coordinates."""
    axis = AxisDescriptor("complex", Quantity([0j, 1 + 1j, 2 + 3j], "s"))

    assert not axis.regular


def test_axis_descriptor_allows_descending_axis_for_nearest_selection():
    """AxisDescriptor itself does not require the ascending slice contract."""
    desc = AxisDescriptor("position", Quantity([3.0, 2.0, 1.0], "m"))

    assert desc.regular
    assert desc.iloc_nearest(2.1 * u.m) == 1


def test_axis_descriptor_irregular_descending_axis_is_not_regular():
    desc = AxisDescriptor("position", Quantity([3.0, 2.0, 0.0], "m"))

    assert not desc.regular


def test_axis_descriptor_non_finite_axis_is_not_regular():
    desc = AxisDescriptor("time", Quantity([0.0, np.nan, 1.0], "s"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert not desc.regular


def test_iloc_nearest():
    idx = Quantity([0, 1, 2, 5, 10], "m")
    desc = AxisDescriptor("pos", idx)

    # 0 -> 0
    assert desc.iloc_nearest(0 * u.m) == 0
    # 1.4 -> 1 (diff 0.4 vs 2 is 0.6)
    assert desc.iloc_nearest(1.4 * u.m) == 1
    # 8 -> 4 (10) (diff 2 vs 5 is 3) or 3 (5)? 8-5=3, 10-8=2. So 10 is nearer.
    assert desc.iloc_nearest(8 * u.m) == 4


def test_iloc_slice():
    idx = Quantity([0, 10, 20, 30, 40], "s")
    desc = AxisDescriptor("time", idx)

    # slice(15, 35) -> 20, 30. Indices 2, 3?
    # searchsorted 'left': 15 -> idx 2 (20).
    # 35 -> idx 4 (40).
    # slice(2, 4) -> indices 2, 3. Values 20, 30. Correct.

    s = slice(15 * u.s, 35 * u.s)
    res = desc.iloc_slice(s)
    assert res.start == 2
    assert res.stop == 4
    assert res.step is None


def test_iloc_slice_irregular_step_error():
    idx = Quantity([0, 1, 4, 9], "m")
    desc = AxisDescriptor("pos", idx)

    with pytest.raises(ValueError):
        desc.iloc_slice(slice(0, 4, 1 * u.m))
