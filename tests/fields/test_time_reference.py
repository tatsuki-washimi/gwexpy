"""Regression coverage for Field time-reference metadata (#725)."""

import pickle

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_array_equal

from gwexpy.fields import ScalarField


def make_field(*, epoch=None, axis0=None):
    """Build a small Field with explicit spatial coordinates."""
    if axis0 is None:
        axis0 = np.array([3.0, 3.25, 3.5, 3.75]) * u.s
    return ScalarField(
        np.arange(32.0).reshape(4, 2, 2, 2),
        epoch=epoch,
        axis0=axis0,
        axis1=np.array([-2.0, 1.0]) * u.m,
        axis2=np.array([4.0, 6.0]) * u.m,
        axis3=np.array([0.0, 5.0]) * u.m,
        axis_names=("t", "x", "y", "z"),
    )


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (None, "relative"),
        (np.nan, "relative"),
        (0.0, "absolute"),
        (1_400_000_000.0, "absolute"),
    ],
)
def test_constructor_classifies_epoch_without_reinterpreting_space_axes(
    epoch, expected
):
    field = make_field(epoch=epoch)

    assert field.time_reference == expected
    assert_array_equal(field._axis1_index.to_value(u.m), [-2.0, 1.0])
    assert_array_equal(field._axis2_index.to_value(u.m), [4.0, 6.0])
    assert_array_equal(field._axis3_index.to_value(u.m), [0.0, 5.0])
    assert field.space_domains == {"x": "real", "y": "real", "z": "real"}
    assert np.all(np.isfinite(field._axis0_index.value))


@pytest.mark.parametrize("epoch", [np.inf, -np.inf])
def test_constructor_rejects_infinite_epoch(epoch):
    with pytest.raises(ValueError, match="epoch must not be infinite"):
        make_field(epoch=epoch)


def test_with_epoch_shifts_time_axis_and_keeps_axis_offsets_and_space_metadata():
    field = make_field()
    original_time = field._axis0_index.copy()
    original_space = tuple(
        getattr(field, f"_axis{index}_index").copy() for index in (1, 2, 3)
    )
    gps = 1_400_000_000.0

    anchored = field.with_epoch(gps)

    assert anchored.time_reference == "absolute"
    assert float(anchored.epoch.gps) == gps
    assert_array_equal(
        anchored._axis0_index.to_value(u.s),
        gps + np.array([3.0, 3.25, 3.5, 3.75]),
    )
    assert_array_equal(field._axis0_index.to_value(u.s), original_time.to_value(u.s))
    for index, expected in zip((1, 2, 3), original_space, strict=True):
        assert_array_equal(
            getattr(anchored, f"_axis{index}_index").to_value(u.m),
            expected.to_value(u.m),
        )
    assert anchored.space_domains == field.space_domains
    assert field.time_reference == "relative"


def test_with_epoch_warns_when_time_step_is_below_ten_gps_ulps():
    field = make_field(axis0=np.arange(4) * 1e-8 * u.s)

    with pytest.warns(UserWarning, match="below 10 ULP"):
        anchored = field.with_epoch(1_400_000_000.0)

    assert anchored.time_reference == "absolute"


def test_with_epoch_after_time_fft_preserves_offset_through_ifft():
    field = make_field()
    frequency_field = field.fft_time()
    gps = 1_400_000_000.0
    original_frequency_axis = frequency_field._axis0_index.copy()

    anchored = frequency_field.with_epoch(gps)
    restored = anchored.ifft_time()

    assert_array_equal(anchored._axis0_index.value, original_frequency_axis.value)
    assert anchored.time_reference == "absolute"
    assert_array_equal(
        restored._axis0_index.to_value(u.s),
        gps + np.array([3.0, 3.25, 3.5, 3.75]),
    )
    assert restored.time_reference == "absolute"


def test_with_epoch_after_time_fft_treats_dimensionless_time_as_seconds():
    offsets = 3.0 + np.arange(4) * 0.25
    field = make_field(axis0=offsets * u.dimensionless_unscaled)
    frequency_field = field.fft_time()
    gps = 1_400_000_000.0

    restored = frequency_field.with_epoch(gps).ifft_time()

    assert_array_equal(restored._axis0_index.to_value(u.s), gps + offsets)


def test_with_epoch_after_time_fft_warns_for_sub_ten_ulp_sampling():
    field = make_field(axis0=np.arange(4) * 1e-8 * u.s)
    frequency_field = field.fft_time()

    with pytest.warns(UserWarning, match="below 10 ULP"):
        anchored = frequency_field.with_epoch(1_400_000_000.0)

    assert anchored.time_reference == "absolute"


def test_time_origin_survives_frequency_field_rebuilds():
    field = make_field()
    frequency_field = field.fft_time()
    rebuilt_fields = (
        frequency_field.fft_space(axes=("x",)),
        frequency_field[:, :1, :, :],
        frequency_field.isel(x=0),
    )
    gps = 1_400_000_000.0

    for rebuilt in rebuilt_fields:
        restored = rebuilt.with_epoch(gps).ifft_time()
        assert_array_equal(restored._axis0_index.to_value(u.s)[0], gps + 3.0)


def test_with_epoch_infers_sampling_interval_for_frequency_field():
    field = ScalarField(
        np.ones((3, 2, 2, 2)),
        axis0=np.array([0.0, 25e6, 50e6]) * u.Hz,
        axis1=np.array([-2.0, 1.0]) * u.m,
        axis2=np.array([4.0, 6.0]) * u.m,
        axis3=np.array([0.0, 5.0]) * u.m,
        axis_names=("f", "x", "y", "z"),
        axis0_domain="frequency",
    )

    with pytest.warns(UserWarning, match="below 10 ULP"):
        anchored = field.with_epoch(1_400_000_000.0)

    assert anchored._axis0_dt.to_value(u.s) == pytest.approx(1e-8)
    restored = anchored.ifft_time()
    assert restored.time_reference == "absolute"
    assert_array_equal(restored._axis0_index.to_value(u.s), [1.4e9] * 4)


def test_with_epoch_rejects_absolute_field():
    with pytest.raises(ValueError, match="requires a relative"):
        make_field(epoch=0.0).with_epoch(1_400_000_000.0)


def test_time_reference_survives_copy_slice_isel_and_pickle():
    field = make_field()
    copied = field.copy()
    sliced = field[:, 1, :, :]
    iseled = field.isel(t=slice(1, 3))
    restored = pickle.loads(pickle.dumps(field))

    cases = (
        (copied, [-2.0, 1.0]),
        (sliced, [1.0]),
        (iseled, [-2.0, 1.0]),
        (restored, [-2.0, 1.0]),
    )
    for result, expected_x in cases:
        assert result.time_reference == "relative"
        assert_array_equal(result._axis1_index.to_value(u.m), expected_x)
        assert result.space_domains == field.space_domains


def test_time_and_space_domain_transforms_preserve_time_reference_and_metadata():
    field = make_field()

    time_roundtrip = field.fft_time().ifft_time()
    space_roundtrip = field.fft_space().ifft_space()

    for result in (time_roundtrip, space_roundtrip):
        assert result.time_reference == "relative"
        assert result.epoch == field.epoch
        assert result.space_domains == field.space_domains
        for index in (1, 2, 3):
            assert_array_equal(
                getattr(result, f"_axis{index}_index").to_value(u.m),
                getattr(field, f"_axis{index}_index").to_value(u.m),
            )
    assert_array_equal(
        time_roundtrip._axis0_index.to_value(u.s),
        field._axis0_index.to_value(u.s),
    )


def test_time_fft_roundtrip_preserves_scaled_time_axis_units():
    offsets = np.array([3.0, 3.25, 3.5, 3.75]) * u.ms

    restored = make_field(axis0=offsets).fft_time().ifft_time()

    assert restored._axis0_index.unit == u.ms
    assert_array_equal(restored._axis0_index.to_value(u.ms), offsets.value)


@pytest.mark.parametrize("reverse", [False, True])
def test_field_arithmetic_rejects_mixed_time_references(reverse):
    relative = make_field()
    absolute = make_field(epoch=1_400_000_000.0)
    left, right = (absolute, relative) if reverse else (relative, absolute)

    with pytest.raises(ValueError, match="time-reference mismatch"):
        left + right


@pytest.mark.parametrize("reverse", [False, True])
def test_field_arithmetic_rejects_different_absolute_epochs(reverse):
    first = make_field(epoch=1_400_000_000.0)
    second = make_field(epoch=1_500_000_000.0)
    left, right = (second, first) if reverse else (first, second)

    with pytest.raises(ValueError, match="epoch mismatch"):
        left + right
