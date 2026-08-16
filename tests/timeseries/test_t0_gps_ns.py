"""Tests for exact GPS nanosecond tracking on ``TimeSeries``."""

import numpy as np
import pytest
from gwpy.time import LIGOTimeGPS

from gwexpy.timeseries import TimeSeries


def test_t0_ns_accepts_integral_values_and_is_read_only():
    ts = TimeSeries([1, 2, 3], t0_ns=1_234_567_890_123_456_789, dt=1)

    assert ts.t0_gps_ns == 1_234_567_890_123_456_789
    assert ts._gwex_t0_gps_precision == "exact"
    with pytest.raises(AttributeError):
        ts.t0_gps_ns = 0


def test_t0_ns_accepts_signed_64_bit_maximum():
    ts = TimeSeries([1], t0_ns=2**63 - 1)

    assert ts.t0_gps_ns == 2**63 - 1
    assert ts._gwex_t0_gps_precision == "exact"


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5, "10", object()])
def test_t0_ns_rejects_non_integral_values(value):
    with pytest.raises(TypeError):
        TimeSeries([1], t0_ns=value)


@pytest.mark.parametrize("value", [-1, 2**63])
def test_t0_ns_rejects_values_outside_signed_64_bit_range(value):
    with pytest.raises(ValueError):
        TimeSeries([1], t0_ns=value)


def test_integral_and_ligo_t0_are_exact_but_float_t0_is_quantized():
    integral = TimeSeries([1], t0=123, dt=1)
    ligo = TimeSeries([1], t0=LIGOTimeGPS(123, 456), dt=1)
    quantized = TimeSeries([1], t0=10.0000000005, dt=1)

    assert integral.t0_gps_ns == 123_000_000_000
    assert integral._gwex_t0_gps_precision == "exact"
    assert ligo.t0_gps_ns == 123_000_000_456
    assert ligo._gwex_t0_gps_precision == "exact"
    assert quantized.t0_gps_ns == 10_000_000_000
    assert quantized._gwex_t0_gps_precision == "quantized"


def test_t0_ns_must_agree_with_t0_or_epoch():
    TimeSeries([1], t0_ns=123_000_000_456, t0=LIGOTimeGPS(123, 456), dt=1)

    with pytest.raises(ValueError, match="nanosecond"):
        TimeSeries([1], t0_ns=123_000_000_457, t0=LIGOTimeGPS(123, 456), dt=1)


def test_t0_ns_accepts_agreeing_t0_and_epoch():
    ts = TimeSeries(
        [1],
        t0_ns=123_000_000_456,
        t0=LIGOTimeGPS(123, 456),
        epoch=LIGOTimeGPS(123, 456),
        dt=1,
    )

    assert ts.t0_gps_ns == 123_000_000_456
    assert ts._gwex_t0_gps_precision == "exact"


def test_t0_ns_rejects_mismatched_t0_when_epoch_agrees():
    with pytest.raises(ValueError, match="t0"):
        TimeSeries(
            [1],
            t0_ns=123_000_000_456,
            t0=LIGOTimeGPS(123, 457),
            epoch=LIGOTimeGPS(123, 456),
            dt=1,
        )


def test_t0_ns_rejects_mismatched_epoch_when_t0_agrees():
    with pytest.raises(ValueError, match="t0_ns and epoch"):
        TimeSeries(
            [1],
            t0_ns=123_000_000_456,
            t0=LIGOTimeGPS(123, 456),
            epoch=LIGOTimeGPS(123, 457),
            dt=1,
        )


@pytest.mark.parametrize("alias", ["t0", "epoch"])
def test_t0_ns_accepts_each_agreeing_single_alias(alias):
    ts = TimeSeries(
        [1],
        t0_ns=123_000_000_456,
        **{alias: LIGOTimeGPS(123, 456)},
        dt=1,
    )

    assert ts.t0_gps_ns == 123_000_000_456
    assert ts._gwex_t0_gps_precision == "exact"


def test_copy_and_view_preserve_gps_nanosecond_state():
    ts = TimeSeries([1, 2], t0_ns=1_000_000_001, dt=1)

    copied = ts.copy()
    viewed = ts.view(TimeSeries)

    assert copied.t0_gps_ns == ts.t0_gps_ns
    assert copied._gwex_t0_gps_precision == "exact"
    assert viewed.t0_gps_ns == ts.t0_gps_ns
    assert viewed._gwex_t0_gps_precision == "exact"


def test_unit_ns_slice_advances_exact_origin():
    ts = TimeSeries([1, 2, 3, 4], t0_ns=1_000_000_001, dt=0.1)

    sliced = ts[2:]

    assert sliced.t0_gps_ns == 1_200_000_001
    assert sliced._gwex_t0_gps_precision == "exact"


def test_materialized_xindex_stays_synchronized_after_large_gps_slice():
    source = TimeSeries(
        np.arange(8, dtype=float),
        t0_ns=1_234_567_890_123_456_789,
        dt=0.1,
    )
    source_x0 = source.x0.copy()
    source_dx = source.dx.copy()
    source_xindex = source.xindex.copy()
    assert "_xindex" in source.__dict__

    result = source[2:6]
    result_xindex = result.xindex
    result_stride = result_xindex[1] - result_xindex[0]
    assert "_xindex" in result.__dict__

    np.testing.assert_array_equal(result.value, np.arange(2, 6, dtype=float))
    assert result.x0 == result_xindex[0]
    assert result.dx == result_stride
    assert result.x0 == source_xindex[2]
    assert result.dx == source_xindex[3] - source_xindex[2]
    assert result.t0_gps_ns == 1_234_567_890_323_456_789
    assert result._gwex_t0_gps_precision == "exact"

    assert source.x0 == source_x0
    assert source.dx == source_dx
    np.testing.assert_array_equal(source.xindex, source_xindex)
    assert source.t0_gps_ns == 1_234_567_890_123_456_789
    assert source._gwex_t0_gps_precision == "exact"


@pytest.mark.parametrize("materialize_source_xindex", [False, True])
def test_exact_slice_xindex_is_stable_for_both_materialization_orders(
    materialize_source_xindex: bool,
) -> None:
    source = TimeSeries(
        np.arange(8, dtype=float),
        t0_ns=1_234_567_890_123_456_789,
        dt=0.1,
    )
    source_x0 = source.x0.copy()
    source_dx = source.dx.copy()
    source_values = source.value.copy()
    assert "_xindex" not in source.__dict__
    if materialize_source_xindex:
        _ = source.xindex

    result = source[2:6]
    result_xindex = result.xindex
    source_xindex = source.xindex

    assert result.x0 == result_xindex[0]
    assert result.x0 == source_xindex[2]
    assert result.dx == result_xindex[1] - result_xindex[0]
    assert result.dx == source_xindex[3] - source_xindex[2]
    assert result._x0 is not source._x0
    assert result._dx is not source._dx
    assert result.xindex is not source.xindex
    assert not np.shares_memory(result.xindex.value, source.xindex.value)
    assert source.x0 == source_x0
    assert source.dx == source_dx
    np.testing.assert_array_equal(source.value, source_values)


@pytest.mark.parametrize("key", [slice(10, None), slice(2, 1), slice(-10, -8)])
def test_empty_slices_clear_gps_nanosecond_state(key):
    ts = TimeSeries([1, 2, 3, 4], t0_ns=1_000_000_001, dt=0.1)

    result = ts[key]

    assert len(result) == 0
    assert result.t0_gps_ns is None
    assert result._gwex_t0_gps_precision is None


@pytest.mark.parametrize(
    ("key", "expected"),
    [(slice(-3, None), 1_100_000_001), (slice(-4, -1), 1_000_000_001)],
)
def test_nonempty_negative_slices_advance_from_normalized_start(key, expected):
    ts = TimeSeries([1, 2, 3, 4], t0_ns=1_000_000_001, dt=0.1)

    result = ts[key]

    assert len(result) > 0
    assert result.t0_gps_ns == expected
    assert result._gwex_t0_gps_precision == "exact"


def test_fractional_ns_slice_quantizes_origin_with_ties_to_even():
    ts = TimeSeries([1, 2], t0_ns=1_000_000_000, dt=0.0000000005)

    sliced = ts[1:]

    assert sliced.t0_gps_ns == 1_000_000_000
    assert sliced._gwex_t0_gps_precision == "quantized"


@pytest.mark.parametrize("key", [slice(None, None, 2), [0, 1], np.array([True, False])])
def test_non_affine_indexing_clears_gps_nanosecond_state(key):
    ts = TimeSeries([1, 2], t0_ns=1_000_000_000, dt=1)

    result = ts[key]

    assert getattr(result, "t0_gps_ns", None) is None
    assert getattr(result, "_gwex_t0_gps_precision", None) is None


def test_irregular_axis_clears_gps_nanosecond_state_after_slice():
    ts = TimeSeries([1, 2, 3], t0_ns=1_000_000_000, times=[0, 1, 2.5])

    result = ts[:2]

    assert result.t0_gps_ns is None
    assert result._gwex_t0_gps_precision is None
