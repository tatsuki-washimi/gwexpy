"""Public contracts for exact integer-nanosecond TimeSeries epochs."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries


def test_t0_ns_is_exact_at_a_large_gps_epoch() -> None:
    epoch_ns = 1_234_567_890_123_456_789

    series = TimeSeries([1.0], t0_ns=epoch_ns, dt=1.0)

    assert series.t0_gps_ns == epoch_ns
    # ``t0`` remains the GWpy-compatible float/Quantity view of the epoch.
    assert float(series.t0.to_value(u.s)) == pytest.approx(epoch_ns / 1e9)


def test_t0_ns_normalizes_independently_of_axis_time_unit() -> None:
    epoch_ns = 1_234_567_890_123_456_789

    series = TimeSeries([1.0], t0_ns=epoch_ns, dt=10 * u.ms)

    assert series.t0_gps_ns == epoch_ns


def test_t0_ns_handles_negative_nanosecond_boundary() -> None:
    series = TimeSeries([1.0, 2.0], t0_ns=-1, dt=1.0)

    assert series.t0_gps_ns == -1
    assert series[1:].t0_gps_ns == 999_999_999


def test_t0_ns_preserves_exact_epoch_through_copy_and_slice() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(np.arange(5.0), t0_ns=epoch_ns, dt=10 * u.ns)

    assert copy.copy(series).t0_gps_ns == epoch_ns
    assert series[3:].t0_gps_ns == epoch_ns + 30


def test_t0_ns_derives_an_exact_interval_from_sample_rate() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(np.arange(4.0), t0_ns=epoch_ns, sample_rate=1000)

    assert series._gwex_dt_gps_ns == 1_000_000
    assert series[1:].t0_gps_ns == epoch_ns + 1_000_000


def test_t0_ns_preserves_exact_epoch_through_gwpy_copy_and_deepcopy() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(np.arange(5.0), t0_ns=epoch_ns, dt=7 * u.ns)

    assert series.view().t0_gps_ns == epoch_ns
    assert series.copy().t0_gps_ns == epoch_ns
    assert copy.deepcopy(series).t0_gps_ns == epoch_ns


@pytest.mark.parametrize("dt", [3 * u.ns, 7 * u.ns, 100 * u.ns, 1000 * u.ns, 1 * u.ms])
def test_t0_ns_slice_uses_exact_unit_aware_sample_offsets(dt: u.Quantity) -> None:
    epoch_ns = 1_234_567_890_123_456_789
    expected_offset_ns = int(dt.to_value(u.ns))
    series = TimeSeries(np.arange(3.0), t0_ns=epoch_ns, dt=dt)

    assert series[1:].t0_gps_ns == epoch_ns + expected_offset_ns
    assert series[(slice(2, None),)].t0_gps_ns == epoch_ns + 2 * expected_offset_ns


def test_t0_ns_slice_step_updates_the_exact_sample_interval() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(np.arange(6.0), t0_ns=epoch_ns, dt=7 * u.ns)

    stepped = series[(slice(1, None, 2),)]

    assert stepped.t0_gps_ns == epoch_ns + 7
    assert stepped._gwex_dt_gps_ns == 14
    assert stepped[1:].t0_gps_ns == epoch_ns + 21
    assert stepped[1:]._gwex_dt_gps_ns == 14


def test_t0_ns_negative_slice_step_updates_epoch_and_interval() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(np.arange(5.0), t0_ns=epoch_ns, dt=7 * u.ns)

    reversed_series = series[4::-2]

    assert reversed_series.t0_gps_ns == epoch_ns + 28
    assert reversed_series._gwex_dt_gps_ns == -14
    assert reversed_series[1:].t0_gps_ns == epoch_ns + 14


def test_t0_ns_zero_slice_step_keeps_python_error_semantics() -> None:
    series = TimeSeries(np.arange(2.0), t0_ns=0, dt=7 * u.ns)

    with pytest.raises(ValueError, match="slice step cannot be zero"):
        series[::0]


def test_t0_ns_slice_drops_exact_authority_for_non_integral_nanosecond_interval() -> (
    None
):
    series = TimeSeries(np.arange(2.0), t0_ns=0, dt=(1 / 3) * u.ns)

    result = series[1:]

    np.testing.assert_array_equal(result.value, [1.0])
    assert not hasattr(result, "_gwex_t0_gps_ns")
    assert not hasattr(result, "_gwex_dt_gps_ns")


def test_t0_ns_crop_drops_exact_authority_for_non_integral_nanosecond_interval() -> (
    None
):
    series = TimeSeries(np.arange(8.0), t0_ns=0, dt=(1 / 3) * u.ns)
    reference = GwpyTimeSeries(np.arange(8.0), t0=0, dt=(1 / 3) * u.ns)

    result = series.crop(0, 1e-9)
    expected = reference.crop(0, 1e-9)

    np.testing.assert_array_equal(result.value, expected.value)
    assert not hasattr(result, "_gwex_t0_gps_ns")
    assert not hasattr(result, "_gwex_dt_gps_ns")


@pytest.mark.parametrize("authority", ["t0", "epoch", "x0", "times"])
def test_t0_ns_rejects_all_simultaneous_epoch_authorities(authority: str) -> None:
    with pytest.raises(TypeError, match="t0_ns cannot be combined"):
        TimeSeries([1.0], t0_ns=0, dt=1.0, **{authority: 0})


def test_t0_gps_ns_is_read_only() -> None:
    series = TimeSeries([1.0], t0_ns=0, dt=1.0)

    with pytest.raises(AttributeError):
        series.t0_gps_ns = 1


def test_t0_gps_ns_remains_available_for_legacy_float_epoch() -> None:
    series = TimeSeries([1.0], t0=123.25, dt=1.0)

    assert series.t0_gps_ns == 123_250_000_000


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_t0_ns_rejects_boolean_values(value: object) -> None:
    with pytest.raises(TypeError, match="integer number of GPS nanoseconds"):
        TimeSeries([1.0], t0_ns=value, dt=1.0)


@pytest.mark.parametrize("attribute", ["t0", "x0"])
def test_epoch_setters_synchronize_exact_authority(attribute: str) -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries([1.0], t0_ns=epoch_ns, dt=1.0)

    setattr(
        series,
        attribute,
        LIGOTimeGPS(epoch_ns // 1_000_000_000, epoch_ns % 1_000_000_000 + 7),
    )

    assert series.t0_gps_ns == epoch_ns + 7


@pytest.mark.parametrize("attribute", ["t0", "x0"])
def test_epoch_setters_reject_non_integral_nanoseconds(attribute: str) -> None:
    series = TimeSeries([1.0], t0_ns=0, dt=1.0)

    with pytest.raises(ValueError, match="integer number of GPS nanoseconds"):
        setattr(series, attribute, 1.5 * u.ns)

    assert series.t0_gps_ns == 0


def test_exact_epoch_metadata_reconstructs_smooth_and_resample_outputs() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(np.arange(20.0), t0_ns=epoch_ns, dt=0.01)

    assert series._get_meta_for_constructor()["t0_ns"] == epoch_ns
    assert series.smooth(3).t0_gps_ns == epoch_ns
    assert series.resample(50).t0_gps_ns == epoch_ns
