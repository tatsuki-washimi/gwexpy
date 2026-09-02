"""Public contracts for exact integer-nanosecond TimeSeries epochs."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


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


@pytest.mark.parametrize("authority", ["t0", "times"])
def test_t0_ns_rejects_explicit_none_epoch_authorities(authority: str) -> None:
    with pytest.raises(TypeError, match=rf"t0_ns.*{authority}"):
        TimeSeries([1.0], t0_ns=0, dt=1.0, **{authority: None})


def test_t0_ns_rejects_explicit_none_positional_t0() -> None:
    with pytest.raises(TypeError, match=r"t0_ns.*t0"):
        TimeSeries([1.0], None, None, t0_ns=0)


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


@pytest.mark.parametrize("attribute", ["t0", "x0"])
def test_bare_epoch_setter_uses_the_current_axis_unit(attribute: str) -> None:
    series = TimeSeries(np.arange(4.0), t0_ns=0, dt=7, xunit=u.ns)
    reference = GwpyTimeSeries(np.arange(4.0), t0=0, dt=7, xunit=u.ns)

    setattr(series, attribute, 42)
    setattr(reference, attribute, 42)

    assert series.t0 == reference.t0
    assert series.x0 == reference.x0
    assert series.t0_gps_ns == 42


@pytest.mark.parametrize("inplace", [True, False], ids=["inplace", "copy"])
def test_resize_false_append_advances_exact_epoch_in_axis_units(
    inplace: bool,
) -> None:
    left = TimeSeries(np.arange(4.0), t0_ns=1000, dt=7, xunit=u.ns)
    right = TimeSeries(np.arange(2.0), t0_ns=1028, dt=7, xunit=u.ns)

    result = left.append(right, inplace=inplace, resize=False)

    assert result.t0_gps_ns == 1014
    assert result._gwex_dt_gps_ns == 7
    if inplace:
        assert result is left
    else:
        assert result is not left
        assert left.t0_gps_ns == 1000


def test_collection_resize_false_append_preserves_exact_epochs() -> None:
    left = TimeSeriesDict(
        {
            key: TimeSeries(np.arange(4.0), t0_ns=1000, dt=7, xunit=u.ns)
            for key in ("first", "second")
        }
    )
    right = TimeSeriesDict(
        {
            key: TimeSeries(np.arange(2.0), t0_ns=1028, dt=7, xunit=u.ns)
            for key in ("first", "second")
        }
    )

    result = left.append(right, resize=False)

    assert result is left
    for series in result.values():
        assert series.t0_gps_ns == 1014
        assert series._gwex_dt_gps_ns == 7


@pytest.mark.parametrize(
    ("initial_dt", "attribute", "value", "expected_dt_ns"),
    [
        pytest.param(7 * u.ns, "dt", 11 * u.ns, 11, id="dt"),
        pytest.param(7 * u.ns, "dx", 13 * u.ns, 13, id="dx"),
        pytest.param(1 * u.s, "sample_rate", 4 * u.Hz, 250_000_000, id="sample-rate"),
    ],
)
def test_cadence_setters_synchronize_exact_interval_for_copy_and_slice(
    initial_dt: u.Quantity,
    attribute: str,
    value: u.Quantity,
    expected_dt_ns: int,
) -> None:
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(np.arange(6.0), t0_ns=epoch_ns, dt=initial_dt)
    reference = GwpyTimeSeries(np.arange(6.0), t0=0, dt=initial_dt)

    setattr(series, attribute, value)
    setattr(reference, attribute, value)

    assert series.dt == reference.dt
    assert series.dx == reference.dx
    assert series.sample_rate == reference.sample_rate
    assert series._gwex_dt_gps_ns == expected_dt_ns
    assert series.copy()._gwex_dt_gps_ns == expected_dt_ns
    sliced = series[2:]
    assert sliced.t0_gps_ns == epoch_ns + 2 * expected_dt_ns
    assert sliced._gwex_dt_gps_ns == expected_dt_ns


def test_nonintegral_cadence_setter_drops_stale_exact_interval() -> None:
    series = TimeSeries(np.arange(4.0), t0_ns=0, dt=7 * u.ns)

    series.dt = (1 / 3) * u.ns

    assert "_gwex_dt_gps_ns" not in series.__dict__
    sliced = series[1:]
    assert "_gwex_t0_gps_ns" not in sliced.__dict__
    assert "_gwex_dt_gps_ns" not in sliced.__dict__
