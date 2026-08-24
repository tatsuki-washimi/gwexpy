"""Public contracts for exact integer-nanosecond TimeSeries epochs."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from astropy import units as u

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


def test_t0_ns_slice_rejects_a_non_integral_nanosecond_interval() -> None:
    series = TimeSeries(np.arange(2.0), t0_ns=0, dt=(1 / 3) * u.ns)

    with pytest.raises(ValueError, match="integer number of nanoseconds"):
        series[1:]


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
