"""Tests for uncertainty-bearing Series metadata, propagation, warn-and-drop, and to_timeseriesdict()."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.types.series import Series


def _make_quantile_ts(
    n: int = 32, *, t0: float = 1000.0, dt: float = 0.25
) -> TimeSeries:
    vals = np.linspace(10.0, 20.0, n)
    ts = TimeSeries(vals, t0=t0, dt=dt * u.s, unit="V", name="sig", channel="H1:SIG")
    q10 = TimeSeries(
        vals - 2.0, t0=t0, dt=dt * u.s, unit="V", name="sig", channel="H1:SIG"
    )
    q50 = TimeSeries(vals, t0=t0, dt=dt * u.s, unit="V", name="sig", channel="H1:SIG")
    q90 = TimeSeries(
        vals + 3.0, t0=t0, dt=dt * u.s, unit="V", name="sig", channel="H1:SIG"
    )
    ts._set_uncertainty(
        quantiles={0.1: q10, 0.5: q50, 0.9: q90},
        estimate_mean=None,
        center_kind="median",
        uncertainty_kind="quantile_interval",
        interval_levels=(0.1, 0.9),
    )
    return ts


def test_uncertainty_defaults_and_mean_median_preserved() -> None:
    ts = TimeSeries([1.0, 2.0, 3.0], dt=0.5, unit="V")
    fs = FrequencySeries([1.0, 2.0, 3.0], df=1.0, unit="V/Hz")
    s = Series([1.0, 2.0, 3.0], unit="m")

    for obj in (ts, fs, s):
        assert obj.lower is None
        assert obj.upper is None
        assert obj.quantiles is None
        assert obj.estimate_mean is None
        assert obj.center_kind is None
        assert obj.uncertainty_kind is None
        assert obj.interval_levels is None
        assert not obj._has_uncertainty()

    assert ts.mean() == 2.0 * u.V
    assert ts.median() == 2.0 * u.V


def test_quantile_derived_and_independent_lower_upper() -> None:
    ts = _make_quantile_ts(8)
    assert ts.lower is ts.quantiles[0.1]
    assert ts.upper is ts.quantiles[0.9]
    assert ts.estimate_mean is None

    # Non-quantile independent lower/upper storage on FrequencySeries
    fs = FrequencySeries([10.0, 20.0, 30.0], df=1.0, unit="V")
    lo = FrequencySeries([9.0, 18.0, 27.0], df=1.0, unit="V")
    hi = FrequencySeries([11.0, 22.0, 33.0], df=1.0, unit="V")
    fs._set_uncertainty(
        lower=lo,
        upper=hi,
        center_kind="mean",
        uncertainty_kind="confidence_interval",
    )
    assert fs.quantiles is None
    assert fs.lower is lo
    assert fs.upper is hi


def test_trivial_operations_propagate_without_warning() -> None:
    ts = _make_quantile_ts(16)
    orig_vals = ts.value.copy()
    orig_lo = ts.lower.value.copy()
    orig_hi = ts.upper.value.copy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        copied = ts.copy()
        assert copied.lower is not ts.lower
        np.testing.assert_allclose(copied.lower.value, orig_lo)
        np.testing.assert_allclose(copied.upper.value, orig_hi)

        sliced = ts[2:10]
        assert len(sliced) == 8
        assert len(sliced.lower) == 8
        np.testing.assert_allclose(sliced.lower.value, orig_lo[2:10])
        np.testing.assert_allclose(sliced.upper.value, orig_hi[2:10])

        cropped = ts.crop(1000.5, 1002.0)
        assert len(cropped) == len(cropped.lower) == len(cropped.upper)

        shifted = ts + 5.0 * u.V
        np.testing.assert_allclose(shifted.value, orig_vals + 5.0)
        np.testing.assert_allclose(shifted.lower.value, orig_lo + 5.0)
        np.testing.assert_allclose(shifted.upper.value, orig_hi + 5.0)

        sub_right = ts - 2.0 * u.V
        np.testing.assert_allclose(sub_right.lower.value, orig_lo - 2.0)
        np.testing.assert_allclose(sub_right.upper.value, orig_hi - 2.0)

        # c - ts reverses order: lower <-> upper and quantiles reversed
        sub_left = 100.0 * u.V - ts
        np.testing.assert_allclose(sub_left.lower.value, 100.0 - orig_hi)
        np.testing.assert_allclose(sub_left.upper.value, 100.0 - orig_lo)
        np.testing.assert_allclose(sub_left.quantiles[0.1].value, 100.0 - orig_hi)
        np.testing.assert_allclose(sub_left.quantiles[0.9].value, 100.0 - orig_lo)

        scaled_pos = ts * 2.0
        np.testing.assert_allclose(scaled_pos.lower.value, orig_lo * 2.0)
        np.testing.assert_allclose(scaled_pos.upper.value, orig_hi * 2.0)

        # Negative scalar swaps lower/upper and reverses quantiles
        scaled_neg = ts * -2.0
        np.testing.assert_allclose(scaled_neg.value, orig_vals * -2.0)
        np.testing.assert_allclose(scaled_neg.lower.value, orig_hi * -2.0)
        np.testing.assert_allclose(scaled_neg.upper.value, orig_lo * -2.0)
        np.testing.assert_allclose(scaled_neg.quantiles[0.1].value, orig_hi * -2.0)
        np.testing.assert_allclose(scaled_neg.quantiles[0.9].value, orig_lo * -2.0)
        assert np.all(scaled_neg.lower.value <= scaled_neg.value)
        assert np.all(scaled_neg.value <= scaled_neg.upper.value)

        div_neg = ts / -2.0
        np.testing.assert_allclose(div_neg.lower.value, orig_hi / -2.0)
        np.testing.assert_allclose(div_neg.upper.value, orig_lo / -2.0)

        converted = ts.to("mV")
        assert converted.unit == u.mV
        assert converted.lower.unit == u.mV
        np.testing.assert_allclose(converted.lower.value, orig_lo * 1000.0)
        np.testing.assert_allclose(converted.upper.value, orig_hi * 1000.0)

    assert caught == []
    # Source must remain unmutated by out-of-place operations.
    np.testing.assert_allclose(ts.value, orig_vals)
    np.testing.assert_allclose(ts.lower.value, orig_lo)
    np.testing.assert_allclose(ts.upper.value, orig_hi)


def test_complex_series_copy_and_slice_propagate_while_arithmetic_drops() -> None:
    cts = TimeSeries(np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]), dt=0.5, unit="V")
    lo = TimeSeries(np.array([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j]), dt=0.5, unit="V")
    hi = TimeSeries(np.array([2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j]), dt=0.5, unit="V")
    cts._set_uncertainty(lower=lo, upper=hi)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copied = cts.copy()
        sliced = cts[1:3]
    assert caught == []
    assert copied.lower is not None
    assert sliced.lower is not None
    assert len(sliced.lower) == 2

    with pytest.warns(UserWarning, match="Uncertainty metadata"):
        scaled = cts * 2.0
    assert not scaled._has_uncertainty()
    assert cts._has_uncertainty()


def test_unsupported_operations_warn_and_drop_without_mutating_source() -> None:
    ts = _make_quantile_ts(64, dt=0.25)
    other = TimeSeries(np.ones(64), t0=1000.0, dt=0.25 * u.s, unit="V")

    for op in [
        lambda: np.sqrt(ts),
        lambda: 1.0 / ts,
        lambda: ts + other,
        lambda: ts * (1.0 + 1.0j),
        lambda: ts.lowpass(1.0),
        lambda: ts.resample(2.0),
        lambda: ts.fft(),
        lambda: ts.psd(fftlength=4.0),
        lambda: ts.asd(fftlength=4.0),
        lambda: ts.smooth(3),
    ]:
        with pytest.warns(UserWarning, match="Uncertainty metadata"):
            res = op()
        assert not res._has_uncertainty()
        assert res.lower is None
        assert res.upper is None
        assert res.quantiles is None
        # Source remains intact
        assert ts._has_uncertainty()


def test_inplace_unsupported_op_warns_and_clears_uncertainty() -> None:
    ts = _make_quantile_ts(16)
    other = TimeSeries(np.ones(16), t0=1000.0, dt=0.25 * u.s, unit="V")

    with pytest.warns(UserWarning, match="Uncertainty metadata"):
        ts += other

    assert not ts._has_uncertainty()
    assert ts.lower is None
    assert ts.quantiles is None


def test_no_warning_when_uncertainty_is_absent_gwpy_parity() -> None:
    ts = TimeSeries(np.linspace(1.0, 10.0, 64), t0=1000.0, dt=0.25 * u.s, unit="V")
    other = TimeSeries(np.ones(64), t0=1000.0, dt=0.25 * u.s, unit="V")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = np.sqrt(ts)
        _ = ts + other
        _ = ts.lowpass(1.0)
        _ = ts.resample(2.0)
        _ = ts.fft()
        _ = ts.smooth(3)
    unc_warnings = [w for w in caught if "Uncertainty metadata" in str(w.message)]
    assert unc_warnings == []


def test_to_timeseriesdict_expands_independent_series_preserving_metadata() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    vals = np.array([10.0, 11.0, 12.0])
    ts = TimeSeries(
        vals, t0_ns=epoch_ns, dt=10 * u.ns, unit="V", name="ch", channel="K1:CH"
    )
    q10 = TimeSeries(
        vals - 1.0, t0_ns=epoch_ns, dt=10 * u.ns, unit="V", name="ch", channel="K1:CH"
    )
    q50 = TimeSeries(
        vals, t0_ns=epoch_ns, dt=10 * u.ns, unit="V", name="ch", channel="K1:CH"
    )
    q90 = TimeSeries(
        vals + 2.0, t0_ns=epoch_ns, dt=10 * u.ns, unit="V", name="ch", channel="K1:CH"
    )
    ts._set_uncertainty(
        quantiles={0.1: q10, 0.5: q50, 0.9: q90},
        center_kind="median",
        uncertainty_kind="quantile_interval",
        interval_levels=(0.1, 0.9),
    )

    parts = ts.to_timeseriesdict()
    assert isinstance(parts, TimeSeriesDict)
    assert list(parts.keys()) == ["value", "lower", "upper", "q0.1", "q0.5", "q0.9"]

    for key, item in parts.items():
        assert isinstance(item, TimeSeries), key
        assert not item._has_uncertainty(), key
        assert item.t0_gps_ns == epoch_ns
        assert item.dt == 10 * u.ns
        assert item.unit == u.V
        assert item.name == "ch"
        assert item.channel == ts.channel

    # Entries are independent copies
    parts["value"][0] = 999.0 * u.V
    assert ts.value[0] == 10.0

    no_q = ts.to_timeseriesdict(include_quantiles=False)
    assert list(no_q.keys()) == ["value", "lower", "upper"]
