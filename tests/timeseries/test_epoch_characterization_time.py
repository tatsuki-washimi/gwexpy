"""Characterization of t0/epoch handling for TimeSeries and TimeSeriesMatrix.

These tests pin the behavior at origin/main (#725, v0.3.0 Step 6).  They do not
describe a desired contract: tests named ``*_current_*`` record behavior that
Step 7 may deliberately change.  A change here should show up as a diff.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesList, TimeSeriesMatrix

OMIT = "omitted"
NONFINITE = [np.nan, np.inf, -np.inf]
GPS = 1.4e9
N = 64
DT = 0.5


def _tol(x):
    return 4 * np.spacing(max(abs(float(x)), 1.0))


def _kw(kind, value):
    return {} if value is OMIT else {kind: value}


def _data():
    return np.sin(np.arange(N, dtype=float))


def _epoch_float(obj):
    ep = obj.epoch
    return None if ep is None else float(getattr(ep, "gps", ep))


@pytest.mark.parametrize("kind", ["t0", "epoch"])
class TestTimeSeriesConstructor:
    @pytest.mark.parametrize("value", [OMIT, None, 0])
    def test_missing_or_zero_becomes_gps_zero(self, kind, value):
        # The public finite values are unchanged (omitted/None/0 all give t0 0,
        # epoch GPS 0).  changed by U-1 (#725): the states now differ.
        ts = TimeSeries(_data(), dt=DT, unit="m", **_kw(kind, value))
        assert ts.t0.value == 0.0
        assert _epoch_float(ts) == 0.0
        assert np.all(np.isfinite(ts.times.value))
        assert ts.dt.value == DT
        assert ts.unit == u.m
        # changed by U-1 (#725): omitted/None is relative, explicit 0 is absolute
        explicit = value is not OMIT and value is not None
        assert ts.time_reference == ("absolute" if explicit else "relative")

    @pytest.mark.parametrize("value", [GPS])
    def test_finite_gps(self, kind, value):
        ts = TimeSeries(_data(), dt=DT, unit="m", **_kw(kind, value))
        assert ts.t0.value == value
        assert _epoch_float(ts) == value
        t = ts.times.value
        assert np.all(np.isfinite(t))
        np.testing.assert_allclose(
            t - value, np.arange(N) * DT, rtol=0, atol=_tol(value)
        )
        assert ts.dt.value == DT
        assert ts.unit == u.m

    # changed by U-1 (#725): NaN no longer poisons the axis; it constructs a
    # relative series with a finite axis.  +/-Inf is a ValueError (below).
    def test_nan_constructs_relative_with_finite_axis(self, kind):
        ts = TimeSeries(_data(), dt=DT, unit="m", **_kw(kind, np.nan))
        assert ts.time_reference == "relative"
        assert ts.t0.value == 0.0
        assert ts.x0.value == 0.0
        assert np.all(np.isfinite(ts.times.value))
        assert _epoch_float(ts) == 0.0
        assert ts.dt.value == DT
        assert ts.unit == u.m

    # changed by U-1 (#725): construction now fails with a clear ValueError.
    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_raises_clear_value_error(self, kind, value):
        with pytest.raises(ValueError, match="must not be infinite"):
            TimeSeries(_data(), dt=DT, unit="m", **_kw(kind, value))

    # changed by U-1 (#725): a NaN-constructed (relative) series transforms
    # like any other; nothing raises and every coordinate is finite.
    def test_nan_transforms_work(self, kind):
        ts = TimeSeries(_data(), dt=DT, unit="m", **_kw(kind, np.nan))
        fs = ts.fft()
        assert fs.unit == u.m
        assert np.all(np.isfinite(fs.frequencies.value))
        psd = ts.psd(fftlength=4.0)
        assert psd.unit == u.m**2 / u.Hz
        assert psd.df.value == pytest.approx(0.25)
        sp = ts.spectrogram(4.0, fftlength=2.0)
        assert np.all(np.isfinite(sp.times.value))
        back = pickle.loads(pickle.dumps(ts))
        assert back.time_reference == "relative"
        assert back.t0.value == 0.0
        assert back.dt.value == DT

    # changed by U-1 (#725): the public t0 stays finite and the state is kept.
    def test_nan_slice_resample_keep_finite_t0_and_state(self, kind):
        ts = TimeSeries(_data(), dt=DT, unit="m", **_kw(kind, np.nan))
        sl = ts[10:20]
        assert len(sl) == 10
        assert sl.t0.value == pytest.approx(10 * DT, abs=_tol(10 * DT))
        assert sl.time_reference == "relative"
        rs = ts.resample(1.0)
        assert len(rs) == N // 2
        assert rs.t0.value == 0.0
        assert rs.time_reference == "relative"
        assert rs.dt.value == 1.0
        assert rs.unit == u.m


@pytest.mark.parametrize(
    ("value", "expected"),
    [(OMIT, 0.0), (0, 0.0), (GPS, GPS)],
    ids=["omitted", "zero", "gps"],
)
class TestTimeSeriesTransforms:
    def _ts(self, value):
        return TimeSeries(_data(), dt=DT, unit="m", **_kw("t0", value))

    def test_fft_epoch_is_gps_zero_even_if_omitted_current(self, value, expected):
        ts = self._ts(value)
        fs = ts.fft()
        assert _epoch_float(fs) == expected
        assert fs.f0.value == 0.0
        assert fs.df.value == pytest.approx(1.0 / (N * DT))
        assert fs.unit == u.m

    def test_fft_ifft_roundtrip_restores_t0(self, value, expected):
        ts = self._ts(value)
        back = ts.fft().ifft()
        assert back.t0.value == expected
        assert back.dt.value == DT
        assert len(back) == N
        assert np.all(np.isfinite(back.times.value))
        assert back.unit == u.m
        np.testing.assert_allclose(
            back.value, ts.value, rtol=0, atol=1e-12 * np.max(np.abs(ts.value))
        )

    def test_psd_spectrogram_carry_epoch(self, value, expected):
        ts = self._ts(value)
        psd = ts.psd(fftlength=4.0)
        assert _epoch_float(psd) == expected
        assert psd.df.value == pytest.approx(0.25)
        assert psd.unit == u.m**2 / u.Hz
        sp = ts.spectrogram(4.0, fftlength=2.0)
        assert sp.t0.value == expected
        assert np.all(np.isfinite(sp.times.value))

    def test_slice_crop_resample_shift_t0(self, value, expected):
        ts = self._ts(value)
        sl = ts[10:20]
        assert sl.t0.value == pytest.approx(expected + 10 * DT, abs=_tol(expected))
        assert len(sl) == 10
        assert sl.dt.value == DT
        cr = ts.crop(expected + 5, None)
        assert cr.t0.value == pytest.approx(expected + 5, abs=_tol(expected))
        assert np.all(np.isfinite(cr.times.value))
        assert cr.unit == u.m
        rs = ts.resample(1.0)
        assert rs.t0.value == expected
        assert rs.dt.value == 1.0
        assert len(rs) == N // 2

    def test_pickle_roundtrip(self, value, expected):
        ts = self._ts(value)
        back = pickle.loads(pickle.dumps(ts))
        assert back.t0.value == expected
        assert back.dt.value == DT
        assert back.unit == u.m
        np.testing.assert_array_equal(back.value, ts.value)

    def test_hdf5_roundtrip(self, value, expected, tmp_path):
        pytest.importorskip("h5py")
        ts = self._ts(value)
        p = tmp_path / "ts.h5"
        ts.write(p, format="hdf5", path="x")
        back = TimeSeries.read(p, format="hdf5", path="x")
        assert back.t0.value == expected
        assert back.dt.value == DT
        assert back.unit == u.m
        np.testing.assert_array_equal(back.value, ts.value)

    # changed by U-1 (#725): a non-finite t0 can no longer be written.  NaN is a
    # relative series and round-trips with t0 0 (the HDF5 file stores an
    # explicit x0, so the relative state itself is lost until Step 7c);
    # +/-Inf fails at construction.
    def test_hdf5_nan_round_trips_inf_cannot_be_built(self, value, expected, tmp_path):
        pytest.importorskip("h5py")
        ts = TimeSeries(_data(), dt=DT, unit="m", t0=np.nan)
        p = tmp_path / "nan.h5"
        ts.write(p, format="hdf5", path="x")
        back = TimeSeries.read(p, format="hdf5", path="x")
        assert back.t0.value == 0.0
        assert back.dt.value == DT
        assert back.unit == u.m
        np.testing.assert_array_equal(back.value, ts.value)
        for bad in (np.inf, -np.inf):
            with pytest.raises(ValueError, match="must not be infinite"):
                TimeSeries(_data(), dt=DT, t0=bad)


@pytest.mark.parametrize("kind", ["t0", "epoch"])
class TestTimeSeriesMatrixConstructor:
    def _m(self, kind, value):
        return TimeSeriesMatrix(
            _data().reshape(1, 1, N), dt=DT, unit="m", **_kw(kind, value)
        )

    @pytest.mark.parametrize("value", [OMIT, None])
    def test_missing_is_gps_zero_float(self, kind, value):
        # characterizes current behavior, not a desired contract (#725)
        m = self._m(kind, value)
        assert m.t0.value == 0.0
        assert m.epoch == 0.0
        assert np.all(np.isfinite(m.times.value))
        assert m.dt.value == DT

    def test_zero_keeps_int_epoch_current(self, kind):
        # characterizes current behavior, not a desired contract (#725):
        # epoch is stored raw (int 0), unlike the omitted case (float 0.0).
        m = self._m(kind, 0)
        assert m.t0.value == 0.0
        assert m.epoch == 0
        assert isinstance(m.epoch, int)

    def test_finite_gps(self, kind):
        m = self._m(kind, GPS)
        assert m.t0.value == GPS
        assert m.epoch == GPS
        np.testing.assert_allclose(
            m.times.value - GPS, np.arange(N) * DT, rtol=0, atol=_tol(GPS)
        )
        assert m.dt.value == DT

    def test_nan_is_relative_with_finite_axis(self, kind):
        # changed by U-1 (#725): NaN used to fail in ``arange``; it is now
        # "no absolute time" (relative) and the axis stays finite, like
        # TimeSeries.
        m = self._m(kind, np.nan)
        assert m.time_reference == "relative"
        assert m.t0.value == 0.0
        assert m.epoch == 0.0
        assert np.all(np.isfinite(m.times.value))
        assert m.dt.value == DT

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_raises_clear_valueerror(self, kind, value):
        # changed by U-1 (#725): the message is no longer numpy's ``arange``.
        with pytest.raises(ValueError, match="must not be infinite"):
            self._m(kind, value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(OMIT, 0.0), (0, 0.0), (GPS, GPS)],
    ids=["omitted", "zero", "gps"],
)
class TestTimeSeriesMatrixConversions:
    def _ts(self, value):
        return TimeSeries(_data(), dt=DT, unit="m", **_kw("t0", value))

    def _m(self, value):
        return TimeSeriesMatrix(
            _data().reshape(1, 1, N), dt=DT, unit="m", **_kw("t0", value)
        )

    def test_matrix_to_series_preserves_axis(self, value, expected):
        ts = self._m(value)[0, 0]
        assert ts.t0.value == expected
        assert ts.dt.value == DT
        assert ts.unit == u.m
        assert np.all(np.isfinite(ts.times.value))

    def test_series_list_to_matrix_epoch_is_quantity_current(self, value, expected):
        # characterizes current behavior, not a desired contract (#725):
        # the epoch type differs from the direct constructor (float/int).
        m = TimeSeriesList([self._ts(value)]).to_matrix()
        assert m.t0.value == expected
        assert isinstance(m.epoch, u.Quantity)
        assert m.epoch.value == expected
        assert m.dt.value == DT

    def test_matrix_pickle_roundtrip(self, value, expected):
        m = self._m(value)
        back = pickle.loads(pickle.dumps(m))
        assert back.t0.value == expected
        assert back.epoch == m.epoch
        assert back.dt.value == DT

    def test_matrix_slice_moves_t0_not_epoch_current(self, value, expected):
        # characterizes current behavior, not a desired contract (#725):
        # the coordinate origin moves while raw ``epoch`` stays put, so the
        # two timing authorities disagree after slicing.
        m = self._m(value)
        sl = m[:, :, 10:20]
        assert sl.t0.value == pytest.approx(expected + 10 * DT, abs=_tol(expected))
        assert sl.epoch == m.epoch
        assert sl.dt.value == DT
