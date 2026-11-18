"""Characterization of epoch/t0 handling for Spectrogram and SpectrogramMatrix.

Pins behavior at origin/main (#725, v0.3.0 Step 6).  Tests named ``*_current``
record behavior that is not a desired contract and may change in Step 7.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram, SpectrogramMatrix
from gwexpy.timeseries import TimeSeries

OMIT = "omitted"
NONFINITE = [np.nan, np.inf, -np.inf]
GPS = 1.4e9
NT, NF = 6, 4
DT = 1.0
DF = 2.0


def _tol(x):
    return 4 * np.spacing(max(abs(float(x)), 1.0))


def _kw(kind, value):
    return {} if value is OMIT else {kind: value}


def _data():
    return np.arange(NT * NF, dtype=float).reshape(NT, NF) + 1.0


def _epoch_float(obj):
    ep = obj.epoch
    return None if ep is None else float(getattr(ep, "gps", getattr(ep, "value", ep)))


def _sg(kind, value):
    return Spectrogram(_data(), dt=DT, f0=0.0, df=DF, unit="m", **_kw(kind, value))


def _sgm(epoch, base):
    times = (np.arange(NT, dtype=float) + base) * u.s
    return SpectrogramMatrix(
        _data().reshape(1, 1, NT, NF),
        times=times,
        frequencies=np.arange(NF) * DF * u.Hz,
        unit="m",
        **_kw("epoch", epoch),
    )


@pytest.mark.parametrize("kind", ["t0", "epoch"])
class TestSpectrogramConstructor:
    @pytest.mark.parametrize("value", [OMIT, None, 0])
    def test_missing_or_zero_is_gps_zero_time(self, kind, value):
        # characterizes the public finite values (#725): they are GPS 0 Time.
        # changed by Plan I (#725): omitted/0 is absolute, explicit None is relative.
        sg = _sg(kind, value)
        assert sg.time_reference == ("relative" if value is None else "absolute")
        assert _epoch_float(sg) == 0.0
        assert sg.t0.value == 0.0
        assert np.all(np.isfinite(sg.times.value))
        assert sg.dt.value == DT
        assert sg.df.value == DF
        assert sg.unit == u.m

    def test_finite_gps(self, kind):
        sg = _sg(kind, GPS)
        assert _epoch_float(sg) == GPS
        np.testing.assert_allclose(
            sg.times.value - GPS, np.arange(NT) * DT, rtol=0, atol=_tol(GPS)
        )
        assert sg.dt.value == DT
        assert sg.df.value == DF

    def test_nan_is_relative_u1(self, kind):
        # changed by U-1 (#725): NaN used to construct and make epoch/times
        # raise later; it now means "unknown" (relative) and is never stored.
        sg = _sg(kind, np.nan)
        assert sg.time_reference == "relative"
        assert np.isfinite(sg.t0.value)
        assert np.all(np.isfinite(sg.times.value))
        assert _epoch_float(sg) == 0.0
        assert sg.dt.value == DT
        assert sg.unit == u.m

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_raises_valueerror_u1(self, kind, value):
        # changed by U-1 (#725): +/-Inf used to construct and make
        # epoch/times raise later.
        with pytest.raises(ValueError, match="must not be infinite"):
            _sg(kind, value)


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        (None, OMIT, 0.0),
        ("epoch", None, 0.0),
        ("epoch", 0, 0.0),
        ("epoch", GPS, GPS),
    ],
    ids=["omitted", "none", "zero", "gps"],
)
class TestSpectrogramOperations:
    def _make(self, kind, value):
        return _sg(kind, value) if kind else _sg("epoch", OMIT)

    def test_pickle_roundtrip(self, kind, value, expected):
        sg = self._make(kind, value)
        back = pickle.loads(pickle.dumps(sg))
        assert _epoch_float(back) == expected
        assert back.t0.value == expected
        assert back.dt.value == DT
        assert back.df.value == DF
        assert back.unit == u.m
        np.testing.assert_array_equal(back.value, sg.value)

    def test_hdf5_roundtrip(self, kind, value, expected, tmp_path):
        pytest.importorskip("h5py")
        sg = self._make(kind, value)
        p = tmp_path / "sg.h5"
        sg.write(p, format="hdf5", path="x")
        back = Spectrogram.read(p, format="hdf5", path="x")
        assert _epoch_float(back) == expected
        assert back.dt.value == DT
        assert back.unit == u.m
        np.testing.assert_array_equal(back.value, sg.value)

    def test_slice_moves_t0_and_epoch_follows(self, kind, value, expected):
        sg = self._make(kind, value)
        sl = sg[1:4]
        assert sl.t0.value == pytest.approx(expected + 1.0, abs=_tol(expected))
        # characterizes current behavior (#725): epoch is re-derived from the
        # sliced time axis; for a zero origin it carries a ~1e-12 s artifact.
        assert _epoch_float(sl) == pytest.approx(expected + 1.0, abs=1e-9)
        assert sl.shape == (3, NF)
        assert sl.df.value == DF
        assert sl.unit == u.m

    def test_to_series_lists_carry_epoch(self, kind, value, expected):
        sg = self._make(kind, value)
        ts_lists = sg.to_timeseries_list()
        assert len(ts_lists) == 2
        first = ts_lists[0]
        assert len(first) == NF
        for ts in first:
            assert ts.t0.value == expected
            assert _epoch_float(ts) == expected
        fs_lists = sg.to_frequencyseries_list()
        assert len(fs_lists) == 2
        for fs in fs_lists[0]:
            assert _epoch_float(fs) == expected

    def test_from_timeseries_spectrogram_carries_t0(self, kind, value, expected):
        ts = TimeSeries(np.sin(np.arange(64.0)), dt=0.5, t0=expected, unit="m")
        sg = ts.spectrogram(4.0, fftlength=2.0)
        assert sg.t0.value == expected
        assert _epoch_float(sg) == expected
        assert np.all(np.isfinite(sg.times.value))
        assert sg.dt.value == 4.0
        assert sg.unit == u.m**2 / u.Hz


class TestSpectrogramMatrix:
    @pytest.mark.parametrize(
        ("value", "expected", "typ"),
        [(OMIT, 0.0, float), (0, 0, int), (GPS, GPS, float)],
        ids=["omitted", "zero", "gps"],
    )
    def test_epoch_stored_raw(self, value, expected, typ):
        # characterizes current behavior, not a desired contract (#725)
        m = _sgm(value, 0.0)
        assert m.epoch == expected
        assert isinstance(m.epoch, typ)
        assert np.all(np.isfinite(np.asarray(m.times.value)))

    def test_none_epoch_kept(self):
        assert _sgm(None, 0.0).epoch is None

    def test_nan_epoch_is_relative_and_not_stored(self):
        # changed by U-1 (#725): NaN is "no absolute time" (relative) and is
        # never stored; times stay finite.
        m = _sgm(np.nan, GPS)
        assert m.epoch == 0.0
        assert m.time_reference == "relative"
        assert np.all(np.isfinite(np.asarray(m.times.value)))

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_epoch_raises_valueerror(self, value):
        # changed by U-1 (#725): +/-Inf used to be stored raw.
        with pytest.raises(ValueError, match="must not be infinite"):
            _sgm(value, GPS)

    @pytest.mark.parametrize(
        ("value", "base", "expected"),
        [(OMIT, 0.0, 0.0), (None, 0.0, 0.0), (0, 0.0, 0.0), (GPS, GPS, GPS)],
        ids=["omitted", "none", "zero", "gps"],
    )
    def test_element_projection(self, value, base, expected):
        # characterizes current behavior (#725): epoch=None projects to a
        # GPS-0 Time on the Spectrogram.
        sg = _sgm(value, base)[0, 0]
        assert _epoch_float(sg) == expected
        assert sg.times.value[0] == expected
        assert sg.dt.value == DT
        assert sg.unit == u.m

    def test_epoch_times_mismatch_projection_uses_explicit_axis(self):
        # changed by Step 7 (#725): the matrix epoch is the coordinate origin,
        # while the explicit times axis supplies the selected sample's time.
        m = _sgm(0, GPS)
        sg = m[0, 0]
        assert sg.time_reference == "absolute"
        assert sg.t0.value == pytest.approx(GPS, rel=0, abs=_tol(GPS))
        assert _epoch_float(sg) == pytest.approx(GPS, rel=0, abs=_tol(GPS))

    def test_time_slice_rejected_current(self):
        # characterizes current behavior (#725): B0 supports structural
        # slicing only.
        with pytest.raises(ValueError, match="structural slicing"):
            _sgm(OMIT, 0.0)[:, :, 1:4]

    @pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
    def test_pickle_roundtrip(self, value):
        base = GPS if value == GPS else 0.0
        m = _sgm(value, base)
        back = pickle.loads(pickle.dumps(m))
        assert back.epoch == m.epoch
        np.testing.assert_array_equal(
            np.asarray(back.times.value), np.asarray(m.times.value)
        )
        np.testing.assert_array_equal(np.asarray(back.value), np.asarray(m.value))

    def test_spectrogram_to_matrix_epoch_is_gps_zero_current(self):
        # characterizes current behavior, not a desired contract (#725):
        # the GPS-anchored Spectrogram's times survive, but the Matrix epoch
        # is reset to 0.0 (the two timing authorities disagree).
        sg = _sg("epoch", GPS)
        m = SpectrogramMatrix(
            np.asarray(sg.value).reshape(1, 1, NT, NF),
            times=sg.times,
            frequencies=sg.frequencies,
            unit="m",
        )
        assert m.epoch == 0.0
        assert np.asarray(m.times.value)[0] == GPS
