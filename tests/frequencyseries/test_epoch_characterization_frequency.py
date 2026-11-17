"""Characterization of epoch handling for FrequencySeries, its Matrix and BifrequencyMap.

Pins behavior at origin/main (#725, v0.3.0 Step 6).  Tests named ``*_current``
record behavior that is not a desired contract and may change in Step 7.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import (
    BifrequencyMap,
    FrequencySeries,
    FrequencySeriesMatrix,
)

OMIT = "omitted"
NONFINITE = [np.nan, np.inf, -np.inf]
GPS = 1.4e9
DF = 0.5
F0 = 1.0
N = 8


def _kw(value):
    return {} if value is OMIT else {"epoch": value}


def _data():
    return np.arange(N, dtype=float) + 1.0


def _epoch_float(obj):
    ep = obj.epoch
    return None if ep is None else float(getattr(ep, "gps", ep))


def _fs(value):
    return FrequencySeries(_data(), df=DF, f0=F0, unit="m", **_kw(value))


def _fsm(value):
    return FrequencySeriesMatrix(_data().reshape(1, 1, N), df=DF, f0=F0, **_kw(value))


class TestFrequencySeriesConstructor:
    @pytest.mark.parametrize("value", [OMIT, None])
    def test_missing_epoch_stays_none(self, value):
        fs = _fs(value)
        assert fs.epoch is None
        assert fs.f0.value == F0
        assert fs.df.value == DF
        assert fs.unit == u.m
        assert np.all(np.isfinite(fs.frequencies.value))

    @pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (GPS, GPS)])
    def test_finite_epoch_is_gps_time(self, value, expected):
        fs = _fs(value)
        assert _epoch_float(fs) == expected
        np.testing.assert_allclose(
            fs.frequencies.value,
            F0 + DF * np.arange(N),
            rtol=0,
            atol=np.spacing(F0 + DF * N),
        )
        assert fs.unit == u.m

    def test_nan_epoch_is_relative_u1(self):
        # changed by U-1 (#725): NaN used to raise RuntimeError("Invalid
        # argument") from the GPS conversion; it now means "unknown" (relative)
        # and is never stored.
        fs = _fs(np.nan)
        assert fs.epoch is None
        assert fs.time_reference == "relative"
        assert np.all(np.isfinite(fs.frequencies.value))
        assert fs.unit == u.m

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_epoch_raises_valueerror_u1(self, value):
        # changed by U-1 (#725): was RuntimeError("Invalid argument").
        with pytest.raises(ValueError, match="must not be infinite"):
            _fs(value)


@pytest.mark.parametrize(
    "value", [OMIT, None, 0, GPS], ids=["omitted", "none", "zero", "gps"]
)
class TestFrequencySeriesOperations:
    def _expected(self, value):
        return None if value in (OMIT, None) else float(value)

    def test_slice_crop_pickle_keep_epoch(self, value):
        fs = _fs(value)
        exp = self._expected(value)
        sl = fs[2:5]
        assert _epoch_float(sl) == exp
        assert sl.f0.value == F0 + 2 * DF
        assert sl.df.value == DF
        cr = fs.crop(2.0, 3.5)
        assert _epoch_float(cr) == exp
        assert cr.f0.value == 2.0
        back = pickle.loads(pickle.dumps(fs))
        assert _epoch_float(back) == exp
        assert back.f0.value == F0
        assert back.unit == u.m
        np.testing.assert_array_equal(back.value, fs.value)

    def test_hdf5_roundtrip_keeps_epoch(self, value, tmp_path):
        pytest.importorskip("h5py")
        fs = _fs(value)
        p = tmp_path / "fs.h5"
        fs.write(p, format="hdf5", path="x")
        back = FrequencySeries.read(p, format="hdf5", path="x")
        assert _epoch_float(back) == self._expected(value)
        assert back.f0.value == F0
        assert back.df.value == DF
        assert back.unit == u.m
        np.testing.assert_array_equal(back.value, fs.value)

    def test_ifft_always_gets_gps_time_epoch_current(self, value):
        # characterizes the public finite values (#725): a missing epoch
        # (None) still projects to a finite GPS-0 Time.
        # changed by U-1 (#725): the result is now relative when the source
        # is, instead of being indistinguishable from absolute GPS 0.
        ts = _fs(value).ifft()
        exp = 0.0 if value in (OMIT, None) else float(value)
        assert ts.t0.value == exp
        assert _epoch_float(ts) == exp
        assert ts.epoch is not None
        assert ts.time_reference == (
            "relative" if value in (OMIT, None) else "absolute"
        )
        assert np.all(np.isfinite(ts.times.value))
        assert ts.dt.value == pytest.approx(1.0 / (2 * (N - 1) * DF))
        assert len(ts) == 2 * (N - 1)


class TestFrequencySeriesMatrix:
    def test_omitted_epoch_is_float_zero_unlike_series_current(self):
        # characterizes current behavior, not a desired contract (#725):
        # FrequencySeries(omitted).epoch is None, the Matrix gives 0.0.
        m = _fsm(OMIT)
        assert m.epoch == 0.0
        assert isinstance(m.epoch, float)
        assert _fs(OMIT).epoch is None
        assert m.f0.value == F0
        assert m.df.value == DF

    def test_none_epoch_is_kept_none(self):
        m = _fsm(None)
        assert m.epoch is None

    def test_zero_epoch_keeps_int_current(self):
        # characterizes current behavior, not a desired contract (#725)
        m = _fsm(0)
        assert m.epoch == 0
        assert isinstance(m.epoch, int)

    def test_gps_epoch(self):
        m = _fsm(GPS)
        assert m.epoch == GPS
        assert m.f0.value == F0
        assert m.df.value == DF

    def test_nan_epoch_is_relative_and_not_stored(self):
        # changed by U-1 (#725): NaN is "no absolute time" (relative) and is
        # never stored; the frequency axis is independent and stays finite.
        m = _fsm(np.nan)
        assert m.epoch == 0.0
        assert m.time_reference == "relative"
        assert np.all(np.isfinite(m.frequencies.value))
        assert m.f0.value == F0
        assert m.df.value == DF

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_epoch_raises_valueerror(self, value):
        # changed by U-1 (#725): +/-Inf used to be stored raw.
        with pytest.raises(ValueError, match="must not be infinite"):
            _fsm(value)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(OMIT, 0.0), (None, None), (0, 0.0), (GPS, GPS)],
        ids=["omitted", "none", "zero", "gps"],
    )
    def test_element_projection_epoch(self, value, expected):
        # characterizes current behavior (#725): the omitted Matrix epoch 0.0
        # becomes a GPS-0 Time on the projected FrequencySeries.
        fs = _fsm(value)[0, 0]
        assert _epoch_float(fs) == expected
        assert fs.f0.value == F0
        assert fs.df.value == DF

    @pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
    def test_slice_and_pickle_keep_matrix_epoch(self, value):
        m = _fsm(value)
        sl = m[:, :, 2:5]
        assert sl.epoch == m.epoch
        assert sl.f0.value == F0 + 2 * DF
        back = pickle.loads(pickle.dumps(m))
        assert back.epoch == m.epoch
        assert back.f0.value == F0
        assert back.df.value == DF


class TestBifrequencyMap:
    @pytest.mark.parametrize("value", [OMIT, None])
    def test_missing_epoch_stays_none(self, value):
        bf = BifrequencyMap(np.ones((4, 4)), unit="m", **_kw(value))
        assert bf.epoch is None
        assert bf.time_reference == "relative"
        assert bf.shape == (4, 4)
        assert bf.unit == u.m
        assert (bf.x0.value, bf.dx.value, bf.y0.value, bf.dy.value) == (
            0.0,
            1.0,
            0.0,
            1.0,
        )

    @pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (GPS, GPS)])
    def test_finite_epoch(self, value, expected):
        bf = BifrequencyMap(np.ones((4, 4)), unit="m", **_kw(value))
        assert _epoch_float(bf) == expected
        assert bf.time_reference == "absolute"

    def test_nan_epoch_is_relative(self):
        bf = BifrequencyMap(np.ones((4, 4)), unit="m", epoch=np.nan)
        assert bf.epoch is None
        assert bf.time_reference == "relative"

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_epoch_raises(self, value):
        with pytest.raises(ValueError, match="must not be infinite"):
            BifrequencyMap(np.ones((4, 4)), unit="m", epoch=value)

    def test_propagate_preserves_input_spectrum_time_reference(self):
        from gwexpy.frequencyseries import FrequencySeries

        freqs = np.array([1.0, 2.0, 3.0, 4.0]) * u.Hz
        bf = BifrequencyMap.from_points(
            np.eye(4), freqs, freqs, unit=u.dimensionless_unscaled
        )

        rel_in = FrequencySeries(np.ones(4), frequencies=freqs, unit=u.m)
        rel_out = bf.propagate(rel_in)
        assert isinstance(rel_out, FrequencySeries)
        assert rel_out.time_reference == "relative"
        assert rel_out.epoch is None
        assert rel_out.unit == u.m
        np.testing.assert_array_equal(rel_out.value, np.ones(4))

        abs_in = FrequencySeries(np.ones(4), frequencies=freqs, unit=u.m, epoch=GPS)
        abs_out = bf.propagate(abs_in)
        assert isinstance(abs_out, FrequencySeries)
        assert abs_out.time_reference == "absolute"
        assert _epoch_float(abs_out) == GPS
        assert abs_out.unit == u.m
        np.testing.assert_array_equal(abs_out.value, np.ones(4))

    @pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
    def test_slice_and_pickle_keep_epoch(self, value):
        bf = BifrequencyMap(np.ones((4, 4)), unit="m", **_kw(value))
        exp = None if value in (OMIT, None) else float(value)
        assert _epoch_float(bf[1:3]) == exp
        assert bf[1:3].shape == (2, 4)
        back = pickle.loads(pickle.dumps(bf))
        assert _epoch_float(back) == exp
        assert back.unit == u.m
