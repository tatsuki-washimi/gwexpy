"""Spectrogram timing-reference state (#725, v0.3.0 Step 7b, decision U-1).

Contract under test (a Spectrogram's ``t0``/``epoch``/``x0`` are one time
origin; ``times`` is its time axis, ``frequencies`` its frequency axis):

* ``t0``/``epoch``/``x0`` omitted, ``None`` or ``NaN`` -> ``"relative"``
* finite value (including an explicit ``0.0``) or an explicit ``times`` axis
  -> ``"absolute"``
* ``+Inf`` / ``-Inf`` -> ``ValueError``
* assignment accepts finite values (state unchanged), rejects ``NaN``/``±Inf``
* the public finite ``epoch``/``t0``/``times`` are the GWpy ones in both states
* the state survives ``OPERATIONS`` and the TimeSeries spectrogram results
* the state never changes a computed value, normalization or axis

Tolerances come from ``np.spacing`` of the quantity compared, or the
comparison is exact because both sides run the same computation (R-15).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from astropy import units as u
from gwpy.spectrogram import Spectrogram as GwpySpectrogram
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.spectrogram import Spectrogram
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
)
from gwexpy.timeseries import TimeSeries

REL = TIME_REFERENCE_RELATIVE
ABS = TIME_REFERENCE_ABSOLUTE
GPS = 1.4e9
NT, NF = 8, 6
DT = 1.0
DF = 2.0
UNIT = u.m

KINDS = ["relative", "absolute_zero", "absolute_gps"]
EXPECTED_STATE = {"relative": REL, "absolute_zero": ABS, "absolute_gps": ABS}
ORIGIN = {"relative": 0.0, "absolute_zero": 0.0, "absolute_gps": GPS}


def _data() -> np.ndarray:
    return np.random.default_rng(4).random((NT, NF)) + 1.0


def _make(kind: str, **extra) -> Spectrogram:
    kw: dict = {}
    if kind == "absolute_zero":
        kw["epoch"] = 0.0
    elif kind == "absolute_gps":
        kw["epoch"] = GPS
    kw.update(extra)
    return Spectrogram(_data(), dt=DT, f0=0.0, df=DF, unit=UNIT, name="x", **kw)


def _tol(*values: float) -> float:
    return 4 * np.spacing(max(1.0, *(abs(float(v)) for v in values)))


def _t0(sg) -> float:
    return float(sg.t0.to_value(u.s))


def _epoch(sg) -> float:
    return float(sg.epoch.gps)


def _assert_origin(sg, expected: float) -> None:
    """Origin of the time axis, to a few ULP of its magnitude."""
    assert abs(_t0(sg) - expected) <= _tol(expected), (_t0(sg), expected)


# ------------------------------------------------------------ construction


@pytest.mark.parametrize("argument", ["t0", "epoch", "x0"])
class TestConstructor:
    @pytest.mark.parametrize("value", ["omitted", None, np.nan, np.nan * u.s])
    def test_missing_or_nan_is_relative(self, argument, value):
        kw = {} if isinstance(value, str) else {argument: value}
        sg = Spectrogram(_data(), dt=DT, f0=0.0, df=DF, unit=UNIT, **kw)
        assert sg.time_reference == REL
        # the public values are finite and unchanged (GWpy)
        assert _t0(sg) == 0.0
        assert _epoch(sg) == 0.0
        assert np.all(np.isfinite(sg.times.value))
        assert sg.dt.value == DT
        assert sg.df.value == DF
        assert sg.unit == UNIT

    @pytest.mark.parametrize("value", [0, 0.0, -0.0, GPS, GPS * u.s])
    def test_finite_is_absolute_including_zero(self, argument, value):
        sg = Spectrogram(_data(), dt=DT, f0=0.0, df=DF, unit=UNIT, **{argument: value})
        assert sg.time_reference == ABS
        expected = float(u.Quantity(value, u.s).value)
        _assert_origin(sg, expected)
        assert sg.dt.value == DT
        assert sg.df.value == DF
        assert sg.unit == UNIT

    @pytest.mark.parametrize("value", [np.inf, -np.inf, np.inf * u.s])
    def test_infinite_raises_valueerror(self, argument, value):
        with pytest.raises(ValueError, match="must not be infinite"):
            Spectrogram(_data(), dt=DT, f0=0.0, df=DF, **{argument: value})


class TestConstructorAxes:
    def test_explicit_times_is_absolute(self):
        times = (np.arange(NT) + 10.0) * u.s
        sg = Spectrogram(_data(), times=times, f0=0.0, df=DF, unit=UNIT)
        assert sg.time_reference == ABS
        np.testing.assert_array_equal(sg.times.value, times.value)

    def test_explicit_times_with_nan_epoch_is_absolute(self):
        times = (np.arange(NT) + 10.0) * u.s
        sg = Spectrogram(_data(), times=times, epoch=np.nan, f0=0.0, df=DF)
        assert sg.time_reference == ABS
        np.testing.assert_array_equal(sg.times.value, times.value)

    def test_frequency_axis_does_not_decide_the_state(self):
        sg = Spectrogram(_data(), dt=DT, frequencies=np.arange(NF) * DF * u.Hz)
        assert sg.time_reference == REL

    def test_positional_t0_follows_the_gwpy_signature(self):
        sg = Spectrogram(_data(), UNIT, GPS, DT, 0.0, DF)
        assert sg.time_reference == ABS
        assert sg.unit == UNIT
        _assert_origin(sg, GPS)
        sg_nan = Spectrogram(_data(), UNIT, np.nan, DT, 0.0, DF)
        assert sg_nan.time_reference == REL
        assert _t0(sg_nan) == 0.0

    def test_conflicting_t0_and_epoch_is_left_to_gwpy(self):
        with pytest.raises(ValueError, match="only one of epoch or t0"):
            Spectrogram(_data(), dt=DT, t0=1.0, epoch=2.0)


class TestConstructorFromSeries:
    def test_relative_stays_relative(self):
        assert Spectrogram(_make("relative")).time_reference == REL

    def test_absolute_stays_absolute(self):
        assert Spectrogram(_make("absolute_zero")).time_reference == ABS

    def test_explicit_epoch_makes_absolute_and_leaves_source(self):
        rel = _make("relative")
        out = Spectrogram(rel, epoch=GPS)
        assert out.time_reference == ABS
        _assert_origin(out, GPS)
        assert rel.time_reference == REL
        assert _t0(rel) == 0.0

    def test_nan_makes_relative(self):
        ab = _make("absolute_gps")
        assert Spectrogram(ab, t0=np.nan).time_reference == REL
        assert ab.time_reference == ABS

    def test_view_of_gwpy_spectrogram_defaults_to_absolute(self):
        g = GwpySpectrogram(_data(), dt=DT, f0=0.0, df=DF, epoch=GPS)
        assert g.view(Spectrogram).time_reference == ABS


# ----------------------------------------------------------------- setters


@pytest.mark.parametrize("alias", ["t0", "epoch", "x0"])
class TestSetter:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, np.nan * u.s])
    def test_nonfinite_assignment_raises_and_changes_nothing(self, alias, kind, bad):
        sg = _make(kind)
        with pytest.raises(ValueError, match="finite"):
            setattr(sg, alias, bad)
        _assert_origin(sg, ORIGIN[kind])
        assert sg.time_reference == EXPECTED_STATE[kind]
        assert np.all(np.isfinite(sg.times.value))

    @pytest.mark.parametrize("kind", KINDS)
    def test_finite_assignment_keeps_state(self, alias, kind):
        sg = _make(kind)
        setattr(sg, alias, 100.0)
        _assert_origin(sg, 100.0)
        assert sg.time_reference == EXPECTED_STATE[kind]


# ----------------------------------------------------------------- transforms

# name -> (function, expected unit as a function of the source unit)
OPERATIONS = {
    "row slice": (lambda t: t[1:4], lambda un: un),
    "step slice": (lambda t: t[::2], lambda un: un),
    "column slice": (lambda t: t[:, 1:3], lambda un: un),
    "crop": (lambda t: t.crop(_t0(t) + 2, _t0(t) + 5), lambda un: un),
    "crop_frequencies": (lambda t: t.crop_frequencies(2, 8), lambda un: un),
    "copy": (lambda t: t.copy(), lambda un: un),
    "copy.copy": (copy.copy, lambda un: un),
    "deepcopy": (copy.deepcopy, lambda un: un),
    "*2": (lambda t: t * 2, lambda un: un),
    "t+t": (lambda t: t + t, lambda un: un),
    "abs": (abs, lambda un: un),
    "to mm": (lambda t: t.to("mm"), lambda un: u.mm),
    "astype": (lambda t: t.astype(np.float32), lambda un: un),
    "ratio": (lambda t: t.ratio("median"), lambda un: u.dimensionless_unscaled),
    "radian": (lambda t: t.radian(), lambda un: u.rad),
    "degree": (lambda t: t.degree(), lambda un: u.deg),
    "normalize": (lambda t: t.normalize(), lambda un: u.dimensionless_unscaled),
    "clean": (lambda t: t.clean(), lambda un: un),
    "rebin(dt)": (lambda t: t.rebin(dt=2.0), lambda un: un),
    "rebin(df)": (lambda t: t.rebin(df=4.0), lambda un: un),
    "Spectrogram(sg)": (lambda t: Spectrogram(t), lambda un: un),
}


class TestOperations:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("name", list(OPERATIONS))
    def test_state_unit_and_axes(self, kind, name):
        fn, unit_of = OPERATIONS[name]
        src = _make(kind)
        out = fn(src)
        assert isinstance(out, Spectrogram), name
        assert out.time_reference == EXPECTED_STATE[kind], name
        assert out.unit == unit_of(src.unit), name
        assert out.df.unit == u.Hz
        assert out.dt.unit == u.s
        assert np.all(np.isfinite(out.times.value)), name
        assert np.all(np.isfinite(out.frequencies.value)), name
        # a relative axis stays finite and relative to the (cropped) origin
        assert out.times.value[0] >= src.times.value[0] - _tol(src.times.value[0])
        # the source is never modified
        assert src.time_reference == EXPECTED_STATE[kind]
        np.testing.assert_array_equal(src.value, _data())

    @pytest.mark.parametrize("kind", KINDS)
    def test_row_slice_moves_origin_not_the_state(self, kind):
        out = _make(kind)[1:4]
        _assert_origin(out, ORIGIN[kind] + 1.0)
        assert out.dt.value == DT
        assert out.df.value == DF
        assert out.shape == (3, NF)
        assert out.time_reference == EXPECTED_STATE[kind]

    @pytest.mark.parametrize("kind", KINDS)
    def test_column_slice_keeps_time_axis(self, kind):
        src = _make(kind)
        out = src[:, 1:3]
        np.testing.assert_array_equal(out.times.value, src.times.value)
        assert out.f0.value == DF
        assert out.df.value == DF
        assert out.time_reference == EXPECTED_STATE[kind]

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("which", ["dt", "df"])
    def test_rebin_axes(self, kind, which):
        src = _make(kind)
        out = src.rebin(**{which: 2.0 * (DT if which == "dt" else DF)})
        if which == "dt":
            assert out.shape == (NT // 2, NF)
            assert out.dt.value == 2 * DT
            # bin centres: mean of the merged sample times
            assert abs(out.times.value[0] - np.mean(src.times.value[:2])) <= _tol(
                out.times.value[0]
            )
        else:
            assert out.shape == (NT, NF // 2)
            np.testing.assert_array_equal(out.times.value, src.times.value)
        assert out.time_reference == EXPECTED_STATE[kind]

    @pytest.mark.parametrize("kind", KINDS)
    def test_to_series_lists_carry_state(self, kind):
        src = _make(kind)
        ts_list, freqs = src.to_timeseries_list()
        assert len(ts_list) == NF
        assert freqs.unit == u.Hz
        for ts in ts_list:
            assert ts.time_reference == EXPECTED_STATE[kind]
            assert ts.unit == UNIT
            np.testing.assert_array_equal(ts.times.value, src.times.value)
        fs_list, times = src.to_frequencyseries_list()
        assert len(fs_list) == NT
        assert times.unit == u.s
        for fs in fs_list:
            assert fs.time_reference == EXPECTED_STATE[kind]
            assert fs.unit == UNIT
            np.testing.assert_array_equal(fs.frequencies.value, src.frequencies.value)

    @pytest.mark.parametrize("kind", KINDS)
    def test_pickle(self, kind):
        src = _make(kind)
        src.provenance = None
        back = pickle.loads(pickle.dumps(src))
        if kind == "relative":
            assert isinstance(back, Spectrogram)
            assert back.time_reference == REL
        else:
            # absolute payloads keep their GWpy-only portability (unchanged)
            assert isinstance(back, GwpySpectrogram)
        np.testing.assert_array_equal(back.value, src.value)
        np.testing.assert_array_equal(back.times.value, src.times.value)
        np.testing.assert_array_equal(back.frequencies.value, src.frequencies.value)
        assert back.unit == src.unit
        assert back.name == "x"

    def test_relative_pickle_keeps_provenance(self):
        src = _make("relative")
        pytest.importorskip("gwexpy.spectrogram.provenance")
        back = pickle.loads(pickle.dumps(src))
        assert back.provenance == src.provenance
        assert back.time_reference == REL

    def test_mixed_arithmetic_follows_first_operand(self):
        rel, ab = _make("relative"), _make("absolute_gps")
        assert (rel + ab).time_reference == REL
        assert (ab + rel).time_reference == ABS

    def test_gwpy_row_and_column_results_have_no_state_current(self):
        # characterizes current behavior (#725): GWpy's _rowclass/_columnclass
        # and its reductions return plain GWpy series, which cannot hold the
        # state; their finite epoch is the GWpy projection.
        rel = _make("relative")
        for out in (rel[2], rel[:, 1], rel.mean(axis=0), rel.percentile(50)):
            assert not hasattr(out, "time_reference")

    def test_state_independent_of_data_scale(self):
        # R-15: the state is a property of the timing, not of the values
        sg = Spectrogram(_data() * 1e-42, dt=DT, f0=0.0, df=DF, unit=UNIT)
        assert sg.time_reference == REL
        assert (sg * 1e-20).time_reference == REL


# ------------------------------------------- TimeSeries -> spectrogram results


def _ts(kind: str) -> TimeSeries:
    kw: dict = {}
    if kind == "absolute_zero":
        kw["t0"] = 0.0
    elif kind == "absolute_gps":
        kw["t0"] = GPS
    x = np.random.default_rng(0).standard_normal(4096)
    return TimeSeries(x, sample_rate=256.0, unit=UNIT, **kw)


SPECTROGRAMS = {
    "spectrogram": lambda t: t.spectrogram(4, fftlength=1),
    "spectrogram2": lambda t: t.spectrogram2(1),
    "rayleigh_spectrogram": lambda t: t.rayleigh_spectrogram(4, fftlength=1),
}


class TestTimeSeriesSpectrograms:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("name", list(SPECTROGRAMS))
    def test_result_carries_state_and_axes(self, kind, name):
        ts = _ts(kind)
        sg = SPECTROGRAMS[name](ts)
        assert isinstance(sg, Spectrogram)
        assert sg.time_reference == EXPECTED_STATE[kind]
        # origin and finite public values are the source's
        assert _t0(sg) == _t0(ts)
        assert float(sg.epoch.gps) == float(ts.epoch.gps)
        assert np.all(np.isfinite(sg.times.value))
        assert sg.df.unit == u.Hz
        assert ts.time_reference == EXPECTED_STATE[kind]

    def test_spectrogram_unit_and_stride(self):
        sg = _ts("relative").spectrogram(4, fftlength=1)
        assert sg.unit == UNIT**2 / u.Hz
        assert sg.dt.value == 4.0
        assert sg.df.value == 1.0
        assert sg.f0.value == 0.0

    @pytest.mark.parametrize("name", list(SPECTROGRAMS))
    def test_values_do_not_depend_on_the_state(self, name):
        # AC-10-2: identical data, relative vs absolute -> identical numbers,
        # axes (up to the origin shift) and units.
        rel = SPECTROGRAMS[name](_ts("relative"))
        for kind in ("absolute_zero", "absolute_gps"):
            ab = SPECTROGRAMS[name](_ts(kind))
            np.testing.assert_array_equal(rel.value, ab.value)
            np.testing.assert_array_equal(rel.frequencies.value, ab.frequencies.value)
            np.testing.assert_allclose(
                ab.times.value - ab.times.value[0],
                rel.times.value - rel.times.value[0],
                rtol=0,
                atol=_tol(ab.times.value[-1]),
            )
            assert rel.unit == ab.unit
            assert rel.dt == ab.dt
            assert rel.df == ab.df

    def test_spectrogram_matches_plain_gwpy(self):
        x = np.random.default_rng(0).standard_normal(4096)
        ours = TimeSeries(x, sample_rate=256.0, unit=UNIT).spectrogram(4, fftlength=1)
        theirs = GwpyTimeSeries(x, sample_rate=256.0, unit=UNIT).spectrogram(
            4, fftlength=1
        )
        np.testing.assert_array_equal(ours.value, theirs.value)
        np.testing.assert_array_equal(ours.times.value, theirs.times.value)
        np.testing.assert_array_equal(ours.frequencies.value, theirs.frequencies.value)
        assert ours.unit == theirs.unit

    @pytest.mark.parametrize("kind", KINDS)
    def test_cropped_relative_origin_is_kept(self, kind):
        ts = _ts(kind).crop(_t0(_ts(kind)) + 3.0, _t0(_ts(kind)) + 15.0)
        assert ts.time_reference == EXPECTED_STATE[kind]
        sg = ts.spectrogram(4, fftlength=1)
        assert sg.time_reference == EXPECTED_STATE[kind]
        assert abs(_t0(sg) - _t0(ts)) <= _tol(_t0(ts))
        if kind == "relative":
            assert _t0(sg) != 0.0  # a cropped relative axis starts away from 0

    def test_documented_in_docstrings(self):
        doc = TimeSeries.spectrogram.__doc__ or ""
        assert "ormaliz" in doc
