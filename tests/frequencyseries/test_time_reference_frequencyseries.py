"""FrequencySeries timing-reference state (#725, v0.3.0 Step 7b, decision U-1).

Contract under test (``epoch`` is the only time origin of a FrequencySeries;
the frequency axis is ``f0``/``df`` and never depends on the state):

* ``epoch`` omitted, ``None`` or ``NaN`` -> ``time_reference == "relative"``
* finite value, including an explicit ``0.0`` -> ``"absolute"``
* ``+Inf`` / ``-Inf`` -> ``ValueError``
* assignment accepts finite values (state unchanged) and rejects ``NaN``/``±Inf``
* the public finite ``epoch`` values are the GWpy ones in both states
* the state survives ``OPERATIONS`` and the spectral results of a TimeSeries
* the state never changes a computed value, normalization or axis

Tolerances come from ``np.spacing`` of the quantity compared, or the
comparison is exact because both sides run the same computation (R-15).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
import scipy.signal as sps
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_NAME,
    TIME_REFERENCE_RELATIVE,
)
from gwexpy.timeseries import TimeSeries

REL = TIME_REFERENCE_RELATIVE
ABS = TIME_REFERENCE_ABSOLUTE
GPS = 1.4e9
N = 33
DF = 0.5
F0 = 1.0
UNIT = u.m

KINDS = ["relative", "absolute_zero", "absolute_gps"]
EXPECTED_STATE = {"relative": REL, "absolute_zero": ABS, "absolute_gps": ABS}
EXPECTED_EPOCH = {"relative": None, "absolute_zero": 0.0, "absolute_gps": GPS}


def _data() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal(N) + 1j * rng.standard_normal(N)


def _make(kind: str) -> FrequencySeries:
    kwargs: dict[str, float | None] = {}
    if kind == "relative":
        kwargs["epoch"] = None
    elif kind == "absolute_zero":
        kwargs["epoch"] = 0.0
    elif kind == "absolute_gps":
        kwargs["epoch"] = GPS
    return FrequencySeries(_data(), df=DF, f0=F0, unit=UNIT, name="x", **kwargs)


def _epoch_gps(obj) -> float | None:
    ep = obj.epoch
    return None if ep is None else float(ep.gps)


def _assert_epoch(obj, expected: float | None) -> None:
    """GWpy stores the epoch as ``Decimal`` and rebuilds a ``Time``: allow a few ULP."""
    actual = _epoch_gps(obj)
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    tol = 16 * np.spacing(max(1.0, abs(expected)))
    assert abs(actual - expected) <= tol, (actual, expected, tol)


# ------------------------------------------------------------ construction


class TestConstructor:
    @pytest.mark.parametrize(
        ("kw", "expected_state"),
        [({}, ABS), ({"epoch": None}, REL), ({"epoch": np.nan}, REL)],
    )
    def test_missing_or_nan_is_relative(self, kw, expected_state):
        fs = FrequencySeries(_data(), df=DF, f0=F0, unit=UNIT, **kw)
        assert fs.time_reference == expected_state
        assert fs.epoch is None  # public value unchanged (GWpy)
        assert fs.f0.value == F0
        assert fs.df.value == DF
        assert fs.unit == UNIT
        assert np.all(np.isfinite(fs.frequencies.value))

    @pytest.mark.parametrize(
        "epoch",
        [
            np.nan * u.s,
            np.float32("nan"),
            float("nan"),
        ],
    )
    def test_nan_variants_are_relative(self, epoch):
        fs = FrequencySeries(_data(), df=DF, f0=F0, epoch=epoch)
        assert fs.time_reference == REL
        assert fs.epoch is None

    @pytest.mark.parametrize(
        ("epoch", "expected"),
        [
            (0, 0.0),
            (0.0, 0.0),
            (-0.0, 0.0),
            (GPS, GPS),
            (GPS * u.s, GPS),
            (LIGOTimeGPS(1400000000), GPS),
        ],
    )
    def test_finite_is_absolute_including_zero(self, epoch, expected):
        fs = FrequencySeries(_data(), df=DF, f0=F0, unit=UNIT, epoch=epoch)
        assert fs.time_reference == ABS
        _assert_epoch(fs, expected)
        assert fs.unit == UNIT
        assert fs.f0.value == F0
        assert fs.df.value == DF

    @pytest.mark.parametrize("epoch", [np.inf, -np.inf, np.inf * u.s])
    def test_infinite_raises_valueerror(self, epoch):
        with pytest.raises(ValueError, match="must not be infinite"):
            FrequencySeries(_data(), df=DF, f0=F0, epoch=epoch)

    def test_frequency_axis_does_not_decide_the_state(self):
        # an explicit frequency axis with epoch=None is relative; omitted is absolute
        fs = FrequencySeries(_data(), frequencies=F0 + DF * np.arange(N), epoch=None)
        assert fs.time_reference == REL
        fs2 = FrequencySeries(_data(), frequencies=F0 + DF * np.arange(N))
        assert fs2.time_reference == ABS
        assert fs2.epoch is None

    def test_state_is_a_read_only_property(self):
        fs = _make("relative")
        assert TIME_REFERENCE_NAME == "time_reference"
        with pytest.raises(AttributeError):
            fs.time_reference = ABS  # type: ignore[misc]


class TestConstructorFromSeries:
    def test_relative_series_stays_relative(self):
        rel = _make("relative")
        assert FrequencySeries(rel).time_reference == REL

    def test_explicit_epoch_makes_absolute_and_leaves_source(self):
        rel = _make("relative")
        out = FrequencySeries(rel, epoch=GPS)
        assert out.time_reference == ABS
        assert _epoch_gps(out) == GPS
        assert rel.time_reference == REL
        assert rel.epoch is None

    def test_nan_epoch_makes_relative(self):
        ab = _make("absolute_gps")
        assert FrequencySeries(ab, epoch=np.nan).time_reference == REL
        assert ab.time_reference == ABS

    def test_absolute_series_stays_absolute(self):
        assert FrequencySeries(_make("absolute_zero")).time_reference == ABS

    def test_plain_gwpy_series_with_finite_epoch_is_absolute(self):
        g = GwpyFrequencySeries(_data(), df=DF, f0=F0, epoch=GPS)
        out = FrequencySeries(g)
        assert out.time_reference == ABS
        assert _epoch_gps(out) == GPS

    def test_plain_gwpy_series_without_epoch_defaults_to_absolute(self):
        g = GwpyFrequencySeries(_data(), df=DF, f0=F0)
        out = FrequencySeries(g)
        assert out.time_reference == ABS
        assert out.epoch is None

    def test_view_of_gwpy_series_defaults_to_absolute(self):
        # no information is available on a bare view: the documented default
        g = GwpyFrequencySeries(_data(), df=DF, f0=F0, epoch=GPS)
        assert g.view(FrequencySeries).time_reference == ABS


# ----------------------------------------------------------------- setters


class TestSetter:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, np.nan * u.s])
    def test_nonfinite_assignment_raises_and_changes_nothing(self, kind, bad):
        fs = _make(kind)
        before = EXPECTED_EPOCH[kind]
        with pytest.raises(ValueError, match="finite"):
            fs.epoch = bad
        assert _epoch_gps(fs) == before
        assert fs.time_reference == EXPECTED_STATE[kind]

    @pytest.mark.parametrize("kind", KINDS)
    def test_finite_assignment_keeps_state(self, kind):
        fs = _make(kind)
        fs.epoch = 123.0
        _assert_epoch(fs, 123.0)
        assert fs.time_reference == EXPECTED_STATE[kind]

    def test_none_assignment_keeps_state(self):
        fs = _make("absolute_gps")
        fs.epoch = None
        assert fs.epoch is None
        assert fs.time_reference == ABS

    def test_delete_epoch(self):
        fs = _make("absolute_gps")
        del fs.epoch
        assert fs.epoch is None


# ----------------------------------------------------------------- transforms

# name -> (function, expected unit as a function of the source unit | None,
#          True if the frequency axis must be identical to the source's)
OPERATIONS = {
    "slice": (lambda t: t[2:5], lambda un: un, False),
    "step slice": (lambda t: t[::2], lambda un: un, False),
    "crop": (lambda t: t.crop(2.0, 5.0), lambda un: un, False),
    "copy": (lambda t: t.copy(), lambda un: un, True),
    "copy.copy": (copy.copy, lambda un: un, True),
    "deepcopy": (copy.deepcopy, lambda un: un, True),
    "*2": (lambda t: t * 2, lambda un: un, True),
    "t+t": (lambda t: t + t, lambda un: un, True),
    "abs": (abs, lambda un: un, True),
    "np.abs": (np.abs, lambda un: un, True),
    "to mm": (lambda t: t.to("mm"), lambda un: u.mm, True),
    "astype": (lambda t: t.astype(np.complex64), lambda un: un, True),
    "conj": (lambda t: t.conj(), lambda un: un, True),
    "pad": (lambda t: t.pad(2), lambda un: un, False),
    "phase": (lambda t: t.phase(), lambda un: u.rad, True),
    "angle": (lambda t: t.angle(), lambda un: u.rad, True),
    "degree": (lambda t: t.degree(), lambda un: u.deg, True),
    "differentiate": (lambda t: t.differentiate(), lambda un: un * u.Hz, True),
    "integrate": (lambda t: t.integrate(), lambda un: un * u.s, True),
    "to_db": (lambda t: t.to_db(), lambda un: u.dB, True),
    "differentiate_time": (
        lambda t: t.differentiate_time(),
        lambda un: un * u.Hz,
        True,
    ),
    "integrate_time": (lambda t: t.integrate_time(), lambda un: un * u.s, True),
    "group_delay": (lambda t: t.group_delay(), lambda un: u.s, True),
    "quadrature_sum": (lambda t: t.quadrature_sum(t), lambda un: un, True),
    "rebin": (lambda t: t.rebin(1.0), lambda un: un, False),
    "smooth": (lambda t: t.smooth(3), lambda un: un, True),
    # GWpy's interpolate drops the unit (existing behavior, not part of #725)
    "interpolate": (
        lambda t: t.interpolate(0.25),
        lambda un: u.dimensionless_unscaled,
        False,
    ),
    "zpk": (lambda t: t.zpk([], [1.0], 1.0), lambda un: un, True),
    "FrequencySeries(fs)": (lambda t: FrequencySeries(t), lambda un: un, True),
}


class TestOperations:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("name", list(OPERATIONS))
    def test_state_unit_and_axis(self, kind, name):
        fn, unit_of, same_axis = OPERATIONS[name]
        src = _make(kind)
        out = fn(src)
        assert out.time_reference == EXPECTED_STATE[kind], name
        assert out.unit == unit_of(src.unit), name
        # the public finite epoch is the GWpy one (unchanged by the state)
        assert _epoch_gps(out) == EXPECTED_EPOCH[kind], name
        assert out.df.unit == u.Hz  # frequency axis stays in Hz
        if same_axis:
            np.testing.assert_array_equal(out.frequencies.value, src.frequencies.value)
        # the source is never modified
        assert src.time_reference == EXPECTED_STATE[kind]
        np.testing.assert_array_equal(src.value, _data())

    @pytest.mark.parametrize("kind", KINDS)
    def test_slice_moves_f0_not_the_state(self, kind):
        src = _make(kind)
        out = src[2:5]
        assert out.f0.value == F0 + 2 * DF
        assert out.df.value == DF
        assert out.time_reference == EXPECTED_STATE[kind]

    @pytest.mark.parametrize("kind", KINDS)
    def test_pickle_of_relative_restores_gwexpy_type(self, kind):
        src = _make(kind)
        back = pickle.loads(pickle.dumps(src))
        if kind == "relative":
            assert isinstance(back, FrequencySeries)
            assert back.time_reference == REL
        else:
            # absolute payloads keep their GWpy-only portability (unchanged)
            assert isinstance(back, GwpyFrequencySeries)
        np.testing.assert_array_equal(back.value, src.value)
        assert back.unit == src.unit
        assert back.f0 == src.f0
        assert back.df == src.df
        assert back.name == "x"

    def test_mixed_arithmetic_follows_first_operand(self):
        rel, ab = _make("relative"), _make("absolute_gps")
        assert (rel + ab).time_reference == REL
        assert (ab + rel).time_reference == ABS

    @pytest.mark.parametrize("kind", KINDS)
    def test_append_and_inplace_ops_keep_state(self, kind):
        src = _make(kind)
        tail = FrequencySeries(_data()[:3], df=DF, f0=F0 + N * DF, unit=UNIT)
        assert src.append(tail, inplace=False).time_reference == EXPECTED_STATE[kind]
        src *= 2
        assert src.time_reference == EXPECTED_STATE[kind]

    def test_state_independent_of_data_scale(self):
        # R-15: the state is a property of the timing, not of the values
        fs = FrequencySeries(_data() * 1e-21, df=DF, f0=F0, unit=UNIT, epoch=None)
        assert fs.time_reference == REL
        assert (fs * 1e-20).time_reference == REL


# --------------------------------------------------------------------- ifft


def _spectrum_of(ts_kind: str) -> FrequencySeries:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(256)
    kw: dict[str, float | None] = {"t0": None}
    if ts_kind == "absolute_gps":
        kw = {"t0": GPS}
    elif ts_kind == "absolute_zero":
        kw = {"t0": 0.0}
    return TimeSeries(x, sample_rate=64.0, unit=UNIT, **kw).fft()


class TestIfft:
    def test_relative_without_epoch_stays_relative_zero_origin(self):
        ts = _make("relative").ifft()
        assert ts.time_reference == REL
        assert float(ts.t0.to_value(u.s)) == 0.0
        assert ts.unit == UNIT

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("mode", ["gwpy", "transient"])
    def test_state_and_origin(self, kind, mode):
        src = _make(kind)
        if mode == "transient":
            src = _spectrum_of(kind)
            src_state = EXPECTED_STATE[kind]
        else:
            src_state = EXPECTED_STATE[kind]
        ts = src.ifft(mode=mode)
        assert ts.time_reference == src_state
        expected_t0 = 0.0 if src.epoch is None else float(src.epoch.gps)
        assert float(ts.t0.to_value(u.s)) == expected_t0
        assert ts.unit == UNIT
        assert src.time_reference == src_state  # source untouched

    @pytest.mark.parametrize("kind", KINDS)
    def test_values_match_plain_gwpy_ifft(self, kind):
        # AC-10-2: the state changes no value, normalization or axis
        src = _make(kind)
        plain = GwpyFrequencySeries(
            src.value, df=DF, f0=F0, unit=UNIT, epoch=src.epoch
        ).ifft()
        out = src.ifft(mode="gwpy")
        np.testing.assert_array_equal(out.value, plain.value)
        assert out.dt == plain.dt

    @pytest.mark.parametrize("kind", KINDS)
    def test_fft_then_ifft_round_trip_keeps_everything(self, kind):
        # transient mode halves the Nyquist bin of an even-length transform
        # (existing behavior, see the ifft docstring), so it uses odd lengths.
        rng = np.random.default_rng(3)
        kw: dict[str, float | None] = {"t0": None}
        if kind == "absolute_zero":
            kw = {"t0": 0.0}
        elif kind == "absolute_gps":
            kw = {"t0": GPS}
        for mode, n in (("gwpy", 512), ("transient", 511)):
            x = rng.standard_normal(n)
            ts = TimeSeries(x, sample_rate=128.0, unit=UNIT, **kw)
            back = ts.fft(mode=mode).ifft()
            assert back.time_reference == EXPECTED_STATE[kind]
            assert back.unit == UNIT
            assert back.size == ts.size
            assert float(back.t0.to_value(u.s)) == float(ts.t0.to_value(u.s))
            assert back.dt == ts.dt
            np.testing.assert_allclose(
                back.value, x, rtol=0, atol=64 * np.spacing(np.max(np.abs(x)))
            )

    def test_idct_keeps_state(self):
        for kind in KINDS:
            assert _make(kind).idct(n=N).time_reference == EXPECTED_STATE[kind]


# ----------------------------------------------- TimeSeries -> spectral results


def _ts(kind: str, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    kw: dict[str, float | None] = {"t0": None}
    if kind == "absolute_zero":
        kw = {"t0": 0.0}
    elif kind == "absolute_gps":
        kw = {"t0": GPS}
    return TimeSeries(rng.standard_normal(4096), sample_rate=256.0, unit=UNIT, **kw)


SPECTRAL = {
    "fft": lambda t, o: t.fft(),
    "fft transient": lambda t, o: t.fft(mode="transient"),
    "psd": lambda t, o: t.psd(fftlength=1),
    "asd": lambda t, o: t.asd(fftlength=1),
    "csd": lambda t, o: t.csd(o, fftlength=1),
    "coherence": lambda t, o: t.coherence(o, fftlength=1),
    "dct": lambda t, o: t.dct(),
    "cepstrum": lambda t, o: t.cepstrum(),
}


class TestSpectralResults:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("name", list(SPECTRAL))
    def test_result_carries_state_and_epoch(self, kind, name):
        ts, other = _ts(kind, 0), _ts(kind, 1)
        res = SPECTRAL[name](ts, other)
        assert isinstance(res, FrequencySeries)
        assert res.time_reference == EXPECTED_STATE[kind], name
        # public epoch is the source's projection, in both states
        assert float(res.epoch.gps) == float(ts.epoch.gps), name
        assert ts.time_reference == EXPECTED_STATE[kind]

    @pytest.mark.parametrize("name", ["psd", "asd", "csd", "coherence"])
    def test_welch_family_axis_and_unit(self, name):
        ts, other = _ts("relative", 0), _ts("relative", 1)
        res = SPECTRAL[name](ts, other)
        assert res.f0.value == 0.0
        assert res.df.value == 1.0  # fftlength = 1 s
        assert res.df.unit == u.Hz
        expected_unit = {
            "psd": UNIT**2 / u.Hz,
            "asd": UNIT / u.Hz**0.5,
            "csd": UNIT**2 / u.Hz,
            "coherence": u.dimensionless_unscaled,
        }[name]
        assert res.unit == expected_unit

    def test_csd_follows_first_operand(self):
        rel, ab = _ts("relative", 0), _ts("absolute_gps", 1)
        assert rel.csd(ab, fftlength=1).time_reference == REL
        assert ab.csd(rel, fftlength=1).time_reference == ABS

    def test_laplace_keeps_state(self):
        ts = _ts("relative", 0)[:256]
        fs = ts.laplace(frequencies=np.array([1.0, 2.0, 4.0]))
        assert fs.time_reference == REL

    @pytest.mark.parametrize("name", list(SPECTRAL))
    def test_values_do_not_depend_on_the_state(self, name):
        # AC-10-2: identical data, relative vs absolute -> identical numbers,
        # axes and units (the two computations are the same code path).
        rel = SPECTRAL[name](_ts("relative", 0), _ts("relative", 1))
        for kind in ("absolute_zero", "absolute_gps"):
            ab = SPECTRAL[name](_ts(kind, 0), _ts(kind, 1))
            np.testing.assert_array_equal(rel.value, ab.value)
            np.testing.assert_array_equal(rel.frequencies.value, ab.frequencies.value)
            assert rel.unit == ab.unit
            assert rel.df == ab.df
            assert rel.f0 == ab.f0


# ----------------------------------------------- normalization (AC-10-3)


class TestNormalizationConventions:
    """The conventions written in the docstrings, checked against references."""

    def test_fft_is_rfft_over_n_with_non_dc_bins_doubled(self):
        x = np.random.default_rng(0).standard_normal(1024)
        for kw in ({}, {"t0": GPS}):
            ts = TimeSeries(x, sample_rate=64.0, unit=UNIT, **kw)
            ref = np.fft.rfft(x) / x.size
            ref[1:] *= 2.0
            np.testing.assert_array_equal(ts.fft().value, ref)
            np.testing.assert_array_equal(
                ts.fft().frequencies.value, np.fft.rfftfreq(x.size, d=1 / 64.0)
            )

    def test_fft_matches_plain_gwpy(self):
        x = np.random.default_rng(1).standard_normal(1024)
        ours = TimeSeries(x, sample_rate=64.0, unit=UNIT).fft()
        theirs = GwpyTimeSeries(x, sample_rate=64.0, unit=UNIT).fft()
        np.testing.assert_array_equal(ours.value, theirs.value)
        np.testing.assert_array_equal(ours.frequencies.value, theirs.frequencies.value)
        assert ours.unit == theirs.unit

    def test_psd_is_one_sided_density_like_scipy_welch(self):
        x = np.random.default_rng(2).standard_normal(4096)
        ts = TimeSeries(x, sample_rate=256.0, unit=UNIT)
        psd = ts.psd(fftlength=1, overlap=0.5, method="welch")
        f, ref = sps.welch(
            x,
            fs=256.0,
            window="hann",
            nperseg=256,
            noverlap=128,
            scaling="density",
            detrend="constant",
        )
        np.testing.assert_array_equal(psd.frequencies.value, f)
        # same algorithm: agreement to a few ULP of the values
        np.testing.assert_allclose(psd.value, ref, rtol=64 * np.spacing(1.0), atol=0)
        assert psd.unit == UNIT**2 / u.Hz

    def test_psd_matches_plain_gwpy(self):
        x = np.random.default_rng(2).standard_normal(4096)
        ours = TimeSeries(x, sample_rate=256.0, unit=UNIT).psd(fftlength=1)
        theirs = GwpyTimeSeries(x, sample_rate=256.0, unit=UNIT).psd(fftlength=1)
        np.testing.assert_array_equal(ours.value, theirs.value)
        assert ours.unit == theirs.unit

    def test_documented_in_docstrings(self):
        for func in (
            TimeSeries.fft,
            TimeSeries.psd,
            TimeSeries.asd,
            TimeSeries.csd,
            TimeSeries.coherence,
            TimeSeries.spectrogram,
            FrequencySeries.ifft,
        ):
            doc = func.__doc__ or ""
            assert "ormaliz" in doc, func.__qualname__
