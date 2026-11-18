"""TimeSeries timing-reference state (#725, v0.3.0 Step 7a, decision U-1).

Contract under test:

* ``t0``/``epoch`` omitted, ``None`` or ``NaN``  -> ``time_reference == "relative"``
* finite value, including an explicit ``0.0``      -> ``"absolute"``
* ``+Inf`` / ``-Inf``                              -> ``ValueError``
* setters accept finite values (state unchanged) and reject ``NaN``/``±Inf``
* the public finite ``t0``/``epoch`` values are the GWpy ones in both states
* the state survives the transforms listed in ``OPERATIONS``

Tolerances are derived from ``np.spacing`` of the quantity compared; no fixed
absolute tolerance is introduced (R-15).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_NAME,
    TIME_REFERENCE_RELATIVE,
    check_time_setter_value,
    classify_time_input,
    get_time_reference,
    inherit_time_reference,
    set_time_reference,
)
from gwexpy.timeseries import TimeSeries

GPS = 1.4e9
N = 1024
FS = 256.0
DT = 1.0 / FS  # exactly representable in binary
UNIT = u.m

REL = TIME_REFERENCE_RELATIVE
ABS = TIME_REFERENCE_ABSOLUTE


def _data() -> np.ndarray:
    return np.sin(np.arange(N, dtype=float) * 0.37) + 0.1 * np.cos(
        np.arange(N, dtype=float) * 1.9
    )


def _tol(*values: float) -> float:
    """4 ULP of the largest magnitude involved (floor at 1.0 s)."""
    return 4 * np.spacing(max(1.0, *(abs(float(v)) for v in values)))


def _assert_close(actual: float, expected: float, *scale: float) -> None:
    tol = _tol(actual, expected, *scale)
    assert abs(float(actual) - float(expected)) <= tol, (actual, expected, tol)


def _make(kind: str) -> TimeSeries:
    if kind == "relative":
        return TimeSeries(_data(), dt=DT, unit=UNIT, name="x", t0=None)
    if kind == "absolute_zero":
        return TimeSeries(_data(), dt=DT, unit=UNIT, name="x", t0=0.0)
    if kind == "absolute_gps":
        return TimeSeries(_data(), dt=DT, unit=UNIT, name="x", t0=GPS)
    raise AssertionError(kind)


KINDS = ["relative", "absolute_zero", "absolute_gps"]
EXPECTED_STATE = {"relative": REL, "absolute_zero": ABS, "absolute_gps": ABS}


def _t0(ts: TimeSeries) -> float:
    return float(ts.t0.to_value(u.s))


# ---------------------------------------------------------------- helpers


class TestClassifyTimeInput:
    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            (None, "missing"),
            (np.nan, "nan"),
            (float("nan"), "nan"),
            (np.float32("nan"), "nan"),
            (np.nan * u.s, "nan"),
            (np.inf, "inf"),
            (-np.inf, "inf"),
            (np.inf * u.s, "inf"),
            (np.array(np.inf), "inf"),
            (0, "finite"),
            (0.0, "finite"),
            (-0.0, "finite"),
            (GPS, "finite"),
            (np.int64(5), "finite"),
            (GPS * u.s, "finite"),
            (LIGOTimeGPS(1400000000), "finite"),
            ("2020-01-01", "finite"),
            (True, "finite"),
        ],
    )
    def test_classification(self, value, kind):
        assert classify_time_input(value) == kind

    def test_huge_int_overflow_is_inf(self):
        assert classify_time_input(10**400) == "inf"

    def test_check_setter_value(self):
        check_time_setter_value(0.0, name="t0")
        check_time_setter_value(None, name="t0")
        for bad in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match="t0 must be a finite time"):
                check_time_setter_value(bad, name="t0")


class TestStateHelpers:
    def test_set_get_and_default(self):
        ts = _make("absolute_zero")
        assert get_time_reference(ts) == ABS
        set_time_reference(ts, REL)
        assert get_time_reference(ts) == REL
        assert ts.time_reference == REL
        set_time_reference(ts, ABS)
        assert get_time_reference(ts) == ABS

    def test_invalid_state_rejected(self):
        with pytest.raises(ValueError, match=TIME_REFERENCE_NAME):
            set_time_reference(_make("relative"), "unknown")

    def test_default_for_object_without_state_is_absolute(self):
        # Documented default when a code path loses the state: the finite
        # public numbers are read as absolute (U-1).
        assert get_time_reference(object()) == ABS
        assert get_time_reference(np.zeros(3)) == ABS

    def test_inherit_only_onto_time_reference_objects(self):
        src = _make("relative")
        plain = np.zeros(3)
        assert inherit_time_reference(plain, src) is plain
        dst = _make("absolute_gps")
        assert inherit_time_reference(dst, src).time_reference == REL

    def test_property_is_read_only(self):
        ts = _make("relative")
        with pytest.raises(AttributeError):
            setattr(ts, TIME_REFERENCE_NAME, ABS)

    def test_public_name_constant(self):
        assert getattr(_make("relative"), TIME_REFERENCE_NAME) == REL


# ------------------------------------------------------------ constructor


@pytest.mark.parametrize("kind", ["t0", "epoch"])
class TestConstructor:
    @pytest.mark.parametrize("missing", ["omitted", None, np.nan, np.nan * u.s])
    def test_missing_or_nan_is_relative_with_gwpy_public_values(self, kind, missing):
        kwargs = {} if isinstance(missing, str) else {kind: missing}
        ts = TimeSeries(_data(), dt=DT, unit=UNIT, **kwargs)
        assert ts.time_reference == (ABS if missing == "omitted" else REL)
        # public finite values are exactly the GWpy ones
        assert ts.t0.value == 0.0
        assert ts.t0.unit == u.s
        assert float(ts.epoch.gps) == 0.0
        times = ts.times.value
        assert np.all(np.isfinite(times))
        np.testing.assert_allclose(
            times, np.arange(N) * DT, rtol=0, atol=_tol(times[-1])
        )
        assert ts.dt.value == DT
        assert ts.unit == UNIT

    @pytest.mark.parametrize("zero", [0, 0.0, -0.0, np.float64(0.0), 0 * u.s])
    def test_explicit_zero_is_absolute_gps_zero(self, kind, zero):
        ts = TimeSeries(_data(), dt=DT, unit=UNIT, **{kind: zero})
        assert ts.time_reference == ABS
        assert ts.t0.value == 0.0
        assert float(ts.epoch.gps) == 0.0

    def test_explicit_none_and_zero_have_identical_numbers_but_different_state(
        self, kind
    ):
        none_ts = TimeSeries(_data(), dt=DT, unit=UNIT, **{kind: None})
        zero = TimeSeries(_data(), dt=DT, unit=UNIT, **{kind: 0.0})
        assert none_ts.t0 == zero.t0
        np.testing.assert_array_equal(none_ts.times.value, zero.times.value)
        np.testing.assert_array_equal(none_ts.value, zero.value)
        assert (none_ts.time_reference, zero.time_reference) == (REL, ABS)

    @pytest.mark.parametrize("gps", [GPS, 1, 1e-3, -5.0, GPS * u.s, 1400000.0 * u.ms])
    def test_finite_is_absolute_and_unchanged(self, kind, gps):
        ts = TimeSeries(_data(), dt=DT, unit=UNIT, **{kind: gps})
        expected = float(u.Quantity(gps, u.s).to_value(u.s))
        assert ts.time_reference == ABS
        _assert_close(_t0(ts), expected)
        np.testing.assert_allclose(
            ts.times.value - expected,
            np.arange(N) * DT,
            rtol=0,
            atol=_tol(expected, ts.times.value[-1]),
        )
        assert ts.dt.value == DT
        assert ts.unit == UNIT

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.inf * u.s])
    def test_infinite_raises_clear_value_error(self, kind, bad):
        with pytest.raises(ValueError, match=rf"{kind} must not be infinite"):
            TimeSeries(_data(), dt=DT, unit=UNIT, **{kind: bad})


class TestConstructorOtherForms:
    def test_x0_keyword_follows_the_same_rule(self):
        assert TimeSeries(_data(), dt=DT, x0=np.nan).time_reference == REL
        assert TimeSeries(_data(), dt=DT, x0=0.0).time_reference == ABS
        with pytest.raises(ValueError, match="x0 must not be infinite"):
            TimeSeries(_data(), dt=DT, x0=np.inf)

    def test_positional_t0(self):
        nan = TimeSeries(_data(), "m", np.nan, DT)
        zero = TimeSeries(_data(), "m", 0.0, DT)
        assert (nan.time_reference, zero.time_reference) == (REL, ABS)
        assert nan.t0.value == 0.0
        assert nan.dt.value == DT
        assert nan.unit == u.m
        with pytest.raises(ValueError, match="t0 must not be infinite"):
            TimeSeries(_data(), "m", -np.inf, DT)

    def test_explicit_axis_without_origin_is_absolute_and_explicit_none_is_relative(
        self,
    ):
        times = np.arange(N) * DT
        assert TimeSeries(_data(), times=times).time_reference == ABS
        assert TimeSeries(_data(), times=times, t0=None).time_reference == REL
        assert TimeSeries(_data(), times=times, t0=np.nan).time_reference == REL
        assert TimeSeries(_data(), times=times, t0=0.0).time_reference == ABS
        assert TimeSeries(_data(), dt=DT, t0_ns=5_000_000_000).time_reference == ABS
        assert (
            TimeSeries(_data(), dt=DT, t0=LIGOTimeGPS(1400000000)).time_reference == ABS
        )

    def test_t0_and_epoch_together_still_fail_in_gwpy(self):
        with pytest.raises(ValueError, match="give only one of epoch or t0"):
            TimeSeries(_data(), dt=DT, t0=1.0, epoch=2.0)
        # even if one of them is NaN: GWpy keeps owning the conflict error
        with pytest.raises(ValueError, match="give only one of epoch or t0"):
            TimeSeries(_data(), dt=DT, t0=np.nan, epoch=2.0)

    def test_cast_keeps_or_sets_state(self):
        rel, zero, gps = (_make(k) for k in KINDS)
        assert TimeSeries(rel).time_reference == REL
        assert TimeSeries(zero).time_reference == ABS
        assert TimeSeries(gps).time_reference == ABS
        anchored = TimeSeries(rel, t0=GPS)
        assert anchored.time_reference == ABS
        _assert_close(_t0(anchored), GPS)
        assert rel.time_reference == REL  # source is not modified
        # NaN on a source that has an axis is an explicit "unknown"
        assert TimeSeries(gps, t0=np.nan).time_reference == REL

    def test_gwpy_series_source_has_an_axis_so_it_is_absolute(self):
        g = GwpyTimeSeries(_data(), dt=DT, t0=3.0, unit=UNIT)
        out = TimeSeries(g)
        assert out.time_reference == ABS
        _assert_close(_t0(out), 3.0)


# ----------------------------------------------------------------- setters


class TestSetters:
    @pytest.mark.parametrize("attr", ["t0", "x0", "epoch"])
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("kind", KINDS)
    def test_non_finite_assignment_raises_and_changes_nothing(self, attr, bad, kind):
        ts = _make(kind)
        before_t0, before_state = _t0(ts), ts.time_reference
        with pytest.raises(ValueError, match=f"{attr} must be a finite time"):
            setattr(ts, attr, bad)
        assert _t0(ts) == before_t0
        assert ts.time_reference == before_state
        assert np.all(np.isfinite(ts.times.value))

    @pytest.mark.parametrize("attr", ["t0", "x0", "epoch"])
    @pytest.mark.parametrize("kind", KINDS)
    def test_finite_assignment_keeps_gwpy_semantics_and_state(self, attr, kind):
        ts = _make(kind)
        state = ts.time_reference
        setattr(ts, attr, 12.5)
        _assert_close(_t0(ts), 12.5)
        assert ts.time_reference == state
        _assert_close(float(ts.times.value[0]), 12.5)
        assert ts.dt.value == DT
        assert ts.unit == UNIT


# ------------------------------------------------- state across transforms

# (id, function, expected unit, t0 offset in samples, expected dt, expected length)
OPERATIONS = [
    ("slice", lambda t: t[100:200], UNIT, 100, DT, 100),
    ("slice_step", lambda t: t[::2], UNIT, 0, 2 * DT, N // 2),
    ("copy", lambda t: t.copy(), UNIT, 0, DT, N),
    ("copy_copy", copy.copy, UNIT, 0, DT, N),
    ("deepcopy", copy.deepcopy, UNIT, 0, DT, N),
    ("crop", lambda t: t.crop(_t0(t) + 0.5, _t0(t) + 2.0), UNIT, 128, DT, 384),
    ("resample", lambda t: t.resample(FS / 2), UNIT, 0, 2 * DT, N // 2),
    ("bandpass", lambda t: t.bandpass(10, 50), UNIT, 0, DT, N),
    ("lowpass", lambda t: t.lowpass(50), UNIT, 0, DT, N),
    ("notch", lambda t: t.notch(60), UNIT, 0, DT, N),
    ("detrend", lambda t: t.detrend("linear"), UNIT, 0, DT, N),
    ("taper", lambda t: t.taper(), UNIT, 0, DT, N),
    ("pad", lambda t: t.pad(10), UNIT, -10, DT, N + 20),
    ("scale", lambda t: t * 2, UNIT, 0, DT, N),
    ("square", lambda t: t * t, UNIT**2, 0, DT, N),
    ("to_mm", lambda t: t.to("mm"), u.mm, 0, DT, N),
    ("abs", abs, UNIT, 0, DT, N),
    ("diff", lambda t: t.diff(), UNIT, 1, DT, N - 1),
    (
        "append",
        lambda t: t.append(
            TimeSeries(_data()[:64], dt=DT, unit=UNIT, t0=_t0(t) + N * DT),
            inplace=False,
        ),
        UNIT,
        0,
        DT,
        N + 64,
    ),
    ("hilbert", lambda t: t.hilbert(), UNIT, 0, DT, N),
    ("envelope", lambda t: t.envelope(), UNIT, 0, DT, N),
    ("instantaneous_phase", lambda t: t.instantaneous_phase(), u.rad, 0, DT, N),
    ("instantaneous_frequency", lambda t: t.instantaneous_frequency(), u.Hz, 0, DT, N),
    ("rolling_mean", lambda t: t.rolling_mean(8), UNIT, 0, DT, N),
    ("smooth", lambda t: t.smooth(5), UNIT, 0, DT, N),
    # ``asfreq`` takes the new sampling *period* (a bare number is seconds)
    ("asfreq", lambda t: t.asfreq(2 * DT), UNIT, 0, 2 * DT, N // 2),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("fn", "unit", "offset", "dt", "length"),
    [op[1:] for op in OPERATIONS],
    ids=[op[0] for op in OPERATIONS],
)
def test_operation_preserves_state_and_metadata(kind, fn, unit, offset, dt, length):
    src = _make(kind)
    out = fn(src)
    # state: a relative series stays relative, an absolute one stays absolute
    assert out.time_reference == EXPECTED_STATE[kind]
    # metadata (AC-10-2): unit from the documented rule, dt, t0 and length
    assert u.Unit(out.unit) == u.Unit(unit)
    assert u.Quantity(out.dt).to_value(u.s) == pytest.approx(dt, rel=_tol(dt))
    _assert_close(_t0(out), _t0(src) + offset * DT, _t0(src))
    assert len(out) == length
    assert np.all(np.isfinite(out.times.value))
    # the source is untouched
    assert src.time_reference == EXPECTED_STATE[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_tail_preserves_state_and_cadence(kind):
    # t0 is not asserted: ``tail`` is ``self[-n:]`` and the t0 of a
    # negative-start slice is not the start of the retained samples in the
    # GWpy slicing code this inherits (pre-existing, unrelated to #725).
    out = _make(kind).tail(10)
    assert out.time_reference == EXPECTED_STATE[kind]
    assert len(out) == 10
    assert out.dt.value == DT
    assert out.unit == UNIT


@pytest.mark.parametrize("kind", KINDS)
def test_whiten_preserves_state(kind):
    out = _make(kind).whiten(fftlength=1, overlap=0.5)
    assert out.time_reference == EXPECTED_STATE[kind]
    assert out.dt.value == DT


def test_relative_stays_distinct_from_explicit_zero_after_each_transform():
    rel, zero = _make("relative"), _make("absolute_zero")
    for fn in (
        lambda t: t.resample(FS / 2),
        lambda t: t.bandpass(10, 50),
        lambda t: t[10:20],
        lambda t: t.crop(0.5, 2.0),
        lambda t: pickle.loads(pickle.dumps(t)),
        lambda t: t.hilbert(),
    ):
        a, b = fn(rel), fn(zero)
        assert a.t0 == b.t0  # identical public numbers ...
        assert getattr(a, "time_reference", ABS) == REL  # ... distinct state
        assert getattr(b, "time_reference", ABS) == ABS


def test_hilbert_analysis_and_emd_dict_results_keep_state():
    rel = _make("relative")
    analysis = rel.hilbert_analysis()
    leaves = analysis.values() if isinstance(analysis, dict) else analysis
    for leaf in leaves:
        if hasattr(leaf, "time_reference"):
            assert leaf.time_reference == REL


def test_mixed_operands_follow_the_first_operand_current():
    # characterizes current behavior: numpy finalizes the result from the
    # first operand, as it already does for units and the epoch.
    rel, zero = _make("relative"), _make("absolute_zero")
    assert (rel + zero).time_reference == REL
    assert (zero + rel).time_reference == ABS


# ------------------------------------------------------------------ pickle


class TestPickle:
    @pytest.mark.parametrize("protocol", [2, 4, pickle.HIGHEST_PROTOCOL])
    def test_relative_round_trip_restores_state_and_metadata(self, protocol):
        src = _make("relative")
        back = pickle.loads(pickle.dumps(src, protocol=protocol))
        assert isinstance(back, TimeSeries)
        assert back.time_reference == REL
        assert back.t0 == src.t0
        assert back.dt == src.dt
        assert back.unit == UNIT
        assert back.name == "x"
        np.testing.assert_array_equal(back.value, src.value)
        np.testing.assert_array_equal(back.times.value, src.times.value)

    def test_relative_after_crop_keeps_nonzero_origin(self):
        src = _make("relative").crop(0.5, 2.0)
        back = pickle.loads(pickle.dumps(src))
        assert back.time_reference == REL
        _assert_close(_t0(back), 0.5)

    @pytest.mark.parametrize("kind", ["absolute_zero", "absolute_gps"])
    def test_absolute_payload_is_unchanged_gwpy_portable(self, kind):
        # characterizes current behavior (not changed by #725): absolute
        # payloads unpickle as plain GWpy series so GWpy-only readers work.
        src = _make(kind)
        back = pickle.loads(pickle.dumps(src))
        assert type(back) is GwpyTimeSeries
        assert getattr(back, TIME_REFERENCE_NAME, ABS) == ABS
        assert back.t0 == src.t0
        assert back.dt == src.dt
        assert back.unit == UNIT

    def test_exact_epoch_round_trip_is_absolute(self):
        src = TimeSeries(_data(), dt=DT, unit=UNIT, t0_ns=1_400_000_000_123_456_789)
        back = pickle.loads(pickle.dumps(src))
        assert back.time_reference == ABS
        assert back.t0_gps_ns == src.t0_gps_ns


# ---------------------------------------------------- edge cases, invariance


class TestEdgeCases:
    @pytest.mark.parametrize(
        "data", [[], [1.0], [np.nan, 1.0]], ids=["empty", "one", "nan"]
    )
    def test_short_and_nan_data_are_relative(self, data):
        ts = TimeSeries(data, dt=DT, unit=UNIT, t0=None)
        assert ts.time_reference == REL
        assert len(ts) == len(data)
        assert ts.t0.value == 0.0
        assert ts.dt.value == DT

    @pytest.mark.parametrize("scale", [1.0, 1e-20, 1e20])
    def test_state_is_independent_of_data_scale(self, scale):
        # R-15: f(X) and f(X * 1e-20) agree; the state never depends on values.
        rel = TimeSeries(_data() * scale, dt=DT, unit=UNIT, t0=None)
        zero = TimeSeries(_data() * scale, dt=DT, unit=UNIT, t0=0.0)
        assert (rel.time_reference, zero.time_reference) == (REL, ABS)
        assert rel.resample(FS / 2).time_reference == REL
        assert zero.resample(FS / 2).time_reference == ABS

    def test_non_second_axis_unit_keeps_state(self):
        ts = TimeSeries(_data(), dt=1.0 * u.ms, unit=UNIT, t0=None)
        assert ts.time_reference == REL
        assert ts.dt.unit == u.ms
        assert ts[10:20].time_reference == REL
        assert TimeSeries(_data(), dt=1.0 * u.ms, t0=0.0).time_reference == ABS

    def test_item_access_returns_plain_quantity(self):
        item = _make("relative")[3]
        assert isinstance(item, u.Quantity)
        assert item.unit == UNIT


# ------------------------------------------- known limitations (7b / 7c)


class TestKnownLimitationsCurrent:
    """Pin behavior outside Step 7a so a later change shows up as a diff."""

    def test_hdf5_round_trip_preserves_the_relative_state(self, tmp_path):
        pytest.importorskip("h5py")
        src = _make("relative")
        path = tmp_path / "rel.h5"
        src.write(path, format="hdf5", path="x")
        back = TimeSeries.read(path, format="hdf5", path="x")
        assert back.t0 == src.t0
        assert back.dt == src.dt
        assert back.unit == UNIT
        assert back.time_reference == REL

    def test_fft_result_epoch_is_gps_zero_even_if_relative_current(self):
        # changed by U-1 (#725): Step 7b gives the FrequencySeries its own
        # timing-reference state.  The public finite epoch is still the GWpy
        # projection (Time(0, gps)); only the state now says it is not GPS 0.
        fs = _make("relative").fft()
        assert float(fs.epoch.gps) == 0.0
        assert hasattr(fs, TIME_REFERENCE_NAME)
        assert fs.time_reference == REL
