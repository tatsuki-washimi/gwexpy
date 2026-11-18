"""TimeSeriesMatrix timing-reference state (#725, v0.3.0 Step 7c-1, decision U-1).

Contract under test (``t0``/``epoch``/``x0`` name the time origin; ``times=``
and ``xindex=`` give an explicit axis):

* ``t0``/``epoch`` omitted, ``None`` or ``NaN`` -> ``time_reference == "relative"``
  and the finite public ``t0 = 0 s``, ``epoch = 0.0`` are unchanged
* a finite value (including an explicit ``0``/``0.0``) or an explicit axis ->
  ``"absolute"``
* ``+Inf`` / ``-Inf`` -> ``ValueError``
* the state agrees with ``TimeSeries`` for every input
* the state survives matrix operations, element extraction, the conversions
  to and from ``TimeSeries`` collections and the spectral methods, and never
  changes a computed value, unit or axis

Tolerances come from ``np.spacing`` of the quantity compared, or the comparison
is exact because both sides run the same computation (R-15).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.spectrogram import SpectrogramMatrix
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
)
from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)

REL = TIME_REFERENCE_RELATIVE
ABS = TIME_REFERENCE_ABSOLUTE
GPS = 1.4e9
FS = 64.0
DT = 1.0 / FS
T = 512
UNIT = u.m

KINDS = ["relative", "absolute_zero", "absolute_gps"]
EXPECTED_STATE = {"relative": REL, "absolute_zero": ABS, "absolute_gps": ABS}
EXPECTED_T0 = {"relative": 0.0, "absolute_zero": 0.0, "absolute_gps": GPS}


def _values() -> np.ndarray:
    rng = np.random.default_rng(11)
    t = np.arange(T) * DT
    out = rng.standard_normal((2, 2, T))
    out[:, 0] += np.sin(2 * np.pi * 5.0 * t)
    return out


def _tkw(kind: str) -> dict[str, float | None]:
    if kind == "relative":
        return {"t0": None}
    return {"t0": EXPECTED_T0[kind]}


def _make(kind: str) -> TimeSeriesMatrix:
    return TimeSeriesMatrix(_values(), dt=DT, unit=UNIT, **_tkw(kind))


def _make_series(kind: str) -> TimeSeries:
    return TimeSeries(_values()[0, 0], dt=DT, unit=UNIT, name="x", **_tkw(kind))


def _tol(*values) -> float:
    """Four spacings of the largest magnitude involved (R-15)."""
    scale = max(1.0, *(float(np.max(np.abs(v))) for v in values))
    return 4 * np.spacing(scale)


def _unit(m) -> u.UnitBase:
    return m.meta[0, 0].unit


# ------------------------------------------------------------ construction


class TestConstructor:
    @pytest.mark.parametrize("name", ["t0", "epoch"])
    @pytest.mark.parametrize("value", ["omitted", None, np.nan, np.nan * u.s])
    def test_missing_or_nan_is_relative(self, name, value):
        kw = {} if isinstance(value, str) else {name: value}
        m = TimeSeriesMatrix(_values(), dt=DT, unit=UNIT, **kw)
        assert m.time_reference == (ABS if value == "omitted" else REL)
        assert m.t0.value == 0.0  # public values unchanged
        assert m.epoch == 0.0
        assert np.all(np.isfinite(m.times.value))
        assert m.dt.value == DT
        assert _unit(m) == UNIT

    @pytest.mark.parametrize("name", ["t0", "epoch"])
    @pytest.mark.parametrize("value", [0, 0.0, -0.0, GPS, GPS * u.s])
    def test_finite_is_absolute_including_zero(self, name, value):
        m = TimeSeriesMatrix(_values(), dt=DT, unit=UNIT, **{name: value})
        assert m.time_reference == ABS
        assert m.t0.value == float(getattr(value, "value", value))
        assert m.dt.value == DT

    @pytest.mark.parametrize("name", ["t0", "epoch", "x0"])
    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_raises_clear_valueerror(self, name, value):
        kw = {name: value}
        with pytest.raises(ValueError, match="must not be infinite"):
            TimeSeriesMatrix(_values(), dt=DT, **kw)

    def test_explicit_times_axis_without_epoch_is_absolute(self):
        m = TimeSeriesMatrix(_values(), times=np.arange(T) * DT * u.s)
        assert m.time_reference == ABS
        assert m.t0.value == 0.0
        m2 = TimeSeriesMatrix(_values(), xindex=np.arange(T) * DT * u.s)
        assert m2.time_reference == ABS
        m3 = TimeSeriesMatrix(_values(), times=np.arange(T) * DT * u.s, epoch=0.0)
        assert m3.time_reference == ABS

    def test_explicit_axis_with_nan_or_none_origin_stays_relative(self):
        m = TimeSeriesMatrix(
            _values(), times=np.arange(T) * DT * u.s, epoch=float("nan")
        )
        assert m.time_reference == REL
        assert np.all(np.isfinite(m.times.value))
        m_none = TimeSeriesMatrix(_values(), times=np.arange(T) * DT * u.s, epoch=None)
        assert m_none.time_reference == REL

    def test_t0_and_epoch_together_keep_the_old_error(self):
        with pytest.raises(ValueError, match="give only one of epoch or t0"):
            TimeSeriesMatrix(_values(), dt=DT, t0=GPS, epoch=GPS)
        with pytest.raises(ValueError, match="give only one of epoch or t0"):
            TimeSeriesMatrix(_values(), dt=DT, t0=float("nan"), epoch=GPS)

    @pytest.mark.parametrize("kind", KINDS)
    def test_rebuild_from_a_matrix_keeps_state(self, kind):
        src = _make(kind)
        rebuilt = TimeSeriesMatrix(src)
        assert rebuilt.time_reference == EXPECTED_STATE[kind]
        assert rebuilt.epoch == src.epoch
        assert _unit(rebuilt) == UNIT

    def test_rebuild_with_explicit_origin_overrides_the_state(self):
        rel = _make("relative")
        assert TimeSeriesMatrix(rel, t0=GPS).time_reference == ABS
        assert rel.time_reference == REL
        assert TimeSeriesMatrix(_make("absolute_gps"), t0=None).time_reference == REL

    def test_scale_invariance(self):
        tiny = TimeSeriesMatrix(_values() * 1e-20, dt=DT, t0=None)
        assert tiny.time_reference == REL
        assert TimeSeriesMatrix(_values() * 1e-20, dt=DT, t0=0).time_reference == ABS


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, ABS),
        ({"t0": None}, REL),
        ({"t0": float("nan")}, REL),
        ({"epoch": float("nan")}, REL),
        ({"t0": 0}, ABS),
        ({"t0": 0.0}, ABS),
        ({"epoch": 0.0}, ABS),
        ({"t0": GPS}, ABS),
        ({"epoch": GPS}, ABS),
    ],
)
def test_state_agrees_with_timeseries_for_every_input(kw, expected):
    """#725 acceptance 5: TimeSeries and TimeSeriesMatrix agree."""
    ts = TimeSeries(_values()[0, 0], dt=DT, **kw)
    m = TimeSeriesMatrix(_values(), dt=DT, **kw)
    assert ts.time_reference == m.time_reference == expected
    assert ts.t0.value == m.t0.value
    assert ts.dt.value == m.dt.value


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", KINDS)
def test_setter_rejects_nonfinite_and_finite_keeps_the_state(kind, bad):
    m = _make(kind)
    with pytest.raises(ValueError, match="epoch must be a finite time"):
        m.epoch = bad
    m.epoch = 5.0
    assert m.epoch == 5.0
    assert m.time_reference == EXPECTED_STATE[kind]


# ------------------------------------------------------ matrix operations

OPERATIONS = {
    "view": (lambda m: m.view(type(m)), UNIT),
    "row_slice": (lambda m: m[:1], UNIT),
    "paired_list_selection": (lambda m: m[[1, 0], [0, 1]], UNIT),
    "submatrix": (lambda m: m.submatrix(["row0"], ["col1"]), UNIT),
    "copy": (lambda m: m.copy(), UNIT),
    "deepcopy": (lambda m: copy.deepcopy(m), UNIT),
    "pickle": (lambda m: pickle.loads(pickle.dumps(m)), UNIT),
    "astype": (lambda m: m.astype(np.float32), UNIT),
    "transpose": (lambda m: m.T, UNIT),
    "add": (lambda m: m + m, UNIT),
    "scale_left": (lambda m: 2 * m, UNIT),
    "square": (lambda m: m**2, UNIT**2),
    "detrend": (lambda m: m.detrend(), UNIT),
    "hilbert": (lambda m: m.hilbert(), UNIT),
    "rolling_mean": (lambda m: m.rolling_mean(5), UNIT),
    "resample_rate": (lambda m: m.resample(FS / 2), UNIT),
    "crop": (lambda m: m.crop(m.xindex[10], m.xindex[200]), UNIT),
    "TimeSeriesMatrix(m)": (lambda m: TimeSeriesMatrix(m), UNIT),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(OPERATIONS))
def test_operation_keeps_state_epoch_and_unit(name, kind):
    op, unit = OPERATIONS[name]
    src = _make(kind)
    result = op(src)
    assert isinstance(result, TimeSeriesMatrix)
    assert result.time_reference == EXPECTED_STATE[kind]
    assert result.epoch == src.epoch
    assert _unit(result) == unit
    # the time coordinate keeps its origin kind: a relative axis starts at 0 s,
    # an absolute one at its t0 (cropping/slicing move it by whole samples)
    assert result.dt.value == pytest.approx(
        src.dt.value * (2 if name == "resample_rate" else 1)
    )
    first = float(result.times.value[0])
    assert first >= EXPECTED_T0[kind] - _tol(first)
    assert np.all(np.isfinite(result.times.value))


@pytest.mark.parametrize("kind", KINDS)
def test_values_are_independent_of_the_state(kind):
    rel, other = _make("relative"), _make(kind)
    for name, (op, _unit_) in OPERATIONS.items():
        if name == "pickle":
            continue
        np.testing.assert_array_equal(
            np.asarray(op(rel).value), np.asarray(op(other).value), err_msg=name
        )


def test_mixed_arithmetic_follows_the_first_operand():
    rel, absolute = _make("relative"), _make("absolute_gps")
    # the axes differ, so build an absolute matrix on the same axis
    same_axis = TimeSeriesMatrix(_values(), dt=DT, t0=0.0, unit=UNIT)
    assert (rel + same_axis).time_reference == REL
    assert (same_axis + rel).time_reference == ABS
    assert absolute.time_reference == ABS


def test_sample_slice_moves_the_axis_but_not_the_epoch_current():
    """Characterizes Step 6 observation 8 (not decided by U-1, not fixed)."""
    m = _make("absolute_gps")
    sl = m[:, :, 10:20]
    assert sl.t0.value == pytest.approx(GPS + 10 * DT, abs=_tol(GPS))
    assert sl.epoch == m.epoch
    assert sl.time_reference == ABS
    rel_slice = _make("relative")[:, :, 10:20]
    assert rel_slice.t0.value == pytest.approx(10 * DT, abs=_tol(10 * DT))
    assert rel_slice.time_reference == REL


def test_resample_keeps_the_source_epoch():
    """Before U-1 an absolute matrix came back with epoch 0.0."""
    m = _make("absolute_gps")
    assert m.resample(FS / 2).epoch == GPS
    assert m.detrend().epoch == GPS


# ------------------------------------------------------ element conversion


@pytest.mark.parametrize("kind", KINDS)
def test_element_extraction_keeps_state_axis_and_unit(kind):
    m = _make(kind)
    ts = m[0, 1]
    assert isinstance(ts, TimeSeries)
    assert ts.time_reference == EXPECTED_STATE[kind]
    assert ts.t0.value == EXPECTED_T0[kind]
    assert ts.dt.value == DT
    assert ts.unit == UNIT
    np.testing.assert_array_equal(ts.value, _values()[0, 1])
    for series in (
        *m.to_series_1Dlist(),
        *m.to_list(),
        *m.to_dict().values(),
        *(s for row in m.to_series_2Dlist() for s in row),
    ):
        assert series.time_reference == EXPECTED_STATE[kind]
        assert series.t0.value == EXPECTED_T0[kind]
        assert series.unit == UNIT


@pytest.mark.parametrize("kind", KINDS)
def test_diagonal_trace_and_det_series_keep_state(kind):
    m = _make(kind)
    for series in m.diagonal():
        assert series.time_reference == EXPECTED_STATE[kind]
        assert series.t0.value == EXPECTED_T0[kind]
    assert m.trace().time_reference == EXPECTED_STATE[kind]
    assert m.det().time_reference == EXPECTED_STATE[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_list_and_dict_to_matrix_keep_state(kind):
    series = [
        TimeSeries(_values()[0, j], dt=DT, unit=UNIT, name=f"c{j}", **_tkw(kind))
        for j in range(2)
    ]
    for matrix in (
        TimeSeriesList(series).to_matrix(),
        TimeSeriesDict({s.name: s for s in series}).to_matrix(),
    ):
        assert matrix.time_reference == EXPECTED_STATE[kind]
        assert matrix.t0.value == EXPECTED_T0[kind]
        assert matrix.dt.value == DT
        np.testing.assert_array_equal(np.asarray(matrix.value)[:, 0], _values()[0, :])
    # Step 6: the Matrix epoch of the list conversion is a Quantity (current)
    assert isinstance(TimeSeriesList(series).to_matrix().epoch, u.Quantity)


@pytest.mark.parametrize("kind", KINDS)
def test_matrix_to_list_to_matrix_round_trip(kind):
    m = _make(kind)
    back = m.to_list().to_matrix()
    assert back.time_reference == EXPECTED_STATE[kind]
    assert back.t0.value == EXPECTED_T0[kind]
    assert back.dt.value == DT
    assert back.shape[-1] == T
    np.testing.assert_array_equal(
        np.asarray(back.value).reshape(4, T), np.asarray(m.value).reshape(4, T)
    )


# --------------------------------------------------------- spectral methods


@pytest.mark.parametrize("kind", KINDS)
def test_fft_psd_asd_keep_state_unit_and_frequency_axis(kind):
    m = _make(kind)
    for name in ("fft", "psd", "asd"):
        kwargs = {} if name == "fft" else {"fftlength": 2}
        result = getattr(m, name)(**kwargs)
        assert isinstance(result, FrequencySeriesMatrix), name
        assert result.time_reference == EXPECTED_STATE[kind], name
        assert result.f0.value == 0.0, name
        assert np.all(np.isfinite(result.frequencies.value)), name
    fft = m.fft()
    assert _unit(fft) == UNIT
    assert fft.df.value == pytest.approx(FS / T)
    psd = m.psd(fftlength=2)
    assert _unit(psd) == UNIT**2 / u.Hz
    assert psd.df.value == pytest.approx(0.5)


@pytest.mark.parametrize("kind", KINDS)
def test_csd_and_coherence_keep_state_and_values(kind):
    m = _make(kind)
    rel = _make("relative")
    for name in ("csd", "coherence"):
        result = getattr(m, name)(m, fftlength=2)
        reference = getattr(rel, name)(rel, fftlength=2)
        assert result.time_reference == EXPECTED_STATE[kind], name
        np.testing.assert_array_equal(
            np.asarray(result.value), np.asarray(reference.value), err_msg=name
        )
    assert _unit(m.csd(m, fftlength=2)) == UNIT**2 / u.Hz
    assert _unit(m.coherence(m, fftlength=2)) == u.dimensionless_unscaled


@pytest.mark.parametrize("kind", KINDS)
def test_spectrogram_keeps_state_unit_and_axes(kind):
    m = _make(kind)
    result = m.spectrogram(2, fftlength=1)
    assert isinstance(result, SpectrogramMatrix)
    assert result.time_reference == EXPECTED_STATE[kind]
    assert np.all(np.isfinite(np.asarray(result.times.value)))
    assert np.asarray(result.frequencies.value)[0] == 0.0
    assert result.dt.value == pytest.approx(2.0)
    assert result.meta[0, 0].unit == UNIT**2 / u.Hz
    # element projection of the result keeps the state as well
    assert result[0, 0].time_reference == EXPECTED_STATE[kind]


def test_fft_values_match_per_element_timeseries_fft():
    """The matrix transform equals the single-series transform (same state)."""
    m = _make("relative")
    fft = m.fft()
    for i in range(2):
        for j in range(2):
            single = m[i, j].fft()
            np.testing.assert_array_equal(np.asarray(fft.value)[i, j], single.value)
            assert single.time_reference == REL


@pytest.mark.parametrize("kind", KINDS)
def test_hdf5_preserves_time_reference(kind, tmp_path):
    src = _make(kind)
    path = tmp_path / "matrix.h5"
    src.write(path)
    result = type(src).read(path)
    assert result.time_reference == src.time_reference
    np.testing.assert_array_equal(result.value, src.value)
    np.testing.assert_array_equal(result.xindex, src.xindex)
    assert result.meta[0, 0].unit == src.meta[0, 0].unit


def test_hdf5_legacy_missing_reference_is_absolute(tmp_path):
    import h5py

    src = _make("relative")
    path = tmp_path / "legacy.h5"
    src.write(path)
    with h5py.File(path, "a") as h5file:
        del h5file.attrs["gwexpy_time_reference"]
    assert type(src).read(path).time_reference == ABS
