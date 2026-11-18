"""SpectrogramMatrix timing-reference state (#725, v0.3.0 Step 7c-1, decision U-1).

Contract under test (``SpectrogramMatrix.epoch`` is independent of ``times``;
the explicit ``times`` axis alone does not make a matrix absolute):

* ``epoch`` omitted, ``None`` or ``NaN`` -> ``time_reference == "relative"``
  and the public ``epoch`` is unchanged (``0.0`` omitted, ``None`` for ``None``)
* a finite value, including an explicit ``0.0`` -> ``"absolute"``
* ``+Inf`` / ``-Inf`` -> ``ValueError``
* the state agrees with ``Spectrogram`` for omitted and explicit origins
* scalar selection uses ``times[0]`` for the selected element's first sample;
  the matrix epoch remains the origin and may differ from a nonzero first axis
* the state survives matrix operations, scalar selection, the ``Spectrogram``
  collection conversions and never changes a value, unit or axis

Tolerances come from ``np.spacing`` of the quantity compared, or the comparison
is exact because both sides run the same computation (R-15).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import (
    Spectrogram,
    SpectrogramDict,
    SpectrogramList,
    SpectrogramMatrix,
)
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
)

REL = TIME_REFERENCE_RELATIVE
ABS = TIME_REFERENCE_ABSOLUTE
GPS = 1.4e9
NT = 6
NF = 4
DT = 2.0
DF = 0.5
UNIT = u.m**2 / u.Hz

KINDS = ["relative", "absolute_zero", "absolute_gps"]
EXPECTED_STATE = {"relative": REL, "absolute_zero": ABS, "absolute_gps": ABS}
EXPECTED_EPOCH = {"relative": 0.0, "absolute_zero": 0.0, "absolute_gps": GPS}
ORIGIN = {"relative": 0.0, "absolute_zero": 0.0, "absolute_gps": GPS}


def _values(shape=(2, 2, NT, NF)) -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.abs(rng.standard_normal(shape)) + 1.0


def _times(origin: float) -> u.Quantity:
    return (origin + DT * np.arange(NT)) * u.s


def _freqs() -> u.Quantity:
    return DF * np.arange(NF) * u.Hz


def _make(kind: str, *, shape=(2, 2, NT, NF)) -> SpectrogramMatrix:
    kw = {"epoch": np.nan} if kind == "relative" else {"epoch": EXPECTED_EPOCH[kind]}
    return SpectrogramMatrix(
        _values(shape),
        times=_times(ORIGIN[kind]),
        frequencies=_freqs(),
        unit=UNIT,
        **kw,
    )


def _tol(*values) -> float:
    scale = max(1.0, *(float(np.max(np.abs(v))) for v in values))
    return 4 * np.spacing(scale)


def _unit(m) -> u.UnitBase:
    return m.meta[0, 0].unit


def _epoch_gps(obj) -> float:
    ep = obj.epoch
    return float(getattr(ep, "gps", ep))


# ------------------------------------------------------------ construction


class TestConstructor:
    @pytest.mark.parametrize(
        ("kw", "expected_state", "public"),
        [
            ({}, ABS, 0.0),  # omitted -> GWpy-compatible absolute
            ({"epoch": None}, REL, None),  # explicit None -> relative
            ({"epoch": np.nan}, REL, 0.0),  # NaN is never stored
            ({"epoch": float("nan") * u.s}, REL, 0.0),
        ],
    )
    def test_missing_or_nan_is_relative(self, kw, expected_state, public):
        m = SpectrogramMatrix(
            _values(), times=_times(0.0), frequencies=_freqs(), unit=UNIT, **kw
        )
        assert m.time_reference == expected_state
        assert m.epoch == public
        assert np.all(np.isfinite(np.asarray(m.times.value)))
        assert m.dt.value == DT
        assert _unit(m) == UNIT

    @pytest.mark.parametrize("epoch", [0, 0.0, -0.0, GPS])
    def test_finite_is_absolute_including_zero(self, epoch):
        m = SpectrogramMatrix(
            _values(), times=_times(0.0), frequencies=_freqs(), epoch=epoch
        )
        assert m.time_reference == ABS
        assert m.epoch == epoch

    @pytest.mark.parametrize("epoch", [np.inf, -np.inf])
    def test_infinite_raises_clear_valueerror(self, epoch):
        with pytest.raises(ValueError, match="must not be infinite"):
            SpectrogramMatrix(
                _values(), times=_times(0.0), frequencies=_freqs(), epoch=epoch
            )

    def test_times_alone_is_absolute_and_explicit_none_is_relative(self):
        """Omitted epoch defaults to absolute; explicit epoch=None is relative."""
        m = SpectrogramMatrix(
            _values(), times=_times(GPS), frequencies=_freqs(), unit=UNIT
        )
        assert m.time_reference == ABS
        assert m.epoch == 0.0
        assert np.asarray(m.times.value)[0] == GPS

        m_rel = SpectrogramMatrix(
            _values(),
            times=_times(GPS),
            frequencies=_freqs(),
            unit=UNIT,
            epoch=None,
        )
        assert m_rel.time_reference == REL
        assert m_rel.epoch is None

    def test_three_dimensional_layout(self):
        m = SpectrogramMatrix(
            _values((3, NT, NF)), times=_times(0.0), frequencies=_freqs(), epoch=0.0
        )
        assert m.time_reference == ABS
        assert m[0].time_reference == ABS
        rel = SpectrogramMatrix(
            _values((3, NT, NF)), times=_times(0.0), frequencies=_freqs(), epoch=None
        )
        assert rel.time_reference == REL
        assert rel[0].time_reference == REL

    @pytest.mark.parametrize("kind", KINDS)
    def test_rebuild_from_a_matrix_keeps_epoch_and_state(self, kind):
        src = _make(kind)
        rebuilt = SpectrogramMatrix(src, times=src.times, frequencies=src.frequencies)
        assert rebuilt.time_reference == EXPECTED_STATE[kind]
        assert rebuilt.epoch == src.epoch

    def test_rebuild_with_explicit_epoch_overrides_the_state(self):
        rel = _make("relative")
        absolute = SpectrogramMatrix(
            rel, times=rel.times, frequencies=rel.frequencies, epoch=GPS
        )
        assert absolute.time_reference == ABS
        assert rel.time_reference == REL


@pytest.mark.parametrize(
    ("t0", "expected"),
    [("omitted", ABS), (None, REL), (float("nan"), REL), (0.0, ABS), (GPS, ABS)],
)
def test_state_agrees_with_spectrogram(t0, expected):
    """#725 acceptance 6: Spectrogram and SpectrogramMatrix agree."""
    kw = {} if t0 == "omitted" else {"t0": t0}
    sg = Spectrogram(_values((NT, NF)), dt=DT, df=DF, unit=UNIT, **kw)
    ekw = {} if t0 == "omitted" else {"epoch": t0}
    m = SpectrogramMatrix(
        _values(), times=_times(0.0), frequencies=_freqs(), unit=UNIT, **ekw
    )
    assert sg.time_reference == m.time_reference == expected


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
    "column_slice": (lambda m: m[:, :1], UNIT),
    "row_of_4d": (lambda m: m[0], UNIT),
    "copy": (lambda m: m.copy(), UNIT),
    "copy_copy": (lambda m: copy.copy(m), UNIT),
    "deepcopy": (lambda m: copy.deepcopy(m), UNIT),
    "pickle": (lambda m: pickle.loads(pickle.dumps(m)), UNIT),
    "astype": (lambda m: m.astype(np.float32), UNIT),
    "real": (lambda m: m.real, UNIT),
    "conj": (lambda m: m.conj(), UNIT),
    "add": (lambda m: m + m, UNIT),
    "scale_left": (lambda m: 2 * m, UNIT),
    "neg": (lambda m: -m, UNIT),
    "square": (lambda m: m**2, UNIT**2),
    "radian": (lambda m: m.radian(), u.rad),
    "degree": (lambda m: m.degree(), u.deg),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(OPERATIONS))
def test_operation_keeps_state_epoch_unit_and_axes(name, kind):
    op, unit = OPERATIONS[name]
    src = _make(kind)
    result = op(src)
    assert isinstance(result, SpectrogramMatrix)
    assert result.time_reference == EXPECTED_STATE[kind]
    assert result.epoch == EXPECTED_EPOCH[kind]
    assert _unit(result) == unit
    times = np.asarray(result.times.value)
    expected_t = np.asarray(src.times.value)
    np.testing.assert_allclose(times, expected_t, rtol=0, atol=_tol(times, expected_t))
    freqs = np.asarray(result.frequencies.value)
    np.testing.assert_array_equal(freqs, np.asarray(_freqs().value))


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
    rel = _make("relative")
    absolute = SpectrogramMatrix(
        _values(), times=_times(0.0), frequencies=_freqs(), unit=UNIT, epoch=0.0
    )
    assert (rel + absolute).time_reference == REL
    assert (absolute + rel).time_reference == ABS


def test_none_epoch_matrix_keeps_none_through_operations():
    m = SpectrogramMatrix(
        _values(), times=_times(0.0), frequencies=_freqs(), unit=UNIT, epoch=None
    )
    for result in (m.copy(), m[:1], m * 2, pickle.loads(pickle.dumps(m))):
        assert result.epoch is None
        assert result.time_reference == REL


# --------------------------------------------------- scalar selection


@pytest.mark.parametrize("kind", KINDS)
def test_scalar_selection_keeps_state_axes_and_unit(kind):
    m = _make(kind)
    for sg in (m[0, 1], *m.to_series_1Dlist(), *m.to_list()):
        assert isinstance(sg, Spectrogram)
        assert sg.time_reference == EXPECTED_STATE[kind]
        assert sg.dt.value == DT
        assert sg.df.value == DF
        assert sg.unit == UNIT
        np.testing.assert_allclose(
            sg.times.value,
            _times(ORIGIN[kind]).value,
            rtol=0,
            atol=_tol(sg.times.value),
        )
    np.testing.assert_array_equal(m[0, 1].value, _values()[0, 1])


def test_relative_matrix_with_gps_times_selects_without_epoch_conflict():
    """Step 6 observation 10: relative matrix selects without epoch conflict."""
    m = SpectrogramMatrix(
        _values(), times=_times(GPS), frequencies=_freqs(), unit=UNIT, epoch=np.nan
    )
    assert m.time_reference == REL
    sg = m[0, 0]
    assert sg.time_reference == REL
    assert sg.times.value[0] == GPS
    assert sg.t0.value == GPS  # the explicit axis decides the element origin
    np.testing.assert_array_equal(sg.value, _values()[0, 0])
    assert sg.unit == UNIT


def test_absolute_epoch_that_disagrees_with_times_uses_explicit_axis():
    """The explicit time axis supplies the selected element's first sample."""
    m = SpectrogramMatrix(
        _values(), times=_times(GPS), frequencies=_freqs(), unit=UNIT, epoch=0
    )
    assert m.time_reference == ABS
    selected = m[0, 0]
    assert selected.time_reference == ABS
    tolerance = 4 * np.spacing(GPS)
    np.testing.assert_allclose(
        selected.times.to_value(u.s), m.times.to_value(u.s), rtol=0, atol=tolerance
    )
    assert selected.t0.value == pytest.approx(
        m.times[0].to_value(u.s), rel=0, abs=tolerance
    )


def test_absolute_epoch_that_agrees_with_times_selects():
    m = SpectrogramMatrix(
        _values(), times=_times(GPS), frequencies=_freqs(), unit=UNIT, epoch=GPS
    )
    sg = m[0, 0]
    assert sg.time_reference == ABS
    assert sg.t0.value == GPS


# ------------------------------------------------- Spectrogram conversions


def _spectrograms(kind: str, count: int = 2) -> list[Spectrogram]:
    kw = {"t0": None} if kind == "relative" else {"t0": ORIGIN[kind]}
    return [
        Spectrogram(
            _values((2, 2, NT, NF))[0, j],
            dt=DT,
            df=DF,
            unit=UNIT,
            name=f"s{j}",
            **kw,
        )
        for j in range(count)
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_list_and_dict_to_matrix_keep_state_and_epoch(kind):
    series = _spectrograms(kind)
    for matrix in (
        SpectrogramList(series).to_matrix(),
        SpectrogramDict({s.name: s for s in series}).to_matrix(),
    ):
        assert isinstance(matrix, SpectrogramMatrix)
        assert matrix.time_reference == EXPECTED_STATE[kind]
        # changed by U-1 (#725, Step 6 observation 10): the epoch is no longer
        # reset to 0.0 by the conversion
        assert matrix.epoch == pytest.approx(
            ORIGIN[kind], abs=16 * np.spacing(max(1.0, ORIGIN[kind]))
        )
        assert matrix.dt.value == DT
        np.testing.assert_allclose(
            np.asarray(matrix.times.value),
            _times(ORIGIN[kind]).value,
            rtol=0,
            atol=_tol(np.asarray(matrix.times.value)),
        )
        np.testing.assert_array_equal(
            np.asarray(matrix.frequencies.value), _freqs().value
        )
        assert matrix.meta[0, 0].unit == UNIT
        for j, s in enumerate(series):
            np.testing.assert_array_equal(np.asarray(matrix.value)[j], s.value)
        # a converted matrix can always be projected back (the epoch agrees)
        for sg in matrix.to_series_1Dlist():
            assert sg.time_reference == EXPECTED_STATE[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_matrix_to_list_to_matrix_round_trip(kind):
    m = _make(kind, shape=(3, NT, NF))
    back = m.to_list().to_matrix()
    assert back.time_reference == EXPECTED_STATE[kind]
    np.testing.assert_array_equal(np.asarray(back.value), np.asarray(m.value))
    np.testing.assert_allclose(
        np.asarray(back.times.value),
        np.asarray(m.times.value),
        rtol=0,
        atol=_tol(np.asarray(m.times.value)),
    )
    assert back.meta[0, 0].unit == UNIT


def test_direct_construction_from_gps_times_stays_epoch_zero_current():
    """Characterizes Step 6 observation 10 for the direct constructor.

    Only the collection conversions follow the series epoch; building a matrix
    from a Spectrogram's ``times`` without ``epoch`` keeps the default 0.0 (now
    a relative matrix, so no epoch/times conflict is raised on selection).
    """
    sg = _spectrograms("absolute_gps", 1)[0]
    m = SpectrogramMatrix(
        np.asarray(sg.value).reshape(1, 1, NT, NF),
        times=sg.times,
        frequencies=sg.frequencies,
        unit=UNIT,
    )
    assert m.epoch == 0.0
    assert m.time_reference == ABS
    assert np.asarray(m.times.value)[0] == GPS

    m_rel = SpectrogramMatrix(
        np.asarray(sg.value).reshape(1, 1, NT, NF),
        times=sg.times,
        frequencies=sg.frequencies,
        unit=UNIT,
        epoch=np.nan,
    )
    assert m_rel.epoch == 0.0
    assert m_rel.time_reference == REL
    assert m_rel[0, 0].time_reference == REL


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
