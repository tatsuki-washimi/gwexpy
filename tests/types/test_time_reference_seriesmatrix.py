"""SeriesMatrix timing-reference state (#725, v0.3.0 Step 7c-1, decision U-1).

Contract under test (the base ``epoch`` is the only time origin the generic
``SeriesMatrix`` knows; ``xindex`` may be any axis):

* ``epoch`` omitted, ``None`` or ``NaN`` -> ``time_reference == "relative"``
* finite value, including an explicit ``0.0`` or int ``0`` -> ``"absolute"``
* ``+Inf`` / ``-Inf`` -> ``ValueError``
* assignment accepts finite values (state unchanged) and rejects ``NaN``/``±Inf``
* the public ``epoch`` value is unchanged (``0.0`` omitted, ``None`` for
  ``None``, the raw finite value otherwise); ``NaN`` is never stored
* the state survives ``OPERATIONS`` (copies, arithmetic, math helpers,
  structure methods, selections and pickle), and mixed arithmetic follows the
  first operand
* the state never changes a computed value, unit or axis

Tolerances come from ``np.spacing`` of the quantity compared, or the comparison
is exact because both sides run the same computation (R-15).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_NAME,
    TIME_REFERENCE_RELATIVE,
    TimeReferenceMixin,
)
from gwexpy.types import SeriesMatrix

REL = TIME_REFERENCE_RELATIVE
ABS = TIME_REFERENCE_ABSOLUTE
GPS = 1.4e9
T = 12
DX = 0.1
UNIT = u.m

KINDS = ["relative", "absolute_zero", "absolute_gps"]
EXPECTED_STATE = {"relative": REL, "absolute_zero": ABS, "absolute_gps": ABS}
EXPECTED_EPOCH = {"relative": 0.0, "absolute_zero": 0.0, "absolute_gps": GPS}


def _values() -> np.ndarray:
    rng = np.random.default_rng(7)
    # a well-conditioned 2x2 stack so inv/det are defined at every sample
    base = rng.standard_normal((2, 2, T))
    return base + 3.0 * np.eye(2)[:, :, None]


def _axis() -> u.Quantity:
    return np.arange(T) * DX * u.s


def _build(**kwargs) -> SeriesMatrix:
    return SeriesMatrix(_values(), xindex=_axis(), xunit="s", unit=UNIT, **kwargs)


def _make(kind: str) -> SeriesMatrix:
    if kind == "relative":
        return _build(epoch=np.nan)
    if kind == "absolute_zero":
        return _build(epoch=0.0)
    return _build(epoch=GPS)


def _tol(*values) -> float:
    """Four spacings of the largest magnitude involved (R-15)."""
    scale = max(1.0, *(float(np.max(np.abs(v))) for v in values))
    return 4 * np.spacing(scale)


# ------------------------------------------------------------ construction


class TestConstructor:
    def test_is_a_time_reference_container(self):
        assert isinstance(_build(), TimeReferenceMixin)
        assert TIME_REFERENCE_NAME == "time_reference"

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
        m = _build(**kw)
        assert m.time_reference == expected_state
        assert m.epoch == public
        assert m.meta[0, 0].unit == UNIT
        assert m.dx.value == pytest.approx(DX)
        np.testing.assert_array_equal(m.xindex.value, _axis().value)

    @pytest.mark.parametrize("epoch", [0, 0.0, -0.0, GPS, GPS * u.s, np.float64(GPS)])
    def test_finite_is_absolute_including_zero(self, epoch):
        m = _build(epoch=epoch)
        assert m.time_reference == ABS
        assert m.epoch == epoch  # the raw public value is unchanged
        np.testing.assert_array_equal(m.xindex.value, _axis().value)

    @pytest.mark.parametrize("epoch", [np.inf, -np.inf, np.inf * u.s])
    def test_infinite_raises_clear_valueerror(self, epoch):
        with pytest.raises(ValueError, match="must not be infinite"):
            _build(epoch=epoch)

    def test_explicit_none_and_zero_are_distinguishable(self):
        assert _build(epoch=np.nan).epoch == _build(epoch=0.0).epoch == 0.0
        assert _build(epoch=np.nan).time_reference == REL
        assert _build(epoch=None).time_reference == REL
        assert _build().time_reference == ABS
        assert _build(epoch=0.0).time_reference == ABS

    def test_state_does_not_depend_on_data_scale(self):
        small = SeriesMatrix(_values() * 1e-20, xindex=_axis(), xunit="s", epoch=None)
        assert small.time_reference == REL
        assert (
            SeriesMatrix(
                _values() * 1e-20, xindex=_axis(), xunit="s", epoch=0.0
            ).time_reference
            == ABS
        )

    @pytest.mark.parametrize("kind", KINDS)
    def test_rebuild_from_a_matrix_keeps_state(self, kind):
        src = _make(kind)
        rebuilt = SeriesMatrix(src)
        assert rebuilt.time_reference == EXPECTED_STATE[kind]
        assert rebuilt.epoch == src.epoch
        assert rebuilt.meta[0, 0].unit == UNIT

    def test_rebuild_with_explicit_epoch_overrides_the_state(self):
        rel = _make("relative")
        absolute = SeriesMatrix(rel, epoch=GPS)
        assert absolute.time_reference == ABS
        assert rel.time_reference == REL  # the source is untouched
        back = SeriesMatrix(_make("absolute_gps"), epoch=None)
        assert back.time_reference == REL


# ------------------------------------------------------------------ setter


class TestSetter:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, np.nan * u.s])
    @pytest.mark.parametrize("kind", KINDS)
    def test_nonfinite_assignment_raises_and_keeps_value(self, kind, bad):
        m = _make(kind)
        before = (m.epoch, m.time_reference)
        with pytest.raises(ValueError, match="epoch must be a finite time"):
            m.epoch = bad
        assert (m.epoch, m.time_reference) == before

    @pytest.mark.parametrize("kind", KINDS)
    def test_finite_assignment_keeps_the_state(self, kind):
        m = _make(kind)
        m.epoch = 5.0
        assert m.epoch == 5.0
        assert m.time_reference == EXPECTED_STATE[kind]

    def test_views_do_not_revalidate_a_stored_value(self):
        m = _make("relative")
        m.__dict__["epoch"] = float("nan")  # a legacy value
        assert np.isnan(m.view(SeriesMatrix).epoch)

    def test_missing_epoch_attribute_is_an_attribute_error(self):
        m = _make("relative")
        del m.__dict__["epoch"]
        assert not hasattr(m, "epoch")


# ------------------------------------------------------------ operations

SQUARE_UNIT = UNIT**2

# name -> (operation, expected unit of cell (0, 0))
OPERATIONS = {
    "view": (lambda m: m.view(type(m)), UNIT),
    "row_slice": (lambda m: m[:1], UNIT),
    "cell_slice": (lambda m: m[:1, :1], UNIT),
    "sample_slice": (lambda m: m[:, :, 2:9], UNIT),
    "paired_list_selection": (lambda m: m[[1, 0], [0, 1]], UNIT),
    "submatrix": (lambda m: m.submatrix(["row0"], ["col1"]), UNIT),
    "copy": (lambda m: m.copy(), UNIT),
    "copy_copy": (lambda m: copy.copy(m), UNIT),
    "deepcopy": (lambda m: copy.deepcopy(m), UNIT),
    "pickle": (lambda m: pickle.loads(pickle.dumps(m)), UNIT),
    "pickle_protocol_2": (lambda m: pickle.loads(pickle.dumps(m, protocol=2)), UNIT),
    "astype": (lambda m: m.astype(np.float32), UNIT),
    "real": (lambda m: m.real, UNIT),
    "imag": (lambda m: m.imag, UNIT),
    "conj": (lambda m: m.conj(), UNIT),
    "transpose": (lambda m: m.transpose(), UNIT),
    "T": (lambda m: m.T, UNIT),
    "reshape": (lambda m: m.reshape(1, 4), UNIT),
    "round": (lambda m: m.round(1), UNIT),
    "crop": (lambda m: m.crop(m.xindex[2], m.xindex[8]), UNIT),
    "diff": (lambda m: m.diff(), UNIT),
    "pad": (lambda m: m.pad(2), UNIT),
    "add": (lambda m: m + m, UNIT),
    "scale_right": (lambda m: m * 2, UNIT),
    "scale_left": (lambda m: 2 * m, UNIT),
    "neg": (lambda m: -m, UNIT),
    "abs": (lambda m: abs(m), UNIT),
    "square": (lambda m: m**2, SQUARE_UNIT),
    "matmul": (lambda m: m @ m, SQUARE_UNIT),
    "diagonal_matrix": (lambda m: m.diagonal(output="matrix"), UNIT),
    "diagonal_vector": (lambda m: m.diagonal(output="vector"), UNIT),
    "inv": (lambda m: m.inv(), 1 / UNIT),
    "schur": (lambda m: m.schur([0]), UNIT),
    "angle": (lambda m: m.angle(), u.rad),
}


def _expected_axis(name: str, m: SeriesMatrix) -> np.ndarray:
    """The sample axis an operation should give (an exact slice of the source)."""
    x = np.asarray(m.xindex.value)
    return {
        "sample_slice": x[2:9],
        "crop": x[2:8],
        "diff": x[1:],
        "pad": np.concatenate(
            [x[0] - DX * np.arange(2, 0, -1), x, x[-1] + DX * np.arange(1, 3)]
        ),
    }.get(name, x)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(OPERATIONS))
def test_operation_keeps_state_epoch_unit_and_axis(name, kind):
    op, unit = OPERATIONS[name]
    src = _make(kind)
    before = (src.time_reference, src.epoch)
    result = op(src)
    assert isinstance(result, SeriesMatrix)
    assert result.time_reference == EXPECTED_STATE[kind]
    # Step 6 observation 8: the epoch is the construction-time origin; the
    # axis origin moves under slicing but the epoch does not.
    assert result.epoch == EXPECTED_EPOCH[kind]
    assert result.meta[0, 0].unit == unit
    expected = _expected_axis(name, src)
    got = np.asarray(result.xindex.value)
    np.testing.assert_allclose(got, expected, rtol=0, atol=_tol(got, expected))
    assert (src.time_reference, src.epoch) == before  # the source is untouched


@pytest.mark.parametrize("kind", KINDS)
def test_values_are_independent_of_the_state(kind):
    """Relative and absolute matrices compute the same numbers."""
    rel, other = _make("relative"), _make(kind)
    for name, (op, _unit) in OPERATIONS.items():
        np.testing.assert_array_equal(
            np.asarray(op(rel).value), np.asarray(op(other).value), err_msg=name
        )


@pytest.mark.parametrize("kind", KINDS)
def test_inplace_arithmetic_keeps_the_state_and_unit(kind):
    m = _make(kind)
    ref = m.copy()
    m *= 2
    assert m.time_reference == EXPECTED_STATE[kind]
    assert m.meta[0, 0].unit == UNIT
    np.testing.assert_array_equal(np.asarray(m.value), 2 * np.asarray(ref.value))


def test_mixed_arithmetic_follows_the_first_operand():
    rel, absolute = _make("relative"), _make("absolute_gps")
    assert (rel + absolute).time_reference == REL
    assert (absolute + rel).time_reference == ABS
    assert (rel - absolute).time_reference == REL
    assert (absolute * rel).time_reference == ABS


def test_sample_slice_moves_the_axis_but_not_the_epoch_current():
    """Characterizes Step 6 observation 8 (not decided by U-1, not fixed)."""
    m = _make("absolute_gps")
    sl = m[:, :, 3:]
    assert sl.x0.value == pytest.approx(3 * DX, abs=_tol(3 * DX))
    assert sl.epoch == m.epoch  # unchanged: the epoch is not re-anchored
    assert sl.time_reference == ABS


def test_paired_list_selection_no_longer_resets_the_epoch():
    """Before U-1 this path went through a plain ndarray and reset epoch to 0.0."""
    m = _make("absolute_gps")
    assert m[[0, 1], [0, 1]].epoch == GPS


@pytest.mark.parametrize("kind", KINDS)
def test_none_epoch_matrix_keeps_none_through_operations(kind):
    m = _build(epoch=None)
    assert m.time_reference == REL
    for result in (m.copy(), m[:1], m * 2, pickle.loads(pickle.dumps(m)), m.T):
        assert result.epoch is None
        assert result.time_reference == REL


# --------------------------------------------------- derived element series


@pytest.mark.parametrize("kind", KINDS)
def test_trace_and_det_series_keep_values_unit_and_axis(kind):
    m = _make(kind)
    tr = m.trace()
    np.testing.assert_allclose(
        np.asarray(tr.value), np.asarray(m.value)[0, 0] + np.asarray(m.value)[1, 1]
    )
    assert tr.unit == UNIT
    det = m.det()
    expected = (
        np.asarray(m.value)[0, 0] * np.asarray(m.value)[1, 1]
        - np.asarray(m.value)[0, 1] * np.asarray(m.value)[1, 0]
    )
    np.testing.assert_allclose(np.asarray(det.value), expected, rtol=1e-12)
    assert det.unit == UNIT**2
    np.testing.assert_array_equal(np.asarray(tr.xindex.value), _axis().value)
    assert len(m.diagonal()) == 2


@pytest.mark.parametrize("kind", KINDS)
def test_hdf5_preserves_time_reference(kind, tmp_path):
    import h5py

    src = _make(kind)
    path = tmp_path / "matrix.h5"
    src.write(path)
    result = SeriesMatrix.read(path)
    assert result.time_reference == src.time_reference
    np.testing.assert_array_equal(result.value, src.value)
    np.testing.assert_array_equal(result.xindex, src.xindex)
    assert result.meta[0, 0].unit == src.meta[0, 0].unit
    with h5py.File(path, "a") as h5file:
        del h5file.attrs["gwexpy_time_reference"]
    assert SeriesMatrix.read(path).time_reference == ABS
