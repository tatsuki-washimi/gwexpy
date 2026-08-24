"""Operator contract for the SeriesMatrix family (issues #575 / #576 / #577).

`SeriesMatrix` and its subclasses set ``__array_ufunc__ = None``.  NumPy then
refuses every ufunc applied to them, which is what stops
``Quantity.__array_ufunc__`` from unwrapping a matrix into a bare ``Quantity``
and losing its per-cell units.  The price is that every supported operator has
to be spelled out explicitly, so this module pins the whole surface: which
operators exist, what they return, which ones are refused, and how units are
converted.

The three concrete classes are exercised together because the defect in #575
affected the family as a whole, and because `SpectrogramMatrix` carries its own
ufunc implementation for the 4-D ``(Row, Col, Time, Freq)`` layout.
"""

import operator
import pickle
import warnings
from copy import deepcopy

import numpy as np
import pytest
from astropy import units as u
from astropy.units import UnitConversionError

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.spectrogram import SpectrogramMatrix
from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix
from gwexpy.types.seriesmatrix import SeriesMatrix

from .series_matrix_contract_manifest import (
    B0_CONTRACT,
    EXPECTED_B0_CELL_COUNT,
    MatrixFamily,
    Phase,
)
from .test_series_matrix_contract_manifest import (
    _assert_source_unchanged,
    _observable_source_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta(unit=u.V):
    """2x2 metadata whose cells carry distinct names and channels."""
    return MetaDataMatrix(
        np.array(
            [
                [
                    MetaData(unit=unit, name="a00", channel="X1:A00"),
                    MetaData(unit=unit, name="a01", channel="X1:A01"),
                ],
                [
                    MetaData(unit=unit, name="a10", channel="X1:A10"),
                    MetaData(unit=unit, name="a11", channel="X1:A11"),
                ],
            ],
            dtype=object,
        )
    )


def make_timeseries_matrix(unit=u.V, values=None):
    """Build a (2, 2, 4) TimeSeriesMatrix with per-cell units."""
    if values is None:
        values = np.arange(16, dtype=float).reshape(2, 2, 4) + 1.0
    return TimeSeriesMatrix(values, meta=_meta(unit), dt=0.5 * u.s, t0=0.0, name="tsm")


def make_frequencyseries_matrix(unit=u.V, values=None):
    """Build a (2, 2, 4) FrequencySeriesMatrix with per-cell units."""
    if values is None:
        values = np.arange(16, dtype=float).reshape(2, 2, 4) + 1.0
    return FrequencySeriesMatrix(
        values, meta=_meta(unit), df=1.0 * u.Hz, f0=0.0, name="fsm"
    )


def make_spectrogram_matrix(unit=u.V, values=None):
    """Build a (2, 2, 2, 2) SpectrogramMatrix with per-cell units."""
    if values is None:
        values = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + 1.0
    return SpectrogramMatrix(
        values,
        times=np.arange(2) * u.s,
        frequencies=np.arange(2) * u.Hz,
        meta=_meta(unit),
        rows=["r0", "r1"],  # type: ignore[list-item]
        cols=["c0", "c1"],  # type: ignore[list-item]
        name="sgm",
    )


MATRIX_FACTORIES = {
    "TimeSeriesMatrix": make_timeseries_matrix,
    "FrequencySeriesMatrix": make_frequencyseries_matrix,
    "SpectrogramMatrix": make_spectrogram_matrix,
}

SERIES_FAMILY_FACTORIES = {
    "TimeSeriesMatrix": make_timeseries_matrix,
    "FrequencySeriesMatrix": make_frequencyseries_matrix,
}


@pytest.fixture(params=list(MATRIX_FACTORIES), ids=list(MATRIX_FACTORIES))
def factory(request):
    """Factory for one matrix class, parametrized over the whole family."""
    return MATRIX_FACTORIES[request.param]


@pytest.fixture
def matrix(factory):
    """A fresh matrix with per-cell unit V and distinct per-cell names."""
    return factory()


@pytest.fixture(params=list(SERIES_FAMILY_FACTORIES), ids=list(SERIES_FAMILY_FACTORIES))
def series_factory(request):
    """Factory restricted to the 3-D ``(row, col, sample)`` classes."""
    return SERIES_FAMILY_FACTORIES[request.param]


def cell_units(matrix):
    """Return the per-cell units as a nested list."""
    return [[matrix.meta[i, j].unit for j in range(2)] for i in range(2)]


def cell_names(matrix):
    """Return the per-cell names as a nested list."""
    return [[matrix.meta[i, j].name for j in range(2)] for i in range(2)]


def test_existing_operator_contract_is_backed_by_canonical_b0_manifest():
    """Keep this executable operator suite tied to the single B0 ledger."""
    assert len(B0_CONTRACT) == EXPECTED_B0_CELL_COUNT
    assert {cell.family for cell in B0_CONTRACT} == set(MatrixFamily)
    assert all(cell.phase is Phase.B0 for cell in B0_CONTRACT)
    assert {"add", "sub", "mul", "truediv", "power"} <= {
        cell.operation for cell in B0_CONTRACT
    }


@pytest.mark.parametrize(
    "factory", list(MATRIX_FACTORIES.values()), ids=list(MATRIX_FACTORIES)
)
@pytest.mark.parametrize("operation", ["copy", "conj"])
def test_structural_metadata_copies_split_shared_nested_cell_payloads(
    factory, operation
):
    """A new logical cell must not inherit a sibling's nested mutable payload."""
    matrix = factory()
    shared = {"nested": {"state": "source"}}
    matrix.meta[0, 0]["shared_payload"] = shared
    matrix.meta[0, 1]["shared_payload"] = shared

    result = matrix.copy() if operation == "copy" else matrix.conj()

    assert (
        result.meta[0, 0]["shared_payload"] is not result.meta[0, 1]["shared_payload"]
    )
    result.meta[0, 0]["shared_payload"]["nested"]["state"] = "result-only"
    assert result.meta[0, 1]["shared_payload"]["nested"]["state"] == "source"
    assert matrix.meta[0, 0]["shared_payload"]["nested"]["state"] == "source"


@pytest.mark.parametrize(
    "operation",
    [
        "shape",
        "dtype",
        "values",
        "slicing",
        "assignment",
        "iteration",
        "copy",
        "astype",
        "real",
        "imag",
        "conj",
        "transpose",
        "reshape",
        "np.asarray",
        "matrix.view(np.ndarray)",
    ],
)
@pytest.mark.parametrize(
    "factory", list(MATRIX_FACTORIES.values()), ids=list(MATRIX_FACTORIES)
)
def test_approved_structure_surface_is_observed_without_runtime_changes(
    factory, operation
):
    """Pin the B0 shape/data surface and its current Spectrogram limitations."""
    matrix = factory()
    if operation == "shape":
        assert isinstance(matrix.shape, tuple)
    elif operation == "dtype":
        assert isinstance(matrix.dtype, np.dtype)
    elif operation == "values":
        np.testing.assert_array_equal(np.asarray(matrix), matrix.value)
    elif operation == "slicing":
        if isinstance(matrix, SpectrogramMatrix):
            with pytest.raises(ValueError):
                matrix[..., :1]
        else:
            sliced = matrix[..., :1]
            assert type(sliced) is type(matrix)
            assert sliced.shape[-1] == 1
    elif operation == "assignment":
        assigned = matrix.copy()
        replacement = np.zeros_like(np.asarray(assigned))
        assigned[...] = replacement
        np.testing.assert_array_equal(np.asarray(assigned), replacement)
    elif operation == "iteration":
        rows = list(matrix)
        assert rows
        assert all(type(row) is type(matrix) for row in rows)
    elif operation == "copy":
        copied = matrix.copy()
        assert type(copied) is type(matrix)
        assert copied is not matrix
        assert copied.meta is not matrix.meta
    elif operation == "astype":
        converted = matrix.astype(np.float32)
        assert type(converted) is type(matrix)
        assert converted.dtype == np.dtype(np.float32)
    elif operation == "real":
        assert type(matrix.real) is type(matrix)
    elif operation == "imag":
        assert type(matrix.imag) is type(matrix)
    elif operation == "conj":
        assert type(matrix.conj()) is type(matrix)
    elif operation in {"transpose", "reshape"}:
        if isinstance(matrix, SpectrogramMatrix):
            with pytest.raises(ValueError):
                getattr(matrix, operation)()
        elif operation == "transpose":
            assert type(matrix.transpose()) is type(matrix)
        else:
            assert type(matrix.reshape(matrix.shape)) is type(matrix)
    elif operation == "np.asarray":
        assert type(np.asarray(matrix)) is np.ndarray
    elif operation == "matrix.view(np.ndarray)":
        assert type(matrix.view(np.ndarray)) is np.ndarray
    else:  # pragma: no cover - protects the manifest-to-test mapping
        raise AssertionError(operation)


# ---------------------------------------------------------------------------
# 1. Issue #575 acceptance: foreign left operands must not swallow the matrix
# ---------------------------------------------------------------------------


LEFT_OPERAND_CASES = {
    "quantity": (2 * u.s, u.V * u.s, 2.0),
    "bare_unit": (u.s, u.V * u.s, 1.0),
    "int": (2, u.V, 2.0),
    "float": (2.0, u.V, 2.0),
    "np_float64": (np.float64(2), u.V, 2.0),
    "ndarray_0d": (np.array(2.0), u.V, 2.0),
}


@pytest.mark.parametrize(
    ("operand", "expected_unit", "factor"),
    list(LEFT_OPERAND_CASES.values()),
    ids=list(LEFT_OPERAND_CASES),
)
def test_multiplication_preserves_matrix_from_either_side(
    matrix, operand, expected_unit, factor
):
    """``x * matrix`` and ``matrix * x`` both keep the matrix and its units.

    This is the acceptance condition of #575: before the fix, a ``Quantity``
    or bare ``Unit`` on the left won the NEP 13 dispatch and returned a bare
    ``Quantity`` whose unit was the operand's alone -- the matrix's V silently
    disappeared.
    """
    expected_values = np.asarray(matrix) * factor

    for result in (operand * matrix, matrix * operand):
        assert type(result) is type(matrix)
        assert result.shape == matrix.shape
        assert result.name == matrix.name
        np.testing.assert_allclose(np.asarray(result), expected_values)
        for row in cell_units(result):
            for unit in row:
                assert unit == expected_unit
        assert cell_names(result) == cell_names(matrix)


def test_multiplication_by_plain_ndarray_from_the_left(matrix):
    """A plain ndarray on the left also comes back through the matrix."""
    operand = np.full(matrix.shape, 2.0)
    result = operand * matrix
    assert type(result) is type(matrix)
    assert cell_units(result) == cell_units(matrix)
    np.testing.assert_allclose(np.asarray(result), np.asarray(matrix) * 2.0)


def test_axis_metadata_survives_left_multiplication(factory):
    """Sample-axis metadata is not collateral damage of #575."""
    matrix = factory()
    result = (2 * u.s) * matrix
    if isinstance(matrix, SpectrogramMatrix):
        np.testing.assert_allclose(
            result.times.to_value(u.s), matrix.times.to_value(u.s)
        )
        np.testing.assert_allclose(
            result.frequencies.to_value(u.Hz), matrix.frequencies.to_value(u.Hz)
        )
    else:
        np.testing.assert_allclose(np.asarray(result.xindex), np.asarray(matrix.xindex))
    assert list(result.rows.keys()) == list(matrix.rows.keys())
    assert list(result.cols.keys()) == list(matrix.cols.keys())


# ---------------------------------------------------------------------------
# 2. Full operator sweep
# ---------------------------------------------------------------------------


BINARY_OPERATORS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "truediv": lambda a, b: a / b,
    "floordiv": lambda a, b: a // b,
    "mod": lambda a, b: a % b,
}

# floordiv/mod are unit-naive (NumPy's remainder/floor-divide do not convert
# units) and are therefore rejected for a unit-bearing operand -- see
# test_modulo_and_floor_divide_do_not_ignore_units. Excluded from the
# "arbitrary matrix operand" sweep below, which uses the default V-unit
# fixture.
UNIT_SAFE_BINARY_OPERATORS = {
    k: v for k, v in BINARY_OPERATORS.items() if k not in ("floordiv", "mod")
}


@pytest.mark.parametrize(
    "op", list(UNIT_SAFE_BINARY_OPERATORS), ids=list(UNIT_SAFE_BINARY_OPERATORS)
)
def test_binary_operators_with_matrix_operand(matrix, factory, op):
    """Every arithmetic operator accepts another matrix and stays in the family."""
    other = factory()
    result = UNIT_SAFE_BINARY_OPERATORS[op](matrix, other)
    assert type(result) is type(matrix)
    assert result.shape == matrix.shape


@pytest.mark.parametrize("op", list(BINARY_OPERATORS), ids=list(BINARY_OPERATORS))
def test_binary_operators_are_reflected_for_plain_arrays(factory, op):
    """The reflected form is reached for a plain ndarray on the left.

    A dimensionless matrix is used so that ``+``/``-`` against a unitless
    array is physically meaningful.
    """
    matrix = factory(unit=u.dimensionless_unscaled)
    other = np.full(matrix.shape, 3.0)
    result = BINARY_OPERATORS[op](other, matrix)
    assert type(result) is type(matrix)
    np.testing.assert_allclose(
        np.asarray(result),
        BINARY_OPERATORS[op](other, np.asarray(matrix)),
    )


def test_unary_operators(matrix):
    """``-``, ``+`` and ``abs()`` keep the class and the per-cell units."""
    for result in (-matrix, +matrix, abs(matrix)):
        assert type(result) is type(matrix)
        assert cell_units(result) == cell_units(matrix)
    np.testing.assert_allclose(np.asarray(-matrix), -np.asarray(matrix))
    np.testing.assert_allclose(np.asarray(abs(matrix)), np.abs(np.asarray(matrix)))


def test_divmod_is_explicitly_unsupported(matrix):
    """``divmod()`` always raises, even for dimensionless operands.

    Composing it from ``//``/``%`` would just re-surface their unit-naive
    result; main never implemented ``__divmod__`` at all, so this restores
    that explicit failure rather than reintroducing a working-but-wrong pair
    (docs/plans/2026-08-04-v0113-contract-rulings.md).
    """
    with pytest.raises(TypeError):
        divmod(matrix, 3)


def test_rdivmod_is_explicitly_unsupported(matrix):
    """The reflected ``divmod()`` is unsupported too."""
    other = np.full(matrix.shape, 30.0)
    with pytest.raises(TypeError):
        divmod(other, matrix)


def test_matmul_requires_two_matrices(series_factory):
    """``@`` works between matrices and refuses a bare array."""
    left = series_factory()
    result = left @ series_factory()
    assert type(result) is type(left)
    assert result.meta[0, 0].unit == u.V**2
    with pytest.raises(TypeError):
        left @ np.ones((2, 2))
    with pytest.raises(TypeError):
        np.ones((2, 2)) @ left


# ---------------------------------------------------------------------------
# 3. Operand shapes accepted by the 3-D classes
# ---------------------------------------------------------------------------


def test_ndarray_operand_broadcast_shapes(series_factory):
    """0-D fills, 1-D spans the sample axis, 2-D spans the cells, 3-D is exact."""
    matrix = series_factory()
    n_row, n_col, n_sample = matrix.shape

    per_sample = np.arange(n_sample, dtype=float) + 1.0
    result = matrix * per_sample
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(matrix) * per_sample.reshape(1, 1, n_sample)
    )

    per_cell = np.arange(n_row * n_col, dtype=float).reshape(n_row, n_col) + 1.0
    result = matrix * per_cell
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(matrix) * per_cell.reshape(n_row, n_col, 1)
    )

    full = np.full(matrix.shape, 2.0)
    np.testing.assert_allclose(np.asarray(matrix * full), np.asarray(matrix) * 2.0)


def test_ndarray_operand_wrong_shape_raises(series_factory):
    """A mis-shaped operand is a ValueError, not a silent broadcast."""
    matrix = series_factory()
    with pytest.raises(ValueError):
        matrix * np.ones(matrix.shape[2] + 1)
    with pytest.raises(ValueError):
        matrix * np.ones((5, 5))
    with pytest.raises(ValueError):
        matrix * np.ones((2, 2, 2, 2))


# ---------------------------------------------------------------------------
# 4. In-place operators
# ---------------------------------------------------------------------------


# The additive operators need a unit-carrying operand, the multiplicative ones
# a plain number; both must land in the existing buffer. floordiv/mod are
# excluded here because the default `matrix` fixture carries unit V, and
# those two now reject unit-bearing operands (see
# test_inplace_floordiv_and_mod_reject_unit_bearing_matrix below for the
# dimensionless in-place case).
INPLACE_CASES = {
    "iadd": ("add", 2 * u.V),
    "isub": ("sub", 2 * u.V),
    "imul": ("mul", 2),
    "itruediv": ("truediv", 2),
}


@pytest.mark.parametrize("op", list(INPLACE_CASES), ids=list(INPLACE_CASES))
def test_inplace_operators_write_the_existing_buffer(matrix, op):
    """In-place operators really mutate the buffer instead of rebinding.

    ``ndarray.__imul__`` used to reach ``__array_ufunc__`` with ``out=self``,
    which was silently dropped, so ``a *= b`` behaved as ``a = a * b``
    (issue #577c).
    """
    plain_op, operand = INPLACE_CASES[op]
    alias = np.asarray(matrix)
    expected = BINARY_OPERATORS[plain_op](np.asarray(matrix).copy(), 2.0)
    identity = id(matrix)

    result = getattr(matrix, f"__{op}__")(operand)

    assert result is matrix
    assert id(result) == identity
    np.testing.assert_allclose(alias, expected)
    np.testing.assert_allclose(np.asarray(matrix), expected)


def test_inplace_floordiv_and_mod_reject_unit_bearing_matrix(matrix):
    """``//=``/``%=`` refuse a unit-bearing buffer, mirroring ``//``/``%``."""
    with pytest.raises(TypeError):
        matrix.__ifloordiv__(2)
    with pytest.raises(TypeError):
        matrix.__imod__(2)


def test_inplace_floordiv_and_mod_work_for_dimensionless(factory):
    """``//=``/``%=`` still mutate the buffer in place for dimensionless operands."""
    matrix = factory(unit=u.dimensionless_unscaled)
    alias = np.asarray(matrix)
    expected_floordiv = np.asarray(matrix).copy() // 3
    identity = id(matrix)

    result = matrix.__ifloordiv__(3)

    assert result is matrix
    assert id(result) == identity
    np.testing.assert_allclose(alias, expected_floordiv)


def test_inplace_multiply_updates_units(matrix):
    """``*=`` recomposes the per-cell units like ``*`` does."""
    matrix *= 2 * u.s
    for row in cell_units(matrix):
        for unit in row:
            assert unit == u.V * u.s


def _iadd_incompatible_unit(m):
    m += 1 * u.s


def _itruediv_zero(m):
    m /= 0


def _imul_bad_shape(m):
    m *= np.ones(99)


def _decorate_rejection_state(matrix):
    """Add nested mutable state covered by the authoritative snapshot helper."""
    shared = {"nested": [np.arange(3, dtype=np.int64)]}
    matrix.attrs["contract"] = shared
    matrix.provenance = {"nested": [shared, {"alias": shared}]}
    return matrix


@pytest.mark.parametrize(
    ("failing_op", "expected_error"),
    [
        (_iadd_incompatible_unit, UnitConversionError),
        (_itruediv_zero, ZeroDivisionError),
        (_imul_bad_shape, (TypeError, ValueError)),
    ],
    ids=["incompatible_unit", "zero_divisor", "bad_shape"],
)
def test_inplace_failure_leaves_the_matrix_untouched(
    matrix, failing_op, expected_error
):
    """A rejected in-place operation must preserve the complete source graph."""
    matrix = _decorate_rejection_state(matrix)
    snapshot = _observable_source_snapshot(matrix)

    with pytest.raises(expected_error):
        failing_op(matrix)

    _assert_source_unchanged(matrix, snapshot)


def test_inplace_rejects_unsafe_dtype_change(matrix):
    """An integer matrix refuses a float in-place update and stays intact."""
    integer_matrix = _decorate_rejection_state(matrix.astype(np.int64))
    snapshot = _observable_source_snapshot(integer_matrix)

    with pytest.raises(TypeError):
        integer_matrix.__imul__(1.5)

    _assert_source_unchanged(integer_matrix, snapshot)


@pytest.mark.parametrize("mutation", ["equal_copy", "nested_mutation"])
def test_rejected_inplace_snapshot_catches_equal_replacement_and_nested_mutation(
    matrix, monkeypatch: pytest.MonkeyPatch, mutation
):
    """The rejection assertion detects topology-preserving and value mutations."""
    matrix = _decorate_rejection_state(matrix)
    snapshot = _observable_source_snapshot(matrix)

    def mutate_then_raise(operand):
        if mutation == "equal_copy":
            matrix.provenance = deepcopy(matrix.provenance)
        else:
            matrix.provenance["nested"][0]["nested"][0][0] = -1
        raise UnitConversionError()

    monkeypatch.setattr(matrix, "__iadd__", mutate_then_raise)
    with pytest.raises(UnitConversionError):
        matrix.__iadd__(1 * u.s)

    with pytest.raises(AssertionError):
        _assert_source_unchanged(matrix, snapshot)


# ---------------------------------------------------------------------------
# 5. Unit conversion for add / sub / comparison (issue #576)
# ---------------------------------------------------------------------------


def test_addition_converts_equivalent_units(factory):
    """``V + mV`` rescales the right operand instead of rejecting it."""
    shape = (2, 2, 2, 2) if factory is make_spectrogram_matrix else (2, 2, 4)
    volts = factory(unit=u.V, values=np.ones(shape))
    millivolts = factory(unit=u.mV, values=np.ones(shape))

    result = volts + millivolts
    for row in cell_units(result):
        for unit in row:
            assert unit == u.V
    np.testing.assert_allclose(np.asarray(result), 1.001)


def test_addition_left_operand_fixes_the_unit(factory):
    """``mV + V`` comes out in millivolts, mirroring astropy's own rule."""
    shape = (2, 2, 2, 2) if factory is make_spectrogram_matrix else (2, 2, 4)
    volts = factory(unit=u.V, values=np.ones(shape))
    millivolts = factory(unit=u.mV, values=np.ones(shape))

    result = millivolts + volts
    for row in cell_units(result):
        for unit in row:
            assert unit == u.mV
    np.testing.assert_allclose(np.asarray(result), 1001.0)


def test_addition_incompatible_units_raise(matrix, factory):
    """Dimensionally wrong units are still a hard error."""
    seconds = factory(unit=u.s)
    with pytest.raises(UnitConversionError):
        matrix + seconds
    with pytest.raises(UnitConversionError):
        matrix + 1 * u.s


def test_comparison_converts_units(factory):
    """Comparisons convert before comparing, so ``1 V > 1 mV`` is True."""
    shape = (2, 2, 2, 2) if factory is make_spectrogram_matrix else (2, 2, 4)
    volts = factory(unit=u.V, values=np.ones(shape))
    millivolts = factory(unit=u.mV, values=np.ones(shape))

    assert np.all(np.asarray(volts > millivolts))
    assert not np.any(np.asarray(volts < millivolts))


def test_comparison_incompatible_units_raise(matrix, factory):
    """A comparison against an incompatible unit raises rather than guessing."""
    with pytest.raises(UnitConversionError):
        matrix > factory(unit=u.s)


# ---------------------------------------------------------------------------
# 6. Bare units are multiplicative only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        lambda m, unit: m + unit,
        lambda m, unit: m - unit,
        lambda m, unit: m > unit,
        lambda m, unit: m**unit,
    ],
    ids=["add", "sub", "gt", "pow"],
)
def test_bare_unit_rejected_outside_multiplication(matrix, op):
    """A bare ``Unit`` carries no values, so only ``*`` and ``/`` accept it."""
    with pytest.raises(TypeError):
        op(matrix, u.s)


def test_bare_unit_accepted_by_multiplication_and_division(matrix):
    """``matrix * u.s`` and ``matrix / u.s`` compose the units."""
    assert (matrix * u.s).meta[0, 0].unit == u.V * u.s
    assert (matrix / u.s).meta[0, 0].unit == u.V / u.s
    assert (u.s * matrix).meta[0, 0].unit == u.V * u.s


# ---------------------------------------------------------------------------
# 7. Metadata independence (issues #577a / #577b)
# ---------------------------------------------------------------------------


def test_result_metadata_cells_are_independent(matrix):
    """Mutating one cell of a result's metadata must not touch its neighbours.

    The uniform-unit fast path used to fill every cell with the *same*
    ``MetaData`` instance via ``np.full`` (issue #577a).
    """
    result = matrix * 2
    assert len({id(cell) for cell in result.meta.reshape(-1)}) == result.meta.size

    result.meta[0, 0].unit = u.Hz
    assert result.meta[0, 1].unit == u.V
    assert result.meta[1, 0].unit == u.V
    assert result.meta[1, 1].unit == u.V
    # ... and the source is untouched as well.
    assert matrix.meta[0, 0].unit == u.V


def test_uniform_units_preserve_per_cell_names(matrix):
    """Uniform units must not collapse every cell onto ``meta[0, 0]``'s name.

    All four cells share unit V here, which is exactly the case the old fast
    path mishandled by adopting cell (0, 0)'s name and channel everywhere
    (issue #577b).
    """
    result = matrix * 2
    assert cell_names(result) == [["a00", "a01"], ["a10", "a11"]]
    assert [str(c) for c in result.meta.channels.reshape(-1)] == [
        "X1:A00",
        "X1:A01",
        "X1:A10",
        "X1:A11",
    ]


def test_non_uniform_units_preserve_per_cell_names(factory):
    """The non-uniform path keeps per-cell names too, so the two paths agree."""
    matrix = factory()
    matrix.meta[0, 1].unit = u.A
    result = matrix * 2
    assert cell_names(result) == [["a00", "a01"], ["a10", "a11"]]
    assert result.meta[0, 0].unit == u.V
    assert result.meta[0, 1].unit == u.A


def test_result_row_and_column_metadata_are_deep_copied(matrix):
    """Row/column metadata is copied, not shared, with the operand."""
    result = matrix * 2
    row_key = next(iter(matrix.rows))
    assert result.rows[row_key] is not matrix.rows[row_key]
    result.rows[row_key].name = "mutated"
    assert matrix.rows[row_key].name != "mutated"


def test_comparison_result_contract(matrix):
    """Comparisons return the same class, bool values, dimensionless units."""
    result = matrix > matrix
    assert type(result) is type(matrix)
    assert np.asarray(result).dtype == np.bool_
    assert result.shape == matrix.shape
    for row in cell_units(result):
        for unit in row:
            assert unit == u.dimensionless_unscaled
    assert cell_names(result) == cell_names(matrix)
    assert result.meta[0, 0] is not matrix.meta[0, 0]
    result.meta[0, 0].name = "mutated"
    assert matrix.meta[0, 0].name == "a00"


# ---------------------------------------------------------------------------
# 8. Power (issue #577e)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda m: m.clip(2.0 * u.V, 5.0 * u.V), lambda a: np.clip(a, 2.0, 5.0)),
        (lambda m: np.clip(m, 2.0 * u.V, 5.0 * u.V), lambda a: np.clip(a, 2.0, 5.0)),
        (lambda m: m.round(), lambda a: np.round(a)),
        (lambda m: np.round(m), lambda a: np.round(a)),
    ],
    ids=["clip_method", "clip_function", "round_method", "round_function"],
)
def test_clip_and_round_keep_units_and_independent_metadata(matrix, call, expected):
    """``clip``/``round`` are ufunc-backed methods kept alive with explicit overrides.

    Without them NumPy's ``_wrapfunc`` fallback would return an object whose
    ``meta`` is literally the source's, so mutating the result would corrupt
    the operand.
    """
    result = call(matrix)
    assert type(result) is type(matrix)
    np.testing.assert_allclose(np.asarray(result), expected(np.asarray(matrix)))
    assert cell_units(result) == cell_units(matrix)
    assert result.meta is not matrix.meta
    assert result.meta[0, 0] is not matrix.meta[0, 0]


def test_clip_and_round_survive_copy_slice_and_pickle_roundtrip(matrix):
    """``clip``/``round`` results keep their axis metadata through further transforms.

    ``clip``/``round`` rebuild via ``self.copy()`` (see ``_rebuild_with_values``),
    so a defect in ``copy()`` -- such as ``SpectrogramMatrix`` silently
    dropping ``frequencies`` -- surfaces here first.
    """
    for result in (matrix.clip(2.0 * u.V, 5.0 * u.V), matrix.round()):
        assert type(result) is type(matrix)

        copied = result.copy()
        assert type(copied) is type(result)
        np.testing.assert_allclose(np.asarray(copied), np.asarray(result))
        assert cell_units(copied) == cell_units(result)
        if isinstance(result, SpectrogramMatrix):
            np.testing.assert_allclose(
                copied.frequencies.to_value(u.Hz), result.frequencies.to_value(u.Hz)
            )
            assert copied.f0 == result.f0
            assert copied.df == result.df

        sliced = result[0]
        assert sliced is not None

        roundtripped = pickle.loads(pickle.dumps(result))
        np.testing.assert_allclose(np.asarray(roundtripped), np.asarray(result))
        assert cell_units(roundtripped) == cell_units(result)
        if isinstance(result, SpectrogramMatrix):
            np.testing.assert_allclose(
                roundtripped.frequencies.to_value(u.Hz),
                result.frequencies.to_value(u.Hz),
            )
            assert roundtripped.f0 == result.f0
            assert roundtripped.df == result.df


def test_spectrogram_clip_preserves_frequency_axis():
    """``SpectrogramMatrix.clip()`` must not drop ``frequencies``/``f0``/``df``.

    ``clip``/``round`` rebuild via ``self.copy()``; the base ``copy()`` only
    resupplies row/col/xindex metadata, so a ``SpectrogramMatrix`` without its
    own ``copy()`` override silently lost its frequency axis here.
    """
    matrix = make_spectrogram_matrix(unit=u.V)
    result = matrix.clip(2.0 * u.V, 5.0 * u.V)
    assert result.frequencies is not None
    np.testing.assert_allclose(
        result.frequencies.to_value(u.Hz), matrix.frequencies.to_value(u.Hz)
    )
    assert result.f0 == matrix.f0
    assert result.df == matrix.df


def test_spectrogram_round_preserves_frequency_axis():
    """``SpectrogramMatrix.round()`` must not drop ``frequencies``/``f0``/``df``."""
    matrix = make_spectrogram_matrix(unit=u.V)
    result = matrix.round()
    assert result.frequencies is not None
    np.testing.assert_allclose(
        result.frequencies.to_value(u.Hz), matrix.frequencies.to_value(u.Hz)
    )
    assert result.f0 == matrix.f0
    assert result.df == matrix.df


def test_clip_accepts_equivalent_quantity_bounds(matrix):
    """A ``Quantity`` bound in an equivalent (but different) unit is converted."""
    result = matrix.clip(200.0 * u.mV, 500.0 * u.mV)
    np.testing.assert_allclose(
        np.asarray(result), np.clip(np.asarray(matrix), 0.2, 0.5)
    )
    assert cell_units(result) == cell_units(matrix)


def test_clip_rejects_dimensionless_bounds_for_unitful_matrix(matrix):
    """A plain number, or a dimensionless ``Quantity``, is refused against unit V.

    Before this fix, a plain-number bound was silently treated as "already in
    the matrix's unit" and a dimensionless ``Quantity`` bound was silently
    accepted as compatible with any unit -- both are issue-#576-class silent
    corruption. All three rejections raise ``UnitConversionError`` rather
    than a bare ``TypeError`` (see test_np_clip_does_not_swallow_the_rejection
    for why).
    """
    with pytest.raises(UnitConversionError):
        matrix.clip(1, 2)
    with pytest.raises(UnitConversionError):
        matrix.clip(6 * u.dimensionless_unscaled, None)
    with pytest.raises(UnitConversionError):
        matrix.clip(1 * u.s, 2 * u.s)


def test_np_clip_does_not_swallow_the_rejection(matrix):
    """``np.clip(matrix, ...)`` must not silently fall back to an unchecked result.

    ``np.clip`` is implemented via NumPy's ``_wrapfunc``, which calls
    ``matrix.clip(...)`` and, if that raises a plain ``TypeError``, silently
    retries through a NumPy-internal path that ignores our unit check
    entirely and returns a result whose metadata *aliases* the source's
    (reintroducing the exact aliasing bug issue #577a fixed). Confirm the
    module-level ``np.clip`` call surfaces the same rejection as the method.
    """
    with pytest.raises(UnitConversionError):
        np.clip(matrix, 1, 2)


def test_clip_out_argument_does_not_bypass_the_unit_check(matrix):
    """``clip(..., out=...)`` must reject rather than silently write ``out``.

    ``np.clip(matrix, ..., out=dest)`` also reaches this class through
    NumPy's ``_wrapfunc``, exactly like the ``out``-less call above. A bare
    ``TypeError`` from the ``out`` guard is swallowed the same way a
    ``TypeError`` from the unit checks would be, and NumPy's fallback then
    writes `dest` with a raw, unit-naive result and aliases the source's
    metadata onto the return value -- reopening the exact hole clip()'s
    unit checks exist to close, just through the ``out`` argument instead
    of the bounds.
    """
    dest = np.empty_like(np.asarray(matrix))
    with pytest.raises(NotImplementedError):
        matrix.clip(1, 2, out=dest)
    with pytest.raises(NotImplementedError):
        np.clip(matrix, 1, 2, out=dest)


def test_round_out_argument_does_not_bypass_the_wrapfunc_swallow(matrix):
    """``round(..., out=...)`` must reject through ``np.round`` too, not just directly."""
    dest = np.empty_like(np.asarray(matrix))
    with pytest.raises(NotImplementedError):
        matrix.round(out=dest)
    with pytest.raises(NotImplementedError):
        np.round(matrix, out=dest)


def test_clip_rejects_equivalent_but_non_identical_per_cell_units():
    """``clip`` refuses a matrix whose cells share a dimension but not a scale.

    ``_all_element_units_equivalent`` only checks dimensional equivalence
    (m and cm are both "length"), so converting a clip *bound* into that
    shared reference unit and applying it to every cell's raw value would
    silently mistreat a cm cell's raw number as if it were already in
    metres -- the same defect class as issue #576, just reached through
    heterogeneous-but-equivalent per-cell units instead of a mismatched
    bound. Until per-cell value conversion is implemented, this must raise
    rather than silently misinterpret the smaller-unit cell's scale.
    """
    meta = MetaDataMatrix(
        np.array(
            [[MetaData(unit=u.m, name="a"), MetaData(unit=u.cm, name="b")]],
            dtype=object,
        )
    )
    matrix = TimeSeriesMatrix(
        np.array([[[1.0, 2.0], [500.0, 600.0]]]), dt=1.0 * u.s, meta=meta
    )
    with pytest.raises(UnitConversionError):
        matrix.clip(max=3 * u.m)


def test_spectrogram_clip_preserves_axes_across_a_dtype_change():
    """``clip()`` must not drop ``frequencies`` or alias ``times`` when it upcasts dtype.

    ``_rebuild_with_values`` (used by ``clip``/``round``) rebuilds via
    ``self.astype(values.dtype)`` instead of ``self.copy()`` whenever the
    operation changes dtype -- e.g. clipping an integer-valued matrix
    against float/``Quantity`` bounds, which NumPy upcasts to float64. Two
    independent defects lived on that path: the base ``astype()`` had no
    ``SpectrogramMatrix``-specific knowledge of ``frequencies`` (the same
    frequencies-blind-spot ``copy()`` had before its own override), and
    separately ``astype()`` passed ``xindex`` through unchanged instead of
    copying it, so the rebuilt result's sample axis aliased the source's --
    mutating one silently corrupted the other.
    """
    values = (np.arange(16).reshape(2, 2, 2, 2) + 1).astype(np.int64)
    matrix = make_spectrogram_matrix(unit=u.V, values=values)
    assert np.asarray(matrix).dtype == np.int64

    result = matrix.clip(2.0 * u.V, 5.0 * u.V)

    assert np.asarray(result).dtype != np.int64
    assert result.frequencies is not None
    np.testing.assert_allclose(
        result.frequencies.to_value(u.Hz), matrix.frequencies.to_value(u.Hz)
    )
    assert result.f0 == matrix.f0
    assert result.df == matrix.df
    assert result.xindex is not matrix.xindex
    assert not np.shares_memory(np.asarray(result.xindex), np.asarray(matrix.xindex))


def test_conjugate_preserves_units(factory):
    """``conj``/``conjugate`` stay available and unit-preserving."""
    shape = (2, 2, 2, 2) if factory is make_spectrogram_matrix else (2, 2, 4)
    matrix = factory(values=np.ones(shape) * (1 + 2j))
    for result in (matrix.conj(), matrix.conjugate()):
        assert type(result) is type(matrix)
        assert cell_units(result) == cell_units(matrix)
        np.testing.assert_allclose(np.asarray(result), np.conjugate(np.asarray(matrix)))


def test_power_squares_the_unit(matrix):
    """``matrix ** 2`` used to raise UnitConversionError while ``np.square`` worked."""
    result = matrix**2
    assert type(result) is type(matrix)
    np.testing.assert_allclose(np.asarray(result), np.asarray(matrix) ** 2)
    for row in cell_units(result):
        for unit in row:
            assert unit == u.V**2


def test_power_matches_square_of_the_raw_values(matrix):
    """The operator and the raw-value ufunc agree numerically."""
    np.testing.assert_allclose(np.asarray(matrix**2), np.square(np.asarray(matrix)))


@pytest.mark.parametrize("exponent", [0.5, -1, np.float64(3), np.array(2.0)])
def test_power_accepts_scalar_exponents(matrix, exponent):
    """Any dimensionless scalar exponent propagates into the unit."""
    result = matrix**exponent
    assert result.meta[0, 0].unit == u.V ** float(np.asarray(exponent))


def test_power_rejects_dimensional_exponent(matrix):
    """``matrix ** (1 * u.s)`` has no meaning and is refused."""
    with pytest.raises(UnitConversionError):
        matrix ** (1 * u.s)


def test_power_rejects_non_scalar_exponent_on_dimensional_base(series_factory):
    """A per-sample exponent cannot be expressed with one unit per cell."""
    matrix = series_factory()
    before = np.asarray(matrix).copy()
    with pytest.raises(UnitConversionError):
        matrix ** np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(np.asarray(matrix), before)
    assert matrix.meta[0, 0].unit == u.V


def test_power_rejects_non_scalar_exponent_on_dimensionless_base(series_factory):
    """A per-sample exponent is unsupported even for a dimensionless base."""
    matrix = series_factory(unit=u.dimensionless_unscaled)
    before = np.asarray(matrix).copy()
    with pytest.raises(UnitConversionError):
        matrix ** np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(np.asarray(matrix), before)
    assert matrix.meta[0, 0].unit == u.dimensionless_unscaled


@pytest.mark.parametrize(
    "factory", list(MATRIX_FACTORIES.values()), ids=list(MATRIX_FACTORIES)
)
@pytest.mark.parametrize(
    "unit", [u.V, u.dimensionless_unscaled], ids=["dimensional", "dimensionless"]
)
def test_power_rejects_every_non_scalar_exponent_before_operand_casting(factory, unit):
    """Lists, tuples, vectors, and matrices all fail atomically at the boundary."""
    prototype = factory(unit=unit)
    exponent_factories = {
        "list": lambda matrix: [1.0] * matrix.shape[-1],
        "tuple": lambda matrix: (1.0,) * matrix.shape[-1],
        "ndarray": lambda matrix: np.ones(matrix.shape[-1]),
        "vector_quantity": lambda matrix: np.ones(matrix.shape[-1]) * u.s,
        "matrix": lambda matrix: matrix.copy(),
    }
    for make_exponent in exponent_factories.values():
        matrix = factory(unit=unit)
        snapshot = _observable_source_snapshot(matrix)
        with pytest.raises(UnitConversionError):
            matrix ** make_exponent(matrix)
        _assert_source_unchanged(matrix, snapshot)

    for base_unit in (u.V, u.dimensionless_unscaled):
        matrix = factory(unit=base_unit)
        snapshot = _observable_source_snapshot(matrix)
        with pytest.raises(UnitConversionError):
            matrix ** (2 * u.s)
        _assert_source_unchanged(matrix, snapshot)

        matrix = factory(unit=base_unit)
        snapshot = _observable_source_snapshot(matrix)
        with pytest.raises(UnitConversionError):
            matrix.__ipow__(2 * u.s)
        _assert_source_unchanged(matrix, snapshot)

        matrix = factory(unit=unit)
        snapshot = _observable_source_snapshot(matrix)
        with pytest.raises(UnitConversionError):
            matrix.__ipow__(make_exponent(matrix))
        _assert_source_unchanged(matrix, snapshot)

    # ``__rpow__`` is deliberately not a B0 composition surface: the matrix
    # would be a per-sample exponent.  Its explicit TypeError is atomically
    # non-mutating, rather than a hidden NumPy composition fallback.
    snapshot = _observable_source_snapshot(prototype)
    with pytest.raises(TypeError):
        2**prototype
    _assert_source_unchanged(prototype, snapshot)


@pytest.mark.parametrize(
    "scalar",
    [
        True,
        False,
        np.bool_(True),
        np.int64(2),
        np.float32(2),
        np.float64(2),
    ],
    ids=[
        "python_true",
        "python_false",
        "numpy_bool",
        "numpy_int",
        "numpy_float32",
        "numpy_float64",
    ],
)
def test_spectrogram_unitless_scalar_categories_match_add_sub_unit_rules(scalar):
    """The advertised NumPy-scalar category includes bool, integer, and float."""
    for operation in (operator.add, operator.sub):
        for reflected in (False, True):
            matrix = make_spectrogram_matrix(unit=u.V)
            snapshot = _observable_source_snapshot(matrix)
            with pytest.raises(UnitConversionError):
                (operation(scalar, matrix) if reflected else operation(matrix, scalar))
            _assert_source_unchanged(matrix, snapshot)

        matrix = make_spectrogram_matrix(unit=u.V)
        snapshot = _observable_source_snapshot(matrix)
        with pytest.raises(UnitConversionError):
            matrix.__iadd__(scalar) if operation is operator.add else matrix.__isub__(
                scalar
            )
        _assert_source_unchanged(matrix, snapshot)

        for reflected in (False, True):
            dimensionless = make_spectrogram_matrix(unit=u.dimensionless_unscaled)
            result = (
                operation(scalar, dimensionless)
                if reflected
                else operation(dimensionless, scalar)
            )
            assert type(result) is SpectrogramMatrix


@pytest.mark.parametrize(
    ("exponent", "expected_power"), [(True, 1), (False, 0)], ids=["true", "false"]
)
def test_bool_power_matches_python_integer_semantics(matrix, exponent, expected_power):
    """``matrix ** True``/``matrix ** False`` behave as ``** 1``/``** 0``.

    Python and NumPy both treat ``True == 1`` and ``False == 0``; a bare
    ``bool`` exponent used to be excluded from the "scalar exponent" fast
    path and fell through to the non-scalar-exponent branch, which rejects a
    non-dimensionless base -- a regression, not the original #577e fix.
    """
    result = matrix**exponent
    np.testing.assert_allclose(np.asarray(result), np.asarray(matrix) ** expected_power)
    for row in cell_units(result):
        for unit in row:
            assert unit == u.V**expected_power


def test_modulo_and_floor_divide_do_not_ignore_units(factory):
    """``%``/``//`` refuse to combine unit-bearing operands without converting.

    NumPy's ``np.mod``/``np.floor_divide`` do not know about units, so
    applying them straight to raw per-cell values silently discards a unit
    mismatch (e.g. ``5000 * u.mm % (2 * u.m)`` naively computing
    ``mod(5000, 2)`` instead of converting to a common unit first) -- the
    same class of defect as issue #576. Full unit-aware floor-division and
    remainder are deferred to the v0.2.0 redesign (issue #637); until then
    this must raise rather than return a silently wrong value.
    """
    unitful = factory(unit=u.m)
    with pytest.raises(TypeError):
        unitful % (2 * u.s)
    with pytest.raises(TypeError):
        unitful // (2 * u.s)
    with pytest.raises(TypeError):
        unitful % 2
    with pytest.raises(TypeError):
        unitful // 2
    with pytest.raises(TypeError):
        (5000 * u.mm) % unitful
    with pytest.raises(TypeError):
        (5000 * u.mm) // unitful

    dimensionless = factory(unit=u.dimensionless_unscaled)
    np.testing.assert_allclose(
        np.asarray(dimensionless % 3), np.asarray(dimensionless) % 3
    )
    np.testing.assert_allclose(
        np.asarray(dimensionless // 3), np.asarray(dimensionless) // 3
    )


def test_modulo_rejects_scaled_dimensionless_units_regardless_of_cell_order():
    """A ``%``-unit (e.g. percent) cell is not safely interchangeable with plain dimensionless.

    ``u.percent`` is *dimensionally* equivalent to ``u.dimensionless_unscaled``
    (both are "dimensionless") but not numerically interchangeable with it
    (``1 == 100%``). The dimensionless fast path used to reuse the general
    equivalence check, which only compares against the first cell's unit, so
    whether a matrix mixing dimensionless and percent cells was treated as
    "safe to mod" depended on which cell happened to be first -- and when it
    was, the percent cell's raw number (e.g. ``500`` for ``500%``) was
    modulo'd directly instead of being interpreted as ``5.0``, silently
    wrong by the unit's scale factor. Both cell orderings must now raise.
    """
    for units in [
        (u.dimensionless_unscaled, u.percent),
        (u.percent, u.dimensionless_unscaled),
    ]:
        meta = MetaDataMatrix(
            np.array(
                [
                    [
                        MetaData(unit=units[0], name="a"),
                        MetaData(unit=units[1], name="b"),
                    ]
                ],
                dtype=object,
            )
        )
        matrix = TimeSeriesMatrix(
            np.array([[[5.0, 6.0], [500.0, 600.0]]]), dt=1.0 * u.s, meta=meta
        )
        with pytest.raises(TypeError):
            matrix % 2
        with pytest.raises(TypeError):
            matrix // 2


# ---------------------------------------------------------------------------
# 9. Refusals
# ---------------------------------------------------------------------------


DIRECT_UFUNCS = {
    "sqrt": lambda m: np.sqrt(m),
    "square": lambda m: np.square(m),
    "absolute": lambda m: np.absolute(m),
    "negative": lambda m: np.negative(m),
    "exp": lambda m: np.exp(m),
    "log": lambda m: np.log(m),
    "sign": lambda m: np.sign(m),
    "isfinite": lambda m: np.isfinite(m),
    "logical_and": lambda m: np.logical_and(m, m),
    "multiply": lambda m: np.multiply(m, m),
    "add_reduce": lambda m: np.add.reduce(m),
    "multiply_accumulate": lambda m: np.multiply.accumulate(m),
}


@pytest.mark.parametrize("name", list(DIRECT_UFUNCS), ids=list(DIRECT_UFUNCS))
def test_direct_ufunc_application_raises(matrix, name):
    """Applying a ufunc straight to a matrix is refused, not silently degraded."""
    with pytest.raises(TypeError):
        DIRECT_UFUNCS[name](matrix)


def test_out_argument_is_rejected_rather_than_discarded(matrix, factory):
    """``out=`` used to be dropped without a word (issue #577c)."""
    destination = factory()
    with pytest.raises(TypeError):
        np.multiply(matrix, 2, out=destination)
    with pytest.raises(TypeError, match="out"):
        matrix._ufunc_dispatch(np.multiply, "__call__", matrix, 2, out=destination)
    with pytest.raises(TypeError, match="where"):
        matrix._ufunc_dispatch(np.multiply, "__call__", matrix, 2, where=False)


def test_non_call_ufunc_methods_are_rejected(matrix):
    """``reduce``/``accumulate`` drop metadata, so they are refused (issue #577d)."""
    with pytest.raises(TypeError):
        matrix._ufunc_dispatch(np.add, "reduce", matrix)


UNSUPPORTED_OPERATORS = {
    "imatmul": lambda m: m.__imatmul__(m),
    "rpow": lambda m: 2**m,
    "and": lambda m: m & m,
    "or": lambda m: m | m,
    "xor": lambda m: m ^ m,
    "invert": lambda m: ~m,
    "lshift": lambda m: m << 1,
    "rshift": lambda m: m >> 1,
    "rand": lambda m: np.ones(m.shape, dtype=bool) & m,
    "three_arg_pow": lambda m: pow(m, 2, 3),
}


@pytest.mark.parametrize(
    "name", list(UNSUPPORTED_OPERATORS), ids=list(UNSUPPORTED_OPERATORS)
)
def test_unsupported_operators_raise_type_error(matrix, name):
    """Operators without a unit-aware meaning fail loudly."""
    with pytest.raises(TypeError):
        UNSUPPORTED_OPERATORS[name](matrix)


# ---------------------------------------------------------------------------
# 10. Zero division
# ---------------------------------------------------------------------------


# floordiv/mod/divmod are exercised separately below: the default `matrix`
# fixture carries unit V, and those three now reject a unit-bearing operand
# (a plain `0` included) before ever reaching the zero check.
ZERO_DIVISION_CASES = {
    "truediv": lambda m, z: m / z,
}


@pytest.mark.parametrize(
    "name", list(ZERO_DIVISION_CASES), ids=list(ZERO_DIVISION_CASES)
)
def test_zero_divisor_raises_and_leaves_operands_intact(matrix, name):
    """A zero divisor is an error, not an ``inf`` sprinkled through the data."""
    before = np.asarray(matrix).copy()
    with pytest.raises(ZeroDivisionError):
        ZERO_DIVISION_CASES[name](matrix, 0)
    np.testing.assert_array_equal(np.asarray(matrix), before)


DIMENSIONLESS_ZERO_DIVISION_CASES = {
    "floordiv": lambda m, z: m // z,
    "mod": lambda m, z: m % z,
}


@pytest.mark.parametrize(
    "name",
    list(DIMENSIONLESS_ZERO_DIVISION_CASES),
    ids=list(DIMENSIONLESS_ZERO_DIVISION_CASES),
)
def test_zero_divisor_raises_for_dimensionless_floordiv_and_mod(factory, name):
    """``//``/``%`` by zero raise even in the dimensionless case they support."""
    matrix = factory(unit=u.dimensionless_unscaled)
    before = np.asarray(matrix).copy()
    with pytest.raises(ZeroDivisionError):
        DIMENSIONLESS_ZERO_DIVISION_CASES[name](matrix, 0)
    np.testing.assert_array_equal(np.asarray(matrix), before)


def test_divmod_by_zero_raises_type_error_not_zero_division_error(matrix):
    """``divmod()`` is unsupported outright, so a zero divisor still just ``TypeError``s."""
    with pytest.raises(TypeError):
        divmod(matrix, 0)


def test_reflected_zero_division_raises(factory):
    """The reflected form is guarded too: dividing by a matrix holding zeros."""
    shape = (2, 2, 2, 2) if factory is make_spectrogram_matrix else (2, 2, 4)
    zeros = factory(values=np.zeros(shape))
    before = np.asarray(zeros).copy()
    with pytest.raises(ZeroDivisionError):
        1.0 / zeros
    np.testing.assert_array_equal(np.asarray(zeros), before)


# ---------------------------------------------------------------------------
# 11. Reductions
# ---------------------------------------------------------------------------


REDUCTIONS = ["sum", "prod", "cumsum", "any", "all", "cumprod"]


@pytest.mark.parametrize("name", REDUCTIONS)
def test_reductions_return_plain_numpy_without_warning(matrix, name):
    """Reductions keep working and stay silent, but drop metadata by design.

    ``ndarray.sum`` goes through ``np.add.reduce``, which the ufunc opt-out
    rejects, so each of these needs an explicit override.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = getattr(matrix, name)()

    assert not isinstance(result, SeriesMatrix)
    expected = getattr(np.asarray(matrix), name)()
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


@pytest.mark.parametrize("name", REDUCTIONS)
def test_module_level_reductions_agree(matrix, name):
    """``np.sum(matrix)`` routes through the same override."""
    np.testing.assert_allclose(
        np.asarray(getattr(np, name)(matrix)),
        np.asarray(getattr(np.asarray(matrix), name)()),
    )


# ---------------------------------------------------------------------------
# 12. astropy's own behaviour for Quantity-left equality
# ---------------------------------------------------------------------------


def test_quantity_left_ordering_comparisons_reach_the_matrix(matrix):
    """``Quantity < matrix`` returns ``NotImplemented`` and reflects to us."""
    quantity = np.ones(matrix.shape) * u.V
    result = quantity < matrix
    assert type(result) is type(matrix)
    assert np.asarray(result).dtype == np.bool_


def test_quantity_left_equality_stays_with_astropy(matrix):
    """``Quantity == matrix`` never reaches the matrix -- pinned as observed.

    ``Quantity.__eq__`` only absorbs ``UnitsError`` from ``_to_own_unit``; it
    never returns ``NotImplemented``, so Python never calls the matrix's
    reflected operator.  With an incompatible unit it short-circuits to a
    scalar ``False``; when the conversion succeeds it falls through to
    ``self.value.__eq__`` and yields a bare ``ndarray``.  Whether the
    conversion succeeds depends on whether the matrix class exposes a single
    ``.unit`` attribute: ``SpectrogramMatrix`` does, the 3-D classes do not,
    so they look dimensionless to astropy.

    This is astropy's behaviour, not a gwexpy contract.  The assertions exist
    so that a future astropy release changing it is detected instead of
    silently altering results.
    """
    incompatible = np.ones(matrix.shape) * u.s
    assert (incompatible == matrix) is False
    assert (incompatible != matrix) is True

    compatible = np.ones(matrix.shape) * (
        u.V if isinstance(matrix, SpectrogramMatrix) else u.dimensionless_unscaled
    )
    equal_result = compatible == matrix
    assert type(equal_result) is np.ndarray
    assert equal_result.dtype == np.bool_
    assert equal_result.shape == matrix.shape

    not_equal_result = compatible != matrix
    assert type(not_equal_result) is np.ndarray
    assert not_equal_result.shape == matrix.shape


def test_matrix_left_equality_is_ours(matrix, factory):
    """``matrix == x`` always goes through the matrix's own operator."""
    result = matrix == factory()
    assert type(result) is type(matrix)
    assert np.all(np.asarray(result))


def test_equality_with_unrelated_object_is_false(matrix):
    """Comparing against something uninterpretable falls back to ``False``."""
    assert (matrix == "not a matrix") is False
    assert (matrix != "not a matrix") is True
    assert (matrix == None) is False  # noqa: E711


# ---------------------------------------------------------------------------
# NumPy integer/float scalar operands (np.int64 does not subclass Python int,
# so it needs its own acceptance path wherever plain int/float is accepted).
# ---------------------------------------------------------------------------


NUMPY_SCALAR_CASES = {
    "np_int64_mul": (lambda m: m * np.int64(2), 2.0),
    "np_int64_pow": (lambda m: m ** np.int64(2), None),  # unit handled separately
    "np_uint32_mul": (lambda m: m * np.uint32(3), 3.0),
    "np_int64_rmul": (lambda m: np.int64(2) * m, 2.0),
}


@pytest.mark.parametrize(
    ("op", "factor"),
    list(NUMPY_SCALAR_CASES.values()),
    ids=list(NUMPY_SCALAR_CASES),
)
def test_operators_accept_bare_numpy_integer_scalars(matrix, op, factor):
    """``matrix * np.int64(n)`` etc. must not raise (previously a regression:
    the scalar-acceptance checks in both ``SeriesMatrix``'s and
    ``SpectrogramMatrix``'s own ufunc dispatch listed ``int``/``float`` but
    not ``np.number``, so a bare NumPy integer scalar was rejected with
    ``TypeError: operand ... does not support ufuncs``).
    """
    result = op(matrix)
    assert type(result) is type(matrix)
    if factor is not None:
        np.testing.assert_allclose(result.value, matrix.value * factor)
        assert cell_units(result) == cell_units(matrix)


def test_power_with_numpy_integer_exponent_matches_python_int(matrix):
    """``matrix ** np.int64(2)`` must equal ``matrix ** 2`` in value and unit."""
    via_numpy_int = matrix ** np.int64(2)
    via_python_int = matrix**2
    np.testing.assert_allclose(via_numpy_int.value, via_python_int.value)
    assert cell_units(via_numpy_int) == cell_units(via_python_int)


def test_power_with_numpy_integer_exponent_does_not_warn(matrix):
    """``matrix ** np.int64(n)`` must not fall back to the per-cell metadata
    loop (previously a regression: ``_scalar_power_exponent`` passed a bare
    ``np.int64`` through to ``MetaDataMatrix.__array_ufunc__``, whose
    ``_to_array`` helper only accepts ``int``/``float``/``complex`` -- not
    ``np.number`` -- so the vectorized path always raised internally, was
    caught, logged a full traceback, and re-raised as a ``PerformanceWarning``
    on every single call).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = matrix ** np.int64(2)
    assert type(result) is type(matrix)
