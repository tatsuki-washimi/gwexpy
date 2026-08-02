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

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.units import UnitConversionError

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.spectrogram import SpectrogramMatrix
from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix
from gwexpy.types.seriesmatrix import SeriesMatrix

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
        rows=["r0", "r1"],
        cols=["c0", "c1"],
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


@pytest.mark.parametrize("op", list(BINARY_OPERATORS), ids=list(BINARY_OPERATORS))
def test_binary_operators_with_matrix_operand(matrix, factory, op):
    """Every arithmetic operator accepts another matrix and stays in the family."""
    other = factory()
    result = BINARY_OPERATORS[op](matrix, other)
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


def test_divmod_matches_floordiv_and_mod(matrix):
    """``divmod`` is exactly the pair of ``//`` and ``%``."""
    quotient, remainder = divmod(matrix, 3)
    assert type(quotient) is type(matrix)
    assert type(remainder) is type(matrix)
    np.testing.assert_allclose(np.asarray(quotient), np.asarray(matrix // 3))
    np.testing.assert_allclose(np.asarray(remainder), np.asarray(matrix % 3))


def test_rdivmod_matches_reflected_floordiv_and_mod(matrix):
    """The reflected ``divmod`` agrees with the reflected operators."""
    other = np.full(matrix.shape, 30.0)
    quotient, remainder = divmod(other, matrix)
    np.testing.assert_allclose(np.asarray(quotient), np.asarray(other // matrix))
    np.testing.assert_allclose(np.asarray(remainder), np.asarray(other % matrix))


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
# a plain number; both must land in the existing buffer.
INPLACE_CASES = {
    "iadd": ("add", 2 * u.V),
    "isub": ("sub", 2 * u.V),
    "imul": ("mul", 2),
    "itruediv": ("truediv", 2),
    "ifloordiv": ("floordiv", 2),
    "imod": ("mod", 2),
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
    """A rejected in-place operation must not have partially written anything."""
    before_values = np.asarray(matrix).copy()
    before_units = cell_units(matrix)

    with pytest.raises(expected_error):
        failing_op(matrix)

    np.testing.assert_array_equal(np.asarray(matrix), before_values)
    assert cell_units(matrix) == before_units


def test_inplace_rejects_unsafe_dtype_change(matrix):
    """An integer matrix refuses a float in-place update and stays intact."""
    integer_matrix = matrix.astype(np.int64)
    before = np.asarray(integer_matrix).copy()

    with pytest.raises(TypeError):
        integer_matrix.__imul__(1.5)

    np.testing.assert_array_equal(np.asarray(integer_matrix), before)
    assert np.asarray(integer_matrix).dtype == np.int64


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
        (lambda m: m.clip(2.0, 5.0), lambda a: np.clip(a, 2.0, 5.0)),
        (lambda m: np.clip(m, 2.0, 5.0), lambda a: np.clip(a, 2.0, 5.0)),
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
    with pytest.raises(UnitConversionError):
        matrix ** np.array([1.0, 2.0, 3.0, 4.0])


def test_power_allows_non_scalar_exponent_on_dimensionless_base(series_factory):
    """A dimensionless base survives an array exponent."""
    matrix = series_factory(unit=u.dimensionless_unscaled)
    result = matrix ** np.array([1.0, 2.0, 3.0, 4.0])
    assert result.meta[0, 0].unit == u.dimensionless_unscaled
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(matrix) ** np.array([1.0, 2.0, 3.0, 4.0])
    )


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


ZERO_DIVISION_CASES = {
    "truediv": lambda m, z: m / z,
    "floordiv": lambda m, z: m // z,
    "mod": lambda m, z: m % z,
    "divmod": lambda m, z: divmod(m, z),
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
