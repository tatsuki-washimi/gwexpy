"""Typed canonical B0 contract for the SeriesMatrix family.

The manifest is the only compatibility ledger used by the adapter.  Every
successful cell carries explicit expectations for result class, units,
metadata, axes, values, and mutation.  Exception cells carry the same typed
fields with ``NOT_APPLICABLE`` values so that no expectation is represented by
free text or inferred from an operation tree in the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
from astropy.units import UnitConversionError


class MatrixFamily(str, Enum):
    TIME_SERIES = "TimeSeriesMatrix"
    FREQUENCY_SERIES = "FrequencySeriesMatrix"
    SPECTROGRAM = "SpectrogramMatrix"


class Phase(str, Enum):
    B0 = "B0"


class Surface(str, Enum):
    STRUCTURE = "structure"
    ARITHMETIC = "arithmetic"
    REFLECTED = "reflected"
    INPLACE = "inplace"
    COMPARISON = "comparison"
    PREDICATE = "predicate"
    REFUSAL = "refusal"


class Side(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    DIRECT = "direct"


class Mutation(str, Enum):
    PURE = "pure"
    INPLACE = "inplace"


class InputScenario(str, Enum):
    DEFAULT = "default"
    DIMENSIONLESS_MATRIX = "dimensionless_matrix"
    DIMENSIONAL_INCOMPATIBLE = "dimensional_incompatible"


class Operand(str, Enum):
    NONE = "not_applicable"
    PYTHON_SCALAR = "python_scalar"
    NUMPY_SCALAR = "numpy_scalar"
    NDARRAY = "ndarray"
    QUANTITY = "quantity"
    UNIT = "unit"
    SAME_CLASS_MATRIX = "same_class_matrix"
    PYTHON_LIST = "python_list"
    PYTHON_TUPLE = "python_tuple"
    VECTOR_QUANTITY = "vector_quantity"


class ResultExpectation(str, Enum):
    EXCEPTION = "exception"
    NONE = "none"
    TUPLE = "tuple"
    NUMPY_DTYPE = "numpy.dtype"
    NUMPY_ARRAY = "numpy.ndarray"
    VALUES_ARRAY = "numpy.ndarray-compatible values"
    MATRIX = "same concrete matrix class"
    BOOL_MATRIX = "same concrete matrix class with numpy.bool_ values"
    ITERATION = "iterator of concrete matrix row views"


class UnitExpectation(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    PRESERVE_CELL_UNITS = "preserve each source cell unit"
    EXACT_DIMENSIONLESS = "exactly dimensionless_unscaled"
    EXACT_V = "exactly V"
    EXACT_V_SQUARED = "exactly V**2"
    EXACT_V_TIMES_S = "exactly V*s"
    EXACT_V_PER_S = "exactly V/s"
    EXACT_S_PER_V = "exactly s/V"
    EXACT_INV_V = "exactly 1/V"


class MetadataExpectation(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    RAW_ARRAY = "raw ndarray has no metadata"
    PRESERVE_SOURCE_CELLS = "preserve source cell identity and labels"
    PRESERVE_SOURCE_CELLS_SHARED_ROWS_COLUMNS = (
        "preserve source cells and share row/column metadata"
    )
    PRESERVE_SOURCE_CELLS_DEEP_ROWS_COLUMNS = (
        "preserve source cells and deep-copy row/column metadata"
    )
    DEEP_COPY_CELLS_SHARED_ROWS_COLUMNS = (
        "deep-copy cells and share row/column metadata"
    )
    DEEP_COPY_CELLS_ROWS_COLUMNS = (
        "deep-independent cells, rows, columns, names, and channels"
    )
    INPLACE_PRESERVE = "preserve in-place metadata"
    ITERATION_ROWS = "preserve row element metadata and labels"


class AxisExpectation(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    RAW_ARRAY = "raw ndarray has no axes"
    PRESERVE_SAMPLE_AXIS = "preserve exact family sample axis"
    SLICE_SAMPLE_AXIS = "preserve exact sliced family sample axis"
    TRANSPOSE_ROW_COLUMN_SAMPLE = "swap row and column labels; preserve sample axis"
    PRESERVE_SPECTROGRAM_AXES = "preserve exact time and frequency axes"
    SPECTROGRAM_TIME_ONLY = "preserve time axis and expose no frequency axis"


class ValueExpectation(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    SHAPE_EXACT = "exact input shape"
    DTYPE_EXACT = "exact input dtype"
    VALUES_EXACT = "exact input values"
    SLICE_EXACT = "exact sliced values"
    ASSIGNMENT_ZERO = "assignment writes exact zeros"
    ITERATION_ROWS_EXACT = "iteration yields exact row values"
    COPY_EXACT = "copy preserves exact values"
    ASTYPE_EXACT = "astype preserves values with exact requested dtype"
    REAL_EXACT = "real preserves exact real values"
    IMAG_EXACT = "imag preserves exact imaginary values"
    CONJ_EXACT = "conj preserves exact conjugate values"
    TRANSPOSE_EXACT = "transpose preserves exact transposed values"
    RESHAPE_EXACT = "reshape preserves exact reshaped values"
    RAW_VIEW_EXACT = "raw ndarray preserves exact values and aliases storage"
    ADD_EXACT = "exact addition result"
    SUB_EXACT = "exact subtraction result"
    MUL_EXACT = "exact multiplication result"
    DIV_EXACT = "exact division result"
    POWER_EXACT = "exact power result"
    COMPARISON_EXACT = "exact boolean comparison result"
    INPLACE_ADD_EXACT = "exact addition result committed in place"
    INPLACE_SUB_EXACT = "exact subtraction result committed in place"
    INPLACE_MUL_EXACT = "exact multiplication result committed in place"
    INPLACE_DIV_EXACT = "exact division result committed in place"
    INPLACE_POWER_EXACT = "exact power result committed in place"


class MutationExpectation(str, Enum):
    PURE_NO_MUTATION = "pure operation leaves source unchanged"
    ASSIGNMENT_MUTATES = "assignment mutates source values"
    RAW_VIEW_ALIASES = "raw ndarray mutation aliases source values"
    INPLACE_COMMIT = "operation mutates existing matrix atomically"
    INPLACE_FAILURE_UNCHANGED = "failed in-place operation leaves source unchanged"


class NameExpectation(str, Enum):
    PRESERVE = "preserve matrix name"
    REAL_SUFFIX = "append .real to matrix name"
    IMAG_SUFFIX = "append .imag to matrix name"
    TRANSPOSE_SUFFIX = "append .T to matrix name"
    ROW_ELEMENT_EMPTY = "row element has an empty matrix name"


class EpochExpectation(str, Enum):
    PRESERVE = "preserve epoch"


class AttrsExpectation(str, Enum):
    DEEP_COPY = "deep-copy attrs"
    SHARED = "share attrs"
    PRESERVE = "preserve attrs in place"
    EMPTY = "result has empty attrs"


@dataclass(frozen=True)
class ContractCell:
    """One fully typed, reviewable B0 compatibility assertion."""

    id: str
    family: MatrixFamily
    phase: Phase
    surface: Surface
    operation: str
    operand: Operand
    side: Side
    mutation: Mutation
    scenario: InputScenario
    expected_result: ResultExpectation
    unit_expectation: UnitExpectation
    metadata_expectation: MetadataExpectation
    axis_expectation: AxisExpectation
    value_expectation: ValueExpectation
    mutation_expectation: MutationExpectation
    name_expectation: NameExpectation = NameExpectation.PRESERVE
    epoch_expectation: EpochExpectation = EpochExpectation.PRESERVE
    attrs_expectation: AttrsExpectation = AttrsExpectation.DEEP_COPY
    exception_class: type[BaseException] | None = None


_FAMILIES: Final[tuple[MatrixFamily, ...]] = tuple(MatrixFamily)
_BINARY_OPERANDS: Final[tuple[Operand, ...]] = tuple(
    operand
    for operand in Operand
    if operand
    not in {
        Operand.NONE,
        Operand.PYTHON_LIST,
        Operand.PYTHON_TUPLE,
        Operand.VECTOR_QUANTITY,
    }
)


def _id(
    family: MatrixFamily,
    surface: Surface,
    operation: str,
    operand: Operand,
    side: Side,
    mutation: Mutation,
    scenario: InputScenario,
    suffix: str,
) -> str:
    return ":".join(
        (
            family.value,
            surface.value,
            operation,
            operand.value,
            side.value,
            mutation.value,
            scenario.value,
            suffix,
        )
    )


def _result(
    family: MatrixFamily,
    operation: str,
    operand: Operand,
    *,
    surface: Surface,
    expected_result: ResultExpectation,
    unit_expectation: UnitExpectation,
    metadata_expectation: MetadataExpectation | None = None,
    axis_expectation: AxisExpectation,
    value_expectation: ValueExpectation,
    mutation_expectation: MutationExpectation = MutationExpectation.PURE_NO_MUTATION,
    name_expectation: NameExpectation = NameExpectation.PRESERVE,
    epoch_expectation: EpochExpectation = EpochExpectation.PRESERVE,
    attrs_expectation: AttrsExpectation | None = None,
    side: Side = Side.NONE,
    mutation: Mutation = Mutation.PURE,
    scenario: InputScenario = InputScenario.DEFAULT,
) -> ContractCell:
    if attrs_expectation is None:
        attrs_expectation = AttrsExpectation.DEEP_COPY
    if metadata_expectation is None:
        metadata_expectation = MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS
    return ContractCell(
        id=_id(
            family,
            surface,
            operation,
            operand,
            side,
            mutation,
            scenario,
            "result",
        ),
        family=family,
        phase=Phase.B0,
        surface=surface,
        operation=operation,
        operand=operand,
        side=side,
        mutation=mutation,
        scenario=scenario,
        expected_result=expected_result,
        unit_expectation=unit_expectation,
        metadata_expectation=metadata_expectation,
        axis_expectation=axis_expectation,
        value_expectation=value_expectation,
        mutation_expectation=mutation_expectation,
        name_expectation=name_expectation,
        epoch_expectation=epoch_expectation,
        attrs_expectation=attrs_expectation,
    )


def _error(
    family: MatrixFamily,
    operation: str,
    operand: Operand,
    exception_class: type[BaseException],
    *,
    surface: Surface,
    side: Side = Side.NONE,
    mutation: Mutation = Mutation.PURE,
    scenario: InputScenario = InputScenario.DEFAULT,
) -> ContractCell:
    return ContractCell(
        id=_id(
            family,
            surface,
            operation,
            operand,
            side,
            mutation,
            scenario,
            f"error:{exception_class.__name__}",
        ),
        family=family,
        phase=Phase.B0,
        surface=surface,
        operation=operation,
        operand=operand,
        side=side,
        mutation=mutation,
        scenario=scenario,
        expected_result=ResultExpectation.EXCEPTION,
        unit_expectation=UnitExpectation.NOT_APPLICABLE,
        metadata_expectation=MetadataExpectation.NOT_APPLICABLE,
        axis_expectation=AxisExpectation.NOT_APPLICABLE,
        value_expectation=ValueExpectation.NOT_APPLICABLE,
        mutation_expectation=(
            MutationExpectation.INPLACE_FAILURE_UNCHANGED
            if mutation is Mutation.INPLACE
            else MutationExpectation.PURE_NO_MUTATION
        ),
        attrs_expectation=AttrsExpectation.PRESERVE,
        exception_class=exception_class,
    )


def _structure_cells(family: MatrixFamily) -> tuple[ContractCell, ...]:
    preserve_axis = (
        AxisExpectation.PRESERVE_SPECTROGRAM_AXES
        if family is MatrixFamily.SPECTROGRAM
        else AxisExpectation.PRESERVE_SAMPLE_AXIS
    )
    cells: list[ContractCell] = [
        _result(
            family,
            "shape",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.TUPLE,
            unit_expectation=UnitExpectation.NOT_APPLICABLE,
            metadata_expectation=MetadataExpectation.NOT_APPLICABLE,
            axis_expectation=AxisExpectation.NOT_APPLICABLE,
            value_expectation=ValueExpectation.SHAPE_EXACT,
        ),
        _result(
            family,
            "dtype",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.NUMPY_DTYPE,
            unit_expectation=UnitExpectation.NOT_APPLICABLE,
            metadata_expectation=MetadataExpectation.NOT_APPLICABLE,
            axis_expectation=AxisExpectation.NOT_APPLICABLE,
            value_expectation=ValueExpectation.DTYPE_EXACT,
        ),
        _result(
            family,
            "values",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.VALUES_ARRAY,
            unit_expectation=UnitExpectation.NOT_APPLICABLE,
            metadata_expectation=MetadataExpectation.NOT_APPLICABLE,
            axis_expectation=AxisExpectation.NOT_APPLICABLE,
            value_expectation=ValueExpectation.VALUES_EXACT,
        ),
        _result(
            family,
            "assignment",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.NONE,
            unit_expectation=UnitExpectation.NOT_APPLICABLE,
            metadata_expectation=MetadataExpectation.INPLACE_PRESERVE,
            axis_expectation=preserve_axis,
            value_expectation=ValueExpectation.ASSIGNMENT_ZERO,
            mutation_expectation=MutationExpectation.ASSIGNMENT_MUTATES,
        ),
        _result(
            family,
            "iteration",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.ITERATION,
            unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
            metadata_expectation=MetadataExpectation.ITERATION_ROWS,
            axis_expectation=preserve_axis,
            value_expectation=ValueExpectation.ITERATION_ROWS_EXACT,
            name_expectation=NameExpectation.ROW_ELEMENT_EMPTY,
            attrs_expectation=AttrsExpectation.DEEP_COPY,
        ),
        _result(
            family,
            "copy",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.MATRIX,
            unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
            axis_expectation=preserve_axis,
            value_expectation=ValueExpectation.COPY_EXACT,
        ),
        _result(
            family,
            "astype",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.MATRIX,
            unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
            axis_expectation=preserve_axis,
            value_expectation=ValueExpectation.ASTYPE_EXACT,
        ),
        _result(
            family,
            "real",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.MATRIX,
            unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
            metadata_expectation=(
                MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS
                if family is MatrixFamily.SPECTROGRAM
                else MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS
            ),
            axis_expectation=preserve_axis,
            value_expectation=ValueExpectation.REAL_EXACT,
            name_expectation=NameExpectation.REAL_SUFFIX,
            attrs_expectation=AttrsExpectation.DEEP_COPY,
        ),
        _result(
            family,
            "imag",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.MATRIX,
            unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
            metadata_expectation=(
                MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS
                if family is MatrixFamily.SPECTROGRAM
                else MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS
            ),
            axis_expectation=preserve_axis,
            value_expectation=ValueExpectation.IMAG_EXACT,
            name_expectation=NameExpectation.IMAG_SUFFIX,
            attrs_expectation=AttrsExpectation.DEEP_COPY,
        ),
        _result(
            family,
            "conj",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.MATRIX,
            unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
            axis_expectation=preserve_axis,
            value_expectation=ValueExpectation.CONJ_EXACT,
        ),
        _result(
            family,
            "np.asarray",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.NUMPY_ARRAY,
            unit_expectation=UnitExpectation.NOT_APPLICABLE,
            metadata_expectation=MetadataExpectation.RAW_ARRAY,
            axis_expectation=AxisExpectation.RAW_ARRAY,
            value_expectation=ValueExpectation.RAW_VIEW_EXACT,
            mutation_expectation=MutationExpectation.RAW_VIEW_ALIASES,
        ),
        _result(
            family,
            "matrix.view(np.ndarray)",
            Operand.NONE,
            surface=Surface.STRUCTURE,
            expected_result=ResultExpectation.NUMPY_ARRAY,
            unit_expectation=UnitExpectation.NOT_APPLICABLE,
            metadata_expectation=MetadataExpectation.RAW_ARRAY,
            axis_expectation=AxisExpectation.RAW_ARRAY,
            value_expectation=ValueExpectation.RAW_VIEW_EXACT,
            mutation_expectation=MutationExpectation.RAW_VIEW_ALIASES,
        ),
    ]
    if family is MatrixFamily.SPECTROGRAM:
        cells.insert(
            3,
            _error(
                family,
                "slicing",
                Operand.NONE,
                ValueError,
                surface=Surface.STRUCTURE,
            ),
        )
        cells.append(
            _error(
                family,
                "transpose",
                Operand.NONE,
                ValueError,
                surface=Surface.STRUCTURE,
            )
        )
        cells.append(
            _error(
                family,
                "reshape",
                Operand.NONE,
                ValueError,
                surface=Surface.STRUCTURE,
            )
        )
    else:
        cells.insert(
            3,
            _result(
                family,
                "slicing",
                Operand.NONE,
                surface=Surface.STRUCTURE,
                expected_result=ResultExpectation.MATRIX,
                unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
                metadata_expectation=MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS,
                axis_expectation=AxisExpectation.SLICE_SAMPLE_AXIS,
                value_expectation=ValueExpectation.SLICE_EXACT,
                attrs_expectation=AttrsExpectation.DEEP_COPY,
            ),
        )
        cells.append(
            _result(
                family,
                "transpose",
                Operand.NONE,
                surface=Surface.STRUCTURE,
                expected_result=ResultExpectation.MATRIX,
                unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
                axis_expectation=AxisExpectation.TRANSPOSE_ROW_COLUMN_SAMPLE,
                value_expectation=ValueExpectation.TRANSPOSE_EXACT,
                name_expectation=NameExpectation.TRANSPOSE_SUFFIX,
            )
        )
        cells.append(
            _result(
                family,
                "reshape",
                Operand.NONE,
                surface=Surface.STRUCTURE,
                expected_result=ResultExpectation.MATRIX,
                unit_expectation=UnitExpectation.PRESERVE_CELL_UNITS,
                axis_expectation=preserve_axis,
                value_expectation=ValueExpectation.RESHAPE_EXACT,
            )
        )
    return tuple(cells)


def _ndarray_add_sub_cell(
    family: MatrixFamily,
    operation: str,
    *,
    side: Side = Side.NONE,
    scenario: InputScenario,
) -> ContractCell:
    surface = Surface.REFLECTED if side is Side.LEFT else Surface.ARITHMETIC
    if scenario is InputScenario.DIMENSIONAL_INCOMPATIBLE:
        return _error(
            family,
            operation,
            Operand.NDARRAY,
            UnitConversionError,
            surface=surface,
            side=side,
            scenario=scenario,
        )
    unit = (
        UnitExpectation.EXACT_DIMENSIONLESS
        if scenario is InputScenario.DIMENSIONLESS_MATRIX
        else UnitExpectation.PRESERVE_CELL_UNITS
    )
    return _result(
        family,
        operation,
        Operand.NDARRAY,
        surface=surface,
        side=side,
        scenario=scenario,
        expected_result=ResultExpectation.MATRIX,
        unit_expectation=unit,
        axis_expectation=(
            AxisExpectation.PRESERVE_SPECTROGRAM_AXES
            if family is MatrixFamily.SPECTROGRAM
            else AxisExpectation.PRESERVE_SAMPLE_AXIS
        ),
        value_expectation=(
            ValueExpectation.ADD_EXACT
            if operation == "add"
            else ValueExpectation.SUB_EXACT
        ),
    )


def _ndarray_add_sub_inplace_cell(
    family: MatrixFamily,
    operation: str,
    *,
    scenario: InputScenario,
) -> ContractCell:
    if scenario is InputScenario.DIMENSIONAL_INCOMPATIBLE:
        return _error(
            family,
            operation,
            Operand.NDARRAY,
            UnitConversionError,
            surface=Surface.INPLACE,
            mutation=Mutation.INPLACE,
            scenario=scenario,
        )
    return _result(
        family,
        operation,
        Operand.NDARRAY,
        surface=Surface.INPLACE,
        mutation=Mutation.INPLACE,
        scenario=scenario,
        expected_result=ResultExpectation.MATRIX,
        unit_expectation=(
            UnitExpectation.EXACT_DIMENSIONLESS
            if scenario is InputScenario.DIMENSIONLESS_MATRIX
            else UnitExpectation.PRESERVE_CELL_UNITS
        ),
        metadata_expectation=MetadataExpectation.INPLACE_PRESERVE,
        axis_expectation=(
            AxisExpectation.PRESERVE_SPECTROGRAM_AXES
            if family is MatrixFamily.SPECTROGRAM
            else AxisExpectation.PRESERVE_SAMPLE_AXIS
        ),
        value_expectation=(
            ValueExpectation.INPLACE_ADD_EXACT
            if operation == "add"
            else ValueExpectation.INPLACE_SUB_EXACT
        ),
        attrs_expectation=AttrsExpectation.PRESERVE,
        mutation_expectation=MutationExpectation.INPLACE_COMMIT,
    )


def _power_error_cell(
    family: MatrixFamily,
    operand: Operand,
    *,
    surface: Surface = Surface.ARITHMETIC,
    mutation: Mutation = Mutation.PURE,
    scenario: InputScenario = InputScenario.DEFAULT,
) -> ContractCell:
    return _error(
        family,
        "power",
        operand,
        UnitConversionError,
        surface=surface,
        mutation=mutation,
        scenario=scenario,
    )


def _scalar_add_sub_cell(
    family: MatrixFamily,
    operation: str,
    operand: Operand,
    *,
    side: Side = Side.NONE,
    scenario: InputScenario,
    mutation: Mutation = Mutation.PURE,
) -> ContractCell:
    surface = (
        Surface.INPLACE
        if mutation is Mutation.INPLACE
        else (Surface.REFLECTED if side is Side.LEFT else Surface.ARITHMETIC)
    )
    if scenario is InputScenario.DIMENSIONAL_INCOMPATIBLE:
        return _error(
            family,
            operation,
            operand,
            UnitConversionError,
            surface=surface,
            side=side,
            mutation=mutation,
            scenario=scenario,
        )
    return _binary_expectation(
        family,
        operation,
        operand,
        surface=surface,
        side=side,
        scenario=scenario,
        mutation=mutation,
    )


def _comparison_cell(
    family: MatrixFamily, operation: str, operand: Operand
) -> ContractCell:
    if family is not MatrixFamily.SPECTROGRAM and operand is Operand.PYTHON_SCALAR:
        return _error(
            family,
            operation,
            operand,
            UnitConversionError,
            surface=Surface.COMPARISON,
        )
    return _result(
        family,
        operation,
        operand,
        surface=Surface.COMPARISON,
        expected_result=ResultExpectation.BOOL_MATRIX,
        unit_expectation=UnitExpectation.EXACT_DIMENSIONLESS,
        axis_expectation=(
            AxisExpectation.PRESERVE_SPECTROGRAM_AXES
            if family is MatrixFamily.SPECTROGRAM
            else AxisExpectation.PRESERVE_SAMPLE_AXIS
        ),
        value_expectation=ValueExpectation.COMPARISON_EXACT,
        scenario=(
            InputScenario.DIMENSIONLESS_MATRIX
            if operand is Operand.NDARRAY
            else InputScenario.DEFAULT
        ),
    )


def _predicate_cell(family: MatrixFamily, operation: str) -> ContractCell:
    if family is MatrixFamily.SPECTROGRAM and operation == "isreal":
        return _result(
            family,
            operation,
            Operand.SAME_CLASS_MATRIX,
            surface=Surface.PREDICATE,
            side=Side.DIRECT,
            expected_result=ResultExpectation.BOOL_MATRIX,
            unit_expectation=UnitExpectation.EXACT_DIMENSIONLESS,
            metadata_expectation=MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS,
            axis_expectation=AxisExpectation.PRESERVE_SPECTROGRAM_AXES,
            value_expectation=ValueExpectation.COMPARISON_EXACT,
            attrs_expectation=AttrsExpectation.DEEP_COPY,
        )
    return _error(
        family,
        operation,
        Operand.SAME_CLASS_MATRIX,
        UnitConversionError if operation == "isreal" else TypeError,
        surface=Surface.PREDICATE,
        side=Side.DIRECT,
    )


def _binary_expectation(
    family: MatrixFamily,
    operation: str,
    operand: Operand,
    *,
    side: Side = Side.NONE,
    surface: Surface,
    scenario: InputScenario = InputScenario.DEFAULT,
    mutation: Mutation = Mutation.PURE,
) -> ContractCell:
    if operation in {"add", "sub"}:
        unit = (
            UnitExpectation.EXACT_DIMENSIONLESS
            if scenario is InputScenario.DIMENSIONLESS_MATRIX
            else UnitExpectation.EXACT_V
        )
    elif operation == "mul":
        if operand in {Operand.QUANTITY, Operand.UNIT}:
            unit = UnitExpectation.EXACT_V_TIMES_S
        elif operand is Operand.SAME_CLASS_MATRIX:
            unit = UnitExpectation.EXACT_V_SQUARED
        else:
            unit = UnitExpectation.EXACT_V
    else:
        if operand in {Operand.QUANTITY, Operand.UNIT}:
            unit = (
                UnitExpectation.EXACT_S_PER_V
                if side is Side.LEFT
                else UnitExpectation.EXACT_V_PER_S
            )
        elif operand is Operand.SAME_CLASS_MATRIX:
            unit = UnitExpectation.EXACT_DIMENSIONLESS
        elif side is Side.LEFT:
            unit = UnitExpectation.EXACT_INV_V
        else:
            unit = UnitExpectation.EXACT_V
    value = {
        "add": ValueExpectation.ADD_EXACT,
        "sub": ValueExpectation.SUB_EXACT,
        "mul": ValueExpectation.MUL_EXACT,
        "truediv": ValueExpectation.DIV_EXACT,
    }[operation]
    axis = (
        AxisExpectation.PRESERVE_SPECTROGRAM_AXES
        if family is MatrixFamily.SPECTROGRAM
        else AxisExpectation.PRESERVE_SAMPLE_AXIS
    )
    return _result(
        family,
        operation,
        operand,
        surface=surface,
        side=side,
        mutation=mutation,
        scenario=scenario,
        expected_result=ResultExpectation.MATRIX,
        unit_expectation=unit,
        metadata_expectation=(
            MetadataExpectation.INPLACE_PRESERVE
            if mutation is Mutation.INPLACE
            else MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS
        ),
        axis_expectation=axis,
        value_expectation=value,
        mutation_expectation=(
            MutationExpectation.INPLACE_COMMIT
            if mutation is Mutation.INPLACE
            else MutationExpectation.PURE_NO_MUTATION
        ),
        attrs_expectation=(
            AttrsExpectation.PRESERVE
            if mutation is Mutation.INPLACE
            else AttrsExpectation.DEEP_COPY
        ),
    )


def _operator_cells(family: MatrixFamily) -> tuple[ContractCell, ...]:
    cells: list[ContractCell] = []
    for operation in ("add", "sub"):
        cells.extend(
            (
                _binary_expectation(
                    family,
                    operation,
                    Operand.QUANTITY,
                    surface=Surface.ARITHMETIC,
                ),
                _binary_expectation(
                    family,
                    operation,
                    Operand.SAME_CLASS_MATRIX,
                    surface=Surface.ARITHMETIC,
                ),
                _ndarray_add_sub_cell(
                    family,
                    operation,
                    scenario=InputScenario.DIMENSIONAL_INCOMPATIBLE,
                ),
                _ndarray_add_sub_cell(
                    family,
                    operation,
                    scenario=InputScenario.DIMENSIONLESS_MATRIX,
                ),
                _binary_expectation(
                    family,
                    operation,
                    Operand.QUANTITY,
                    surface=Surface.REFLECTED,
                    side=Side.LEFT,
                ),
                _ndarray_add_sub_cell(
                    family,
                    operation,
                    side=Side.LEFT,
                    scenario=InputScenario.DIMENSIONAL_INCOMPATIBLE,
                ),
                _ndarray_add_sub_cell(
                    family,
                    operation,
                    side=Side.LEFT,
                    scenario=InputScenario.DIMENSIONLESS_MATRIX,
                ),
            )
        )
        for operand in (Operand.PYTHON_SCALAR, Operand.NUMPY_SCALAR):
            for scenario in (
                InputScenario.DIMENSIONAL_INCOMPATIBLE,
                InputScenario.DIMENSIONLESS_MATRIX,
            ):
                cells.extend(
                    (
                        _scalar_add_sub_cell(
                            family, operation, operand, scenario=scenario
                        ),
                        _scalar_add_sub_cell(
                            family,
                            operation,
                            operand,
                            side=Side.LEFT,
                            scenario=scenario,
                        ),
                    )
                )
    for operation in ("mul", "truediv"):
        for operand in _BINARY_OPERANDS:
            cells.append(
                _binary_expectation(
                    family,
                    operation,
                    operand,
                    surface=Surface.ARITHMETIC,
                )
            )
        for operand in _BINARY_OPERANDS:
            cells.append(
                _binary_expectation(
                    family,
                    operation,
                    operand,
                    surface=Surface.REFLECTED,
                    side=Side.LEFT,
                )
            )
    for operand in (Operand.PYTHON_SCALAR, Operand.NUMPY_SCALAR):
        cells.append(
            _result(
                family,
                "power",
                operand,
                surface=Surface.ARITHMETIC,
                expected_result=ResultExpectation.MATRIX,
                unit_expectation=UnitExpectation.EXACT_V_SQUARED,
                axis_expectation=(
                    AxisExpectation.PRESERVE_SPECTROGRAM_AXES
                    if family is MatrixFamily.SPECTROGRAM
                    else AxisExpectation.PRESERVE_SAMPLE_AXIS
                ),
                value_expectation=ValueExpectation.POWER_EXACT,
            )
        )
    for operand in (
        Operand.NDARRAY,
        Operand.PYTHON_LIST,
        Operand.PYTHON_TUPLE,
        Operand.VECTOR_QUANTITY,
        Operand.SAME_CLASS_MATRIX,
    ):
        for scenario in (
            InputScenario.DEFAULT,
            InputScenario.DIMENSIONLESS_MATRIX,
        ):
            cells.append(_power_error_cell(family, operand, scenario=scenario))
    for scenario in (
        InputScenario.DEFAULT,
        InputScenario.DIMENSIONLESS_MATRIX,
    ):
        cells.append(_power_error_cell(family, Operand.QUANTITY, scenario=scenario))
    cells.append(
        _error(
            family,
            "power",
            Operand.UNIT,
            TypeError,
            surface=Surface.ARITHMETIC,
        )
    )
    for scenario in (
        InputScenario.DEFAULT,
        InputScenario.DIMENSIONLESS_MATRIX,
    ):
        cells.append(
            _error(
                family,
                "power",
                Operand.PYTHON_SCALAR,
                TypeError,
                surface=Surface.REFLECTED,
                side=Side.LEFT,
                scenario=scenario,
            )
        )
    for operation in ("add", "sub"):
        cells.append(
            _binary_expectation(
                family,
                operation,
                Operand.QUANTITY,
                surface=Surface.INPLACE,
                mutation=Mutation.INPLACE,
            )
        )
        cells.append(
            _error(
                family,
                operation,
                Operand.QUANTITY,
                UnitConversionError,
                surface=Surface.INPLACE,
                mutation=Mutation.INPLACE,
                scenario=InputScenario.DIMENSIONAL_INCOMPATIBLE,
            )
        )
        cells.extend(
            (
                _ndarray_add_sub_inplace_cell(
                    family,
                    operation,
                    scenario=InputScenario.DIMENSIONLESS_MATRIX,
                ),
                _ndarray_add_sub_inplace_cell(
                    family,
                    operation,
                    scenario=InputScenario.DIMENSIONAL_INCOMPATIBLE,
                ),
            )
        )
        for operand in (Operand.PYTHON_SCALAR, Operand.NUMPY_SCALAR):
            for scenario in (
                InputScenario.DIMENSIONAL_INCOMPATIBLE,
                InputScenario.DIMENSIONLESS_MATRIX,
            ):
                cells.append(
                    _scalar_add_sub_cell(
                        family,
                        operation,
                        operand,
                        mutation=Mutation.INPLACE,
                        scenario=scenario,
                    )
                )
    for operation in ("mul", "truediv"):
        for operand in (Operand.PYTHON_SCALAR, Operand.UNIT):
            cells.append(
                _binary_expectation(
                    family,
                    operation,
                    operand,
                    surface=Surface.INPLACE,
                    mutation=Mutation.INPLACE,
                )
            )
    cells.append(
        _result(
            family,
            "power",
            Operand.PYTHON_SCALAR,
            surface=Surface.INPLACE,
            mutation=Mutation.INPLACE,
            expected_result=ResultExpectation.MATRIX,
            unit_expectation=UnitExpectation.EXACT_V_SQUARED,
            metadata_expectation=MetadataExpectation.INPLACE_PRESERVE,
            axis_expectation=(
                AxisExpectation.PRESERVE_SPECTROGRAM_AXES
                if family is MatrixFamily.SPECTROGRAM
                else AxisExpectation.PRESERVE_SAMPLE_AXIS
            ),
            value_expectation=ValueExpectation.INPLACE_POWER_EXACT,
            mutation_expectation=MutationExpectation.INPLACE_COMMIT,
            attrs_expectation=AttrsExpectation.PRESERVE,
        )
    )
    for operand in (
        Operand.NDARRAY,
        Operand.PYTHON_LIST,
        Operand.PYTHON_TUPLE,
        Operand.VECTOR_QUANTITY,
        Operand.SAME_CLASS_MATRIX,
    ):
        for scenario in (
            InputScenario.DEFAULT,
            InputScenario.DIMENSIONLESS_MATRIX,
        ):
            cells.append(
                _power_error_cell(
                    family,
                    operand,
                    surface=Surface.INPLACE,
                    mutation=Mutation.INPLACE,
                    scenario=scenario,
                )
            )
    for scenario in (
        InputScenario.DEFAULT,
        InputScenario.DIMENSIONLESS_MATRIX,
    ):
        cells.append(
            _power_error_cell(
                family,
                Operand.QUANTITY,
                surface=Surface.INPLACE,
                mutation=Mutation.INPLACE,
                scenario=scenario,
            )
        )
    for operation in ("lt", "le", "eq", "ne", "gt", "ge"):
        for operand in (
            Operand.PYTHON_SCALAR,
            Operand.NDARRAY,
            Operand.QUANTITY,
            Operand.SAME_CLASS_MATRIX,
        ):
            cells.append(_comparison_cell(family, operation, operand))
    for operation in ("isfinite", "isnan", "isreal"):
        cells.append(_predicate_cell(family, operation))
    for operation in ("sqrt", "log", "exp"):
        cells.append(
            _error(
                family,
                operation,
                Operand.SAME_CLASS_MATRIX,
                TypeError,
                surface=Surface.REFUSAL,
                side=Side.DIRECT,
            )
        )
    cells.append(
        _error(
            family,
            "direct_ufunc",
            Operand.SAME_CLASS_MATRIX,
            TypeError,
            surface=Surface.REFUSAL,
            side=Side.DIRECT,
        )
    )
    cells.append(
        _error(
            family,
            "unsupported_operator",
            Operand.SAME_CLASS_MATRIX,
            TypeError,
            surface=Surface.REFUSAL,
        )
    )
    return tuple(cells)


def _build_manifest() -> tuple[ContractCell, ...]:
    cells: list[ContractCell] = []
    for family in _FAMILIES:
        cells.extend(_structure_cells(family))
        cells.extend(_operator_cells(family))
    return tuple(cells)


B0_CONTRACT: Final[tuple[ContractCell, ...]] = _build_manifest()
EXPECTED_B0_CELL_COUNT: Final[int] = 453
assert len(B0_CONTRACT) == EXPECTED_B0_CELL_COUNT


__all__ = [
    "AxisExpectation",
    "AttrsExpectation",
    "B0_CONTRACT",
    "ContractCell",
    "EXPECTED_B0_CELL_COUNT",
    "EpochExpectation",
    "InputScenario",
    "MatrixFamily",
    "MetadataExpectation",
    "Mutation",
    "MutationExpectation",
    "NameExpectation",
    "Operand",
    "Phase",
    "ResultExpectation",
    "Side",
    "Surface",
    "UnitExpectation",
    "ValueExpectation",
]
