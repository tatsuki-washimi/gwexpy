"""Executable invariants for the typed SeriesMatrix B0 contract manifest."""

from __future__ import annotations

import operator
import sys
from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.units import UnitConversionError

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.spectrogram import SpectrogramMatrix
from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix

from .series_matrix_contract_manifest import (
    B0_CONTRACT,
    EXPECTED_B0_CELL_COUNT,
    AttrsExpectation,
    AxisExpectation,
    ContractCell,
    EpochExpectation,
    InputScenario,
    MatrixFamily,
    MetadataExpectation,
    Mutation,
    MutationExpectation,
    NameExpectation,
    Operand,
    Phase,
    ResultExpectation,
    Side,
    Surface,
    UnitExpectation,
    ValueExpectation,
)


class _SlotMutableSequence(MutableSequence[object]):
    """A mutable sequence without ``__dict__`` for topology regressions."""

    __slots__ = ("_items",)

    def __init__(self, items: list[object]) -> None:
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value) -> None:
        self._items[index] = value

    def __delitem__(self, index) -> None:
        del self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def insert(self, index: int, value: object) -> None:
        self._items.insert(index, value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SlotMutableSequence) and self._items == other._items


class _SlotMutableMapping(MutableMapping[str, object]):
    """A mutable mapping without ``__dict__`` for topology regressions."""

    __slots__ = ("_items",)

    def __init__(self, items: dict[str, object]) -> None:
        self._items = dict(items)

    def __getitem__(self, key: str) -> object:
        return self._items[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._items[key] = value

    def __delitem__(self, key: str) -> None:
        del self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SlotMutableMapping) and self._items == other._items


MUTABLE_CONTAINER_FACTORIES = (
    pytest.param(lambda: {1, 2}, id="set"),
    pytest.param(
        lambda: _SlotMutableSequence(["one", {"nested"}]), id="mutable-sequence"
    ),
    pytest.param(
        lambda: _SlotMutableMapping({"nested": {"value"}}), id="mutable-mapping"
    ),
)


@dataclass(frozen=True)
class ContractObservation:
    cell_id: str
    matches_manifest: bool


def _meta(unit: u.UnitBase) -> MetaDataMatrix:
    metadata = MetaDataMatrix(
        np.array(
            [
                [
                    MetaData(
                        unit=unit,
                        name="a00",
                        channel="X1:A00",
                        calibration={"nested": ["a00", {"gain": 1.0}]},
                    ),
                    MetaData(
                        unit=unit,
                        name="a01",
                        channel="X1:A01",
                        calibration={"nested": ["a01", {"gain": 1.1}]},
                    ),
                ],
                [
                    MetaData(
                        unit=unit,
                        name="a10",
                        channel="X1:A10",
                        calibration={"nested": ["a10", {"gain": 1.2}]},
                    ),
                    MetaData(
                        unit=unit,
                        name="a11",
                        channel="X1:A11",
                        calibration={"nested": ["a11", {"gain": 1.3}]},
                    ),
                ],
            ],
            dtype=object,
        )
    )
    metadata.row_keys = ["meta-row-a", "meta-row-b"]
    metadata.col_keys = ["meta-col-a", "meta-col-b"]
    return metadata


def _matrix(cell: ContractCell):
    unit = (
        u.dimensionless_unscaled
        if cell.scenario is InputScenario.DIMENSIONLESS_MATRIX
        else u.V
    )

    def decorate(matrix):
        matrix.attrs["contract"] = {"nested": ["value", {"calibration": [1, 2, 3]}]}
        matrix.rows[next(iter(matrix.rows))]["calibration"] = {"nested": ["row"]}
        matrix.cols[next(iter(matrix.cols))]["calibration"] = {"nested": ["col"]}
        matrix.provenance = {
            "schema": "contract",
            "nested": [{"array": np.arange(3, dtype=np.int64)}],
        }
        return matrix

    if cell.family is MatrixFamily.TIME_SERIES:
        return decorate(
            TimeSeriesMatrix(
                np.arange(16, dtype=float).reshape(2, 2, 4) + 1,
                meta=_meta(unit),
                dt=0.5 * u.s,
                t0=0.0,
                name="tsm",
            )
        )
    if cell.family is MatrixFamily.FREQUENCY_SERIES:
        return decorate(
            FrequencySeriesMatrix(
                np.arange(16, dtype=float).reshape(2, 2, 4) + 1,
                meta=_meta(unit),
                df=1.0 * u.Hz,
                f0=0.0,
                name="fsm",
            )
        )
    return decorate(
        SpectrogramMatrix(
            np.arange(16, dtype=float).reshape(2, 2, 2, 2) + 1,
            times=np.arange(2) * u.s,
            frequencies=np.arange(2) * u.Hz,
            meta=_meta(unit),
            rows=["r0", "r1"],  # type: ignore[list-item]
            cols=["c0", "c1"],  # type: ignore[list-item]
            name="sgm",
        )
    )


@pytest.mark.parametrize("family", list(MatrixFamily))
def test_contract_fixture_retains_arbitrary_metadata_and_explicit_keys(
    family: MatrixFamily,
) -> None:
    cell = next(cell for cell in B0_CONTRACT if cell.family is family)
    matrix = _matrix(cell)
    assert matrix.meta.row_keys == ["meta-row-a", "meta-row-b"]
    assert matrix.meta.col_keys == ["meta-col-a", "meta-col-b"]
    assert matrix.meta[0, 0]["calibration"] == {"nested": ["a00", {"gain": 1.0}]}
    assert matrix.rows[next(iter(matrix.rows))]["calibration"] == {"nested": ["row"]}
    assert matrix.cols[next(iter(matrix.cols))]["calibration"] == {"nested": ["col"]}


def _operand(cell: ContractCell, matrix):
    if cell.operand.value == "python_scalar":
        return 2
    if cell.operand.value == "numpy_scalar":
        return np.float64(2)
    if cell.operand.value == "ndarray":
        return np.full(matrix.shape, 2.0)
    if cell.operand.value == "quantity":
        if cell.operation in {"mul", "truediv"}:
            return 2 * u.s
        if cell.scenario is InputScenario.DIMENSIONAL_INCOMPATIBLE:
            return 2 * u.s
        return 2 * u.V
    if cell.operand.value == "unit":
        return u.s
    if cell.operand.value == "same_class_matrix":
        return matrix.copy()
    if cell.operand is Operand.PYTHON_LIST:
        return [1.0] * matrix.shape[-1]
    if cell.operand is Operand.PYTHON_TUPLE:
        return (1.0,) * matrix.shape[-1]
    if cell.operand is Operand.VECTOR_QUANTITY:
        return np.ones(matrix.shape[-1]) * u.s
    raise AssertionError(f"no operand adapter for {cell.operand}")


def _invoke(cell: ContractCell, matrix):
    if cell.surface is Surface.STRUCTURE:
        operations = {
            "shape": lambda: matrix.shape,
            "dtype": lambda: matrix.dtype,
            "values": lambda: np.asarray(matrix),
            "slicing": lambda: matrix[..., :1],
            "assignment": lambda: _assign(matrix),
            "iteration": lambda: list(matrix),
            "copy": matrix.copy,
            "astype": lambda: matrix.astype(np.float32),
            "real": lambda: matrix.real,
            "imag": lambda: matrix.imag,
            "conj": matrix.conj,
            "transpose": matrix.transpose,
            "reshape": lambda: matrix.reshape(matrix.shape),
            "np.asarray": lambda: np.asarray(matrix),
            "matrix.view(np.ndarray)": lambda: matrix.view(np.ndarray),
        }
        return operations[cell.operation]()
    if cell.surface in {Surface.ARITHMETIC, Surface.REFLECTED}:
        operand = _operand(cell, matrix)
        if cell.operation == "power" and cell.side is Side.LEFT:
            return operator.pow(operand, matrix)
        operation = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "truediv": operator.truediv,
            "power": operator.pow,
        }[cell.operation]
        return (
            operation(operand, matrix)
            if cell.side is Side.LEFT
            else operation(matrix, operand)
        )
    if cell.surface is Surface.INPLACE:
        operand = _operand(cell, matrix)
        inplace_operation = {
            "add": "iadd",
            "sub": "isub",
            "mul": "imul",
            "truediv": "itruediv",
            "power": "ipow",
        }[cell.operation]
        return getattr(matrix, f"__{inplace_operation}__")(operand)
    if cell.surface is Surface.COMPARISON:
        return getattr(operator, cell.operation)(matrix, _operand(cell, matrix))
    if cell.operation in {"isfinite", "isnan", "isreal", "sqrt", "log", "exp"}:
        return getattr(np, cell.operation)(matrix)
    if cell.operation == "direct_ufunc":
        return np.add(matrix, matrix)
    if cell.operation == "unsupported_operator":
        return matrix & 1
    raise AssertionError(f"no operation adapter for {cell.id}")


def _assign(matrix) -> None:
    matrix[...] = np.zeros_like(np.asarray(matrix))


def _units(matrix) -> tuple[u.UnitBase, ...]:
    return tuple(metadata.unit for metadata in matrix.meta.reshape(-1))


def _labels(matrix) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(metadata.name), str(metadata.channel))
        for metadata in matrix.meta.reshape(-1)
    )


def _metadata_payload(metadata: MetaData) -> dict:
    """Return all custom metadata fields while leaving unit assertions separate."""
    payload = deepcopy(dict(metadata))
    payload.pop("unit")
    return payload


def _assert_deep_metadata_payload(source, result) -> None:
    """Assert public metadata content and nested independence for a new result."""
    transposed = result.name == f"{source.name}.T"
    assert result.meta.row_keys == (
        source.meta.col_keys if transposed else source.meta.row_keys
    )
    assert result.meta.col_keys == (
        source.meta.row_keys if transposed else source.meta.col_keys
    )
    for index in np.ndindex(result.meta.shape):
        source_index = index[::-1] if transposed else index
        _assert_deep_equal(
            _metadata_payload(source.meta[source_index]),
            _metadata_payload(result.meta[index]),
        )
        assert (
            result.meta[index]["calibration"]
            is not source.meta[source_index]["calibration"]
        )
    result.meta[(0, 0)]["calibration"]["nested"][1]["gain"] = "result-only"
    assert (
        source.meta[(0, 0) if not transposed else (0, 0)]["calibration"]["nested"][1][
            "gain"
        ]
        != "result-only"
    )
    source_rows = source.cols if transposed else source.rows
    source_cols = source.rows if transposed else source.cols
    assert list(result.rows) == list(source_rows)
    assert list(result.cols) == list(source_cols)
    for key in source_rows:
        _assert_deep_equal(dict(source_rows[key]), dict(result.rows[key]))
        assert result.rows[key] is not source_rows[key]
    for key in source_cols:
        _assert_deep_equal(dict(source_cols[key]), dict(result.cols[key]))
        assert result.cols[key] is not source_cols[key]
    first_row_key = next(iter(result.rows))
    result.rows[first_row_key]["calibration"]["nested"].append("result-only")
    assert "result-only" not in source_rows[first_row_key]["calibration"]["nested"]


def _slot_names(value) -> tuple[str, ...]:
    names: list[str] = []
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(name for name in slots if name not in {"__dict__", "__weakref__"})
    return tuple(names)


def _observable_source_state(matrix):
    """Return project-owned public state, excluding third-party lazy caches."""
    state = {}
    for name in (
        "unit",
        "meta",
        "rows",
        "cols",
        "xindex",
        "times",
        "frequencies",
        "name",
        "channel",
        "epoch",
        "attrs",
        "provenance",
    ):
        try:
            state[name] = getattr(matrix, name)
        except AttributeError:
            pass
    return state


def _observable_aliases(matrix):
    """Return identities for every mutable object in observable source state."""
    aliases = {"matrix": id(matrix)}
    seen: set[int] = set()

    def visit(value, path: str) -> None:
        value_id = id(value)
        if isinstance(value, np.ndarray):
            aliases[path] = value_id
            if value_id in seen:
                return
            seen.add(value_id)
            if value.dtype == object:
                for array_index in np.ndindex(value.shape):
                    visit(value[array_index], f"{path}[{array_index!r}]")
            return
        if isinstance(value, Mapping):
            aliases[path] = value_id
            if value_id in seen:
                return
            seen.add(value_id)
            for key, item in value.items():
                visit(item, f"{path}[{key!r}]")
            return
        if isinstance(value, MutableSequence):
            aliases[path] = value_id
            if value_id in seen:
                return
            seen.add(value_id)
            for sequence_index, item in enumerate(value):
                visit(item, f"{path}[{sequence_index}]")
            return
        if isinstance(value, MutableSet):
            aliases[path] = value_id
            if value_id in seen:
                return
            seen.add(value_id)
            for set_index, item in enumerate(value):
                visit(item, f"{path}[{set_index}]")
            return
        if isinstance(value, tuple):
            if value_id in seen:
                return
            seen.add(value_id)
            for tuple_index, item in enumerate(value):
                visit(item, f"{path}[{tuple_index}]")
            return

    for name, value in _observable_source_state(matrix).items():
        visit(value, name)
    return aliases


def _observable_source_snapshot(matrix):
    """Capture the complete observable source state for pure operations."""
    return deepcopy(_observable_source_state(matrix)), _observable_aliases(matrix)


def _assert_deep_equal(expected, actual, _seen=None) -> None:
    """Compare nested observable state without allowing ndarray shortcuts."""
    if _seen is None:
        _seen = set()
    pair = (id(expected), id(actual))
    if pair in _seen:
        return
    _seen.add(pair)
    if isinstance(expected, u.Quantity):
        assert isinstance(actual, u.Quantity)
        assert actual.unit == expected.unit
        np.testing.assert_array_equal(actual.value, expected.value)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        if expected.dtype == object:
            for index in np.ndindex(expected.shape):
                _assert_deep_equal(expected[index], actual[index], _seen)
        else:
            np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, Mapping):
        assert type(actual) is type(expected)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_deep_equal(expected[key], actual[key], _seen)
        return
    if isinstance(expected, MutableSet):
        assert type(actual) is type(expected)
        assert actual == expected
        return
    if isinstance(expected, MutableSequence):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for expected_item, actual_item in zip(expected, actual):
            _assert_deep_equal(expected_item, actual_item, _seen)
        return
    if isinstance(expected, tuple):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for expected_item, actual_item in zip(expected, actual):
            _assert_deep_equal(expected_item, actual_item, _seen)
        return
    assert actual == expected


def _assert_source_unchanged(matrix, snapshot) -> None:
    before_state, before_aliases = snapshot
    _assert_deep_equal(before_state, _observable_source_state(matrix))
    assert _observable_aliases(matrix) == before_aliases


def test_pure_alias_snapshot_accepts_unchanged_nested_provenance() -> None:
    cell = next(cell for cell in B0_CONTRACT if cell.operation == "shape")
    matrix = _matrix(cell)
    snapshot = _observable_source_snapshot(matrix)

    _assert_source_unchanged(matrix, snapshot)


def test_public_snapshot_ignores_third_party_lazy_caches() -> None:
    """Astropy/GWpy implementation caches are outside the B0 mutation boundary."""
    cell = next(cell for cell in B0_CONTRACT if cell.operation == "shape")
    matrix = _matrix(cell)
    channel = matrix.meta[0, 0].channel
    channel._b0_lazy_cache = {"ready": False}
    snapshot = _observable_source_snapshot(matrix)

    channel._b0_lazy_cache["ready"] = True

    _assert_source_unchanged(matrix, snapshot)


@pytest.mark.parametrize("field", ["provenance", "attrs", "array"])
def test_pure_alias_snapshot_rejects_equal_nested_replacements(field: str) -> None:
    cell = next(cell for cell in B0_CONTRACT if cell.operation == "shape")
    matrix = _matrix(cell)
    snapshot = _observable_source_snapshot(matrix)

    if field == "provenance":
        matrix.provenance = deepcopy(matrix.provenance)
    elif field == "attrs":
        matrix.attrs["contract"]["nested"] = deepcopy(
            matrix.attrs["contract"]["nested"]
        )
    else:
        matrix.provenance["nested"][0]["array"] = np.array(
            matrix.provenance["nested"][0]["array"], copy=True
        )

    with pytest.raises(AssertionError):
        _assert_source_unchanged(matrix, snapshot)


@pytest.mark.parametrize("container_factory", MUTABLE_CONTAINER_FACTORIES)
def test_pure_alias_snapshot_rejects_equal_nested_mutable_container_replacements(
    container_factory,
) -> None:
    cell = next(cell for cell in B0_CONTRACT if cell.operation == "shape")
    matrix = _matrix(cell)
    matrix.provenance["nested"][0]["container"] = container_factory()
    snapshot = _observable_source_snapshot(matrix)
    matrix.provenance["nested"][0]["container"] = deepcopy(
        matrix.provenance["nested"][0]["container"]
    )

    with pytest.raises(AssertionError):
        _assert_source_unchanged(matrix, snapshot)


@pytest.mark.parametrize("container_factory", MUTABLE_CONTAINER_FACTORIES)
def test_pure_alias_snapshot_accepts_unchanged_nested_mutable_containers(
    container_factory,
) -> None:
    cell = next(cell for cell in B0_CONTRACT if cell.operation == "shape")
    matrix = _matrix(cell)
    matrix.provenance["nested"][0]["container"] = container_factory()
    snapshot = _observable_source_snapshot(matrix)

    _assert_source_unchanged(matrix, snapshot)


def test_exception_path_rejects_equal_nested_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(cell for cell in B0_CONTRACT if cell.exception_class is not None)

    def mutate_then_raise(cell, matrix):
        matrix.provenance["nested"][0] = deepcopy(matrix.provenance["nested"][0])
        raise cell.exception_class()

    monkeypatch.setattr(sys.modules[__name__], "_invoke", mutate_then_raise)
    with pytest.raises(AssertionError):
        execute_contract_cell(cell)


@pytest.mark.parametrize("container_factory", MUTABLE_CONTAINER_FACTORIES)
def test_exception_path_rejects_equal_nested_mutable_container_replacement(
    monkeypatch: pytest.MonkeyPatch, container_factory
) -> None:
    cell = next(cell for cell in B0_CONTRACT if cell.exception_class is not None)
    original_container = container_factory()
    original_matrix = _matrix

    def mutate_then_raise(cell, matrix):
        matrix.provenance["nested"][0]["container"] = deepcopy(original_container)
        raise cell.exception_class()

    def prepare_matrix(cell):
        matrix = original_matrix(cell)
        matrix.provenance["nested"][0]["container"] = original_container
        return matrix

    monkeypatch.setattr(sys.modules[__name__], "_matrix", prepare_matrix)
    monkeypatch.setattr(sys.modules[__name__], "_invoke", mutate_then_raise)
    with pytest.raises(AssertionError):
        execute_contract_cell(cell)


def _numeric_operand(
    cell: ContractCell, matrix, before: np.ndarray
) -> np.ndarray | float:
    if cell.operand.value == "same_class_matrix":
        return before
    if cell.operand.value == "ndarray":
        return np.full(matrix.shape, 2.0)
    if cell.operand.value in {"quantity", "unit"}:
        return 2.0 if cell.operand.value == "quantity" else 1.0
    return 2.0


def _expected_values(
    cell: ContractCell, matrix, before: np.ndarray
) -> np.ndarray | None:
    operand = _numeric_operand(cell, matrix, before)
    expected = cell.value_expectation
    if expected in {
        ValueExpectation.SHAPE_EXACT,
        ValueExpectation.DTYPE_EXACT,
        ValueExpectation.VALUES_EXACT,
        ValueExpectation.COPY_EXACT,
        ValueExpectation.RAW_VIEW_EXACT,
    }:
        return before
    if expected is ValueExpectation.SLICE_EXACT:
        return before[..., :1]
    if expected is ValueExpectation.ASSIGNMENT_ZERO:
        return np.zeros_like(before)
    if expected is ValueExpectation.ASTYPE_EXACT:
        return before.astype(np.float32)
    if expected is ValueExpectation.REAL_EXACT:
        return np.real(before)
    if expected is ValueExpectation.IMAG_EXACT:
        return np.imag(before)
    if expected is ValueExpectation.CONJ_EXACT:
        return np.conj(before)
    if expected is ValueExpectation.TRANSPOSE_EXACT:
        return before.transpose(1, 0, 2)
    if expected is ValueExpectation.RESHAPE_EXACT:
        return before.reshape(matrix.shape)
    if expected in {
        ValueExpectation.ITERATION_ROWS_EXACT,
        ValueExpectation.COMPARISON_EXACT,
    }:
        return None
    if expected in {ValueExpectation.ADD_EXACT, ValueExpectation.INPLACE_ADD_EXACT}:
        return before + operand
    if expected in {ValueExpectation.SUB_EXACT, ValueExpectation.INPLACE_SUB_EXACT}:
        return operand - before if cell.side is Side.LEFT else before - operand
    if expected in {ValueExpectation.MUL_EXACT, ValueExpectation.INPLACE_MUL_EXACT}:
        return before * operand
    if expected in {ValueExpectation.DIV_EXACT, ValueExpectation.INPLACE_DIV_EXACT}:
        return operand / before if cell.side is Side.LEFT else before / operand
    if expected in {ValueExpectation.POWER_EXACT, ValueExpectation.INPLACE_POWER_EXACT}:
        return operand**before if cell.side is Side.LEFT else before**operand
    raise AssertionError(f"no value expectation adapter for {cell.id}")


def _assert_units(
    cell: ContractCell,
    matrix,
    result,
    before_units: tuple[u.UnitBase, ...],
) -> None:
    expectation = cell.unit_expectation
    if expectation is UnitExpectation.NOT_APPLICABLE:
        return
    exact = {
        UnitExpectation.EXACT_DIMENSIONLESS: u.dimensionless_unscaled,
        UnitExpectation.EXACT_V: u.V,
        UnitExpectation.EXACT_V_SQUARED: u.V**2,
        UnitExpectation.EXACT_V_TIMES_S: u.V * u.s,
        UnitExpectation.EXACT_V_PER_S: u.V / u.s,
        UnitExpectation.EXACT_S_PER_V: u.s / u.V,
        UnitExpectation.EXACT_INV_V: 1 / u.V,
    }
    if expectation is UnitExpectation.PRESERVE_CELL_UNITS:
        expected = before_units
    else:
        expected = (exact[expectation],) * len(before_units)
    observed = _units(matrix if cell.mutation is Mutation.INPLACE else result)
    assert observed == expected


def _assert_metadata(cell: ContractCell, matrix, result) -> None:
    expectation = cell.metadata_expectation
    if expectation in {
        MetadataExpectation.NOT_APPLICABLE,
        MetadataExpectation.RAW_ARRAY,
    }:
        return
    if expectation is MetadataExpectation.ITERATION_ROWS:
        for index, row in enumerate(result):
            assert type(row) is type(matrix)
            assert _labels(row) == _labels(matrix)[index * 2 : (index + 1) * 2]
        return
    if expectation is MetadataExpectation.INPLACE_PRESERVE:
        assert result is matrix
        assert result.meta is matrix.meta
        return
    if expectation in {
        MetadataExpectation.PRESERVE_SOURCE_CELLS,
        MetadataExpectation.PRESERVE_SOURCE_CELLS_SHARED_ROWS_COLUMNS,
        MetadataExpectation.PRESERVE_SOURCE_CELLS_DEEP_ROWS_COLUMNS,
    }:
        assert result.meta is not matrix.meta
        assert all(
            result.meta[index] is matrix.meta[index]
            for index in np.ndindex(result.meta.shape)
        )
        assert _labels(result) == _labels(matrix)
        for key in matrix.rows:
            if (
                expectation
                is MetadataExpectation.PRESERVE_SOURCE_CELLS_SHARED_ROWS_COLUMNS
            ):
                assert result.rows[key] is matrix.rows[key]
            elif key in result.rows:
                assert result.rows[key] is not matrix.rows[key]
        for key in matrix.cols:
            if (
                expectation
                is MetadataExpectation.PRESERVE_SOURCE_CELLS_SHARED_ROWS_COLUMNS
            ):
                assert result.cols[key] is matrix.cols[key]
            elif key in result.cols:
                assert result.cols[key] is not matrix.cols[key]
        return
    if expectation is MetadataExpectation.DEEP_COPY_CELLS_SHARED_ROWS_COLUMNS:
        assert result.meta is not matrix.meta
        assert all(
            result.meta[index] is not matrix.meta[index]
            for index in np.ndindex(result.meta.shape)
        )
        assert _labels(result) == _labels(matrix)
        for key in matrix.rows:
            assert result.rows[key] is matrix.rows[key]
        for key in matrix.cols:
            assert result.cols[key] is matrix.cols[key]
        return
    assert expectation is MetadataExpectation.DEEP_COPY_CELLS_ROWS_COLUMNS
    assert result.meta is not matrix.meta
    assert all(
        result.meta[index] is not matrix.meta[index]
        for index in np.ndindex(result.meta.shape)
    )
    if cell.operation == "transpose":
        expected_labels = tuple(
            (str(matrix.meta[j, i].name), str(matrix.meta[j, i].channel))
            for i, j in np.ndindex(result.meta.shape)
        )
    else:
        expected_labels = _labels(matrix)
    assert _labels(result) == expected_labels
    assert (
        list(result.rows.keys()) == list(matrix.rows.keys())
        or cell.operation == "transpose"
    )
    assert (
        list(result.cols.keys()) == list(matrix.cols.keys())
        or cell.operation == "transpose"
    )
    for key in matrix.rows:
        if key in result.rows:
            assert result.rows[key] is not matrix.rows[key]
    for key in matrix.cols:
        if key in result.cols:
            assert result.cols[key] is not matrix.cols[key]
    _assert_deep_metadata_payload(matrix, result)


def _assert_object_metadata(cell: ContractCell, matrix, result) -> None:
    assert cell.epoch_expectation is EpochExpectation.PRESERVE
    assert result.epoch == matrix.epoch
    name = {
        NameExpectation.PRESERVE: matrix.name,
        NameExpectation.REAL_SUFFIX: f"{matrix.name}.real",
        NameExpectation.IMAG_SUFFIX: f"{matrix.name}.imag",
        NameExpectation.TRANSPOSE_SUFFIX: f"{matrix.name}.T",
    }[cell.name_expectation]
    assert result.name == name
    if cell.attrs_expectation is AttrsExpectation.DEEP_COPY:
        assert result.attrs is not matrix.attrs
        _assert_deep_equal(matrix.attrs, result.attrs)
        if "contract" in matrix.attrs:
            assert result.attrs["contract"] is not matrix.attrs["contract"]
            assert (
                result.attrs["contract"]["nested"][1]
                is not matrix.attrs["contract"]["nested"][1]
            )
            result.attrs["contract"]["nested"][1]["calibration"].append("result-only")
            assert (
                "result-only"
                not in matrix.attrs["contract"]["nested"][1]["calibration"]
            )
    elif cell.attrs_expectation is AttrsExpectation.SHARED:
        assert result.attrs is matrix.attrs
    elif cell.attrs_expectation is AttrsExpectation.PRESERVE:
        assert result.attrs is matrix.attrs
    elif cell.attrs_expectation is AttrsExpectation.EMPTY:
        assert result.attrs == {}
        assert result.attrs is not matrix.attrs
    else:  # pragma: no cover - every enum value is intentionally handled
        raise AssertionError(cell.attrs_expectation)


def _assert_axes(cell: ContractCell, matrix, result) -> None:
    expectation = cell.axis_expectation
    if expectation in {AxisExpectation.NOT_APPLICABLE, AxisExpectation.RAW_ARRAY}:
        return
    target = matrix if cell.mutation is Mutation.INPLACE else result
    if expectation is AxisExpectation.PRESERVE_SAMPLE_AXIS:
        np.testing.assert_array_equal(
            target.xindex.to_value(target.xindex.unit),
            matrix.xindex.to_value(matrix.xindex.unit),
        )
    elif expectation is AxisExpectation.SLICE_SAMPLE_AXIS:
        np.testing.assert_array_equal(
            target.xindex.to_value(target.xindex.unit),
            matrix.xindex.to_value(matrix.xindex.unit)[:1],
        )
    elif expectation is AxisExpectation.TRANSPOSE_ROW_COLUMN_SAMPLE:
        assert list(target.rows.keys()) == list(matrix.cols.keys())
        assert list(target.cols.keys()) == list(matrix.rows.keys())
        np.testing.assert_array_equal(
            target.xindex.to_value(target.xindex.unit),
            matrix.xindex.to_value(matrix.xindex.unit),
        )
    elif expectation is AxisExpectation.PRESERVE_SPECTROGRAM_AXES:
        np.testing.assert_array_equal(
            target.times.to_value(u.s), matrix.times.to_value(u.s)
        )
        np.testing.assert_array_equal(
            target.frequencies.to_value(u.Hz), matrix.frequencies.to_value(u.Hz)
        )
    elif expectation is AxisExpectation.SPECTROGRAM_TIME_ONLY:
        np.testing.assert_array_equal(
            target.times.to_value(u.s), matrix.times.to_value(u.s)
        )
        assert target.frequencies is None
    else:  # pragma: no cover - every enum value is intentionally handled
        raise AssertionError(expectation)


def _assert_values(cell: ContractCell, matrix, result, before: np.ndarray) -> None:
    expected = _expected_values(cell, matrix, before)
    if cell.value_expectation is ValueExpectation.SHAPE_EXACT:
        assert result == matrix.shape
    elif cell.value_expectation is ValueExpectation.DTYPE_EXACT:
        assert result == before.dtype
    elif cell.value_expectation is ValueExpectation.ITERATION_ROWS_EXACT:
        for index, row in enumerate(result):
            expected_row = (
                before[index : index + 1] if before.ndim == 3 else before[index]
            )
            np.testing.assert_array_equal(np.asarray(row), expected_row)
    elif cell.value_expectation is ValueExpectation.COMPARISON_EXACT:
        if cell.operation == "isreal":
            expected_bool = np.isreal(before)
        else:
            operand = _numeric_operand(cell, matrix, before)
            expected_bool = getattr(operator, cell.operation)(before, operand)
        np.testing.assert_array_equal(np.asarray(result), expected_bool)
    elif cell.value_expectation is ValueExpectation.ASSIGNMENT_ZERO:
        np.testing.assert_array_equal(np.asarray(matrix), np.zeros_like(before))
    elif expected is not None:
        np.testing.assert_array_equal(np.asarray(result), expected)


def _assert_iteration_contract(
    cell: ContractCell,
    matrix,
    rows,
    before_units: tuple[u.UnitBase, ...],
) -> None:
    assert cell.unit_expectation is UnitExpectation.PRESERVE_CELL_UNITS
    assert cell.axis_expectation in {
        AxisExpectation.PRESERVE_SAMPLE_AXIS,
        AxisExpectation.PRESERVE_SPECTROGRAM_AXES,
    }
    for index, row in enumerate(rows):
        assert cell.name_expectation is NameExpectation.ROW_ELEMENT_EMPTY
        assert row.name == ""
        assert row.epoch == matrix.epoch
        if cell.attrs_expectation is AttrsExpectation.DEEP_COPY:
            assert row.attrs == matrix.attrs
            assert row.attrs is not matrix.attrs
            assert row.attrs["contract"] is not matrix.attrs["contract"]
            row.attrs["contract"]["nested"][1]["calibration"].append("row-only")
            assert (
                "row-only" not in matrix.attrs["contract"]["nested"][1]["calibration"]
            )
        else:
            assert cell.attrs_expectation is AttrsExpectation.EMPTY
            assert row.attrs == {}
            assert row.attrs is not matrix.attrs
        width = row.meta.size
        expected_units = before_units[index * width : (index + 1) * width]
        assert _units(row) == expected_units
        if isinstance(matrix, SpectrogramMatrix):
            np.testing.assert_array_equal(
                row.times.to_value(u.s), matrix.times.to_value(u.s)
            )
            np.testing.assert_array_equal(
                row.frequencies.to_value(u.Hz), matrix.frequencies.to_value(u.Hz)
            )
        else:
            np.testing.assert_array_equal(
                row.xindex.to_value(row.xindex.unit),
                matrix.xindex.to_value(matrix.xindex.unit),
            )


def _assert_observed_result(
    cell: ContractCell,
    matrix,
    result,
    before: np.ndarray,
    before_units: tuple[u.UnitBase, ...],
    source_snapshot,
) -> None:
    expected = cell.expected_result
    if expected is ResultExpectation.NONE:
        assert result is None
        _assert_values(cell, matrix, result, before)
        return
    if expected is ResultExpectation.TUPLE:
        assert type(result) is tuple
    elif expected is ResultExpectation.NUMPY_DTYPE:
        assert isinstance(result, np.dtype)
    elif expected in {ResultExpectation.NUMPY_ARRAY, ResultExpectation.VALUES_ARRAY}:
        assert type(result) is np.ndarray
    elif expected is ResultExpectation.ITERATION:
        assert type(result) is list
        _assert_iteration_contract(cell, matrix, result, before_units)
    elif expected in {ResultExpectation.MATRIX, ResultExpectation.BOOL_MATRIX}:
        assert type(result) is type(matrix)
        if expected is ResultExpectation.BOOL_MATRIX:
            assert result.dtype == np.dtype(np.bool_)
    else:  # pragma: no cover - EXCEPTION is handled before this function
        raise AssertionError(expected)
    if expected in {ResultExpectation.MATRIX, ResultExpectation.BOOL_MATRIX}:
        _assert_units(cell, matrix, result, before_units)
        _assert_metadata(cell, matrix, result)
        _assert_object_metadata(cell, matrix, result)
        _assert_axes(cell, matrix, result)
    _assert_values(cell, matrix, result, before)
    if cell.mutation_expectation is MutationExpectation.PURE_NO_MUTATION:
        _assert_source_unchanged(matrix, source_snapshot)
    elif cell.mutation_expectation is MutationExpectation.INPLACE_COMMIT:
        assert result is matrix
    elif cell.mutation_expectation is MutationExpectation.ASSIGNMENT_MUTATES:
        np.testing.assert_array_equal(np.asarray(matrix), np.zeros_like(before))
    elif cell.mutation_expectation is MutationExpectation.RAW_VIEW_ALIASES:
        np.testing.assert_array_equal(result, before)
        result.flat[0] = result.flat[0] + 1
        assert np.asarray(matrix).flat[0] == result.flat[0]


def execute_contract_cell(cell: ContractCell) -> ContractObservation:
    matrix = _matrix(cell)
    before = np.asarray(matrix).copy()
    before_units = _units(matrix)
    source_snapshot = _observable_source_snapshot(matrix)
    try:
        result = _invoke(cell, matrix)
    except BaseException as exc:
        if cell.exception_class is None or type(exc) is not cell.exception_class:
            raise AssertionError(
                f"unexpected exception for {cell.id}: {exc!r}"
            ) from exc
        if cell.mutation_expectation in {
            MutationExpectation.PURE_NO_MUTATION,
            MutationExpectation.INPLACE_FAILURE_UNCHANGED,
        }:
            _assert_source_unchanged(matrix, source_snapshot)
        return ContractObservation(cell.id, True)
    if cell.exception_class is not None:
        raise AssertionError(f"expected {cell.exception_class.__name__} for {cell.id}")
    _assert_observed_result(cell, matrix, result, before, before_units, source_snapshot)
    return ContractObservation(cell.id, True)


def test_b0_manifest_has_a_literal_cell_count_and_unique_ids() -> None:
    assert len(B0_CONTRACT) == EXPECTED_B0_CELL_COUNT
    assert EXPECTED_B0_CELL_COUNT == 453
    ids = [cell.id for cell in B0_CONTRACT]
    assert len(ids) == len(set(ids))


def test_b0_manifest_cells_have_typed_complete_expectations() -> None:
    for cell in B0_CONTRACT:
        assert isinstance(cell.family, MatrixFamily)
        assert isinstance(cell.phase, Phase)
        assert isinstance(cell.surface, Surface)
        assert isinstance(cell.expected_result, ResultExpectation)
        assert isinstance(cell.unit_expectation, UnitExpectation)
        assert isinstance(cell.metadata_expectation, MetadataExpectation)
        assert isinstance(cell.axis_expectation, AxisExpectation)
        assert isinstance(cell.value_expectation, ValueExpectation)
        assert isinstance(cell.mutation_expectation, MutationExpectation)
        assert isinstance(cell.name_expectation, NameExpectation)
        assert isinstance(cell.epoch_expectation, EpochExpectation)
        assert isinstance(cell.attrs_expectation, AttrsExpectation)
        assert (cell.expected_result is ResultExpectation.EXCEPTION) == (
            cell.exception_class is not None
        )


def test_every_b0_cell_executes_once_through_the_typed_adapter() -> None:
    observed = [execute_contract_cell(cell) for cell in B0_CONTRACT]
    assert len(observed) == EXPECTED_B0_CELL_COUNT
    assert {item.cell_id for item in observed} == {cell.id for cell in B0_CONTRACT}
    assert all(item.matches_manifest for item in observed)


def test_mutating_a_manifest_expectation_changes_adapter_outcome() -> None:
    original = next(cell for cell in B0_CONTRACT if cell.operation == "shape")
    mutated = replace(original, expected_result=ResultExpectation.NUMPY_ARRAY)
    with pytest.raises(AssertionError):
        execute_contract_cell(mutated)


def test_ndarray_add_sub_cells_cover_out_of_place_and_in_place_dimensions() -> None:
    cells = [
        cell
        for cell in B0_CONTRACT
        if cell.operand is Operand.NDARRAY and cell.operation in {"add", "sub"}
    ]
    assert {cell.scenario for cell in cells} == {
        InputScenario.DIMENSIONLESS_MATRIX,
        InputScenario.DIMENSIONAL_INCOMPATIBLE,
    }
    assert sum(cell.mutation is Mutation.INPLACE for cell in cells) == 12
    assert (
        sum(
            cell.exception_class is None
            for cell in cells
            if cell.scenario is InputScenario.DIMENSIONLESS_MATRIX
        )
        == 18
    )
    dimensional = [
        cell
        for cell in cells
        if cell.scenario is InputScenario.DIMENSIONAL_INCOMPATIBLE
    ]
    assert all(
        cell.expected_result is ResultExpectation.EXCEPTION
        and cell.exception_class is UnitConversionError
        for cell in dimensional
    )


def test_container_arithmetic_docs_match_spectrogram_ndarray_manifest() -> None:
    document = " ".join(
        (
            Path(__file__).parents[2]
            / "docs/developers/contracts/container_arithmetic_contract.md"
        )
        .read_text(encoding="utf-8")
        .split()
    )
    assert "all six dimensional SpectrogramMatrix raw-ndarray add/sub cells" in document
    assert "pure, reflected, and in-place" in document
    assert "exact atomic `UnitConversionError` failures" in document
    assert "dimensionless ndarray cases succeed" in document
    assert "preserves dimensional cells" not in document


def test_b0_manifest_covers_approved_surface_for_every_family() -> None:
    required = {
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
    }
    for family in MatrixFamily:
        assert required <= {
            cell.operation for cell in B0_CONTRACT if cell.family is family
        }


def test_b0_manifest_covers_operand_and_operator_categories() -> None:
    operations = {cell.operation for cell in B0_CONTRACT}
    assert {"add", "sub", "mul", "truediv", "power", "sqrt", "log", "exp"} <= operations
    assert {"lt", "le", "eq", "ne", "gt", "ge"} <= operations
    assert {"isfinite", "isnan", "isreal"} <= operations
    assert {"direct_ufunc", "unsupported_operator"} <= operations


def test_add_sub_scalar_cross_product_is_complete() -> None:
    for family in MatrixFamily:
        for operation in ("add", "sub"):
            for operand in (Operand.PYTHON_SCALAR, Operand.NUMPY_SCALAR):
                for scenario in (
                    InputScenario.DIMENSIONAL_INCOMPATIBLE,
                    InputScenario.DIMENSIONLESS_MATRIX,
                ):
                    for surface, side, mutation in (
                        (Surface.ARITHMETIC, Side.NONE, Mutation.PURE),
                        (Surface.REFLECTED, Side.LEFT, Mutation.PURE),
                        (Surface.INPLACE, Side.NONE, Mutation.INPLACE),
                    ):
                        matches = [
                            cell
                            for cell in B0_CONTRACT
                            if (
                                cell.family is family
                                and cell.operation == operation
                                and cell.operand is operand
                                and cell.surface is surface
                                and cell.side is side
                                and cell.mutation is mutation
                                and cell.scenario is scenario
                            )
                        ]
                        assert len(matches) == 1
                        cell = matches[0]
                        if scenario is InputScenario.DIMENSIONAL_INCOMPATIBLE:
                            assert cell.exception_class is UnitConversionError
                        else:
                            assert cell.expected_result is ResultExpectation.MATRIX


def test_non_scalar_power_cross_product_is_complete() -> None:
    operands = {
        Operand.NDARRAY,
        Operand.PYTHON_LIST,
        Operand.PYTHON_TUPLE,
        Operand.VECTOR_QUANTITY,
        Operand.SAME_CLASS_MATRIX,
    }
    for family in MatrixFamily:
        for operand in operands:
            for scenario in (
                InputScenario.DEFAULT,
                InputScenario.DIMENSIONLESS_MATRIX,
            ):
                for surface, mutation in (
                    (Surface.ARITHMETIC, Mutation.PURE),
                    (Surface.INPLACE, Mutation.INPLACE),
                ):
                    matches = [
                        cell
                        for cell in B0_CONTRACT
                        if cell.family is family
                        and cell.operation == "power"
                        and cell.operand is operand
                        and cell.surface is surface
                        and cell.mutation is mutation
                        and cell.scenario is scenario
                    ]
                    assert len(matches) == 1
                    assert matches[0].exception_class is UnitConversionError
        for scenario in (
            InputScenario.DEFAULT,
            InputScenario.DIMENSIONLESS_MATRIX,
        ):
            reflected = [
                cell
                for cell in B0_CONTRACT
                if cell.family is family
                and cell.operation == "power"
                and cell.operand is Operand.PYTHON_SCALAR
                and cell.surface is Surface.REFLECTED
                and cell.side is Side.LEFT
                and cell.scenario is scenario
            ]
            assert len(reflected) == 1
            assert reflected[0].exception_class is TypeError


def test_direct_ufunc_cells_honestly_pin_b0_rejection() -> None:
    direct = [cell for cell in B0_CONTRACT if cell.operation == "direct_ufunc"]
    assert direct
    assert all(cell.exception_class is TypeError for cell in direct)
    assert all(cell.expected_result is ResultExpectation.EXCEPTION for cell in direct)


def test_contract_tests_forbid_xfail_escape_hatch() -> None:
    source = (
        Path(__file__).with_name("test_series_matrix_operator_contract.py").read_text()
    )
    assert "pytest.mark.xfail" not in source
    assert "xfail=" not in source


@pytest.mark.parametrize("family", list(MatrixFamily))
def test_manifest_is_not_only_for_one_matrix_family(family: MatrixFamily) -> None:
    assert any(cell.family is family for cell in B0_CONTRACT)
