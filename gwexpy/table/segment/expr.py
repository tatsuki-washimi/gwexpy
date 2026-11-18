"""Column expression AST (``col()``) and typed column view (``SegmentColumn``)."""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

from gwexpy.table.segment.cell import CellStatus
from gwexpy.table.segment.status import is_scalar_nan

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable

__all__ = [
    "BinOp",
    "Call",
    "ColumnRef",
    "Expr",
    "Literal",
    "SegmentColumn",
    "UnaryOp",
    "col",
]

_STATUS_PREDICATES = frozenset(
    {"is_valid", "is_missing", "is_failed", "is_loaded", "isna", "notna"}
)
_SEGMENT_ATTRS = frozenset({"start", "end", "duration"})
_PAYLOAD_REDUCTIONS = frozenset(
    {"min", "max", "mean", "median", "std", "sum", "quantile", "apply"}
)

_BINOP_MAP: dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "&": operator.and_,
    "|": operator.or_,
}

_COMPARE_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_LOGIC_OPS = frozenset({"&", "|"})


def _ensure_expr(other: Any) -> Expr:
    if isinstance(other, Expr):
        return other
    return Literal(other)


def _safe_elementwise_compare(
    op_fn: Callable[[Any, Any], Any], lhs: Any, rhs: Any
) -> np.ndarray:
    """Compare elementwise while returning False for None/NaN values."""
    n = len(lhs)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lv = lhs[i]
        rv = rhs[i]
        if is_scalar_nan(lv) or is_scalar_nan(rv):
            out[i] = False
            continue
        try:
            out[i] = bool(op_fn(lv, rv))
        except TypeError:
            out[i] = False
    return out


class Expr:
    """Base class for lazy column-expression AST nodes."""

    __hash__ = None  # type: ignore[assignment]

    def required_columns(self) -> frozenset[str]:
        """Return the set of column names referenced by this expression."""
        raise NotImplementedError

    def evaluate(self, table: SegmentTable) -> np.ndarray:
        """Evaluate this expression eagerly against *table*."""
        raise NotImplementedError

    def __bool__(self) -> bool:
        raise TypeError(
            "Cannot convert column expression to bool; use '&', '|', and '~' "
            "for elementwise boolean logic."
        )

    # Arithmetic
    def __add__(self, other: Any) -> BinOp:
        return BinOp(self, "+", _ensure_expr(other))

    def __radd__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "+", self)

    def __sub__(self, other: Any) -> BinOp:
        return BinOp(self, "-", _ensure_expr(other))

    def __rsub__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "-", self)

    def __mul__(self, other: Any) -> BinOp:
        return BinOp(self, "*", _ensure_expr(other))

    def __rmul__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "*", self)

    def __truediv__(self, other: Any) -> BinOp:
        return BinOp(self, "/", _ensure_expr(other))

    def __rtruediv__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "/", self)

    def __floordiv__(self, other: Any) -> BinOp:
        return BinOp(self, "//", _ensure_expr(other))

    def __rfloordiv__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "//", self)

    def __mod__(self, other: Any) -> BinOp:
        return BinOp(self, "%", _ensure_expr(other))

    def __rmod__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "%", self)

    def __pow__(self, other: Any) -> BinOp:
        return BinOp(self, "**", _ensure_expr(other))

    def __rpow__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "**", self)

    # Comparisons
    def __eq__(self, other: Any) -> BinOp:  # type: ignore[override]
        return BinOp(self, "==", _ensure_expr(other))

    def __ne__(self, other: Any) -> BinOp:  # type: ignore[override]
        return BinOp(self, "!=", _ensure_expr(other))

    def __lt__(self, other: Any) -> BinOp:
        return BinOp(self, "<", _ensure_expr(other))

    def __le__(self, other: Any) -> BinOp:
        return BinOp(self, "<=", _ensure_expr(other))

    def __gt__(self, other: Any) -> BinOp:
        return BinOp(self, ">", _ensure_expr(other))

    def __ge__(self, other: Any) -> BinOp:
        return BinOp(self, ">=", _ensure_expr(other))

    # Logical
    def __and__(self, other: Any) -> BinOp:
        return BinOp(self, "&", _ensure_expr(other))

    def __rand__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "&", self)

    def __or__(self, other: Any) -> BinOp:
        return BinOp(self, "|", _ensure_expr(other))

    def __ror__(self, other: Any) -> BinOp:
        return BinOp(_ensure_expr(other), "|", self)

    def __invert__(self) -> UnaryOp:
        return UnaryOp("~", self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("-", self)

    def __pos__(self) -> UnaryOp:
        return UnaryOp("+", self)

    def __abs__(self) -> Call:
        return Call(self, "abs")

    # Meta / general expression methods
    def isin(self, values: Iterable[Any]) -> Call:
        """Return a boolean predicate checking membership in *values*."""
        return Call(self, "isin", args=(tuple(values),))

    def between(self, low: Any, high: Any, *, inclusive: str = "both") -> Call:
        """Return a boolean predicate checking ``low <= val <= high``."""
        return Call(self, "between", args=(low, high), kwargs={"inclusive": inclusive})

    def contains(self, pattern: str, *, case: bool = True, regex: bool = False) -> Call:
        """Return a boolean predicate checking string substring/regex match."""
        return Call(
            self,
            "contains",
            args=(pattern,),
            kwargs={"case": case, "regex": regex},
        )

    def abs(self) -> Call:
        """Return elementwise absolute value."""
        return Call(self, "abs")


class Literal(Expr):
    """Literal constant value node."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def required_columns(self) -> frozenset[str]:
        """Return empty column set for literals."""
        return frozenset()

    def evaluate(self, table: SegmentTable) -> np.ndarray:
        """Broadcast the literal value across all rows of *table*."""
        n = len(table)
        val = self.value
        if isinstance(val, (bool, np.bool_)):
            return np.full(n, bool(val), dtype=bool)
        if isinstance(val, (int, float, np.number)):
            return np.full(n, val)
        arr = np.empty(n, dtype=object)
        arr[:] = [val] * n
        return arr

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


class ColumnRef(Expr):
    """Reference to a column in a :class:`SegmentTable`."""

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Column name must be a non-empty string, got {name!r}.")
        self.name = name

    def required_columns(self) -> frozenset[str]:
        """Return the singleton set containing this column name."""
        return frozenset({self.name})

    def evaluate(self, table: SegmentTable) -> np.ndarray:
        """Evaluate this column reference against *table*."""
        if self.name not in table._schema:
            raise KeyError(f"Column {self.name!r} not found in SegmentTable.")
        if self.name in table._payload:
            raise ValueError(
                f"Cannot directly evaluate payload column {self.name!r} in a column "
                "expression; use status predicates (.is_valid(), .is_missing(), ...) "
                "or pass load=True to a payload reduction (.max(load=True), ...)."
            )
        return table._meta[self.name].to_numpy()

    # Non-loading status predicates
    def is_valid(self) -> Call:
        """Non-loading predicate: True where cell status is ``VALID``."""
        return Call(self, "is_valid")

    def is_missing(self) -> Call:
        """Non-loading predicate: True where cell status is ``MISSING``."""
        return Call(self, "is_missing")

    def is_failed(self) -> Call:
        """Non-loading predicate: True where cell status is ``FAILED``."""
        return Call(self, "is_failed")

    def is_loaded(self) -> Call:
        """Non-loading predicate: True where payload cell is already loaded."""
        return Call(self, "is_loaded")

    def isna(self) -> Call:
        """Non-loading predicate: True where cell is missing, failed, masked, or N/A."""
        return Call(self, "isna")

    def notna(self) -> Call:
        """Non-loading predicate: negation of :meth:`isna` (keeps ``UNLOADED``)."""
        return Call(self, "notna")

    # Segment / span helpers
    @property
    def start(self) -> Call:
        """Extract segment start time from a segment column."""
        return Call(self, "start")

    @property
    def end(self) -> Call:
        """Extract segment end time from a segment column."""
        return Call(self, "end")

    @property
    def duration(self) -> Call:
        """Extract segment duration (``end - start``) from a segment column."""
        return Call(self, "duration")

    # Payload content reductions (require load=True on payload columns)
    def min(self, *, load: bool = False) -> Call:
        """Per-row minimum (requires ``load=True`` for payload columns)."""
        return Call(self, "min", kwargs={"load": load})

    def max(self, *, load: bool = False) -> Call:
        """Per-row maximum (requires ``load=True`` for payload columns)."""
        return Call(self, "max", kwargs={"load": load})

    def mean(self, *, load: bool = False) -> Call:
        """Per-row mean (requires ``load=True`` for payload columns)."""
        return Call(self, "mean", kwargs={"load": load})

    def median(self, *, load: bool = False) -> Call:
        """Per-row median (requires ``load=True`` for payload columns)."""
        return Call(self, "median", kwargs={"load": load})

    def std(self, *, load: bool = False) -> Call:
        """Per-row standard deviation (requires ``load=True`` for payload columns)."""
        return Call(self, "std", kwargs={"load": load})

    def sum(self, *, load: bool = False) -> Call:
        """Per-row sum (requires ``load=True`` for payload columns)."""
        return Call(self, "sum", kwargs={"load": load})

    def quantile(self, q: float, *, load: bool = False) -> Call:
        """Per-row quantile *q* (requires ``load=True`` for payload columns)."""
        return Call(self, "quantile", args=(q,), kwargs={"load": load})

    def apply(self, func: Callable[[Any], Any], *, load: bool = False) -> Call:
        """Per-row callable *func* (requires ``load=True`` for payload columns)."""
        return Call(self, "apply", args=(func,), kwargs={"load": load})

    def __repr__(self) -> str:
        return f"col({self.name!r})"


class BinOp(Expr):
    """Binary operation node combining two expressions."""

    def __init__(self, left: Expr, op: str, right: Expr) -> None:
        if op not in _BINOP_MAP:
            raise ValueError(f"Unsupported binary operator {op!r}.")
        self.left = left
        self.op = op
        self.right = right

    def required_columns(self) -> frozenset[str]:
        """Return the union of columns required by left and right operands."""
        return self.left.required_columns() | self.right.required_columns()

    def evaluate(self, table: SegmentTable) -> np.ndarray:
        """Evaluate the binary operation elementwise against *table*."""
        lhs = self.left.evaluate(table)
        rhs = self.right.evaluate(table)
        op_fn = _BINOP_MAP[self.op]

        if self.op in _LOGIC_OPS:
            l_bool = np.asarray(lhs, dtype=bool)
            r_bool = np.asarray(rhs, dtype=bool)
            return np.asarray(op_fn(l_bool, r_bool), dtype=bool)

        if self.op in _COMPARE_OPS:
            if lhs.dtype == object or rhs.dtype == object:
                return _safe_elementwise_compare(op_fn, lhs, rhs)
            with np.errstate(invalid="ignore"):
                res = op_fn(lhs, rhs)
            return np.asarray(res, dtype=bool)

        return np.asarray(op_fn(lhs, rhs))

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


class UnaryOp(Expr):
    """Unary operation node (``-``, ``+``, ``~``)."""

    def __init__(self, op: str, operand: Expr) -> None:
        if op not in ("-", "+", "~"):
            raise ValueError(f"Unsupported unary operator {op!r}.")
        self.op = op
        self.operand = operand

    def required_columns(self) -> frozenset[str]:
        """Return columns required by the unary operand."""
        return self.operand.required_columns()

    def evaluate(self, table: SegmentTable) -> np.ndarray:
        """Evaluate the unary operation elementwise against *table*."""
        val = self.operand.evaluate(table)
        if self.op == "-":
            return np.asarray(-val)
        if self.op == "+":
            return np.asarray(+val)
        return np.asarray(np.logical_not(np.asarray(val, dtype=bool)), dtype=bool)

    def __repr__(self) -> str:
        return f"({self.op}{self.operand!r})"


class Call(Expr):
    """Method/predicate call AST node on a column or expression."""

    def __init__(
        self,
        target: Expr,
        func: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.target = target
        self.func = func
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})

    def required_columns(self) -> frozenset[str]:
        """Return columns required by the call target."""
        return self.target.required_columns()

    def _eval_status_predicate(self, table: SegmentTable) -> np.ndarray:
        if not isinstance(self.target, ColumnRef):
            raise TypeError(
                f"Status predicate .{self.func}() requires a ColumnRef target."
            )
        col_name = self.target.name
        if col_name not in table._schema:
            raise KeyError(f"Column {col_name!r} not found in SegmentTable.")

        if col_name in table._payload:
            cells = table._payload[col_name]
            if self.func == "is_valid":
                flags = [c.status == CellStatus.VALID for c in cells]
            elif self.func == "is_missing":
                flags = [c.status == CellStatus.MISSING for c in cells]
            elif self.func == "is_failed":
                flags = [c.status == CellStatus.FAILED for c in cells]
            elif self.func == "is_loaded":
                flags = [c.is_loaded() for c in cells]
            elif self.func == "isna":
                flags = [
                    c.status
                    in (
                        CellStatus.MISSING,
                        CellStatus.FAILED,
                        CellStatus.MASKED,
                        CellStatus.NOT_APPLICABLE,
                    )
                    for c in cells
                ]
            else:
                # notna
                flags = [
                    c.status in (CellStatus.VALID, CellStatus.UNLOADED) for c in cells
                ]
            return np.asarray(flags, dtype=bool)

        # Meta column
        n = len(table)
        statuses = [table._meta_row_status(col_name, i) for i in range(n)]
        if self.func == "is_valid":
            flags = [s == CellStatus.VALID for s in statuses]
        elif self.func == "is_missing":
            flags = [s == CellStatus.MISSING for s in statuses]
        elif self.func == "is_failed":
            flags = [s == CellStatus.FAILED for s in statuses]
        elif self.func == "is_loaded":
            flags = [True] * n
        elif self.func == "isna":
            flags = [s != CellStatus.VALID for s in statuses]
        else:
            flags = [s == CellStatus.VALID for s in statuses]
        return np.asarray(flags, dtype=bool)

    def _eval_payload_reduction(self, table: SegmentTable) -> np.ndarray:
        load = bool(self.kwargs.get("load", False))
        if isinstance(self.target, ColumnRef) and self.target.name in table._payload:
            col_name = self.target.name
            if not load:
                raise ValueError(
                    f"Evaluating .{self.func}() on payload column {col_name!r} "
                    "inspects payload contents and requires load=True."
                )
            cells = table._payload[col_name]
            out: list[Any] = []
            for cell in cells:
                if cell.status in (
                    CellStatus.MISSING,
                    CellStatus.MASKED,
                    CellStatus.NOT_APPLICABLE,
                ):
                    out.append(np.nan)
                    continue
                val = cell.get()
                out.append(self._reduce_single(val))
            dtype = object if self.func == "apply" else float
            return np.asarray(out, dtype=dtype)

        # Meta column evaluation
        arr = self.target.evaluate(table)
        if self.func == "apply":
            fn = self.args[0]
            return np.asarray([fn(v) for v in arr], dtype=object)
        return np.asarray([self._reduce_single(v) for v in arr], dtype=float)

    def _reduce_single(self, val: Any) -> Any:
        if self.func == "apply":
            return self.args[0](val)
        if val is None or is_scalar_nan(val):
            return np.nan
        raw = np.asarray(val.value if hasattr(val, "value") else val, dtype=float)
        if raw.size == 0:
            return np.nan
        if self.func == "min":
            return float(np.nanmin(raw))
        if self.func == "max":
            return float(np.nanmax(raw))
        if self.func == "mean":
            return float(np.nanmean(raw))
        if self.func == "median":
            return float(np.nanmedian(raw))
        if self.func == "std":
            return float(np.nanstd(raw))
        if self.func == "sum":
            return float(np.nansum(raw))
        if self.func == "quantile":
            return float(np.nanquantile(raw, float(self.args[0])))
        raise ValueError(f"Unsupported reduction {self.func!r}.")

    def evaluate(self, table: SegmentTable) -> np.ndarray:
        """Evaluate this method/predicate call against *table*."""
        if self.func in _STATUS_PREDICATES:
            return self._eval_status_predicate(table)

        if self.func in _PAYLOAD_REDUCTIONS:
            return self._eval_payload_reduction(table)

        arr = self.target.evaluate(table)
        if self.func in _SEGMENT_ATTRS:
            if self.func == "start":
                return np.asarray([float(s[0]) for s in arr], dtype=float)
            if self.func == "end":
                return np.asarray([float(s[1]) for s in arr], dtype=float)
            return np.asarray([float(s[1] - s[0]) for s in arr], dtype=float)

        if self.func == "abs":
            return np.asarray(np.abs(arr))

        if self.func == "isin":
            allowed = self.args[0]
            return np.asarray(
                [False if is_scalar_nan(v) else v in allowed for v in arr],
                dtype=bool,
            )

        if self.func == "between":
            low, high = self.args
            inclusive = self.kwargs.get("inclusive", "both")
            if inclusive not in ("both", "neither", "left", "right"):
                raise ValueError(
                    f"Invalid inclusive={inclusive!r}; expected 'both', 'neither', 'left', or 'right'."
                )
            left_op = operator.ge if inclusive in ("both", "left") else operator.gt
            right_op = operator.le if inclusive in ("both", "right") else operator.lt
            flags: list[bool] = []
            for v in arr:
                if is_scalar_nan(v):
                    flags.append(False)
                else:
                    try:
                        flags.append(bool(left_op(v, low) and right_op(v, high)))
                    except TypeError:
                        flags.append(False)
            return np.asarray(flags, dtype=bool)

        if self.func == "contains":
            pattern = self.args[0]
            case = bool(self.kwargs.get("case", True))
            regex = bool(self.kwargs.get("regex", False))
            s = pd.Series(arr, dtype="string")
            matched = s.str.contains(pattern, case=case, regex=regex, na=False)
            return matched.to_numpy(dtype=bool)

        raise ValueError(f"Unsupported Call function {self.func!r}.")

    def __repr__(self) -> str:
        args_repr = ", ".join(repr(a) for a in self.args)
        kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        all_args = ", ".join(part for part in (args_repr, kwargs_repr) if part)
        return f"{self.target!r}.{self.func}({all_args})"


def col(name: str) -> ColumnRef:
    """Create a column expression reference for *name*."""
    return ColumnRef(name)


class SegmentColumn:
    """Typed column view returned by ``table[col_name]``.

    Provides metadata properties (``unit``, ``kind``, ``status``), non-loading
    ``isna()``, summary statistics (``mean()``, ``median()``, ``quantile()``),
    and explicit export helpers (``to_numpy()``, ``to_list()``).
    """

    def __init__(self, table: SegmentTable, name: str) -> None:
        if name not in table._schema:
            raise KeyError(f"Column {name!r} not found in SegmentTable.")
        self._table = table
        self._name = name

    @property
    def name(self) -> str:
        """Column name."""
        return self._name

    @property
    def kind(self) -> str:
        """Column kind from the table schema."""
        return self._table._schema[self._name]

    @property
    def unit(self) -> Any:
        """Column unit if stored or inferable from already-loaded payloads."""
        if self._name in self._table._units:
            return self._table._units[self._name]
        if self._name in self._table._payload:
            for cell in self._table._payload[self._name]:
                if cell.status == CellStatus.VALID:
                    val = cell.get()
                    if hasattr(val, "unit"):
                        return val.unit
                    break
        return None

    @property
    def status(self) -> np.ndarray:
        """1D object array of :class:`CellStatus` values (non-loading)."""
        n = len(self._table)
        if self._name in self._table._payload:
            return np.array(
                [c.status for c in self._table._payload[self._name]],
                dtype=object,
            )
        return np.array(
            [self._table._meta_row_status(self._name, i) for i in range(n)],
            dtype=object,
        )

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        n = len(self)
        if idx < -n or idx >= n:
            raise IndexError(f"Index {idx} out of range for column of length {n}.")
        if idx < 0:
            idx = n + idx
        if self._name in self._table._payload:
            return self._table._payload[self._name][idx].get()
        return self._table._meta.at[idx, self._name]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def isna(self, *, include_failed: bool = True, deep: bool = False) -> np.ndarray:
        """Return 1D boolean mask indicating missing/failed cells."""
        df = self._table.isna(
            columns=[self._name], include_failed=include_failed, deep=deep
        )
        return df[self._name].to_numpy(dtype=bool)

    def to_numpy(self, *, load: bool = False, missing_as: str = "nan") -> np.ndarray:
        """Export column values as a NumPy array.

        For payload columns, ``load=True`` is required if any cells are
        ``UNLOADED``.
        """
        if missing_as not in ("nan", "summary"):
            raise ValueError(
                f"Invalid missing_as {missing_as!r}; expected 'nan' or 'summary'."
            )
        if self._name not in self._table._payload:
            return self._table._meta[self._name].to_numpy()

        cells = self._table._payload[self._name]
        if not load and any(c.status == CellStatus.UNLOADED for c in cells):
            raise ValueError(
                f"Payload column {self._name!r} contains unloaded cells; "
                "pass load=True to materialize before calling to_numpy()."
            )
        out: list[Any] = []
        for c in cells:
            if c.status == CellStatus.VALID or (
                load and c.status == CellStatus.UNLOADED and c.loader is not None
            ):
                if c.status == CellStatus.UNLOADED:
                    try:
                        out.append(c.get())
                    except Exception:
                        out.append(
                            np.nan if missing_as == "nan" else c._summary(self.kind)
                        )
                else:
                    out.append(c.get())
            elif missing_as == "summary":
                out.append(c._summary(self.kind))
            else:
                out.append(np.nan)
        if all(isinstance(v, (bool, int, float, np.number)) or v is None for v in out):
            return np.asarray(out)
        arr = np.empty(len(out), dtype=object)
        arr[:] = out
        return arr

    def to_list(self, *, load: bool = False) -> list[Any]:
        """Export column values as a Python ``list``."""
        if self._name not in self._table._payload:
            return self._table._meta[self._name].tolist()

        cells = self._table._payload[self._name]
        if not load and any(c.status == CellStatus.UNLOADED for c in cells):
            raise ValueError(
                f"Payload column {self._name!r} contains unloaded cells; "
                "pass load=True to materialize before calling to_list()."
            )
        out: list[Any] = []
        for c in cells:
            if c.status == CellStatus.VALID:
                out.append(c.get())
            elif load and c.status == CellStatus.UNLOADED and c.loader is not None:
                try:
                    out.append(c.get())
                except Exception:
                    out.append(None)
            else:
                out.append(None)
        return out

    def _numeric_values(self, *, load: bool = False) -> np.ndarray:
        arr = self.to_numpy(load=load, missing_as="nan")
        try:
            return np.asarray(arr, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Column {self._name!r} is not numeric and does not support "
                "scalar statistics."
            ) from exc

    def mean(self, *, skipna: bool = True, load: bool = False) -> float:
        """Compute arithmetic mean across rows."""
        arr = self._numeric_values(load=load)
        if arr.size == 0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean(arr) if skipna else np.mean(arr))

    def median(self, *, skipna: bool = True, load: bool = False) -> float:
        """Compute median across rows."""
        arr = self._numeric_values(load=load)
        if arr.size == 0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmedian(arr) if skipna else np.median(arr))

    def quantile(self, q: float, *, skipna: bool = True, load: bool = False) -> float:
        """Compute *q*-th quantile (in ``[0, 1]``) across rows."""
        arr = self._numeric_values(load=load)
        if arr.size == 0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanquantile(arr, q) if skipna else np.quantile(arr, q))

    def min(self, *, skipna: bool = True, load: bool = False) -> float:
        """Compute minimum across rows."""
        arr = self._numeric_values(load=load)
        if arr.size == 0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmin(arr) if skipna else np.min(arr))

    def max(self, *, skipna: bool = True, load: bool = False) -> float:
        """Compute maximum across rows."""
        arr = self._numeric_values(load=load)
        if arr.size == 0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmax(arr) if skipna else np.max(arr))

    def std(self, *, skipna: bool = True, load: bool = False) -> float:
        """Compute standard deviation across rows."""
        arr = self._numeric_values(load=load)
        if arr.size == 0:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanstd(arr) if skipna else np.std(arr))

    def __repr__(self) -> str:
        return (
            f"SegmentColumn(name={self._name!r}, kind={self.kind!r}, "
            f"len={len(self)}, unit={self.unit!r})"
        )
