"""Uncertainty metadata support for GWexpy 1-D Series classes.

This module implements :class:`UncertaintyMixin` so that ``Series``,
``TimeSeries``, and ``FrequencySeries`` can carry optional uncertainty
metadata (``.lower``, ``.upper``, ``.quantiles``, ``.estimate_mean``,
``.center_kind``, ``.uncertainty_kind``, ``.interval_levels``) without
introducing wrapper classes or altering default GWpy behavior when no
uncertainty metadata is attached.

Propagation policy
------------------
Uncertainty metadata is automatically propagated only across mathematically
trivial GWexpy-controlled operations:

* ``copy()`` (real or complex)
* slicing / ``crop()`` (real or complex)
* real scalar ``+c``, ``-c``, ``*c``, ``/c`` and unary ``+`` / ``-``
  (swapping ``.lower``/``.upper`` and reversing ``.quantiles`` /
  ``.interval_levels`` on negative scalars; letting ``/ 0`` follow standard
  Python/NumPy behavior)
* unit conversion ``.to()`` on real-valued series

All other operations (nonlinear ufuncs, multi-Series arithmetic, filters,
resampling, FFT, complex arithmetic, etc.) emit a :class:`UserWarning` only
when uncertainty metadata is present and drop all uncertainty attributes
together from the result (including in-place operations), without mutating
the source object on out-of-place operations.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Real
from typing import Any, Literal, cast

import numpy as np
from astropy import units as u

__all__ = [
    "UNCERTAINTY_STATE_KEYS",
    "UncertaintyMixin",
    "suppress_uncertainty_drop_warning",
    "warn_uncertainty_dropped",
]

UNCERTAINTY_STATE_KEYS: frozenset[str] = frozenset(
    {
        "_gwex_unc_lower",
        "_gwex_unc_upper",
        "_gwex_unc_quantiles",
        "_gwex_unc_estimate_mean",
        "_gwex_unc_center_kind",
        "_gwex_unc_uncertainty_kind",
        "_gwex_unc_interval_levels",
    }
)

# Context depth for operations that explicitly handle uncertainty themselves
# (copy, slice, trivial ufunc, unit conversion) or have already emitted a
# drop warning, preventing duplicate warnings from __metadata_finalize__.
_SUPPRESS_UNCERTAINTY_DROP_WARNING: ContextVar[int] = ContextVar(
    "_SUPPRESS_UNCERTAINTY_DROP_WARNING", default=0
)


@contextmanager
def suppress_uncertainty_drop_warning() -> Iterator[None]:
    """Temporarily suppress automatic drop warnings in ``__metadata_finalize__``."""
    token = _SUPPRESS_UNCERTAINTY_DROP_WARNING.set(
        _SUPPRESS_UNCERTAINTY_DROP_WARNING.get() + 1
    )
    try:
        yield
    finally:
        _SUPPRESS_UNCERTAINTY_DROP_WARNING.reset(token)


def warn_uncertainty_dropped(operation: str, *, stacklevel: int = 3) -> None:
    """Emit a ``UserWarning`` when uncertainty metadata is dropped."""
    if _SUPPRESS_UNCERTAINTY_DROP_WARNING.get() > 0:
        return
    warnings.warn(
        f"Uncertainty metadata cannot be automatically propagated across "
        f"{operation}; dropping uncertainty attributes from the result. "
        f"Use .to_timeseriesdict() for manual propagation.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _canonical_level(level: float) -> float:
    """Normalize a quantile probability to avoid floating-point key drift."""
    return round(float(level), 12)


def _is_real_scalar(value: Any) -> bool:
    """Return True iff *value* is a finite-capable real scalar (not bool/complex/array)."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, u.Quantity):
        if value.ndim != 0:
            return False
        raw = value.value
        if isinstance(raw, (bool, np.bool_)) or np.iscomplexobj(raw):
            return False
        return isinstance(raw, (Real, np.integer, np.floating))
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return False
        if value.dtype.kind not in "iuf":
            return False
        return True
    if np.iscomplexobj(value):
        return False
    return isinstance(value, (Real, np.integer, np.floating))


def _real_scalar_float(value: Any) -> float:
    """Extract the underlying real float value from a validated real scalar."""
    if isinstance(value, u.Quantity):
        return float(np.asarray(value.value).item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _validate_interval_levels(
    levels: tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    """Validate and normalize ``(low, high)`` interval probability levels."""
    if levels is None:
        return None
    if isinstance(levels, (str, bytes)):
        raise ValueError("interval_levels must be a 2-tuple of probabilities in (0, 1)")
    try:
        pair = tuple(levels)
    except TypeError as exc:
        raise ValueError(
            "interval_levels must be a 2-tuple of probabilities in (0, 1)"
        ) from exc
    if len(pair) != 2:
        raise ValueError("interval_levels must have length 2")
    out: list[float] = []
    for item in pair:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise ValueError("interval_levels must contain real numeric probabilities")
        val = float(item)
        if not np.isfinite(val) or not 0.0 < val < 1.0:
            raise ValueError("interval_levels must be strictly between 0 and 1")
        out.append(_canonical_level(val))
    if not out[0] < out[1]:
        raise ValueError("interval_levels must satisfy low < high")
    return (out[0], out[1])


def _validate_quantiles_mapping(
    quantiles: Mapping[float, Any] | None,
) -> dict[float, Any] | None:
    """Validate and normalize a quantile mapping."""
    if quantiles is None:
        return None
    if not isinstance(quantiles, Mapping):
        raise TypeError("quantiles must be a mapping from probability levels to Series")
    normalized: dict[float, Any] = {}
    for key, series in quantiles.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, Real):
            raise ValueError("quantile keys must be real numeric levels in (0, 1)")
        level = float(key)
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("quantile keys must be strictly between 0 and 1")
        normalized[_canonical_level(level)] = series
    return normalized


class UncertaintyMixin:
    """Mixin adding optional uncertainty metadata and propagation hooks to Series."""

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _has_uncertainty(self) -> bool:
        """Return True if any uncertainty attribute is set on this instance."""
        d = self.__dict__
        return any(d.get(key) is not None for key in UNCERTAINTY_STATE_KEYS)

    def _clear_uncertainty(self) -> None:
        """Remove all uncertainty attributes from this instance."""
        d = self.__dict__
        for key in UNCERTAINTY_STATE_KEYS:
            d.pop(key, None)

    def _set_uncertainty(
        self,
        *,
        lower: Any = None,
        upper: Any = None,
        quantiles: Mapping[float, Any] | None = None,
        estimate_mean: Any = None,
        center_kind: str | None = None,
        uncertainty_kind: str | None = None,
        interval_levels: tuple[float, float] | list[float] | None = None,
    ) -> None:
        """Atomically assign uncertainty metadata on this instance."""
        norm_quantiles = _validate_quantiles_mapping(quantiles)
        norm_levels = _validate_interval_levels(interval_levels)
        if (
            norm_quantiles is not None
            and norm_levels is not None
            and (lower is not None or upper is not None)
        ):
            raise ValueError(
                "Cannot specify explicit lower/upper alongside quantile-derived "
                "bounds (quantiles + interval_levels); lower/upper are derived "
                "from quantiles[interval_levels]."
            )
        self._clear_uncertainty()
        d = self.__dict__
        if lower is not None:
            d["_gwex_unc_lower"] = lower
        if upper is not None:
            d["_gwex_unc_upper"] = upper
        if norm_quantiles is not None:
            d["_gwex_unc_quantiles"] = norm_quantiles
        if estimate_mean is not None:
            d["_gwex_unc_estimate_mean"] = estimate_mean
        if center_kind is not None:
            d["_gwex_unc_center_kind"] = center_kind
        if uncertainty_kind is not None:
            d["_gwex_unc_uncertainty_kind"] = uncertainty_kind
        if norm_levels is not None:
            d["_gwex_unc_interval_levels"] = norm_levels

    def _uncertainty_snapshot(self) -> dict[str, Any]:
        """Return a shallow snapshot of raw stored uncertainty keys."""
        d = self.__dict__
        return {
            key: d.get(key) for key in UNCERTAINTY_STATE_KEYS if d.get(key) is not None
        }

    def _apply_uncertainty_transform(
        self,
        target: UncertaintyMixin,
        transform: Any,
        *,
        reverse_bounds: bool = False,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Transform uncertainty series from ``self`` (or ``snapshot``) onto ``target``."""
        raw = self._uncertainty_snapshot() if snapshot is None else snapshot
        if not raw:
            target._clear_uncertainty()
            return

        lower = raw.get("_gwex_unc_lower")
        upper = raw.get("_gwex_unc_upper")
        quantiles = raw.get("_gwex_unc_quantiles")
        estimate_mean = raw.get("_gwex_unc_estimate_mean")
        center_kind = raw.get("_gwex_unc_center_kind")
        uncertainty_kind = raw.get("_gwex_unc_uncertainty_kind")
        interval_levels = raw.get("_gwex_unc_interval_levels")

        new_lower = transform(lower) if lower is not None else None
        new_upper = transform(upper) if upper is not None else None
        if reverse_bounds:
            new_lower, new_upper = new_upper, new_lower

        new_quantiles: dict[float, Any] | None = None
        if quantiles is not None:
            if reverse_bounds:
                new_quantiles = {
                    _canonical_level(1.0 - level): transform(series)
                    for level, series in quantiles.items()
                }
            else:
                new_quantiles = {
                    level: transform(series) for level, series in quantiles.items()
                }

        new_levels = interval_levels
        if reverse_bounds and interval_levels is not None:
            lo, hi = interval_levels
            new_levels = (_canonical_level(1.0 - hi), _canonical_level(1.0 - lo))

        new_mean = transform(estimate_mean) if estimate_mean is not None else None

        target._set_uncertainty(
            lower=new_lower,
            upper=new_upper,
            quantiles=new_quantiles,
            estimate_mean=new_mean,
            center_kind=center_kind,
            uncertainty_kind=uncertainty_kind,
            interval_levels=new_levels,
        )

    def _copy_uncertainty_to(
        self, target: UncertaintyMixin, *, order: str = "C"
    ) -> None:
        """Deep-copy uncertainty attributes from ``self`` onto ``target``."""
        if not self._has_uncertainty():
            return
        with suppress_uncertainty_drop_warning():
            self._apply_uncertainty_transform(
                target,
                lambda s: s.copy(order=order) if hasattr(s, "copy") else s,
            )

    def _slice_uncertainty_to(self, target: UncertaintyMixin, item: Any) -> None:
        """Slice uncertainty attributes from ``self`` onto ``target``."""
        if not self._has_uncertainty():
            return
        with suppress_uncertainty_drop_warning():
            self._apply_uncertainty_transform(target, lambda s: s[item])

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def lower(self) -> Any:
        """Lower bound series (derived from ``quantiles`` at ``interval_levels[0]`` when quantile-based)."""
        explicit = self.__dict__.get("_gwex_unc_lower")
        if explicit is not None:
            return explicit
        quantiles = self.__dict__.get("_gwex_unc_quantiles")
        levels = self.__dict__.get("_gwex_unc_interval_levels")
        if quantiles is not None and levels is not None:
            return quantiles.get(levels[0])
        return None

    @lower.setter
    def lower(self, value: Any) -> None:
        if value is None:
            self.__dict__.pop("_gwex_unc_lower", None)
            return
        if (
            self.__dict__.get("_gwex_unc_quantiles") is not None
            and self.__dict__.get("_gwex_unc_interval_levels") is not None
        ):
            raise ValueError(
                "Cannot overwrite .lower independently while quantile-derived "
                "bounds (quantiles + interval_levels) are active; update "
                ".quantiles or .interval_levels, or clear .quantiles first."
            )
        self.__dict__["_gwex_unc_lower"] = value

    @property
    def upper(self) -> Any:
        """Upper bound series (derived from ``quantiles`` at ``interval_levels[1]`` when quantile-based)."""
        explicit = self.__dict__.get("_gwex_unc_upper")
        if explicit is not None:
            return explicit
        quantiles = self.__dict__.get("_gwex_unc_quantiles")
        levels = self.__dict__.get("_gwex_unc_interval_levels")
        if quantiles is not None and levels is not None:
            return quantiles.get(levels[1])
        return None

    @upper.setter
    def upper(self, value: Any) -> None:
        if value is None:
            self.__dict__.pop("_gwex_unc_upper", None)
            return
        if (
            self.__dict__.get("_gwex_unc_quantiles") is not None
            and self.__dict__.get("_gwex_unc_interval_levels") is not None
        ):
            raise ValueError(
                "Cannot overwrite .upper independently while quantile-derived "
                "bounds (quantiles + interval_levels) are active; update "
                ".quantiles or .interval_levels, or clear .quantiles first."
            )
        self.__dict__["_gwex_unc_upper"] = value

    @property
    def quantiles(self) -> dict[float, Any] | None:
        """Optional mapping from quantile levels in ``(0, 1)`` to Series."""
        return self.__dict__.get("_gwex_unc_quantiles")

    @property
    def percentiles(self) -> dict[float, Any] | None:
        """Quantile series indexed by percentile, derived from ``quantiles``."""
        quantiles = self.quantiles
        if quantiles is None:
            return None
        return {round(level * 100, 12): series for level, series in quantiles.items()}

    @quantiles.setter
    def quantiles(self, value: Mapping[float, Any] | None) -> None:
        normalized = _validate_quantiles_mapping(value)
        if normalized is None:
            self.__dict__.pop("_gwex_unc_quantiles", None)
        else:
            self.__dict__["_gwex_unc_quantiles"] = normalized

    @property
    def estimate_mean(self) -> Any:
        """Optional independent predicted mean Series (``None`` if unavailable)."""
        return self.__dict__.get("_gwex_unc_estimate_mean")

    @estimate_mean.setter
    def estimate_mean(self, value: Any) -> None:
        if value is None:
            self.__dict__.pop("_gwex_unc_estimate_mean", None)
        else:
            self.__dict__["_gwex_unc_estimate_mean"] = value

    @property
    def center_kind(self) -> str | None:
        """Kind of central representative value (e.g. ``'median'``, ``'mean'``)."""
        return self.__dict__.get("_gwex_unc_center_kind")

    @center_kind.setter
    def center_kind(self, value: str | None) -> None:
        if value is None:
            self.__dict__.pop("_gwex_unc_center_kind", None)
        else:
            self.__dict__["_gwex_unc_center_kind"] = str(value)

    @property
    def uncertainty_kind(self) -> str | None:
        """Kind of uncertainty interval (e.g. ``'predictive'``)."""
        return self.__dict__.get("_gwex_unc_uncertainty_kind")

    @uncertainty_kind.setter
    def uncertainty_kind(self, value: str | None) -> None:
        if value is None:
            self.__dict__.pop("_gwex_unc_uncertainty_kind", None)
        else:
            self.__dict__["_gwex_unc_uncertainty_kind"] = str(value)

    @property
    def interval_levels(self) -> tuple[float, float] | None:
        """Probability levels ``(low, high)`` defining ``lower`` and ``upper``."""
        return self.__dict__.get("_gwex_unc_interval_levels")

    @interval_levels.setter
    def interval_levels(self, value: tuple[float, float] | list[float] | None) -> None:
        normalized = _validate_interval_levels(value)
        if normalized is None:
            self.__dict__.pop("_gwex_unc_interval_levels", None)
        else:
            self.__dict__["_gwex_unc_interval_levels"] = normalized

    # ------------------------------------------------------------------
    # Propagation & warn-and-drop hooks
    # ------------------------------------------------------------------

    def copy(self, order: Literal["C", "F", "A", "K"] = "C") -> Any:
        """Copy this series and propagate any attached uncertainty metadata."""
        if not self._has_uncertainty():
            return super().copy(order=order)  # type: ignore[misc]
        with suppress_uncertainty_drop_warning():
            result = super().copy(order=order)  # type: ignore[misc]
        if isinstance(result, UncertaintyMixin):
            self._copy_uncertainty_to(result, order=order)
        return result

    def __getitem__(self, item: Any) -> Any:
        """Slice this series and propagate uncertainty metadata on slice views."""
        if not self._has_uncertainty():
            return super().__getitem__(item)  # type: ignore[misc]
        slice_item = item[0] if isinstance(item, tuple) and len(item) == 1 else item
        is_slice = isinstance(slice_item, slice)
        if is_slice:
            with suppress_uncertainty_drop_warning():
                result = super().__getitem__(item)  # type: ignore[misc]
            if isinstance(result, type(self)) and getattr(result, "ndim", None) == 1:
                self._slice_uncertainty_to(result, item)
            return result
        result = super().__getitem__(item)  # type: ignore[misc]
        if isinstance(result, UncertaintyMixin):
            result._clear_uncertainty()
            warn_uncertainty_dropped("non-slice indexing", stacklevel=2)
        return result

    def to(self, unit: Any, equivalencies: Any = (), copy: bool = True) -> Any:
        """Convert physical unit and propagate uncertainty on real-valued series."""
        if not self._has_uncertainty():
            return super().to(unit, equivalencies=equivalencies, copy=copy)  # type: ignore[misc]
        with suppress_uncertainty_drop_warning():
            result = super().to(unit, equivalencies=equivalencies, copy=copy)  # type: ignore[misc]
        if not isinstance(result, UncertaintyMixin):
            return result
        if np.iscomplexobj(cast(Any, self)):
            if result is not self:
                result._clear_uncertainty()
            warn_uncertainty_dropped("complex unit conversion", stacklevel=2)
            return result
        if result is self:
            return result
        with suppress_uncertainty_drop_warning():
            self._apply_uncertainty_transform(
                result,
                lambda s: s.to(unit, equivalencies=equivalencies, copy=True),
            )
        return result

    def __metadata_finalize__(self, obj: Any, *, force: bool = False) -> None:
        """Drop uncertainty metadata on GWpy-rebuilt derived series with warning."""
        super().__metadata_finalize__(obj, force=force)  # type: ignore[misc]
        self._clear_uncertainty()
        if isinstance(obj, UncertaintyMixin) and obj._has_uncertainty():
            warn_uncertainty_dropped("series transformation", stacklevel=3)

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Propagate uncertainty on trivial real scalar arithmetic; warn-and-drop otherwise."""
        unc_inputs = [
            (idx, inp)
            for idx, inp in enumerate(inputs)
            if isinstance(inp, UncertaintyMixin) and inp._has_uncertainty()
        ]
        if not unc_inputs:
            return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # type: ignore[misc]

        # Snapshot source uncertainty before super() runs (in case of in-place out=self).
        primary_idx, primary = unc_inputs[0]
        primary_snapshot = primary._uncertainty_snapshot()

        # Check if this call qualifies as a mathematically trivial real scalar op.
        trivial_transform: Any = None
        reverse_bounds = False

        if (
            method == "__call__"
            and len(unc_inputs) == 1
            and not np.iscomplexobj(cast(Any, primary))
        ):
            if ufunc is np.positive and len(inputs) == 1:
                trivial_transform = lambda s: +s  # noqa: E731
            elif ufunc is np.negative and len(inputs) == 1:
                trivial_transform = lambda s: -s  # noqa: E731
                reverse_bounds = True
            elif len(inputs) == 2 and ufunc in (
                np.add,
                np.subtract,
                np.multiply,
                np.true_divide,
            ):
                other_idx = 1 - primary_idx
                other = inputs[other_idx]
                if _is_real_scalar(other):
                    scalar_val = _real_scalar_float(other)
                    if np.isfinite(scalar_val):
                        if ufunc is np.add:
                            trivial_transform = (
                                (lambda s, c=other: s + c)
                                if primary_idx == 0
                                else (lambda s, c=other: c + s)
                            )
                        elif ufunc is np.subtract:
                            if primary_idx == 0:
                                trivial_transform = lambda s, c=other: s - c  # noqa: E731
                            else:
                                trivial_transform = lambda s, c=other: c - s  # noqa: E731
                                reverse_bounds = True
                        elif ufunc is np.multiply:
                            trivial_transform = (
                                (lambda s, c=other: s * c)
                                if primary_idx == 0
                                else (lambda s, c=other: c * s)
                            )
                            reverse_bounds = scalar_val < 0
                        elif ufunc is np.true_divide and primary_idx == 0:
                            if scalar_val != 0.0:
                                trivial_transform = lambda s, c=other: s / c  # noqa: E731
                                reverse_bounds = scalar_val < 0

        with suppress_uncertainty_drop_warning():
            result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # type: ignore[misc]

        if result is NotImplemented:
            return NotImplemented

        out_targets = kwargs.get("out")
        if trivial_transform is not None and isinstance(result, UncertaintyMixin):
            # Guard against result becoming complex (e.g., unexpected dtype promotion).
            if not np.iscomplexobj(cast(Any, result)):
                with suppress_uncertainty_drop_warning():
                    primary._apply_uncertainty_transform(
                        result,
                        trivial_transform,
                        reverse_bounds=reverse_bounds,
                        snapshot=primary_snapshot,
                    )
                return result

        # Unsupported / nonlinear / multi-Series / complex / /0 operation:
        # drop uncertainty from result (and any in-place out targets) and warn.
        if isinstance(result, UncertaintyMixin):
            result._clear_uncertainty()
        if out_targets is not None:
            targets = out_targets if isinstance(out_targets, tuple) else (out_targets,)
            for target in targets:
                if isinstance(target, UncertaintyMixin):
                    target._clear_uncertainty()

        op_name = getattr(ufunc, "__name__", str(ufunc))
        warn_uncertainty_dropped(f"ufunc '{op_name}'", stacklevel=2)
        return result
