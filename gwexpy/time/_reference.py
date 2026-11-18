"""Timing-reference state shared by GWexpy containers (#725, v0.3.0 U-1).

A GWexpy time-axis container is in one of two states:

``"absolute"``
    The axis origin (``t0``/``epoch``) is a GPS time.
``"relative"``
    No absolute time is known.  The axis still has a finite origin (``0 s``
    unless the data were cropped or shifted), and the public finite
    ``t0``/``epoch`` values are the same ones GWpy returns.  ``t0 == 0``
    therefore does **not** mean GPS 0 unless the state is ``"absolute"``.

Contract for constructor arguments (``t0``, ``epoch``, ``x0``):

====================  =======================================
omitted, ``None``     relative
``NaN``               relative
finite (incl. 0.0)    absolute (an explicit ``0.0`` is GPS 0)
``+Inf`` / ``-Inf``   ``ValueError``
====================  =======================================

Assigning to ``t0``/``x0``/``epoch`` after construction accepts finite
values (GWpy semantics, the state is unchanged) and raises ``ValueError`` for
``NaN`` and ``±Inf``.  Moving between states is a separate, explicit
operation (``with_epoch()``, v0.3.0 Step 7c).

The state is stored as one private key in the instance ``__dict__`` and only
for ``"relative"``; a missing key means ``"absolute"``.  That is also the
default when an unknown code path loses the key: the object then carries only
its finite public numbers, which U-1 defines as absolute.

The public name is provisional (#725 open question 2).  To rename it, change
``TIME_REFERENCE_NAME`` and the property name in :class:`TimeReferenceMixin`.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from typing import Any, Literal, TypeVar

import numpy as np
from astropy import units as u

__all__ = [
    "TIME_INPUT_UNSET",
    "TIME_REFERENCE_ABSOLUTE",
    "TIME_REFERENCE_NAME",
    "TIME_REFERENCE_RELATIVE",
    "TIME_REFERENCE_STATE_KEY",
    "TimeReferenceMixin",
    "check_time_setter_value",
    "classify_time_input",
    "get_time_reference",
    "inherit_time_reference",
    "inherits_time_reference",
    "require_uniform_time_reference",
    "rebuild_with_time_reference",
    "resolve_matrix_epoch",
    "resolve_time_inputs",
    "set_time_reference",
]

#: Public attribute name of the state (provisional, #725 open question 2).
TIME_REFERENCE_NAME = "time_reference"
TIME_REFERENCE_ABSOLUTE = "absolute"
TIME_REFERENCE_RELATIVE = "relative"

_STATE_KEY = f"_gwex_{TIME_REFERENCE_NAME}"
#: Key of the state in an instance ``__dict__`` (for containers that pickle
#: selected ``__dict__`` entries, such as ``SeriesMatrix``).
TIME_REFERENCE_STATE_KEY = _STATE_KEY
#: Sentinel for an omitted constructor time-origin argument (distinct from ``None``).
TIME_INPUT_UNSET = object()
_VALID_STATES = (TIME_REFERENCE_ABSOLUTE, TIME_REFERENCE_RELATIVE)

TimeInputKind = Literal["missing", "nan", "inf", "finite"]
_F = TypeVar("_F", bound=Callable[..., Any])


def classify_time_input(value: object) -> TimeInputKind:
    """Classify one ``t0``/``epoch``/``x0`` argument.

    Only real scalar numbers (and quantities of them) can be ``"nan"`` or
    ``"inf"``.  Anything else (strings, ``datetime``, ``Time``,
    ``LIGOTimeGPS``, arrays) is reported as ``"finite"`` so that the existing
    GWpy/GWexpy conversion and its own errors stay in charge.
    """
    if value is None:
        return "missing"
    raw = value.value if isinstance(value, u.Quantity) else value
    if isinstance(raw, (bool, np.bool_)):
        return "finite"
    if isinstance(raw, np.ndarray):
        if raw.ndim != 0 or raw.dtype.kind not in "iuf":
            return "finite"
        raw = raw.item()
    if not isinstance(raw, (int, float, np.integer, np.floating, Decimal)):
        return "finite"
    try:
        number = float(raw)
    except OverflowError:
        return "inf"
    if np.isnan(number):
        return "nan"
    if np.isinf(number):
        return "inf"
    return "finite"


def resolve_time_inputs(
    candidates: Mapping[str, object],
    *,
    explicit_axis: bool,
    inherit: bool,
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve the state requested by constructor time arguments (U-1).

    Parameters
    ----------
    candidates
        The constructor arguments that name the time origin (for example
        ``{"t0": t0, "epoch": epoch, "x0": x0}``).  ``None`` means omitted.
    explicit_axis
        Retained for caller compatibility. Under #725 Option A, an explicit
        time axis (``times=``/``xindex=``) without a finite ``t0``/``epoch``/
        ``x0`` does not establish an absolute GPS reference and remains
        ``"relative"`` (unless inherited from ``data``).
    inherit
        ``True`` when the data already carries a state (a series that has an
        axis); an omitted origin then keeps that state.

    Returns
    -------
    state, clear
        ``state`` is ``"absolute"`` or ``"relative"`` when the arguments decide
        it, or ``None`` when it must be inherited from the data.  ``clear``
        names the arguments that must be dropped (set to ``None``) before the
        values reach GWpy, so that a ``NaN`` never poisons the axis.

    Raises
    ------
    ValueError
        If the single given origin is ``+Inf`` or ``-Inf``.

    """
    given = {
        name: value
        for name, value in candidates.items()
        if value is not TIME_INPUT_UNSET
    }
    if len(given) > 1:
        non_none = {name: value for name, value in given.items() if value is not None}
        if not non_none:
            return TIME_REFERENCE_RELATIVE, ()
        if len(non_none) > 1:
            # Conflicting authorities: leave the failure (or GWpy's own
            # precedence) to GWpy and treat what survives as explicit.
            return TIME_REFERENCE_ABSOLUTE, ()
        given = non_none
    if given:
        (name, value), *_ = given.items()
        if value is None:
            return TIME_REFERENCE_RELATIVE, ()
        kind = classify_time_input(value)
        if kind == "inf":
            raise ValueError(
                f"{name} must not be infinite (got {value!r}); pass a finite "
                f"GPS time, or pass None/NaN when the absolute time is unknown"
            )
        if kind == "nan":
            return TIME_REFERENCE_RELATIVE, (name,)
        return TIME_REFERENCE_ABSOLUTE, ()
    del explicit_axis
    return (None if inherit else TIME_REFERENCE_ABSOLUTE), ()


def resolve_matrix_epoch(
    epoch: object,
    *,
    omitted: bool,
    default: object,
    inherit: bool,
    explicit_axis: bool = False,
) -> tuple[object, str | None]:
    """Resolve the ``epoch`` constructor argument of a matrix container (U-1).

    Matrices keep ``epoch`` as a raw public attribute, so the finite value
    that GWexpy has always reported is kept and only the state is added.

    Parameters
    ----------
    epoch
        The argument as given (ignored when ``omitted``).
    omitted
        ``True`` when the caller did not pass ``epoch`` at all.
    default
        Public value used when ``epoch`` is omitted (for example ``0.0`` or the
        source matrix's ``epoch``).
    inherit
        ``True`` when an omitted ``epoch`` should keep the state of a source
        matrix (the state is then returned as ``None``).
    explicit_axis
        ``True`` when a complete time axis decides the state (see
        :func:`resolve_time_inputs`).

    Returns
    -------
    value, state
        ``value`` is what to store in ``epoch``: ``default`` when omitted,
        ``None`` when ``None`` was given, ``0.0`` (the ``NaN`` is never stored)
        for ``NaN`` and the argument itself when finite.  ``state`` is
        ``"absolute"``/``"relative"``, or ``None`` to inherit it.

    Raises
    ------
    ValueError
        If ``epoch`` is ``+Inf`` or ``-Inf``.

    """
    state, cleared = resolve_time_inputs(
        {"epoch": TIME_INPUT_UNSET if omitted else epoch},
        explicit_axis=explicit_axis,
        inherit=inherit and omitted,
    )
    if omitted:
        return default, state
    if cleared:
        return 0.0, state
    return epoch, state


def check_time_setter_value(value: object, *, name: str) -> None:
    """Raise ``ValueError`` if ``value`` is ``NaN`` or ``±Inf`` (setter rule)."""
    kind = classify_time_input(value)
    if kind in ("nan", "inf"):
        raise ValueError(
            f"{name} must be a finite time when assigned, got {value!r}. "
            "A non-finite value cannot be stored; to express an unknown "
            "absolute time construct the object without t0/epoch (relative)."
        )


def get_time_reference(obj: object) -> str:
    """Return ``"absolute"`` or ``"relative"`` for ``obj`` (default absolute)."""
    state = getattr(obj, "__dict__", {}).get(_STATE_KEY)
    return (
        TIME_REFERENCE_RELATIVE
        if state == TIME_REFERENCE_RELATIVE
        else (TIME_REFERENCE_ABSOLUTE)
    )


def require_uniform_time_reference(objects: Iterable[object]) -> str | None:
    """Reject a collection whose items disagree on absolute vs relative time.

    A single matrix-level ``time_reference`` cannot preserve mixed per-item
    states.  Call this before converting a collection to a matrix, which would
    otherwise inherit only the first item's meaning.
    """
    states = {get_time_reference(obj) for obj in objects}
    if len(states) > 1:
        raise ValueError("cannot convert a collection with mixed time_reference states")
    return next(iter(states), None)


def set_time_reference(obj: Any, state: str) -> None:
    """Record ``state`` on ``obj`` (only ``"relative"`` is stored)."""
    if state not in _VALID_STATES:
        raise ValueError(
            f"{TIME_REFERENCE_NAME} must be one of {_VALID_STATES}, got {state!r}"
        )
    if state == TIME_REFERENCE_RELATIVE:
        obj.__dict__[_STATE_KEY] = TIME_REFERENCE_RELATIVE
    else:
        obj.__dict__.pop(_STATE_KEY, None)


def inherit_time_reference(result: Any, source: object) -> Any:
    """Copy ``source``'s state onto ``result`` (a derived object) and return it."""
    if isinstance(result, TimeReferenceMixin) and source is not None:
        set_time_reference(result, get_time_reference(source))
    return result


def _inherit_into(result: Any, source: object) -> Any:
    if isinstance(result, (tuple, list)):
        for item in result:
            inherit_time_reference(item, source)
    elif isinstance(result, dict):
        for item in result.values():
            inherit_time_reference(item, source)
    else:
        inherit_time_reference(result, source)
    return result


def inherits_time_reference(func: _F) -> _F:
    """Decorate a method so its returned series keep the first argument's state.

    For methods that rebuild their result from raw arrays with an explicit
    ``x0``/``t0`` (which would otherwise read as an absolute GPS time).
    Tuples, lists and dicts of series are handled one level deep.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        return _inherit_into(result, args[0] if args else None)

    return wrapper  # type: ignore[return-value]


def rebuild_with_time_reference(
    builder: Callable[..., Any], args: tuple[Any, ...], state: str
) -> Any:
    """Unpickle helper: build the object, then restore its state."""
    obj = builder(*args)
    set_time_reference(obj, state)
    return obj


class TimeReferenceMixin:
    """Read-only ``time_reference`` property that survives GWpy metadata copies.

    GWpy rebuilds results of ``resample``, the filters, ``pad`` and similar
    methods through ``__metadata_finalize__``; this mixin carries the state
    through that hook.
    """

    def with_epoch(self, epoch: Any) -> Any:
        """Return a copy anchored at GPS ``epoch``.

        The existing relative coordinates are offsets from the supplied GPS
        origin, including offsets retained by slices. Values and sampling
        interval are unchanged. The method accepts relative objects only;
        shifting an already absolute object requires an explicit other operation.

        Warn when the interval is below ten float64 ULPs at the anchor: at
        that scale materialized GPS coordinates can have appreciable rounding.
        """
        from gwpy.time import to_gps

        from gwexpy.types.seriesmatrix_base import SeriesMatrix

        if get_time_reference(self) != TIME_REFERENCE_RELATIVE:
            raise ValueError("with_epoch requires a relative time reference")
        if classify_time_input(epoch) != "finite":
            raise ValueError("epoch must be a finite GPS time")
        value = epoch.to_value(u.s) if isinstance(epoch, u.Quantity) else epoch
        anchor = float(to_gps(value))
        if not np.isfinite(anchor):
            raise ValueError("epoch must be a finite GPS time")
        previous_epoch = getattr(self, "epoch", None)
        if previous_epoch is None:
            offset = 0.0
        elif isinstance(previous_epoch, u.Quantity):
            offset = float(previous_epoch.to_value(u.s))
        else:
            offset = float(getattr(previous_epoch, "gps", previous_epoch))
        if hasattr(self, "times"):
            try:
                interval = self.dt  # type: ignore[attr-defined]
                if hasattr(interval, "to_value"):
                    try:
                        dt = float(interval.to_value(u.s))
                    except u.UnitConversionError:
                        interval_unit = getattr(interval, "unit", None)
                        if interval_unit is None or not interval_unit.is_equivalent(
                            u.dimensionless_unscaled
                        ):
                            raise
                        # SpectrogramMatrix accepts bare numeric time axes;
                        # with_epoch interprets those coordinates as seconds.
                        dt = float(interval.to_value(u.dimensionless_unscaled))
                else:
                    dt = float(interval)
            except (AttributeError, IndexError, TypeError, ValueError):
                pass
            else:
                ulp = abs(float(np.spacing(anchor)))
                if abs(dt) < 10 * ulp:
                    warnings.warn(
                        f"dt={dt:g} s is below 10 ULP of epoch ({ulp:g} s); "
                        "absolute time coordinates may be rounded",
                        UserWarning,
                        stacklevel=2,
                    )
        result = self.copy()  # type: ignore[attr-defined]
        if isinstance(self, SeriesMatrix):
            if hasattr(self, "times"):
                try:
                    spacing = self.dt  # type: ignore[attr-defined]
                except (AttributeError, IndexError):
                    spacing = None
                axis: Any = self.xindex
                if isinstance(axis, u.Quantity):
                    if axis.unit.is_equivalent(u.dimensionless_unscaled):
                        # Bare and dimensionless time coordinates are
                        # interpreted as seconds by SpectrogramMatrix.
                        shifted_axis = (
                            axis.to_value(u.dimensionless_unscaled) * u.s + anchor * u.s
                        )
                    else:
                        shifted_axis = axis + anchor * u.s
                else:
                    shifted_axis = axis + anchor
                result.xindex = shifted_axis
                if spacing is not None:
                    if (
                        isinstance(axis, u.Quantity)
                        and axis.unit.is_equivalent(u.dimensionless_unscaled)
                        and isinstance(spacing, u.Quantity)
                        and spacing.unit.is_equivalent(u.dimensionless_unscaled)
                    ):
                        spacing = u.Quantity(
                            spacing.to_value(u.dimensionless_unscaled), u.s
                        )
                    result._dx = spacing
            # TimeSeriesMatrix and SpectrogramMatrix store time coordinates
            # directly on xindex; anchoring shifts those coordinates by anchor,
            # so epoch is anchor (not anchor + offset).
            result.epoch = anchor if hasattr(self, "times") else anchor + offset
        elif hasattr(self, "times"):
            index = self.xindex  # type: ignore[attr-defined]
            if not getattr(index, "regular", True):
                result.xindex = index + anchor * u.s
            else:
                result.x0 = self.x0 + anchor * u.s  # type: ignore[attr-defined]
        else:
            result.epoch = anchor + offset
        set_time_reference(result, TIME_REFERENCE_ABSOLUTE)
        return result

    @property
    def time_reference(self) -> str:
        """``"absolute"`` (``t0`` is a GPS time) or ``"relative"`` (unknown).

        ``t0 == 0`` means GPS 0 only when this is ``"absolute"``.
        """
        return get_time_reference(self)

    #: Private attributes that hold the time-axis origin.  While none of them is
    #: set the object's timing has not been decided yet, so GWpy's metadata copy
    #: (which fills exactly those attributes from ``obj``) must bring the state
    #: along, even if the constructor tentatively marked it relative.
    _time_reference_axis_attrs: tuple[str, ...] = ("_x0", "_xindex")

    def __metadata_finalize__(self, obj: Any, *, force: bool = False) -> None:
        timing_undecided = not any(
            attr in self.__dict__ for attr in self._time_reference_axis_attrs
        )
        super().__metadata_finalize__(obj, force=force)  # type: ignore[misc]
        if force or timing_undecided:
            inherit_time_reference(self, obj)
