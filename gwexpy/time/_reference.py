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
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Literal, TypeVar

import numpy as np
from astropy import units as u

__all__ = [
    "TIME_REFERENCE_ABSOLUTE",
    "TIME_REFERENCE_NAME",
    "TIME_REFERENCE_RELATIVE",
    "TimeReferenceMixin",
    "check_time_setter_value",
    "classify_time_input",
    "get_time_reference",
    "inherit_time_reference",
    "inherits_time_reference",
    "rebuild_with_time_reference",
    "set_time_reference",
]

#: Public attribute name of the state (provisional, #725 open question 2).
TIME_REFERENCE_NAME = "time_reference"
TIME_REFERENCE_ABSOLUTE = "absolute"
TIME_REFERENCE_RELATIVE = "relative"

_STATE_KEY = f"_gwex_{TIME_REFERENCE_NAME}"
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
