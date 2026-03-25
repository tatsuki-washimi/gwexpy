"""Converter registry for breaking circular dependencies.

This module provides a lightweight registry that allows modules to look up
concrete classes (e.g., TimeSeries, FrequencySeries) without importing
them directly, breaking circular import chains.

Classes are registered at module load time in each subpackage's ``__init__.py``.

Usage
-----
**Registration** (in subpackage ``__init__.py``)::

    from gwexpy.interop._registry import ConverterRegistry as _CR
    from .timeseries import TimeSeries

    _CR.register_constructor("TimeSeries", TimeSeries)

**Lookup** (in any module that needs the class)::

    from gwexpy.interop._registry import ConverterRegistry

    def my_transform(self):
        FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
        return FrequencySeries(data, frequencies=freqs)

See Also
--------
docs/developers/guides/coding_standards.md : Section 6 for full guidelines.
"""

from __future__ import annotations

import warnings
from typing import Any


class ConverterRegistry:
    """Registry mapping string names to concrete classes and converter functions.

    This avoids circular imports by allowing modules to look up classes
    by name instead of importing them directly.

    The registry is a class-level singleton — all state lives on the class
    itself (``_constructors``, ``_converters``), so no instantiation is needed.

    Notes
    -----
    * **Thread safety**: Registration happens at import time (module ``__init__``),
      which is serialized by the GIL. Lookups are dict reads and are safe.
    * **Error messages**: ``get_constructor`` / ``get_converter`` list all
      available keys on ``KeyError`` for easy debugging.
    """

    _constructors: dict[str, type] = {}
    _converters: dict[str, Any] = {}

    @classmethod
    def register_constructor(cls, name: str, klass: type) -> None:
        """Register a concrete class by name.

        Parameters
        ----------
        name : str
            Lookup key (e.g., ``"TimeSeries"``, ``"FrequencySeriesDict"``).
        klass : type
            The concrete class to register.
        """
        existing = cls._constructors.get(name)
        if existing is not None and existing is not klass:
            warnings.warn(
                f"ConverterRegistry: overwriting constructor {name!r} "
                f"({existing!r} -> {klass!r})",
                stacklevel=2,
            )
        cls._constructors[name] = klass

    @classmethod
    def get_constructor(cls, name: str) -> type:
        """Retrieve a registered class by name.

        Parameters
        ----------
        name : str
            Lookup key.

        Returns
        -------
        type
            The registered class.

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        try:
            return cls._constructors[name]
        except KeyError:
            raise KeyError(
                f"Constructor {name!r} not registered. "
                f"Available: {sorted(cls._constructors)}. "
                f"Hint: call gwexpy.register_all() to ensure all "
                f"constructors are loaded."
            ) from None

    @classmethod
    def register_converter(cls, name: str, func: Any) -> None:
        """Register a converter function by name."""
        existing = cls._converters.get(name)
        if existing is not None and existing is not func:
            warnings.warn(
                f"ConverterRegistry: overwriting converter {name!r} "
                f"({existing!r} -> {func!r})",
                stacklevel=2,
            )
        cls._converters[name] = func

    @classmethod
    def get_converter(cls, name: str) -> Any:
        """Retrieve a registered converter function by name."""
        try:
            return cls._converters[name]
        except KeyError:
            raise KeyError(
                f"Converter {name!r} not registered. "
                f"Available: {sorted(cls._converters)}. "
                f"Hint: call gwexpy.register_all() to ensure all "
                f"converters are loaded."
            ) from None

    @classmethod
    def has_constructor(cls, name: str) -> bool:
        """Check if a constructor is registered."""
        return name in cls._constructors

    @classmethod
    def has_converter(cls, name: str) -> bool:
        """Check if a converter is registered."""
        return name in cls._converters
