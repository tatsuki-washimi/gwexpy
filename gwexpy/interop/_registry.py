"""Converter registry for breaking circular dependencies.

This module provides a lightweight registry that allows modules to look up
concrete classes (e.g., TimeSeries, FrequencySeries) without importing
them directly, breaking circular import chains.

Classes are registered at module load time in each subpackage's __init__.py.
"""

from __future__ import annotations

from typing import Any


class ConverterRegistry:
    """Registry mapping string names to concrete classes and converter functions.

    This avoids circular imports by allowing modules to look up classes
    by name instead of importing them directly.
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
                f"Available: {sorted(cls._constructors)}"
            ) from None

    @classmethod
    def register_converter(cls, name: str, func: Any) -> None:
        """Register a converter function by name."""
        cls._converters[name] = func

    @classmethod
    def get_converter(cls, name: str) -> Any:
        """Retrieve a registered converter function by name."""
        try:
            return cls._converters[name]
        except KeyError:
            raise KeyError(
                f"Converter {name!r} not registered. "
                f"Available: {sorted(cls._converters)}"
            ) from None

    @classmethod
    def has_constructor(cls, name: str) -> bool:
        """Check if a constructor is registered."""
        return name in cls._constructors

    @classmethod
    def has_converter(cls, name: str) -> bool:
        """Check if a converter is registered."""
        return name in cls._converters
