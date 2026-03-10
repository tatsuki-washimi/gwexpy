"""Mixins and factories for eliminating dict/list delegation boilerplate."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol, cast


class _DictLike(Protocol):
    def items(self) -> Iterable[tuple[Any, Any]]: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...


class _ListLike(Protocol):
    def __iter__(self) -> Iterator[Any]: ...
    def append(self, value: Any) -> None: ...


class DictMapMixin:
    """Mixin providing ``_map_new`` for dict-like collections."""

    def _map_new(self, method_name: str, *args: Any, **kwargs: Any):
        """Apply *method_name* to each value, returning a new collection."""
        this = cast(_DictLike, self)
        new = cast(_DictLike, self.__class__())
        for key, val in this.items():
            new[key] = getattr(val, method_name)(*args, **kwargs)
        return new


class ListMapMixin:
    """Mixin providing ``_map_new_list`` for list-like collections."""

    def _map_new_list(self, method_name: str, *args: Any, **kwargs: Any):
        """Apply *method_name* to each element, returning a new collection."""
        this = cast(_ListLike, self)
        new = cast(_ListLike, self.__class__())
        for item in this:
            new.append(getattr(item, method_name)(*args, **kwargs))
        return new


def _make_dict_map_method(
    method_name: str,
    *,
    doc: str = "",
    result_class_path: str = "",
) -> Any:
    """Create a delegation method for dict-like collections.

    Parameters
    ----------
    result_class_path : str
        Dotted import path for the result collection class
        (e.g. ``"gwexpy.frequencyseries.FrequencySeriesDict"``).
        Uses deferred import to avoid circular dependencies.
        If empty, uses ``self.__class__()`` (same type as input).
    """
    if result_class_path:

        def method(self, *args, **kwargs):
            import importlib

            mod_path, cls_name = result_class_path.rsplit(".", 1)
            cls = getattr(importlib.import_module(mod_path), cls_name)
            new = cls()
            for key, val in self.items():
                new[key] = getattr(val, method_name)(*args, **kwargs)
            return new

    else:

        def method(self, *args, **kwargs):
            return self._map_new(method_name, *args, **kwargs)

    method.__name__ = method_name
    method.__qualname__ = method_name
    method.__doc__ = doc or f"Apply ``{method_name}`` to each element."
    return method


def _make_dict_plain_method(method_name: str, *, doc: str = "") -> Any:
    """Create a delegation method returning a plain dict (not a collection)."""

    def method(self, *args, **kwargs):
        return {
            key: getattr(val, method_name)(*args, **kwargs)
            for key, val in self.items()
        }

    method.__name__ = method_name
    method.__qualname__ = method_name
    method.__doc__ = doc or f"Apply ``{method_name}`` to each element."
    return method


def _make_list_map_method(
    method_name: str,
    *,
    doc: str = "",
    result_class_path: str = "",
) -> Any:
    """Create a delegation method for list-like collections.

    Parameters
    ----------
    result_class_path : str
        Dotted import path for the result collection class.
        Uses deferred import to avoid circular dependencies.
        If empty, uses ``self.__class__()`` (same type as input).
    """
    if result_class_path:

        def method(self, *args, **kwargs):
            import importlib

            mod_path, cls_name = result_class_path.rsplit(".", 1)
            cls = getattr(importlib.import_module(mod_path), cls_name)
            new = cls()
            for item in self:
                new.append(getattr(item, method_name)(*args, **kwargs))
            return new

    else:

        def method(self, *args, **kwargs):
            return self._map_new_list(method_name, *args, **kwargs)

    method.__name__ = method_name
    method.__qualname__ = method_name
    method.__doc__ = doc or f"Apply ``{method_name}`` to each element."
    return method
