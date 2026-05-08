from __future__ import annotations

from importlib import import_module
from types import ModuleType


def public_names(module: ModuleType) -> list[str]:
    public = getattr(module, "__all__", None)
    if public is not None:
        return list(public)
    return [name for name in dir(module) if not name.startswith("_")]


def bind_gwpy_proxy(namespace: dict[str, object], module_name: str) -> list[str]:
    module = import_module(module_name)
    names = public_names(module)
    namespace.update({name: getattr(module, name) for name in names})
    return names


__all__ = ["bind_gwpy_proxy", "public_names"]
