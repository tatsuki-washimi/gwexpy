"""gwexpy.spectrogram - Spectrogram data containers and operations."""

from __future__ import annotations

import importlib
import sys

__all__ = [
    "Spectrogram",
    "SpectrogramList",
    "SpectrogramDict",
    "SpectrogramMatrix",
]

_EXPORTS = {
    "Spectrogram": (".spectrogram", "Spectrogram"),
    "SpectrogramDict": (".collections", "SpectrogramDict"),
    "SpectrogramList": (".collections", "SpectrogramList"),
    "SpectrogramMatrix": (".matrix", "SpectrogramMatrix"),
}


def _register_constructors() -> None:
    """Load and register all container constructors for explicit bootstrap."""
    from gwexpy.interop._registry import ConverterRegistry

    for name in __all__:
        value = _load_export(name)
        ConverterRegistry.register_constructor(name, value)


def _load_export(name: str):
    module_name, attribute_name = _EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    from gwexpy.interop._registry import ConverterRegistry

    ConverterRegistry.register_constructor(name, value)
    globals()[name] = value
    return value


def __getattr__(name):
    if name in _EXPORTS:
        return _load_export(name)
    # Preserve GWpy's compatibility namespace, but only pay its import cost
    # when an unknown attribute is requested explicitly.
    gwpy_spectrogram = importlib.import_module("gwpy.spectrogram")
    return getattr(gwpy_spectrogram, name)


def __dir__():
    fallback_names = {"connect", "io", "spectrogram"}
    gwpy_spectrogram = sys.modules.get("gwpy.spectrogram")
    if gwpy_spectrogram is not None:
        fallback_names.update(dir(gwpy_spectrogram))
    return sorted(set(__all__) | set(globals()) | fallback_names)
