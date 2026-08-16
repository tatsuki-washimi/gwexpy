"""Explicit bootstrap for gwexpy registry initialization.

Importing ``gwexpy`` keeps its public subpackages and constructors lazy.  Call
``gwexpy.register_all()`` when constructor and I/O format registration is
needed.  If a user or tool imports a submodule directly (e.g.,
``from gwexpy.interop._registry import ConverterRegistry``), registrations
from other subpackages will not have run yet.

``register_all()`` forces all subpackage imports so the registry is fully
populated regardless of import order.

Examples
--------
>>> from gwexpy import register_all
>>> register_all()

>>> # Importing gwexpy alone does not bootstrap optional registrations:
>>> import gwexpy
>>> gwexpy.register_all()

"""

from __future__ import annotations

from typing import Any

_constructors_bootstrapped = False
_io_bootstrapped = False
_bootstrapped = False


def _ensure_io_registry_compat() -> None:
    """Provide compatibility shims required by older or minimal GWpy builds."""
    try:  # pragma: no cover - defensive compatibility path
        from gwpy.io import registry as registry_module

        registries: list[Any] = [registry_module]
        default_registry = getattr(registry_module, "default_registry", None)
        if default_registry is not None:
            registries.append(default_registry)

        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        for registry in registries:
            for name in ("register_reader", "register_identifier", "register_writer"):
                if not hasattr(registry, name):
                    setattr(registry, name, _noop)
    except (ImportError, AttributeError):
        pass


def _bootstrap_constructors() -> None:
    """Import the packages whose module initializers register constructors."""
    global _constructors_bootstrapped
    if _constructors_bootstrapped:
        return

    import gwexpy.frequencyseries  # noqa: F401
    import gwexpy.histogram  # noqa: F401
    import gwexpy.plot  # noqa: F401
    import gwexpy.spectrogram as spectrogram

    spectrogram._register_constructors()
    import gwexpy.timeseries  # noqa: F401
    import gwexpy.types  # noqa: F401

    _constructors_bootstrapped = True


def _bootstrap_io() -> None:
    """Import all GWexpy-owned I/O registration modules exactly once."""
    global _io_bootstrapped
    if _io_bootstrapped:
        return

    _ensure_io_registry_compat()
    import gwexpy.frequencyseries.io  # noqa: F401
    import gwexpy.spectrogram.io  # noqa: F401
    import gwexpy.timeseries.io  # noqa: F401
    from gwexpy.io.hdf5_sidecar import register_hdf5_sidecars

    register_hdf5_sidecars()

    _io_bootstrapped = True


def ensure_io_registered() -> None:
    """Register GWexpy I/O handlers when a public I/O operation needs them."""
    _bootstrap_io()


def register_all(*, include_io: bool = True) -> None:
    """Ensure all constructors and (optionally) I/O formats are registered.

    This function is idempotent — calling it multiple times is safe and
    effectively free after the first call.

    Parameters
    ----------
    include_io : bool, optional
        If ``True`` (default), also trigger I/O format registration
        (readers, writers, identifiers).  Set to ``False`` to register
        only constructors.

    """
    global _bootstrapped
    _bootstrap_constructors()
    if include_io:
        _bootstrap_io()

    _bootstrapped = _constructors_bootstrapped and _io_bootstrapped
