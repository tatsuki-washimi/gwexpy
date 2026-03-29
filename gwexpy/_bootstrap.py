"""Explicit bootstrap for gwexpy registry initialization.

Normally, ``import gwexpy`` eagerly imports all subpackages, which triggers
constructor and I/O format registration as a side effect.  However, if a
user or tool imports a submodule directly (e.g.,
``from gwexpy.interop._registry import ConverterRegistry``), registrations
from other subpackages will not have run yet.

``register_all()`` forces all subpackage imports so the registry is fully
populated regardless of import order.

Examples
--------
>>> from gwexpy import register_all
>>> register_all()

>>> # Or equivalently, just import the top-level package:
>>> import gwexpy  # register_all() is called automatically
"""

from __future__ import annotations

_bootstrapped = False


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
    if _bootstrapped:
        return

    # Force-import subpackages that register constructors.
    # Python caches modules, so repeated imports are no-ops.
    import gwexpy.frequencyseries  # noqa: F811
    import gwexpy.histogram  # noqa: F811
    import gwexpy.plot  # noqa: F811
    import gwexpy.spectrogram  # noqa: F811
    import gwexpy.timeseries  # noqa: F811
    import gwexpy.types  # noqa: F811

    if include_io:
        # frequencyseries/__init__.py does not import .io,
        # so we must trigger it explicitly here.
        import gwexpy.frequencyseries.io  # noqa: F401
        import gwexpy.timeseries.io  # noqa: F401

    _bootstrapped = True
