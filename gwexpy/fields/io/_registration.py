"""I/O registry helpers for GWexpy field classes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from gwpy.io.registry import default_registry as io_registry

__all__ = [
    "register_field_format",
    "get_field_lister",
    "registered_field_formats",
]

_FIELD_LISTERS: dict[str, Callable[..., list[str]]] = {}


def _normalize_extensions(extension: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if extension is None:
        return ()
    if isinstance(extension, str):
        items: tuple[str, ...] = (extension,)
    else:
        items = tuple(extension)
    return tuple(f".{ext.lower().lstrip('.')}" for ext in items)


def get_field_lister(format_name: str) -> Callable[..., list[str]] | None:
    """Return the lightweight field-discovery callable for *format_name*, if any."""
    return _FIELD_LISTERS.get(format_name)


def registered_field_formats() -> tuple[str, ...]:
    """Return canonical/aliased format names registered via :func:`register_field_format`."""
    return tuple(_FIELD_LISTERS.keys())


def register_field_format(
    format_name: str,
    *,
    reader: Callable[..., Any] | None = None,
    writer: Callable[..., Any] | None = None,
    reader_scalar: Callable[..., Any] | None = None,
    reader_vector: Callable[..., Any] | None = None,
    reader_tensor: Callable[..., Any] | None = None,
    reader_dict: Callable[..., Any] | None = None,
    writer_scalar: Callable[..., Any] | None = None,
    writer_vector: Callable[..., Any] | None = None,
    writer_tensor: Callable[..., Any] | None = None,
    writer_dict: Callable[..., Any] | None = None,
    identifier: Callable[..., bool] | None = None,
    magic_identifier: Callable[..., bool] | None = None,
    extension: str | tuple[str, ...] | None = None,
    lister: Callable[..., list[str]] | None = None,
    aliases: tuple[str, ...] = (),
    force: bool = True,
) -> None:
    """Register readers, writers, identifiers, and field listers for Field classes.

    Targets :class:`~gwexpy.fields.ScalarField`,
    :class:`~gwexpy.fields.VectorField`,
    :class:`~gwexpy.fields.TensorField`, and
    :class:`~gwexpy.fields.FieldDict` on ``gwpy.io.registry.default_registry``.
    """
    from ..collections import FieldDict
    from ..scalar import ScalarField
    from ..tensor import TensorField
    from ..vector import VectorField

    if reader is not None and reader_scalar is None:
        reader_scalar = reader
    if writer is not None and writer_scalar is None:
        writer_scalar = writer

    if reader_dict is None and reader_scalar is not None:

        def _read_scalar_dict(
            source: Any,
            names: Iterable[str] | str | None = None,
            *,
            start: Any = None,
            end: Any = None,
            **kwargs: Any,
        ) -> FieldDict:
            """Adapt a scalar-only reader to a selected, ordered FieldDict."""
            from .core import read_scalarfield

            if names is None:
                available = list(lister(source, **kwargs)) if lister is not None else []
                raise ValueError(
                    "Field names must be specified for FieldDict.read(). "
                    f"Available fields: {available}"
                )
            requested = [names] if isinstance(names, str) else list(names)
            if not requested or any(not isinstance(n, str) or not n for n in requested):
                raise ValueError("FieldDict reader requires nonempty field names.")
            if lister is not None:
                available = list(lister(source, **kwargs))
                missing = [name for name in requested if name not in available]
                if missing:
                    raise KeyError(
                        f"Requested field(s) {missing} not found in {source!r} "
                        f"(available: {available})."
                    )
            return FieldDict(
                {
                    name: read_scalarfield(
                        source,
                        name,
                        format=format_name,
                        start=start,
                        end=end,
                        **kwargs,
                    )
                    for name in requested
                }
            )

        reader_dict = _read_scalar_dict

    format_names = (format_name, *aliases)
    extensions = _normalize_extensions(extension)

    readers_by_class: list[tuple[type[Any], Callable[..., Any] | None]] = [
        (ScalarField, reader_scalar),
        (VectorField, reader_vector),
        (TensorField, reader_tensor),
        (FieldDict, reader_dict),
    ]
    writers_by_class: list[tuple[type[Any], Callable[..., Any] | None]] = [
        (ScalarField, writer_scalar),
        (VectorField, writer_vector),
        (TensorField, writer_tensor),
        (FieldDict, writer_dict),
    ]

    active_classes: list[type[Any]] = []
    for cls, fn in readers_by_class:
        if fn is not None:
            active_classes.append(cls)
            for reg_name in format_names:
                io_registry.register_reader(reg_name, cls, fn, force=force)

    for cls, fn in writers_by_class:
        if fn is not None:
            if cls not in active_classes:
                active_classes.append(cls)
            for reg_name in format_names:
                io_registry.register_writer(reg_name, cls, fn, force=force)

    if lister is not None:
        for reg_name in format_names:
            _FIELD_LISTERS[reg_name] = lister

    resolved_identifier = identifier
    if resolved_identifier is None and (magic_identifier is not None or extensions):

        def _combined_identifier(origin, filepath, fileobj, *args, **kwargs):
            if magic_identifier is not None:
                try:
                    if magic_identifier(origin, filepath, fileobj, *args, **kwargs):
                        return True
                except (AttributeError, TypeError, ValueError, OSError, EOFError):
                    pass
            if extensions and filepath is not None:
                lower = str(filepath).lower()
                return any(lower.endswith(ext) for ext in extensions)
            return False

        resolved_identifier = _combined_identifier

    if resolved_identifier is not None:
        for cls in active_classes:
            io_registry.register_identifier(
                format_name, cls, resolved_identifier, force=force
            )
