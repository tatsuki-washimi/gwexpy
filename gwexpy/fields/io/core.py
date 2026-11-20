"""Shared Field I/O dispatch, format identification, and discovery."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from astropy.io.registry.base import IORegistryError
from gwpy.io.registry import default_registry as io_registry

from gwexpy.io.utils import ensure_dependency

from ._registration import get_field_lister

if TYPE_CHECKING:
    from ..collections import FieldDict
    from ..scalar import ScalarField
    from ..tensor import TensorField
    from ..vector import VectorField

__all__ = [
    "list_fields",
    "read_scalarfield",
    "read_vectorfield",
    "read_tensorfield",
    "read_fielddict",
    "write_scalarfield",
    "write_vectorfield",
    "write_tensorfield",
    "write_fielddict",
]

_FORBIDDEN_COMMON_KWARGS = frozenset(
    {
        "field",
        "fields",
        "profile",
        "times",
        "grid_mode",
        "conversion",
        "representation",
        "target_grid",
        "outside",
        "alignment",
        "selectors",
    }
)

_TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
_GEOTIFF_FORMATS = frozenset({"geotiff", "tif", "tiff"})


def _reject_forbidden_kwargs(kwargs: dict[str, Any]) -> None:
    bad = sorted(k for k in kwargs if k in _FORBIDDEN_COMMON_KWARGS)
    if bad:
        raise TypeError(
            f"Unsupported Field I/O keyword argument(s): {', '.join(bad)}. "
            "Pass selector name/names as the second positional argument, "
            "'format=' for format selection, and 'start='/'end=' for axis-0 ranges."
        )


def _first_probe_source(source: Any) -> Any:
    if isinstance(source, (str, bytes, os.PathLike)) or hasattr(source, "read"):
        return source
    if isinstance(source, Sequence):
        if len(source) == 0:
            raise ValueError("Input source sequence must not be empty.")
        return source[0]
    return source


def _split_path_and_fileobj(probe: Any) -> tuple[str | None, Any]:
    if isinstance(probe, (str, os.PathLike)):
        return os.fspath(probe), None
    if hasattr(probe, "attrs") and hasattr(probe, "file"):
        # h5py.File or h5py.Group
        filename = getattr(probe.file, "filename", None)
        path_str = (
            os.fspath(filename) if isinstance(filename, (str, os.PathLike)) else None
        )
        return path_str, probe
    if hasattr(probe, "read"):
        name = getattr(probe, "name", None)
        path_str = os.fspath(name) if isinstance(name, (str, os.PathLike)) else None
        return path_str, probe
    return None, None


def _looks_like_geotiff(probe: Any, *, allow_file_read: bool = True) -> bool:
    path_str, fileobj = _split_path_and_fileobj(probe)
    if path_str is not None and Path(path_str).suffix.lower() in (".tif", ".tiff"):
        return True
    if not allow_file_read:
        return False
    try:
        if fileobj is not None and hasattr(fileobj, "seek"):
            pos = fileobj.tell()
            head = fileobj.read(4)
            fileobj.seek(pos)
            return head in _TIFF_MAGICS
        if path_str is not None and os.path.isfile(path_str):
            with open(path_str, "rb") as fh:
                return fh.read(4) in _TIFF_MAGICS
    except OSError:
        pass
    return False


def _ensure_geotiff_if_requested(format_name: str | None, probe: Any) -> None:
    if format_name in _GEOTIFF_FORMATS:
        ensure_dependency("rasterio", extra="geospatial")
        return
    if format_name is None and _looks_like_geotiff(probe):
        ensure_dependency("rasterio", extra="geospatial")


def _resolve_read_format(
    source: Any,
    target_classes: tuple[type[Any], ...],
    format_name: str | None,
    kwargs: dict[str, Any],
) -> str:
    probe = _first_probe_source(source)
    if format_name is not None:
        if format_name in _GEOTIFF_FORMATS:
            ensure_dependency("rasterio", extra="geospatial")
        return format_name

    path_str, fileobj = _split_path_and_fileobj(probe)
    matched: list[str] = []
    for cls in target_classes:
        try:
            candidates = io_registry.identify_format(
                "read", cls, path_str, fileobj, (), kwargs
            )
        except Exception:
            candidates = []
        for fmt in candidates:
            if fmt not in matched:
                matched.append(fmt)

    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise ValueError(
            f"Ambiguous format for {source!r}: {matched}. Please specify 'format='."
        )

    _ensure_geotiff_if_requested(None, probe)
    raise ValueError(
        f"Could not identify format for {source!r}. Please specify 'format' argument."
    )


def _call_with_accepted_kwargs(
    fn: Any,
    source: Any,
    *,
    name_arg: Any = None,
    has_name_arg: bool = False,
    start: Any = None,
    end: Any = None,
    kwargs: dict[str, Any],
) -> tuple[Any, bool, bool]:
    """Call *fn* forwarding only selector/range arguments accepted by its signature.

    Returns ``(result, name_forwarded, range_forwarded)``.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    call_kwargs = dict(kwargs)
    name_forwarded = False
    range_forwarded = False

    if has_name_arg and name_arg is not None:
        if "name" in params:
            call_kwargs["name"] = name_arg
            name_forwarded = True
        elif "names" in params:
            call_kwargs["names"] = name_arg
            name_forwarded = True
        elif accepts_varkw:
            call_kwargs["name"] = name_arg
            name_forwarded = True

    if start is not None or end is not None:
        if "start" in params and "end" in params:
            call_kwargs["start"] = start
            call_kwargs["end"] = end
            range_forwarded = True

    return fn(source, **call_kwargs), name_forwarded, range_forwarded


def _apply_axis0_range(field: Any, start: Any, end: Any) -> Any:
    if start is None and end is None:
        return field
    ax0 = field._axis0_index
    ax0_unit = getattr(ax0, "unit", u.dimensionless_unscaled)
    vals = np.asarray(ax0.value, dtype=float)
    mask = np.ones(vals.shape, dtype=bool)
    if start is not None:
        s_val = (
            float(start.to_value(ax0_unit))
            if isinstance(start, u.Quantity)
            else float(start)
        )
        mask &= vals >= s_val
    if end is not None:
        e_val = (
            float(end.to_value(ax0_unit)) if isinstance(end, u.Quantity) else float(end)
        )
        mask &= vals <= e_val
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(
            f"No axis-0 samples fall within requested range start={start!r}, end={end!r}."
        )
    if np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        return field[indices[0] : indices[-1] + 1, :, :, :]
    raise ValueError("Axis-0 range selection requires contiguous samples.")


def _crop_field(field: Any, start: Any, end: Any) -> Any:
    from ..scalar import ScalarField
    from ..tensor import TensorField
    from ..vector import VectorField

    if isinstance(field, ScalarField):
        return _apply_axis0_range(field, start, end)
    if isinstance(field, (VectorField, TensorField)):
        components = {
            key: _apply_axis0_range(component, start, end)
            for key, component in field.items()
        }
        if isinstance(field, VectorField):
            return VectorField(components, basis=field.basis, validate=False)
        return TensorField(components, rank=field.rank, validate=False)
    raise TypeError(
        f"Axis-0 range selection is unsupported for {type(field).__name__}."
    )


def _order_fields(result: FieldDict, requested: list[str], fmt: str) -> FieldDict:
    from ..collections import FieldDict

    unqualified: dict[str, list[str]] = {}
    if fmt in {"exo", "exo.flow3d"}:
        for key in result:
            if "/" in key:
                unqualified.setdefault(key.rsplit("/", 1)[1], []).append(key)

    ordered = FieldDict()
    for name in requested:
        keys = [name] if name in result else unqualified.get(name, [])
        if not keys:
            raise KeyError(f"Requested field {name!r} missing from reader result.")
        for key in keys:
            ordered[key] = result[key]
    return ordered


def _apply_read_aliases(
    field: ScalarField,
    *,
    fill_missing_opt: Any,
    projected_opt: Any,
) -> ScalarField:
    if fill_missing_opt is not None and fill_missing_opt is not False:
        if fill_missing_opt is True:
            field = field.fill_missing()
        elif isinstance(fill_missing_opt, str):
            field = field.fill_missing(method=fill_missing_opt)  # type: ignore[arg-type]
        elif isinstance(fill_missing_opt, dict):
            field = field.fill_missing(**fill_missing_opt)
        else:
            raise TypeError(
                "fill_missing= alias must be a method string, options dict, bool, or None."
            )

    if projected_opt is not None:
        if isinstance(projected_opt, str):
            field = field.to_projected(projected_opt)
        elif isinstance(projected_opt, dict):
            opts = dict(projected_opt)
            target_crs = opts.pop("target_crs", None) or opts.pop("crs", None)
            if not target_crs:
                raise ValueError(
                    "projected= dict must include 'target_crs' (or 'crs')."
                )
            field = field.to_projected(target_crs, **opts)
        else:
            raise TypeError(
                "projected= alias must be a target CRS string or options dict."
            )
    return field


def list_fields(
    source: Any,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> list[str]:
    """Discover available field names in *source* without loading full arrays."""
    from .. import io as _io  # noqa: F401
    from ..collections import FieldDict
    from ..scalar import ScalarField
    from ..tensor import TensorField
    from ..vector import VectorField

    _reject_forbidden_kwargs(kwargs)
    fmt = _resolve_read_format(
        source, (ScalarField, VectorField, TensorField, FieldDict), format, kwargs
    )
    lister = get_field_lister(fmt)
    if lister is None:
        raise ValueError(
            f"Format {fmt!r} does not register a lightweight field-listing handler."
        )
    return list(lister(source, **kwargs))


def read_scalarfield(
    source: Any,
    name: str | None = None,
    *,
    format: str | None = None,
    start: Any = None,
    end: Any = None,
    **kwargs: Any,
) -> ScalarField:
    """Read a :class:`~gwexpy.fields.ScalarField` from *source*."""
    from .. import io as _io  # noqa: F401
    from ..scalar import ScalarField

    _reject_forbidden_kwargs(kwargs)
    fill_missing_opt = kwargs.pop("fill_missing", None)
    projected_opt = kwargs.pop("projected", None)

    fmt = _resolve_read_format(source, (ScalarField,), format, kwargs)
    reader = io_registry.get_reader(fmt, ScalarField)

    result, name_forwarded, range_forwarded = _call_with_accepted_kwargs(
        reader,
        source,
        name_arg=name,
        has_name_arg=True,
        start=start,
        end=end,
        kwargs=kwargs,
    )
    if not isinstance(result, ScalarField):
        raise TypeError(
            f"Reader for format {fmt!r} returned {type(result).__name__}, "
            "expected ScalarField."
        )
    if not name_forwarded and name is not None:
        result.name = name
    if not range_forwarded and (start is not None or end is not None):
        result = _apply_axis0_range(result, start, end)

    return _apply_read_aliases(
        result,
        fill_missing_opt=fill_missing_opt,
        projected_opt=projected_opt,
    )


def read_vectorfield(
    source: Any,
    name: str | None = None,
    *,
    format: str | None = None,
    start: Any = None,
    end: Any = None,
    **kwargs: Any,
) -> VectorField:
    """Read a :class:`~gwexpy.fields.VectorField` from *source*."""
    from .. import io as _io  # noqa: F401
    from ..vector import VectorField

    _reject_forbidden_kwargs(kwargs)
    fmt = _resolve_read_format(source, (VectorField,), format, kwargs)
    reader = io_registry.get_reader(fmt, VectorField)
    result, _, range_forwarded = _call_with_accepted_kwargs(
        reader,
        source,
        name_arg=name,
        has_name_arg=True,
        start=start,
        end=end,
        kwargs=kwargs,
    )
    if not isinstance(result, VectorField):
        raise TypeError(
            f"Reader for format {fmt!r} returned {type(result).__name__}, "
            "expected VectorField."
        )
    if not range_forwarded and (start is not None or end is not None):
        result = _crop_field(result, start, end)
    return result


def read_tensorfield(
    source: Any,
    name: str | None = None,
    *,
    format: str | None = None,
    start: Any = None,
    end: Any = None,
    **kwargs: Any,
) -> TensorField:
    """Read a :class:`~gwexpy.fields.TensorField` from *source*."""
    from .. import io as _io  # noqa: F401
    from ..tensor import TensorField

    _reject_forbidden_kwargs(kwargs)
    fmt = _resolve_read_format(source, (TensorField,), format, kwargs)
    reader = io_registry.get_reader(fmt, TensorField)
    result, _, range_forwarded = _call_with_accepted_kwargs(
        reader,
        source,
        name_arg=name,
        has_name_arg=True,
        start=start,
        end=end,
        kwargs=kwargs,
    )
    if not isinstance(result, TensorField):
        raise TypeError(
            f"Reader for format {fmt!r} returned {type(result).__name__}, "
            "expected TensorField."
        )
    if not range_forwarded and (start is not None or end is not None):
        result = _crop_field(result, start, end)
    return result


def read_fielddict(
    source: Any,
    names: Sequence[str] | str | None = None,
    *,
    format: str | None = None,
    start: Any = None,
    end: Any = None,
    **kwargs: Any,
) -> FieldDict:
    """Read named physical fields from *source* into an ordered :class:`FieldDict`."""
    from .. import io as _io  # noqa: F401
    from ..collections import FieldDict
    from ..scalar import ScalarField

    _reject_forbidden_kwargs(kwargs)
    fmt = _resolve_read_format(source, (FieldDict, ScalarField), format, kwargs)

    if names is None:
        lister = get_field_lister(fmt)
        if lister is not None:
            avail = list(lister(source, **kwargs))
            raise ValueError(
                f"Field names must be specified for FieldDict.read(). "
                f"Available fields: {avail}"
            )
        raise ValueError("Field names must be specified for FieldDict.read().")

    if isinstance(names, str):
        req_names = [names]
    elif isinstance(names, Iterable):
        req_names = list(names)
    else:
        raise TypeError(
            f"names must be a str or iterable of str, got {type(names).__name__}"
        )

    if any(not isinstance(name, str) or not name for name in req_names):
        raise TypeError("Requested field names must be nonempty strings.")
    if not req_names:
        raise ValueError("FieldDict.read() requires at least one requested field name.")

    try:
        dict_reader = io_registry.get_reader(fmt, FieldDict)
    except IORegistryError:
        dict_reader = None

    if dict_reader is not None:
        result, _, range_forwarded = _call_with_accepted_kwargs(
            dict_reader,
            source,
            name_arg=req_names,
            has_name_arg=True,
            start=start,
            end=end,
            kwargs=kwargs,
        )
        if not isinstance(result, FieldDict):
            raise TypeError(
                f"Reader for format {fmt!r} returned {type(result).__name__}, "
                "expected FieldDict."
            )
        ordered = _order_fields(result, req_names, fmt)
        if not range_forwarded and (start is not None or end is not None):
            ordered = FieldDict(
                (key, _crop_field(field, start, end)) for key, field in ordered.items()
            )
        return ordered

    # Fallback for ScalarField-only formats (e.g., gsi-dem / geotiff)
    scalar_reader = io_registry.get_reader(fmt, ScalarField)
    lister = get_field_lister(fmt)
    available = list(lister(source, **kwargs)) if lister is not None else None
    if available is not None:
        missing = [n for n in req_names if n not in available]
        if missing:
            raise KeyError(
                f"Requested field(s) {missing} not found in {source!r} "
                f"(available: {available})."
            )

    collected: dict[str, Any] = {}
    for req in req_names:
        sf, name_forwarded, range_forwarded = _call_with_accepted_kwargs(
            scalar_reader,
            source,
            name_arg=req,
            has_name_arg=True,
            start=start,
            end=end,
            kwargs=kwargs,
        )
        if not isinstance(sf, ScalarField):
            raise TypeError(f"Reader for format {fmt!r} must return ScalarField.")
        if not name_forwarded:
            sf.name = req
        if not range_forwarded and (start is not None or end is not None):
            sf = _apply_axis0_range(sf, start, end)
        collected[req] = sf
    return FieldDict(collected)


def _write_field_object(
    field: Any,
    cls: type[Any],
    target: Any,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> None:
    from .. import io as _io  # noqa: F401

    probe = target
    path_str, fileobj = _split_path_and_fileobj(probe)
    fmt = format
    if fmt is None:
        try:
            candidates = io_registry.identify_format(
                "write", cls, path_str, fileobj, (), kwargs
            )
        except Exception:
            candidates = []
        if len(candidates) == 1:
            fmt = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"Ambiguous format for {target!r}: {candidates}. Please specify 'format='."
            )
        else:
            if _looks_like_geotiff(probe, allow_file_read=False):
                ensure_dependency("rasterio", extra="geospatial")
            raise ValueError(
                f"Could not identify format for {target!r}. Please specify 'format' argument."
            )
    elif fmt in _GEOTIFF_FORMATS:
        ensure_dependency("rasterio", extra="geospatial")

    writer = io_registry.get_writer(fmt, cls)
    writer(field, target, **kwargs)


def write_scalarfield(
    field: ScalarField,
    target: Any,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a :class:`~gwexpy.fields.ScalarField` to *target*."""
    from ..scalar import ScalarField

    _write_field_object(field, ScalarField, target, format=format, **kwargs)


def write_vectorfield(
    field: VectorField,
    target: Any,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a :class:`~gwexpy.fields.VectorField` to *target*."""
    from ..vector import VectorField

    _write_field_object(field, VectorField, target, format=format, **kwargs)


def write_tensorfield(
    field: TensorField,
    target: Any,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a :class:`~gwexpy.fields.TensorField` to *target*."""
    from ..tensor import TensorField

    _write_field_object(field, TensorField, target, format=format, **kwargs)


def write_fielddict(
    field_dict: FieldDict,
    target: Any,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a :class:`~gwexpy.fields.FieldDict` to *target*."""
    from ..collections import FieldDict

    _write_field_object(field_dict, FieldDict, target, format=format, **kwargs)
