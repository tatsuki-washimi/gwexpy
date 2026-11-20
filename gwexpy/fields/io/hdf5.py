"""Canonical HDF5 reader/writer for GWexpy Field classes (#520, #658).

Implements the U-4 / #658 on-disk schema versioning and unknown-field policy
for :class:`~gwexpy.fields.ScalarField`, :class:`~gwexpy.fields.VectorField`,
:class:`~gwexpy.fields.TensorField`, and :class:`~gwexpy.fields.FieldDict`:

- Root attributes ``gwexpy_schema`` (``"gwexpy.field"`` or
  ``"gwexpy.field_dict"``) and ``gwexpy_schema_version`` (``1``) distinguish
  Field HDF5 files from ``SegmentTable`` (``"gwexpy.segment_table"``),
  GWDama (``"gwexpy.hdf.gwdama"``), and GWpy Series HDF5 files.
- A bare ``.h5`` / ``.hdf5`` extension does not uniquely identify a file on
  read: content identification checks ``gwexpy_schema`` (or an explicitly
  supported unversioned legacy Field layout containing ``data`` and
  ``axes/axis0..axis3``).
- Unknown attributes are partitioned via
  :func:`gwexpy.io.schema.classify_unknown_attrs`: unknown ``gwexpy_*`` /
  ``_gwexpy_*`` attributes emit a ``UserWarning`` (or raise ``ValueError``
  when ``strict_schema=True``), whereas external (non-GWexpy) attributes are
  preserved on ``field._gwex_hdf5_external_attrs`` for lossless round-trip.
- Lossless JSON provenance contract:
  - :class:`~gwexpy.fields.geo.GeoMetadata` (``_gwex_geo``) is reconstructed
    as a typed frozen dataclass.
  - Known simulation provenance dataclasses (``OpenFOAMProvenance``,
    ``ExodusProvenance``, ``Flow3DProvenance``, ``Specfem3DProvenance``, plus
    legacy ``*Metadata`` aliases) are
    reconstructed with their exact frozen dataclass types (including auxiliary
    boolean/float NumPy mask arrays stored in a dedicated ``provenance_arrays``
    subgroup) when their interop module is available, or as tagged dictionaries
    otherwise.
  - Other ``_gwex_*`` provenance attributes containing JSON-representable
    primitives (``None``, ``bool``, ``int``, ``float``, ``str``), ``tuple``,
    ``list``, ``dict``, 1D/ND ``numpy.ndarray``, or ``astropy.units.Quantity``
    are round-tripped losslessly with explicit type tags. Arbitrary Python
    objects outside this set fail closed with ``TypeError`` on write unless
    ``skip_unsupported_provenance=True``.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import h5py
import numpy as np
from astropy import units as u

from gwexpy.io.schema import (
    SCHEMA_ATTR,
    SCHEMA_ATTRS,
    SCHEMA_VERSION_ATTR,
    classify_unknown_attrs,
    read_and_validate_schema,
    write_schema_attrs,
)
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
    get_time_reference,
    set_time_reference,
)

from ..collections import FieldDict
from ..geo import GeoMetadata
from ..scalar import ScalarField
from ..tensor import TensorField
from ..vector import VectorField
from ._registration import register_field_format

__all__ = [
    "FIELD_DICT_SCHEMA_NAME",
    "FIELD_SCHEMA_NAME",
    "FIELD_SCHEMA_VERSION",
    "MAX_SUPPORTED_FIELD_SCHEMA_VERSION",
    "identify_field_hdf5",
    "list_fields_hdf5",
    "read_fielddict_hdf5",
    "read_scalarfield_hdf5",
    "read_tensorfield_hdf5",
    "read_vectorfield_hdf5",
    "register_hdf5_field_formats",
    "write_fielddict_hdf5",
    "write_scalarfield_hdf5",
    "write_tensorfield_hdf5",
    "write_vectorfield_hdf5",
]

FIELD_SCHEMA_NAME = "gwexpy.field"
FIELD_DICT_SCHEMA_NAME = "gwexpy.field_dict"
FIELD_SCHEMA_VERSION = 1
MAX_SUPPORTED_FIELD_SCHEMA_VERSION = 1

_EXTERNAL_ATTRS_SLOT = "_gwex_hdf5_external_attrs"
_INTERNAL_TRANSIENT_GWEX_ATTRS = frozenset({_EXTERNAL_ATTRS_SLOT})

_KNOWN_ROOT_ATTRS = SCHEMA_ATTRS | frozenset(
    {
        "gwexpy_field_kind",
        "gwexpy_default_name",
        "gwexpy_field_order",
    }
)

_KNOWN_SCALAR_GROUP_ATTRS = SCHEMA_ATTRS | frozenset(
    {
        "gwexpy_field_kind",
        "gwexpy_name",
        "gwexpy_channel",
        "gwexpy_unit",
        "gwexpy_epoch",
        "gwexpy_has_epoch",
        "gwexpy_time_reference",
        "gwexpy_axis0_domain",
        "gwexpy_axis_names",
        "gwexpy_space_domains",
        "gwexpy_axis0_offset_val",
        "gwexpy_axis0_offset_unit",
        "gwexpy_axis0_dt_val",
        "gwexpy_axis0_dt_unit",
        "gwexpy_geo_json",
        "gwexpy_provenance_json",
    }
)

_KNOWN_COMPOUND_GROUP_ATTRS = SCHEMA_ATTRS | frozenset(
    {
        "gwexpy_field_kind",
        "gwexpy_basis",
        "gwexpy_rank",
        "gwexpy_component_order",
    }
)

_KNOWN_DATACLASS_REGISTRY: dict[str, tuple[str, str]] = {
    "GeoMetadata": ("gwexpy.fields.geo", "GeoMetadata"),
    "OpenFOAMProvenance": ("gwexpy.interop._openfoam.model", "OpenFOAMProvenance"),
    "ExodusProvenance": ("gwexpy.interop._exodus.model", "ExodusProvenance"),
    "Flow3DProvenance": ("gwexpy.interop._exodus.model", "Flow3DProvenance"),
    "Specfem3DProvenance": ("gwexpy.interop._specfem3d.model", "Specfem3DProvenance"),
    # Backward-compatible aliases
    "OpenFOAMMetadata": ("gwexpy.interop._openfoam.model", "OpenFOAMProvenance"),
    "ExodusMetadata": ("gwexpy.interop._exodus.model", "ExodusProvenance"),
    "Flow3DMetadata": ("gwexpy.interop._exodus.model", "Flow3DProvenance"),
    "Specfem3DFieldMetadata": (
        "gwexpy.interop._specfem3d.model",
        "Specfem3DProvenance",
    ),
}


def _decode_str(val: Any) -> str:
    if isinstance(val, np.ndarray) and val.ndim == 0:
        val = val.item()
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    return str(val)


def _detach_attr_value(val: Any) -> Any:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray):
        return np.array(val, copy=True)
    if isinstance(val, np.generic):
        return val.item()
    return val


@contextmanager
def _open_h5(target: Any, mode: str):
    if isinstance(target, (h5py.File, h5py.Group)):
        yield target
        return
    if isinstance(target, (str, os.PathLike)):
        with h5py.File(os.fspath(target), mode) as h5f:
            yield h5f
        return
    if hasattr(target, "read") or hasattr(target, "write"):
        with h5py.File(target, mode) as h5f:
            yield h5f
        return
    raise TypeError(
        f"Unsupported HDF5 source/target type {type(target).__name__}; "
        "expected path, file-like object, h5py.File, or h5py.Group."
    )


def _is_legacy_scalar_group(grp: h5py.Group) -> bool:
    """Return True if *grp* has the known unversioned Field dataset+axes layout."""
    if "data" not in grp or not isinstance(grp["data"], h5py.Dataset):
        return False
    if "axes" not in grp or not isinstance(grp["axes"], h5py.Group):
        return False
    axes_grp = grp["axes"]
    return all(f"axis{i}" in axes_grp for i in range(4))


def _is_legacy_field_container(h5obj: h5py.File | h5py.Group) -> bool:
    if _is_legacy_scalar_group(h5obj):
        return True
    if "fields" in h5obj and isinstance(h5obj["fields"], h5py.Group):
        fgrp = h5obj["fields"]
        if len(fgrp) > 0:
            return all(
                isinstance(sub, h5py.Group)
                and (_is_legacy_scalar_group(sub) or "components" in sub)
                for sub in fgrp.values()
            )
    return False


def identify_field_hdf5(
    origin: str,
    filepath: str | None,
    fileobj: Any,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Identify Field HDF5 format without claiming non-Field ``.h5`` files.

    A bare ``.h5`` / ``.hdf5`` extension does *not* identify a file as a Field
    HDF5 container on ``read``: we inspect the HDF5 root ``gwexpy_schema``
    attribute (or verify the known unversioned legacy Field layout). On
    ``write`` with an ``h5py.File`` or ``h5py.Group`` target, we return True;
    file-path writes require explicit ``format='hdf5'``.
    """
    if origin == "write":
        return isinstance(fileobj, (h5py.File, h5py.Group))

    probe = fileobj if fileobj is not None else filepath
    if probe is None:
        return False

    try:
        if isinstance(probe, (h5py.File, h5py.Group)):
            return _check_h5_group_is_field(probe)
        if isinstance(probe, (str, os.PathLike)):
            path_str = os.fspath(probe)
            if not os.path.isfile(path_str) or not h5py.is_hdf5(path_str):
                return False
            with h5py.File(path_str, "r") as h5f:
                return _check_h5_group_is_field(h5f)
        if hasattr(probe, "read") and hasattr(probe, "seek"):
            pos = probe.tell()
            try:
                with h5py.File(probe, "r") as h5f:
                    return _check_h5_group_is_field(h5f)
            finally:
                probe.seek(pos)
    except Exception:
        return False
    return False


def _check_h5_group_is_field(h5obj: h5py.File | h5py.Group) -> bool:
    if SCHEMA_ATTR in h5obj.attrs:
        schema = _decode_str(h5obj.attrs[SCHEMA_ATTR])
        return schema in (FIELD_SCHEMA_NAME, FIELD_DICT_SCHEMA_NAME)
    if SCHEMA_VERSION_ATTR in h5obj.attrs:
        # Malformed versioned file: let the reader fail closed if explicitly
        # invoked or if layout matches Field.
        return _is_legacy_field_container(h5obj)
    return _is_legacy_field_container(h5obj)


# ---------------------------------------------------------------------------
# Provenance & GeoMetadata JSON encoding/decoding
# ---------------------------------------------------------------------------


def _encode_geo(geo: GeoMetadata) -> str:
    payload = {
        "crs": geo.crs,
        "transform": list(geo.transform) if geo.transform is not None else None,
        "vertical_datum": geo.vertical_datum,
        "nodata": geo.nodata,
        "band": geo.band,
        "source": list(geo.source),
        "source_format": geo.source_format,
        "active": bool(geo.active),
        "local_origin": (
            list(geo.local_origin) if geo.local_origin is not None else None
        ),
        "history": list(geo.history),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _decode_geo(raw: str) -> GeoMetadata:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Invalid gwexpy_geo_json payload: expected JSON object.")
    tf = data.get("transform")
    lo = data.get("local_origin")
    src = data.get("source", ())
    hist = data.get("history", ())
    return GeoMetadata(
        crs=data.get("crs"),
        transform=tuple(float(x) for x in tf) if tf is not None else None,
        vertical_datum=str(data.get("vertical_datum", "geoid")),
        nodata=float(data["nodata"]) if data.get("nodata") is not None else None,
        band=int(data["band"]) if data.get("band") is not None else None,
        source=tuple(str(x) for x in src),
        source_format=data.get("source_format"),
        active=bool(data.get("active", True)),
        local_origin=(
            (float(lo[0]), float(lo[1])) if lo is not None and len(lo) == 2 else None
        ),
        history=tuple(str(x) for x in hist),
    )


def _encode_provenance_value(
    val: Any,
    arrays_out: dict[str, np.ndarray],
    path_prefix: str,
) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, u.Quantity):
        arr_val = np.asarray(val.value)
        if arr_val.ndim == 0:
            return {
                "__gwexpy_type__": "quantity_scalar",
                "value": float(arr_val.item()),
                "unit": str(val.unit),
            }
        arr_key = f"arr_{len(arrays_out)}_{path_prefix}"
        arrays_out[arr_key] = np.array(arr_val, copy=True)
        return {
            "__gwexpy_type__": "quantity_array",
            "dataset": arr_key,
            "unit": str(val.unit),
        }
    if isinstance(val, u.UnitBase):
        return {
            "__gwexpy_type__": "unit",
            "unit": str(val),
        }
    if isinstance(val, np.ndarray):
        arr_key = f"arr_{len(arrays_out)}_{path_prefix}"
        arrays_out[arr_key] = np.array(val, copy=True)
        return {
            "__gwexpy_type__": "ndarray",
            "dataset": arr_key,
        }
    if isinstance(val, tuple):
        return {
            "__gwexpy_type__": "tuple",
            "items": [
                _encode_provenance_value(item, arrays_out, f"{path_prefix}_{i}")
                for i, item in enumerate(val)
            ],
        }
    if isinstance(val, list):
        return [
            _encode_provenance_value(item, arrays_out, f"{path_prefix}_{i}")
            for i, item in enumerate(val)
        ]
    if isinstance(val, dict):
        if any(not isinstance(k, str) for k in val):
            raise TypeError(
                "Provenance dictionary keys must be strings for lossless JSON serialization."
            )
        return {
            k: _encode_provenance_value(v, arrays_out, f"{path_prefix}_{k}")
            for k, v in val.items()
        }
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        cls_name = type(val).__name__
        if cls_name not in _KNOWN_DATACLASS_REGISTRY:
            raise TypeError(
                f"Provenance dataclass {cls_name!r} is not in the registered "
                "lossless Field HDF5 provenance set."
            )
        fields_dict = {}
        for f in dataclasses.fields(val):
            fields_dict[f.name] = _encode_provenance_value(
                getattr(val, f.name),
                arrays_out,
                f"{path_prefix}_{f.name}",
            )
        return {
            "__gwexpy_type__": "dataclass",
            "class_name": cls_name,
            "module": type(val).__module__,
            "fields": fields_dict,
        }
    raise TypeError(
        f"Provenance value of type {type(val).__name__!r} is not covered by the "
        "Field HDF5 lossless JSON provenance contract."
    )


def _decode_provenance_value(
    raw: Any,
    arrays_group: h5py.Group | None,
) -> Any:
    if raw is None or isinstance(raw, (bool, int, float, str)):
        return raw
    if isinstance(raw, list):
        return [_decode_provenance_value(item, arrays_group) for item in raw]
    if isinstance(raw, dict):
        tag = raw.get("__gwexpy_type__")
        if tag == "tuple":
            return tuple(
                _decode_provenance_value(item, arrays_group) for item in raw["items"]
            )
        if tag == "quantity_scalar":
            return float(raw["value"]) * u.Unit(raw["unit"])
        if tag == "quantity_array":
            if arrays_group is None or raw["dataset"] not in arrays_group:
                raise ValueError(
                    f"Missing provenance array dataset {raw.get('dataset')!r}."
                )
            arr = np.array(arrays_group[raw["dataset"]][()], copy=True)
            return arr * u.Unit(raw["unit"])
        if tag == "unit":
            return u.Unit(raw["unit"])
        if tag == "ndarray":
            if arrays_group is None or raw["dataset"] not in arrays_group:
                raise ValueError(
                    f"Missing provenance array dataset {raw.get('dataset')!r}."
                )
            return np.array(arrays_group[raw["dataset"]][()], copy=True)
        if tag == "dataclass":
            cls_name = str(raw.get("class_name", ""))
            fields_decoded = {
                k: _decode_provenance_value(v, arrays_group)
                for k, v in raw.get("fields", {}).items()
            }
            if cls_name in _KNOWN_DATACLASS_REGISTRY:
                mod_name, attr_name = _KNOWN_DATACLASS_REGISTRY[cls_name]
                try:
                    mod = importlib.import_module(mod_name)
                    dc_cls = getattr(mod, attr_name)
                    return dc_cls(**fields_decoded)
                except Exception:
                    pass
            return fields_decoded
        return {k: _decode_provenance_value(v, arrays_group) for k, v in raw.items()}
    return raw


# ---------------------------------------------------------------------------
# ScalarField group writer / reader
# ---------------------------------------------------------------------------


def _write_scalar_group(
    grp: h5py.Group,
    field: ScalarField,
    *,
    skip_unsupported_provenance: bool = False,
) -> None:
    grp.attrs["gwexpy_field_kind"] = "scalar"
    grp.attrs["gwexpy_name"] = str(field.name or "")
    grp.attrs["gwexpy_channel"] = (
        str(field.channel) if getattr(field, "channel", None) is not None else ""
    )
    unit_str = str(field.unit) if field.unit is not None else ""
    grp.attrs["gwexpy_unit"] = unit_str

    time_ref = get_time_reference(field)
    grp.attrs["gwexpy_time_reference"] = time_ref

    epoch_val = getattr(field, "_epoch", None)
    if epoch_val is not None:
        try:
            ep_float = float(
                epoch_val.to_value(u.s)
                if isinstance(epoch_val, u.Quantity)
                else float(epoch_val)
            )
            if np.isfinite(ep_float):
                grp.attrs["gwexpy_has_epoch"] = True
                grp.attrs["gwexpy_epoch"] = ep_float
            else:
                grp.attrs["gwexpy_has_epoch"] = False
        except Exception:
            grp.attrs["gwexpy_has_epoch"] = False
    else:
        grp.attrs["gwexpy_has_epoch"] = False

    grp.attrs["gwexpy_axis0_domain"] = str(field.axis0_domain)
    grp.attrs["gwexpy_axis_names"] = json.dumps(list(field.axis_names))
    grp.attrs["gwexpy_space_domains"] = json.dumps(dict(field.space_domains))

    ax0_offset = getattr(field, "_axis0_offset", None)
    if ax0_offset is not None:
        off_unit = getattr(ax0_offset, "unit", u.dimensionless_unscaled)
        off_val = float(getattr(ax0_offset, "value", ax0_offset))
        grp.attrs["gwexpy_axis0_offset_val"] = off_val
        grp.attrs["gwexpy_axis0_offset_unit"] = str(off_unit)

    ax0_dt = getattr(field, "_axis0_dt", None)
    if ax0_dt is not None:
        dt_unit = getattr(ax0_dt, "unit", u.dimensionless_unscaled)
        dt_val = float(getattr(ax0_dt, "value", ax0_dt))
        grp.attrs["gwexpy_axis0_dt_val"] = dt_val
        grp.attrs["gwexpy_axis0_dt_unit"] = str(dt_unit)

    # GeoMetadata
    if field.geo is not None:
        grp.attrs["gwexpy_geo_json"] = _encode_geo(field.geo)

    # Provenance (_gwex_* except _gwex_geo and transient slots)
    prov_dict: dict[str, Any] = {}
    prov_arrays: dict[str, np.ndarray] = {}
    for attr_name, attr_val in field.__dict__.items():
        if (
            not attr_name.startswith("_gwex_")
            or attr_name == "_gwex_geo"
            or attr_name in _INTERNAL_TRANSIENT_GWEX_ATTRS
        ):
            continue
        try:
            prov_dict[attr_name] = _encode_provenance_value(
                attr_val, prov_arrays, attr_name.lstrip("_")
            )
        except TypeError:
            if not skip_unsupported_provenance:
                raise

    if prov_dict:
        grp.attrs["gwexpy_provenance_json"] = json.dumps(
            prov_dict, separators=(",", ":"), sort_keys=True
        )
    if prov_arrays:
        arr_grp = grp.create_group("provenance_arrays")
        for arr_key, arr_val in prov_arrays.items():
            arr_grp.create_dataset(arr_key, data=arr_val)

    # Re-emit preserved external attributes
    ext_attrs = getattr(field, _EXTERNAL_ATTRS_SLOT, None)
    if isinstance(ext_attrs, dict):
        for ext_k, ext_v in ext_attrs.items():
            if ext_k not in grp.attrs and not ext_k.startswith(("gwexpy_", "_gwexpy_")):
                grp.attrs[ext_k] = ext_v

    # Data and 4 axes (exact 1D coordinate arrays preserved verbatim)
    grp.create_dataset("data", data=np.asarray(field.value))
    axes_grp = grp.create_group("axes")
    for idx in range(4):
        ax_q = getattr(field, f"_axis{idx}_index")
        ax_unit = getattr(ax_q, "unit", u.dimensionless_unscaled)
        ax_vals = np.asarray(getattr(ax_q, "value", ax_q))
        ds = axes_grp.create_dataset(f"axis{idx}", data=ax_vals)
        ds.attrs["name"] = str(field.axis_names[idx])
        ds.attrs["unit"] = str(ax_unit)


def _read_scalar_group(
    grp: h5py.Group,
    *,
    strict_schema: bool = False,
    root_external_attrs: dict[str, Any] | None = None,
) -> ScalarField:
    if not _is_legacy_scalar_group(grp):
        raise ValueError(
            f"HDF5 group {grp.name!r} does not contain a valid Field layout "
            "(expected 'data' dataset and 'axes/axis0..axis3')."
        )

    group_external = classify_unknown_attrs(
        grp.attrs,
        known_keys=_KNOWN_SCALAR_GROUP_ATTRS,
        strict=strict_schema,
    )

    data = np.array(grp["data"][()], copy=True)
    if data.ndim != 4:
        raise ValueError(
            f"Field HDF5 'data' dataset must be 4-dimensional, got {data.ndim}D."
        )

    axes_grp = grp["axes"]
    axes_quantities: list[u.Quantity] = []
    axis_names_from_ds: list[str] = []
    default_names = ("t", "x", "y", "z")
    for idx in range(4):
        ds = axes_grp[f"axis{idx}"]
        vals = np.array(ds[()], copy=True)
        unit_raw = ds.attrs.get("unit", "")
        unit_str = _decode_str(unit_raw) if unit_raw is not None else ""
        ax_unit = u.Unit(unit_str) if unit_str else u.dimensionless_unscaled
        axes_quantities.append(vals * ax_unit)
        name_raw = ds.attrs.get("name", default_names[idx])
        axis_names_from_ds.append(_decode_str(name_raw))

    if "gwexpy_axis_names" in grp.attrs:
        axis_names = list(json.loads(_decode_str(grp.attrs["gwexpy_axis_names"])))
    else:
        axis_names = axis_names_from_ds

    axis0_domain = (
        _decode_str(grp.attrs["gwexpy_axis0_domain"])
        if "gwexpy_axis0_domain" in grp.attrs
        else "time"
    )
    if "gwexpy_space_domains" in grp.attrs:
        space_domains: str | dict[str, str] = dict(
            json.loads(_decode_str(grp.attrs["gwexpy_space_domains"]))
        )
    else:
        space_domains = "real"

    unit_attr = (
        _decode_str(grp.attrs["gwexpy_unit"])
        if "gwexpy_unit" in grp.attrs
        else _decode_str(grp["data"].attrs.get("unit", ""))
    )
    field_unit = u.Unit(unit_attr) if unit_attr else u.dimensionless_unscaled

    name_attr = (
        _decode_str(grp.attrs["gwexpy_name"]) if "gwexpy_name" in grp.attrs else ""
    )
    channel_attr = (
        _decode_str(grp.attrs["gwexpy_channel"])
        if "gwexpy_channel" in grp.attrs
        else ""
    )

    has_epoch = bool(grp.attrs.get("gwexpy_has_epoch", False))
    epoch_val: float | None = (
        float(grp.attrs["gwexpy_epoch"])
        if has_epoch and "gwexpy_epoch" in grp.attrs
        else None
    )

    time_ref = (
        _decode_str(grp.attrs["gwexpy_time_reference"])
        if "gwexpy_time_reference" in grp.attrs
        else (
            TIME_REFERENCE_ABSOLUTE
            if epoch_val is not None
            else TIME_REFERENCE_RELATIVE
        )
    )
    if time_ref not in (TIME_REFERENCE_ABSOLUTE, TIME_REFERENCE_RELATIVE):
        raise ValueError(f"Invalid gwexpy_time_reference {time_ref!r} in HDF5.")

    sf = ScalarField(
        data,
        unit=field_unit,
        name=name_attr or None,
        channel=channel_attr or None,
        epoch=epoch_val,
        axis0=axes_quantities[0],
        axis1=axes_quantities[1],
        axis2=axes_quantities[2],
        axis3=axes_quantities[3],
        axis_names=axis_names,
        axis0_domain=axis0_domain,
        space_domain=space_domains,
    )
    set_time_reference(sf, time_ref)

    if (
        "gwexpy_axis0_offset_val" in grp.attrs
        and "gwexpy_axis0_offset_unit" in grp.attrs
    ):
        off_u = _decode_str(grp.attrs["gwexpy_axis0_offset_unit"])
        sf._axis0_offset = float(grp.attrs["gwexpy_axis0_offset_val"]) * (
            u.Unit(off_u) if off_u else u.dimensionless_unscaled
        )
    if "gwexpy_axis0_dt_val" in grp.attrs and "gwexpy_axis0_dt_unit" in grp.attrs:
        dt_u = _decode_str(grp.attrs["gwexpy_axis0_dt_unit"])
        sf._axis0_dt = float(grp.attrs["gwexpy_axis0_dt_val"]) * (
            u.Unit(dt_u) if dt_u else u.dimensionless_unscaled
        )

    if "gwexpy_geo_json" in grp.attrs:
        sf.geo = _decode_geo(_decode_str(grp.attrs["gwexpy_geo_json"]))

    if "gwexpy_provenance_json" in grp.attrs:
        raw_prov = json.loads(_decode_str(grp.attrs["gwexpy_provenance_json"]))
        arr_grp = (
            grp["provenance_arrays"]
            if "provenance_arrays" in grp
            and isinstance(grp["provenance_arrays"], h5py.Group)
            else None
        )
        for p_key, p_val in raw_prov.items():
            if isinstance(p_key, str) and p_key.startswith("_gwex_"):
                setattr(sf, p_key, _decode_provenance_value(p_val, arr_grp))

    merged_external: dict[str, Any] = {}
    if root_external_attrs:
        for k, v in root_external_attrs.items():
            merged_external[k] = _detach_attr_value(v)
    for k, v in group_external.items():
        merged_external[k] = _detach_attr_value(v)
    if merged_external:
        setattr(sf, _EXTERNAL_ATTRS_SLOT, merged_external)

    return sf


# ---------------------------------------------------------------------------
# VectorField and TensorField group writers / readers
# ---------------------------------------------------------------------------


def _write_vector_group(
    grp: h5py.Group,
    field: VectorField,
    *,
    skip_unsupported_provenance: bool = False,
) -> None:
    grp.attrs["gwexpy_field_kind"] = "vector"
    grp.attrs["gwexpy_basis"] = str(getattr(field, "basis", "cartesian"))
    keys = [str(k) for k in field.keys()]
    grp.attrs["gwexpy_component_order"] = json.dumps(keys)
    comp_grp = grp.create_group("components")
    for idx, key in enumerate(keys):
        sub = comp_grp.create_group(f"comp_{idx}")
        sub.attrs["key"] = key
        _write_scalar_group(
            sub,
            field[key],
            skip_unsupported_provenance=skip_unsupported_provenance,
        )


def _read_vector_group(
    grp: h5py.Group,
    *,
    strict_schema: bool = False,
    root_external_attrs: dict[str, Any] | None = None,
) -> VectorField:
    group_external = classify_unknown_attrs(
        grp.attrs,
        known_keys=_KNOWN_COMPOUND_GROUP_ATTRS,
        strict=strict_schema,
    )
    basis_str = (
        _decode_str(grp.attrs["gwexpy_basis"])
        if "gwexpy_basis" in grp.attrs
        else "cartesian"
    )
    if "components" not in grp or not isinstance(grp["components"], h5py.Group):
        raise ValueError(
            f"VectorField HDF5 group {grp.name!r} is missing 'components' subgroup."
        )
    comp_grp = grp["components"]
    order = (
        list(json.loads(_decode_str(grp.attrs["gwexpy_component_order"])))
        if "gwexpy_component_order" in grp.attrs
        else sorted(comp_grp.keys())
    )
    merged_ext = dict(root_external_attrs or {})
    for k, v in group_external.items():
        merged_ext[k] = _detach_attr_value(v)

    components: dict[str, ScalarField] = {}
    for idx, expected_key in enumerate(order):
        sub_name = f"comp_{idx}" if f"comp_{idx}" in comp_grp else expected_key
        sub = comp_grp[sub_name]
        comp_key = (
            _decode_str(sub.attrs["key"]) if "key" in sub.attrs else str(expected_key)
        )
        known_sub = _KNOWN_SCALAR_GROUP_ATTRS | frozenset({"key"})
        sub_ext = classify_unknown_attrs(
            sub.attrs, known_keys=known_sub, strict=strict_schema
        )
        comp_sf = _read_scalar_group(
            sub,
            strict_schema=False,
            root_external_attrs={**merged_ext, **sub_ext},
        )
        components[comp_key] = comp_sf

    vf = VectorField(components, basis=basis_str, validate=False)  # type: ignore[arg-type]
    if merged_ext:
        setattr(vf, _EXTERNAL_ATTRS_SLOT, merged_ext)
    return vf


def _encode_tensor_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return {
            "__tuple__": [
                int(x) if isinstance(x, (int, np.integer)) else x for x in key
            ]
        }
    return key


def _decode_tensor_key(raw: Any) -> Any:
    if isinstance(raw, dict) and "__tuple__" in raw:
        return tuple(raw["__tuple__"])
    return raw


def _write_tensor_group(
    grp: h5py.Group,
    field: TensorField,
    *,
    skip_unsupported_provenance: bool = False,
) -> None:
    grp.attrs["gwexpy_field_kind"] = "tensor"
    if field.rank is not None:
        grp.attrs["gwexpy_rank"] = int(field.rank)
    keys = list(field.keys())
    grp.attrs["gwexpy_component_order"] = json.dumps(
        [_encode_tensor_key(k) for k in keys]
    )
    comp_grp = grp.create_group("components")
    for idx, key in enumerate(keys):
        sub = comp_grp.create_group(f"comp_{idx}")
        sub.attrs["key_json"] = json.dumps(_encode_tensor_key(key))
        _write_scalar_group(
            sub,
            field[key],
            skip_unsupported_provenance=skip_unsupported_provenance,
        )


def _read_tensor_group(
    grp: h5py.Group,
    *,
    strict_schema: bool = False,
    root_external_attrs: dict[str, Any] | None = None,
) -> TensorField:
    group_external = classify_unknown_attrs(
        grp.attrs,
        known_keys=_KNOWN_COMPOUND_GROUP_ATTRS,
        strict=strict_schema,
    )
    rank = int(grp.attrs["gwexpy_rank"]) if "gwexpy_rank" in grp.attrs else None
    if "components" not in grp or not isinstance(grp["components"], h5py.Group):
        raise ValueError(
            f"TensorField HDF5 group {grp.name!r} is missing 'components' subgroup."
        )
    comp_grp = grp["components"]
    raw_order = (
        list(json.loads(_decode_str(grp.attrs["gwexpy_component_order"])))
        if "gwexpy_component_order" in grp.attrs
        else []
    )
    merged_ext = dict(root_external_attrs or {})
    for k, v in group_external.items():
        merged_ext[k] = _detach_attr_value(v)

    components: dict[Any, ScalarField] = {}
    for idx in range(len(raw_order) if raw_order else len(comp_grp)):
        sub = comp_grp[f"comp_{idx}"]
        if "key_json" in sub.attrs:
            comp_key = _decode_tensor_key(
                json.loads(_decode_str(sub.attrs["key_json"]))
            )
        else:
            comp_key = _decode_tensor_key(raw_order[idx])
        known_sub = _KNOWN_SCALAR_GROUP_ATTRS | frozenset({"key_json"})
        sub_ext = classify_unknown_attrs(
            sub.attrs, known_keys=known_sub, strict=strict_schema
        )
        comp_sf = _read_scalar_group(
            sub,
            strict_schema=False,
            root_external_attrs={**merged_ext, **sub_ext},
        )
        components[comp_key] = comp_sf

    tf = TensorField(components, rank=rank, validate=False)
    if merged_ext:
        setattr(tf, _EXTERNAL_ATTRS_SLOT, merged_ext)
    return tf


def _write_any_field_group(
    grp: h5py.Group,
    field: Any,
    *,
    skip_unsupported_provenance: bool = False,
) -> None:
    if isinstance(field, ScalarField):
        _write_scalar_group(
            grp, field, skip_unsupported_provenance=skip_unsupported_provenance
        )
    elif isinstance(field, VectorField):
        _write_vector_group(
            grp, field, skip_unsupported_provenance=skip_unsupported_provenance
        )
    elif isinstance(field, TensorField):
        _write_tensor_group(
            grp, field, skip_unsupported_provenance=skip_unsupported_provenance
        )
    else:
        raise TypeError(
            f"Unsupported field type {type(field).__name__} for Field HDF5 persistence."
        )


def _read_any_field_group(
    grp: h5py.Group,
    *,
    strict_schema: bool = False,
    root_external_attrs: dict[str, Any] | None = None,
) -> ScalarField | VectorField | TensorField:
    kind = (
        _decode_str(grp.attrs["gwexpy_field_kind"])
        if "gwexpy_field_kind" in grp.attrs
        else "scalar"
    )
    if kind == "scalar":
        return _read_scalar_group(
            grp,
            strict_schema=strict_schema,
            root_external_attrs=root_external_attrs,
        )
    if kind == "vector":
        return _read_vector_group(
            grp,
            strict_schema=strict_schema,
            root_external_attrs=root_external_attrs,
        )
    if kind == "tensor":
        return _read_tensor_group(
            grp,
            strict_schema=strict_schema,
            root_external_attrs=root_external_attrs,
        )
    raise ValueError(f"Unsupported gwexpy_field_kind {kind!r} in HDF5 group.")


# ---------------------------------------------------------------------------
# Public writers and readers
# ---------------------------------------------------------------------------


def _validate_root_for_read(
    root: h5py.File | h5py.Group,
    *,
    allowed_schemas: tuple[str, ...],
    strict_schema: bool,
    warn_on_legacy: bool,
) -> tuple[str | None, dict[str, Any]]:
    if SCHEMA_ATTR in root.attrs:
        actual_schema = _decode_str(root.attrs[SCHEMA_ATTR])
        if actual_schema not in allowed_schemas:
            raise ValueError(
                f"Schema mismatch: expected one of {allowed_schemas!r}, "
                f"got {actual_schema!r}"
            )
        read_and_validate_schema(
            root,
            actual_schema,
            MAX_SUPPORTED_FIELD_SCHEMA_VERSION,
            warn_on_legacy=warn_on_legacy,
        )
    else:
        # Either legacy unversioned or missing schema name while version is present
        read_and_validate_schema(
            root,
            allowed_schemas[0],
            MAX_SUPPORTED_FIELD_SCHEMA_VERSION,
            warn_on_legacy=warn_on_legacy,
        )
        if not _is_legacy_field_container(root):
            raise ValueError(
                "Unversioned HDF5 file does not match any supported legacy "
                "GWexpy Field layout."
            )
        actual_schema = None

    known_root = (
        _KNOWN_ROOT_ATTRS | _KNOWN_SCALAR_GROUP_ATTRS | _KNOWN_COMPOUND_GROUP_ATTRS
    )
    root_ext = classify_unknown_attrs(
        root.attrs,
        known_keys=known_root,
        strict=strict_schema,
    )
    return actual_schema, {k: _detach_attr_value(v) for k, v in root_ext.items()}


def _resolve_field_group_for_read(
    root: h5py.File | h5py.Group,
    name: str | None,
) -> tuple[h5py.Group, str | None]:
    if "fields" in root and isinstance(root["fields"], h5py.Group):
        fgrp = root["fields"]
        available = _list_names_in_fields_group(root, fgrp)
        if name is None:
            if len(available) == 1:
                name = available[0]
            elif "gwexpy_default_name" in root.attrs:
                default_n = _decode_str(root.attrs["gwexpy_default_name"])
                if default_n in available:
                    name = default_n
            if name is None:
                raise ValueError(
                    f"Field name must be specified when reading from multi-field "
                    f"HDF5 container (available: {available})."
                )
        sub = _lookup_named_subgroup(fgrp, name, available)
        return sub, name

    # Direct single-field layout at root
    if name is not None:
        root_name = (
            _decode_str(root.attrs["gwexpy_name"])
            if "gwexpy_name" in root.attrs
            else None
        )
        if root_name and root_name != name:
            raise KeyError(
                f"Requested field {name!r} not found in single-field HDF5 file "
                f"(available: {[root_name]})."
            )
    return root, name


def _list_names_in_fields_group(
    root: h5py.File | h5py.Group,
    fgrp: h5py.Group,
) -> list[str]:
    if "gwexpy_field_order" in root.attrs:
        return list(json.loads(_decode_str(root.attrs["gwexpy_field_order"])))
    names: list[str] = []
    for k, sub in fgrp.items():
        if isinstance(sub, h5py.Group) and "field_key" in sub.attrs:
            names.append(_decode_str(sub.attrs["field_key"]))
        else:
            names.append(str(k))
    return names


def _lookup_named_subgroup(
    fgrp: h5py.Group,
    requested_name: str,
    available: list[str],
) -> h5py.Group:
    if requested_name not in available:
        raise KeyError(
            f"Requested field {requested_name!r} not found in HDF5 file "
            f"(available: {available})."
        )
    idx = available.index(requested_name)
    indexed_key = f"field_{idx}"
    if indexed_key in fgrp and isinstance(fgrp[indexed_key], h5py.Group):
        sub = fgrp[indexed_key]
        if (
            "field_key" not in sub.attrs
            or _decode_str(sub.attrs["field_key"]) == requested_name
        ):
            return sub
    if requested_name in fgrp and isinstance(fgrp[requested_name], h5py.Group):
        return fgrp[requested_name]
    for sub in fgrp.values():
        if (
            isinstance(sub, h5py.Group)
            and "field_key" in sub.attrs
            and _decode_str(sub.attrs["field_key"]) == requested_name
        ):
            return sub
    raise KeyError(
        f"Requested field {requested_name!r} not found in HDF5 file "
        f"(available: {available})."
    )


def list_fields_hdf5(
    source: Any,
    *,
    strict_schema: bool = False,
    warn_on_legacy: bool = True,
    **kwargs: Any,
) -> list[str]:
    """List available field names in a Field or FieldDict HDF5 file."""
    with _open_h5(source, "r") as root:
        _validate_root_for_read(
            root,
            allowed_schemas=(FIELD_SCHEMA_NAME, FIELD_DICT_SCHEMA_NAME),
            strict_schema=strict_schema,
            warn_on_legacy=warn_on_legacy,
        )
        if "fields" in root and isinstance(root["fields"], h5py.Group):
            return _list_names_in_fields_group(root, root["fields"])
        if "gwexpy_name" in root.attrs:
            n = _decode_str(root.attrs["gwexpy_name"])
            return [n] if n else ["field"]
        return ["field"]


def write_scalarfield_hdf5(
    field: ScalarField,
    target: Any,
    *,
    overwrite: bool = True,
    skip_unsupported_provenance: bool = False,
    **kwargs: Any,
) -> None:
    """Write a :class:`ScalarField` to an HDF5 file or group."""
    mode = "w" if overwrite else "x"
    with _open_h5(target, mode) as root:
        write_schema_attrs(root, FIELD_SCHEMA_NAME, FIELD_SCHEMA_VERSION)
        _write_scalar_group(
            root,
            field,
            skip_unsupported_provenance=skip_unsupported_provenance,
        )


def read_scalarfield_hdf5(
    source: Any,
    name: str | None = None,
    *,
    strict_schema: bool = False,
    warn_on_legacy: bool = True,
    **kwargs: Any,
) -> ScalarField:
    """Read a :class:`ScalarField` from an HDF5 file or group."""
    with _open_h5(source, "r") as root:
        _, root_ext = _validate_root_for_read(
            root,
            allowed_schemas=(FIELD_SCHEMA_NAME, FIELD_DICT_SCHEMA_NAME),
            strict_schema=strict_schema,
            warn_on_legacy=warn_on_legacy,
        )
        grp, resolved_name = _resolve_field_group_for_read(root, name)
        kind = (
            _decode_str(grp.attrs["gwexpy_field_kind"])
            if "gwexpy_field_kind" in grp.attrs
            else "scalar"
        )
        if kind != "scalar":
            raise ValueError(
                f"Field {resolved_name or grp.name!r} has kind {kind!r}, "
                "expected 'scalar'."
            )
        sf = _read_scalar_group(
            grp,
            strict_schema=strict_schema,
            root_external_attrs=root_ext,
        )
        if resolved_name is not None and not sf.name:
            sf.name = resolved_name
        return sf


def write_vectorfield_hdf5(
    field: VectorField,
    target: Any,
    *,
    overwrite: bool = True,
    skip_unsupported_provenance: bool = False,
    **kwargs: Any,
) -> None:
    """Write a :class:`VectorField` to an HDF5 file or group."""
    mode = "w" if overwrite else "x"
    with _open_h5(target, mode) as root:
        write_schema_attrs(root, FIELD_SCHEMA_NAME, FIELD_SCHEMA_VERSION)
        _write_vector_group(
            root,
            field,
            skip_unsupported_provenance=skip_unsupported_provenance,
        )
        ext_attrs = getattr(field, _EXTERNAL_ATTRS_SLOT, None)
        if isinstance(ext_attrs, dict):
            for k, v in ext_attrs.items():
                if k not in root.attrs and not k.startswith(("gwexpy_", "_gwexpy_")):
                    root.attrs[k] = v


def read_vectorfield_hdf5(
    source: Any,
    name: str | None = None,
    *,
    strict_schema: bool = False,
    warn_on_legacy: bool = True,
    **kwargs: Any,
) -> VectorField:
    """Read a :class:`VectorField` from an HDF5 file or group."""
    with _open_h5(source, "r") as root:
        _, root_ext = _validate_root_for_read(
            root,
            allowed_schemas=(FIELD_SCHEMA_NAME, FIELD_DICT_SCHEMA_NAME),
            strict_schema=strict_schema,
            warn_on_legacy=warn_on_legacy,
        )
        grp, resolved_name = _resolve_field_group_for_read(root, name)
        kind = (
            _decode_str(grp.attrs["gwexpy_field_kind"])
            if "gwexpy_field_kind" in grp.attrs
            else "scalar"
        )
        if kind != "vector":
            raise ValueError(
                f"Field {resolved_name or grp.name!r} has kind {kind!r}, "
                "expected 'vector'."
            )
        return _read_vector_group(
            grp,
            strict_schema=strict_schema,
            root_external_attrs=root_ext,
        )


def write_tensorfield_hdf5(
    field: TensorField,
    target: Any,
    *,
    overwrite: bool = True,
    skip_unsupported_provenance: bool = False,
    **kwargs: Any,
) -> None:
    """Write a :class:`TensorField` to an HDF5 file or group."""
    mode = "w" if overwrite else "x"
    with _open_h5(target, mode) as root:
        write_schema_attrs(root, FIELD_SCHEMA_NAME, FIELD_SCHEMA_VERSION)
        _write_tensor_group(
            root,
            field,
            skip_unsupported_provenance=skip_unsupported_provenance,
        )
        ext_attrs = getattr(field, _EXTERNAL_ATTRS_SLOT, None)
        if isinstance(ext_attrs, dict):
            for k, v in ext_attrs.items():
                if k not in root.attrs and not k.startswith(("gwexpy_", "_gwexpy_")):
                    root.attrs[k] = v


def read_tensorfield_hdf5(
    source: Any,
    name: str | None = None,
    *,
    strict_schema: bool = False,
    warn_on_legacy: bool = True,
    **kwargs: Any,
) -> TensorField:
    """Read a :class:`TensorField` from an HDF5 file or group."""
    with _open_h5(source, "r") as root:
        _, root_ext = _validate_root_for_read(
            root,
            allowed_schemas=(FIELD_SCHEMA_NAME, FIELD_DICT_SCHEMA_NAME),
            strict_schema=strict_schema,
            warn_on_legacy=warn_on_legacy,
        )
        grp, resolved_name = _resolve_field_group_for_read(root, name)
        kind = (
            _decode_str(grp.attrs["gwexpy_field_kind"])
            if "gwexpy_field_kind" in grp.attrs
            else "scalar"
        )
        if kind != "tensor":
            raise ValueError(
                f"Field {resolved_name or grp.name!r} has kind {kind!r}, "
                "expected 'tensor'."
            )
        return _read_tensor_group(
            grp,
            strict_schema=strict_schema,
            root_external_attrs=root_ext,
        )


def write_fielddict_hdf5(
    field_dict: FieldDict,
    target: Any,
    *,
    overwrite: bool = True,
    skip_unsupported_provenance: bool = False,
    **kwargs: Any,
) -> None:
    """Write a heterogeneous :class:`FieldDict` to an HDF5 file or group."""
    mode = "w" if overwrite else "x"
    with _open_h5(target, mode) as root:
        write_schema_attrs(root, FIELD_DICT_SCHEMA_NAME, FIELD_SCHEMA_VERSION)
        keys = [str(k) for k in field_dict.keys()]
        root.attrs["gwexpy_field_order"] = json.dumps(keys)
        fgrp = root.create_group("fields")
        for idx, key in enumerate(keys):
            sub = fgrp.create_group(f"field_{idx}")
            sub.attrs["field_key"] = key
            _write_any_field_group(
                sub,
                field_dict[key],
                skip_unsupported_provenance=skip_unsupported_provenance,
            )
        ext_attrs = getattr(field_dict, _EXTERNAL_ATTRS_SLOT, None)
        if isinstance(ext_attrs, dict):
            for k, v in ext_attrs.items():
                if k not in root.attrs and not k.startswith(("gwexpy_", "_gwexpy_")):
                    root.attrs[k] = v


def read_fielddict_hdf5(
    source: Any,
    names: Sequence[str] | str | None = None,
    *,
    strict_schema: bool = False,
    warn_on_legacy: bool = True,
    **kwargs: Any,
) -> FieldDict:
    """Read a :class:`FieldDict` from an HDF5 file or group."""
    if isinstance(names, str):
        req_names = [names]
    elif names is not None:
        req_names = list(names)
    else:
        req_names = []

    with _open_h5(source, "r") as root:
        _, root_ext = _validate_root_for_read(
            root,
            allowed_schemas=(FIELD_DICT_SCHEMA_NAME, FIELD_SCHEMA_NAME),
            strict_schema=strict_schema,
            warn_on_legacy=warn_on_legacy,
        )
        if "fields" in root and isinstance(root["fields"], h5py.Group):
            fgrp = root["fields"]
            available = _list_names_in_fields_group(root, fgrp)
            selected = req_names if req_names else available
            out = FieldDict()
            for req in selected:
                sub = _lookup_named_subgroup(fgrp, req, available)
                known_sub = (
                    _KNOWN_SCALAR_GROUP_ATTRS
                    | _KNOWN_COMPOUND_GROUP_ATTRS
                    | frozenset({"field_key"})
                )
                sub_ext = classify_unknown_attrs(
                    sub.attrs, known_keys=known_sub, strict=strict_schema
                )
                out[req] = _read_any_field_group(
                    sub,
                    strict_schema=False,
                    root_external_attrs={**root_ext, **sub_ext},
                )
            if root_ext:
                setattr(out, _EXTERNAL_ATTRS_SLOT, root_ext)
            return out

        # Single-field file read via FieldDict.read()
        single_name = (
            _decode_str(root.attrs["gwexpy_name"])
            if "gwexpy_name" in root.attrs and _decode_str(root.attrs["gwexpy_name"])
            else "field"
        )
        available = [single_name]
        selected = req_names if req_names else available
        out = FieldDict()
        for req in selected:
            if req not in available:
                raise KeyError(
                    f"Requested field {req!r} not found in HDF5 file "
                    f"(available: {available})."
                )
            out[req] = _read_any_field_group(
                root,
                strict_schema=strict_schema,
                root_external_attrs=root_ext,
            )
        if root_ext:
            setattr(out, _EXTERNAL_ATTRS_SLOT, root_ext)
        return out


def register_hdf5_field_formats() -> None:
    """Register canonical ``hdf5`` (and ``h5``, ``hdf.field``) on Field I/O."""
    register_field_format(
        "hdf5",
        reader_scalar=read_scalarfield_hdf5,
        reader_vector=read_vectorfield_hdf5,
        reader_tensor=read_tensorfield_hdf5,
        reader_dict=read_fielddict_hdf5,
        writer_scalar=write_scalarfield_hdf5,
        writer_vector=write_vectorfield_hdf5,
        writer_tensor=write_tensorfield_hdf5,
        writer_dict=write_fielddict_hdf5,
        identifier=identify_field_hdf5,
        lister=list_fields_hdf5,
        aliases=("h5", "hdf.field"),
    )


register_hdf5_field_formats()
