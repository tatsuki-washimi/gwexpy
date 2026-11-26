"""HDF5/CSV persistence, schema versioning, provenance, and payload serializer registry."""

from __future__ import annotations

import json
import math
import os
import pickle
import warnings
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import h5py
import numpy as np
import pandas as pd
from astropy import units as u

from gwexpy.table.segment.cell import _UNSET, CellStatus, SegmentCell
from gwexpy.table.segment.status import is_scalar_nan

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable

__all__ = [
    "KNOWN_ROOT_ATTRS",
    "MAX_SUPPORTED_SCHEMA_VERSION",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "get_payload_serializer",
    "register_payload_serializer",
]

SCHEMA_NAME = "gwexpy.segment_table"
SCHEMA_VERSION = 1
MAX_SUPPORTED_SCHEMA_VERSION = 1
KNOWN_ROOT_ATTRS = frozenset(
    {
        "gwexpy_schema",
        "gwexpy_schema_version",
        "gwexpy_provenance",
        "_gwexpy_time_reference_v1",
        "_gwexpy_sidecar_json_v1",
        "_gwexpy_sidecar_json_v2",
    }
)

PayloadWriter = Callable[[h5py.Group, Any], None]
PayloadReader = Callable[[h5py.Group], Any]

_PAYLOAD_SERIALIZERS: dict[str, tuple[PayloadWriter, PayloadReader]] = {}


def register_payload_serializer(
    kind: str,
    writer: PayloadWriter,
    reader: PayloadReader,
    *,
    overwrite: bool = False,
) -> None:
    """Register an HDF5 ``(writer, reader)`` pair for payload *kind*."""
    if kind in _PAYLOAD_SERIALIZERS and not overwrite:
        raise ValueError(f"Payload serializer for kind {kind!r} is already registered.")
    _PAYLOAD_SERIALIZERS[kind] = (writer, reader)


def get_payload_serializer(
    kind: str,
) -> Optional[tuple[PayloadWriter, PayloadReader]]:
    """Return the registered ``(writer, reader)`` for *kind*, or ``None``."""
    return _PAYLOAD_SERIALIZERS.get(kind)


# ---------------------------------------------------------------------------
# Built-in payload serializers (no pickle)
# ---------------------------------------------------------------------------


def _write_timeseries(group: h5py.Group, val: Any) -> None:
    val.write(group, format="hdf5", path="data")
    group.attrs["serializer"] = "timeseries"


def _read_timeseries(group: h5py.Group) -> Any:
    from gwexpy.timeseries import TimeSeries

    return TimeSeries.read(group, format="hdf5", path="data")


def _write_frequencyseries(group: h5py.Group, val: Any) -> None:
    val.write(group, format="hdf5", path="data")
    group.attrs["serializer"] = "frequencyseries"


def _read_frequencyseries(group: h5py.Group) -> Any:
    from gwexpy.frequencyseries import FrequencySeries

    return FrequencySeries.read(group, format="hdf5", path="data")


def _write_spectrogram(group: h5py.Group, val: Any) -> None:
    val.write(group, format="hdf5", path="data")
    group.attrs["serializer"] = "spectrogram"


def _read_spectrogram(group: h5py.Group) -> Any:
    from gwexpy.spectrogram import Spectrogram

    return Spectrogram.read(group, format="hdf5", path="data")


def _write_series_dict(
    group: h5py.Group, val: Any, item_writer: PayloadWriter, serializer_name: str
) -> None:
    keys = list(val.keys())
    group.attrs["serializer"] = serializer_name
    group.attrs["dict_kind"] = "dict" if type(val) is dict else serializer_name
    group.attrs["order"] = json.dumps([str(k) for k in keys])
    for idx, key in enumerate(keys):
        sub = group.create_group(f"item_{idx}")
        sub.attrs["key"] = str(key)
        item_writer(sub, val[key])


def _read_series_dict(
    group: h5py.Group, item_reader: PayloadReader, dict_cls: type[Any]
) -> Any:
    raw_order = group.attrs.get("order", "[]")
    if isinstance(raw_order, bytes):
        raw_order = raw_order.decode("utf-8")
    keys = json.loads(raw_order)
    dict_kind = group.attrs.get("dict_kind", "")
    if isinstance(dict_kind, bytes):
        dict_kind = dict_kind.decode("utf-8")
    out: Any = {} if dict_kind == "dict" else dict_cls()
    for idx, key in enumerate(keys):
        sub = cast(h5py.Group, group[f"item_{idx}"])
        out[key] = item_reader(sub)
    return out


def _write_timeseriesdict(group: h5py.Group, val: Any) -> None:
    _write_series_dict(group, val, _write_timeseries, "timeseriesdict")


def _read_timeseriesdict(group: h5py.Group) -> Any:
    from gwexpy.timeseries import TimeSeriesDict

    return _read_series_dict(group, _read_timeseries, TimeSeriesDict)


def _write_frequencyseriesdict(group: h5py.Group, val: Any) -> None:
    _write_series_dict(group, val, _write_frequencyseries, "frequencyseriesdict")


def _read_frequencyseriesdict(group: h5py.Group) -> Any:
    from gwexpy.frequencyseries import FrequencySeriesDict

    return _read_series_dict(group, _read_frequencyseries, FrequencySeriesDict)


def _write_matrix(group: h5py.Group, val: Any) -> None:
    from gwexpy.time._reference import get_time_reference
    from gwexpy.types.series_matrix_io import (
        _encode_attrs_json,
        _to_json_native,
        _write_axis_group,
    )

    group.attrs["serializer"] = "matrix"
    group.attrs["matrix_class"] = f"{type(val).__module__}.{type(val).__name__}"
    group.attrs["name"] = str(getattr(val, "name", "") or "")
    group.attrs["gwexpy_time_reference"] = get_time_reference(val)

    attrs_json = _encode_attrs_json(getattr(val, "attrs", None))
    if attrs_json is not None:
        group.attrs["attrs_json"] = attrs_json

    try:
        cadence = val.dx
    except (AttributeError, IndexError):
        cadence = None
    if cadence is not None:
        group.attrs["gwexpy_dx"] = float(cadence.value)
        group.attrs["gwexpy_dx_unit"] = str(cadence.unit)

    try:
        group.attrs["epoch"] = float(getattr(val, "epoch", 0.0))
    except (IndexError, KeyError, TypeError, ValueError, AttributeError):
        pass

    group.create_dataset("data", data=val.value)
    _write_axis_group(group, "xindex", val.xindex)
    for axis_name in getattr(val, "_hdf5_extra_axes", ()):
        axis = getattr(val, axis_name, None)
        if axis is not None:
            _write_axis_group(group, axis_name, axis)

    meta_grp = group.create_group("meta")
    units = np.vectorize(lambda u_: "" if u_ is None else str(u_))(val.meta.units)
    names = np.vectorize(lambda n: "" if n is None else str(n))(val.meta.names)
    channels = np.vectorize(lambda c: "" if c is None else str(c))(val.meta.channels)
    meta_grp.create_dataset("units", data=units.astype("S"))
    meta_grp.create_dataset("names", data=names.astype("S"))
    meta_grp.create_dataset("channels", data=channels.astype("S"))

    for grp_name, axis_dict in (("rows", val.rows), ("cols", val.cols)):
        ag = group.create_group(grp_name)
        ag.attrs["key_format"] = "json"
        ag.create_dataset(
            "keys",
            data=np.array(
                [json.dumps(_to_json_native(k)) for k in axis_dict.keys()], dtype="S"
            ),
        )
        ag.create_dataset(
            "names",
            data=np.array([str(v.name) for v in axis_dict.values()], dtype="S"),
        )
        ag.create_dataset(
            "units",
            data=np.array([str(v.unit) for v in axis_dict.values()], dtype="S"),
        )
        ag.create_dataset(
            "channels",
            data=np.array([str(v.channel) for v in axis_dict.values()], dtype="S"),
        )


def _read_matrix(group: h5py.Group) -> Any:
    from gwexpy.frequencyseries import FrequencySeriesMatrix
    from gwexpy.time._reference import set_time_reference
    from gwexpy.timeseries import TimeSeriesMatrix
    from gwexpy.types import SeriesMatrix
    from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix
    from gwexpy.types.series_matrix_io import _decode_matrix_key, _read_axis_group

    cls_name = group.attrs.get("matrix_class", "")
    if isinstance(cls_name, bytes):
        cls_name = cls_name.decode("utf-8")

    matrix_cls: type[Any] = SeriesMatrix
    if cls_name.endswith(".TimeSeriesMatrix"):
        matrix_cls = TimeSeriesMatrix
    elif cls_name.endswith(".FrequencySeriesMatrix"):
        matrix_cls = FrequencySeriesMatrix

    data = group["data"][:]
    xindex = _read_axis_group(group, "xindex")
    extra_axes = {
        axis_name: _read_axis_group(group, axis_name)
        for axis_name in getattr(matrix_cls, "_hdf5_extra_axes", ())
        if axis_name in group
    }
    name = group.attrs.get("name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    time_reference = group.attrs.get("gwexpy_time_reference", "absolute")
    if isinstance(time_reference, bytes):
        time_reference = time_reference.decode("utf-8")
    epoch = group.attrs.get("epoch", 0.0)

    cadence = None
    if "gwexpy_dx" in group.attrs and "gwexpy_dx_unit" in group.attrs:
        dx_unit = group.attrs["gwexpy_dx_unit"]
        if isinstance(dx_unit, bytes):
            dx_unit = dx_unit.decode("utf-8")
        cadence = u.Quantity(
            float(group.attrs["gwexpy_dx"]),
            u.Unit(dx_unit) if dx_unit else u.dimensionless_unscaled,
        )

    attrs_raw = group.attrs.get("attrs_json")
    if isinstance(attrs_raw, bytes):
        attrs_raw = attrs_raw.decode("utf-8")
    attrs = json.loads(attrs_raw) if attrs_raw else {}

    meta_grp = cast(h5py.Group, group["meta"])
    units_raw = meta_grp["units"][:].astype(str)
    names_raw = meta_grp["names"][:].astype(str)
    channels_raw = meta_grp["channels"][:].astype(str)
    nrows, ncols = units_raw.shape
    meta_arr = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            u_str = units_raw[r, c]
            meta_arr[r, c] = MetaData(
                unit=u.Unit(u_str) if u_str else u.dimensionless_unscaled,
                name=names_raw[r, c] or None,
                channel=channels_raw[r, c] or None,
            )
    meta_matrix = MetaDataMatrix(meta_arr)

    def _read_axis_dict(grp_name: str, prefix: str) -> MetaDataDict:
        ag = cast(h5py.Group, group[grp_name])
        use_json = ag.attrs.get("key_format") == "json"
        keys = [
            _decode_matrix_key(k)
            if use_json
            else (k.decode() if isinstance(k, bytes) else k)
            for k in ag["keys"][:]
        ]
        ns = [x.decode() if isinstance(x, bytes) else str(x) for x in ag["names"][:]]
        us = [x.decode() if isinstance(x, bytes) else str(x) for x in ag["units"][:]]
        cs = [x.decode() if isinstance(x, bytes) else str(x) for x in ag["channels"][:]]
        od: OrderedDict[Any, MetaData] = OrderedDict()
        for k, n, u_, ch in zip(keys, ns, us, cs):
            od[k] = MetaData(
                unit=u.Unit(u_) if u_ else u.dimensionless_unscaled,
                name=n or None,
                channel=ch or None,
            )
        return MetaDataDict(od, expected_size=len(keys), key_prefix=prefix)

    rows = _read_axis_dict("rows", "row")
    cols = _read_axis_dict("cols", "col")

    result = matrix_cls(
        data,
        xindex=xindex,
        meta=meta_matrix,
        rows=rows,
        cols=cols,
        name=name,
        epoch=epoch,
        attrs=attrs,
        dx=cadence,
        **extra_axes,
    )
    if cadence is not None:
        result._dx = cadence
    set_time_reference(result, time_reference)
    return result


def _write_array(group: h5py.Group, val: Any) -> None:
    from gwpy.types import Array as GwpyArray
    from gwpy.types.io.hdf5 import write_hdf5_array

    group.attrs["serializer"] = "array"
    if isinstance(val, GwpyArray):
        group.attrs["array_kind"] = "gwpy_array"
        write_hdf5_array(val, group, path="data")
    elif isinstance(val, u.Quantity):
        group.attrs["array_kind"] = "quantity"
        group.attrs["unit"] = str(val.unit)
        group.create_dataset("data", data=np.asarray(val.value))
    else:
        group.attrs["array_kind"] = "ndarray"
        group.create_dataset("data", data=np.asarray(val))


def _read_array(group: h5py.Group) -> Any:
    from gwpy.types import Array as GwpyArray
    from gwpy.types.io.hdf5 import read_hdf5_array

    ak = group.attrs.get("array_kind", "ndarray")
    if isinstance(ak, bytes):
        ak = ak.decode("utf-8")
    if ak == "gwpy_array":
        return read_hdf5_array(group, path="data", array_type=GwpyArray)
    arr = group["data"][:]
    if ak == "quantity":
        u_str = group.attrs.get("unit", "")
        if isinstance(u_str, bytes):
            u_str = u_str.decode("utf-8")
        return u.Quantity(arr, u.Unit(u_str) if u_str else u.dimensionless_unscaled)
    return arr


def _encode_scalar_json(val: Any) -> str:
    if val is None:
        return json.dumps({"t": "none", "v": None})
    if isinstance(val, (bool, np.bool_)):
        return json.dumps({"t": "bool", "v": bool(val)})
    if isinstance(val, (int, np.integer)):
        return json.dumps({"t": "int", "v": int(val)})
    if isinstance(val, (float, np.floating)):
        fv = float(val)
        return json.dumps(
            {"t": "float", "v": repr(fv) if not math.isfinite(fv) else fv}
        )
    if isinstance(val, (complex, np.complexfloating)):
        cv = complex(val)
        return json.dumps({"t": "complex", "v": [repr(cv.real), repr(cv.imag)]})
    if isinstance(val, str):
        return json.dumps({"t": "str", "v": val})
    raise TypeError(
        f"Scalar payload serializer cannot encode {type(val).__name__!r} without pickle."
    )


def _decode_scalar_json(raw: Any) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    doc = json.loads(raw)
    t = doc["t"]
    v = doc["v"]
    if t == "none":
        return None
    if t == "bool":
        return bool(v)
    if t == "int":
        return int(v)
    if t == "float":
        return float(v)
    if t == "complex":
        return complex(float(v[0]), float(v[1]))
    if t == "str":
        return str(v)
    raise ValueError(f"Unknown scalar payload type tag {t!r}.")


def _write_scalar(group: h5py.Group, val: Any) -> None:
    group.attrs["serializer"] = "scalar"
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("data", data=_encode_scalar_json(val), dtype=dt)


def _read_scalar(group: h5py.Group) -> Any:
    raw = group["data"][()]
    return _decode_scalar_json(raw)


register_payload_serializer("timeseries", _write_timeseries, _read_timeseries)
register_payload_serializer(
    "frequencyseries", _write_frequencyseries, _read_frequencyseries
)
register_payload_serializer("spectrogram", _write_spectrogram, _read_spectrogram)
register_payload_serializer(
    "timeseriesdict", _write_timeseriesdict, _read_timeseriesdict
)
register_payload_serializer(
    "frequencyseriesdict", _write_frequencyseriesdict, _read_frequencyseriesdict
)
register_payload_serializer("matrix", _write_matrix, _read_matrix)
register_payload_serializer("array", _write_array, _read_array)
register_payload_serializer("scalar", _write_scalar, _read_scalar)


# ---------------------------------------------------------------------------
# Metadata encoding / decoding helpers
# ---------------------------------------------------------------------------


def _encode_meta_column(
    values: list[Any],
) -> tuple[str, np.ndarray, Optional[np.ndarray]]:
    """Classify and encode a meta column without pickle."""
    n = len(values)
    na_flags = [is_scalar_nan(v) for v in values]
    non_na = [v for v, is_na in zip(values, na_flags) if not is_na]

    if not non_na:
        return "float", np.full(n, np.nan, dtype=np.float64), None

    if not any(na_flags) and all(isinstance(v, (bool, np.bool_)) for v in non_na):
        return "bool", np.array([bool(v) for v in values], dtype=bool), None

    if not any(na_flags) and all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        for v in non_na
    ):
        return "int", np.array([int(v) for v in values], dtype=np.int64), None

    if all(
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, (bool, np.bool_))
        for v in non_na
    ):
        arr = np.array(
            [np.nan if is_na else float(v) for v, is_na in zip(values, na_flags)],
            dtype=np.float64,
        )
        return "float", arr, None

    if all(isinstance(v, str) for v in non_na):
        str_arr = np.array(
            ["" if is_na else str(v) for v, is_na in zip(values, na_flags)],
            dtype=object,
        )
        na_arr = np.array(na_flags, dtype=bool) if any(na_flags) else None
        return "str", str_arr, na_arr

    # JSON fallback for mixed primitive meta columns
    encoded = [
        "null" if is_na else json.dumps(v.item() if isinstance(v, np.generic) else v)
        for v, is_na in zip(values, na_flags)
    ]
    return "json", np.array(encoded, dtype=object), None


def _write_meta_group(
    root: h5py.Group,
    df: pd.DataFrame,
    span_cols: tuple[str, str],
) -> None:
    meta_grp = root.create_group("metadata")
    s1, s2 = span_cols
    spans = list(df["span"])
    meta_grp.create_dataset(
        s1, data=np.array([float(s[0]) for s in spans], dtype=np.float64)
    )
    meta_grp.create_dataset(
        s2, data=np.array([float(s[1]) for s in spans], dtype=np.float64)
    )

    str_dt = h5py.string_dtype(encoding="utf-8")
    for col_name in df.columns:
        if col_name == "span":
            continue
        encoding, data_arr, na_arr = _encode_meta_column(df[col_name].tolist())
        if encoding in ("str", "json"):
            ds = meta_grp.create_dataset(col_name, data=data_arr, dtype=str_dt)
        else:
            ds = meta_grp.create_dataset(col_name, data=data_arr)
        ds.attrs["meta_encoding"] = encoding
        if na_arr is not None:
            meta_grp.create_dataset(f"{col_name}.__isna__", data=na_arr)


def _read_meta_group(
    root: h5py.Group,
    schema_cols: list[dict[str, Any]],
    span_cols: tuple[str, str],
) -> pd.DataFrame:
    from gwpy.segments import Segment

    meta_grp = cast(h5py.Group, root["metadata"])
    s1, s2 = span_cols
    starts = meta_grp[s1][:]
    ends = meta_grp[s2][:]
    spans = [Segment(float(a), float(b)) for a, b in zip(starts, ends)]
    data: dict[str, list[Any]] = {"span": spans}

    for entry in schema_cols:
        if entry["storage"] != "meta":
            continue
        col_name = entry["name"]
        if col_name == "span":
            continue
        ds = meta_grp[col_name]
        enc = ds.attrs.get("meta_encoding", "")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8")
        raw = ds[:]
        na_key = f"{col_name}.__isna__"
        na_mask = meta_grp[na_key][:] if na_key in meta_grp else None
        if enc == "str":
            vals: list[Any] = []
            for idx, item in enumerate(raw):
                if na_mask is not None and bool(na_mask[idx]):
                    vals.append(np.nan)
                elif isinstance(item, bytes):
                    vals.append(item.decode("utf-8"))
                else:
                    vals.append(str(item))
            data[col_name] = vals
        elif enc == "json":
            vals = []
            for item in raw:
                text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                vals.append(np.nan if text == "null" else json.loads(text))
            data[col_name] = vals
        else:
            data[col_name] = raw.tolist()

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# HDF5 root validation (U-4 / #658) and lazy loader helper
# ---------------------------------------------------------------------------


def _decode_attr_str(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _validate_hdf5_root(root: h5py.Group) -> dict[str, Any]:
    """Validate root schema/version/unknown gwexpy_* attrs and return external attrs."""
    external_attrs: dict[str, Any] = {}
    for attr_name in root.attrs:
        if attr_name.startswith("gwexpy_") or attr_name.startswith("_gwexpy_"):
            if attr_name not in KNOWN_ROOT_ATTRS:
                raise ValueError(
                    f"Unknown gwexpy root attribute {attr_name!r} in SegmentTable HDF5 file."
                )
        else:
            external_attrs[attr_name] = root.attrs[attr_name]

    if "gwexpy_schema" in root.attrs:
        schema_val = _decode_attr_str(root.attrs["gwexpy_schema"])
        if schema_val != SCHEMA_NAME:
            raise ValueError(
                f"Unsupported gwexpy_schema {schema_val!r}; expected {SCHEMA_NAME!r}."
            )

    if "gwexpy_schema_version" not in root.attrs:
        if "metadata" not in root or "schema" not in root:
            raise ValueError(
                "HDF5 root is missing required SegmentTable '/metadata' or '/schema'."
            )
        warnings.warn(
            "Reading legacy SegmentTable HDF5 file without 'gwexpy_schema_version'; "
            f"assuming schema version {SCHEMA_VERSION}.",
            UserWarning,
            stacklevel=4,
        )
    else:
        raw_ver = root.attrs["gwexpy_schema_version"]
        if isinstance(raw_ver, (bool, np.bool_)) or not isinstance(
            raw_ver, (int, np.integer)
        ):
            raise ValueError(
                f"Invalid gwexpy_schema_version {raw_ver!r}; expected an integer."
            )
        ver = int(raw_ver)
        if ver < 1:
            raise ValueError(
                f"Invalid gwexpy_schema_version {ver}; version must be >= 1."
            )
        if ver > MAX_SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SegmentTable HDF5 schema version {ver}; "
                f"maximum supported version is {MAX_SUPPORTED_SCHEMA_VERSION}."
            )

    return external_attrs


def _read_cell_group(cell_grp: h5py.Group, kind: str, *, allow_pickle: bool) -> Any:
    ser = cell_grp.attrs.get("serializer", "")
    if isinstance(ser, bytes):
        ser = ser.decode("utf-8")
    if ser == "pickle":
        if not allow_pickle:
            raise ValueError(
                "HDF5 cell payload is pickled and allow_pickle=False was specified."
            )
        raw_bytes = bytes(cell_grp["pickle"][:])
        return pickle.loads(raw_bytes)  # noqa: S301
    pair = get_payload_serializer(ser or kind)
    if pair is None:
        raise ValueError(
            f"No payload serializer registered for kind/serializer {ser or kind!r}."
        )
    return pair[1](cell_grp)


def _make_hdf5_cell_loader(
    filepath: Path,
    group_path: str,
    col_name: str,
    row_idx: int,
    kind: str,
    allow_pickle: bool,
) -> Callable[[], Any]:
    """Create a file-handle-safe lazy loader that opens HDF5 per read."""

    def _load() -> Any:
        with h5py.File(filepath, "r") as h5f:
            root = h5f if group_path == "/" else cast(h5py.Group, h5f[group_path])
            cell_grp = cast(h5py.Group, root["payload"][col_name][str(row_idx)])
            return _read_cell_group(cell_grp, kind, allow_pickle=allow_pickle)

    return _load


def _restore_error(err_info: Optional[dict[str, str]]) -> BaseException:
    if not err_info:
        return RuntimeError("Recorded failure in SegmentTable HDF5 cell.")
    etype = err_info.get("type", "RuntimeError")
    msg = err_info.get("message", "")
    builtins_map: dict[str, type[Exception]] = {
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "RuntimeError": RuntimeError,
        "OSError": OSError,
        "ZeroDivisionError": ZeroDivisionError,
    }
    exc_cls = builtins_map.get(etype, RuntimeError)
    if exc_cls is RuntimeError and etype != "RuntimeError":
        return RuntimeError(f"{etype}: {msg}" if msg else etype)
    return exc_cls(msg)


def _path_suffix(source: Any) -> str:
    if isinstance(source, (str, os.PathLike)):
        return Path(source).suffix.lower()
    return ""


class _IOMixin:
    """``read()``, ``read_csv()``, ``write()``, and ``provenance`` for SegmentTable."""

    _meta: pd.DataFrame
    _payload: dict[str, list[SegmentCell]]
    _schema: dict[str, str]
    _units: dict[str, Any]
    _provenance: list[dict[str, Any]]
    _external_attrs: dict[str, Any]
    _provenance_doc: Optional[dict[str, Any]]

    @property
    def provenance(self) -> dict[str, Any]:
        """Return provenance metadata and operation history."""
        base = dict(getattr(self, "_provenance_doc", None) or {})
        base["history"] = [dict(entry) for entry in self._provenance]
        return base

    @classmethod
    def read_csv(
        cls,
        filepath: str,
        span_cols: tuple[str, str] = ("start", "end"),
        **kwargs: Any,
    ) -> SegmentTable:
        """Read a :class:`SegmentTable` from a CSV file.

        Parameters
        ----------
        filepath:
            Path to the CSV file.
        span_cols:
            The two column names to use as the ``span`` start and end values.
            Ignored if a column named ``"span"`` already exists (containing
            Segment objects or compatible).
        **kwargs:
            Forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        SegmentTable

        """
        from gwpy.segments import Segment

        from gwexpy.table.segment.core import _coerce_span_value

        df = pd.read_csv(filepath, **kwargs)
        if "span" not in df.columns:
            s1, s2 = span_cols
            df["span"] = [Segment(float(s), float(e)) for s, e in zip(df[s1], df[s2])]
        else:
            df["span"] = [_coerce_span_value(value) for value in df["span"]]
        return cast("SegmentTable", cls(df))  # type: ignore[call-arg]

    @classmethod
    def read(
        cls,
        source: Any,
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> SegmentTable:
        """Read a :class:`SegmentTable` from CSV or HDF5.

        Parameters
        ----------
        source:
            Input file path, path-like, or open :class:`h5py.Group`.
        format:
            Explicit format (``"csv"`` or ``"hdf5"``).  When ``None``,
            inferred from the file extension or object type.
        **kwargs:
            Format-specific keyword arguments forwarded to :meth:`read_csv`
            or the HDF5 reader (e.g., ``lazy=True``, ``path="/"``).

        Returns
        -------
        SegmentTable

        """
        fmt = format.lower() if isinstance(format, str) else format
        if fmt is None:
            if isinstance(source, h5py.Group):
                fmt = "hdf5"
            else:
                suffix = _path_suffix(source)
                if suffix == ".csv":
                    fmt = "csv"
                elif suffix in (".h5", ".hdf5", ".hdf"):
                    fmt = "hdf5"
                elif isinstance(source, (str, os.PathLike)) and os.path.exists(source):
                    if h5py.is_hdf5(source):
                        fmt = "hdf5"

        if fmt == "csv":
            return cls.read_csv(source, **kwargs)
        if fmt in ("hdf5", "h5", "hdf"):
            return cls._read_hdf5(source, **kwargs)
        if fmt is None:
            raise ValueError(
                f"Could not determine format for {source!r}; "
                "specify format='csv' or format='hdf5'."
            )
        raise ValueError(
            f"Unsupported format {format!r} for SegmentTable.read(); "
            "expected 'csv' or 'hdf5'."
        )

    @classmethod
    def _read_hdf5(
        cls,
        source: Any,
        *,
        lazy: bool = True,
        path: str = "/",
        allow_pickle: bool = False,
    ) -> SegmentTable:
        if isinstance(source, h5py.Group):
            root = source if path == "/" else cast(h5py.Group, source[path])
            filename = getattr(source.file, "filename", None)
            file_path = (
                Path(filename).resolve()
                if filename and os.path.exists(filename)
                else None
            )
            group_path = root.name or "/"
            return cls._read_hdf5_group(
                root,
                file_path=file_path,
                group_path=group_path,
                lazy=lazy,
                allow_pickle=allow_pickle,
            )

        file_path = Path(source).resolve()
        with h5py.File(file_path, "r") as h5f:
            root = h5f if path == "/" else cast(h5py.Group, h5f[path])
            group_path = root.name or "/"
            return cls._read_hdf5_group(
                root,
                file_path=file_path,
                group_path=group_path,
                lazy=lazy,
                allow_pickle=allow_pickle,
            )

    @classmethod
    def _read_hdf5_group(
        cls,
        root: h5py.Group,
        *,
        file_path: Optional[Path],
        group_path: str,
        lazy: bool,
        allow_pickle: bool,
    ) -> SegmentTable:
        external_attrs = _validate_hdf5_root(root)

        if "schema" not in root or "metadata" not in root:
            raise ValueError(
                "HDF5 group does not contain required '/schema' and '/metadata' entries."
            )

        raw_schema = root["schema"][()]
        if isinstance(raw_schema, bytes):
            raw_schema = raw_schema.decode("utf-8")
        schema_doc = json.loads(raw_schema)
        schema_cols: list[dict[str, Any]] = schema_doc["columns"]
        span_cols_raw = schema_doc.get("span_cols", ["start", "end"])
        span_cols = (str(span_cols_raw[0]), str(span_cols_raw[1]))

        df = _read_meta_group(root, schema_cols, span_cols)
        table = cast("SegmentTable", cls(df))  # type: ignore[call-arg]
        table._external_attrs = external_attrs

        for entry in schema_cols:
            col_name = entry["name"]
            kind = entry["kind"]
            unit_str = entry.get("unit")
            if unit_str:
                table._units[col_name] = u.Unit(unit_str)
            if entry["storage"] == "meta" and col_name != "span":
                table._schema[col_name] = kind

        n_rows = len(df)
        status_grp = cast(h5py.Group, root["status"]) if "status" in root else None
        payload_grp = cast(h5py.Group, root["payload"]) if "payload" in root else None

        for entry in schema_cols:
            if entry["storage"] != "payload":
                continue
            col_name = entry["name"]
            kind = entry["kind"]
            table._schema[col_name] = kind

            if status_grp is not None and col_name in status_grp:
                raw_statuses = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in status_grp[col_name][:]
                ]
            else:
                raw_statuses = [CellStatus.UNLOADED.value] * n_rows

            err_key = f"{col_name}.__errors__"
            err_list: list[Optional[dict[str, str]]] = [None] * n_rows
            if status_grp is not None and err_key in status_grp:
                raw_errs = status_grp[err_key][()]
                if isinstance(raw_errs, bytes):
                    raw_errs = raw_errs.decode("utf-8")
                err_list = json.loads(raw_errs)

            cells: list[SegmentCell] = []
            col_payload_grp = (
                cast(h5py.Group, payload_grp[col_name])
                if payload_grp is not None and col_name in payload_grp
                else None
            )
            for row_idx in range(n_rows):
                st = CellStatus(raw_statuses[row_idx])
                if st == CellStatus.VALID:
                    if lazy and file_path is not None:
                        loader = _make_hdf5_cell_loader(
                            file_path,
                            group_path,
                            col_name,
                            row_idx,
                            kind,
                            allow_pickle,
                        )
                        cells.append(SegmentCell(loader=loader, cacheable=True))
                    else:
                        if (
                            col_payload_grp is None
                            or str(row_idx) not in col_payload_grp
                        ):
                            raise ValueError(
                                f"Missing HDF5 payload at '/payload/{col_name}/{row_idx}'."
                            )
                        val = _read_cell_group(
                            cast(h5py.Group, col_payload_grp[str(row_idx)]),
                            kind,
                            allow_pickle=allow_pickle,
                        )
                        cells.append(SegmentCell.valid(val))
                elif st == CellStatus.MISSING:
                    cells.append(SegmentCell.missing())
                elif st == CellStatus.MASKED:
                    cells.append(SegmentCell.masked())
                elif st == CellStatus.NOT_APPLICABLE:
                    cells.append(SegmentCell.not_applicable())
                elif st == CellStatus.FAILED:
                    cells.append(SegmentCell.failed(_restore_error(err_list[row_idx])))
                else:
                    cells.append(SegmentCell())
            table._payload[col_name] = cells

        if "provenance" in root:
            raw_prov = root["provenance"][()]
            if isinstance(raw_prov, bytes):
                raw_prov = raw_prov.decode("utf-8")
            prov_doc = json.loads(raw_prov)
            if not allow_pickle and prov_doc.get("used_pickle"):
                raise ValueError(
                    "HDF5 SegmentTable contains pickled payloads and allow_pickle=False was specified."
                )
            table._provenance_doc = prov_doc
            table._provenance = [dict(x) for x in prov_doc.get("history", [])]

        if not allow_pickle and payload_grp is not None:
            for entry in schema_cols:
                if entry["storage"] != "payload":
                    continue
                col_name = entry["name"]
                if col_name not in payload_grp:
                    continue
                col_grp = cast(h5py.Group, payload_grp[col_name])
                for row_key in col_grp.keys():
                    ser = cast(h5py.Group, col_grp[row_key]).attrs.get("serializer", "")
                    if isinstance(ser, bytes):
                        ser = ser.decode("utf-8")
                    if ser == "pickle":
                        raise ValueError(
                            "HDF5 cell payload is pickled and allow_pickle=False was specified."
                        )

        return table

    def write(
        self,
        target: Any,
        format: Optional[str] = None,
        *,
        allow_pickle: bool = False,
        **kwargs: Any,
    ) -> None:
        """Write this :class:`SegmentTable` to HDF5 or CSV.

        Parameters
        ----------
        target:
            Destination file path, path-like, or writable :class:`h5py.Group`.
        format:
            Output format (``"hdf5"`` or ``"csv"``).  Inferred from *target*
            extension or type when ``None``.
        allow_pickle:
            If ``False`` (default), reject unregistered ``object`` payloads
            rather than falling back to pickle.  When ``True``, pickle
            fallback is permitted and recorded in ``/provenance``.
        **kwargs:
            Additional format-specific options (e.g. ``overwrite=True``).

        """
        fmt = format.lower() if isinstance(format, str) else format
        if fmt is None:
            if isinstance(target, h5py.Group):
                fmt = "hdf5"
            else:
                suffix = _path_suffix(target)
                if suffix in (".h5", ".hdf5", ".hdf"):
                    fmt = "hdf5"
                elif suffix == ".csv":
                    fmt = "csv"

        if fmt in ("hdf5", "h5", "hdf"):
            self._write_hdf5(target, allow_pickle=allow_pickle, **kwargs)
            return
        if fmt == "csv":
            overwrite = bool(kwargs.pop("overwrite", True))
            if (
                isinstance(target, (str, os.PathLike))
                and not overwrite
                and os.path.exists(target)
            ):
                raise FileExistsError(f"Target file already exists: {target}")
            df = self.to_pandas(meta_only=True)  # type: ignore[attr-defined]
            df["span"] = [f"({s[0]}, {s[1]})" for s in df["span"]]
            df.to_csv(target, index=False, **kwargs)
            return
        if fmt is None:
            raise ValueError(
                f"Could not determine format for {target!r}; "
                "specify format='hdf5' or format='csv'."
            )
        raise ValueError(
            f"Unsupported format {format!r} for SegmentTable.write(); "
            "expected 'hdf5' or 'csv'."
        )

    def _write_hdf5(
        self,
        target: Any,
        *,
        allow_pickle: bool = False,
        overwrite: bool = True,
        path: str = "/",
        source_files: Optional[list[str]] = None,
    ) -> None:
        # Pre-validate meta encoding and resolve/validate payload cells BEFORE
        # opening/truncating any destination file.
        meta_cols = set(self._meta.columns)
        span_cols = (
            ("__span_start__", "__span_end__")
            if ("start" in meta_cols or "end" in meta_cols)
            else ("start", "end")
        )
        for col_name in self._meta.columns:
            if col_name != "span":
                _encode_meta_column(self._meta[col_name].tolist())

        resolved_payloads: dict[
            str, list[tuple[CellStatus, Any, Optional[BaseException]]]
        ] = {}
        used_pickle = False

        for col_name, cells in self._payload.items():
            kind = self._schema[col_name]
            has_serializer = get_payload_serializer(kind) is not None
            col_entries: list[tuple[CellStatus, Any, Optional[BaseException]]] = []
            for cell in cells:
                if cell.status == CellStatus.VALID:
                    val = cell.get()
                    status = CellStatus.VALID
                    err = None
                elif cell.status == CellStatus.UNLOADED and cell.loader is not None:
                    val = cell.loader()
                    status = CellStatus.VALID
                    err = None
                else:
                    val = _UNSET
                    status = cell.status
                    err = cell.error

                if status == CellStatus.VALID:
                    if not has_serializer:
                        if not allow_pickle:
                            raise ValueError(
                                f"Cannot serialize payload column {col_name!r} "
                                f"with unregistered kind {kind!r} when allow_pickle=False. "
                                "Register a serializer or pass allow_pickle=True."
                            )
                        pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
                        used_pickle = True
                    elif kind == "scalar":
                        _encode_scalar_json(val)
                col_entries.append((status, val, err))
            resolved_payloads[col_name] = col_entries

        if isinstance(target, h5py.Group):
            root = target if path == "/" else target.require_group(path)
            self._populate_hdf5_group(
                root,
                span_cols=span_cols,
                resolved_payloads=resolved_payloads,
                allow_pickle=allow_pickle,
                used_pickle=used_pickle,
                source_files=source_files,
            )
            return

        target_path = Path(target)
        if not overwrite and target_path.exists():
            raise FileExistsError(f"Target file already exists: {target_path}")

        with h5py.File(target_path, "w") as h5f:
            root = h5f if path == "/" else h5f.require_group(path)
            self._populate_hdf5_group(
                root,
                span_cols=span_cols,
                resolved_payloads=resolved_payloads,
                allow_pickle=allow_pickle,
                used_pickle=used_pickle,
                source_files=source_files,
            )

    def _populate_hdf5_group(
        self,
        root: h5py.Group,
        *,
        span_cols: tuple[str, str],
        resolved_payloads: dict[
            str, list[tuple[CellStatus, Any, Optional[BaseException]]]
        ],
        allow_pickle: bool,
        used_pickle: bool,
        source_files: Optional[list[str]],
    ) -> None:
        # Preserve external attrs first
        for k, v in self._external_attrs.items():
            if not (k.startswith("gwexpy_") or k.startswith("_gwexpy_")):
                root.attrs[k] = v

        # Canonical U-4 / #658 root attrs
        root.attrs["gwexpy_schema"] = SCHEMA_NAME
        root.attrs["gwexpy_schema_version"] = int(SCHEMA_VERSION)

        _write_meta_group(root, self._meta, span_cols)

        schema_entries: list[dict[str, Any]] = []
        for col_name in self.columns:  # type: ignore[attr-defined]
            unit_obj = self._units.get(col_name)
            schema_entries.append(
                {
                    "name": col_name,
                    "kind": self._schema[col_name],
                    "storage": "payload" if col_name in self._payload else "meta",
                    "unit": str(unit_obj) if unit_obj is not None else None,
                }
            )
        str_dt = h5py.string_dtype(encoding="utf-8")
        root.create_dataset(
            "schema",
            data=json.dumps({"columns": schema_entries, "span_cols": list(span_cols)}),
            dtype=str_dt,
        )

        status_grp = root.create_group("status")
        payload_grp = root.create_group("payload")
        for col_name, entries in resolved_payloads.items():
            kind = self._schema[col_name]
            status_arr = np.array([st.value for st, _, _ in entries], dtype=object)
            status_grp.create_dataset(col_name, data=status_arr, dtype=str_dt)

            if any(st == CellStatus.FAILED for st, _, _ in entries):
                errs_json = [
                    {"type": type(err).__name__, "message": str(err)}
                    if err is not None
                    else None
                    for st, _, err in entries
                ]
                status_grp.create_dataset(
                    f"{col_name}.__errors__",
                    data=json.dumps(errs_json),
                    dtype=str_dt,
                )

            col_grp = payload_grp.create_group(col_name)
            pair = get_payload_serializer(kind)
            for row_idx, (st, val, _) in enumerate(entries):
                if st != CellStatus.VALID:
                    continue
                cell_grp = col_grp.create_group(str(row_idx))
                if pair is not None:
                    pair[0](cell_grp, val)
                else:
                    assert allow_pickle
                    raw = pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
                    cell_grp.create_dataset(
                        "pickle", data=np.frombuffer(raw, dtype=np.uint8)
                    )
                    cell_grp.attrs["serializer"] = "pickle"

        try:
            from gwexpy._version import __version__ as gwexpy_version
        except Exception:
            gwexpy_version = "unknown"

        prov_doc = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "gwexpy_version": gwexpy_version,
            "created_utc": datetime.now(UTC).isoformat(),
            "allow_pickle": bool(allow_pickle),
            "used_pickle": bool(used_pickle),
            "source_files": list(source_files) if source_files is not None else [],
            "history": list(self._provenance),
        }
        root.create_dataset("provenance", data=json.dumps(prov_doc), dtype=str_dt)
