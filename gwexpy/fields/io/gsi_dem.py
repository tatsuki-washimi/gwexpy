"""Reader for Geospatial Information Authority of Japan (GSI) FGD DEM XML/ZIP files."""

from __future__ import annotations

import os
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from astropy import units as u

from ..geo import GeoMetadata
from ._registration import register_field_format

if TYPE_CHECKING:
    from ..scalar import ScalarField

__all__ = [
    "identify_gsi_dem",
    "list_fields_gsi_dem",
    "read_scalarfield_gsi_dem",
]

_GSI_FILENAME_RE = re.compile(r"^FG-GML-.*DEM.*\.(xml|zip)$", re.IGNORECASE)

_SRS_MAP: dict[str, tuple[str, str]] = {
    "fguuid:jgd2011.bl": ("EPSG:6668", "T.P. (JGD2011 vertical)"),
    "fguuid:jgd2000.bl": ("EPSG:4612", "T.P. (JGD2000 vertical)"),
}


@dataclass(frozen=True)
class _TileHeader:
    label: str
    srs_name: str
    crs: str
    vertical_datum: str
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float
    low_x: int
    low_y: int
    nx: int
    ny: int
    dlon: float
    dlat: float


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem:
        if _local_tag(child.tag) == name:
            return child
    return None


def _find_descendant(elem: ET.Element, name: str) -> ET.Element | None:
    for node in elem.iter():
        if _local_tag(node.tag) == name:
            return node
    return None


def _resolve_srs(srs_name: str) -> tuple[str, str]:
    key = srs_name.strip()
    lower = key.lower()
    if lower in _SRS_MAP:
        return _SRS_MAP[lower]
    if lower.startswith("epsg:"):
        return f"EPSG:{key.split(':', 1)[1]}", "T.P."
    if "epsg::" in lower:
        code = key.rsplit("::", 1)[-1]
        return f"EPSG:{code}", "T.P."
    raise ValueError(f"Unsupported or unrecognized GSI DEM srsName: {srs_name!r}")


def _looks_like_fgd_dem_xml_bytes(head: bytes) -> bool:
    if b"FGD_GMLSchema" in head or b"fgd.gsi.go.jp" in head:
        return b"<DEM" in head or b":DEM" in head or b"fgd" in head.lower()
    return (b"<Dataset" in head and b"<DEM" in head) or (
        b"<DEM" in head and b"<tupleList" in head
    )


def _probe_zip_for_gsi_dem(target: Any) -> bool:
    try:
        with zipfile.ZipFile(target, "r") as zf:
            xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            for xml_name in xml_names[:4]:
                with zf.open(xml_name, "r") as member:
                    head = member.read(4096)
                if _looks_like_fgd_dem_xml_bytes(head):
                    return True
    except (zipfile.BadZipFile, OSError, KeyError, RuntimeError):
        return False
    return False


def identify_gsi_dem(
    origin: Any,
    filepath: str | None,
    fileobj: Any,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Identify whether *filepath* / *fileobj* is a GSI FGD DEM XML or ZIP archive."""
    if fileobj is not None and hasattr(fileobj, "read") and hasattr(fileobj, "seek"):
        try:
            pos = fileobj.tell()
            head = fileobj.read(4096)
            fileobj.seek(pos)
            if isinstance(head, str):
                head = head.encode("utf-8", errors="ignore")
            if head.startswith(b"PK\x03\x04"):
                return _probe_zip_for_gsi_dem(fileobj)
            if _looks_like_fgd_dem_xml_bytes(head):
                return True
        except (OSError, ValueError):
            pass

    if filepath is not None:
        path = Path(filepath)
        if path.is_file():
            try:
                with path.open("rb") as fh:
                    head = fh.read(4096)
                if head.startswith(b"PK\x03\x04") or path.suffix.lower() == ".zip":
                    return _probe_zip_for_gsi_dem(path)
                if path.suffix.lower() == ".xml" and _looks_like_fgd_dem_xml_bytes(
                    head
                ):
                    return True
            except OSError:
                return False
        elif _GSI_FILENAME_RE.match(path.name):
            return True
    return False


def _iter_xml_inputs(source: Any) -> list[tuple[str, bytes]]:
    """Expand XML/ZIP path(s) or binary file-like input into ``(label, xml_bytes)`` pairs."""
    if hasattr(source, "read"):
        pos = source.tell() if hasattr(source, "seek") else None
        raw = source.read()
        if pos is not None:
            source.seek(pos)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        name = str(getattr(source, "name", "<stream>"))
        if raw.startswith(b"PK\x03\x04"):
            return _extract_zip_bytes(raw, name)
        return [(name, raw)]

    if isinstance(source, (str, os.PathLike)):
        items: Sequence[Any] = (source,)
    elif isinstance(source, Sequence):
        if len(source) == 0:
            raise ValueError("GSI DEM source sequence must not be empty.")
        items = source
    else:
        raise TypeError(f"Unsupported GSI DEM source type: {type(source).__name__}.")

    outputs: list[tuple[str, bytes]] = []
    for item in items:
        path = Path(os.fspath(item))
        if not path.is_file():
            raise FileNotFoundError(f"GSI DEM file not found: {path}")
        if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                xml_names = sorted(
                    n
                    for n in zf.namelist()
                    if n.lower().endswith(".xml") and not n.endswith("/")
                )
                if not xml_names:
                    raise ValueError(f"ZIP archive contains no XML files: {path}")
                for member in xml_names:
                    outputs.append((f"{path}::{member}", zf.read(member)))
        else:
            outputs.append((str(path), path.read_bytes()))
    return outputs


def _extract_zip_bytes(raw: bytes, label: str) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(BytesIO(raw), "r") as zf:
        xml_names = sorted(
            n
            for n in zf.namelist()
            if n.lower().endswith(".xml") and not n.endswith("/")
        )
        if not xml_names:
            raise ValueError(f"ZIP stream contains no XML files: {label}")
        return [(f"{label}::{member}", zf.read(member)) for member in xml_names]


def _extract_dem_elements(root: ET.Element, label: str) -> list[ET.Element]:
    if _local_tag(root.tag) == "DEM":
        return [root]
    dems = [el for el in root.iter() if _local_tag(el.tag) == "DEM"]
    if not dems:
        raise ValueError(f"No <DEM> coverage element found in {label}.")
    return dems


def _parse_tile_header(dem_elem: ET.Element, label: str) -> _TileHeader:
    cov = _find_child(dem_elem, "coverage")
    coverage = cov if cov is not None else dem_elem
    envelope = _find_descendant(coverage, "Envelope")
    if envelope is None:
        raise ValueError(f"Missing gml:Envelope in GSI DEM {label}.")
    srs_name = envelope.attrib.get("srsName", "").strip()
    if not srs_name:
        raise ValueError(f"Missing srsName on Envelope in GSI DEM {label}.")
    crs, vertical_datum = _resolve_srs(srs_name)

    lower_el = _find_child(envelope, "lowerCorner")
    upper_el = _find_child(envelope, "upperCorner")
    if lower_el is None or upper_el is None or not lower_el.text or not upper_el.text:
        raise ValueError(f"Missing lowerCorner/upperCorner in GSI DEM {label}.")

    lat_min, lon_min = (float(tok) for tok in lower_el.text.split()[:2])
    lat_max, lon_max = (float(tok) for tok in upper_el.text.split()[:2])
    if not (
        np.isfinite(lat_min)
        and np.isfinite(lon_min)
        and np.isfinite(lat_max)
        and np.isfinite(lon_max)
    ):
        raise ValueError(f"Non-finite Envelope corners in GSI DEM {label}.")
    if lon_max <= lon_min or lat_max <= lat_min:
        raise ValueError(f"Invalid Envelope ordering in GSI DEM {label}.")

    grid_env = _find_descendant(coverage, "GridEnvelope")
    if grid_env is None:
        raise ValueError(f"Missing gml:GridEnvelope in GSI DEM {label}.")
    low_el = _find_child(grid_env, "low")
    high_el = _find_child(grid_env, "high")
    if low_el is None or high_el is None or not low_el.text or not high_el.text:
        raise ValueError(f"Missing low/high limits in GSI DEM {label}.")
    low_x, low_y = (int(tok) for tok in low_el.text.split()[:2])
    high_x, high_y = (int(tok) for tok in high_el.text.split()[:2])
    nx = high_x - low_x + 1
    ny = high_y - low_y + 1
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Non-positive GridEnvelope size ({nx}, {ny}) in {label}.")

    dlon = (lon_max - lon_min) / float(nx)
    dlat = (lat_max - lat_min) / float(ny)
    return _TileHeader(
        label=label,
        srs_name=srs_name,
        crs=crs,
        vertical_datum=vertical_datum,
        lat_min=lat_min,
        lon_min=lon_min,
        lat_max=lat_max,
        lon_max=lon_max,
        low_x=low_x,
        low_y=low_y,
        nx=nx,
        ny=ny,
        dlon=dlon,
        dlat=dlat,
    )


def _parse_tile_values(dem_elem: ET.Element, hdr: _TileHeader) -> np.ndarray:
    """Parse ``tupleList`` values into an ``(nx, ny)`` array ordered ``(lon asc, lat desc)``."""
    cov = _find_child(dem_elem, "coverage")
    coverage = cov if cov is not None else dem_elem
    seq_rule = _find_descendant(coverage, "sequenceRule")
    if seq_rule is None:
        raise ValueError(f"Missing gml:sequenceRule in GSI DEM {hdr.label}.")
    axis_order_attr = seq_rule.attrib.get("axisOrder", "").strip()
    if axis_order_attr != "+x-y":
        raise ValueError(
            f"Unsupported GSI DEM sequenceRule axisOrder {axis_order_attr!r} "
            f"in {hdr.label}; expected '+x-y'."
        )

    start_point = _find_descendant(coverage, "startPoint")
    if start_point is not None and start_point.text and start_point.text.strip():
        sx, sy = (int(tok) for tok in start_point.text.split()[:2])
    else:
        sx, sy = hdr.low_x, hdr.low_y

    offset = (sx - hdr.low_x) + (sy - hdr.low_y) * hdr.nx
    total = hdr.nx * hdr.ny
    if offset < 0 or offset >= total:
        raise ValueError(
            f"Invalid startPoint ({sx}, {sy}) relative to low=({hdr.low_x}, {hdr.low_y}) "
            f"and size=({hdr.nx}, {hdr.ny}) in {hdr.label}."
        )

    tuple_el = _find_descendant(coverage, "tupleList")
    if tuple_el is None:
        raise ValueError(f"Missing gml:tupleList in GSI DEM {hdr.label}.")
    raw_text = tuple_el.text or ""

    values: list[float] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "," in line:
            token = line.rsplit(",", 1)[-1].strip()
            val = float(token)
            values.append(np.nan if val == -9999.0 else val)
        else:
            for token in line.split():
                val = float(token)
                values.append(np.nan if val == -9999.0 else val)

    if len(values) + offset > total:
        raise ValueError(
            f"tupleList length {len(values)} with startPoint offset {offset} "
            f"exceeds grid cell count {total} in {hdr.label}."
        )

    flat = np.full(total, np.nan, dtype=np.float64)
    if values:
        flat[offset : offset + len(values)] = np.asarray(values, dtype=np.float64)

    grid_yx = flat.reshape((hdr.ny, hdr.nx))
    return np.ascontiguousarray(grid_yx.T)


def _tiles_intersect_bounds(
    hdr: _TileHeader,
    bounds: tuple[float, float, float, float] | None,
) -> bool:
    if bounds is None:
        return True
    b_xmin, b_ymin, b_xmax, b_ymax = bounds
    return not (
        hdr.lon_max <= b_xmin
        or hdr.lon_min >= b_xmax
        or hdr.lat_max <= b_ymin
        or hdr.lat_min >= b_ymax
    )


def _validate_bounds(bounds: tuple[float, float, float, float] | None) -> None:
    if bounds is None:
        return
    if len(bounds) != 4:
        raise ValueError("bounds must be a 4-tuple (xmin, ymin, xmax, ymax).")
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    if not (
        np.isfinite(xmin)
        and np.isfinite(ymin)
        and np.isfinite(xmax)
        and np.isfinite(ymax)
    ):
        raise ValueError("bounds coordinates must be finite.")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            f"Invalid bounds {bounds}: require xmin < xmax and ymin < ymax."
        )


def list_fields_gsi_dem(source: Any, **kwargs: Any) -> list[str]:
    """List available field names in GSI DEM source without parsing elevation arrays."""
    xml_entries = _iter_xml_inputs(source)
    found_dem = False
    for label, raw in xml_entries:
        root = ET.fromstring(raw)
        dems = _extract_dem_elements(root, label)
        for dem_el in dems:
            _parse_tile_header(dem_el, label)
            found_dem = True
    if not found_dem:
        raise ValueError("No DEM coverages found in GSI DEM source.")
    return ["elevation"]


def read_scalarfield_gsi_dem(
    source: Any,
    name: str | None = None,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    axis_order: Literal["native", "ascending"] = "native",
) -> ScalarField:
    """Read one or more GSI FGD DEM XML/ZIP files into a :class:`ScalarField`.

    Parameters
    ----------
    source : str, Path, file-like, or sequence of paths
        Input GSI DEM ``.xml`` or ``.zip`` archive(s). Multiple tiles are
        automatically mosaicked on a common lat/lon lattice.
    name : str, optional
        Field name selector. Only ``None`` or ``'elevation'`` is valid.
    bounds : tuple of (xmin, ymin, xmax, ymax), optional
        Longitude/latitude bounding box for tile pre-filtering and cropping.
    axis_order : {'native', 'ascending'}, default 'native'
        ``'native'`` keeps descending latitude order from FGD ``+x-y`` scan order;
        ``'ascending'`` reverses latitude to strictly increasing order.

    Returns
    -------
    ScalarField
        4D field with shape ``(1, n_lon, n_lat, 1)``, ``axis_names=('t', 'lon', 'lat', 'z')``,
        ``space_domain={'lon': 'angle', 'lat': 'angle', 'z': 'real'}``,
        elevation unit ``m``, and active :class:`GeoMetadata`.

    """
    from ..scalar import ScalarField

    if name is not None and name != "elevation":
        raise KeyError(
            f"Field {name!r} not found in GSI DEM source (available: ['elevation'])."
        )
    if axis_order not in ("native", "ascending"):
        raise ValueError(
            f"axis_order must be 'native' or 'ascending', got {axis_order!r}."
        )
    _validate_bounds(bounds)

    xml_entries = _iter_xml_inputs(source)
    parsed_tiles: list[tuple[_TileHeader, ET.Element]] = []
    for label, raw in xml_entries:
        root = ET.fromstring(raw)
        for dem_el in _extract_dem_elements(root, label):
            hdr = _parse_tile_header(dem_el, label)
            if _tiles_intersect_bounds(hdr, bounds):
                parsed_tiles.append((hdr, dem_el))

    if not parsed_tiles:
        raise ValueError(
            f"No GSI DEM tiles intersect bounds {bounds}."
            if bounds is not None
            else "No GSI DEM tiles found in source."
        )

    ref_hdr = parsed_tiles[0][0]
    ref_dlon = ref_hdr.dlon
    ref_dlat = ref_hdr.dlat
    ref_lon_center0 = ref_hdr.lon_min + 0.5 * ref_dlon
    ref_lat_center0 = ref_hdr.lat_max - 0.5 * ref_dlat

    offsets: list[tuple[int, int, _TileHeader, ET.Element]] = []
    for hdr, dem_el in parsed_tiles:
        if hdr.crs != ref_hdr.crs:
            raise ValueError(
                f"Cannot mosaic GSI DEM tiles with mismatched CRS: "
                f"{ref_hdr.crs!r} vs {hdr.crs!r} ({hdr.label})."
            )
        if (
            abs(hdr.dlon - ref_dlon) / ref_dlon > 1e-6
            or abs(hdr.dlat - ref_dlat) / ref_dlat > 1e-6
        ):
            raise ValueError(
                f"Cannot mosaic GSI DEM tiles with mismatched resolution: "
                f"({ref_dlon}, {ref_dlat}) vs ({hdr.dlon}, {hdr.dlat}) ({hdr.label})."
            )
        lon_c0 = hdr.lon_min + 0.5 * ref_dlon
        lat_c0 = hdr.lat_max - 0.5 * ref_dlat
        fx = (lon_c0 - ref_lon_center0) / ref_dlon
        fy = (ref_lat_center0 - lat_c0) / ref_dlat
        ix0 = round(fx)
        iy0 = round(fy)
        if abs(fx - ix0) > 0.01 or abs(fy - iy0) > 0.01:
            raise ValueError(
                f"Cannot mosaic GSI DEM tile {hdr.label}: pixel lattice misalignment "
                f"exceeds 0.01 px (fractional offset: dx={abs(fx - ix0):.4g}, dy={abs(fy - iy0):.4g})."
            )
        offsets.append((ix0, iy0, hdr, dem_el))

    min_ix = min(ix0 for ix0, _, _, _ in offsets)
    max_ix = max(ix0 + hdr.nx for ix0, _, hdr, _ in offsets)
    min_iy = min(iy0 for _, iy0, _, _ in offsets)
    max_iy = max(iy0 + hdr.ny for _, iy0, hdr, _ in offsets)

    total_nx = max_ix - min_ix
    total_ny = max_iy - min_iy

    mosaic = np.full((total_nx, total_ny), np.nan, dtype=np.float64)
    seen = np.zeros((total_nx, total_ny), dtype=bool)
    warned_overlap = False

    for ix0, iy0, hdr, dem_el in offsets:
        tile_xy = _parse_tile_values(dem_el, hdr)
        gx0 = ix0 - min_ix
        gy0 = iy0 - min_iy
        sl_x = slice(gx0, gx0 + hdr.nx)
        sl_y = slice(gy0, gy0 + hdr.ny)
        if not warned_overlap and np.any(seen[sl_x, sl_y]):
            warnings.warn(
                "Overlapping GSI DEM tiles detected; later tile values overwrite earlier ones.",
                UserWarning,
                stacklevel=2,
            )
            warned_overlap = True
        mosaic[sl_x, sl_y] = tile_xy
        seen[sl_x, sl_y] = True

    lon_vals = (
        ref_lon_center0 + (np.arange(min_ix, max_ix, dtype=np.float64)) * ref_dlon
    )
    lat_vals = (
        ref_lat_center0 - (np.arange(min_iy, max_iy, dtype=np.float64)) * ref_dlat
    )

    if bounds is not None:
        b_xmin, b_ymin, b_xmax, b_ymax = bounds
        tol_x = 1e-12 * ref_dlon
        tol_y = 1e-12 * ref_dlat
        lon_mask = (lon_vals >= b_xmin - tol_x) & (lon_vals <= b_xmax + tol_x)
        lat_mask = (lat_vals >= b_ymin - tol_y) & (lat_vals <= b_ymax + tol_y)
        if not np.any(lon_mask):
            lon_mask = ((lon_vals + 0.5 * ref_dlon) > b_xmin + tol_x) & (
                (lon_vals - 0.5 * ref_dlon) < b_xmax - tol_x
            )
        if not np.any(lat_mask):
            lat_mask = ((lat_vals + 0.5 * ref_dlat) > b_ymin + tol_y) & (
                (lat_vals - 0.5 * ref_dlat) < b_ymax - tol_y
            )
        if not np.any(lon_mask) or not np.any(lat_mask):
            raise ValueError(f"Cropped GSI DEM region for bounds {bounds} is empty.")
        lon_vals = lon_vals[lon_mask]
        lat_vals = lat_vals[lat_mask]
        mosaic = mosaic[np.ix_(lon_mask, lat_mask)]

    if np.all(np.isnan(mosaic)):
        raise ValueError("GSI DEM contains only missing (-9999) elevation values.")

    dy_actual = -ref_dlat
    if axis_order == "ascending" and lat_vals.size >= 2 and lat_vals[1] < lat_vals[0]:
        lat_vals = lat_vals[::-1].copy()
        mosaic = mosaic[:, ::-1]
        dy_actual = ref_dlat

    x_ul = float(lon_vals[0]) - 0.5 * ref_dlon
    y_ul = float(lat_vals[0]) - 0.5 * dy_actual
    transform = (x_ul, float(ref_dlon), 0.0, y_ul, 0.0, float(dy_actual))

    if isinstance(source, (str, os.PathLike)):
        src_tuple: tuple[str, ...] = (str(source),)
    elif isinstance(source, Sequence):
        src_tuple = tuple(str(os.fspath(s)) for s in source)
    else:
        src_tuple = (str(getattr(source, "name", "<stream>")),)

    data_4d = np.ascontiguousarray(mosaic)[np.newaxis, :, :, np.newaxis]
    field = ScalarField(
        data_4d,
        unit=u.m,
        name=name or "elevation",
        axis0=u.Quantity([0.0], u.s),
        axis1=u.Quantity(lon_vals, u.deg),
        axis2=u.Quantity(lat_vals, u.deg),
        axis3=u.Quantity([0.0], u.m),
        axis_names=("t", "lon", "lat", "z"),
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )
    field.geo = GeoMetadata(
        crs=ref_hdr.crs,
        transform=transform,
        vertical_datum=ref_hdr.vertical_datum,
        nodata=-9999.0,
        band=1,
        source=src_tuple,
        source_format="gsi-dem",
        active=True,
    )
    return field


register_field_format(
    "gsi-dem",
    reader_scalar=read_scalarfield_gsi_dem,
    identifier=identify_gsi_dem,
    lister=list_fields_gsi_dem,
    aliases=("fgd-dem",),
)
