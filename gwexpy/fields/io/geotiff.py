"""GeoTIFF reader and writer for :class:`~gwexpy.fields.ScalarField` (via rasterio)."""

from __future__ import annotations

import importlib.util
import math
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from astropy import units as u

from gwexpy.io.utils import ensure_dependency

from ..geo import GeoMetadata
from ._registration import register_field_format

if TYPE_CHECKING:
    from ..scalar import ScalarField

__all__ = [
    "list_fields_geotiff",
    "read_scalarfield_geotiff",
    "write_scalarfield_geotiff",
]

_TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None


def _identify_geotiff_magic(
    origin: Any,
    filepath: str | None,
    fileobj: Any,
    *args: Any,
    **kwargs: Any,
) -> bool:
    try:
        if (
            fileobj is not None
            and hasattr(fileobj, "read")
            and hasattr(fileobj, "seek")
        ):
            pos = fileobj.tell()
            head = fileobj.read(4)
            fileobj.seek(pos)
            return head in _TIFF_MAGICS
        if filepath is not None and os.path.isfile(filepath):
            with open(filepath, "rb") as fh:
                return fh.read(4) in _TIFF_MAGICS
    except OSError:
        return False
    return False


def _parse_geotiff_unit(unit_tag: str | None) -> u.UnitBase:
    if not unit_tag or not str(unit_tag).strip():
        return u.dimensionless_unscaled
    text = str(unit_tag).strip()
    aliases = {
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
    }
    text = aliases.get(text.lower(), text)
    try:
        return u.Unit(text)
    except ValueError:
        if text.lower().endswith("s") and len(text) > 1:
            try:
                return u.Unit(text[:-1])
            except ValueError:
                pass
        warnings.warn(
            f"Unrecognized GeoTIFF band unit {unit_tag!r}; falling back to dimensionless.",
            UserWarning,
            stacklevel=3,
        )
        return u.dimensionless_unscaled


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


def list_fields_geotiff(source: Any, **kwargs: Any) -> list[str]:
    """List band names in a GeoTIFF raster without loading full pixel arrays."""
    rasterio = ensure_dependency("rasterio", extra="geospatial")
    with rasterio.open(source) as src:
        return [f"band_{i}" for i in range(1, src.count + 1)]


def read_scalarfield_geotiff(
    source: Any,
    name: str | None = None,
    *,
    band: int | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    axis_order: Literal["native", "ascending"] = "native",
    unit: str | u.UnitBase | None = None,
    nodata: float | None = None,
    vertical_datum: str = "geoid",
) -> ScalarField:
    """Read a GeoTIFF band into a :class:`~gwexpy.fields.ScalarField`.

    Parameters
    ----------
    source : str, Path, or file-like
        Path or file object readable by ``rasterio.open``.
    name : str, optional
        Band selector (e.g. ``'band_1'`` or a band description). Defaults to ``band_1``.
    band : int, optional
        1-based band index. Overrides default band 1 when ``name`` is omitted.
    bounds : tuple of (xmin, ymin, xmax, ymax), optional
        Spatial window in source CRS coordinates.
    axis_order : {'native', 'ascending'}, default 'native'
        ``'native'`` preserves north-up descending y order; ``'ascending'`` flips
        y to strictly increasing order.
    unit : str or UnitBase, optional
        Explicit value unit overriding the GeoTIFF band unit tag.
    nodata : float, optional
        Explicit nodata sentinel overriding the dataset nodata value.
    vertical_datum : str, default 'geoid'
        Vertical datum label recorded in :class:`GeoMetadata`.

    """
    from ..scalar import ScalarField

    rasterio = ensure_dependency("rasterio", extra="geospatial")
    from rasterio.windows import Window
    from rasterio.windows import from_bounds as window_from_bounds

    if axis_order not in ("native", "ascending"):
        raise ValueError(
            f"axis_order must be 'native' or 'ascending', got {axis_order!r}."
        )
    _validate_bounds(bounds)

    with rasterio.open(source) as src:
        available = [f"band_{i}" for i in range(1, src.count + 1)]

        if band is not None:
            band_idx = int(band)
            if band_idx < 1 or band_idx > src.count:
                raise ValueError(
                    f"GeoTIFF band index {band_idx} out of range [1, {src.count}]."
                )
            if name is not None:
                desc = (
                    src.descriptions[band_idx - 1]
                    if src.descriptions and len(src.descriptions) >= band_idx
                    else None
                )
                if name != f"band_{band_idx}" and name != desc:
                    raise ValueError(
                        f"Conflicting band selector name={name!r} and band={band_idx}."
                    )
            field_name = name or f"band_{band_idx}"
        elif name is not None:
            if name in available:
                band_idx = available.index(name) + 1
            elif src.descriptions and name in src.descriptions:
                band_idx = list(src.descriptions).index(name) + 1
            else:
                raise KeyError(
                    f"Field {name!r} not found in GeoTIFF {source!r} "
                    f"(available: {available})."
                )
            field_name = name
        else:
            band_idx = 1
            field_name = "band_1"

        if src.crs is None:
            raise ValueError("GeoTIFF dataset has no CRS metadata.")

        c, a, b, f, d, e = src.transform.to_gdal()
        if abs(b) > 1e-12 or abs(d) > 1e-12:
            raise ValueError(
                "Rotated or sheared GeoTIFF transforms (b != 0 or d != 0) are not supported."
            )
        if a <= 0:
            raise ValueError(
                f"GeoTIFF must have positive x pixel width (dx > 0), got dx={a}."
            )
        if e >= 0:
            raise ValueError(
                f"GeoTIFF must have north-up transform (dy < 0), got dy={e}."
            )

        if bounds is not None:
            b_xmin, b_ymin, b_xmax, b_ymax = bounds
            raw_win = window_from_bounds(
                b_xmin, b_ymin, b_xmax, b_ymax, transform=src.transform
            )
            col_off = max(0, int(math.floor(raw_win.col_off + 1e-9)))
            row_off = max(0, int(math.floor(raw_win.row_off + 1e-9)))
            col_end = min(
                src.width, int(math.ceil(raw_win.col_off + raw_win.width - 1e-9))
            )
            row_end = min(
                src.height, int(math.ceil(raw_win.row_off + raw_win.height - 1e-9))
            )
            if col_end <= col_off or row_end <= row_off:
                raise ValueError(f"Bounds {bounds} do not intersect GeoTIFF extent.")
            win = Window(col_off, row_off, col_end - col_off, row_end - row_off)
            win_transform = src.window_transform(win)
            data_yx = src.read(band_idx, window=win, out_dtype="float64")
        else:
            win_transform = src.transform
            data_yx = src.read(band_idx, out_dtype="float64")

        eff_nodata = src.nodata if nodata is None else nodata
        if eff_nodata is not None:
            if isinstance(eff_nodata, float) and np.isnan(eff_nodata):
                pass
            else:
                data_yx = np.where(data_yx == eff_nodata, np.nan, data_yx)

        scale = (
            float(src.scales[band_idx - 1])
            if src.scales and len(src.scales) >= band_idx
            else 1.0
        )
        offset = (
            float(src.offsets[band_idx - 1])
            if src.offsets and len(src.offsets) >= band_idx
            else 0.0
        )
        if scale != 1.0 or offset != 0.0:
            data_yx = data_yx * scale + offset

        if unit is not None:
            resolved_unit = u.Unit(unit)
        else:
            band_unit_tag = (
                src.units[band_idx - 1]
                if src.units and len(src.units) >= band_idx
                else None
            )
            resolved_unit = _parse_geotiff_unit(band_unit_tag)

        is_geo = bool(src.crs.is_geographic)
        crs_str = src.crs.to_string() or src.crs.to_wkt()
        if not is_geo and getattr(src.crs, "axis_info", None):
            factor = float(src.crs.axis_info[0].unit_conversion_factor)
            proj_unit = u.m if np.isclose(factor, 1.0) else u.m * factor
        else:
            proj_unit = u.m

        wc, wa, _, wf, _, we = win_transform.to_gdal()

    ny, nx = data_yx.shape
    x_vals = wc + (np.arange(nx, dtype=np.float64) + 0.5) * wa
    y_vals = wf + (np.arange(ny, dtype=np.float64) + 0.5) * we
    data_xy = np.ascontiguousarray(data_yx.T)

    dy_actual = float(we)
    if axis_order == "ascending" and y_vals.size >= 2 and y_vals[1] < y_vals[0]:
        y_vals = y_vals[::-1].copy()
        data_xy = np.ascontiguousarray(data_xy[:, ::-1])
        dy_actual = -float(we)

    x_ul = float(x_vals[0]) - 0.5 * float(wa)
    y_ul = float(y_vals[0]) - 0.5 * dy_actual
    tf_tuple = (x_ul, float(wa), 0.0, y_ul, 0.0, dy_actual)

    if is_geo:
        axis_unit = u.deg
        ax_names = ("t", "lon", "lat", "z")
        s_dom: str | dict[str, str] = {"lon": "angle", "lat": "angle", "z": "real"}
    else:
        axis_unit = proj_unit
        ax_names = ("t", "x", "y", "z")
        s_dom = "real"

    src_tuple = (str(source),) if isinstance(source, (str, os.PathLike)) else ()
    field = ScalarField(
        data_xy[np.newaxis, :, :, np.newaxis],
        unit=resolved_unit,
        name=field_name,
        axis0=u.Quantity([0.0], u.s),
        axis1=u.Quantity(x_vals, axis_unit),
        axis2=u.Quantity(y_vals, axis_unit),
        axis3=u.Quantity([0.0], u.m),
        axis_names=ax_names,
        space_domain=s_dom,
    )
    field.geo = GeoMetadata(
        crs=crs_str,
        transform=tf_tuple,
        vertical_datum=vertical_datum,
        nodata=eff_nodata,
        band=band_idx,
        source=src_tuple,
        source_format="geotiff",
        active=True,
    )
    return field


def write_scalarfield_geotiff(
    field: ScalarField,
    target: Any,
    *,
    crs: str | None = None,
    nodata: float | None = -9999.0,
    compress: str | None = "deflate",
) -> None:
    """Write a 2D spatial :class:`~gwexpy.fields.ScalarField` to a north-up GeoTIFF."""
    rasterio = ensure_dependency("rasterio", extra="geospatial")
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    if field.ndim != 4 or field.shape[0] != 1 or field.shape[3] != 1:
        raise ValueError(
            "GeoTIFF export requires a 2D spatial ScalarField with shape (1, nx, ny, 1)."
        )

    if crs is not None:
        crs_str = crs
    elif field.geo is not None and field.geo.active and field.geo.crs:
        crs_str = field.geo.crs
    else:
        raise ValueError(
            "GeoTIFF export requires active GeoMetadata with 'crs' "
            "(or pass explicit crs=...)."
        )

    dst_crs = CRS.from_user_input(crs_str)
    if dst_crs.is_geographic:
        coord_unit = u.deg
    else:
        factor = (
            float(dst_crs.axis_info[0].unit_conversion_factor)
            if getattr(dst_crs, "axis_info", None)
            else 1.0
        )
        coord_unit = u.m if np.isclose(factor, 1.0) else u.m * factor

    try:
        x_vals = np.asarray(
            u.Quantity(field._axis1_index, copy=False).to_value(coord_unit),
            dtype=np.float64,
        )
        y_vals = np.asarray(
            u.Quantity(field._axis2_index, copy=False).to_value(coord_unit),
            dtype=np.float64,
        )
    except u.UnitConversionError as exc:
        raise ValueError(
            f"Spatial axis units incompatible with GeoTIFF CRS {crs_str!r}: {exc}"
        ) from exc

    nx, ny = x_vals.size, y_vals.size
    if nx == 0 or ny == 0:
        raise ValueError("Cannot write empty spatial field to GeoTIFF.")

    tf_fallback = field.geo.transform if field.geo is not None else None
    if nx >= 2:
        if not field.is_axis_regular(1):
            raise ValueError("GeoTIFF export requires a uniformly spaced x axis.")
        dx_q = field.axis_mean_delta(1)
        if dx_q is None:
            raise ValueError("GeoTIFF export failed to compute x pixel spacing.")
        dx = float(u.Quantity(dx_q, copy=False).to_value(coord_unit))
    elif tf_fallback is not None and len(tf_fallback) >= 6:
        dx = float(tf_fallback[1])
    else:
        raise ValueError(
            "Cannot determine pixel width for singleton x axis without GeoMetadata.transform."
        )

    if ny >= 2:
        if not field.is_axis_regular(2):
            raise ValueError("GeoTIFF export requires a uniformly spaced y axis.")
        dy_q = field.axis_mean_delta(2)
        if dy_q is None:
            raise ValueError("GeoTIFF export failed to compute y pixel spacing.")
        dy = float(u.Quantity(dy_q, copy=False).to_value(coord_unit))
    elif tf_fallback is not None and len(tf_fallback) >= 6:
        dy = float(tf_fallback[5])
    else:
        raise ValueError(
            "Cannot determine pixel height for singleton y axis without GeoMetadata.transform."
        )

    arr_xy = np.asarray(field.value[0, :, :, 0], dtype=np.float64)
    if dx < 0:
        x_vals = x_vals[::-1]
        arr_xy = arr_xy[::-1, :]
        dx = -dx
    if dy > 0:
        # GeoTIFF convention stores rows top-to-bottom (north-up, dy < 0).
        y_vals = y_vals[::-1]
        arr_xy = arr_xy[:, ::-1]
        dy = -dy

    arr_yx = np.ascontiguousarray(arr_xy.T)
    if nodata is not None:
        arr_yx = np.where(~np.isfinite(arr_yx), float(nodata), arr_yx)

    x_ul = float(x_vals[0]) - 0.5 * dx
    y_ul = float(y_vals[0]) - 0.5 * dy
    transform = Affine.from_gdal(x_ul, dx, 0.0, y_ul, 0.0, dy)

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float64",
        "crs": dst_crs,
        "transform": transform,
        "nodata": nodata,
    }
    if compress:
        profile["compress"] = compress

    with rasterio.open(target, "w", **profile) as dst:
        dst.write(arr_yx, 1)
        if field.unit is not None and field.unit != u.dimensionless_unscaled:
            dst.units = (str(field.unit),)


if _HAS_RASTERIO:
    register_field_format(
        "geotiff",
        reader_scalar=read_scalarfield_geotiff,
        writer_scalar=write_scalarfield_geotiff,
        lister=list_fields_geotiff,
        extension=(".tif", ".tiff"),
        magic_identifier=_identify_geotiff_magic,
        aliases=("tif", "tiff"),
    )
