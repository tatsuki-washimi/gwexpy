"""Synthetic GSI FGD DEM XML/ZIP generator helpers (stdlib only)."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def render_gsi_dem_xml(
    grid_yx: np.ndarray,
    *,
    lon_min: float = 139.0,
    lat_min: float = 35.0,
    lon_max: float = 139.2,
    lat_max: float = 35.2,
    srs_name: str = "fguuid:jgd2011.bl",
    axis_order: str = "+x-y",
    start_point: tuple[int, int] | None = None,
    nodata_value: float = -9999.0,
    skip_leading_cells: int = 0,
) -> str:
    """Render a minimal valid GSI FGD DEM GML XML string from a ``(ny, nx)`` array.

    Rows of ``grid_yx`` are north-to-south (row 0 = northernmost) and columns
    are west-to-east, matching GSI ``+x-y`` sequence order.
    """
    arr = np.asarray(grid_yx, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("grid_yx must be 2D (ny, nx).")
    ny, nx = arr.shape
    flat = arr.reshape(-1)
    if start_point is None:
        sx, sy = 0, 0
    else:
        sx, sy = start_point
    if skip_leading_cells > 0:
        flat = flat[skip_leading_cells:]

    tuple_lines = []
    for val in flat:
        if np.isnan(val):
            tuple_lines.append(f"データなし,{nodata_value:g}.")
        else:
            tuple_lines.append(f"地表面,{val:.6f}")
    tuple_body = "\n".join(tuple_lines)
    start_xml = (
        f"<gml:startPoint>{sx} {sy}</gml:startPoint>" if start_point is not None else ""
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Dataset xmlns="http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema"
         xmlns:gml="http://www.opengis.net/gml/3.2"
         xmlns:xlink="http://www.w3.org/1999/xlink"
         gml:id="Dataset1">
  <DEM gml:id="DEM1">
    <fid>fgd-dem-synthetic</fid>
    <type>5mメッシュ（標高）</type>
    <mesh>53390000</mesh>
    <coverage gml:id="COV1">
      <gml:boundedBy>
        <gml:Envelope srsName="{srs_name}">
          <gml:lowerCorner>{lat_min:.10f} {lon_min:.10f}</gml:lowerCorner>
          <gml:upperCorner>{lat_max:.10f} {lon_max:.10f}</gml:upperCorner>
        </gml:Envelope>
      </gml:boundedBy>
      <gml:gridDomain>
        <gml:Grid dimension="2" gml:id="G1">
          <gml:limits>
            <gml:GridEnvelope>
              <gml:low>0 0</gml:low>
              <gml:high>{nx - 1} {ny - 1}</gml:high>
            </gml:GridEnvelope>
          </gml:limits>
          <gml:axisName>x</gml:axisName>
          <gml:axisName>y</gml:axisName>
        </gml:Grid>
      </gml:gridDomain>
      <gml:rangeSet>
        <gml:DataBlock>
          <gml:rangeParameters/>
          <gml:tupleList>
{tuple_body}
          </gml:tupleList>
        </gml:DataBlock>
      </gml:rangeSet>
      <gml:coverageFunction>
        <gml:GridFunction>
          <gml:sequenceRule axisOrder="{axis_order}">Linear</gml:sequenceRule>
          {start_xml}
        </gml:GridFunction>
      </gml:coverageFunction>
    </coverage>
  </DEM>
</Dataset>
"""


def write_gsi_dem_xml(path: Path | str, grid_yx: np.ndarray, **kwargs: Any) -> Path:
    out = Path(path)
    out.write_text(render_gsi_dem_xml(grid_yx, **kwargs), encoding="utf-8")
    return out


def write_gsi_dem_zip(
    path: Path | str,
    tiles: Sequence[tuple[str, np.ndarray, dict[str, Any]]],
) -> Path:
    out = Path(path)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member_name, grid_yx, kw in tiles:
            zf.writestr(member_name, render_gsi_dem_xml(grid_yx, **kw))
    return out
