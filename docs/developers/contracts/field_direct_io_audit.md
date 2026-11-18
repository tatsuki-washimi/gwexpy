# Field direct-I/O contract audit (#553)

Date: 2026-11-19. Scope: the terrain formats introduced by #547, #548 and
#551, on the v0.3.0 field integration branch.

## Decision

`gsi-dem` and `geotiff` remain outside
`public_io_contract.json` for this contract revision. Their class methods are
available and documented in the terrain I/O guide, but inclusion in the
**published conformance schema** would currently claim a CI gate that does not
exercise these Field routes. This is an explicit contract-scope decision, not a
claim that the readers are absent or that terrain I/O is unsupported.

The current v3 conformance loader maps formats to time-series-oriented fixture
generators and optional CI jobs. It has no Field validator for four axes, units,
`GeoMetadata`, requested-name order, or 2D GeoTIFF round trips, and no
`io-geospatial` job with `rasterio`. Adding format rows as `planned` would
advertise a published contract without the test gate required by the
[execution rule](public_io_contract.md#execution-rule).

## Audited routes

| Format | Live direct API | Registry and identification | Dependency and unavailable behavior | Synthetic evidence |
|---|---|---|---|---|
| `gsi-dem` (`fgd-dem` alias) | `ScalarField.read(source, name, ...)`, `FieldDict.read(source, names, ...)`; no writer | Both reader classes registered; `FieldDict` handler selects names through the scalar reader; content-based GML/ZIP identifier | Base install; no optional package | `tests/io_conformance/generators/gsi_dem.py`; `tests/io/test_gsi_dem_reader.py`, `test_gsi_dem_public_io.py` |
| `geotiff` (`tif`, `tiff` aliases) | `ScalarField.read(source, name, ...)`, `FieldDict.read(source, names, ...)`, `ScalarField.write(target, ...)` | Both reader classes and the scalar writer register when `rasterio` is importable; `FieldDict` handler selects names through the scalar reader; TIFF magic/extension identification | `rasterio` from `geospatial`; explicit `.read()`/`.write()` raises `ImportError` naming the extra when absent; registry entries may be absent | `tests/io/test_geotiff_reader.py`, `test_geotiff_public_io.py` use synthetic rasters when `rasterio` is installed; missing-dependency behavior is tested without it |

`VectorField` and `TensorField` have shared `.read()` dispatch, but neither
terrain format registers a reader for them. `FieldDict` preserves requested
name order. `gsi-dem` keeps missing elevations as `NaN` and pixel-center
longitude/latitude axes in degrees; `geotiff` supports only a two-dimensional
exchange representation on write. The optional `pyproj` dependency is needed
for `to_projected()`, not for GeoTIFF file I/O itself. The `geospatial` extra
contains both packages.

## Entry conditions for a future schema expansion

1. Add Field classes to the schema class map and define axis-0 time semantics
   for static terrain fields without claiming the time-series-only rules.
2. Register a deterministic GSI generator and a GeoTIFF generator in the
   conformance harness; validate values, axis order and units, `GeoMetadata`,
   selection order, and read/write behavior.
3. Add a required `io-geospatial` CI job with `rasterio`, plus a base-install
   missing-dependency gate. Update format coverage status and optional job
   routing together with the new JSON entries.
4. Reconcile the English/Japanese capability guides with those entries in the
   same change.

Until those gates exist, the targeted terrain tests above and the existing
`tests/io` gate protect the implementation. The JSON contract remains focused
on its currently qualified formats.
