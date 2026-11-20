# Field API: Layered Geographic and Terrain Visualization

This tutorial shows how to compose **basemap tiles**, **DEM terrain relief / contours**, **scalar and vector physical fields**, **detector markers**, and **static-background animations** on a single figure using `FieldPlot` (Matplotlib) and `GeoMap` (PyGMT).

## Layer Model and `zorder` Conventions

`FieldPlot` stacks layers in a fixed default `zorder` hierarchy (all overridable via keyword arguments):

| Layer | Method | Default `zorder` | Default `colorbar` |
| :--- | :--- | :--- | :--- |
| Basemap XYZ tiles | `FieldPlot.add_basemap()` | `0.0` | `—` |
| Terrain DEM | `FieldPlot.add_terrain()` | `1.0` | `False` |
| Scalar field | `FieldPlot.add_scalar()` | `2.0` | `True` |
| Vector field | `FieldPlot.add_vector()` | `3.0` | `—` |
| Markers / detectors | `FieldPlot.add_markers()`, `FieldPlot.add_detector()` | `4.0` | `—` |

Layers on the same `Axes` share a `GeoPlotContext`:

- Mixing `angle`-domain axes (`lon`/`lat`) with `real`/`local`-Cartesian axes (`x`/`y` in meters) raises `ValueError` with guidance to align coordinates via `to_local_cartesian()`.
- Active CRS mismatches and differing `local_origin` coordinates emit `UserWarning`.
- Calling `add_basemap()` or `add_detector()` on a plain (non-georeferenced) field raises `ValueError`.

## 1. Synthetic KAGRA Terrain and Wavefield Setup

When reading a real Geospatial Information Authority of Japan (GSI) DEM (e.g. via `ScalarField.read("dem.xml", format="gsi-dem")`), you can convert geographic coordinates to a local tangent plane in meters with `dem.to_local_cartesian(origin=(137.31, 36.41))`. Below we construct an equivalent synthetic DEM and wavefield centered on KAGRA:

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.fields import ScalarField, VectorField
from gwexpy.plot.field import FieldPlot

# Local Cartesian grid centered on KAGRA (137.31 E, 36.41 N)
nx, ny = 60, 60
x = np.linspace(-1500.0, 1500.0, nx) * u.m
y = np.linspace(-1500.0, 1500.0, ny) * u.m
z0 = np.array([0.0]) * u.m
t_static = np.array([0.0]) * u.s

X, Y = np.meshgrid(x.value, y.value, indexing="ij")
elev = 400.0 + 700.0 * np.exp(-((X - 200.0) ** 2 + (Y + 150.0) ** 2) / (2 * 600.0**2))

dem = ScalarField(
    elev[np.newaxis, :, :, np.newaxis],
    unit=u.m,
    name="elevation",
    axis0=t_static,
    axis1=x,
    axis2=y,
    axis3=z0,
    space_domain="real",
)

# Attach local-Cartesian origin provenance (populated automatically by to_local_cartesian())
from types import SimpleNamespace
dem._gwex_geo = SimpleNamespace(active=False, crs="EPSG:6668", local_origin=(137.31, 36.41))

# Synthetic seismic wavefield propagating across the same grid
t = np.linspace(0.0, 2.0, 10) * u.s
wave_data = np.sin(
    2.0 * np.pi * (0.5 * t.value[:, None, None, None] - X[None, :, :, None] / 800.0)
) * np.exp(-(X[None, :, :, None] ** 2 + Y[None, :, :, None] ** 2) / (2 * 700.0**2))

wave = ScalarField(
    wave_data,
    unit=u.um / u.s,
    name="ground_velocity",
    axis0=t,
    axis1=x,
    axis2=y,
    axis3=z0,
    space_domain="real",
)
wave._gwex_geo = dem._gwex_geo
```

## 2. Building a Layered Figure Step by Step

Use `FieldPlot` to compose terrain shading, contours, a semi-transparent scalar wavefield, and the KAGRA detector marker:

```python
fp = FieldPlot()
fp.add_terrain(dem, style="hillshade")
fp.add_terrain(dem, style="contour", levels=6, colors="black", alpha=0.35, linewidths=0.6)
fp.add_scalar(wave, x="x", y="y", slice_kwargs={"t": 0, "z": 0}, cmap="RdBu_r", alpha=0.6)
fp.add_detector("K1")

# Optional online XYZ basemap tiles (uses GSI standard/pale tiles by default):
# fp.add_basemap(source="gsi-pale", cache_dir=".tile_cache")

plt.show()
```

Pass `colorbar=False` to `add_scalar()` when stacking multiple scalar layers to keep the layout uncluttered.

## 3. One-Call Overlay Sugar in `FieldBase.plot()`

Once your workflow is established, `ScalarField.plot()` accepts `terrain=`, `terrain_style=`, `basemap=`, `vector=`, and `detectors=` keywords and dispatches them in the standard `zorder` sequence:

```python
flow = VectorField({"x": wave, "y": wave * 0.5})
fp = wave.plot(
    x="x",
    y="y",
    t=0,
    z=0,
    terrain=dem,
    terrain_style="shaded_relief",
    # basemap="gsi-pale",
    vector=flow,
    detectors=["K1"],
    alpha=0.55,
    cmap="magma",
)
```

## 4. Static-Background Animation with `animate_scalar()`

`FieldPlot.animate_scalar()` creates a single `QuadMesh` and colorbar once and updates only the mesh array (`set_array()`) and axes title per frame, leaving static basemap, terrain, and marker layers untouched:

```python
fp_anim = FieldPlot()
fp_anim.add_terrain(dem, style="hillshade")
fp_anim.add_detector("K1")
ani = fp_anim.animate_scalar(wave, x="x", y="y", axis="t", interval=150, cmap="RdBu_r", alpha=0.6)
```

> **Note:** `shading="gouraud"` is rejected in `animate_scalar()` because `QuadMesh.set_array()` requires flat or auto shading.

## 5. Wide-Area Geodetic Maps with `GeoMap`

For wide-area or global projections where tile extent placement would introduce noticeable WebMercator distortion (a `UserWarning` is emitted when latitude span exceeds 2 degrees), use `GeoMap` (requires the `plotting` extra / `pygmt` and `xarray`):

```python
from gwexpy.plot import GeoMap

gmap = GeoMap(projection="Mercator", region=[135.0, 140.0, 34.0, 38.0])
gmap.add_relief(resolution="15s", shading=True)  # downloads @earth_relief_15s
# gmap.add_field(angle_dem, cmap="viridis", alpha=0.7)
gmap.plot_detector("K1")
gmap.add_colorbar(label="Elevation [m]")
```

## Tile Providers and Attribution

Built-in providers in `gwexpy.plot._tiles.PROVIDERS`:

- `"gsi-std"`, `"gsi-pale"`, `"gsi-photo"`, `"gsi-relief"` — Geospatial Information Authority of Japan (GSI Tiles). Attribution is required by the GSI terms of use (`attribution=True` is enabled by default). See https://maps.gsi.go.jp/development/ichiran.html for details.
- `"osm"` — OpenStreetMap standard tiles. Bulk automated scraping is discouraged by the OSM tile usage policy (https://operations.osmfoundation.org/policies/tiles/); prefer `cache_dir=` when iterating locally.
