"""Smoke and unit tests for layered geographic and terrain plotting (#560-#567)."""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from matplotlib.quiver import Quiver
from PIL import Image

from gwexpy.fields import ScalarField, VectorField
from gwexpy.plot._detectors import DETECTORS
from gwexpy.plot._geo_utils import (
    GeoPlotContext,
    compute_hillshade,
    local_xy_to_lonlat,
    lonlat_to_local_xy,
    resolve_geo_context,
)
from gwexpy.plot._tiles import (
    PROVIDERS,
    auto_zoom,
    deg2tile,
    fetch_tile,
    tile_extent_deg,
)
from gwexpy.plot.field import FieldPlot
from gwexpy.plot.geomap import DETECTORS as GEOMAP_DETECTORS


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _AngleGeoField:
    """Duck-typed 4D angle-domain field compatible with FieldPlot and origin/main.

    Mirrors the #545/#546 contract (``space_domains`` with ``"angle"`` and
    ``geo`` metadata) so layered plotting tests run both pre- and post-merge.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        lon: u.Quantity,
        lat: u.Quantity,
        t: u.Quantity | None = None,
        z: u.Quantity | None = None,
        unit: u.UnitBase = u.m,
        name: str = "elevation",
        geo: Any = None,
    ):
        self.value = np.asarray(data, dtype=float)
        self.shape = self.value.shape
        self.unit = unit
        self.name = name
        self._axis0_name = "t"
        self._axis1_name = "lon"
        self._axis2_name = "lat"
        self._axis3_name = "z"
        self.axis_names = ("t", "lon", "lat", "z")
        self._axis0_index = (
            t if t is not None else np.arange(self.shape[0], dtype=float) * u.s
        )
        self._axis1_index = lon
        self._axis2_index = lat
        self._axis3_index = (
            z if z is not None else np.zeros(self.shape[3], dtype=float) * u.m
        )
        self.space_domains = {"lon": "angle", "lat": "angle", "z": "real"}
        self.geo = geo

    def get_slice(self, x_axis=None, y_axis=None, **fixed_coords):
        all_axes = list(self.axis_names)
        if x_axis is None or y_axis is None:
            candidates = [a for a in all_axes if a not in fixed_coords]
            if x_axis is None:
                x_axis = candidates[0]
            if y_axis is None:
                y_axis = candidates[1]
        slices: list[Any] = [slice(None)] * 4
        for i, ax_name in enumerate(all_axes):
            if ax_name in (x_axis, y_axis):
                continue
            if ax_name in fixed_coords:
                val = fixed_coords[ax_name]
                if isinstance(val, int) and not isinstance(val, u.Quantity):
                    slices[i] = val
                else:
                    idx_arr = getattr(self, f"_axis{i}_index")
                    target = (
                        val.to_value(idx_arr.unit)
                        if isinstance(val, u.Quantity)
                        else val
                    )
                    slices[i] = int(np.argmin(np.abs(idx_arr.value - target)))
            else:
                slices[i] = 0
        x_pos = all_axes.index(x_axis)
        y_pos = all_axes.index(y_axis)
        return (
            self.value[tuple(slices)] * self.unit,
            getattr(self, f"_axis{x_pos}_index"),
            getattr(self, f"_axis{y_pos}_index"),
            x_axis,
            y_axis,
        )


def _make_angle_dem(
    *, descending_lat: bool = True, with_nan: bool = False, crs: str = "EPSG:6668"
):
    nx, ny = 8, 6
    lon = np.linspace(137.28, 137.34, nx) * u.deg
    lat = (
        np.linspace(36.44, 36.38, ny) * u.deg
        if descending_lat
        else np.linspace(36.38, 36.44, ny) * u.deg
    )
    lon_grid, lat_grid = np.meshgrid(lon.value, lat.value, indexing="ij")
    elev = 400.0 + 900.0 * np.exp(
        -((lon_grid - 137.31) ** 2 + (lat_grid - 36.41) ** 2) / (2 * 0.015**2)
    )
    if with_nan:
        elev[2, 3] = np.nan
    geo = SimpleNamespace(active=True, crs=crs, local_origin=None)
    return _AngleGeoField(
        elev[np.newaxis, :, :, np.newaxis],
        lon=lon,
        lat=lat,
        geo=geo,
    )


def _make_local_field(*, origin=(137.31, 36.41), nt=1, name="local_field"):
    nx, ny = 6, 5
    x = np.linspace(-500.0, 500.0, nx) * u.m
    y = np.linspace(-400.0, 400.0, ny) * u.m
    t = np.arange(nt, dtype=float) * u.s
    data = np.arange(nt * nx * ny, dtype=float).reshape(nt, nx, ny, 1)
    sf = ScalarField(
        data,
        unit=u.m,
        name=name,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=np.array([0.0]) * u.m,
        space_domain="real",
    )
    sf._gwex_geo = SimpleNamespace(active=False, crs="EPSG:6668", local_origin=origin)
    return sf


def _fake_tile_factory():
    def _fake_fetch(url, *, timeout=10.0, cache_dir=None, provider=None):
        arr = np.full((256, 256, 4), 200, dtype=np.uint8)
        arr[:, :, 3] = 255
        return Image.fromarray(arr, mode="RGBA")

    return _fake_fetch


# ---------------------------------------------------------------------------
# C-0 & C-1: VectorField stride + add_scalar layering groundwork
# ---------------------------------------------------------------------------


def test_vector_field_plot_stride_actually_decimates_quiver():
    data = np.ones((1, 4, 4, 1), dtype=float)
    sf = ScalarField(
        data,
        unit=u.m / u.s,
        axis0=np.array([0.0]) * u.s,
        axis1=np.arange(4, dtype=float) * u.m,
        axis2=np.arange(4, dtype=float) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    vf = VectorField({"x": sf, "y": sf})
    fp = vf.plot(x="x", y="y", t=0, z=0, stride=2)
    quivers = [c for c in fp.gca().collections if isinstance(c, Quiver)]
    assert len(quivers) == 1
    assert quivers[0].get_offsets().shape[0] == 4


def test_add_scalar_colorbar_false_and_zorder():
    sf = ScalarField(
        np.arange(24, dtype=float).reshape(1, 4, 6, 1),
        unit=u.V,
        axis0=np.array([0.0]) * u.s,
        axis1=np.arange(4, dtype=float) * u.m,
        axis2=np.arange(6, dtype=float) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    fp = FieldPlot()
    mesh = fp.add_scalar(
        sf, x="x", y="y", slice_kwargs={"t": 0, "z": 0}, colorbar=False
    )
    assert isinstance(mesh, QuadMesh)
    assert fp.last_field_colorbar is None
    assert len(fp.figure.axes) == 1
    assert len(fp.gca().child_axes) == 0
    assert mesh.get_zorder() == pytest.approx(2.0)

    mesh_custom = fp.add_scalar(
        sf, x="x", y="y", slice_kwargs={"t": 0, "z": 0}, colorbar=False, zorder=5.5
    )
    assert mesh_custom.get_zorder() == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# C-2: GeoPlotContext, consistency checks, hillshade & tangent-plane math
# ---------------------------------------------------------------------------


def test_compute_hillshade_synthetic_slope_and_nan():
    # Row 0 is North (top): values increasing toward bottom-right face NW (bright under az=315)
    rows, cols = np.meshgrid(
        np.arange(5, dtype=float), np.arange(5, dtype=float), indexing="ij"
    )
    z_nw = rows + cols
    z_se = -(rows + cols)
    hs_nw = compute_hillshade(z_nw, dx=1.0, dy=1.0, azdeg=315.0, altdeg=45.0)
    hs_se = compute_hillshade(z_se, dx=1.0, dy=1.0, azdeg=315.0, altdeg=45.0)
    assert hs_nw[2, 2] > 0.7
    assert hs_se[2, 2] < 0.3

    z_nan = z_nw.copy()
    z_nan[2, 2] = np.nan
    hs_nan = compute_hillshade(z_nan, dx=1.0, dy=1.0)
    assert np.isnan(hs_nan[2, 2])
    assert np.isfinite(hs_nan[1, 2])

    hs_all_nan = compute_hillshade(np.full((3, 3), np.nan))
    assert np.all(np.isnan(hs_all_nan))


def test_local_tangent_plane_roundtrip():
    lon0, lat0 = 137.31, 36.41
    x0, y0 = lonlat_to_local_xy(lon0, lat0, lon0, lat0)
    assert x0 == pytest.approx(0.0, abs=1e-12)
    assert y0 == pytest.approx(0.0, abs=1e-12)

    lon_pts = np.array([137.30, 137.32])
    lat_pts = np.array([36.40, 36.42])
    x, y = lonlat_to_local_xy(lon_pts, lat_pts, lon0, lat0)
    lon_rt, lat_rt = local_xy_to_lonlat(x, y, lon0, lat0)
    np.testing.assert_allclose(lon_rt, lon_pts, atol=1e-12)
    np.testing.assert_allclose(lat_rt, lat_pts, atol=1e-12)
    assert x[1] > 0 and y[1] > 0


def test_geo_context_consistency_rules():
    dem_angle = _make_angle_dem()
    local_sf = _make_local_field()
    plain_sf = ScalarField(
        np.ones((1, 4, 4, 1)),
        axis0=np.array([0.0]) * u.s,
        axis1=np.arange(4, dtype=float) * u.m,
        axis2=np.arange(4, dtype=float) * u.m,
        axis3=np.array([0.0]) * u.m,
    )

    # 1. angle + real/local -> ValueError
    fig, ax = plt.subplots()
    ctx = resolve_geo_context(ax, dem_angle, "lon", "lat")
    assert ctx.kind == "angle"
    with pytest.raises(ValueError, match="to_local_cartesian"):
        resolve_geo_context(ax, local_sf, "x", "y")
    with pytest.raises(ValueError, match="to_local_cartesian"):
        resolve_geo_context(ax, plain_sf, "x", "y")

    # 2. real/local + angle -> ValueError
    fig2, ax2 = plt.subplots()
    resolve_geo_context(ax2, local_sf, "x", "y")
    with pytest.raises(ValueError, match="to_local_cartesian"):
        resolve_geo_context(ax2, dem_angle, "lon", "lat")

    # 3. active CRS mismatch -> UserWarning
    fig3, ax3 = plt.subplots()
    resolve_geo_context(ax3, _make_angle_dem(crs="EPSG:6668"), "lon", "lat")
    with pytest.warns(UserWarning, match="CRS mismatch"):
        resolve_geo_context(ax3, _make_angle_dem(crs="EPSG:4326"), "lon", "lat")

    # 4. local_origin mismatch -> UserWarning with offset in meters
    fig4, ax4 = plt.subplots()
    resolve_geo_context(ax4, _make_local_field(origin=(137.31, 36.41)), "x", "y")
    with pytest.warns(UserWarning, match="local_origin mismatch.*m offset"):
        resolve_geo_context(ax4, _make_local_field(origin=(137.32, 36.41)), "x", "y")

    # 5. deg/rad mixing -> accepted and converted on plot
    dem_rad = _AngleGeoField(
        dem_angle.value,
        lon=dem_angle._axis1_index.to(u.rad),
        lat=dem_angle._axis2_index.to(u.rad),
        geo=dem_angle.geo,
    )
    fp = FieldPlot()
    fp.add_terrain(dem_angle)
    mesh = fp.add_scalar(
        dem_rad,
        x="lon",
        y="lat",
        slice_kwargs={"t": 0, "z": 0},
        alpha=0.5,
        colorbar=False,
    )
    assert fp.gca().get_xlabel() == "lon [deg]"
    coords = mesh.get_coordinates()
    assert np.all(coords[..., 0] > 100.0)  # converted from rad (~2.4) to deg (~137.3)


# ---------------------------------------------------------------------------
# C-3: FieldPlot.add_terrain
# ---------------------------------------------------------------------------


def test_add_terrain_styles_nan_and_descending_lat():
    dem_desc = _make_angle_dem(descending_lat=True, with_nan=True)
    fp = FieldPlot()
    im = fp.add_terrain(dem_desc, style="hillshade")
    assert isinstance(im, AxesImage)
    assert im.get_zorder() == pytest.approx(1.0)
    assert fp.last_field_colorbar is None
    assert len(fp.figure.axes) == 1
    assert fp.gca().get_xlabel() == "lon [deg]"
    assert fp.gca().get_ylabel() == "lat [deg]"

    # Descending lat must still display North (larger lat) at top (non-inverted ylim)
    ylim = fp.gca().get_ylim()
    assert ylim[0] < ylim[1]

    # NaN pixel must render transparent (alpha == 0)
    arr = im.get_array()
    assert np.ma.getmaskarray(arr)[3, 2]
    rgba = im.to_rgba(arr)
    assert rgba[3, 2, 3] == pytest.approx(0.0)

    # Shaded relief + colorbar=True
    fp2 = FieldPlot()
    im_sr = fp2.add_terrain(dem_desc, style="shaded_relief", colorbar=True)
    assert isinstance(im_sr, AxesImage)
    assert im_sr.get_array().shape[-1] == 4
    assert im_sr.get_array()[3, 2, 3] == pytest.approx(0.0)
    assert fp2.last_field_colorbar is not None

    # Orientation check: ascending and descending lat yield matching RGB at North/South
    dem_asc = _make_angle_dem(descending_lat=False, with_nan=False)
    dem_desc_clean = _make_angle_dem(descending_lat=True, with_nan=False)
    fp_desc = FieldPlot()
    im_d = fp_desc.add_terrain(dem_desc_clean, style="shaded_relief")
    fp_asc = FieldPlot()
    im_a = fp_asc.add_terrain(dem_asc, style="shaded_relief")
    assert fp_asc.gca().get_ylim()[0] < fp_asc.gca().get_ylim()[1]
    np.testing.assert_allclose(im_d.get_array(), im_a.get_array()[::-1], atol=1e-6)

    # Contours
    fp3 = FieldPlot()
    cs = fp3.add_terrain(dem_desc, style="contour", levels=5)
    assert isinstance(cs, QuadContourSet)
    fp4 = FieldPlot()
    csf = fp4.add_terrain(dem_desc, style="contourf", levels=5)
    assert isinstance(csf, QuadContourSet)

    with pytest.raises(ValueError, match="Unknown terrain style"):
        fp.add_terrain(dem_desc, style="invalid")


def test_add_terrain_local_cartesian_field():
    local_dem = _make_local_field(name="dem_local")
    fp = FieldPlot()
    im = fp.add_terrain(local_dem, style="hillshade")
    assert isinstance(im, AxesImage)
    assert fp.gca().get_xlabel() == "x [m]"
    assert fp.gca().get_ylabel() == "y [m]"


# ---------------------------------------------------------------------------
# C-4: _detectors, add_markers, add_detector
# ---------------------------------------------------------------------------


def test_detectors_reexport_and_markers():
    assert GEOMAP_DETECTORS is DETECTORS
    assert "K1" in DETECTORS

    # Angle axes
    dem = _make_angle_dem()
    fp = FieldPlot()
    fp.add_terrain(dem)
    pc = fp.add_detector("K1")
    assert isinstance(pc, PathCollection)
    assert pc.get_zorder() == pytest.approx(4.0)
    offsets = pc.get_offsets()
    np.testing.assert_allclose(
        offsets[0], [DETECTORS["K1"]["lon"], DETECTORS["K1"]["lat"]]
    )

    # Local-Cartesian axes
    local_sf = _make_local_field(origin=(137.30, 36.40))
    fp_loc = FieldPlot()
    fp_loc.add_scalar(local_sf, x="x", y="y", slice_kwargs={"t": 0, "z": 0})
    pc_loc = fp_loc.add_detector("K1")
    exp_x, exp_y = lonlat_to_local_xy(
        DETECTORS["K1"]["lon"], DETECTORS["K1"]["lat"], 137.30, 36.40
    )
    np.testing.assert_allclose(pc_loc.get_offsets()[0], [exp_x, exp_y])

    # Plain field -> ValueError for add_detector
    plain_sf = ScalarField(
        np.ones((1, 4, 4, 1)),
        axis0=np.array([0.0]) * u.s,
        axis1=np.arange(4, dtype=float) * u.m,
        axis2=np.arange(4, dtype=float) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    fp_plain = plain_sf.plot(x="x", y="y", t=0, z=0)
    with pytest.raises(ValueError, match="plain"):
        fp_plain.add_detector("K1")

    # Generic add_markers works on plain field
    m_artist = fp_plain.add_markers([1.0, 2.0], [1.5, 2.5], labels=["A", "B"])
    assert isinstance(m_artist, PathCollection)

    with pytest.raises(ValueError, match="Unknown detector"):
        fp.add_detector("UNKNOWN")


# ---------------------------------------------------------------------------
# C-5: _tiles and FieldPlot.add_basemap
# ---------------------------------------------------------------------------


def test_tile_math_pure_functions():
    assert {"gsi-std", "gsi-pale", "gsi-photo", "gsi-relief", "osm"}.issubset(PROVIDERS)
    tx, ty = deg2tile(137.31, 36.41, 12)
    lon0, lon1, lat0, lat1 = tile_extent_deg(tx, ty, 12)
    assert lon0 <= 137.31 <= lon1
    assert lat0 <= 36.41 <= lat1
    z = auto_zoom(137.28, 137.34, 36.38, 36.44, max_tiles=16)
    assert 0 <= z <= 18


def test_fetch_tile_cache_and_error_propagation(tmp_path, monkeypatch):
    import urllib.request
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(buf, format="PNG")
    raw_png = buf.getvalue()
    calls = {"count": 0}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return raw_png

    def _fake_urlopen(req, timeout=10.0):
        calls["count"] += 1
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    im1 = fetch_tile(
        "https://example.com/12/1/2.png", cache_dir=tmp_path, provider="gsi-std"
    )
    im2 = fetch_tile(
        "https://example.com/12/1/2.png", cache_dir=tmp_path, provider="gsi-std"
    )
    assert im1.size == (8, 8)
    assert im2.size == (8, 8)
    assert calls["count"] == 1  # second call served from disk cache

    def _failing_urlopen(req, timeout=10.0):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _failing_urlopen)
    with pytest.raises(urllib.error.URLError, match="gsi-pale"):
        fetch_tile("https://example.com/12/9/9.png", provider="gsi-pale")


def test_add_basemap_mocked(monkeypatch):
    import gwexpy.plot.field as field_mod

    monkeypatch.setattr(field_mod, "fetch_tile", _fake_tile_factory())

    dem = _make_angle_dem()
    fp = FieldPlot()
    fp.add_terrain(dem)
    imgs = fp.add_basemap(source="gsi-pale", zoom=12)
    assert len(imgs) >= 1
    assert all(isinstance(im, AxesImage) for im in imgs)
    assert all(im.get_zorder() == pytest.approx(0.0) for im in imgs)
    assert any("GSI" in t.get_text() for t in fp.gca().texts)

    # Attribution disabled
    fp_no_attr = FieldPlot()
    fp_no_attr.add_basemap(
        source="osm", extent=(137.28, 137.34, 36.38, 36.44), zoom=12, attribution=False
    )
    assert len(fp_no_attr.gca().texts) == 0

    # Wide latitude span (> 2 deg) emits UserWarning
    fp_wide = FieldPlot()
    with pytest.warns(UserWarning, match="GeoMap"):
        fp_wide.add_basemap(source="gsi-std", extent=(135.0, 138.0, 34.0, 37.5), zoom=7)

    # Local-Cartesian basemap placement
    local_sf = _make_local_field()
    fp_local = FieldPlot()
    fp_local.add_scalar(local_sf, x="x", y="y", slice_kwargs={"t": 0, "z": 0})
    imgs_local = fp_local.add_basemap(source="gsi-std", zoom=14)
    assert len(imgs_local) >= 1

    # Plain field -> ValueError
    plain_sf = ScalarField(
        np.ones((1, 4, 4, 1)),
        axis0=np.array([0.0]) * u.s,
        axis1=np.arange(4, dtype=float) * u.m,
        axis2=np.arange(4, dtype=float) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    fp_plain = plain_sf.plot(x="x", y="y", t=0, z=0)
    with pytest.raises(ValueError, match="plain"):
        fp_plain.add_basemap(source="gsi-std")

    # Empty axes without extent -> ValueError
    with pytest.raises(ValueError, match="extent"):
        FieldPlot().add_basemap(source="gsi-std")


@pytest.mark.network
def test_fetch_tile_live_gsi():
    tx, ty = deg2tile(137.31, 36.41, 12)
    url = PROVIDERS["gsi-pale"]["url"].format(z=12, x=tx, y=ty)
    im = fetch_tile(url, provider="gsi-pale")
    assert im.size == (256, 256)


# ---------------------------------------------------------------------------
# C-6: FieldPlot.animate_scalar preserves static layers
# ---------------------------------------------------------------------------


def test_animate_scalar_preserves_static_layers_and_rejects_gouraud(monkeypatch):
    import gwexpy.plot.field as field_mod

    monkeypatch.setattr(field_mod, "fetch_tile", _fake_tile_factory())

    dem = _make_angle_dem()
    nt, nx, ny = 3, dem.shape[1], dem.shape[2]
    wave_data = np.arange(nt * nx * ny, dtype=float).reshape(nt, nx, ny, 1)
    wave = _AngleGeoField(
        wave_data,
        lon=dem._axis1_index,
        lat=dem._axis2_index,
        t=np.arange(nt, dtype=float) * u.s,
        unit=u.m / u.s,
        name="velocity",
        geo=dem.geo,
    )

    fp = FieldPlot()
    fp.add_terrain(dem, style="hillshade")
    fp.add_basemap(source="gsi-pale", zoom=12)
    fp.add_detector("K1")

    ax = fp.gca()
    n_images_before = len(ax.images)
    n_texts_before = len(ax.texts)
    n_collections_before = len(ax.collections)

    ani = fp.animate_scalar(wave, x="lon", y="lat", axis="t", alpha=0.6)
    n_axes_init = len(fp.figure.axes) + sum(len(a.child_axes) for a in fp.figure.axes)
    assert len(ax.images) == n_images_before
    assert len(ax.texts) == n_texts_before
    assert len(ax.collections) == n_collections_before + 1

    for frame in wave._axis0_index:
        ani._draw_frame(frame)
        n_axes_now = len(fp.figure.axes) + sum(
            len(a.child_axes) for a in fp.figure.axes
        )
        assert n_axes_now == n_axes_init
        assert len(ax.images) == n_images_before
        assert len(ax.texts) == n_texts_before
        assert len(ax.collections) == n_collections_before + 1

    with pytest.raises(ValueError, match="gouraud"):
        FieldPlot().animate_scalar(wave, x="lon", y="lat", shading="gouraud")


# ---------------------------------------------------------------------------
# C-8: FieldBase.plot one-call overlays
# ---------------------------------------------------------------------------


def test_field_base_plot_one_call_overlays(monkeypatch):
    import gwexpy.plot.field as field_mod

    monkeypatch.setattr(field_mod, "fetch_tile", _fake_tile_factory())

    origin = (137.31, 36.41)
    dem = _make_local_field(origin=origin, nt=1, name="dem")
    wave = _make_local_field(origin=origin, nt=2, name="wave")
    vec = VectorField({"x": wave, "y": wave})

    fp = wave.plot(
        x="x",
        y="y",
        t=0,
        z=0,
        terrain=dem,
        terrain_style="hillshade",
        basemap="gsi-pale",
        vector=vec,
        detectors=["K1"],
        alpha=0.6,
        cmap="magma",
    )
    assert isinstance(fp, FieldPlot)
    ax = fp.gca()
    images = [a for a in ax.images]
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    quivers = [c for c in ax.collections if isinstance(c, Quiver)]
    markers = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(images) >= 2  # basemap + terrain
    assert len(meshes) == 1
    assert len(quivers) == 1
    assert len(markers) == 1
    assert min(im.get_zorder() for im in images) == pytest.approx(0.0)
    assert max(im.get_zorder() for im in images) == pytest.approx(1.0)
    assert meshes[0].get_zorder() == pytest.approx(2.0)
    assert quivers[0].get_zorder() == pytest.approx(3.0)
    assert markers[0].get_zorder() == pytest.approx(4.0)
