import pytest

from gwexpy.plot import GeoMap


def _require_pygmt_runtime():
    try:
        import pygmt
        from pygmt.exceptions import GMTCLibError
    except Exception as exc:
        pytest.skip(f"PyGMT runtime not available ({exc})")
    try:
        pygmt.Figure()
    except GMTCLibError as exc:
        pytest.skip(f"GMT runtime not available ({exc})")


def test_geomap_init():
    """Test GeoMap initialization."""
    _require_pygmt_runtime()

    gmap = GeoMap(projection="Robinson")
    assert gmap.projection.startswith("N")
    assert gmap.region == "d"


def test_geomap_methods():
    """Test GeoMap drawing methods."""
    _require_pygmt_runtime()

    gmap = GeoMap(projection="Mercator", center_lon=135)
    gmap.add_coastlines()
    gmap.fill_continents(color="green")
    gmap.fill_oceans(color="blue")
    gmap.plot(137.31, 36.41, marker="*")
    gmap.text(137.31, 36.41 + 2, text="KAGRA")
    gmap.plot_detector("H1")


def test_geomap_save(tmp_path):
    """Test GeoMap saving."""
    _require_pygmt_runtime()

    gmap = GeoMap()
    gmap.add_coastlines()
    p = tmp_path / "test_map.png"
    gmap.save(str(p))
    assert p.exists()


def test_geomap_field_to_grid_preserves_values_after_lat_sort():
    """``_field_to_grid`` squeezes 4D angle fields and flips descending lat."""
    pytest.importorskip("xarray")
    from types import SimpleNamespace

    import numpy as np
    from astropy import units as u

    from gwexpy.plot.geomap import _field_to_grid

    lon = np.array([137.0, 137.2, 137.4]) * u.deg
    lat_desc = np.array([36.6, 36.4, 36.2, 36.0]) * u.deg
    # data[0, ix, iy, 0] = 100 * ix + iy so every cell is uniquely identifiable
    raw = np.zeros((1, 3, 4, 1), dtype=float)
    for ix in range(3):
        for iy in range(4):
            raw[0, ix, iy, 0] = 100.0 * ix + iy

    angle_field = SimpleNamespace(
        value=raw,
        shape=raw.shape,
        unit=u.m,
        axis_names=("t", "lon", "lat", "z"),
        _axis0_name="t",
        _axis1_name="lon",
        _axis2_name="lat",
        _axis3_name="z",
        _axis0_index=np.array([0.0]) * u.s,
        _axis1_index=lon,
        _axis2_index=lat_desc,
        _axis3_index=np.array([0.0]) * u.m,
        space_domains={"lon": "angle", "lat": "angle", "z": "real"},
    )

    da = _field_to_grid(angle_field)
    assert da.dims == ("lat", "lon")
    assert np.all(np.diff(da.coords["lat"].values) > 0)
    # Value at (lon=137.4 [ix=2], lat=36.6 [original iy=0]) must remain 200.0
    assert float(da.sel(lon=137.4, lat=36.6).values) == pytest.approx(200.0)
    # Value at (lon=137.2 [ix=1], lat=36.0 [original iy=3]) must remain 103.0
    assert float(da.sel(lon=137.2, lat=36.0).values) == pytest.approx(103.0)


def test_geomap_add_field_rejects_real_domain():
    """Real-space fields raise ValueError when passed to GeoMap.add_field."""
    _require_pygmt_runtime()
    import numpy as np
    from astropy import units as u

    from gwexpy.fields import ScalarField

    sf = ScalarField(
        np.ones((1, 3, 3, 1)),
        axis0=np.array([0.0]) * u.s,
        axis1=np.arange(3, dtype=float) * u.m,
        axis2=np.arange(3, dtype=float) * u.m,
        axis3=np.array([0.0]) * u.m,
        space_domain="real",
    )
    gmap = GeoMap(projection="Mercator", region=[137.0, 137.5, 36.0, 36.5])
    with pytest.raises(ValueError, match="FieldPlot"):
        gmap.add_field(sf)


def test_geomap_add_field_and_colorbar():
    """GeoMap.add_field renders an angle-domain field and colorbar."""
    _require_pygmt_runtime()
    pytest.importorskip("xarray")
    from types import SimpleNamespace

    import numpy as np
    from astropy import units as u

    lon = np.linspace(137.0, 137.4, 4) * u.deg
    lat = np.linspace(36.5, 36.1, 4) * u.deg
    raw = np.arange(16, dtype=float).reshape(1, 4, 4, 1)
    angle_field = SimpleNamespace(
        value=raw,
        shape=raw.shape,
        unit=u.m,
        axis_names=("t", "lon", "lat", "z"),
        _axis0_name="t",
        _axis1_name="lon",
        _axis2_name="lat",
        _axis3_name="z",
        _axis0_index=np.array([0.0]) * u.s,
        _axis1_index=lon,
        _axis2_index=lat,
        _axis3_index=np.array([0.0]) * u.m,
        space_domains={"lon": "angle", "lat": "angle", "z": "real"},
    )
    gmap = GeoMap(projection="Mercator", region=[137.0, 137.4, 36.1, 36.5])
    gmap.add_field(angle_field, cmap="viridis", alpha=0.8)
    gmap.add_colorbar(label="Elevation [m]")


@pytest.mark.network
def test_geomap_add_relief():
    """GeoMap.add_relief fetches GMT earth_relief data (network required)."""
    _require_pygmt_runtime()
    gmap = GeoMap(projection="Mercator", region=[137.2, 137.4, 36.3, 36.5])
    gmap.add_relief(resolution="15s")


def test_geomap_extensions_dispatch_with_mocked_pygmt(monkeypatch):
    """Verify add_relief, add_field, and add_colorbar forward expected PyGMT args."""
    pytest.importorskip("xarray")
    from types import SimpleNamespace

    import numpy as np
    from astropy import units as u

    import gwexpy.plot.geomap as geomap_mod

    calls = []

    class _FakeFigure:
        def basemap(self, **kwargs):
            calls.append(("basemap", kwargs))

        def grdimage(self, **kwargs):
            calls.append(("grdimage", kwargs))

        def colorbar(self, **kwargs):
            calls.append(("colorbar", kwargs))

    fake_pygmt = SimpleNamespace(Figure=_FakeFigure)
    monkeypatch.setattr(geomap_mod, "HAS_PYGMT", True)
    monkeypatch.setattr(geomap_mod, "pygmt", fake_pygmt)

    gmap = geomap_mod.GeoMap(projection="Mercator", region=[137.0, 137.4, 36.1, 36.5])
    gmap.add_relief(resolution="03s", shading=False, cmap="terrain")
    assert calls[-1][0] == "grdimage"
    assert calls[-1][1]["grid"] == "@earth_relief_03s"
    assert calls[-1][1]["shading"] is False
    assert calls[-1][1]["cmap"] == "terrain"

    lon = np.linspace(137.0, 137.4, 4) * u.deg
    lat = np.linspace(36.5, 36.1, 4) * u.deg
    angle_field = SimpleNamespace(
        value=np.arange(16, dtype=float).reshape(1, 4, 4, 1),
        shape=(1, 4, 4, 1),
        unit=u.m,
        axis_names=("t", "lon", "lat", "z"),
        _axis0_name="t",
        _axis1_name="lon",
        _axis2_name="lat",
        _axis3_name="z",
        _axis0_index=np.array([0.0]) * u.s,
        _axis1_index=lon,
        _axis2_index=lat,
        _axis3_index=np.array([0.0]) * u.m,
        space_domains={"lon": "angle", "lat": "angle", "z": "real"},
    )
    gmap.add_field(angle_field, cmap="viridis", alpha=0.75)
    assert calls[-1][0] == "grdimage"
    assert calls[-1][1]["cmap"] == "viridis"
    assert calls[-1][1]["transparency"] == pytest.approx(25.0)
    assert calls[-1][1]["grid"].dims == ("lat", "lon")

    with pytest.raises(ValueError, match="alpha"):
        gmap.add_field(angle_field, alpha=1.5)

    gmap.add_colorbar(label="Elevation [m]")
    assert calls[-1] == ("colorbar", {"frame": "af+lElevation [m]"})
