import pytest

from gwexpy.plot import GeoMap


def _require_pygmt_runtime():
    try:
        import pygmt
        from pygmt.exceptions import GMTCLibError
    except ImportError:
        pytest.skip("pygmt not available")
    try:
        pygmt.Figure()
    except GMTCLibError as exc:
        pytest.skip(f"GMT runtime not available ({exc})")

def test_geomap_init():
    """Test GeoMap initialization."""
    _require_pygmt_runtime()

    gmap = GeoMap(projection='Robinson')
    assert gmap.projection.startswith('N')
    assert gmap.region == 'd'

def test_geomap_methods():
    """Test GeoMap drawing methods."""
    _require_pygmt_runtime()

    gmap = GeoMap(projection='Mercator', center_lon=135)
    gmap.add_coastlines()
    gmap.fill_continents(color='green')
    gmap.fill_oceans(color='blue')
    gmap.plot(137.31, 36.41, marker='*')
    gmap.text(137.31, 36.41 + 2, text="KAGRA")
    gmap.plot_detector('H1')

def test_geomap_save(tmp_path):
    """Test GeoMap saving."""
    _require_pygmt_runtime()

    gmap = GeoMap()
    gmap.add_coastlines()
    p = tmp_path / "test_map.png"
    gmap.save(str(p))
    assert p.exists()
