import numpy as np
import pytest
from matplotlib.projections import get_projection_class

try:
    get_projection_class("astro hours mollweide")
except Exception as exc:
    pytest.skip(
        f"sky map projection unavailable: {exc}",
        allow_module_level=True,
    )

from gwexpy.plot import SkyMap


def test_skymap_init():
    """Test SkyMap initialization."""
    fig = SkyMap()
    assert len(fig.axes) == 1
    # Check default projection (astro hours mollweide usually results in WCSAxes or polar-like depending on version,
    # but we just check if it's created).
    fig.close()

def test_mark_target():
    """Test mark_target method."""
    fig = SkyMap()
    # Test with float (degrees)
    fig.mark_target(83.63, 22.01, label="Crab")
    # Test with array
    ras = [10, 20, 30]
    decs = [0, 10, 20]
    fig.mark_target(ras, decs)
    fig.close()

def test_add_healpix():
    """Test add_healpix if ligo.skymap is available."""
    try:
        import ligo.skymap  # noqa: F401 - availability check
    except ImportError:
        pytest.skip("ligo.skymap not available")

    # Create a small HEALPix map (nside=1 has 12 pixels)
    map_data = np.random.rand(12)
    fig = SkyMap()
    fig.add_healpix(map_data)
    fig.close()

def test_add_heatmap():
    """Test add_heatmap method."""
    fig = SkyMap()
    ra = np.linspace(0, 360, 10)
    dec = np.linspace(-90, 90, 5)
    ra_grid, dec_grid = np.meshgrid(ra, dec)
    values = np.sin(np.deg2rad(ra_grid)) * np.cos(np.deg2rad(dec_grid))
    fig.add_heatmap(ra_grid, dec_grid, values)
    fig.close()
