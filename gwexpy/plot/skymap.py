import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .plot import Plot

# Attempt to import ligo.skymap for HEALPix support
try:
    # Workaround for scipy 1.14+ where trapz was removed but still used by healpy/ligo.skymap
    import scipy.integrate

    if not hasattr(scipy.integrate, "trapz"):
        scipy.integrate.trapz = getattr(scipy.integrate, "trapezoid", None)

    # Also patch sys.modules to be extremely safe
    import sys

    if "scipy.integrate" in sys.modules:
        setattr(sys.modules["scipy.integrate"], "trapz", scipy.integrate.trapz)

    import ligo.skymap.plot  # noqa: F401

    HAS_LIGO_SKYMAP = True
except (ImportError, AttributeError):
    logging.getLogger(__name__).debug(
        "ligo.skymap not found or could not be patched. SkyMap features will be limited."
    )
    HAS_LIGO_SKYMAP = False

__all__ = ["SkyMap"]


class SkyMap(Plot):
    """A Plot subclass for all‑sky maps.

    This class provides convenient methods to display HEALPix probability maps
    (using :mod:`ligo.skymap`) and to overlay astronomical targets.
    """

    def __init__(self, *args, **kwargs):
        """Create a SkyMap.

        Parameters
        ----------
        *args, **kwargs : passed to :class:`gwexpy.plot.Plot`.
        """
        # Default projection: Mollweide in hour angle (RA) units
        if "projection" not in kwargs and "geometry" not in kwargs:
            kwargs.setdefault("projection", "astro hours mollweide")

        # Ensure at least one axis if no data provided
        if not args and "geometry" not in kwargs:
            kwargs.setdefault("geometry", (1, 1))

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # HEALPix handling
    # ------------------------------------------------------------------
    def add_healpix(self, map_data, **kwargs):
        """Add a HEALPix probability map to the sky plot.

        Parameters
        ----------
        map_data : array‑like
            HEALPix map data (e.g., a ``numpy`` array of probabilities).
        **kwargs : additional keyword arguments passed to ``ligo.skymap.plot.imshow_hpx``.
        """
        if not HAS_LIGO_SKYMAP:
            # Try once more to patch and import if it was not successful initially
            try:
                import scipy.integrate

                scipy.integrate.trapz = getattr(scipy.integrate, "trapezoid", None)
            except (ImportError, AttributeError):
                raise ImportError(
                    "ligo.skymap is required for add_healpix. Install with: pip install ligo.skymap"
                )
        ax = self.gca()
        # ``imshow_hpx`` handles the projection internally.
        im = ax.imshow_hpx(map_data, **kwargs)
        # Add a colorbar if not suppressed by the caller.
        if kwargs.get("colorbar", True):
            self.colorbar(im, ax=ax, label="Probability")
        return im

    # ------------------------------------------------------------------
    # Target markers
    # ------------------------------------------------------------------
    def mark_target(self, ra, dec, label=None, **kwargs):
        """Mark a sky position on the map.

        Parameters
        ----------
        ra, dec : array‑like or ``Quantity``
            Right‑ascension and declination. If plain numbers are supplied they are
            interpreted as degrees.
        label : str, optional
            Text label to place next to the marker.
        **kwargs : additional arguments passed to the underlying Matplotlib ``scatter``
            or ``plot_coord`` call.
        """
        # Ensure astropy quantities
        if not isinstance(ra, u.Quantity):
            ra = np.atleast_1d(ra) * u.deg
        else:
            ra = np.atleast_1d(ra)
        if not isinstance(dec, u.Quantity):
            dec = np.atleast_1d(dec) * u.deg
        else:
            dec = np.atleast_1d(dec)

        # Convert to SkyCoord for convenience
        coord = SkyCoord(ra, dec, frame="icrs")
        ax = self.gca()
        # Prefer gwpy's ``plot_coord`` if available; otherwise fall back to scatter.
        try:
            ax.plot_coord(coord, **kwargs)
        except (AttributeError, TypeError, ValueError):
            # Matplotlib expects radians for Mollweide
            ra_rad = coord.ra.to(u.rad).value
            dec_rad = coord.dec.to(u.rad).value
            # Shift RA to [-π, π] for the Mollweide projection
            ra_rad[ra_rad > np.pi] -= 2 * np.pi
            ax.scatter(ra_rad, dec_rad, **kwargs)
        if label:
            # Place label slightly offset from the marker
            # We must pass scalars to ax.text
            tx = (
                float(coord.ra.rad[0]) if not coord.ra.isscalar else float(coord.ra.rad)
            )
            ty = (
                float(coord.dec.rad[0])
                if not coord.dec.isscalar
                else float(coord.dec.rad)
            )
            # Fallback for standard Matplotlib geo projections if ligo.skymap is missing
            try:
                transform = ax.get_transform("world")
            except (AttributeError, TypeError):
                transform = ax.transData
                # Shift RA to [-π, π] for standard geo axes
                if tx > np.pi:
                    tx -= 2 * np.pi
            ax.text(tx, ty, f" {label}", transform=transform)

    # ------------------------------------------------------------------
    # Heatmap overlay
    # ------------------------------------------------------------------
    def add_heatmap(self, ra, dec, values, **kwargs):
        """Overlay a heatmap defined on sky coordinates.

        Parameters
        ----------
        ra, dec : array‑like (degrees)
            Grid of right‑ascension and declination values.
        values : 2‑D array
            Data values corresponding to each (ra, dec) point.
        **kwargs : additional arguments passed to ``pcolormesh``.
        """
        # Convert to radians for the Mollweide projection
        ra = np.asarray(ra)
        dec = np.asarray(dec)
        if not isinstance(ra.flat[0], u.Quantity):
            ra = ra * u.deg
        if not isinstance(dec.flat[0], u.Quantity):
            dec = dec * u.deg
        ra_rad = ra.to(u.rad).value
        dec_rad = dec.to(u.rad).value
        # Shift RA to the range expected by Mollweide
        ra_rad[ra_rad > np.pi] -= 2 * np.pi
        ax = self.gca()
        mesh = ax.pcolormesh(ra_rad, dec_rad, values, **kwargs)
        self.colorbar(mesh, ax=ax)
        return mesh
