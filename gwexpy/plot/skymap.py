
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from .plot import Plot

__all__ = ["SkyMap"]

class SkyMap(Plot):
    """
    An extension of :class:`gwexpy.plot.Plot` for all-sky maps using astropy.
    """
    def __init__(self, *args, **kwargs):
        # Default to Mollweide projection if not specified
        if 'projection' not in kwargs and 'geometry' not in kwargs:
             kwargs.setdefault('projection', 'mollweide')
        
        super().__init__(*args, **kwargs)

    def add_markers(self, ra, dec, **kwargs):
        """
        Add markers to the sky map.
        
        Parameters
        ----------
        ra : array-like
            Right Ascension in degrees or astropy.units.Quantity.
        dec : array-like
            Declination in degrees or astropy.units.Quantity.
        **kwargs
            Additional arguments passed to :meth:`matplotlib.axes.Axes.scatter`.
        """
        # Ensure units
        if not isinstance(ra, u.Quantity):
            ra = ra * u.deg
        if not isinstance(dec, u.Quantity):
            dec = dec * u.deg
            
        # Convert to radians for matplotlib projection if needed, 
        # but typically astropy/matplotlib handling might vary. 
        # For standard matplotlib projections (mollweide, aitoff), 
        # coords are expected in radians, and usually mapped [-pi, pi] for RA (longitude)
        # and [-pi/2, pi/2] for Dec (latitude).
        
        # However, gwpy/gwexpy Plot wraps matplotlib. 
        # If we use standard matplotlib projections, we need manual conversion.
        # If we successfully integrated wcsaxes, we would use SkyCoord.
        
        # For simplicity and robustness with standard matplotlib projections:
        ra_rad = ra.to(u.rad).value
        dec_rad = dec.to(u.rad).value
        
        # Shift RA to [-pi, pi]
        ra_rad[ra_rad > np.pi] -= 2 * np.pi
        
        for ax in self.axes:
            ax.scatter(ra_rad, dec_rad, **kwargs)
            ax.grid(True)
            
    def add_heatmap(self, ra, dec, values, **kwargs):
        # Placeholder for more complex heatmap logic if needed
        pass
