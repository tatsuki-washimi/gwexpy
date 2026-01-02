"""Geographic map plotting using PyGMT backend with Cartopy-like interface."""

import numpy as np

try:
    import pygmt
    HAS_PYGMT = True
except ImportError:
    HAS_PYGMT = False

# 検出器座標データベース (簡易版)
DETECTORS = {
    'K1': {'lon': 137.31, 'lat': 36.41, 'name': 'KAGRA', 'color': 'darkgreen'},
    'H1': {'lon': -119.41, 'lat': 46.45, 'name': 'LIGO Hanford', 'color': 'darkred'},
    'L1': {'lon': -90.77, 'lat': 30.56, 'name': 'LIGO Livingston', 'color': 'darkblue'},
    'V1': {'lon': 10.5, 'lat': 43.63, 'name': 'Virgo', 'color': 'darkorange'},
    'G1': {'lon': 9.80, 'lat': 52.24, 'name': 'GEO600', 'color': 'darkmagenta'},
}

# 投影法のマッピング (Friendly Name -> GMT Code)
PROJECTIONS = {
    'Mercator': 'M',
    'Robinson': 'N',
    'Mollweide': 'W',
    'PlateCarree': 'Q',  # Cylindrical Equidistant
    'Equidistant': 'Q',
}

# マーカーのマッピング (Matplotlib -> GMT)
MARKERS = {
    'o': 'c', # circle
    's': 's', # square
    '^': 't', # triangle
    'd': 'd', # diamond
    '+': '+', # plus
    'x': 'x', # cross
    '*': 'a', # star
}

class GeoMap:
    """A PyGMT wrapper that provides a Cartopy-like interface.
    """
    def __init__(self, projection='Robinson', center_lon=0, width='15c', **kwargs):
        """
        Initialize a new GeoMap.

        Parameters
        ----------
        projection : str, optional
            The projection name (e.g., 'Robinson', 'Mercator', 'Mollweide').
        center_lon : float, optional
            Central longitude for the projection.
        width : str, optional
            Width of the map (GMT-style, e.g., '15c' for 15 cm).
        **kwargs
            Additional arguments:
            region: GMT region (default 'd' for global).
            frame: GMT frame (default 'afg').
        """
        if not HAS_PYGMT:
            raise ImportError("pygmt is required for GeoMap. Install with: pip install pygmt")
        
        self.fig = pygmt.Figure()
        
        # 投影法の設定
        proj_code = PROJECTIONS.get(projection, projection)
        
        # GMTの投影指定を作成
        # region 'd' is -180/180/-90/90
        self.region = kwargs.get('region', 'd')
        
        if center_lon != 0:
            self.projection = f"{proj_code}{center_lon}/{width}"
        else:
            self.projection = f"{proj_code}{width}"
            
        # GMT Error: Option -R: Cannot include south/north poles with Mercator projection!
        if proj_code == 'M' and self.region == 'd':
            self.region = [-180, 180, -80, 80] # Trim poles for Mercator
            
        self.frame = kwargs.get('frame', 'ag') # auto, grid

        # ベースマップの初期化
        self.fig.basemap(region=self.region, projection=self.projection, frame=self.frame)

    def add_coastlines(self, resolution='low', color='black', linewidth=0.5):
        """Draw coastlines (Cartopy-like API)."""
        res_map = {'low': 'l', 'medium': 'i', 'high': 'h', 'crude': 'c', 'full': 'f'}
        res = res_map.get(resolution, 'l')
        
        pen = f"{linewidth}p,{color}"
        self.fig.coast(shorelines=pen, resolution=res, area_thresh=10000)

    def fill_continents(self, color='lightgray'):
        """Fill land colors."""
        self.fig.coast(land=color)

    def fill_oceans(self, color='azure'):
        """Fill water colors."""
        self.fig.coast(water=color)

    def plot(self, x, y, color='blue', marker='o', markersize=10, label=None, **kwargs):
        """Plot points (Matplotlib-like API)."""
        gmt_marker = MARKERS.get(marker, 'c')
        style = f"{gmt_marker}{markersize}p"
        
        self.fig.plot(
            x=x, y=y,
            style=style,
            fill=color,
            pen="0.5p,black",
            label=label,
            **kwargs
        )

    def text(self, x, y, text, **kwargs):
        """Add text at (x, y)."""
        # PyGMT text expects justify, font, etc.
        self.fig.text(x=x, y=y, text=text, **kwargs)

    def plot_detector(self, name, label=True, **kwargs):
        """Plot a gravitational wave detector by name (e.g., 'K1')."""
        if name not in DETECTORS:
            raise ValueError(f"Unknown detector: {name}")
        
        det = DETECTORS[name]
        
        plot_kwargs = {'marker': '*', 'markersize': 15, 'color': det['color']}
        plot_kwargs.update(kwargs)
        
        self.plot(det['lon'], det['lat'], **plot_kwargs)
        
        if label:
            self.text(det['lon'], det['lat'] + 5, text=det['name'], 
                      font="10p,Helvetica-Bold,black", justify="CM")

    def show(self):
        """Display the plot."""
        self.fig.show()
    
    def save(self, filename, **kwargs):
        """Save the plot."""
        self.fig.savefig(filename, **kwargs)
