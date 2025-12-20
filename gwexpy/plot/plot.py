
from typing import TYPE_CHECKING
import numpy as np
import math
from gwpy.plot import Plot as BasePlot

if TYPE_CHECKING:
    from gwexpy.types.seriesmatrix import SeriesMatrix
    from gwexpy.frequencyseries import FrequencySeriesMatrix
    from gwpy.types.series import Series

__all__ = ["Plot"]

class Plot(BasePlot):
    """
    An extension of :class:`gwpy.plot.Plot` that automatically handles
    :class:`gwexpy.types.SeriesMatrix` arguments by expanding them into
    individual :class:`gwpy.types.Series` objects, while preserving
    matrix layout and metadata where possible.
    """

    def __init__(self, *args, **kwargs):
        # Local import to avoid circular dependency
        from gwexpy.types.seriesmatrix import SeriesMatrix
        from gwexpy.frequencyseries import FrequencySeriesMatrix
        from astropy import units as u
        
        # 1. Inspect args to find SeriesMatrix objects
        matrices = [arg for arg in args if isinstance(arg, SeriesMatrix)]
        
        # We peek at 'separate' but do NOT pop it, so gwpy can use it too.
        separate = kwargs.get('separate', None)
        
        use_smart_layout = False
        
        # Only intervene with geometry calculation if we have Matrices
        if matrices and separate:
            use_smart_layout = True
            
            if 'geometry' not in kwargs:
                ref_matrix = matrices[0]
                nrow = len(ref_matrix.row_keys())
                ncol = len(ref_matrix.col_keys())
                kwargs['geometry'] = (nrow, ncol)
        
            # Set default figsize if geometry is present
            if 'figsize' not in kwargs:
                 nrow, ncol = kwargs['geometry']
                 # Increase default size per subplot to prevent crowding
                 # 6 inches width per col, 4 inches height per row is a good robust default
                 # But we clamp it to avoid massive figures that crash notebooks
                 fig_width = min(24, 6 * ncol)
                 fig_height = min(24, 4 * nrow) 
                 kwargs['figsize'] = (fig_width, fig_height)
            
            # Use constrained_layout for better automatic spacing preventing overlap
            kwargs.setdefault('constrained_layout', True)

            # Enforce sharex/sharey for matrix grids to reduce clutter
            kwargs.setdefault('sharex', True)
            kwargs.setdefault('sharey', True)
             
            # Default log scales for FrequencySeries
            if isinstance(matrices[0], FrequencySeriesMatrix):
                 ref = matrices[0]
                 
                 # Y Scale Logic: Default to log unless unit is Phase/Angle/dB/GroupDelay
                 if 'yscale' not in kwargs:
                     try:
                         # Attempt to check unit of first element
                         # Currently ref.meta[0,0] is safest accessor without computing Full Series
                         unit = ref.meta[0, 0].unit
                         
                         is_linear = False
                         
                         # Check strict matches
                         if unit == u.deg or unit == u.rad or unit == u.dB:
                             is_linear = True
                         else:
                             # String based checks for composite units or non-standard ones
                             u_str = str(unit)
                             if 'deg' in u_str or 'rad' in u_str or 'dB' in u_str:
                                 is_linear = True
                         
                         # Check name for keywords
                         name_str = str(getattr(ref, 'name', '')).lower()
                         if 'delay' in name_str or 'phase' in name_str or 'angle' in name_str:
                             is_linear = True
                         
                         if not is_linear:
                             kwargs['yscale'] = 'log'
                     
                     except Exception:
                         # If checking fails, default to log for FrequencySeries is standard behavior
                         kwargs['yscale'] = 'log'

                 # X Scale Logic: Default to log if sample count is large (>128)
                 if 'xscale' not in kwargs:
                     # Check sample size
                     # ref.shape is (nrow, ncol, nsample)
                     if len(ref.shape) >= 3:
                         nsample = ref.shape[-1]
                         if nsample > 256:
                             kwargs['xscale'] = 'log'

        # 2. Expand args
        new_args = []
        
        if use_smart_layout:
             # Matrix Grid Expansion (Row-Major)
             ref_matrix = matrices[0]
             r_keys = ref_matrix.row_keys()
             c_keys = ref_matrix.col_keys()
             
             for r in r_keys:
                 for c in c_keys:
                     for arg in args:
                         if isinstance(arg, SeriesMatrix):
                             val = arg[r, c]
                             new_args.append(val)
                         else:
                             # Append extra args. gwpy will distribute them to the grid.
                             new_args.append(arg)
        else:
             # Default behavior (Flat or Stacked)
             for arg in args:
                if isinstance(arg, SeriesMatrix):
                    new_args.extend(arg.to_series_1Dlist())
                else:
                    new_args.append(arg)

        # 3. Call super
        super().__init__(*new_args, **kwargs)
        
        # 4. Apply Metadata (Labels/Titles) for Matrix Layout
        if use_smart_layout:
             axes = self.axes
             ref_matrix = matrices[0]
             
             row_names = ref_matrix.rows.names
             col_names = ref_matrix.cols.names
             
             if 'geometry' in kwargs:
                 nrow, ncol = kwargs['geometry']
             else:
                 nrow, ncol = len(row_names), len(col_names)

             # Apply Row Labels (Y-axis of first column)
             for i, name in enumerate(row_names):
                 idx = i * ncol
                 if idx < len(axes):
                     if name:
                         axes[idx].set_ylabel(str(name))
            
             # Apply Col Labels (Title of first row)
             for j, name in enumerate(col_names):
                 idx = j
                 if idx < len(axes):
                     if name:
                         axes[idx].set_title(str(name))
                         
             # Extra polish: If constrained_layout is NOT used (e.g. user set it False), 
             # try tight_layout. But we defaulted it to True above.
             # self.tight_layout() # Not needed if constrained_layout=True

             # Add Legends
             for ax in axes:
                 handles, labels = ax.get_legend_handles_labels()
                 if labels:
                     ax.legend()
