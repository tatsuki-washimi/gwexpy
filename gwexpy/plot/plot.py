
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
        from gwexpy.spectrogram import Spectrogram, SpectrogramMatrix
        from astropy import units as u
        
        # 1. Inspect args to find SeriesMatrix objects
        matrices = [arg for arg in args if isinstance(arg, (SeriesMatrix, SpectrogramMatrix))]
        
        # subplots argument support (as alias/expansion of separate)
        subplots = kwargs.pop('subplots', None)
        separate = kwargs.pop('separate', None)
        
        # Priority: explicit 'separate' then 'subplots'
        if separate is None:
            separate = subplots
        
        subplots_orig = separate

        if separate == 'row':
             # One axes per row, all columns overlaid
             use_smart_layout = True
             separate = True
             if matrices:
                  ref = matrices[0]
                  kwargs.setdefault('geometry', (len(ref.row_keys()), 1))
        elif separate == 'col':
             # One axes per column, all rows overlaid
             use_smart_layout = True
             separate = True
             if matrices:
                  ref = matrices[0]
                  kwargs.setdefault('geometry', (1, len(ref.col_keys())))
        elif separate is True:
             use_smart_layout = True
             if matrices and 'geometry' not in kwargs:
                 ref = matrices[0]
                 kwargs['geometry'] = (len(ref.row_keys()), len(ref.col_keys()))
        elif separate is None and matrices:
             # Default for matrices
             use_smart_layout = True
             separate = True
             ref = matrices[0]
             kwargs.setdefault('geometry', (len(ref.row_keys()), len(ref.col_keys())))

        kwargs['separate'] = separate

        # If we have matrices and are using smart layout, determine axes/sharing
        if use_smart_layout and matrices:
            # Disable sharex/sharey if mixed types or units are present
            from gwexpy.timeseries import TimeSeriesMatrix
            has_time = any(isinstance(m, TimeSeriesMatrix) for m in matrices)
            has_freq = any(isinstance(m, FrequencySeriesMatrix) for m in matrices)
            
            units_consistent = True
            for m in matrices:
                consistent, _ = m._all_element_units_equivalent()
                if not consistent:
                    units_consistent = False
                    break
            
            if (has_time and has_freq) or not units_consistent:
                 kwargs.setdefault('sharex', False)
                 kwargs.setdefault('sharey', False)
            else:
                 kwargs.setdefault('sharex', True)
                 kwargs.setdefault('sharey', True)
        
        # Detect if we are plotting Spectrograms
        is_spectrogram = any(isinstance(arg, Spectrogram) for arg in args)
        if is_spectrogram:
             kwargs.setdefault('yscale', 'log')
             kwargs.setdefault('xscale', 'linear')
             # For spectrograms, sharing axes is often tricky with colorbars, 
             # but we can try to enable it if requested or by default.
             kwargs.setdefault('sharex', True)
             # sharey=True is good for spectrograms to keep frequency axes aligned
             kwargs.setdefault('sharey', True)
             
             # Adjust figsize for 2D plots which need more space for colorbars
             if 'geometry' in kwargs and 'figsize' not in kwargs:
                  nrow, ncol = kwargs['geometry']
                  fig_width = min(24, 8 * ncol)
                  fig_height = min(24, 5 * nrow)
                  kwargs['figsize'] = (fig_width, fig_height)
        
        # Set default figsize if geometry is present (for matrices)
        if use_smart_layout and matrices:
            if 'figsize' not in kwargs and 'geometry' in kwargs:
                 nrow, ncol = kwargs['geometry']
                 # Increase default size per subplot to prevent crowding
                 # 6 inches width per col, 4 inches height per row is a good robust default
                 # But we clamp it to avoid massive figures that crash notebooks
                 fig_width = min(24, 6 * ncol)
                 fig_height = min(24, 4 * nrow) 
                 kwargs['figsize'] = (fig_width, fig_height)
            
            # Use constrained_layout for better automatic spacing preventing overlap
            kwargs.setdefault('constrained_layout', True)

            # Enforce sharex for matrix grids to reduce clutter
            kwargs.setdefault('sharex', True)

            # Default log scales for FrequencySeries
            if isinstance(matrices[0], FrequencySeriesMatrix):
                 ref = matrices[0]
                 
                 # Y Scale Logic: Default to log unless unit is Phase/Angle/dB/GroupDelay
                 if 'yscale' not in kwargs:
                     try:
                         # Attempt to check unit of first element
                         # Currently ref.meta[0,0] is safest accessor without computing Full Series
                         if ref.shape[0] > 0 and ref.shape[1] > 0:
                             try:
                                 # Standard accessor
                                 unit = ref.meta[0, 0].unit
                             except (AttributeError, TypeError, IndexError):
                                 # Robust fallback: use indexing if meta attribute is missing or non-subscriptable
                                 r0 = ref.row_keys()[0]
                                 c0 = ref.col_keys()[0]
                                 unit = ref[r0, c0].unit
                             
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
        matrix_args = [arg for arg in args if isinstance(arg, (SeriesMatrix, SpectrogramMatrix))]
        other_args = [arg for arg in args if not isinstance(arg, (SeriesMatrix, SpectrogramMatrix))]
        
        new_args = []
        use_overlay = False
        
        if use_smart_layout and matrix_args:
             # Matrix Grid Expansion
             base_m = matrix_args[0]
             r_keys = base_m.row_keys()
             c_keys = base_m.col_keys()
             
             # subplots_orig stores the original row/col hint if provided
             if subplots_orig == 'row':
                  # One axes per row, overlay columns
                  for r in r_keys:
                       row_elements = []
                       for c in c_keys:
                            val = base_m[r, c]
                            if not getattr(val, 'name', None):
                                 val.name = f"{r} / {c}"
                            row_elements.append(val)
                       new_args.append(row_elements)
                  use_overlay = True
             elif subplots_orig == 'col':
                  # One axes per column, overlay rows
                  for c in c_keys:
                       col_elements = []
                       for r in r_keys:
                            val = base_m[r, c]
                            if not getattr(val, 'name', None):
                                 val.name = f"{r} / {c}"
                            col_elements.append(val)
                       new_args.append(col_elements)
                  use_overlay = True
             else:
                  # Full Grid (Row-Major)
                  for r in r_keys:
                      for c in c_keys:
                          val = base_m[r, c]
                          if not getattr(val, 'name', None):
                               val.name = f"{r} / {c}"
                          new_args.append(val)
                  use_overlay = True
             
             # Any additional matrices or single series will be overlayed manually
             extra_matrix_args = matrix_args[1:]
        else:
             # Default behavior (Flat or Stacked)
             for arg in args:
                if isinstance(arg, SeriesMatrix):
                    new_args.extend(arg.to_series_1Dlist())
                else:
                    new_args.append(arg)

        # 3. Call super
        # Separate figure-level and gwpy-specific args from artist args
        # because GWpy often passes kwargs to line plotting calls.
        layout_kwargs = {}
        for k in ['separate', 'geometry', 'sharex', 'sharey', 'xscale', 'yscale', 
                  'xlim', 'ylim', 'xlabel', 'ylabel', 'title', 'legend']:
            if k in kwargs:
                layout_kwargs[k] = kwargs.pop(k)
        
        # Figure constructor args
        fig_params = {}
        for k in ['figsize', 'dpi', 'facecolor', 'edgecolor', 'linewidth', 
                  'frameon', 'subplotpars']:
            if k in kwargs:
                fig_params[k] = kwargs.pop(k)
        
        # Layout engines (often cause issues if passed to super when super passes them to plot())
        use_cl = kwargs.pop('constrained_layout', False)
        use_tl = kwargs.pop('tight_layout', False)
        
        super().__init__(*new_args, **layout_kwargs, **fig_params, **kwargs)
        
        # Apply layout engines manually to the figure
        if use_cl:
             try:
                 self.set_constrained_layout(True)
             except:
                 pass
        if use_tl:
             try:
                 self.tight_layout()
             except:
                 pass
        
        # Overlay additional matrices and other_args on separate grid if applicable
        if use_overlay:
             axes = self.axes
             ref_matrix = matrix_args[0]
             r_keys = list(ref_matrix.row_keys())
             c_keys = list(ref_matrix.col_keys())
             ncol = len(c_keys)
             
             # Helper to plot on axis
             def _plot_on_ax(ax, other):
                 if hasattr(other, 'times'):
                      ax.plot(other.times, other.value, label=getattr(other, 'name', None))
                 elif hasattr(other, 'frequencies'):
                      ax.plot(other.frequencies, other.value, label=getattr(other, 'name', None))
                 else:
                      ax.plot(other)

             for i, r in enumerate(r_keys):
                  for j, c in enumerate(c_keys):
                       if subplots_orig == 'row':
                            ax_idx = i
                       elif subplots_orig == 'col':
                            ax_idx = j
                       else:
                            ax_idx = i * ncol + j
                            
                       if ax_idx >= len(axes):
                            break
                       ax = axes[ax_idx]
                       
                       # 1. Overlay extra matrices
                       for m in extra_matrix_args:
                            try:
                                 val = m[r, c]
                                 _plot_on_ax(ax, val)
                            except:
                                 pass
                                 
                       # 2. Overlay single series (only once per axis)
                       if (subplots_orig == 'row' and j == 0) or \
                          (subplots_orig == 'col' and i == 0) or \
                          (subplots_orig not in ('row', 'col')):
                            for other in other_args:
                                 _plot_on_ax(ax, other)
        
        # 4. Apply Metadata (Labels/Titles) for Matrix Layout
        if use_smart_layout and matrices:
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
             show_legend = layout_kwargs.get('legend', True)
             if show_legend:
                  for ax in axes:
                      handles, labels = ax.get_legend_handles_labels()
                      if labels:
                          ax.legend()

        # 5. Automatic Colorbars for Spectrograms
        if is_spectrogram:
             # Find all spectrogram image artists and add colorbars to their axes
             # GWpy's add_colorbar works on the current figure and can take an ax
             for ax in self.axes:
                  # Check if this axes has an image (spectrograms are plotted as collections or images)
                  # Usually they are QuadMesh objects from pcolormesh
                  from matplotlib.collections import QuadMesh
                  from matplotlib.image import AxesImage
                  
                  mappable = None
                  for child in ax.get_children():
                       if isinstance(child, (QuadMesh, AxesImage)):
                            mappable = child
                            break
                  
                  if mappable:
                       try:
                            # We use self.add_colorbar which is a gwpy method
                            # It creates a new axis for the colorbar
                            self.add_colorbar(ax=ax, mappable=mappable, fraction=0.046, pad=0.04)
                       except Exception:
                            # Fallback to standard matplotlib if needed
                            try:
                                 self.colorbar(mappable, ax=ax)
                            except:
                                 pass

        # Final layout polish
        if kwargs.get('constrained_layout', True):
             try:
                 self.set_constrained_layout(True)
             except:
                 pass
