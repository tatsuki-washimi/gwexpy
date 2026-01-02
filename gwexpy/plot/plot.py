import numpy as np
from typing import TYPE_CHECKING
from gwpy.plot import Plot as BasePlot

if TYPE_CHECKING:
    pass

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
        from gwexpy.frequencyseries import (
            FrequencySeriesMatrix,
            FrequencySeriesList,
            FrequencySeriesDict,
            FrequencySeries
        )
        from gwexpy.spectrogram import (
            Spectrogram,
            SpectrogramMatrix,
            SpectrogramList,
            SpectrogramDict
        )
        from gwexpy.plot import defaults

        # 1. Unpack Collections
        matrices = []
        expanded_args = []
        
        has_matrix = False

        for arg in args:
            if isinstance(arg, (SeriesMatrix, SpectrogramMatrix)):
                matrices.append(arg)
                has_matrix = True
            elif isinstance(arg, (FrequencySeriesList, SpectrogramList)):
                expanded_args.extend(arg)
            elif isinstance(arg, (FrequencySeriesDict, SpectrogramDict)):
                expanded_args.extend(arg.values())
            elif isinstance(arg, (list, tuple)):
                expanded_args.extend(arg)
            elif isinstance(arg, dict):
                expanded_args.extend(arg.values())
            else:
                expanded_args.append(arg)


        # 2. Defaults (Figsize, Scales, Geometry)
        all_data = expanded_args[:]
        if matrices:
            all_data.append(matrices[0])
        
        # Geometry / Separate
        separate = kwargs.get('separate')
        geometry = kwargs.get('geometry')
        
        separate, geometry = defaults.determine_geometry_and_separate(
            all_data, separate=separate, geometry=geometry
        )
        
        if separate is not None:
             kwargs['separate'] = separate
        if geometry is not None:
             kwargs['geometry'] = geometry

        # Scales
        kwargs['xscale'] = defaults.determine_xscale(all_data, current_value=kwargs.get('xscale'))
        kwargs['yscale'] = defaults.determine_yscale(all_data, current_value=kwargs.get('yscale'))
        
        det_xlabel = defaults.determine_xlabel(all_data, current_value=kwargs.get('xlabel'))
        if det_xlabel is not None:
             kwargs['xlabel'] = det_xlabel
             
        det_ylabel = defaults.determine_ylabel(all_data, current_value=kwargs.get('ylabel'))
        if det_ylabel is not None:
             kwargs['ylabel'] = det_ylabel
        
        new_norm = defaults.determine_norm(all_data, current_value=kwargs.get('norm'))
        if new_norm is not None:
             kwargs['norm'] = new_norm

        det_clabel = defaults.determine_clabel(all_data)

        # Figsize
        if 'figsize' not in kwargs and 'geometry' in kwargs:
             nrow, ncol = kwargs['geometry']
             kwargs['figsize'] = defaults.calculate_default_figsize((nrow, ncol), nrow, ncol)


        # 3. Matrix Logic (Legacy adaptation)
        matrix_args = [arg for arg in args if isinstance(arg, (SeriesMatrix, SpectrogramMatrix))]
        
        use_smart_layout = False
        subplots = kwargs.pop('subplots', None)
        separate = kwargs.get('separate')

        if separate is None:
            separate = subplots

        subplots_orig = separate

        if separate == 'row' and matrix_args:
             use_smart_layout = True
             separate = True
             ref = matrix_args[0]
             kwargs.setdefault('geometry', (len(ref.row_keys()), 1))
        elif separate == 'col' and matrix_args:
             use_smart_layout = True
             separate = True
             ref = matrix_args[0]
             kwargs.setdefault('geometry', (1, len(ref.col_keys())))
        elif separate is True:
             use_smart_layout = True
             if matrix_args and 'geometry' not in kwargs:
                 ref = matrix_args[0]
                 kwargs['geometry'] = (len(ref.row_keys()), len(ref.col_keys()))
        elif separate is None and matrix_args:
             use_smart_layout = True
             separate = True
             ref = matrix_args[0]
             kwargs.setdefault('geometry', (len(ref.row_keys()), len(ref.col_keys())))
        
        # Enforce geometry update
        if 'geometry' in kwargs and 'figsize' not in kwargs:
             nrow, ncol = kwargs['geometry']
             kwargs['figsize'] = defaults.calculate_default_figsize(None, nrow, ncol)

        kwargs['separate'] = separate

        # Share Axis Logic for Matrices
        if use_smart_layout and matrix_args:
            from gwexpy.timeseries import TimeSeriesMatrix
            has_time = any(isinstance(m, TimeSeriesMatrix) for m in matrix_args)
            has_freq = any(isinstance(m, FrequencySeriesMatrix) for m in matrix_args)

            units_consistent = True
            for m in matrix_args:
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

        is_spectrogram = any(isinstance(a, (Spectrogram, SpectrogramMatrix)) for a in all_data)
        if is_spectrogram:
             kwargs.setdefault('sharex', True)
             kwargs.setdefault('sharey', True)
             # Default to pcolormesh for Spectrograms if not specified
             kwargs.setdefault('method', 'pcolormesh')

        kwargs.setdefault('constrained_layout', False)
        kwargs.setdefault('tight_layout', True)

        # 4. Final Args
        final_args = []
        use_overlay = False

        if use_smart_layout and matrix_args:
             base_m = matrix_args[0]
             r_keys = base_m.row_keys()
             c_keys = base_m.col_keys()

             if subplots_orig == 'row':
                  for r in r_keys:
                       row_elements = []
                       for c in c_keys:
                            val = base_m[r, c]
                            if not getattr(val, 'name', None):
                                 val.name = f"{r} / {c}"
                            row_elements.append(val)
                       final_args.append(row_elements)
                  use_overlay = True
             elif subplots_orig == 'col':
                  for c in c_keys:
                       col_elements = []
                       for r in r_keys:
                            val = base_m[r, c]
                            if not getattr(val, 'name', None):
                                 val.name = f"{r} / {c}"
                            col_elements.append(val)
                       final_args.append(col_elements)
                  use_overlay = True
             else:
                  for r in r_keys:
                      for c in c_keys:
                          # Wrap in Spectrogram if it's SpectrogramMatrix
                          if type(base_m).__name__ == 'SpectrogramMatrix':
                              from gwexpy.spectrogram import Spectrogram
                              ri = base_m.row_index(r)
                              ci = base_m.col_index(c)
                              if getattr(base_m, 'ndim', 0) == 3:
                                  val_data = base_m[ri]
                              else:
                                  val_data = base_m[ri, ci]
                              val = Spectrogram(val_data.view(np.ndarray), 
                                               times=base_m.times, frequencies=base_m.frequencies,
                                               unit=base_m.meta[ri, ci].unit, name=base_m.meta[ri, ci].name)
                          else:
                              # For TimeSeriesMatrix, FrequencySeriesMatrix, etc.
                              val = base_m[r, c]
                              if not getattr(val, 'name', None):
                                   val.name = f"{r} / {c}"
                          final_args.append(val)
                  use_overlay = True
        else:
             for m in matrix_args:
                 final_args.extend(m.to_series_1Dlist())
             final_args.extend(expanded_args)

        # 4.5. Optimize Large TimeSeries (Adaptive Decimation)
        from gwexpy.plot.utils import adaptive_decimate
        from gwexpy.timeseries import TimeSeries

        decimate_threshold = kwargs.pop('decimate_threshold', 50000)
        decimate_points = kwargs.pop('decimate_points', 10000)

        def _optimize_if_needed(arg):
            if isinstance(arg, TimeSeries) and len(arg) > decimate_threshold:
                return adaptive_decimate(arg, target_points=decimate_points)
            if isinstance(arg, list):
                return [_optimize_if_needed(a) for a in arg]
            if isinstance(arg, tuple):
                return tuple(_optimize_if_needed(a) for a in arg)
            if isinstance(arg, dict):
                return {k: _optimize_if_needed(v) for k, v in arg.items()}
            return arg

        final_args = [_optimize_if_needed(arg) for arg in final_args]

        # 5. Super Init
        layout_kwargs = {}
        for k in ['separate', 'geometry', 'sharex', 'sharey', 'xscale', 'yscale', 'norm',
                  'xlim', 'ylim', 'xlabel', 'ylabel', 'title', 'legend']:
            if k in kwargs:
                layout_kwargs[k] = kwargs.pop(k)

        fig_params = {}
        for k in ['figsize', 'dpi', 'facecolor', 'edgecolor', 'linewidth',
                  'frameon', 'subplotpars']:
            if k in kwargs:
                fig_params[k] = kwargs.pop(k)

        use_cl = kwargs.pop('constrained_layout', False)
        use_tl = kwargs.pop('tight_layout', False)

        # Store labels to ensure application
        force_ylabel = layout_kwargs.get('ylabel')
        force_xlabel = layout_kwargs.get('xlabel')

        # Pop 'ax' if it exists to avoid passing it to BasePlot which might pass it down to artists
        kwargs.pop('ax', None)

        super().__init__(*final_args, **layout_kwargs, **fig_params, **kwargs)

        # Explicitly apply labels to all axes if they were provided but not applied
        # This fixes an issue where gwpy/matplotlib might only label the last axis or specific columns
        
        # 1. Y-Label
        candidate_ylabel = force_ylabel
        if candidate_ylabel is None:
             # Scan for any existing label applied by gwpy
             for ax in self.axes:
                 yl = ax.get_ylabel()
                 if yl:
                      candidate_ylabel = yl
                      break
        
        if candidate_ylabel:
            for ax in self.axes:
                if not ax.get_ylabel():
                    ax.set_ylabel(candidate_ylabel)
        
        # 2. X-Label
        candidate_xlabel = force_xlabel
        if candidate_xlabel is None:
             for ax in self.axes:
                 xl = ax.get_xlabel()
                 if xl:
                      candidate_xlabel = xl
                      break

        if candidate_xlabel:
             for ax in self.axes:
                 if not ax.get_xlabel():
                      ax.set_xlabel(candidate_xlabel)

        if use_cl:
             try:
                 self.set_constrained_layout(True)
             except (TypeError, ValueError, AttributeError):
                 pass
        if use_tl:
             try:
                 self.tight_layout()
             except (TypeError, ValueError, AttributeError):
                 # Sometimes tight_layout fails with constrained_layout
                 pass

        # 6. Post-Plotting
        if use_overlay and matrix_args:
             axes = self.axes
             ref_matrix = matrix_args[0]
             r_keys = list(ref_matrix.row_keys())
             c_keys = list(ref_matrix.col_keys())
             ncol = len(c_keys)
             extra_matrix_args = matrix_args[1:]

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

                       for m in extra_matrix_args:
                            try:
                                 val = m[r, c]
                                 _plot_on_ax(ax, val)
                            except (TypeError, ValueError, AttributeError):
                                 pass
                       
                       if (subplots_orig == 'row' and j == 0) or \
                          (subplots_orig == 'col' and i == 0) or \
                          (subplots_orig not in ('row', 'col')):
                             for other in expanded_args:
                                  _plot_on_ax(ax, other)

             row_keys = list(ref_matrix.row_keys())
             col_keys = list(ref_matrix.col_keys())

             def _get_label(md_dict, key):
                 entry = md_dict.get(key)
                 name = getattr(entry, 'name', None) if entry else None
                 if name:
                     return str(name)
                 return str(key)

             current_geom = layout_kwargs.get('geometry')
             if current_geom:
                 nrow_g, ncol_g = current_geom
                 # Label rows (left column only)
                 for i, rk in enumerate(row_keys):
                     idx = i * ncol
                     if idx < len(axes):
                         name_label = _get_label(ref_matrix.rows, rk)
                         current_yl = axes[idx].get_ylabel()
                         if current_yl:
                             # Prepend row name to current ylabel (unit derived)
                             axes[idx].set_ylabel(f"{name_label}\n{current_yl}")
                         else:
                             axes[idx].set_ylabel(name_label)

                 # Label columns (top row only)
                 for j, ck in enumerate(col_keys):
                     idx = j
                     if idx < len(axes):
                         label = _get_label(ref_matrix.cols, ck)
                         axes[idx].set_title(label)
             
             if layout_kwargs.get('legend', True):
                  for ax in axes:
                      handles, labels = ax.get_legend_handles_labels()
                      if labels:
                          ax.legend()

        if is_spectrogram:
             for ax in self.axes:
                  from matplotlib.collections import QuadMesh
                  from matplotlib.image import AxesImage
                  mappable = None
                  for child in ax.get_children():
                       if isinstance(child, (QuadMesh, AxesImage)):
                            mappable = child
                            break
                  if mappable:
                       try:
                            self.colorbar(mappable, ax=ax, label=det_clabel)
                       except (TypeError, ValueError, AttributeError):
                            pass
        
        # Hide x-labels for non-bottom rows when sharex is True
        current_geom = layout_kwargs.get('geometry')
        if current_geom and layout_kwargs.get('sharex', False):
            nrow_g, ncol_g = current_geom
            for i, ax in enumerate(self.axes):
                row_idx = i // ncol_g
                if row_idx < nrow_g - 1:
                    ax.set_xlabel('')
                    ax.tick_params(labelbottom=False)
        # Final layout polish

    def plot_mmm(self, median, min_s, max_s, ax=None, **kwargs):
        """
        Plot Median, Min, and Max series with a filled area between Min and Max.
        
        Parameters
        ----------
        median : Series
        min_s : Series
        max_s : Series
        ax : Axes, optional
        **kwargs
            Passed to ax.plot for the median line.
        """
        if ax is None:
            ax = self.gca()
        
        # Plot fill between
        from matplotlib import pyplot as plt
        color = kwargs.get('color')
        if color is None:
            # Get next color in cycle if not provided
            line, = ax.plot(median.xindex.value, median.value, alpha=0)
            color = line.get_color()
            line.remove()
        
        alpha = kwargs.pop('alpha_fill', 0.2)
        ax.fill_between(min_s.xindex.value, min_s.value, max_s.value, color=color, alpha=alpha)
        
        # Plot median line
        label = kwargs.pop('label', median.name)
        return ax.plot(median.xindex.value, median.value, color=color, label=label, **kwargs)

def plot_summary(sg_collection, fmin=None, fmax=None, title='', **kwargs):
    """
    Plot a grid of Spectrograms and their percentile summaries side-by-side.
    
    Suitable for ASD, PSD, Coherence, and other spectrograms.
    
    Parameters
    ----------
    sg_collection : SpectrogramList, SpectrogramDict, or SpectrogramMatrix
    fmin, fmax : float, optional
        Frequency range.
    title : str, optional
    **kwargs
        Passed to Plot constructor for global settings.
    """
    from matplotlib import pyplot as plt
    from gwexpy.spectrogram import Spectrogram, SpectrogramList, SpectrogramDict, SpectrogramMatrix
    from gwexpy.plot import Plot
    import numpy as np

    # Normalize collection to a dict-like or list of (name, spectrogram)
    if isinstance(sg_collection, SpectrogramMatrix):
        # We assume 3D (Batch, Time, Freq) for now as typical use case
        if sg_collection.ndim == 3:
            names = list(sg_collection.row_keys())
            if not names:
                names = [f"Channel {i}" for i in range(sg_collection.shape[0])]
            sgs = sg_collection.to_series_1Dlist()
            items = list(zip(names, sgs))
        else:
            # Flatten 4D if needed, but 3D is primary target for "list of spectrograms"
            items = []
            r_keys = sg_collection.row_keys()
            c_keys = sg_collection.col_keys()
            for r in r_keys:
                for c in c_keys:
                    items.append((f"{r}/{c}", sg_collection[r, c]))
    elif isinstance(sg_collection, SpectrogramDict):
        items = list(sg_collection.items())
    elif isinstance(sg_collection, (SpectrogramList, list)):
        items = [(getattr(s, 'name', f"Channel {i}"), s) for i, s in enumerate(sg_collection)]
    else:
        raise TypeError(f"Unsupported collection type: {type(sg_collection)}")

    num_rows = len(items)
    if num_rows == 0:
        return None

    # Determine frequency limits if not provided
    if fmin is None:
        fmin = min(s.frequencies.value[0] for _, s in items)
    if fmax is None:
        fmax = max(s.frequencies.value[-1] for _, s in items)

    # Crop frequencies
    items = [(name, s.crop_frequencies(fmin, fmax)) for name, s in items]

    figsize = kwargs.pop('figsize', (16, num_rows * 3.5))
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [2, 1], "wspace": 0.05})
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    if title:
        fig.suptitle(title)

    # Import gwpy scales for auto-gps
    try:
        from gwpy.plot import Plot as GwpyPlot
        # This import registers the 'auto-gps' scale with matplotlib
    except ImportError:
        pass

    for i, (name, sg) in enumerate(items):
        # 1. Percentile Summary Plot (Right)
        ax_asd = axes[i, 1]
        p50 = sg.percentile(50)
        p10 = sg.percentile(10)
        p90 = sg.percentile(90)
        
        # Plot directly on the axis without creating extra figures
        freqs = p50.frequencies.value
        ax_asd.fill_between(freqs, p10.value, p90.value, alpha=0.3, label='10%-90%')
        ax_asd.plot(freqs, p50.value, label='50%')
        
        ax_asd.set_xscale('log')
        ax_asd.set_yscale('log')
        
        unit_str = sg.unit.to_string('latex_inline').replace('Hz^{-1/2}', r'/\sqrt{Hz}')
        
        # Auto-detect title based on unit
        unit_lower = str(sg.unit).lower()
        name_lower = str(getattr(sg, 'name', '')).lower()
        if 'coherence' in name_lower or sg.unit.is_equivalent(''):
            summary_title = 'Coherence'
        elif 'hz^{-1/2}' in unit_str.lower() or 'hz^-1/2' in unit_lower:
            summary_title = f'Amplitude Spectral Density [{unit_str}]'
        elif 'hz^{-1}' in unit_str.lower() or '/hz' in unit_lower:
            summary_title = f'Power Spectral Density [{unit_str}]'
        else:
            summary_title = f'Percentile Summary [{unit_str}]'
        ax_asd.set_title(summary_title)
        ax_asd.set_xlim(fmin, fmax)
        ax_asd.legend(loc='upper right', fontsize=8)
        
        # Only show x-label on bottom row
        if i == num_rows - 1:
            ax_asd.set_xlabel('Frequency [Hz]')
        else:
            ax_asd.tick_params(labelbottom=False)
        
        clim = ax_asd.get_ylim()
        
        # 2. Spectrogram Plot (Left)
        ax_sg = axes[i, 0]
        # Ensure log norm for spectrogram
        from matplotlib.colors import LogNorm
        vmin = clim[0] if clim[0] > 0 else 1e-25
        vmax = clim[1] if clim[1] > 0 else 1e-15
        
        # Plot spectrogram directly using pcolormesh
        times = sg.times.value
        freqs = sg.frequencies.value
        mesh = ax_sg.pcolormesh(times, freqs, sg.value.T, norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
        ax_sg.set_title(name)
        ax_sg.set_ylim(fmin, fmax)
        ax_sg.set_ylabel('Frequency [Hz]')
        
        # Apply auto-gps scale if available
        try:
            ax_sg.set_xscale('auto-gps')
        except ValueError:
            pass  # Fall back to default scale
        
        # Only show x-label on bottom row - clear any auto-generated labels on non-bottom rows
        if i == num_rows - 1:
            pass  # Let auto-gps handle the xlabel for bottom row
        else:
            ax_sg.set_xlabel('')
            ax_sg.tick_params(labelbottom=False)
        
        # Add colorbar
        fig.colorbar(mesh, ax=ax_sg, label=sg.unit.to_string('latex_inline'))
        
    fig.tight_layout()

    return fig, axes

