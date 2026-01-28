from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from gwpy.plot import Plot as BasePlot

if TYPE_CHECKING:
    pass

__all__ = ["Plot"]


def plot_mmm(median, min_s, max_s, ax=None, **kwargs):
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
        from matplotlib import pyplot as plt

        ax = plt.gca()

    # Plot fill between
    color = kwargs.pop("color", None)
    if color is None:
        # Get next color in cycle if not provided
        (line,) = ax.plot(median.xindex.value, median.value, alpha=0)
        color = line.get_color()
        line.remove()

    alpha = kwargs.pop("alpha_fill", 0.2)
    label_fill = kwargs.pop("label_fill", None)
    ax.fill_between(
        min_s.xindex.value,
        min_s.value,
        max_s.value,
        color=color,
        alpha=alpha,
        label=label_fill,
    )

    # Plot median line
    label = kwargs.pop("label", median.name)
    return ax.plot(
        median.xindex.value, median.value, color=color, label=label, **kwargs
    )


class Plot(BasePlot):
    """
    An extension of :class:`gwpy.plot.Plot` that automatically handles
    :class:`gwexpy.types.SeriesMatrix` arguments by expanding them into
    individual :class:`gwpy.types.Series` objects, while preserving
    matrix layout and metadata where possible.
    """

    # Suppress _repr_html_ to prevent double plotting (repr + backend)
    _repr_html_ = None

    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    def __init__(self, *args, **kwargs):
        # Local import to avoid circular dependency
        from gwexpy.frequencyseries import FrequencySeriesDict, FrequencySeriesList
        from gwexpy.plot import defaults
        from gwexpy.spectrogram import (
            Spectrogram,
            SpectrogramDict,
            SpectrogramList,
            SpectrogramMatrix,
        )
        from gwexpy.types.seriesmatrix import SeriesMatrix

        # 0. Handle monitor filtering
        monitor = kwargs.get("monitor")
        if monitor is not None:
            new_args = []
            for arg in args:
                if isinstance(arg, (SeriesMatrix, SpectrogramMatrix)):
                    # Apply filtering
                    try:
                        # For SpectrogramMatrix 4D, map a flat index to (row, col)
                        if type(arg).__name__ == "SpectrogramMatrix" and isinstance(
                            monitor, (int, np.integer)
                        ):
                            if arg.ndim == 4:
                                nrow, ncol = arg.shape[:2]
                                row_idx = monitor // ncol
                                col_idx = monitor % ncol
                                new_args.append(arg[row_idx, col_idx])
                                continue
                        # Use the object's own indexing (supporting labels, lists, etc. if implemented)
                        filtered_arg = arg[monitor]
                        new_args.append(filtered_arg)
                    except (IndexError, KeyError, TypeError, ValueError):
                        # Fallback: if single integer monitor on SpectrogramMatrix fails (e.g. 4D vs 3D confusion)
                        if type(arg).__name__ == "SpectrogramMatrix" and isinstance(
                            monitor, (int, np.integer)
                        ):
                            if arg.ndim == 4:
                                nrow, ncol = arg.shape[:2]
                                row_idx = monitor // ncol
                                col_idx = monitor % ncol
                                new_args.append(arg[row_idx, col_idx])
                            else:
                                new_args.append(arg[monitor])
                        else:
                            new_args.append(arg)
                else:
                    new_args.append(arg)
            args = tuple(new_args)

        # 1. Determine Layout Mode (Separate / Geometry)
        if "subplots" in kwargs and "separate" not in kwargs:
            kwargs["separate"] = kwargs["subplots"]
        kwargs.pop("subplots", None)
        separate = kwargs.get("separate")
        geometry = kwargs.get("geometry")

        # Use defaults logic to infer separate/geometry from inputs
        separate, geometry = defaults.determine_geometry_and_separate(
            list(args), separate=separate, geometry=geometry
        )

        # If monitor is active, we force separation (unless explicitly overridden?)
        if monitor is not None and separate is None:
            separate = True

        # 2. Expand Arguments based on Separate Mode
        final_args = []

        for arg in args:
            is_matrix = isinstance(arg, (SeriesMatrix, SpectrogramMatrix))

            if separate is True:
                # Grid / Separate mode: Flatten everything to individual items
                if is_matrix:
                    final_args.extend(arg.to_series_1Dlist())
                elif isinstance(arg, (list, tuple)):
                    final_args.extend(arg)
                elif isinstance(arg, dict):
                    final_args.extend(arg.values())
                elif isinstance(arg, (FrequencySeriesList, SpectrogramList)):
                    final_args.extend(arg)
                elif isinstance(arg, (FrequencySeriesDict, SpectrogramDict)):
                    final_args.extend(arg.values())
                else:
                    final_args.append(arg)

            elif separate == "row" and is_matrix:
                # Row mode: Each row is a group
                # Assuming SeriesMatrix has row_keys iterator or similar
                r_keys = arg.row_keys()
                c_keys = arg.col_keys()
                for r in r_keys:
                    row_items = []
                    for c in c_keys:
                        try:
                            val = arg[r, c]
                            # Ensure name for legend
                            if not getattr(val, "name", None):
                                val.name = f"{r} / {c}"
                            row_items.append(val)
                        except (TypeError, ValueError, IndexError):
                            pass
                    final_args.append(row_items)

            elif separate == "col" and is_matrix:
                # Col mode: Each col is a group
                r_keys = arg.row_keys()
                c_keys = arg.col_keys()
                for c in c_keys:
                    col_items = []
                    for r in r_keys:
                        try:
                            val = arg[r, c]
                            if not getattr(val, "name", None):
                                val.name = f"{r} / {c}"
                            col_items.append(val)
                        except (TypeError, ValueError, IndexError):
                            pass
                    final_args.append(col_items)

            else:
                # separate=False (Overlay) or None (Default behavior)
                # For Matrix: Flatten into ONE group
                if is_matrix:
                    final_args.append(arg.to_series_1Dlist())
                # For List/Dict: Keep as ONE group (Preserve structure for gwpy)
                elif isinstance(arg, dict):
                    final_args.append(list(arg.values()))
                elif isinstance(arg, (FrequencySeriesDict, SpectrogramDict)):
                    final_args.append(list(arg.values()))
                elif isinstance(arg, (FrequencySeriesList, SpectrogramList)):
                    final_args.append(list(arg))
                else:
                    final_args.append(arg)

        # 2.5 TimeSeries Optimization (Adaptive Decimation)
        from gwexpy.plot.utils import adaptive_decimate
        from gwexpy.timeseries import TimeSeries

        decimate_threshold = kwargs.pop("decimate_threshold", 50000)
        decimate_points = kwargs.pop("decimate_points", 10000)

        def _optimize_if_needed(val):
            if isinstance(val, TimeSeries) and len(val) > decimate_threshold:
                return adaptive_decimate(val, target_points=decimate_points)
            if isinstance(val, list):
                return [_optimize_if_needed(v) for v in val]
            if isinstance(val, tuple):
                return tuple(_optimize_if_needed(v) for v in val)
            return val

        final_args = [_optimize_if_needed(a) for a in final_args]

        # 3. Determine Scales and Labels (using flattened/inspected data)
        # We need a flat list of all data to scan for scales
        def _flatten_scan(ax_args):
            flat = []
            for a in ax_args:
                if isinstance(a, (list, tuple)):
                    flat.extend(_flatten_scan(a))
                else:
                    flat.append(a)
            return flat

        scan_data = _flatten_scan(final_args)

        # Ensure spectrograms use a 2D plotting method instead of line plotting.
        if "method" not in kwargs:
            has_spectrogram = any(
                isinstance(a, (Spectrogram, SpectrogramMatrix)) for a in scan_data
            )
            if has_spectrogram:
                kwargs["method"] = "pcolormesh"

        if "xscale" not in kwargs:
            det_xscale = defaults.determine_xscale(scan_data)
            if det_xscale is not None:
                kwargs["xscale"] = det_xscale

        if "yscale" not in kwargs:
            det_yscale = defaults.determine_yscale(scan_data)
            if det_yscale is not None:
                kwargs["yscale"] = det_yscale

        if "xlabel" not in kwargs:
            det_xlabel = defaults.determine_xlabel(scan_data)
            if det_xlabel is not None:
                kwargs["xlabel"] = det_xlabel

        if "ylabel" not in kwargs:
            # Only determine global ylabel if units are consistent across all data
            # Check consistency
            units_set: set[str] = set()
            for x in scan_data:
                u_val = getattr(x, "unit", None)
                # Treat None and Dimensionless as distinct? gwpy might handle them.
                # For simplicity, store string representation unless None
                if u_val is not None and hasattr(u_val, "to_string"):
                    units_set.add(u_val.to_string())
                else:
                    units_set.add(str(u_val))

            if len(units_set) <= 1:
                det_ylabel = defaults.determine_ylabel(scan_data)
                if det_ylabel is not None:
                    kwargs["ylabel"] = det_ylabel
            # Else: do not set global ylabel, let individual axes handle it (or handle in post-processing)
            else:
                # Explicitly ensure NO global label is enforced so gwpy doesn't pick just one?
                # If we pass nothing, gwpy might autopick from the first one?
                # We might need to handle this in post-processing.
                pass

        if "norm" not in kwargs:
            det_norm = defaults.determine_norm(scan_data)
            if det_norm is not None:
                kwargs["norm"] = det_norm

        if "clabel" not in kwargs:
            det_clabel = defaults.determine_clabel(scan_data)
            # clabel is handled separately later for colorbars or passed?
            # If we want to pass it, we should check if valid.
            # But clabel is not a standard Plot kwarg usually.
            pass

        if "ylim" not in kwargs:
            det_ylim = defaults.determine_ylim(scan_data, yscale=kwargs.get("yscale"))
            if det_ylim is not None:
                kwargs["ylim"] = det_ylim

        # 4. Geometry Check
        if geometry is not None:
            kwargs["geometry"] = geometry

        if separate is not None:
            # Pass separate to base plot
            kwargs["separate"] = True if isinstance(separate, str) else separate

        # Figsize Logic
        if "figsize" not in kwargs and geometry is not None:
            kwargs["figsize"] = defaults.calculate_default_figsize(
                geometry, geometry[0], geometry[1]
            )
        elif "figsize" not in kwargs and separate is True:
            # Infer geometry from final_args length
            n_axes = len(final_args)
            kwargs["figsize"] = defaults.calculate_default_figsize(None, n_axes, 1)

        # Definitions for Legacy/Post-Init Logic
        use_overlay = any(
            isinstance(fa, (list, tuple)) and len(fa) > 1 for fa in final_args
        )
        matrix_args = [
            a for a in args if isinstance(a, (SeriesMatrix, SpectrogramMatrix))
        ]
        is_spectrogram = any(
            isinstance(a, (Spectrogram, SpectrogramMatrix)) for a in scan_data
        )
        subplots_orig = separate

        expanded_args: list[Any] = []
        for arg in args:
            if not isinstance(arg, (SeriesMatrix, SpectrogramMatrix)):
                if isinstance(arg, (list, tuple)):
                    expanded_args.extend(arg)
                elif isinstance(arg, dict):
                    expanded_args.extend(arg.values())
                elif isinstance(arg, (FrequencySeriesList, SpectrogramList)):
                    expanded_args.extend(arg)
                elif isinstance(arg, (FrequencySeriesDict, SpectrogramDict)):
                    expanded_args.extend(arg.values())
                else:
                    expanded_args.append(arg)

        # 5. Super Init
        layout_kwargs = {}
        for k in [
            "separate",
            "geometry",
            "sharex",
            "sharey",
            "xscale",
            "yscale",
            "norm",
            "xlim",
            "ylim",
            "xlabel",
            "ylabel",
            "title",
            "legend",
            "method",
        ]:
            if k in kwargs:
                layout_kwargs[k] = kwargs.pop(k)

        # Handle logx/logy aliases
        if kwargs.pop("logx", False):
            layout_kwargs["xscale"] = "log"
        if kwargs.pop("logy", False):
            layout_kwargs["yscale"] = "log"

        fig_params = {}
        for k in [
            "figsize",
            "dpi",
            "facecolor",
            "edgecolor",
            "linewidth",
            "frameon",
            "subplotpars",
        ]:
            if k in kwargs:
                fig_params[k] = kwargs.pop(k)

        use_cl = kwargs.pop("constrained_layout", False)
        use_tl = kwargs.pop("tight_layout", False)

        # Store labels to ensure application
        force_ylabel = layout_kwargs.get("ylabel")
        force_xlabel = layout_kwargs.get("xlabel")

        # Pop redundant or gwexpy-specific keywords that shouldn't go to BasePlot/Matplotlib
        kwargs.pop("ax", None)
        kwargs.pop("monitor", None)
        kwargs.pop("show", None)

        # Handle list of labels (which causes ValueError in matplotlib if passed to plot())
        labels_list = None
        if "label" in kwargs and isinstance(kwargs["label"], (list, tuple)):
            labels_list = kwargs.pop("label")
        elif "labels" in kwargs:
            labels_list = kwargs.pop("labels")

        super().__init__(*final_args, **layout_kwargs, **fig_params, **kwargs)

        # Apply list labels if provided
        if labels_list:
            # Iterate over all lines in all axes and assign labels sequentially
            # This assumes the plotting order matches the label list order
            line_idx = 0
            for ax in self.axes:
                for line in ax.get_lines():
                    if line_idx < len(labels_list):
                        line.set_label(labels_list[line_idx])
                        line_idx += 1
                # If we also have collections (like scatter/pcolormesh), should we label them?
                # Plot usually makes lines for 1D. BifrequencyMap diagonal returns FrequencySeries -> Lines.

            # Re-generate legend if labels were updated
            if layout_kwargs.get("legend", True):
                for ax in self.axes:
                    # Check if ax has lines with labels
                    handles, lbls = ax.get_legend_handles_labels()
                    if lbls:
                        ax.legend()

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
            # Detect axes in the first column vs others
            first_col_axes = []
            other_axes = []
            for ax in self.axes:
                try:
                    is_left = ax.get_subplotspec().is_first_col()
                except (AttributeError, ValueError, IndexError):
                    is_left = True
                if is_left:
                    first_col_axes.append(ax)
                else:
                    other_axes.append(ax)

            # If many rows in the first column share the same label, only label the middle one
            if len(first_col_axes) > 2:
                all_same = True
                for ax in first_col_axes:
                    yl = ax.get_ylabel()
                    if yl and yl != candidate_ylabel:
                        all_same = False
                        break

                if all_same:
                    mid_idx = len(first_col_axes) // 2
                    for i, ax in enumerate(first_col_axes):
                        if i == mid_idx:
                            if not ax.get_ylabel():
                                ax.set_ylabel(candidate_ylabel)
                        else:
                            ax.set_ylabel("")
                else:
                    for ax in first_col_axes:
                        if not ax.get_ylabel():
                            ax.set_ylabel(candidate_ylabel)
            else:
                for ax in first_col_axes:
                    if not ax.get_ylabel():
                        ax.set_ylabel(candidate_ylabel)

            # Clear labels in other columns if sharing Y
            if layout_kwargs.get("sharey", False):
                for ax in other_axes:
                    if ax.get_ylabel() == candidate_ylabel:
                        ax.set_ylabel("")

        # 6. Auto-Label Individual Axes if needed
        # If no global ylabel was enforced (e.g. diverse units), try to label each axis individually
        if force_ylabel is None:
            # Heuristic: map final_args to axes one-to-one
            # This works well for separate=True or standard gwpy behavior
            if len(self.axes) == len(final_args):
                for ax, data_item in zip(self.axes, final_args):
                    if not ax.get_ylabel():
                        # data_item might be a list (if grouped) or single obj
                        d_list = (
                            data_item
                            if isinstance(data_item, (list, tuple))
                            else [data_item]
                        )
                        lbl = defaults.determine_ylabel(d_list)
                        if lbl:
                            ax.set_ylabel(lbl)

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
                # Only apply if xlabel is empty AND NOT in sharex mode
                # In sharex mode, gwpy handles the labels correctly (usually non-bottom are empty)
                if not ax.get_xlabel() and not layout_kwargs.get("sharex", False):
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

        # Force scales if they were determined but not applied/overridden by super class
        if layout_kwargs.get("xscale"):
            for ax in self.axes:
                try:
                    ax.set_xscale(layout_kwargs["xscale"])
                    ax.autoscale_view()
                except (ValueError, TypeError):
                    pass
        if layout_kwargs.get("yscale"):
            for ax in self.axes:
                try:
                    ax.set_yscale(layout_kwargs["yscale"])
                    ax.autoscale_view()
                except (ValueError, TypeError):
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
                if hasattr(other, "times"):
                    ax.plot(other, label=getattr(other, "name", None))
                elif hasattr(other, "frequencies"):
                    ax.plot(other, label=getattr(other, "name", None))
                else:
                    ax.plot(other)

            for i, r in enumerate(r_keys):
                for j, c in enumerate(c_keys):
                    if subplots_orig == "row":
                        ax_idx = i
                    elif subplots_orig == "col":
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

                    if (
                        (subplots_orig == "row" and j == 0)
                        or (subplots_orig == "col" and i == 0)
                        or (subplots_orig not in ("row", "col"))
                    ):
                        for other in expanded_args:
                            _plot_on_ax(ax, other)

            row_keys = list(ref_matrix.row_keys())
            col_keys = list(ref_matrix.col_keys())

            def _get_label(md_dict, key):
                entry = md_dict.get(key)
                name = getattr(entry, "name", None) if entry else None
                if name:
                    return str(name)
                return str(key)

            current_geom = layout_kwargs.get("geometry")
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

            if layout_kwargs.get("legend", True):
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
        current_geom = layout_kwargs.get("geometry")
        if current_geom and layout_kwargs.get("sharex", False):
            # Recalculate based on current axes length if needed, but geometry is more reliable
            nrow_g, ncol_g = current_geom
            for i, ax in enumerate(self.axes):
                row_idx = i // ncol_g
                if row_idx < nrow_g - 1:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)

        # Ensure x-axis label for bottom row is applied if sharex=True but label is missing
        if layout_kwargs.get("sharex", False) and candidate_xlabel:
            nrow_g, ncol_g = current_geom if current_geom else (len(self.axes), 1)
            for i, ax in enumerate(self.axes):
                row_idx = i // ncol_g
                if row_idx == nrow_g - 1:
                    if not ax.get_xlabel():
                        ax.set_xlabel(candidate_xlabel)
        # Final layout polish

    def plot_mmm(self, median, min_s, max_s, ax=None, **kwargs):
        """Plot median line with a min/max envelope.

        This is a convenience wrapper around :func:`gwexpy.plot.plot_mmm`.

        Parameters
        ----------
        median : gwpy.types.Series
            Median series to plot as a line.
        min_s : gwpy.types.Series
            Minimum series defining the lower envelope.
        max_s : gwpy.types.Series
            Maximum series defining the upper envelope.
        ax : matplotlib.axes.Axes, optional
            Target axes. If omitted, uses the current axes of this Plot.
        **kwargs
            Passed to ``ax.plot`` for the median line. Additional keys supported:

            - ``alpha_fill``: opacity for the filled envelope
            - ``label_fill``: label for the filled envelope

        Returns
        -------
        list
            The list of Line2D objects returned by ``ax.plot``.
        """
        if ax is None:
            ax = self.gca()
        return plot_mmm(median, min_s, max_s, ax=ax, **kwargs)

    def show(self, warn=True):
        """Show the figure and close it to prevent double display."""
        import matplotlib.pyplot as plt

        plt.show()
        plt.close(self)
        return None


def plot_summary(sg_collection, fmin=None, fmax=None, title="", **kwargs):
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
    import numpy as np
    from matplotlib import pyplot as plt

    from gwexpy.spectrogram import SpectrogramDict, SpectrogramList, SpectrogramMatrix

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
        items = [
            (getattr(s, "name", f"Channel {i}"), s) for i, s in enumerate(sg_collection)
        ]
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

    figsize = kwargs.pop("figsize", (16, num_rows * 3.5))
    fig, axes = plt.subplots(
        num_rows,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.05},
    )
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    if title:
        fig.suptitle(title)

    # Import gwpy scales for auto-gps
    try:
        from gwpy.plot import Plot as GwpyPlot  # noqa: F401 - registers auto-gps scale
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
        ax_asd.fill_between(
            p10.frequencies.value, p10.value, p90.value, alpha=0.3, label="10%-90%"
        )
        ax_asd.plot(p50.frequencies.value, p50.value, label="50%")

        ax_asd.set_xscale("log")
        ax_asd.set_yscale("log")

        unit_str = sg.unit.to_string("latex_inline").replace("Hz^{-1/2}", r"/\sqrt{Hz}")

        # Auto-detect title based on unit
        unit_lower = str(sg.unit).lower()
        name_lower = str(getattr(sg, "name", "")).lower()
        if "coherence" in name_lower or sg.unit.is_equivalent(""):
            summary_title = "Coherence"
        elif "hz^{-1/2}" in unit_str.lower() or "hz^-1/2" in unit_lower:
            summary_title = f"Amplitude Spectral Density [{unit_str}]"
        elif "hz^{-1}" in unit_str.lower() or "/hz" in unit_lower:
            summary_title = f"Power Spectral Density [{unit_str}]"
        else:
            summary_title = f"Percentile Summary [{unit_str}]"
        ax_asd.set_title(summary_title)
        # Avoid warning by ensuring xlim > 0 for log axis
        pos_freqs = p50.frequencies.value[p50.frequencies.value > 0]
        fmin_plot = (
            max(fmin, pos_freqs[0])
            if len(pos_freqs) > 0
            else (fmin if fmin > 0 else 1e-3)
        )
        ax_asd.set_xlim(fmin_plot, fmax)
        ax_asd.legend(loc="upper right", fontsize=8)

        # Only show x-label on bottom row
        if i == num_rows - 1:
            ax_asd.set_xlabel("Frequency [Hz]")
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
        mesh = ax_sg.pcolormesh(
            times, freqs, sg.value.T, norm=LogNorm(vmin=vmin, vmax=vmax), shading="auto"
        )
        ax_sg.set_title(name)
        ax_sg.set_ylim(fmin, fmax)
        ax_sg.set_ylabel("Frequency [Hz]")

        # Apply auto-gps scale if available
        try:
            ax_sg.set_xscale("auto-gps")
        except ValueError:
            pass  # Fall back to default scale

        # Only show x-label on bottom row - clear any auto-generated labels on non-bottom rows
        if i == num_rows - 1:
            pass  # Let auto-gps handle the xlabel for bottom row
        else:
            ax_sg.set_xlabel("")
            ax_sg.tick_params(labelbottom=False)

        # Add colorbar
        fig.colorbar(mesh, ax=ax_sg, label=sg.unit.to_string("latex_inline"))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
        )
        fig.tight_layout()

    return fig, axes
