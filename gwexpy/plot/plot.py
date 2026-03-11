from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

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
        from gwexpy.interop._registry import ConverterRegistry
        from gwexpy.plot import defaults
        from gwexpy.plot._init_helpers import (
            _adaptive_decimate_args,
            _add_spectrogram_colorbars,
            _apply_individual_axis_labels,
            _apply_layout_polish,
            _apply_list_labels,
            _apply_xlabel,
            _apply_ylabel,
            _determine_scales_and_labels,
            _expand_args,
            _extract_layout_and_fig_params,
            _filter_monitor_args,
            _flatten_scan,
            _force_scales,
            _manage_sharex_labels,
            _post_plot_overlay,
        )

        # Retrieve concrete types from registry (avoids circular imports)
        SeriesMatrix = ConverterRegistry.get_constructor("SeriesMatrix")
        SpectrogramMatrix = ConverterRegistry.get_constructor("SpectrogramMatrix")
        FrequencySeriesList = ConverterRegistry.get_constructor("FrequencySeriesList")
        FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
        SpectrogramList = ConverterRegistry.get_constructor("SpectrogramList")
        SpectrogramDict = ConverterRegistry.get_constructor("SpectrogramDict")
        Spectrogram = ConverterRegistry.get_constructor("Spectrogram")

        # 0. Handle monitor filtering
        monitor = kwargs.get("monitor")
        if monitor is not None:
            args = _filter_monitor_args(args, monitor, SeriesMatrix, SpectrogramMatrix)

        # 1. Determine Layout Mode
        if "subplots" in kwargs and "separate" not in kwargs:
            kwargs["separate"] = kwargs["subplots"]
        kwargs.pop("subplots", None)
        separate = kwargs.get("separate")
        geometry = kwargs.get("geometry")
        separate, geometry = defaults.determine_geometry_and_separate(
            list(args), separate=separate, geometry=geometry
        )
        if monitor is not None and separate is None:
            separate = True

        # 2. Expand Arguments
        final_args: list = []
        _expand_args(
            args,
            separate,
            final_args,
            SeriesMatrix=SeriesMatrix,
            SpectrogramMatrix=SpectrogramMatrix,
            FrequencySeriesList=FrequencySeriesList,
            FrequencySeriesDict=FrequencySeriesDict,
            SpectrogramList=SpectrogramList,
            SpectrogramDict=SpectrogramDict,
        )

        # 2.5 Adaptive Decimation
        decimate_threshold = kwargs.pop("decimate_threshold", 50000)
        decimate_points = kwargs.pop("decimate_points", 10000)
        final_args = _adaptive_decimate_args(
            final_args, decimate_threshold, decimate_points
        )

        # 3. Determine Scales and Labels
        scan_data = _flatten_scan(final_args)
        det_clabel = _determine_scales_and_labels(
            scan_data, kwargs, defaults, Spectrogram, SpectrogramMatrix
        )

        # 4. Definitions for Post-Init Logic
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

        expanded_args: list = []
        for arg in args:
            if not isinstance(arg, (SeriesMatrix, SpectrogramMatrix)):
                if isinstance(arg, (list, tuple)):
                    expanded_args.extend(cast(Iterable[Any], arg))
                elif isinstance(arg, dict):
                    expanded_args.extend(cast(Iterable[Any], cast(Any, arg).values()))
                elif isinstance(arg, (FrequencySeriesList, SpectrogramList)):
                    expanded_args.extend(cast(Iterable[Any], arg))
                elif isinstance(arg, (FrequencySeriesDict, SpectrogramDict)):
                    expanded_args.extend(cast(Iterable[Any], cast(Any, arg).values()))
                else:
                    expanded_args.append(arg)

        # 5. Extract params and call super().__init__
        layout_kwargs, fig_params, use_cl, use_tl, labels_list = (
            _extract_layout_and_fig_params(
                kwargs, separate, geometry, final_args, defaults
            )
        )
        force_ylabel = layout_kwargs.get("ylabel")
        force_xlabel = layout_kwargs.get("xlabel")

        super().__init__(*final_args, **layout_kwargs, **fig_params, **kwargs)

        # 6. Post-Init: Labels
        if labels_list:
            _apply_list_labels(self, labels_list, layout_kwargs)

        _apply_ylabel(self, force_ylabel, layout_kwargs)
        _apply_individual_axis_labels(self, final_args, force_ylabel, defaults)
        candidate_xlabel = _apply_xlabel(self, force_xlabel, layout_kwargs)

        # 7. Layout polish and force scales
        _apply_layout_polish(self, use_cl, use_tl)
        _force_scales(self, layout_kwargs)

        # 8. Post-Plotting overlay
        _post_plot_overlay(
            self,
            use_overlay,
            matrix_args,
            subplots_orig,
            expanded_args,
            layout_kwargs,
            SeriesMatrix,
            SpectrogramMatrix,
        )

        # 9. Spectrogram colorbars
        _add_spectrogram_colorbars(self, is_spectrogram, det_clabel)

        # 10. Sharex label management
        _manage_sharex_labels(self, layout_kwargs, candidate_xlabel)

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

    def show(
        self, warn: bool = True, close: bool = True, block: bool | None = None
    ) -> None:
        """Show the figure.

        Parameters
        ----------
        warn : bool, optional
            Unused, kept for API compatibility. Default is True.
        close : bool, optional
            If True (default), close the figure after showing to free
            resources and prevent double display in Jupyter.
            Set to False if you need to call savefig() after show().
        block : bool or None, optional
            Whether to block execution until the figure window is closed.
            If None (default), uses matplotlib's default behavior.
            Set to False to continue execution immediately after showing,
            which allows savefig() to work in scripted workflows.

        Examples
        --------
        >>> plot = Plot(data)
        >>> plot.show(close=False, block=False)  # Non-blocking, keep figure
        >>> plot.savefig("output.png")  # Works because figure is still open

        Notes
        -----
        In Jupyter notebooks, setting ``close=True`` prevents the figure
        from being displayed twice (once by show() and once by the
        notebook's automatic display).

        In script mode, ``plt.show()`` blocks by default until the window
        is closed. Use ``block=False`` if you need to save after showing.
        """
        import matplotlib.pyplot as plt

        plt.show(block=block)
        if close:
            plt.close(self)

    def _repr_png_(self) -> bytes | None:
        """Return PNG representation for Jupyter display.

        This ensures the figure can be displayed in Jupyter notebooks
        when _repr_html_ is disabled to prevent double plotting.

        Returns
        -------
        bytes or None
            PNG image data, or None if the figure has been closed.
        """
        from io import BytesIO

        try:
            buf = BytesIO()
            self.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            return buf.read()
        except (ValueError, AttributeError):
            # Figure may have been closed
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
    from matplotlib import pyplot as plt

    from gwexpy.interop._registry import ConverterRegistry

    SpectrogramMatrix = cast(
        Any, ConverterRegistry.get_constructor("SpectrogramMatrix")
    )
    SpectrogramDict = cast(Any, ConverterRegistry.get_constructor("SpectrogramDict"))
    SpectrogramList = cast(Any, ConverterRegistry.get_constructor("SpectrogramList"))

    # Normalize collection to a dict-like or list of (name, spectrogram)
    if isinstance(sg_collection, SpectrogramMatrix):
        sg_obj = cast(Any, sg_collection)
        # We assume 3D (Batch, Time, Freq) for now as typical use case
        if sg_obj.ndim == 3:
            names = list(sg_obj.row_keys())
            if not names:
                names = [f"Channel {i}" for i in range(sg_obj.shape[0])]
            sgs = sg_obj.to_series_1Dlist()
            items = list(zip(names, sgs))
        else:
            # Flatten 4D if needed, but 3D is primary target for "list of spectrograms"
            items = []
            r_keys = sg_obj.row_keys()
            c_keys = sg_obj.col_keys()
            for r in r_keys:
                for c in c_keys:
                    items.append((f"{r}/{c}", sg_obj[r, c]))
    elif isinstance(sg_collection, SpectrogramDict):
        items = list(cast(Any, sg_collection).items())
    elif isinstance(sg_collection, (SpectrogramList, list)):
        sg_list = cast(list[Any], sg_collection)
        items = [
            (getattr(s, "name", f"Channel {i}"), s) for i, s in enumerate(sg_list)
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
