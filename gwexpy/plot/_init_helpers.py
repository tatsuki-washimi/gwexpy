"""Helper functions extracted from Plot.__init__ to reduce complexity."""

from __future__ import annotations

from typing import Any, cast


def _filter_monitor_args(
    args: tuple[Any, ...], monitor: Any, SeriesMatrix: type, SpectrogramMatrix: type
) -> tuple[Any, ...]:
    """Filter args by monitor index for SeriesMatrix/SpectrogramMatrix."""
    import numpy as np

    new_args = []
    for arg in args:
        if isinstance(arg, (SeriesMatrix, SpectrogramMatrix)):
            matrix_arg = cast(Any, arg)
            try:
                if type(matrix_arg).__name__ == "SpectrogramMatrix" and isinstance(
                    monitor, (int, np.integer)
                ):
                    if matrix_arg.ndim == 4:
                        nrow, ncol = matrix_arg.shape[:2]
                        row_idx = monitor // ncol
                        col_idx = monitor % ncol
                        new_args.append(matrix_arg[row_idx, col_idx])
                        continue
                filtered_arg = matrix_arg[monitor]
                new_args.append(filtered_arg)
            except (IndexError, KeyError, TypeError, ValueError):
                if type(matrix_arg).__name__ == "SpectrogramMatrix" and isinstance(
                    monitor, (int, np.integer)
                ):
                    if matrix_arg.ndim == 4:
                        nrow, ncol = matrix_arg.shape[:2]
                        row_idx = monitor // ncol
                        col_idx = monitor % ncol
                        new_args.append(matrix_arg[row_idx, col_idx])
                    else:
                        new_args.append(matrix_arg[monitor])
                else:
                    new_args.append(matrix_arg)
        else:
            new_args.append(arg)
    return tuple(new_args)


def _expand_args(
    args,
    separate,
    final_args_out,
    *,
    SeriesMatrix,
    SpectrogramMatrix,
    FrequencySeriesList,
    FrequencySeriesDict,
    SpectrogramList,
    SpectrogramDict,
):
    """Expand arguments based on separate mode, appending to final_args_out."""
    for arg in args:
        is_matrix = isinstance(arg, (SeriesMatrix, SpectrogramMatrix))

        if separate is True:
            if is_matrix:
                final_args_out.extend(arg.to_series_1Dlist())
            elif isinstance(arg, (list, tuple)):
                final_args_out.extend(arg)
            elif isinstance(arg, dict):
                final_args_out.extend(arg.values())
            elif isinstance(arg, (FrequencySeriesList, SpectrogramList)):
                final_args_out.extend(arg)
            elif isinstance(arg, (FrequencySeriesDict, SpectrogramDict)):
                final_args_out.extend(arg.values())
            else:
                final_args_out.append(arg)

        elif separate == "row" and is_matrix:
            for r in arg.row_keys():
                row_items = []
                for c in arg.col_keys():
                    try:
                        val = arg[r, c]
                        if not getattr(val, "name", None):
                            val.name = f"{r} / {c}"
                        row_items.append(val)
                    except (TypeError, ValueError, IndexError):
                        pass
                final_args_out.append(row_items)

        elif separate == "col" and is_matrix:
            for c in arg.col_keys():
                col_items = []
                for r in arg.row_keys():
                    try:
                        val = arg[r, c]
                        if not getattr(val, "name", None):
                            val.name = f"{r} / {c}"
                        col_items.append(val)
                    except (TypeError, ValueError, IndexError):
                        pass
                final_args_out.append(col_items)

        else:
            if is_matrix:
                final_args_out.append(arg.to_series_1Dlist())
            elif isinstance(arg, dict):
                final_args_out.append(list(arg.values()))
            elif isinstance(arg, (FrequencySeriesDict, SpectrogramDict)):
                final_args_out.append(list(arg.values()))
            elif isinstance(arg, (FrequencySeriesList, SpectrogramList)):
                final_args_out.append(list(arg))
            else:
                final_args_out.append(arg)


def _adaptive_decimate_args(final_args, decimate_threshold, decimate_points):
    """Apply adaptive decimation to TimeSeries that exceed threshold."""
    from gwexpy.plot.utils import adaptive_decimate
    from gwexpy.timeseries import TimeSeries

    def _optimize_if_needed(val):
        if isinstance(val, TimeSeries) and len(val) > decimate_threshold:
            return adaptive_decimate(val, target_points=decimate_points)
        if isinstance(val, list):
            return [_optimize_if_needed(v) for v in val]
        if isinstance(val, tuple):
            return tuple(_optimize_if_needed(v) for v in val)
        return val

    return [_optimize_if_needed(a) for a in final_args]


def _flatten_scan(ax_args):
    """Flatten nested lists/tuples into a flat list for scanning."""
    flat = []
    for a in ax_args:
        if isinstance(a, (list, tuple)):
            flat.extend(_flatten_scan(a))
        else:
            flat.append(a)
    return flat


def _determine_scales_and_labels(
    scan_data, kwargs, defaults, Spectrogram, SpectrogramMatrix
):
    """Determine scales, labels, norms, and limits from scanned data."""
    if "method" not in kwargs:
        has_spectrogram = any(
            isinstance(a, (Spectrogram, SpectrogramMatrix)) for a in scan_data
        )
        if has_spectrogram:
            kwargs["method"] = "pcolormesh"

    for key, determine_fn in [
        ("xscale", defaults.determine_xscale),
        ("yscale", defaults.determine_yscale),
        ("xlabel", defaults.determine_xlabel),
    ]:
        if key not in kwargs:
            det = determine_fn(scan_data)
            if det is not None:
                kwargs[key] = det

    if "ylabel" not in kwargs:
        units_set: set[str] = set()
        for x in scan_data:
            u_val = getattr(x, "unit", None)
            if u_val is not None and hasattr(u_val, "to_string"):
                units_set.add(u_val.to_string())
            else:
                units_set.add(str(u_val))
        if len(units_set) <= 1:
            det_ylabel = defaults.determine_ylabel(scan_data)
            if det_ylabel is not None:
                kwargs["ylabel"] = det_ylabel

    if "norm" not in kwargs:
        det_norm = defaults.determine_norm(scan_data)
        if det_norm is not None:
            kwargs["norm"] = det_norm

    # clabel is handled separately
    det_clabel = None
    if "clabel" not in kwargs:
        det_clabel = defaults.determine_clabel(scan_data)

    if "ylim" not in kwargs:
        det_ylim = defaults.determine_ylim(scan_data, yscale=kwargs.get("yscale"))
        if det_ylim is not None:
            kwargs["ylim"] = det_ylim

    return det_clabel


def _extract_layout_and_fig_params(kwargs, separate, geometry, final_args, defaults):
    """Extract layout_kwargs and fig_params from kwargs, returning them."""
    if geometry is not None:
        kwargs["geometry"] = geometry
    if separate is not None:
        kwargs["separate"] = True if isinstance(separate, str) else separate

    if "figsize" not in kwargs and geometry is not None:
        kwargs["figsize"] = defaults.calculate_default_figsize(
            geometry, geometry[0], geometry[1]
        )
    elif "figsize" not in kwargs and separate is True:
        n_axes = len(final_args)
        kwargs["figsize"] = defaults.calculate_default_figsize(None, n_axes, 1)

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

    kwargs.pop("ax", None)
    kwargs.pop("monitor", None)
    kwargs.pop("show", None)

    labels_list = None
    if "label" in kwargs and isinstance(kwargs["label"], (list, tuple)):
        labels_list = kwargs.pop("label")
    elif "labels" in kwargs:
        labels_list = kwargs.pop("labels")

    return layout_kwargs, fig_params, use_cl, use_tl, labels_list


def _apply_list_labels(fig, labels_list, layout_kwargs):
    """Apply a list of labels to plot lines sequentially."""
    line_idx = 0
    for ax in fig.axes:
        for line in ax.get_lines():
            if line_idx < len(labels_list):
                line.set_label(labels_list[line_idx])
                line_idx += 1
    if layout_kwargs.get("legend", True):
        for ax in fig.axes:
            handles, lbls = ax.get_legend_handles_labels()
            if lbls:
                ax.legend()


def _apply_ylabel(fig, force_ylabel, layout_kwargs):
    """Apply ylabel to axes, handling multi-row layouts."""
    candidate_ylabel = force_ylabel
    if candidate_ylabel is None:
        for ax in fig.axes:
            yl = ax.get_ylabel()
            if yl:
                candidate_ylabel = yl
                break

    if not candidate_ylabel:
        return

    first_col_axes = []
    other_axes = []
    for ax in fig.axes:
        try:
            is_left = ax.get_subplotspec().is_first_col()
        except (AttributeError, ValueError, IndexError):
            is_left = True
        if is_left:
            first_col_axes.append(ax)
        else:
            other_axes.append(ax)

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

    if layout_kwargs.get("sharey", False):
        for ax in other_axes:
            if ax.get_ylabel() == candidate_ylabel:
                ax.set_ylabel("")


def _apply_individual_axis_labels(fig, final_args, force_ylabel, defaults):
    """Apply per-axis ylabel when no global ylabel was set and units differ."""
    if force_ylabel is not None:
        return
    if len(fig.axes) != len(final_args):
        return
    for ax, data_item in zip(fig.axes, final_args):
        if not ax.get_ylabel():
            d_list = data_item if isinstance(data_item, (list, tuple)) else [data_item]
            lbl = defaults.determine_ylabel(d_list)
            if lbl:
                ax.set_ylabel(lbl)


def _apply_xlabel(fig, force_xlabel, layout_kwargs):
    """Apply xlabel to axes."""
    candidate_xlabel = force_xlabel
    if candidate_xlabel is None:
        for ax in fig.axes:
            xl = ax.get_xlabel()
            if xl:
                candidate_xlabel = xl
                break

    if candidate_xlabel:
        for ax in fig.axes:
            if not ax.get_xlabel() and not layout_kwargs.get("sharex", False):
                ax.set_xlabel(candidate_xlabel)

    return candidate_xlabel


def _apply_layout_polish(fig, use_cl, use_tl):
    """Apply constrained_layout or tight_layout."""
    if use_cl:
        try:
            fig.set_constrained_layout(True)
        except (TypeError, ValueError, AttributeError):
            pass
    if use_tl:
        try:
            fig.tight_layout()
        except (TypeError, ValueError, AttributeError):
            pass


def _force_scales(fig, layout_kwargs):
    """Force xscale/yscale on all axes if specified."""
    if layout_kwargs.get("xscale"):
        for ax in fig.axes:
            try:
                ax.set_xscale(layout_kwargs["xscale"])
                ax.autoscale_view()
            except (ValueError, TypeError):
                pass
    if layout_kwargs.get("yscale"):
        for ax in fig.axes:
            try:
                ax.set_yscale(layout_kwargs["yscale"])
                ax.autoscale_view()
            except (ValueError, TypeError):
                pass


def _post_plot_overlay(
    fig,
    use_overlay,
    matrix_args,
    subplots_orig,
    expanded_args,
    layout_kwargs,
    SeriesMatrix,
    SpectrogramMatrix,
):
    """Handle post-plotting overlay for matrix args."""
    if not (use_overlay and matrix_args):
        return

    axes = fig.axes
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

    # Label rows and columns from matrix metadata
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
        for i, rk in enumerate(row_keys):
            idx = i * ncol
            if idx < len(axes):
                name_label = _get_label(ref_matrix.rows, rk)
                current_yl = axes[idx].get_ylabel()
                if current_yl:
                    axes[idx].set_ylabel(f"{name_label}\n{current_yl}")
                else:
                    axes[idx].set_ylabel(name_label)

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


def _add_spectrogram_colorbars(fig, is_spectrogram, det_clabel):
    """Add colorbars for spectrogram plots."""
    if not is_spectrogram:
        return

    from matplotlib.collections import QuadMesh
    from matplotlib.image import AxesImage

    for ax in fig.axes:
        mappable = None
        for child in ax.get_children():
            if isinstance(child, (QuadMesh, AxesImage)):
                mappable = child
                break
        if mappable:
            try:
                fig.colorbar(mappable, ax=ax, label=det_clabel)
            except (TypeError, ValueError, AttributeError):
                pass


def _manage_sharex_labels(fig, layout_kwargs, candidate_xlabel):
    """Hide x-labels for non-bottom rows when sharex is True."""
    current_geom = layout_kwargs.get("geometry")
    if not (current_geom and layout_kwargs.get("sharex", False)):
        return

    nrow_g, ncol_g = current_geom
    for i, ax in enumerate(fig.axes):
        row_idx = i // ncol_g
        if row_idx < nrow_g - 1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    if candidate_xlabel:
        for i, ax in enumerate(fig.axes):
            row_idx = i // ncol_g
            if row_idx == nrow_g - 1:
                if not ax.get_xlabel():
                    ax.set_xlabel(candidate_xlabel)
