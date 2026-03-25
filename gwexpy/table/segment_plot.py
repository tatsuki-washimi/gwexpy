"""segment_plot — drawing functions for SegmentTable.

All public functions accept a :class:`~gwexpy.table.segment_table.SegmentTable`
as their first argument and return a :class:`gwpy.plot.Plot`.
None of the functions call :meth:`matplotlib.figure.Figure.show` internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gwexpy.table.segment_table import SegmentTable

__all__ = [
    "plot_segment_table",
    "scatter_segment_table",
    "hist_segment_table",
    "segments_segment_table",
    "overlay_segment_table",
    "overlay_spectra_segment_table",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_plot() -> Any:
    """Return a new gwpy Plot instance."""
    from gwpy.plot import Plot

    return Plot()


def _require_column(st: SegmentTable, col: str) -> None:
    if col not in st.schema:
        raise KeyError(f"Column {col!r} not found in SegmentTable.")


def _require_scalar_column(st: SegmentTable, col: str) -> None:
    _require_column(st, col)
    kind = st.schema[col]
    if kind not in ("meta", "object"):
        raise TypeError(
            f"Column {col!r} has kind {kind!r}; only scalar/meta columns are "
            "supported for this plot type."
        )


# ---------------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------------


def plot_segment_table(
    st: SegmentTable,
    column: Optional[str] = None,
    row: Optional[int] = None,
    mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """General-purpose plot entry point for :class:`SegmentTable`.

    When both *column* and *row* are given and the payload is a GWpy
    time/frequency object, delegates to that object's ``.plot()`` method.

    Parameters
    ----------
    st:
        Source table.
    column:
        Payload column name.
    row:
        Row index.
    mode:
        Reserved for future use.
    **kwargs:
        Forwarded to the underlying plot method.

    Returns
    -------
    gwpy.plot.Plot

    Raises
    ------
    ValueError
        If *column* and *row* are not both specified, or the payload is not
        a plottable GWpy object.
    """
    if column is None or row is None:
        raise ValueError(
            "plot() requires both 'column' and 'row' to be specified. "
            "For table-level visualisations use scatter(), hist(), or segments()."
        )

    _require_column(st, column)
    kind = st.schema[column]
    if kind not in ("timeseries", "timeseriesdict", "frequencyseries", "frequencyseriesdict"):
        raise ValueError(
            f"plot() can only delegate to GWpy objects, "
            f"but column {column!r} has kind {kind!r}."
        )

    payload = st.row(row)[column]
    return payload.plot(**kwargs)


# ---------------------------------------------------------------------------
# scatter()
# ---------------------------------------------------------------------------


def scatter_segment_table(
    st: SegmentTable,
    x: str,
    y: str,
    color: Optional[str] = None,
    *,
    selection: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Scatter plot of two scalar columns.

    Parameters
    ----------
    st:
        Source table.
    x, y:
        Scalar/meta column names for the axes.
    color:
        Optional scalar column for point colour.
    selection:
        Optional boolean mask to subselect rows.
    **kwargs:
        Forwarded to :func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    gwpy.plot.Plot
    """
    _require_scalar_column(st, x)
    _require_scalar_column(st, y)

    df = st.to_pandas(meta_only=True)
    if selection is not None:
        df = df[list(selection)]


    plot = _get_plot()
    ax = plot.add_subplot(111)

    scatter_kwargs: dict[str, Any] = {}
    if color is not None:
        _require_scalar_column(st, color)
        scatter_kwargs["c"] = df[color].tolist()
        scatter_kwargs["cmap"] = kwargs.pop("cmap", "viridis")

    ax.scatter(df[x].tolist(), df[y].tolist(), **scatter_kwargs, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return plot


# ---------------------------------------------------------------------------
# hist()
# ---------------------------------------------------------------------------


def hist_segment_table(
    st: SegmentTable,
    column: str,
    *,
    bins: int = 10,
    range: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Histogram of a scalar column.

    Parameters
    ----------
    st:
        Source table.
    column:
        Scalar/meta column to histogram.
    bins:
        Number of bins.
    range:
        Optional (min, max) tuple.
    **kwargs:
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    gwpy.plot.Plot
    """
    _require_scalar_column(st, column)

    df = st.to_pandas(meta_only=True)
    plot = _get_plot()
    ax = plot.add_subplot(111)
    hist_kwargs: dict[str, Any] = {"bins": bins}
    if range is not None:
        hist_kwargs["range"] = range
    hist_kwargs.update(kwargs)
    ax.hist(df[column].tolist(), **hist_kwargs)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    return plot


# ---------------------------------------------------------------------------
# segments()
# ---------------------------------------------------------------------------


def segments_segment_table(
    st: SegmentTable,
    *,
    y: Optional[str] = None,
    color: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Draw each row's ``span`` as a horizontal bar.

    Parameters
    ----------
    st:
        Source table.
    y:
        Optional meta column; used to group rows on the y-axis.
    color:
        Optional meta column for bar colour.
    **kwargs:
        Forwarded to :func:`matplotlib.patches.Rectangle` / broken_barh.

    Returns
    -------
    gwpy.plot.Plot
    """

    df = st.to_pandas(meta_only=True)
    plot = _get_plot()
    ax = plot.add_subplot(111)

    # Resolve y values
    if y is not None:
        _require_scalar_column(st, y)
        y_vals = df[y].tolist()
    else:
        y_vals = list(range(len(df)))

    # Resolve colours
    colors: list[Any]
    if color is not None:
        _require_scalar_column(st, color)
        unique_vals = sorted(set(df[color].tolist()), key=str)
        import matplotlib as _mpl

        cmap = _mpl.colormaps.get_cmap("tab10").resampled(len(unique_vals))
        val_to_color = {v: cmap(i / max(len(unique_vals) - 1, 1)) for i, v in enumerate(unique_vals)}
        colors = [val_to_color[v] for v in df[color].tolist()]
    else:
        colors = ["steelblue"] * len(df)

    for i, (span, yv, c) in enumerate(zip(df["span"], y_vals, colors)):
        start, end = float(span[0]), float(span[1])
        bar_kwargs = {k: v for k, v in kwargs.items()}
        bar_kwargs.setdefault("edgecolor", "none")
        ax.broken_barh([(start, end - start)], (float(yv) - 0.4, 0.8), facecolors=[c], **bar_kwargs)

    ax.set_xlabel("time")
    ax.set_ylabel(y if y else "row index")
    return plot


# ---------------------------------------------------------------------------
# overlay()
# ---------------------------------------------------------------------------


def overlay_segment_table(
    st: SegmentTable,
    column: str,
    rows: list[int],
    *,
    separate: bool = False,
    sharex: bool = True,
    **kwargs: Any,
) -> Any:
    """Overlay (or juxtapose) payload from multiple rows.

    Parameters
    ----------
    st:
        Source table.
    column:
        Payload column (``timeseries``, ``timeseriesdict``,
        ``frequencyseries``, or ``frequencyseriesdict``).
    rows:
        Row indices to include.
    separate:
        If ``True``, each row gets its own axes.
    sharex:
        Share x-axis when ``separate=True``.
    **kwargs:
        Forwarded to the individual ``.plot()`` calls.

    Returns
    -------
    gwpy.plot.Plot
    """
    _require_column(st, column)

    payloads = [st.row(i)[column] for i in rows]

    if separate:
        n = len(payloads)
        plot = _get_plot()
        axes = plot.subplots(n, 1, sharex=sharex)
        if n == 1:
            axes = [axes]
        for ax, payload in zip(axes, payloads):
            payload.plot(ax=ax, **kwargs)
    else:
        plot = _get_plot()
        ax = plot.add_subplot(111)
        for payload in payloads:
            payload.plot(ax=ax, **kwargs)

    return plot


# ---------------------------------------------------------------------------
# overlay_spectra() — representative API
# ---------------------------------------------------------------------------


def overlay_spectra_segment_table(
    st: SegmentTable,
    column: str,
    *,
    channel: Optional[str] = None,
    rows: Optional[list[int]] = None,
    color_by: Optional[str] = None,
    sort_by: Optional[str] = None,
    cmap: str = "viridis",
    alpha: float = 0.7,
    linewidth: float = 0.8,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    xscale: str = "log",
    yscale: str = "log",
    xlim: Optional[Any] = None,
    ylim: Optional[Any] = None,
    ax: Optional[Any] = None,
) -> Any:
    """Overlay frequency spectra from multiple rows with colour-graded lines.

    This is one of the two *representative* APIs of :class:`SegmentTable`
    (alongside :func:`segments_segment_table`).  It combines the time-ordered
    colour-grading idiom with GWpy's ``FrequencySeries`` API to produce
    publication-quality overlaid spectra.

    Parameters
    ----------
    st:
        Source table.
    column:
        Payload column of kind ``"frequencyseries"`` or
        ``"frequencyseriesdict"``.
    channel:
        Required when *column* has kind ``"frequencyseriesdict"``.  Ignored
        for ``"frequencyseries"``.
    rows:
        Row indices to include.  ``None`` means all rows.
    color_by:
        Colour-grading basis.  One of:

        * ``None`` — default: sort & colour by ``span.start`` (time order).
        * ``"row"`` — row index ordering.
        * ``"t0"`` — ``span.start`` (GPS seconds).
        * Any scalar/meta column name.
    sort_by:
        Overlay order.  Same options as *color_by*.  Defaults to the same
        value as *color_by* (or ``"t0"`` when *color_by* is ``None``).
    cmap:
        Matplotlib colormap name.
    alpha:
        Line transparency.
    linewidth:
        Line width in points.
    colorbar:
        Whether to add a colorbar when *color_by* is a continuous quantity.
    colorbar_label:
        Label for the colorbar.  Auto-generated when omitted.
    xscale, yscale:
        Axis scales (default ``"log"`` for both — standard for spectra).
    xlim, ylim:
        Axis limits.
    ax:
        Existing :class:`matplotlib.axes.Axes` to draw into.  A new
        :class:`~gwpy.plot.Plot` is created when ``None``.

    Returns
    -------
    gwpy.plot.Plot

    Raises
    ------
    KeyError
        If *column*, *channel*, or the *color_by*/*sort_by* column does not
        exist.
    TypeError
        If *column* kind is not ``"frequencyseries"`` or
        ``"frequencyseriesdict"``, or if *color_by* column is non-numeric.
    ValueError
        If *rows* is empty, or ``channel=None`` for a
        ``"frequencyseriesdict"`` column.

    Examples
    --------
    >>> plot = st.overlay_spectra(
    ...     "asd",
    ...     channel="H1:STRAIN",
    ...     color_by="t0",
    ...     cmap="plasma",
    ... )
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors

    _require_column(st, column)
    kind = st.schema[column]
    if kind not in ("frequencyseries", "frequencyseriesdict"):
        raise TypeError(
            f"overlay_spectra() requires a 'frequencyseries' or "
            f"'frequencyseriesdict' column, got kind={kind!r}."
        )
    if kind == "frequencyseriesdict" and channel is None:
        raise ValueError(
            "channel must be specified when column kind is 'frequencyseriesdict'."
        )

    # Resolve row indices
    row_indices: list[int] = list(rows) if rows is not None else list(range(len(st)))
    if not row_indices:
        raise ValueError("rows must not be empty.")

    # Resolve color_by / sort_by
    _color_by = color_by
    _sort_by = sort_by if sort_by is not None else (_color_by if _color_by is not None else "t0")

    # Build ordering values
    def _get_scalar(col_id: str, idx: int) -> float:
        if col_id == "row":
            return float(idx)
        if col_id == "t0":
            return float(st.row(idx)["span"][0])
        # meta column
        if col_id not in st.schema:
            raise KeyError(f"Column {col_id!r} not found.")
        val = st.row(idx)[col_id]
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"color_by/sort_by column {col_id!r} must be numeric, "
                f"got {type(val).__name__!r}."
            ) from exc

    sort_vals = [_get_scalar(_sort_by, i) for i in row_indices]
    order = sorted(range(len(row_indices)), key=lambda k: sort_vals[k])
    sorted_indices = [row_indices[k] for k in order]

    # Build colour values
    if _color_by is None:
        color_vals = [float(st.row(i)["span"][0]) for i in sorted_indices]
    else:
        color_vals = [_get_scalar(_color_by, i) for i in sorted_indices]

    vmin, vmax = min(color_vals), max(color_vals)
    if vmin == vmax:
        norm = _mcolors.Normalize(vmin=vmin - 1, vmax=vmax + 1)
    else:
        norm = _mcolors.Normalize(vmin=vmin, vmax=vmax)
    import matplotlib as _mpl
    cmap_obj = _mpl.colormaps.get_cmap(cmap)

    # Set up axes
    if ax is not None:
        _ax = ax
        # Reuse the figure that owns the provided axes
        plot = ax.figure
    else:
        plot = _get_plot()
        _ax = plot.add_subplot(111)

    # Draw
    for z, (row_idx, cval) in enumerate(zip(sorted_indices, color_vals)):
        payload = st.row(row_idx)[column]
        if kind == "frequencyseriesdict":
            if channel not in payload:
                raise KeyError(f"Channel {channel!r} not found in FrequencySeriesDict.")
            fs = payload[channel]
        else:
            fs = payload

        color = cmap_obj(norm(cval))
        _ax.plot(
            fs.frequencies.value,
            fs.value,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=z,
        )

    _ax.set_xscale(xscale)
    _ax.set_yscale(yscale)
    if xlim is not None:
        _ax.set_xlim(*xlim)
    if ylim is not None:
        _ax.set_ylim(*ylim)
    _ax.set_xlabel("Frequency [Hz]")
    _ax.set_ylabel(column)

    # Colorbar
    if colorbar and len(color_vals) > 1:
        sm = _cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cb = plot.colorbar(sm, ax=_ax)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)
        else:
            default_labels = {"row": "row index", "t0": "segment start"}
            cb.set_label(default_labels.get(_color_by or "t0", _color_by or "t0"))

    return plot
