# Plot

**Inherits from:** Plot


An extension of :class:`gwpy.plot.Plot` that automatically handles
:class:`gwexpy.types.SeriesMatrix` arguments by expanding them into
individual :class:`gwpy.types.Series` objects, while preserving
matrix layout and metadata where possible.


## Methods

### `__init__`

```python
__init__(self, *args, **kwargs)
```

Parameters
----------
figsize : 2-tuple of floats, default: :rc:`figure.figsize`
    Figure dimension ``(width, height)`` in inches.

dpi : float, default: :rc:`figure.dpi`
    Dots per inch.

facecolor : default: :rc:`figure.facecolor`
    The figure patch facecolor.

edgecolor : default: :rc:`figure.edgecolor`
    The figure patch edge color.

linewidth : float
    The linewidth of the frame (i.e. the edge linewidth of the figure
    patch).

frameon : bool, default: :rc:`figure.frameon`
    If ``False``, suppress drawing the figure background patch.

subplotpars : `SubplotParams`
    Subplot parameters. If not given, the default subplot
    parameters :rc:`figure.subplot.*` are used.

tight_layout : bool or dict, default: :rc:`figure.autolayout`
    Whether to use the tight layout mechanism. See `.set_tight_layout`.

    .. admonition:: Discouraged

        The use of this parameter is discouraged. Please use
        ``layout='tight'`` instead for the common case of
        ``tight_layout=True`` and use `.set_tight_layout` otherwise.

constrained_layout : bool, default: :rc:`figure.constrained_layout.use`
    This is equal to ``layout='constrained'``.

    .. admonition:: Discouraged

        The use of this parameter is discouraged. Please use
        ``layout='constrained'`` instead.

layout : {'constrained', 'compressed', 'tight', `.LayoutEngine`, None}
    The layout mechanism for positioning of plot elements to avoid
    overlapping Axes decorations (labels, ticks, etc). Note that
    layout managers can have significant performance penalties.
    Defaults to *None*.

    - 'constrained': The constrained layout solver adjusts axes sizes
       to avoid overlapping axes decorations.  Can handle complex plot
       layouts and colorbars, and is thus recommended.

      See :doc:`/tutorials/intermediate/constrainedlayout_guide`
      for examples.

    - 'compressed': uses the same algorithm as 'constrained', but
      removes extra space between fixed-aspect-ratio Axes.  Best for
      simple grids of axes.

    - 'tight': Use the tight layout mechanism. This is a relatively
      simple algorithm that adjusts the subplot parameters so that
      decorations do not overlap. See `.Figure.set_tight_layout` for
      further details.

    - A `.LayoutEngine` instance. Builtin layout classes are
      `.ConstrainedLayoutEngine` and `.TightLayoutEngine`, more easily
      accessible by 'constrained' and 'tight'.  Passing an instance
      allows third parties to provide their own layout engine.

    If not given, fall back to using the parameters *tight_layout* and
    *constrained_layout*, including their config defaults
    :rc:`figure.autolayout` and :rc:`figure.constrained_layout.use`.

Other Parameters
----------------
**kwargs : `.Figure` properties, optional

    Properties:
    agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array and two offsets from the bottom left corner of the image
    alpha: scalar or None
    animated: bool
    canvas: FigureCanvas
    clip_box: `.Bbox`
    clip_on: bool
    clip_path: Patch or (Path, Transform) or None
    constrained_layout: unknown
    constrained_layout_pads: unknown
    dpi: float
    edgecolor: color
    facecolor: color
    figheight: float
    figure: `.Figure`
    figwidth: float
    frameon: bool
    gid: str
    in_layout: bool
    label: object
    layout_engine: unknown
    linewidth: number
    mouseover: bool
    path_effects: `.AbstractPathEffect`
    picker: None or bool or float or callable
    rasterized: bool
    size_inches: (float, float) or float
    sketch_params: (scale: float, length: float, randomness: float)
    snap: bool or None
    tight_layout: unknown
    transform: `.Transform`
    url: str
    visible: bool
    zorder: float

*(Inherited from `Figure`)*

### `add_segments_bar`

```python
add_segments_bar(self, segments, ax=None, height=0.14, pad=0.1, sharex=True, location='bottom', **plotargs)
```

Add a segment bar `Plot` indicating state information.

By default, segments are displayed in a thin horizontal set of Axes
sitting immediately below the x-axis of the main,
similarly to a colorbar.

Parameters
----------
segments : `~gwpy.segments.DataQualityFlag`
    A data-quality flag, or `SegmentList` denoting state segments
    about this Plot

ax : `Axes`, optional
    Specific `Axes` relative to which to position new `Axes`,
    defaults to :func:`~matplotlib.pyplot.gca()`

height : `float, `optional
    Height of the new axes, as a fraction of the anchor axes

pad : `float`, optional
    Padding between the new axes and the anchor, as a fraction of
    the anchor axes dimension

sharex : `True`, `~matplotlib.axes.Axes`, optional
    Either `True` to set ``sharex=ax`` for the new segment axes,
    or an `Axes` to use directly

location : `str`, optional
    Location for new segment axes, defaults to ``'bottom'``,
    acceptable values are ``'top'`` or ``'bottom'``.

**plotargs
    extra keyword arguments are passed to
    :meth:`~gwpy.plot.SegmentAxes.plot`


### `close`

```python
close(self)
```

Close the plot and release its memory.
        

### `colorbar`

```python
colorbar(self, mappable=None, cax=None, ax=None, fraction=0.0, use_axesgrid=True, emit=True, **kwargs)
```

Add a colorbar to the current `Plot`.

This method differs from the default
:meth:`matplotlib.figure.Figure.colorbar` in that it doesn't
resize the parent `Axes` to accommodate the colorbar, but rather
draws a new Axes alongside it.

Parameters
----------
mappable : matplotlib data collection
    Collection against which to map the colouring

cax : `~matplotlib.axes.Axes`
    Axes on which to draw colorbar

ax : `~matplotlib.axes.Axes`
    Axes relative to which to position colorbar

fraction : `float`, optional
    Fraction of original axes to use for colorbar.
    The default (``fraction=0``) is to not resize the
    original axes at all.

use_axesgrid : `bool`
    Use :mod:`mpl_toolkits.axes_grid1` to generate the
    colorbar axes (default: `True`).
    This takes precedence over the ``use_gridspec``
    keyword argument from the upstream
    :meth:`~matplotlib.figure.Figure.colorbar` method.

emit : `bool`, optional
    If `True` update all mappables on `Axes` to match the same
    colouring as the colorbar.

**kwargs
    other keyword arguments to be passed to the
    :meth:`~matplotlib.figure.Figure.colorbar`

Returns
-------
cbar : `~matplotlib.colorbar.Colorbar`
    the newly added `Colorbar`

Notes
-----
To revert to the default matplotlib behaviour, pass
``use_axesgrid=False, fraction=0.15``.

See also
--------
matplotlib.figure.Figure.colorbar
matplotlib.colorbar.Colorbar

Examples
--------
>>> import numpy
>>> from gwpy.plot import Plot

To plot a simple image and add a colorbar:

>>> plot = Plot()
>>> ax = plot.gca()
>>> ax.imshow(numpy.random.randn(120).reshape((10, 12)))
>>> plot.colorbar(label='Value')
>>> plot.show()

Colorbars can also be generated by directly referencing the parent
axes:

>>> Plot = Plot()
>>> ax = plot.gca()
>>> ax.imshow(numpy.random.randn(120).reshape((10, 12)))
>>> ax.colorbar(label='Value')
>>> plot.show()


### `get_axes`

```python
get_axes(self, projection=None)
```

Find all `Axes`, optionally matching the given projection

Parameters
----------
projection : `str`
    name of axes types to return

Returns
-------
axlist : `list` of `~matplotlib.axes.Axes`


### `refresh`

```python
refresh(self)
```

Refresh the current figure
        

### `save`

```python
save(self, *args, **kwargs)
```

Save the figure to disk.

This method is an alias to :meth:`~matplotlib.figure.Figure.savefig`,
all arguments are passed directory to that method.


### `show`

```python
show(self, block=None, warn=True)
```

Display the current figure (if possible).

If blocking, this method replicates the behaviour of
:func:`matplotlib.pyplot.show()`, otherwise it just calls up to
:meth:`~matplotlib.figure.Figure.show`.

This method also supports repeatedly showing the same figure, even
after closing the display window, which isn't supported by
`pyplot.show` (AFAIK).

Parameters
----------
block : `bool`, optional
    open the figure and block until the figure is closed, otherwise
    open the figure as a detached window, default: `None`.
    If `None`, block if using an interactive backend and _not_
    inside IPython.

warn : `bool`, optional
    print a warning if matplotlib is not running in an interactive
    backend and cannot display the figure, default: `True`.


