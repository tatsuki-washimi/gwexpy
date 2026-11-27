from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union, cast

import numpy as np

from ._axis import affine_representable_axis, exactly_affine_axis
from ._optional import require_optional
from .base import to_plain_array

if TYPE_CHECKING:
    import ROOT

    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.timeseries import TimeSeries

T_s = TypeVar("T_s", bound=Union["TimeSeries", "FrequencySeries", "Spectrogram"])

_ROOT_HISTOGRAM_DTYPES = {
    "C": np.dtype("int8"),
    "S": np.dtype("int16"),
    "I": np.dtype("int32"),
    "L": np.dtype("int64"),
    "F": np.dtype("float32"),
    "D": np.dtype("float64"),
}


def _root_histogram_dtype(class_name: str) -> np.dtype | None:
    """Return the native ROOT TH scalar dtype, if the class is known."""
    match = class_name.rsplit("TH", 1)[-1]
    if not match:
        return None
    return cast(np.dtype | None, _ROOT_HISTOGRAM_DTYPES.get(match[-1]))


def _unsupported_root_histogram(class_name: str) -> bool:
    """Profiles and polygon histograms have non-TH bin semantics."""
    return class_name.startswith("TProfile") or class_name == "TH2Poly"


def _get_label(obj: Any, unit: Any, default_name: str = "x") -> str:
    name = getattr(obj, "name", None) or default_name
    unit_str = str(unit) if unit else ""
    if unit_str:
        return f"{name} [{unit_str}]"
    return str(name)


_CENTER_ROUNDOFF_ULPS = 2.0


def _centers_match(actual: np.ndarray, centers: np.ndarray, edges: np.ndarray) -> bool:
    """Return whether bin centres equal *centers* up to edge round-off.

    A histogram stores edges; each centre is derived from two of them in
    floating point, so it can differ from the sample by a few ULPs of the
    neighbouring edges. Anything larger means the sample was moved.
    """
    scale = np.maximum(np.abs(edges[:-1]), np.abs(edges[1:]))
    tolerance = _CENTER_ROUNDOFF_ULPS * np.spacing(scale)
    return bool(np.all(np.abs(np.asarray(actual) - centers) <= tolerance))


def _edges_for_exact_centers(centers: np.ndarray) -> np.ndarray:
    """Build increasing bin edges whose centres are the given *centers*.

    Requiring ``(e[k] + e[k+1]) / 2 == c[k]`` fixes every edge once the first
    edge ``e0`` is chosen: ``e[k] = s[k] + (-1)**k * e0`` with
    ``s[k+1] = 2 c[k] - s[k]``. The widths are positive exactly when
    ``c[k] > e[k]`` for every ``k``, which bounds ``e0`` to an open interval.
    The conventional half-interval first edge is used when it lies in that
    interval, otherwise its midpoint. ``ValueError`` is raised when the
    interval is empty, i.e. no increasing edges can centre a bin on every
    sample.
    """
    values = np.asarray(centers, dtype=float)
    n = values.size
    if n == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    partial = np.empty(n + 1, dtype=float)
    partial[0] = 0.0
    for index, center in enumerate(values):
        partial[index + 1] = 2.0 * center - partial[index]
    upper = float(np.min(values[0::2] - partial[0:n:2]))
    lower = float(np.max(partial[1:n:2] - values[1::2]))
    conventional = float(values[0] - (values[1] - values[0]) / 2.0)
    if lower < conventional < upper:
        first = conventional
    else:
        first = 0.5 * (lower + upper)

    edges = np.empty(n + 1, dtype=float)
    edges[0] = first
    for index, center in enumerate(values):
        edges[index + 1] = 2.0 * center - edges[index]
    if (
        not lower < upper
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0)
        or not _centers_match(edges[:-1] + 0.5 * np.diff(edges), values, edges)
    ):
        raise ValueError(
            "ROOT histogram bin edges cannot preserve the exact sample coordinates "
            "(no increasing bin edges centre a bin on every sample); "
            "use a TGraph for one-dimensional data"
        )
    return edges


def _histogram_binning(centers: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(edges, uniform)`` for histogram bins centred on *centers*.

    A ROOT histogram is defined by its bin edges; series samples become bin
    centres. When the samples form a uniform grid (up to floating-point
    round-off), uniform binning is the faithful representation and ROOT
    derives the centres from the edges. Otherwise variable bins are built so
    that every centre equals its sample up to edge round-off, or
    ``ValueError`` is raised instead of moving the samples to
    midpoint-derived centres.
    """
    values = np.asarray(centers, dtype=float)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5]), True
    if affine_representable_axis(values):
        step = values[1] - values[0]
        edges = np.linspace(
            values[0] - step / 2, values[-1] + step / 2, values.size + 1
        )
        return edges, True
    return _edges_for_exact_centers(values), False


def _root_axis_unit(axis: Any) -> str | None:
    """Extract the bracketed unit from a ROOT axis title, if present."""
    import re

    match = re.search(r"\[(.*?)\]", axis.GetTitle())
    return match.group(1) if match else None


def _error_values_in_data_unit(values, error_unit, series):
    """Return error ``values`` given in ``error_unit`` in the unit of ``series``.

    Raises
    ------
    astropy.units.UnitConversionError
        If ``error_unit`` cannot be converted to the data unit.  An error bar
        in an incompatible unit has no magnitude on the data axis, so its raw
        number is never used.

    """
    from astropy import units as u

    data_unit = getattr(series, "unit", None)
    if data_unit is None or error_unit is None:
        return values
    try:
        return u.Quantity(values, error_unit).to(data_unit).value
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(
            f"error unit '{error_unit}' is not convertible to the data unit "
            f"'{data_unit}'"
        ) from exc


def _extract_error_array(series, error):
    """Extract an error array, in the data unit, from supported input types.

    A plain array or list is taken to be in the data unit already.  An
    `~astropy.units.Quantity` is converted to the data unit.  A series-like
    error is converted as well, unless it is dimensionless: that is the
    default of a series built without a unit and makes no unit claim.
    """
    from astropy import units as u

    # If it's a gwpy/gwexpy Series
    if hasattr(error, "value") and hasattr(error, "xindex"):
        if len(error) != len(series):
            raise ValueError(
                f"Error series length ({len(error)}) does not match data length ({len(series)})"
            )
        error_unit = getattr(error, "unit", None)
        if error_unit == u.dimensionless_unscaled:
            error_unit = None
        return _error_values_in_data_unit(
            to_plain_array(error.value), error_unit, series
        )

    # If it's an astropy Quantity
    if isinstance(error, u.Quantity):
        if error.shape != series.shape:
            raise ValueError(
                f"Error shape {error.shape} does not match data shape {series.shape}"
            )
        return _error_values_in_data_unit(error.value, error.unit, series)

    # If it's a plain numpy array or list
    err_arr = np.asarray(error)
    if err_arr.shape != series.shape:
        raise ValueError(
            f"Error shape {err_arr.shape} does not match data shape {series.shape}"
        )
    return err_arr


def to_tgraph(series: Any, error: Optional[Any] = None) -> ROOT.TGraph:
    """Convert a 1D series to a ROOT ``TGraph`` or ``TGraphErrors``."""
    ROOT = require_optional("ROOT")

    x = to_plain_array(series.xindex).astype(float)
    y = to_plain_array(series.value).astype(float)
    n = len(x)

    if error is not None:
        ey = _extract_error_array(series, error).astype(float)
        ex = np.zeros(n)
        graph = ROOT.TGraphErrors(n, x, y, ex, ey)
    else:
        graph = ROOT.TGraph(n, x, y)

    name = str(series.name or "graph")
    graph.SetName(name)
    graph.SetTitle(name)

    # Axis labels
    str(series.xunit) if hasattr(series, "xunit") else ""
    str(series.unit) if hasattr(series, "unit") else ""

    graph.GetXaxis().SetTitle(_get_label(series.xindex, series.xunit, default_name="x"))
    graph.GetYaxis().SetTitle(_get_label(series, series.unit, default_name="y"))

    return graph


def to_th1d(series: Any, error: Optional[Any] = None) -> ROOT.TH1D:
    """Convert a 1D series to a ROOT ``TH1D``."""
    ROOT = require_optional("ROOT")

    # Handle Histogram objects (composite, not ndarray-based)
    if hasattr(series, "edges") and hasattr(series, "values"):
        # Get edges and values
        edges = to_plain_array(series.edges).astype(float)
        y = to_plain_array(series.values).astype(float)
        n = len(y)

        name = str(getattr(series, "name", None) or "hist")
        title = name
        hist = ROOT.TH1D(name, title, n, edges)

        # Fill bins
        # ROOT bins: 0 = underflow, 1..n = content, n+1 = overflow
        content = np.zeros(n + 2, dtype=np.float64)
        content[1:-1] = y
        content[0] = to_plain_array(getattr(series, "underflow", 0.0)).astype(float)
        content[-1] = to_plain_array(getattr(series, "overflow", 0.0)).astype(float)
        hist.SetContent(content)

        # Errors: prefer sumw2 if available, otherwise diagonal of cov, otherwise derived from error arg
        if error is not None:
            ey = _extract_error_array(series, error).astype(float)
        elif hasattr(series, "errors") and series.errors is not None:
            ey = to_plain_array(series.errors).astype(float)
        else:
            ey = None

        if ey is not None:
            err_content = np.zeros(n + 2, dtype=np.float64)
            err_content[1:-1] = ey
            # Underflow/overflow errors from sumw2 if available
            sw2_under = getattr(series, "underflow_sumw2", None)
            if sw2_under is not None:
                err_content[0] = np.sqrt(to_plain_array(sw2_under).astype(float))
            sw2_over = getattr(series, "overflow_sumw2", None)
            if sw2_over is not None:
                err_content[-1] = np.sqrt(to_plain_array(sw2_over).astype(float))

            hist.SetError(err_content)

        # Labels
        hist.GetXaxis().SetTitle(
            _get_label(series.edges, series.xunit, default_name="x")
        )
        hist.GetYaxis().SetTitle(_get_label(series, series.unit, default_name="y"))

        return hist

    # Fallback to existing Series logic
    x = to_plain_array(series.xindex).astype(float)
    y = to_plain_array(series.value).astype(float)
    n = len(x)

    if n < 2:
        raise ValueError("Series must have at least 2 points for TH1D")

    # TH1D is defined by bin edges and the series samples become bin centres.
    # A uniform sample grid maps to fixed-width binning; any other grid must
    # keep each sample as the exact centre of its variable-width bin.
    edges, uniform = _histogram_binning(x)

    name = str(series.name or "hist")
    title = name

    if uniform:
        hist = ROOT.TH1D(name, title, n, float(edges[0]), float(edges[-1]))
    else:
        hist = ROOT.TH1D(name, title, n, edges)
        actual_centers = np.array(
            [hist.GetXaxis().GetBinCenter(i + 1) for i in range(n)]
        )
        if not _centers_match(actual_centers, x, edges):
            raise ValueError(
                "ROOT TH1D cannot preserve the exact sample coordinates; "
                "use to_tgraph()"
            )

    # Fill bins (vectorized)
    # TH1 has n+2 bins (0=underflow, n+1=overflow)
    content = np.zeros(n + 2, dtype=np.float64)
    content[1:-1] = y
    hist.SetContent(content)

    if error is not None:
        ey = _extract_error_array(series, error).astype(float)
        # Check if SetError is available (usually is for TH1)
        err_content = np.zeros(n + 2, dtype=np.float64)
        err_content[1:-1] = ey
        hist.SetError(err_content)

    # Labels
    str(series.xunit) if hasattr(series, "xunit") else ""
    str(series.unit) if hasattr(series, "unit") else ""
    hist.GetXaxis().SetTitle(_get_label(series.xindex, series.xunit, default_name="x"))
    hist.GetYaxis().SetTitle(_get_label(series, series.unit, default_name="y"))

    return hist


def to_th2d(spec: Spectrogram, error: Optional[Any] = None) -> ROOT.TH2D:
    """Convert a spectrogram to a ROOT ``TH2D``."""
    ROOT = require_optional("ROOT")

    times = to_plain_array(spec.times).astype(float)
    freqs = to_plain_array(spec.frequencies).astype(float)
    data = to_plain_array(spec.value).astype(float)

    nt = len(times)
    nf = len(freqs)

    # Uniform axes use uniform bin edges; other axes keep exact bin centres
    # (see ``_histogram_binning``).
    t_edges, t_uniform = _histogram_binning(times)
    f_edges, f_uniform = _histogram_binning(freqs)

    name = str(spec.name or "spectrogram")
    hist = ROOT.TH2D(name, name, nt, t_edges, nf, f_edges)
    for axis, centers, axis_edges, uniform in (
        (hist.GetXaxis(), times, t_edges, t_uniform),
        (hist.GetYaxis(), freqs, f_edges, f_uniform),
    ):
        if uniform:
            continue
        actual = np.array([axis.GetBinCenter(i + 1) for i in range(len(centers))])
        if not _centers_match(actual, centers, axis_edges):
            raise ValueError("ROOT TH2D cannot preserve the exact sample coordinates")

    # Fill (vectorized)
    # ROOT stores data as linearized array with index = x + (nx+2)*y
    # This corresponds to a C-style flattened array of shape (ny+2, nx+2)
    # where y is major (slow) and x is minor (fast).
    # We construct the 2D array in (ny+2, nx+2) shape, then flatten.

    full_content = np.zeros((nf + 2, nt + 2), dtype=np.float64)
    # Assign data. data is (nt, nf) -> tranpose to (nf, nt) to match [y, x]
    full_content[1:-1, 1:-1] = data.T

    hist.SetContent(full_content.flatten())

    if error is not None:
        err_arr = np.asarray(error).astype(float)
        if err_arr.shape != data.shape:
            raise ValueError("Error shape mismatch")
        full_error = np.zeros((nf + 2, nt + 2), dtype=np.float64)
        full_error[1:-1, 1:-1] = err_arr.T
        hist.SetError(full_error.flatten())

    # Labels
    hist.GetXaxis().SetTitle(
        _get_label(spec.times, spec.times.unit, default_name="time")
    )
    hist.GetYaxis().SetTitle(
        _get_label(spec.frequencies, spec.frequencies.unit, default_name="frequency")
    )
    hist.GetZaxis().SetTitle(_get_label(spec, spec.unit, default_name="value"))

    return hist


def from_root(
    cls: type[T_s], obj: Union[ROOT.TGraph, ROOT.TH1], return_error: bool = False
) -> Union[T_s, tuple[T_s, T_s]]:
    """Create a GWexpy object from a ROOT ``TGraph`` or ``TH1``.

    ``TH1`` objects can be converted to ``Histogram``, ``TimeSeries``, or
    ``FrequencySeries``. For the series targets, bin contents become the
    samples at the ROOT bin centers. Underflow and overflow contents and
    errors cannot be represented in a series; nonzero flow bins are discarded
    with a :class:`UserWarning`. With ``return_error=True``, return a pair of
    data and error series. ROOT's ``GetBinError`` is used, so explicit
    ``Sumw2`` errors are preserved and ROOT's default errors are used otherwise.
    """
    ROOT = require_optional("ROOT")
    from gwexpy.frequencyseries import FrequencySeries as FrequencySeriesType
    from gwexpy.histogram import Histogram as HistogramType
    from gwexpy.timeseries import TimeSeries as TimeSeriesType

    class_name = obj.ClassName() if hasattr(obj, "ClassName") else type(obj).__name__
    if _unsupported_root_histogram(class_name):
        raise TypeError(
            f"from_root does not support {class_name}; profile and polygon "
            "histograms have incompatible bin semantics."
        )

    # Reject collection/container types with a clear, actionable message before
    # the generic type check below.
    if isinstance(obj, ROOT.TMultiGraph):
        raise TypeError(
            "from_root does not accept a TMultiGraph; convert each TGraph "
            "individually, e.g. [from_root(cls, g) for g in obj.GetListOfGraphs()]."
        )
    if isinstance(obj, ROOT.TFile):
        raise TypeError(
            "from_root does not accept a TFile; read an individual object from "
            "the file first, e.g. from_root(cls, tfile.Get('name'))."
        )

    # Check type
    is_hist = isinstance(obj, ROOT.TH1)
    is_hist2d = isinstance(obj, ROOT.TH2)
    is_graph = isinstance(obj, ROOT.TGraph)

    if not is_hist and not is_graph and not is_hist2d:
        raise TypeError(f"Object {obj} is neither TH1, TH2 nor TGraph")

    is_histogram_cls = issubclass(cls, HistogramType)
    is_frequencyseries_cls = issubclass(cls, FrequencySeriesType)
    is_timeseries_cls = issubclass(cls, TimeSeriesType)

    if is_hist2d:
        if is_histogram_cls:
            raise TypeError(
                "Histogram class does not support 2D ROOT objects yet. Use Spectrogram."
            )
        nx = obj.GetNbinsX()
        ny = obj.GetNbinsY()
        x = np.array([obj.GetXaxis().GetBinCenter(i + 1) for i in range(nx)])
        y = np.array([obj.GetYaxis().GetBinCenter(j + 1) for j in range(ny)])
        value_dtype = _root_histogram_dtype(class_name) or np.dtype("float64")
        z = np.asarray(
            [
                [obj.GetBinContent(ix + 1, iy + 1) for iy in range(ny)]
                for ix in range(nx)
            ],
            dtype=value_dtype,
        )

        if return_error:
            ez = np.asarray(
                [
                    [obj.GetBinError(ix + 1, iy + 1) for iy in range(ny)]
                    for ix in range(nx)
                ],
                dtype=np.float64,
            )

        # In GWpy Spectrogram, typically shape is (Time, Freq)
        name = obj.GetName()
        unit = None
        # ... logic for unit extraction from Z axis if possible ...
        z_title = obj.GetZaxis().GetTitle()
        if "[" in z_title and "]" in z_title:
            import re

            match = re.search(r"\[(.*?)\]", z_title)
            if match:
                unit = match.group(1)

        res = cls(z, times=x, frequencies=y, unit=unit, name=name)
        if return_error:
            from typing import cast

            err_res = cls(
                cast(Any, ez), times=x, frequencies=y, unit=unit, name=f"{name}_error"
            )
            return res, err_res
        return res

    if is_hist:
        # TH1
        n = obj.GetNbinsX()

        # Edges: length n + 1
        edges = np.array([obj.GetXaxis().GetBinLowEdge(i + 1) for i in range(n + 1)])

        # `GetArray` has implementation-dependent element type.  Read through
        # ROOT's bin accessor to preserve the native TH{C,S,I,L,F,D} dtype;
        # this is also the conservative fallback for an unknown rectangular TH.
        value_dtype = _root_histogram_dtype(class_name) or np.dtype("float64")
        full_y = np.asarray(
            [obj.GetBinContent(index) for index in range(n + 2)], dtype=value_dtype
        )
        y = full_y[1:-1]
        underflow = full_y[0]
        overflow = full_y[-1]

        if return_error or is_histogram_cls:
            # GetBinError is ROOT's public error oracle: it handles fSumw2
            # and ROOT's default error rule uniformly.  Store variances in
            # float64 regardless of the TH content dtype.
            err_full = np.asarray(
                [obj.GetBinError(index) for index in range(n + 2)], dtype=np.float64
            )
            ey = err_full[1:-1]
            sw2 = np.square(ey, dtype=np.float64)
            underflow_sw2 = float(err_full[0] ** 2)
            overflow_sw2 = float(err_full[-1] ** 2)
        else:
            ey = None
            sw2 = None
            underflow_sw2 = None
            overflow_sw2 = None

        if is_histogram_cls:
            name = obj.GetName()
            # Unit extraction
            y_title = obj.GetYaxis().GetTitle()
            x_title = obj.GetXaxis().GetTitle()
            unit = None
            xunit = None
            import re

            if "[" in y_title and "]" in y_title:
                m = re.search(r"\[(.*?)\]", y_title)
                if m:
                    unit = m.group(1)
            if "[" in x_title and "]" in x_title:
                m = re.search(r"\[(.*?)\]", x_title)
                if m:
                    xunit = m.group(1)

            histogram_cls: Any = cls
            return histogram_cls(
                y,
                edges,
                unit=unit,
                xunit=xunit,
                sumw2=sw2,
                underflow=underflow,
                overflow=overflow,
                underflow_sumw2=underflow_sw2,
                overflow_sumw2=overflow_sw2,
                name=name,
            )

        if not is_frequencyseries_cls and not is_timeseries_cls:
            raise NotImplementedError(
                f"TH1 to {cls.__name__} conversion is not implemented; use "
                "Histogram, TimeSeries, or FrequencySeries."
            )

        # Series objects have no bins for values outside their coordinate
        # range. Do not silently lose nonzero flow-bin information.
        flow_errors = (obj.GetBinError(0), obj.GetBinError(n + 1))
        if underflow != 0 or overflow != 0 or any(err != 0 for err in flow_errors):
            warnings.warn(
                "Nonzero underflow or overflow content/error is discarded "
                "when converting a TH1 to a series.",
                UserWarning,
                stacklevel=2,
            )

        axis = obj.GetXaxis()
        x = np.array([axis.GetBinCenter(i + 1) for i in range(n)])
        bin_widths = np.array([axis.GetBinWidth(i + 1) for i in range(n)])
        name = obj.GetName()
        unit = _root_axis_unit(obj.GetYaxis())
        xunit = _root_axis_unit(obj.GetXaxis())
        coordinate_name = "frequencies" if is_frequencyseries_cls else "times"
        series_kwargs: dict[str, Any] = {
            "unit": unit,
            "name": name,
        }
        # A histogram's coordinates are defined by its bin edges. Fixed-width
        # binning is uniform by construction, so the series keeps the affine
        # origin and cadence. With variable binning, the bin centres become
        # explicit coordinates unless an origin and cadence rebuild every
        # centre exactly.
        uniform_binning = n > 0 and (
            not axis.IsVariableBinSize()
            or exactly_affine_axis(x, step=float(bin_widths[0]))
        )
        if uniform_binning:
            origin_name = "f0" if is_frequencyseries_cls else "x0"
            interval_name = "df" if is_frequencyseries_cls else "dt"
            series_kwargs[origin_name] = x[0]
            series_kwargs[interval_name] = bin_widths[0]
        else:
            series_kwargs[coordinate_name] = x
        if xunit is not None:
            series_kwargs["xunit"] = xunit

        res = cls(y, **series_kwargs)
        if return_error:
            assert ey is not None
            error_kwargs = dict(series_kwargs)
            error_kwargs["name"] = f"{name}_error"
            err_res = cls(ey, **error_kwargs)
            return res, err_res
        return res

    else:  # is_graph
        if is_histogram_cls:
            raise TypeError("Cannot create Histogram from TGraph. Use TH1 instead.")
        n = obj.GetN()
        # buffer access is faster
        x = np.frombuffer(obj.GetX(), dtype=np.float64, count=n).copy()
        y = np.frombuffer(obj.GetY(), dtype=np.float64, count=n).copy()
        if return_error:
            if hasattr(obj, "GetEY"):
                ey = np.frombuffer(obj.GetEY(), dtype=np.float64, count=n).copy()
            else:
                ey = np.zeros(n)
        else:
            ey = None

    # Try to extract name and unit
    name = obj.GetName()
    title = obj.GetYaxis().GetTitle()
    unit = None
    if "[" in title and "]" in title:
        # Simple parser "Name [Unit]"
        import re

        match = re.search(r"\[(.*?)\]", title)
        if match:
            unit = match.group(1)

    # Regularity check
    if is_graph:
        if not is_frequencyseries_cls and not is_timeseries_cls:
            raise TypeError(
                "TGraph can only be converted to TimeSeries or FrequencySeries."
            )
        if n > 1:
            # Graph points are the coordinates themselves: keep them unless an
            # origin and cadence rebuild every point exactly.
            dx_vals = np.diff(x)
            if exactly_affine_axis(x):
                res = cls(y, x0=float(x[0]), dx=float(dx_vals[0]), unit=unit, name=name)
            else:
                if is_frequencyseries_cls:
                    res = cls(y, frequencies=x, unit=unit, name=name)
                else:
                    res = cls(y, times=x, unit=unit, name=name)
        else:
            res = cls(y, x0=float(x[0]) if n == 1 else 0, unit=unit, name=name)

        if return_error:
            from typing import cast

            # Create a matching series for error
            if is_frequencyseries_cls:
                err_res = cls(
                    cast(Any, ey), frequencies=x, unit=unit, name=f"{name}_error"
                )
            else:
                err_res = cls(cast(Any, ey), times=x, unit=unit, name=f"{name}_error")
            return res, err_res
        return res

    raise TypeError("Unsupported ROOT object type for from_root().")


def to_tmultigraph(collection, name: Optional[str] = None) -> Any:
    """Convert a collection of series to a ROOT ``TMultiGraph``."""
    ROOT = require_optional("ROOT")
    mg = ROOT.TMultiGraph()
    title = name or getattr(collection, "name", "multigraph")
    mg.SetName(str(title))
    mg.SetTitle(str(title))

    # Handle dict or list
    if hasattr(collection, "items"):
        items = collection.items()
    else:
        items = enumerate(collection)

    for i, (key, series) in enumerate(items):
        if hasattr(series, "to_tgraph"):
            g = series.to_tgraph()
        else:
            g = to_tgraph(series)

        # Set default colors (ROOT color cycle: 1=Black, 2=Red, 3=Green, 4=Blue, ...)
        # Skip 0 (White) and handle large i
        color = (i % 9) + 1
        g.SetLineColor(color)
        g.SetMarkerColor(color)

        # Ensure it has a meaningful name in the legend
        if g.GetName() in ["graph", ""]:
            g.SetName(str(key))
            g.SetTitle(str(key))

        mg.Add(g)

    return mg


def write_root_file(collection, filename: str, **kwargs: Any) -> None:
    """Write a collection of series to a ROOT ``TFile``."""
    ROOT = require_optional("ROOT")
    mode = kwargs.get("mode", "recreate")

    # Save current directory to restore later
    old_dir = ROOT.gDirectory.GetDirectory("")

    f = ROOT.TFile.Open(filename, mode)
    if not f or f.IsZombie():
        raise OSError(f"Failed to open {filename} for writing")

    if hasattr(collection, "items"):
        items = collection.items()
    else:
        items = enumerate(collection)

    for key, series in items:
        if hasattr(series, "to_th1d"):
            obj = series.to_th1d()
        elif hasattr(series, "to_th2d"):
            obj = series.to_th2d()
        elif hasattr(series, "to_tgraph"):
            obj = series.to_tgraph()
        else:
            obj = to_tgraph(series)

        # Determine a good name for the object in the ROOT file
        obj_name = str(key)
        if isinstance(key, int):
            # For lists, try to use the object's own name if it has one
            obj_name = getattr(series, "name", None) or str(key)

        obj.SetName(str(obj_name))
        obj.Write()

    f.Close()
    if old_dir:
        old_dir.cd()
