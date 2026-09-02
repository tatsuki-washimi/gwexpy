from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, SupportsIndex, cast

import h5py

logger = logging.getLogger(__name__)
from astropy import units as u

try:
    import scipy.signal  # noqa: F401 - availability check
except ImportError:
    pass  # scipy is optional dependency for gwpy but required here for hilbert


from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries import TimeSeriesDict as BaseTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as BaseTimeSeriesList

# --- Monkey Patch TimeSeriesDict ---
from gwexpy.interop._registry import ConverterRegistry
from gwexpy.io.hdf5_collection import (
    LAYOUT_DATASET,
    LAYOUT_GROUP,
    detect_hdf5_layout,
    ensure_hdf5_file,
    normalize_layout,
    read_hdf5_keymap,
    read_hdf5_order,
    safe_hdf5_key,
    unique_hdf5_key,
    write_hdf5_manifest,
)
from gwexpy.io.time_selection import apply_time_selection
from gwexpy.types.mixin import PhaseMethodsMixin
from gwexpy.types.mixin._collection_mixin import (
    DictMapMixin,
    ListMapMixin,
    _make_dict_map_method,
    _make_list_map_method,
)
from gwexpy.types.mixin._plot_mixin import PlotMixin

from ._gwf_io import (
    _GWF_BACKENDS,
    _GWF_PARALLEL_HELP,
    _extract_gwf_read_args,
    _format_gwf_import_error,
    _gwf_parallel_read_signature,
    _GWFParallelContractError,
    _normalize_gwf_parallel_kwargs,
    _prepare_gwf_parallel_source,
    _source_for_gwf_channel_listing,
    _validate_gwf_parallel_source,
    read_gwf_timeseriesdict,
)
from .spectral import coherence_matrix_from_collection, csd_matrix_from_collection


def _parse_fft_positional_args(
    args: tuple[Any, ...],
    *,
    fftlength: Any,
    overlap: Any,
    method_name: str,
) -> tuple[Any, Any]:
    """Parse optional positional ``(fftlength, overlap)`` compatibility args."""
    if not args:
        return fftlength, overlap
    if len(args) > 2:
        raise TypeError(
            f"{method_name}() accepts at most two positional spectral arguments: "
            "fftlength, overlap."
        )
    if fftlength is not None or overlap is not None:
        raise TypeError(
            f"{method_name}() cannot mix positional fftlength/overlap with "
            "keyword fftlength/overlap."
        )
    return args[0], (args[1] if len(args) == 2 else None)


def _coerce_reader_result(cls, reader_result):
    """Wrap a collection reader result while retaining collection provenance."""
    result = cls(reader_result)
    provenance = getattr(reader_result, "_gwexpy_io", None)
    if isinstance(provenance, dict):
        result._gwexpy_io = {**provenance}
    return result


def _is_timeseries_hdf5_dataset(
    obj: Any, *, allow_missing_xunit: bool = False
) -> bool:
    """Return whether an HDF5 dataset is eligible as a TimeSeries entry."""
    if not isinstance(obj, h5py.Dataset) or obj.ndim != 1:
        return False
    kind = obj.dtype.kind
    if kind not in {"b", "u", "i", "f", "c"} and not (
        kind == "V" and obj.dtype.fields is None
    ):
        return False
    if "xunit" not in obj.attrs:
        return allow_missing_xunit
    xunit = obj.attrs["xunit"]
    if getattr(xunit, "shape", ()) != ():
        return False
    try:
        return u.s.is_equivalent(xunit)
    except (TypeError, ValueError, u.UnitsError):
        return False


class TimeSeriesDict(PlotMixin, DictMapMixin, PhaseMethodsMixin, BaseTimeSeriesDict):
    """A dictionary of TimeSeries, indexed by name.

    `TimeSeriesDict` is a specialized dictionary designed to hold and
    manipulate multiple `TimeSeries` objects simultaneously. It provides
    batch processing methods (e.g., `resample`, `filter`, `fft`) that
    operate on all entries at once, and supports advanced I/O for
    multi-channel data (HDF5, Zarr, CSV).

    Parameters
    ----------
    *args
        A mapping or iterable of `(key, TimeSeries)` pairs.

    **kwargs
        Additional keyword arguments for the dictionary.

    Notes
    -----
    This class is highly interoperable, supporting conversions to and from
    Pandas DataFrames, Polars DataFrames, and MNE Raw objects. It also
    supports matrix conversion via `to_matrix()`.

    Key methods:

    .. autosummary::

       ~TimeSeriesDict.read
       ~TimeSeriesDict.write
       ~TimeSeriesDict.plot
       ~TimeSeriesDict.resample
       ~TimeSeriesDict.fft
       ~TimeSeriesDict.psd

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries, TimeSeriesDict
    >>> tsd = TimeSeriesDict()
    >>> tsd['H1'] = TimeSeries([1, 2], sample_rate=1)
    >>> tsd
    {'H1': <TimeSeries([1, 2],
                unit=Unit(dimensionless),
                t0=<Quantity 0. s>,
                dt=<Quantity 1. s>,
                name=None,
                channel=None)>}

    """

    @classmethod
    def read(cls, source, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Read a `TimeSeriesDict` from a supported source."""
        from gwexpy._bootstrap import register_all

        register_all()

        fmt = kwargs.get("format")
        if fmt in {"nc", "netcdf4"}:
            from gwexpy.timeseries.io.netcdf4_ import read_timeseriesdict_netcdf4

            reader_kwargs = dict(kwargs)
            reader_kwargs.pop("format", None)
            return _coerce_reader_result(
                cls, read_timeseriesdict_netcdf4(source, **reader_kwargs)
            )
        if fmt == "zarr":
            from gwexpy.timeseries.io.zarr_ import read_timeseriesdict_zarr

            reader_kwargs = dict(kwargs)
            reader_kwargs.pop("format", None)
            return _coerce_reader_result(
                cls, read_timeseriesdict_zarr(source, *args, **reader_kwargs)
            )
        if fmt == "ats.mth5":
            raise TypeError(
                "format 'ats.mth5' supports TimeSeries.read only; "
                "TimeSeriesDict.read is not supported"
            )
        source, gwf_format = _prepare_gwf_parallel_source(source, fmt, kwargs)
        try:
            p = Path(source)
        except TypeError:
            p = None
        if fmt in {
            "mseed",
            "miniseed",
            "sac",
            "gse2",
            "knet",
            "win",
            "win32",
            "ats",
            "gbd",
            "tdms",
            "xml.diaggui",
            "dttxml",
        }:
            direct_readers = {
                "mseed": (
                    "gwexpy.timeseries.io.seismic",
                    "read_miniseed_timeseriesdict",
                ),
                "miniseed": (
                    "gwexpy.timeseries.io.seismic",
                    "read_miniseed_timeseriesdict",
                ),
                "sac": ("gwexpy.timeseries.io.seismic", "read_sac_timeseriesdict"),
                "gse2": ("gwexpy.timeseries.io.seismic", "read_gse2_timeseriesdict"),
                "knet": ("gwexpy.timeseries.io.seismic", "read_knet_timeseriesdict"),
                "win": ("gwexpy.timeseries.io.win", "read_win_file"),
                "win32": ("gwexpy.timeseries.io.win", "read_win_file"),
                "ats": ("gwexpy.timeseries.io.ats", "read_timeseriesdict_ats"),
                "gbd": ("gwexpy.timeseries.io.gbd", "read_timeseriesdict_gbd"),
                "tdms": ("gwexpy.timeseries.io.tdms", "read_timeseriesdict_tdms"),
                "xml.diaggui": (
                    "gwexpy.timeseries.io.dttxml",
                    "read_timeseriesdict_dttxml",
                ),
                "dttxml": ("gwexpy.timeseries.io.dttxml", "read_timeseriesdict_dttxml"),
            }
            module_name, func_name = direct_readers[fmt]
            module = __import__(module_name, fromlist=[func_name])
            reader = getattr(module, func_name)
            reader_kwargs = dict(kwargs)
            reader_kwargs.pop("format", None)
            return _coerce_reader_result(cls, reader(source, *args, **reader_kwargs))
        if gwf_format is not None:
            channels, start, end, gwf_kwargs = _extract_gwf_read_args(
                args,
                kwargs,
                allow_multiple_channels=True,
            )
            source = _validate_gwf_parallel_source(source, gwf_kwargs)
            _, parallel_workers = _normalize_gwf_parallel_kwargs(
                dict(gwf_kwargs),
                number_of_spans=len(source) if isinstance(source, (list, tuple)) else 1,
            )
            TimeSeries = cast(Any, ConverterRegistry.get_constructor("TimeSeries"))
            backend = gwf_kwargs.pop("backend", _GWF_BACKENDS[gwf_format])
            try:
                from gwpy.io.gwf.core import get_channel_names

                if channels is None:
                    channel_source = _source_for_gwf_channel_listing(source)
                    channels = get_channel_names(channel_source, backend=backend)
                    if not channels:
                        raise ValueError(f"No channels found in GWF source: {source}")
                return read_gwf_timeseriesdict(
                    source,
                    channels,
                    start=start,
                    end=end,
                    backend=backend,
                    dict_class=cls,
                    series_class=TimeSeries,
                    **gwf_kwargs,
                )
            except ImportError as exc:
                if parallel_workers > 1:
                    # Preserve the original worker/backend ImportError.  The
                    # formatted dependency hint is a serial compatibility path.
                    raise
                raise _format_gwf_import_error(gwf_format, exc)
            except _GWFParallelContractError:
                raise
            except TypeError as exc:
                if parallel_workers > 1:
                    raise
                # Keep existing ValueError contract for malformed user inputs.
                raise ValueError(f"Invalid input for GWF read: {exc}") from exc
        if p is not None and fmt is None and str(p).lower().endswith(".zarr"):
            from gwexpy.timeseries.io.zarr_ import read_timeseriesdict_zarr

            return _coerce_reader_result(
                cls,
                read_timeseriesdict_zarr(
                    p, **{k: v for k, v in kwargs.items() if k != "format"}
                ),
            )
        if p is not None and p.is_dir() and (fmt in (None, "csv", "txt")):
            from gwexpy.io.collection_dir import read_collection_dir
            from gwexpy.io.utils import _reject_timezone_reinterpretation, apply_unit

            if fmt == "txt":
                _reject_timezone_reinterpretation(
                    "txt", kwargs.pop("timezone", None), None
                )

            TimeSeries = cast(Any, ConverterRegistry.get_constructor("TimeSeries"))

            _, items = read_collection_dir(
                p,
                expected_kind="TimeSeriesDict",
                entry_format=fmt,
                reader=lambda path, f: TimeSeries.read(path, format=f),
            )
            out = cls()
            for k, v, meta in items:
                unit = meta.get("unit")
                v = apply_unit(v, unit) if unit else v
                out[k] = v
            # The per-file reader above is called without the bounds, so this
            # branch used to return every sample in the directory for a windowed
            # request (issue #611).
            return apply_time_selection(out, kwargs.get("start"), kwargs.get("end"))
        if fmt in ("hdf5", "h5", "hdf"):
            from gwexpy.io.utils import _reject_timezone_reinterpretation

            _reject_timezone_reinterpretation(
                "hdf5", kwargs.pop("timezone", None), None
            )
            TimeSeries = cast(Any, ConverterRegistry.get_constructor("TimeSeries"))

            # This branch reopens the file and re-reads each dataset itself
            # instead of going through the registered reader, and used to drop
            # ``start``/``end`` on the way (issue #611) — so a bounded dict read
            # silently returned the whole file even though the single-series
            # path cropped correctly.  Read whole, then crop, which is the
            # documented oracle.  Read out of ``kwargs`` rather than popped,
            # because the fall-through below hands the untouched kwargs to the
            # registry reader, which applies them itself.
            start = kwargs.get("start")
            end = kwargs.get("end")

            with h5py.File(source, "r") as h5f:
                layout = detect_hdf5_layout(h5f)
                keymap = read_hdf5_keymap(h5f)
                order = read_hdf5_order(h5f)
                keys = order or list(h5f.keys())
                out = cls()
                if layout == LAYOUT_DATASET or layout is None:
                    for ds_name in keys:
                        try:
                            ts = TimeSeries.read(h5f, format="hdf5", path=ds_name)
                        except (KeyError, ValueError, TypeError, OSError) as e:
                            logger.debug("Skipping dataset %s: %s", ds_name, e)
                            continue
                        orig_key = keymap.get(ds_name, ds_name)
                        out[orig_key] = ts
                    return apply_time_selection(out, start, end)
                if layout == LAYOUT_GROUP:
                    for grp_name in keys:
                        try:
                            grp = h5f[grp_name]
                            ts = TimeSeries.read(grp, format="hdf5", path="data")
                        except (KeyError, ValueError, TypeError, OSError):
                            try:
                                ts = TimeSeries.read(grp, format="hdf5")
                            except (KeyError, ValueError, TypeError, OSError) as e2:
                                logger.debug("Skipping group %s: %s", grp_name, e2)
                                continue
                        orig_key = keymap.get(grp_name, grp_name)
                        out[orig_key] = ts
                    return apply_time_selection(out, start, end)
        return super().read(source, *args, **kwargs)

    def __reduce_ex__(self, protocol: SupportsIndex):
        from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

        return (GwpyTimeSeriesDict, (dict(self),))

    asfreq = _make_dict_map_method(
        "asfreq", doc="Apply asfreq to each TimeSeries in the dict."
    )

    def resample(self, rate, **kwargs):
        """Resample items in the TimeSeriesDict.

        In-place operation (updates the dict contents).

        If rate is time-like, performs time-bin resampling.
        Otherwise performs signal processing resampling (gwpy's native behavior).
        """
        is_time_bin = False
        if isinstance(rate, str):
            is_time_bin = True
        elif isinstance(rate, u.Quantity) and rate.unit.physical_type == "time":
            is_time_bin = True

        if is_time_bin:
            # Time-bin logic: replace items in-place
            # We can't strictly modify the objects in-place easily
            # (asfreq/resample return new objects usually),
            # so we replace the values in the dict.
            for key in list(self.keys()):
                self[key] = self[key].resample(rate, **kwargs)
            return self
        else:
            # Native gwpy resample (signal processing)
            # gwpy's TimeSeriesDict.resample is in-place
            return super().resample(rate, **kwargs)

    hilbert = _make_dict_map_method(
        "hilbert", doc="Apply Hilbert transform to each item."
    )
    envelope = _make_dict_map_method("envelope", doc="Apply envelope to each item.")
    instantaneous_phase = _make_dict_map_method(
        "instantaneous_phase", doc="Apply instantaneous_phase to each item."
    )

    # ===============================
    # P2 Methods (Domain Specific)
    # ===============================

    def to_mne(self, info=None, picks=None):
        """Convert to mne.io.RawArray."""
        from gwexpy.interop import to_mne_rawarray

        return to_mne_rawarray(self, info=info, picks=picks)

    @classmethod
    def from_mne(cls, raw, *, unit_map=None):
        """Create from mne.io.Raw."""
        from gwexpy.interop import from_mne_raw

        return from_mne_raw(cls, raw, unit_map=unit_map)

    def to_obspy(self, *, stats_extra=None, dtype=None):
        """Convert to an obspy.Stream (one Trace per TimeSeries)."""
        from gwexpy.interop import to_obspy

        return to_obspy(self, stats_extra=stats_extra, dtype=dtype)

    @classmethod
    def from_obspy(cls, stream, *, unit=None, name_policy="id"):
        """Create a TimeSeriesDict from an obspy.Stream (or Trace).

        Each Trace in the Stream becomes a TimeSeries, keyed by its name
        (per ``name_policy``).
        """
        from gwexpy.interop import from_obspy

        return from_obspy(cls, stream, unit=unit, name_policy=name_policy)

    @classmethod
    def from_control(cls, response: Any, **kwargs) -> TimeSeriesDict:
        """Create TimeSeriesDict from python-control TimeResponseData.

        Parameters
        ----------
        response : control.TimeResponseData
            The simulation result from python-control.
        **kwargs : dict
            Additional arguments passed to the TimeSeries constructor.

        Returns
        -------
        TimeSeriesDict
            The converted time-domain data.

        """
        from gwexpy.interop import from_control_response

        res = from_control_response(cls, response, **kwargs)
        if not isinstance(res, cls):
            # Wrap in a Dictionary if it isn't one already
            obj = cls()
            name = getattr(res, "name", "output")
            obj[name] = res
            return obj
        return res

    radian = _make_dict_map_method(
        "radian", doc="Compute instantaneous phase (in radians) of each item."
    )
    degree = _make_dict_map_method(
        "degree", doc="Compute instantaneous phase (in degrees) of each item."
    )

    # phase() and angle() are provided by PhaseMethodsMixin

    unwrap_phase = _make_dict_map_method(
        "unwrap_phase", doc="Apply unwrap_phase to each item."
    )
    instantaneous_frequency = _make_dict_map_method(
        "instantaneous_frequency", doc="Apply instantaneous_frequency to each item."
    )
    mix_down = _make_dict_map_method("mix_down", doc="Apply mix_down to each item.")
    baseband = _make_dict_map_method("baseband", doc="Apply baseband to each item.")
    heterodyne = _make_dict_map_method(
        "heterodyne", doc="Apply heterodyne to each item."
    )

    def lock_in(self, *args, **kwargs):
        """Apply lock_in to each item.

        Returns TimeSeriesDict (if output='complex') or tuple of TimeSeriesDicts.
        """
        # We need to know the output structure (tuple vs single)
        # Peek first item
        if not self:
            return self.__class__()

        keys = list(self.keys())
        first_res = self[keys[0]].lock_in(*args, **kwargs)

        if isinstance(first_res, tuple):
            # Tuple return (e.g. mag, phase or i, q)
            # Assume logic dictates uniform return type
            dict_tuple = tuple(self.__class__() for _ in first_res)

            for key, ts in self.items():
                res = ts.lock_in(*args, **kwargs)
                for i, val in enumerate(res):
                    dict_tuple[i][key] = val
            return dict_tuple
        else:
            # Single return
            new_dict = self.__class__()
            new_dict[keys[0]] = first_res
            for key in keys[1:]:
                new_dict[key] = self[key].lock_in(*args, **kwargs)
            return new_dict

    def csd_matrix(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        hermitian=True,
        include_diagonal=True,
        **kwargs,
    ):
        """Compute Cross-Spectral Density matrix for all pairs.

        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Another collection for cross-CSD. If None, compute self-CSD matrix.
        *args
            Positional arguments forwarded to `TimeSeries.csd`.
        fftlength : float, optional
            FFT length in seconds.
        overlap : float, optional
            Overlap between segments in seconds.
        window : str, optional
            Window function name (default 'hann').
        hermitian : bool, optional
            If True, exploit Hermitian symmetry (default True).
        include_diagonal : bool, optional
            Must be True for CSD matrices; False raises ValueError because the
            diagonal is always the PSD.
        **kwargs
            Additional keyword arguments forwarded to `TimeSeries.csd`.

        Returns
        -------
        FrequencySeriesMatrix
            The CSD matrix.

        Notes
        -----
        The diagonal of a self-CSD matrix is always computed as the PSD. Any
        uncomputed elements are represented as complex NaN. The frequency axis
        is taken from the first computed element without alignment/truncation;
        dt and fftlength consistency is enforced before computation.

        """
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.csd_matrix",
        )
        return csd_matrix_from_collection(
            self,
            other,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            hermitian=hermitian,
            include_diagonal=include_diagonal,
            **kwargs,
        )

    def coherence_matrix(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        symmetric=True,
        include_diagonal=True,
        diagonal_value=1.0,
        **kwargs,
    ):
        """Compute coherence matrix for all pairs.

        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Another collection for cross-coherence.
        *args
            Positional arguments forwarded to `TimeSeries.coherence`.
        fftlength : float, optional
            FFT length in seconds.
        overlap : float, optional
            Overlap between segments in seconds.
        window : str, optional
            Window function name (default 'hann').
        symmetric : bool, optional
            If True, exploit symmetry (default True).
        include_diagonal : bool, optional
            Whether to include diagonal elements (default True).
        diagonal_value : float, optional
            Value for diagonal elements (default 1.0).
        **kwargs
            Additional keyword arguments forwarded to `TimeSeries.coherence`.

        Returns
        -------
        FrequencySeriesMatrix
            The coherence matrix.

        Notes
        -----
        If include_diagonal is True and diagonal_value is not None, the
        diagonal is filled with that value without computation. If
        diagonal_value is None, the diagonal coherence is computed. Uncomputed
        elements are represented as NaN. The frequency axis is taken from the
        first computed element without alignment/truncation; dt and fftlength
        consistency is enforced before computation.

        """
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.coherence_matrix",
        )
        return coherence_matrix_from_collection(
            self,
            other,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            symmetric=symmetric,
            include_diagonal=include_diagonal,
            diagonal_value=diagonal_value,
            **kwargs,
        )

    def csd(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        hermitian=True,
        include_diagonal=True,
        **kwargs,
    ):
        """Compute CSD for each element or as a matrix depending on `other`."""
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.csd",
        )
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.csd_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.interop._registry import ConverterRegistry

            FrequencySeriesDict = ConverterRegistry.get_constructor(
                "FrequencySeriesDict"
            )
            new_dict = FrequencySeriesDict()
            for key, ts in self.items():
                new_dict[key] = ts.csd(
                    other, fftlength=fftlength, overlap=overlap, window=window, **kwargs
                )
            return new_dict

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return csd_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        raise TypeError("other must be TimeSeries, TimeSeriesList/Dict, or None/'self'")

    def coherence(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        symmetric=True,
        include_diagonal=True,
        diagonal_value=1.0,
        **kwargs,
    ):
        """Compute coherence for each element or as a matrix depending on `other`."""
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.coherence",
        )
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.coherence_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.interop._registry import ConverterRegistry

            FrequencySeriesDict = ConverterRegistry.get_constructor(
                "FrequencySeriesDict"
            )
            new_dict = FrequencySeriesDict()
            for key, ts in self.items():
                new_dict[key] = ts.coherence(
                    other, fftlength=fftlength, overlap=overlap, window=window, **kwargs
                )
            return new_dict

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return coherence_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        raise TypeError("other must be TimeSeries, TimeSeriesList/Dict, or None/'self'")

    psd = _make_dict_map_method(
        "psd",
        doc="Compute PSD for each TimeSeries. Returns a FrequencySeriesDict.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesDict",
    )
    asd = _make_dict_map_method(
        "asd",
        doc="Compute ASD for each TimeSeries. Returns a FrequencySeriesDict.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesDict",
    )
    spectrogram = _make_dict_map_method(
        "spectrogram",
        doc="Compute spectrogram for each TimeSeries. Returns a SpectrogramDict.",
        result_class_path="gwexpy.spectrogram.SpectrogramDict",
    )
    spectrogram2 = _make_dict_map_method(
        "spectrogram2",
        doc="Compute spectrogram2 for each TimeSeries. Returns a SpectrogramDict.",
        result_class_path="gwexpy.spectrogram.SpectrogramDict",
    )
    q_transform = _make_dict_map_method(
        "q_transform",
        doc="Compute Q-transform for each TimeSeries. Returns a SpectrogramDict.",
        result_class_path="gwexpy.spectrogram.SpectrogramDict",
    )
    histogram = _make_dict_map_method(
        "histogram",
        doc="Compute Histogram for each TimeSeries. Returns a HistogramDict.",
        result_class_path="gwexpy.histogram.collections.HistogramDict",
    )

    # ===============================
    # Interoperability Methods (P0)
    # ===============================

    def to_pandas(self, index="datetime", *, copy=False):
        """Convert to pandas.DataFrame."""
        from gwexpy.interop import to_pandas_dataframe

        return to_pandas_dataframe(self, index=index, copy=copy)

    @classmethod
    def from_pandas(cls, df, *, unit_map=None, t0=None, dt=None):
        """Create TimeSeriesDict from pandas.DataFrame."""
        from gwexpy.interop import from_pandas_dataframe

        return from_pandas_dataframe(cls, df, unit_map=unit_map, t0=t0, dt=dt)

    def to_polars(self, time_column="time", time_unit="datetime"):
        """Convert to polars.DataFrame."""
        from gwexpy.interop import to_polars_dict

        return to_polars_dict(self, index_column=time_column, time_unit=time_unit)

    @classmethod
    def from_polars(cls, df, *, time_column="time", unit_map=None):
        """Create TimeSeriesDict from polars.DataFrame."""
        from gwexpy.interop import from_polars_dict

        return from_polars_dict(cls, df, index_column=time_column, unit_map=unit_map)

    def to_tmultigraph(self, name: str | None = None) -> Any:
        """Convert to ROOT TMultiGraph."""
        from gwexpy.interop import to_tmultigraph

        return to_tmultigraph(self, name=name)

    def write(self, target: str, *args: Any, **kwargs: Any) -> Any:
        """Write the collection to a supported target."""
        from gwexpy._bootstrap import register_all

        register_all()

        fmt = kwargs.get("format")
        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
            from gwexpy.interop.root_ import write_root_file

            return write_root_file(self, target, **kwargs)
        if fmt in ("csv", "txt"):
            from gwexpy.io.collection_dir import write_collection_dir

            overwrite = bool(kwargs.pop("overwrite", False))
            # Each entry is written as a standalone GWpy-compatible CSV/TXT.
            return write_collection_dir(
                target,
                kind="TimeSeriesDict",
                entry_format=str(fmt),
                entries=list(self.items()),
                writer=lambda ts, path, f: ts.write(path, format=f),
                meta_getter=lambda ts: {"unit": str(getattr(ts, "unit", "") or "")},
                overwrite=overwrite,
            )
        if fmt in ("hdf5", "h5", "hdf"):
            overwrite = bool(kwargs.pop("overwrite", False))
            append = bool(kwargs.pop("append", False))
            mode = kwargs.pop("mode", None)
            if append and mode in {"w", "w-", "x"}:
                raise ValueError(
                    f"append=True is incompatible with HDF5 create mode {mode!r}"
                )
            if append and mode is None:
                mode = "a"
            merge = append or mode in ("a", "r+")
            layout = normalize_layout(kwargs.pop("layout", "gwpy"))
            with ensure_hdf5_file(
                target,
                mode=mode,
                overwrite=overwrite and not append,
            ) as h5f:
                root_names = list(h5f.keys())
                used = set(root_names)
                if merge:
                    from gwexpy.timeseries.io.hdf5 import _ROLLBACK_PREFIX

                    stored_keymap = read_hdf5_keymap(h5f)
                    stored_order = read_hdf5_order(h5f)
                    manifest_explicit = set(stored_order) | set(stored_keymap)
                    eligible = []
                    for name in root_names:
                        if name.startswith(_ROLLBACK_PREFIX):
                            continue
                        link = h5f.get(name, getlink=True)
                        if not isinstance(link, h5py.HardLink):
                            continue
                        obj = h5f[name]
                        allow_missing_xunit = name in manifest_explicit
                        if layout == LAYOUT_DATASET:
                            if _is_timeseries_hdf5_dataset(
                                obj,
                                allow_missing_xunit=allow_missing_xunit,
                            ):
                                eligible.append(name)
                            continue
                        if not isinstance(obj, h5py.Group):
                            continue
                        data_link = obj.get("data", getlink=True)
                        if not isinstance(data_link, h5py.HardLink):
                            continue
                        if _is_timeseries_hdf5_dataset(
                            obj.get("data"),
                            allow_missing_xunit=allow_missing_xunit,
                        ):
                            eligible.append(name)
                    eligible_set = set(eligible)
                    keymap = {
                        name: stored_keymap.get(name, name) for name in eligible
                    }

                    logical_to_physical: dict[str, str] = {}
                    for physical, logical in keymap.items():
                        if logical in logical_to_physical:
                            raise ValueError(
                                "ambiguous existing HDF5 logical key "
                                f"{logical!r}"
                            )
                        logical_to_physical[logical] = physical

                    incoming_logical = [str(key) for key in self]
                    if len(incoming_logical) != len(set(incoming_logical)):
                        raise ValueError("HDF5 merge contains duplicate logical keys")
                    for logical in incoming_logical:
                        if logical in logical_to_physical:
                            raise ValueError(
                                f"HDF5 merge logical key already exists: {logical!r}"
                            )

                    order = []
                    ordered = set()
                    for name in [*stored_order, *eligible]:
                        if name in eligible_set and name not in ordered:
                            order.append(name)
                            ordered.add(name)
                else:
                    keymap = read_hdf5_keymap(h5f)
                    order = read_hdf5_order(h5f) or root_names
                for key, ts in self.items():
                    safe = safe_hdf5_key(str(key))
                    name = unique_hdf5_key(safe, used=used)
                    if layout == LAYOUT_DATASET:
                        ts.write(h5f, format="hdf5", path=name)
                    else:
                        grp = h5f.create_group(name)
                        ts.write(grp, format="hdf5", path="data")
                    keymap[name] = str(key)
                    order.append(name)
                write_hdf5_manifest(
                    h5f,
                    kind=type(self).__name__,
                    layout=layout,
                    keymap=keymap,
                    order=order,
                )
            return target
        return super().write(target, *args, **kwargs)

    def plot_all(self, *args: Any, **kwargs: Any):
        """Alias for plot(). Plots all series."""
        return self.plot(*args, **kwargs)

    impute = _make_dict_map_method("impute", doc="Apply impute to each item.")

    def rolling_mean(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling mean to each item."""
        from gwexpy.timeseries.rolling import rolling_mean

        return rolling_mean(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_std(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ddof=0,
        ignore_nan=None,
    ):
        """Apply rolling std to each item."""
        from gwexpy.timeseries.rolling import rolling_std

        return rolling_std(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ddof=ddof,
            ignore_nan=ignore_nan,
        )

    def rolling_median(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling median to each item."""
        from gwexpy.timeseries.rolling import rolling_median

        return rolling_median(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_min(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling min to each item."""
        from gwexpy.timeseries.rolling import rolling_min

        return rolling_min(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_max(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling max to each item."""
        from gwexpy.timeseries.rolling import rolling_max

        return rolling_max(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def to_matrix(self, *, align="intersection", **kwargs):
        """Convert the dictionary to a `TimeSeriesMatrix` with alignment."""
        from gwexpy.timeseries.preprocess import align_timeseries_collection

        # Ensure consistent order (keys sorted) or specific
        # Dicts are ordered in modern python but keys() usually safe
        keys = list(self.keys())
        series_list = [self[k] for k in keys]

        vals, times, meta = align_timeseries_collection(
            series_list, how=align, **kwargs
        )

        # SeriesMatrix expects 3D usually (rows, cols, time) or checks last axis
        # vals: (samples, channels).
        # We create (channels, 1, samples).
        data = vals.T[:, None, :]

        from .matrix import TimeSeriesMatrix

        matrix = TimeSeriesMatrix(
            data,
            times=times,
            # meta might contain channel_names from original list (names of TS objects)
            # But converting dict to matrix usually implies keys become channel names?
            # User requirement: "preserve labels"
            # TimeSeries from dict usually inherit name from key if created via read?
            # Not always. We should force keys as names?
            # "Must preserve channel ordering from input."
        )
        exact_epochs: list[int | None] = [
            getattr(series, "_gwex_t0_gps_ns", None) for series in series_list
        ]
        if (
            all(epoch is not None for epoch in exact_epochs)
            and len(set(exact_epochs)) == 1
        ):
            setattr(matrix, "_gwex_t0_gps_ns", exact_epochs[0])
        exact_intervals: list[int | None] = [
            getattr(series, "_gwex_dt_gps_ns", None) for series in series_list
        ]
        if (
            all(interval is not None for interval in exact_intervals)
            and len(set(exact_intervals)) == 1
        ):
            setattr(matrix, "_gwex_dt_gps_ns", exact_intervals[0])
        # Assign channel names from keys
        matrix.channel_names = keys
        return matrix

    # ===============================
    # Batch Processing Methods (P1)
    # ===============================

    # --- Waveform Operations ---

    def crop(self, start=None, end=None, copy=False) -> TimeSeriesDict:
        """Crop each TimeSeries in the dict.

        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        Returns a new TimeSeriesDict.
        """
        from gwexpy.time import to_gps

        # Convert inputs to GPS if provided
        if start is not None:
            start = float(to_gps(start))
        if end is not None:
            end = float(to_gps(end))

        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.crop(start=start, end=end, copy=copy)
        provenance = getattr(self, "_gwexpy_io", None)
        if isinstance(provenance, dict):
            new_dict._gwexpy_io = {**provenance}
        return new_dict

    def append(self, other, copy=True, **kwargs) -> TimeSeriesDict:
        """Append another mapping of `TimeSeries` or a single `TimeSeries` to each item."""
        if isinstance(other, BaseTimeSeries):
            for ts in self.values():
                ts.append(other, **kwargs)
            return self

        # If 'copy' key is present in 'other' (can happen with some readers),
        # it will cause super().append to fail if 'copy' is not a TimeSeries.
        # We should filter it out if it's not a TimeSeries.
        if (
            hasattr(other, "pop")
            and "copy" in other
            and not isinstance(other["copy"], BaseTimeSeries)
        ):
            other.pop("copy")

        # Ensure we don't pass 'copy' twice if it's already in kwargs
        if "copy" in kwargs:
            copy = kwargs.pop("copy")

        return super().append(other, copy=copy, **kwargs)

    def prepend(self, *args, **kwargs) -> TimeSeriesDict:
        """Prepend to each TimeSeries in the dict (in-place).

        Returns self.
        """
        for ts in self.values():
            ts.prepend(*args, **kwargs)
        return self

    shift = _make_dict_map_method("shift", doc="Shift each TimeSeries in the dict.")

    gate = _make_dict_map_method("gate", doc="Gate each TimeSeries in the dict.")
    mask = _make_dict_map_method("mask", doc="Mask each TimeSeries in the dict.")

    # --- Signal Processing ---

    decimate = _make_dict_map_method(
        "decimate", doc="Decimate each TimeSeries in the dict."
    )
    filter = _make_dict_map_method("filter", doc="Filter each TimeSeries in the dict.")
    whiten = _make_dict_map_method("whiten", doc="Whiten each TimeSeries in the dict.")
    notch = _make_dict_map_method(
        "notch", doc="Notch filter each TimeSeries in the dict."
    )
    zpk = _make_dict_map_method(
        "zpk", doc="Apply ZPK filter to each TimeSeries in the dict."
    )
    detrend = _make_dict_map_method(
        "detrend", doc="Detrend each TimeSeries in the dict."
    )
    taper = _make_dict_map_method("taper", doc="Taper each TimeSeries in the dict.")

    # --- Spectral Conversion ---

    fft = _make_dict_map_method(
        "fft",
        doc="Apply FFT to each TimeSeries. Returns a FrequencySeriesDict.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesDict",
    )
    average_fft = _make_dict_map_method(
        "average_fft",
        doc="Apply average_fft to each TimeSeries. Returns a FrequencySeriesDict.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesDict",
    )

    # --- Statistics & Measurements ---

    def _apply_scalar_or_map(self, method_name, *args, **kwargs):
        """Apply a method that can return `TimeSeries` or scalar values.

        If TimeSeries -> return TimeSeriesDict.
        If scalar -> return pandas.Series.
        """
        import pandas as pd

        results: dict[Any, Any] = {}
        is_ts = False
        first = True

        for key, ts in self.items():
            method = getattr(ts, method_name)
            res = method(*args, **kwargs)

            if first:
                first = False
                # Check for TimeSeries-like structure
                if hasattr(res, "value") and hasattr(res, "dt"):
                    is_ts = True
                    results = self.__class__()

            if is_ts:
                # Ensure consistency
                if not (hasattr(res, "value") and hasattr(res, "dt")):
                    # Mixed types not supported cleanly here, defaulting to dict of objects
                    pass

            results[key] = res

        if is_ts:
            return results
        else:
            return pd.Series(results)

    def value_at(self, *args, **kwargs):
        """Get value at a specific time for each TimeSeries."""
        return self._apply_scalar_or_map("value_at", *args, **kwargs)

    def is_contiguous(self, *args, **kwargs):
        """Check contiguity with another object for each TimeSeries."""
        return self._apply_scalar_or_map("is_contiguous", *args, **kwargs)

    def skewness(self, **kwargs):
        """Compute skewness. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().skewness(**kwargs)

    def kurtosis(self, **kwargs):
        """Compute kurtosis. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().kurtosis(**kwargs)

    def mean(self, **kwargs):
        """Compute mean. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().mean(**kwargs)

    def std(self, **kwargs):
        """Compute standard deviation. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().std(**kwargs)

    def rms(self, **kwargs):
        """Compute root-mean-square. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().rms(**kwargs)

    def min(self, **kwargs):
        """Compute minimum. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().min(**kwargs)

    def max(self, **kwargs):
        """Compute maximum. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().max(**kwargs)

    def correlation(self, other=None, **kwargs):
        """Compute correlation. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().correlation(other=other, **kwargs)

    def mic(self, other, **kwargs):
        """Compute MIC. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().mic(other, **kwargs)

    def distance_correlation(self, other, **kwargs):
        """Compute distance correlation. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().distance_correlation(other, **kwargs)

    def pcc(self, other, **kwargs):
        """Compute Pearson correlation. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().pcc(other, **kwargs)

    def ktau(self, other, **kwargs):
        """Compute Kendall's tau. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().ktau(other, **kwargs)

    # --- State Analysis ---

    def state_segments(self, *args, **kwargs):
        """Run state_segments on each item (returns Series of SegmentLists)."""
        return self._apply_scalar_or_map("state_segments", *args, **kwargs)

    # --- Multivariate ---

    def pca(self, *args, **kwargs):
        """Perform PCA decomposition across channels."""
        return self.to_matrix().pca(*args, **kwargs)

    def ica(self, *args, **kwargs):
        """Perform ICA decomposition across channels."""
        return self.to_matrix().ica(*args, **kwargs)


class TimeSeriesList(PlotMixin, ListMapMixin, PhaseMethodsMixin, BaseTimeSeriesList):
    """A list of TimeSeries objects.

    `TimeSeriesList` is a specialized list designed to hold and manipulate
    multiple `TimeSeries` objects. It provides batch processing methods
    that operate on all entries at once.

    Parameters
    ----------
    *args
        An iterable of `TimeSeries` objects.

    Notes
    -----
    Key methods:

    .. autosummary::

       ~TimeSeriesList.plot
       ~TimeSeriesList.append
       ~TimeSeriesList.extend
       ~TimeSeriesList.csd_matrix
       ~TimeSeriesList.coherence_matrix

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries, TimeSeriesList
    >>> tsl = TimeSeriesList([TimeSeries([1, 2], sample_rate=1)])
    >>> tsl
    [<TimeSeries([1, 2],
                unit=Unit(dimensionless),
                t0=<Quantity 0. s>,
                dt=<Quantity 1. s>,
                name=None,
                channel=None)>]

    """

    def csd_matrix(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        hermitian=True,
        include_diagonal=True,
        **kwargs,
    ):
        """Compute Cross Spectral Density Matrix.

        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Other collection for cross-CSD.
        *args
            Positional arguments forwarded to `TimeSeries.csd`.
        fftlength, overlap, window :
            See TimeSeries.csd() arguments.
        hermitian : bool, default=True
            If True and other is None, compute only upper triangle and conjugate fill lower.
        include_diagonal : bool, default=True
            Must be True for CSD matrices; False raises ValueError because the
            diagonal is always the PSD.
        **kwargs
            Additional keyword arguments forwarded to `TimeSeries.csd`.

        Returns
        -------
        FrequencySeriesMatrix

        Notes
        -----
        The diagonal of a self-CSD matrix is always computed as the PSD. Any
        uncomputed elements are represented as complex NaN. The frequency axis
        is taken from the first computed element without alignment/truncation;
        dt and fftlength consistency is enforced before computation.

        """
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.csd_matrix",
        )
        return csd_matrix_from_collection(
            self,
            other,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            hermitian=hermitian,
            include_diagonal=include_diagonal,
            **kwargs,
        )

    def coherence_matrix(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        symmetric=True,
        include_diagonal=True,
        diagonal_value=1.0,
        **kwargs,
    ):
        """Compute Coherence Matrix.

        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Other collection.
        *args
            Positional arguments forwarded to `TimeSeries.coherence`.
        fftlength, overlap, window :
            See TimeSeries.coherence().
        symmetric : bool, default=True
            If True and other is None, compute only upper triangle and copy to lower.
        include_diagonal : bool, default=True
            Include diagonal.
        diagonal_value : float or None, default=1.0
            Value to fill diagonal if include_diagonal is True. If None, compute diagonal coherence.
        **kwargs
            Additional keyword arguments forwarded to `TimeSeries.coherence`.

        Returns
        -------
        FrequencySeriesMatrix

        Notes
        -----
        If include_diagonal is True and diagonal_value is not None, the
        diagonal is filled with that value without computation. If
        diagonal_value is None, the diagonal coherence is computed. Uncomputed
        elements are represented as NaN. The frequency axis is taken from the
        first computed element without alignment/truncation; dt and fftlength
        consistency is enforced before computation.

        """
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.coherence_matrix",
        )
        return coherence_matrix_from_collection(
            self,
            other,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            symmetric=symmetric,
            include_diagonal=include_diagonal,
            diagonal_value=diagonal_value,
            **kwargs,
        )

    def csd(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        hermitian=True,
        include_diagonal=True,
        **kwargs,
    ):
        """Compute CSD for each element or as a matrix depending on `other`."""
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.csd",
        )
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.csd_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.interop._registry import ConverterRegistry

            FrequencySeriesList = ConverterRegistry.get_constructor(
                "FrequencySeriesList"
            )
            new_list = FrequencySeriesList()
            for ts in self:
                list.append(
                    new_list,
                    ts.csd(
                        other,
                        fftlength=fftlength,
                        overlap=overlap,
                        window=window,
                        **kwargs,
                    ),
                )
            return new_list

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return csd_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        raise TypeError("other must be TimeSeries, TimeSeriesList/Dict, or None/'self'")

    def coherence(
        self,
        other=None,
        *args,
        fftlength=None,
        overlap=None,
        window="hann",
        symmetric=True,
        include_diagonal=True,
        diagonal_value=1.0,
        **kwargs,
    ):
        """Compute coherence for each element or as a matrix depending on `other`."""
        fftlength, overlap = _parse_fft_positional_args(
            args,
            fftlength=fftlength,
            overlap=overlap,
            method_name=f"{type(self).__name__}.coherence",
        )
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.coherence_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.interop._registry import ConverterRegistry

            FrequencySeriesList = ConverterRegistry.get_constructor(
                "FrequencySeriesList"
            )
            new_list = FrequencySeriesList()
            for ts in self:
                list.append(
                    new_list,
                    ts.coherence(
                        other,
                        fftlength=fftlength,
                        overlap=overlap,
                        window=window,
                        **kwargs,
                    ),
                )
            return new_list

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return coherence_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        raise TypeError("other must be TimeSeries, TimeSeriesList/Dict, or None/'self'")

    def impute(
        self, *, method="interpolate", limit=None, axis="time", max_gap=None, **kwargs
    ):
        """Impute missing data (NaNs) in each TimeSeries.

        Parameters
        ----------
        method : str, optional
            Imputation method ('interpolate', 'fill', etc.).
        limit : int, optional
            Maximum number of consecutive NaNs to fill.
        axis : str, optional
            Axis to impute along.
        max_gap : float, optional
            Maximum gap size to fill (in seconds).
        **kwargs
            Passed to TimeSeries.impute().

        Returns
        -------
        TimeSeriesList

        """
        new_list = self.__class__()
        for ts in self:
            list.append(
                new_list,
                ts.impute(
                    method=method, limit=limit, axis=axis, max_gap=max_gap, **kwargs
                ),
            )
        return new_list

    def rolling_mean(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling mean to each element."""
        from gwexpy.timeseries.rolling import rolling_mean

        return rolling_mean(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_std(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ddof=0,
        ignore_nan=None,
    ):
        """Apply rolling std to each element."""
        from gwexpy.timeseries.rolling import rolling_std

        return rolling_std(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ddof=ddof,
            ignore_nan=ignore_nan,
        )

    def rolling_median(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling median to each element."""
        from gwexpy.timeseries.rolling import rolling_median

        return rolling_median(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_min(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling min to each element."""
        from gwexpy.timeseries.rolling import rolling_min

        return rolling_min(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_max(
        self,
        window,
        *,
        center=False,
        min_count=1,
        nan_policy="omit",
        backend="auto",
        ignore_nan=None,
    ):
        """Apply rolling max to each element."""
        from gwexpy.timeseries.rolling import rolling_max

        return rolling_max(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def to_matrix(self, *, align="intersection", **kwargs):
        """Convert list to TimeSeriesMatrix with alignment.

        Parameters
        ----------
        align : str, optional
            Alignment strategy ('intersection', 'union', etc.). Default 'intersection'.
        **kwargs
            Additional arguments passed to alignment function.

        Returns
        -------
        TimeSeriesMatrix
            Matrix with all series aligned to common time axis.

        """
        from gwexpy.timeseries.matrix import TimeSeriesMatrix
        from gwexpy.timeseries.preprocess import align_timeseries_collection

        vals, times, meta = align_timeseries_collection(list(self), how=align, **kwargs)
        # Use names from metadata (from TS objects)
        names = meta.get("channel_names")

        data = vals.T[:, None, :]

        matrix = TimeSeriesMatrix(
            data,
            times=times,
        )
        exact_epochs: list[int | None] = [
            getattr(series, "_gwex_t0_gps_ns", None) for series in self
        ]
        if (
            all(epoch is not None for epoch in exact_epochs)
            and len(set(exact_epochs)) == 1
        ):
            setattr(matrix, "_gwex_t0_gps_ns", exact_epochs[0])
        exact_intervals: list[int | None] = [
            getattr(series, "_gwex_dt_gps_ns", None) for series in self
        ]
        if (
            all(interval is not None for interval in exact_intervals)
            and len(set(exact_intervals)) == 1
        ):
            setattr(matrix, "_gwex_dt_gps_ns", exact_intervals[0])
        if names:
            matrix.channel_names = names
        return matrix

    # ===============================
    # Batch Processing Methods (P1)
    # ===============================

    # --- Waveform Operations ---

    def crop(self, start=None, end=None, copy=False) -> TimeSeriesList:
        """Crop each ``TimeSeries`` in the list.

        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        Returns a new TimeSeriesList.
        """
        from gwexpy.time import to_gps

        # Convert inputs to GPS if provided
        if start is not None:
            start = float(to_gps(start))
        if end is not None:
            end = float(to_gps(end))

        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.crop(start=start, end=end, copy=copy))
        return new_list

    shift = _make_list_map_method("shift", doc="Shift each TimeSeries in the list.")
    gate = _make_list_map_method("gate", doc="Gate each TimeSeries in the list.")
    mask = _make_list_map_method("mask", doc="Mask each TimeSeries in the list.")

    # --- Signal Processing ---

    resample = _make_list_map_method(
        "resample", doc="Resample each TimeSeries in the list."
    )
    decimate = _make_list_map_method(
        "decimate", doc="Decimate each TimeSeries in the list."
    )
    filter = _make_list_map_method("filter", doc="Filter each TimeSeries in the list.")
    whiten = _make_list_map_method("whiten", doc="Whiten each TimeSeries in the list.")
    notch = _make_list_map_method(
        "notch", doc="Notch filter each TimeSeries in the list."
    )
    zpk = _make_list_map_method("zpk", doc="ZPK filter each TimeSeries in the list.")
    detrend = _make_list_map_method(
        "detrend", doc="Detrend each TimeSeries in the list."
    )
    taper = _make_list_map_method("taper", doc="Taper each TimeSeries in the list.")
    hilbert = _make_list_map_method(
        "hilbert", doc="Apply Hilbert transform to each item."
    )
    envelope = _make_list_map_method("envelope", doc="Apply envelope to each item.")
    instantaneous_phase = _make_list_map_method(
        "instantaneous_phase", doc="Apply instantaneous_phase to each item."
    )
    unwrap_phase = _make_list_map_method(
        "unwrap_phase", doc="Apply unwrap_phase to each item."
    )
    instantaneous_frequency = _make_list_map_method(
        "instantaneous_frequency", doc="Apply instantaneous_frequency to each item."
    )
    mix_down = _make_list_map_method("mix_down", doc="Apply mix_down to each item.")
    baseband = _make_list_map_method("baseband", doc="Apply baseband to each item.")
    heterodyne = _make_list_map_method(
        "heterodyne", doc="Apply heterodyne to each item."
    )

    def lock_in(self, *args, **kwargs):
        """Apply lock_in to each item."""
        if not self:
            return self.__class__()

        # Peek first
        first_res = self[0].lock_in(*args, **kwargs)
        if isinstance(first_res, tuple):
            res_lists = tuple(self.__class__() for _ in first_res)
            for ts in self:
                res = ts.lock_in(*args, **kwargs)
                for i, val in enumerate(res):
                    list.append(res_lists[i], val)
            return res_lists
        else:
            new_list = self.__class__()
            for ts in self:
                list.append(new_list, ts.lock_in(*args, **kwargs))
            return new_list

    # --- Spectral Conversion ---

    fft = _make_list_map_method(
        "fft",
        doc="Apply FFT to each TimeSeries. Returns a FrequencySeriesList.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesList",
    )
    average_fft = _make_list_map_method(
        "average_fft",
        doc="Apply average_fft to each TimeSeries. Returns a FrequencySeriesList.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesList",
    )
    psd = _make_list_map_method(
        "psd",
        doc="Compute PSD for each TimeSeries. Returns a FrequencySeriesList.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesList",
    )
    asd = _make_list_map_method(
        "asd",
        doc="Compute ASD for each TimeSeries. Returns a FrequencySeriesList.",
        result_class_path="gwexpy.frequencyseries.FrequencySeriesList",
    )
    spectrogram = _make_list_map_method(
        "spectrogram",
        doc="Compute spectrogram for each TimeSeries. Returns a SpectrogramList.",
        result_class_path="gwexpy.spectrogram.SpectrogramList",
    )
    spectrogram2 = _make_list_map_method(
        "spectrogram2",
        doc="Compute spectrogram2 for each TimeSeries. Returns a SpectrogramList.",
        result_class_path="gwexpy.spectrogram.SpectrogramList",
    )
    q_transform = _make_list_map_method(
        "q_transform",
        doc="Compute Q-transform for each TimeSeries. Returns a SpectrogramList.",
        result_class_path="gwexpy.spectrogram.SpectrogramList",
    )

    # --- Statistics & Measurements ---

    def _apply_scalar_or_map(self, method_name, *args, **kwargs):
        """Apply a method that can return ``TimeSeries`` or scalar values.

        If TimeSeries -> return TimeSeriesList.
        If scalar -> return list.
        """
        results: list[Any] | TimeSeriesList = []
        is_ts = False
        first = True

        for ts in self:
            method = getattr(ts, method_name)
            res = method(*args, **kwargs)

            if first:
                first = False
                if hasattr(res, "value") and hasattr(res, "dt"):
                    is_ts = True
                    results = self.__class__()

            if is_ts:
                # Type check?
                pass

            if isinstance(results, self.__class__):
                list.append(results, res)
            else:
                list.append(results, res)

        return results

    def value_at(self, *args, **kwargs):
        """Get value at a specific time for each TimeSeries."""
        return self._apply_scalar_or_map("value_at", *args, **kwargs)

    def is_contiguous(self, *args, **kwargs):
        """Check contiguity with another object for each TimeSeries."""
        return self._apply_scalar_or_map("is_contiguous", *args, **kwargs)

    def skewness(self, **kwargs):
        """Compute skewness. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().skewness(**kwargs)

    def kurtosis(self, **kwargs):
        """Compute kurtosis. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().kurtosis(**kwargs)

    def mean(self, **kwargs):
        """Compute mean. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().mean(**kwargs)

    def std(self, **kwargs):
        """Compute standard deviation. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().std(**kwargs)

    def rms(self, **kwargs):
        """Compute root-mean-square. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().rms(**kwargs)

    def min(self, **kwargs):
        """Compute minimum. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().min(**kwargs)

    def max(self, **kwargs):
        """Compute maximum. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().max(**kwargs)

    def correlation(self, other=None, **kwargs):
        """Compute correlation. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().correlation(other=other, **kwargs)

    def mic(self, other, **kwargs):
        """Compute MIC. Vectorized via TimeSeriesMatrix."""
        return self.to_matrix().mic(other, **kwargs)

    # --- State Analysis ---

    # --- Multivariate ---

    def to_pandas(self, **kwargs):
        """Convert a ``TimeSeriesList`` to a pandas ``DataFrame``.

        Each element becomes a column.
        ASSUMES common time axis.
        """
        import pandas as pd

        data = {}
        for i, ts in enumerate(self):
            name = ts.name or f"series_{i}"
            if hasattr(ts, "to_pandas"):
                s = ts.to_pandas()
            else:
                s = pd.Series(ts.value, index=ts.times.value)

            data[name] = s

        return pd.DataFrame(data)

    def to_tmultigraph(self, name: str | None = None) -> Any:
        """Convert to ROOT TMultiGraph."""
        from gwexpy.interop import to_tmultigraph

        return to_tmultigraph(self, name=name)

    @classmethod
    def read(cls, source, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Read a ``TimeSeriesList`` from a supported source."""
        from gwexpy._bootstrap import register_all

        register_all()

        fmt = kwargs.get("format")
        # Both branches below re-read each entry themselves rather than going
        # through a registered reader, and neither forwarded the bounds — so a
        # windowed read returned every sample (issue #611).  A list has no span
        # of its own, so the window is applied per entry.
        start = kwargs.get("start")
        end = kwargs.get("end")
        try:
            p = Path(source)
        except TypeError:
            p = None
        if p is not None and p.is_dir() and (fmt in (None, "csv", "txt")):
            from gwexpy.io.collection_dir import read_collection_dir
            from gwexpy.io.utils import apply_unit

            TimeSeries = cast(Any, ConverterRegistry.get_constructor("TimeSeries"))

            _, items = read_collection_dir(
                p,
                expected_kind="TimeSeriesList",
                entry_format=fmt,
                reader=lambda path, f: TimeSeries.read(path, format=f),
            )
            dir_items = []
            for _, v, meta in items:
                unit = meta.get("unit")
                v = apply_unit(v, unit) if unit else v
                dir_items.append(apply_time_selection(v, start, end))
            return cls(*dir_items)
        if fmt in ("hdf5", "h5", "hdf"):
            TimeSeries = cast(Any, ConverterRegistry.get_constructor("TimeSeries"))

            with h5py.File(source, "r") as h5f:
                layout = detect_hdf5_layout(h5f)
                order = read_hdf5_order(h5f) or list(h5f.keys())
                out_items: list[Any] = []
                if layout == LAYOUT_DATASET or layout is None:
                    for ds_name in order:
                        try:
                            ts = TimeSeries.read(h5f, format="hdf5", path=ds_name)
                        except (KeyError, ValueError, TypeError, OSError) as e:
                            logger.debug("Skipping dataset %s: %s", ds_name, e)
                            continue
                        out_items.append(apply_time_selection(ts, start, end))
                    return cls(*out_items)
                if layout == LAYOUT_GROUP:
                    for grp_name in order:
                        try:
                            grp = h5f[grp_name]
                            ts = TimeSeries.read(grp, format="hdf5", path="data")
                        except (KeyError, ValueError, TypeError, OSError):
                            try:
                                ts = TimeSeries.read(grp, format="hdf5")
                            except (KeyError, ValueError, TypeError, OSError) as e2:
                                logger.debug("Skipping group %s: %s", grp_name, e2)
                                continue
                        out_items.append(apply_time_selection(ts, start, end))
                    return cls(*out_items)
        raise TypeError(
            "TimeSeriesList.read currently supports only directory sources for csv/txt"
        )

    def __reduce_ex__(self, protocol: SupportsIndex):
        from gwpy.timeseries import TimeSeriesList as GwpyTimeSeriesList

        return (GwpyTimeSeriesList, tuple(self))

    def write(self, target: str, *args: Any, **kwargs: Any) -> Any:
        """Write TimeSeriesList to file (HDF5, ROOT, etc.)."""
        from gwexpy._bootstrap import register_all

        register_all()

        fmt = kwargs.get("format")
        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
            from gwexpy.interop.root_ import write_root_file

            return write_root_file(self, target, **kwargs)
        if fmt in ("csv", "txt"):
            from gwexpy.io.collection_dir import write_collection_dir

            overwrite = bool(kwargs.pop("overwrite", False))
            pairs: list[tuple[str, Any]] = []
            for i, ts in enumerate(self):
                key = ts.name or f"series_{i}"
                pairs.append((key, ts))
            return write_collection_dir(
                target,
                kind="TimeSeriesList",
                entry_format=str(fmt),
                entries=pairs,
                writer=lambda ts, path, f: ts.write(path, format=f),
                meta_getter=lambda ts: {"unit": str(getattr(ts, "unit", "") or "")},
                overwrite=overwrite,
            )
        if fmt in ("hdf5", "h5", "hdf"):
            overwrite = bool(kwargs.pop("overwrite", False))
            mode = kwargs.pop("mode", None)
            layout = normalize_layout(kwargs.pop("layout", "gwpy"))
            used: set[str] = set()
            order: list[str] = []
            with ensure_hdf5_file(target, mode=mode, overwrite=overwrite) as h5f:
                for i, ts in enumerate(self):
                    key = safe_hdf5_key(str(i))
                    name = unique_hdf5_key(key, used=used)
                    if layout == LAYOUT_DATASET:
                        ts.write(h5f, format="hdf5", path=name)
                    else:
                        grp = h5f.create_group(name)
                        ts.write(grp, format="hdf5", path="data")
                    order.append(name)
                write_hdf5_manifest(
                    h5f,
                    kind=type(self).__name__,
                    layout=layout,
                    keymap={},
                    order=order,
                )
            return target
        from astropy.io import registry

        return registry.write(self, target, *args, **kwargs)

    def pca(self, *args, **kwargs):
        """Perform PCA decomposition across channels."""
        return self.to_matrix().pca(*args, **kwargs)

    def ica(self, *args, **kwargs):
        """Perform ICA decomposition across channels."""
        return self.to_matrix().ica(*args, **kwargs)

    def plot_all(self, *args: Any, **kwargs: Any):
        """Alias for plot(). Plots all series."""
        return self.plot(*args, **kwargs)

    histogram = _make_list_map_method(
        "histogram",
        doc="Compute Histogram for each TimeSeries. Returns a HistogramList.",
        result_class_path="gwexpy.histogram.collections.HistogramList",
    )

    radian = _make_list_map_method(
        "radian", doc="Compute instantaneous phase (in radians) of each item."
    )
    degree = _make_list_map_method(
        "degree", doc="Compute instantaneous phase (in degrees) of each item."
    )

    # phase() and angle() are provided by PhaseMethodsMixin


def _patch_gwpy_collections() -> None:
    patches = (
        (BaseTimeSeriesDict, TimeSeriesDict, ("csd_matrix", "coherence_matrix")),
        (BaseTimeSeriesList, TimeSeriesList, ("csd_matrix", "coherence_matrix")),
    )
    for base_cls, impl_cls, method_names in patches:
        for name in method_names:
            if not hasattr(base_cls, name):
                setattr(base_cls, name, getattr(impl_cls, name))


_patch_gwpy_collections()

_timeseriesdict_read = cast(Any, TimeSeriesDict.read).__func__
_timeseriesdict_read.__signature__ = _gwf_parallel_read_signature(_timeseriesdict_read)
_timeseriesdict_read.__doc__ = f"{_timeseriesdict_read.__doc__}{_GWF_PARALLEL_HELP}"
