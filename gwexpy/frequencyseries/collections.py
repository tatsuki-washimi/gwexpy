from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, SupportsIndex, TypeVar

import h5py
import numpy as np

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
from gwexpy.types.mixin._collection_mixin import (
    DictMapMixin,
    ListMapMixin,
    _make_dict_map_method,
    _make_dict_plain_method,
    _make_list_map_method,
)
from gwexpy.types.mixin._plot_mixin import PlotMixin

from .frequencyseries import FrequencySeries, as_series_dict_class

_FS = TypeVar("_FS", bound=FrequencySeries)
logger = logging.getLogger(__name__)


class FrequencySeriesBaseDict(OrderedDict[str, _FS]):
    """Ordered mapping container for `FrequencySeries` objects.

    This is a lightweight GWpy-inspired container:
    - enforces `EntryClass` on insertion/update
    - provides map-style helpers (`copy`, `crop`, `plot`)
    - default values for setdefault() must be FrequencySeries (None not allowed)

    Non-trivial operations (I/O, fetching, axis coercion, joins) are
    intentionally out-of-scope for this MVP.
    """

    EntryClass = FrequencySeries

    @property
    def span(self):
        """Frequency extent across all elements (based on xspan)."""
        from gwpy.segments import SegmentList

        span = SegmentList([val.xspan for val in self.values()])
        try:
            return span.extent()
        except ValueError as exc:  # empty
            exc.args = (f"cannot calculate span for empty {type(self).__name__}",)
            raise

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    def __setitem__(self, key: str, value: _FS) -> None:
        if not isinstance(value, self.EntryClass):
            raise TypeError(
                f"Cannot set key '{key}' to type '{type(value).__name__}' in {type(self).__name__}"
            )
        super().__setitem__(key, value)

    def setdefault(self, key: str, default: _FS | None = None) -> _FS:  # type: ignore[override]
        if key in self:
            return self[key]
        if default is None:
            raise TypeError(
                f"Cannot set default None for {type(self).__name__}; expected {self.EntryClass.__name__}"
            )
        if not isinstance(default, self.EntryClass):
            raise TypeError(
                f"Cannot set default type '{type(default).__name__}' in {type(self).__name__}"
            )
        self[key] = default
        return default

    def copy(self) -> FrequencySeriesBaseDict[_FS]:
        new = self.__class__()
        for key, val in self.items():
            new[key] = val.copy()
        return new

    def crop(
        self, start: Any = None, end: Any = None, copy: bool = False
    ) -> FrequencySeriesBaseDict[_FS]:
        for key, val in list(self.items()):
            self[key] = val.crop(start=start, end=end, copy=copy)
        return self

    def plot(
        self,
        label: str = "key",
        method: str = "plot",
        figsize: Any | None = None,
        **kwargs: Any,
    ):
        """
        Plot data.

        Parameters
        ----------
        label : str, optional
            labelling method, one of

            - ``'key'``: use dictionary key (default)
            - ``'name'``: use ``name`` attribute of each item
        method : str, optional
            method of :class:`~gwpy.plot.Plot` to call, default: ``'plot'``
        figsize : tuple, optional
            (width, height) tuple in inches
        **kwargs
            other keyword arguments passed to the plot method
        """
        from gwexpy.interop._registry import ConverterRegistry

        Plot = ConverterRegistry.get_constructor("Plot")

        kwargs = dict(kwargs)
        separate = kwargs.get("separate", False)
        if figsize is not None:
            kwargs.setdefault("figsize", figsize)
        kwargs.update({"label": label, "method": method})

        # We pass the dict directly if separate=True (or False),
        # but gwexpy.plot.Plot can handle list unpacking now.
        # To maintain label logic ("key" vs "name"), we might need
        # to adjust the input items or labels beforehand if not handled by Plot.
        # However, gwpy.plot.Plot handles dicts by default for labeling if separate=False.

        # If separate=True, we want subplots. gwexpy.plot.Plot handles this via defaults now.

        # For 'key' labeling, we rely on the input being a dict/values iteration.

        if separate:
            # If separate, Plot(...) with *values usually works in gwpy
            # gwexpy Plot now supports unpacking
            plot = Plot(self, **kwargs)
        else:
            plot = Plot(self, **kwargs)

        artmap = {"plot": "lines", "scatter": "collections"}
        artists = [
            artist
            for ax in plot.axes
            for artist in getattr(ax, artmap.get(method, "lines"))
        ]

        label_key = label.lower()
        for key, artist in zip(self, artists):
            if label_key == "name":
                lab = self[key].name
            elif label_key == "key":
                lab = key
            else:
                lab = label
            artist.set_label(lab)

        return plot

    def plot_all(self, *args: Any, **kwargs: Any):
        """Alias for plot(). Plots all series in the dict."""
        return self.plot(*args, **kwargs)

    @classmethod
    def read(cls, source, *args, **kwargs):
        """
        Read data into a `FrequencySeriesDict`.

        Parameters
        ----------
        source : str, file-like
            Source of data, either a file path or a file-like object.
        *args
            Arguments passed to the underlying reader.
        **kwargs
            Keyword arguments passed to the underlying reader.

        Returns
        -------
        FrequencySeriesDict
            A new dict containing the data read from the source.
        """
        fmt = kwargs.get("format")
        try:
            p = Path(source)
        except TypeError:
            p = None
        if p is not None and p.is_dir() and (fmt in (None, "csv", "txt")):
            from gwexpy.io.collection_dir import read_collection_dir
            from gwexpy.io.utils import apply_unit

            _, items = read_collection_dir(
                p,
                expected_kind=cls.__name__,
                entry_format=fmt,
                reader=lambda path, f: FrequencySeries.read(path, format=f),
            )
            out = cls()
            for k, v, meta in items:
                unit = meta.get("unit")
                v = apply_unit(v, unit) if unit else v
                out[k] = v
            return out
        if fmt in ("hdf5", "h5", "hdf"):
            with h5py.File(source, "r") as h5f:
                layout = detect_hdf5_layout(h5f)
                keymap = read_hdf5_keymap(h5f)
                order = read_hdf5_order(h5f)
                keys = order or list(h5f.keys())
                out = cls()
                if layout == LAYOUT_DATASET or layout is None:
                    for ds_name in keys:
                        try:
                            fs = FrequencySeries.read(h5f, format="hdf5", path=ds_name)
                        except (KeyError, ValueError, TypeError, OSError) as e:
                            logger.debug("Skipping dataset %s: %s", ds_name, e)
                            continue
                        orig_key = keymap.get(ds_name, ds_name)
                        out[orig_key] = fs
                    return out
                if layout == LAYOUT_GROUP:
                    for grp_name in keys:
                        try:
                            grp = h5f[grp_name]
                            fs = FrequencySeries.read(grp, format="hdf5", path="data")
                        except (KeyError, ValueError, TypeError, OSError):
                            try:
                                fs = FrequencySeries.read(grp, format="hdf5")
                            except (KeyError, ValueError, TypeError, OSError) as e2:
                                logger.debug("Skipping group %s: %s", grp_name, e2)
                                continue
                        orig_key = keymap.get(grp_name, grp_name)
                        out[orig_key] = fs
                    return out
        from astropy.io import registry

        return registry.read(cls, source, *args, **kwargs)

    def __reduce_ex__(self, protocol: SupportsIndex):
        return (dict, (dict(self),))

    def write(self, target, *args, **kwargs):
        from astropy.io import registry

        return registry.write(self, target, *args, **kwargs)


@as_series_dict_class(FrequencySeries)
class FrequencySeriesDict(DictMapMixin, FrequencySeriesBaseDict[FrequencySeries]):
    """Ordered mapping of `FrequencySeries` objects keyed by label."""

    EntryClass = FrequencySeries

    # ===============================
    # 1. Axis & Edit Operations
    # ===============================

    def crop(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Crop each FrequencySeries in the dict.
        In-place operation (GWpy-compatible). Returns self.
        """
        super().crop(*args, **kwargs)
        return self

    pad = _make_dict_map_method("pad", doc="Pad each FrequencySeries in the dict.")
    interpolate = _make_dict_map_method(
        "interpolate", doc="Interpolate each FrequencySeries in the dict."
    )

    # --- In-place Element Operations ---

    def append(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Append to each FrequencySeries in the dict (in-place).
        Returns self.
        """
        for fs in self.values():
            fs.append(*args, **kwargs)
        return self

    def prepend(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Prepend to each FrequencySeries in the dict (in-place).
        Returns self.
        """
        for fs in self.values():
            fs.prepend(*args, **kwargs)
        return self

    # ===============================
    # 2. Filter & Response
    # ===============================

    zpk = _make_dict_map_method("zpk", doc="Apply ZPK filter to each FrequencySeries.")
    filter = _make_dict_map_method(
        "filter", doc="Apply filter to each FrequencySeries."
    )
    apply_response = _make_dict_map_method(
        "apply_response", doc="Apply response to each FrequencySeries."
    )

    # ===============================
    # 3. Analysis & Conversion
    # ===============================

    phase = _make_dict_map_method("phase", doc="Compute phase of each FrequencySeries.")

    def angle(self, *args, **kwargs) -> FrequencySeriesDict:
        """Alias for phase(). Returns a new FrequencySeriesDict."""
        return self.phase(*args, **kwargs)

    degree = _make_dict_map_method(
        "degree", doc="Compute phase (in degrees) of each FrequencySeries."
    )
    to_db = _make_dict_map_method("to_db", doc="Convert each FrequencySeries to dB.")
    differentiate_time = _make_dict_map_method(
        "differentiate_time", doc="Apply time differentiation to each item."
    )
    integrate_time = _make_dict_map_method(
        "integrate_time", doc="Apply time integration to each item."
    )
    group_delay = _make_dict_map_method(
        "group_delay", doc="Compute group delay of each item."
    )
    smooth = _make_dict_map_method("smooth", doc="Smooth each FrequencySeries.")
    rebin = _make_dict_map_method(
        "rebin", doc="Rebin each FrequencySeries in the dict."
    )

    # ===============================
    # 4. Time Domain Conversion
    # ===============================

    ifft = _make_dict_map_method(
        "ifft",
        doc="Compute IFFT of each FrequencySeries. Returns a TimeSeriesDict.",
        result_class_path="gwexpy.timeseries.TimeSeriesDict",
    )

    # ===============================
    # 5. External Library Interop
    # ===============================

    to_control_frd = _make_dict_plain_method(
        "to_control_frd", doc="Convert each item to control.FRD. Returns a dict."
    )
    to_torch = _make_dict_plain_method(
        "to_torch", doc="Convert each item to torch.Tensor. Returns a dict."
    )
    to_tensorflow = _make_dict_plain_method(
        "to_tensorflow", doc="Convert each item to tensorflow.Tensor. Returns a dict."
    )
    to_jax = _make_dict_plain_method(
        "to_jax", doc="Convert each item to jax.Array. Returns a dict."
    )
    to_cupy = _make_dict_plain_method(
        "to_cupy", doc="Convert each item to cupy.ndarray. Returns a dict."
    )

    # ===============================
    # 6. Aggregation
    # ===============================

    def to_pandas(self, **kwargs):
        """
        Convert to pandas.DataFrame.
        Keys become columns.
        """
        import pandas as pd

        data = {}
        for key, fs in self.items():
            # Extract Series with index from FrequencySeries
            if hasattr(fs, "to_pandas"):
                # to_pandas(index="frequency") ensures index is frequency
                s = fs.to_pandas(index="frequency", copy=False)
                # s could be Series or DataFrame (gwpy usually returns Series for 1D)
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                elif not isinstance(s, pd.Series):  # Fallback
                    s = pd.Series(fs.value, index=fs.frequencies.value)
            else:
                s = pd.Series(fs.value, index=fs.frequencies.value)
            data[key] = s

        return pd.DataFrame(data)

    def to_xarray(self):
        """
        Convert to xarray.Dataset.
        Keys become data variables.
        """
        import xarray as xr

        ds = xr.Dataset()
        for key, fs in self.items():
            # FrequencySeries.to_xarray returns DataArray
            ds[key] = fs.to_xarray()
        return ds

    def to_matrix(self):
        """Convert this FrequencySeriesDict to a FrequencySeriesMatrix (Nx1)."""
        from gwexpy.types.metadata import MetaData, MetaDataMatrix

        from .matrix import FrequencySeriesMatrix

        keys = list(self.keys())
        if not keys:
            return FrequencySeriesMatrix(np.empty((0, 0, 0)))

        first = self[keys[0]]
        n_row = len(keys)
        n_samp = len(first)
        data = np.empty((n_row, 1, n_samp), dtype=first.value.dtype)

        meta_arr = np.empty((n_row, 1), dtype=object)
        for i, key in enumerate(keys):
            fs = self[key]
            if len(fs) != n_samp:
                raise ValueError(
                    f"FrequencySeriesDict items must have same length to convert to matrix. '{key}' has {len(fs)}, expected {n_samp}"
                )
            data[i, 0] = fs.value
            meta_arr[i, 0] = MetaData(unit=fs.unit, name=fs.name, channel=fs.channel)

        return FrequencySeriesMatrix(
            data,
            frequencies=first.frequencies,
            rows=keys,
            cols=["value"],
            meta=MetaDataMatrix(meta_arr),
            epoch=first.epoch.gps if hasattr(first.epoch, "gps") else first.epoch,
        )

    def to_tmultigraph(self, name: str | None = None) -> Any:
        """Convert to ROOT TMultiGraph."""
        from gwexpy.interop import to_tmultigraph

        return to_tmultigraph(self, name=name)

    # ===============================
    # 7. Finesse 3 Interop
    # ===============================

    @classmethod
    def from_finesse_frequency_response(
        cls, sol: Any, *, unit: Any | None = None
    ) -> FrequencySeriesDict:
        """Create from finesse FrequencyResponseSolution.

        Returns a dict keyed by ``"output -> input"`` for all DOF pairs.

        Parameters
        ----------
        sol : finesse.analysis.actions.lti.FrequencyResponseSolution
            The frequency response solution from a Finesse 3 simulation.
        unit : str or astropy.units.Unit, optional
            Unit to assign to the data.
        """
        from gwexpy.interop import from_finesse_frequency_response

        return from_finesse_frequency_response(cls, sol, unit=unit)

    @classmethod
    def from_finesse_noise(
        cls, sol: Any, *, output: Any | None = None, unit: Any | None = None
    ) -> FrequencySeriesDict:
        """Create from finesse NoiseProjectionSolution.

        Returns a dict keyed by ``"output: noise_source"`` strings.

        Parameters
        ----------
        sol : finesse.analysis.actions.noise.NoiseProjectionSolution
            The noise projection solution from a Finesse 3 simulation.
        output : str or object, optional
            Output node name. If *None*, all outputs are included.
        unit : str or astropy.units.Unit, optional
            Unit to assign to the data (e.g., ``"m/sqrt(Hz)"``).
        """
        from gwexpy.interop import from_finesse_noise

        return from_finesse_noise(cls, sol, output=output, unit=unit)

    # ===============================
    # 8. PySpice Interop
    # ===============================

    @classmethod
    def from_pyspice_ac(
        cls, analysis: Any, *, unit: Any | None = None
    ) -> FrequencySeriesDict:
        """Create from a PySpice AcAnalysis.

        Returns a dict keyed by signal name for all nodes and branches.

        Parameters
        ----------
        analysis : PySpice.Spice.Simulation.AcAnalysis
            The AC analysis result from a PySpice simulation.
        unit : str or astropy.units.Unit, optional
            Unit to assign to the data.
        """
        from gwexpy.interop import from_pyspice_ac

        return from_pyspice_ac(cls, analysis, unit=unit)

    @classmethod
    def from_pyspice_noise(
        cls, analysis: Any, *, unit: Any | None = None
    ) -> FrequencySeriesDict:
        """Create from a PySpice NoiseAnalysis.

        Returns a dict keyed by signal name for all noise nodes.

        Parameters
        ----------
        analysis : PySpice.Spice.Simulation.NoiseAnalysis
            The noise analysis result from a PySpice simulation.
        unit : str or astropy.units.Unit, optional
            Unit to assign to the data.
        """
        from gwexpy.interop import from_pyspice_noise

        return from_pyspice_noise(cls, analysis, unit=unit)

    # ===============================
    # 9. scikit-rf Interop
    # ===============================

    @classmethod
    def from_skrf_network(
        cls,
        ntwk: Any,
        *,
        parameter: str = "s",
        unit: Any | None = None,
    ) -> FrequencySeriesDict:
        """Create from a scikit-rf Network.

        Returns a dict keyed by port-pair labels (e.g. ``"S11"``, ``"S21"``).

        Parameters
        ----------
        ntwk : skrf.Network
            The scikit-rf Network object.
        parameter : str, default ``"s"``
            Which network parameter to extract (``"s"``, ``"z"``, ``"y"``,
            ``"a"``, ``"t"``, or ``"h"``).
        unit : str or astropy.units.Unit, optional
            Unit to assign to the data.
        """
        from gwexpy.interop import from_skrf_network

        return from_skrf_network(cls, ntwk, parameter=parameter, unit=unit)

    def write(self, target: str, *args: Any, **kwargs: Any) -> Any:
        """Write dict to file (HDF5, ROOT, etc.)."""
        fmt = kwargs.get("format")
        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
            from gwexpy.interop.root_ import write_root_file

            return write_root_file(self, target, **kwargs)
        if fmt in ("csv", "txt"):
            from gwexpy.io.collection_dir import write_collection_dir

            overwrite = bool(kwargs.pop("overwrite", False))
            return write_collection_dir(
                target,
                kind=type(self).__name__,
                entry_format=str(fmt),
                entries=list(self.items()),
                writer=lambda fs, path, f: fs.write(path, format=f),
                meta_getter=lambda fs: {"unit": str(getattr(fs, "unit", "") or "")},
                overwrite=overwrite,
            )
        if fmt in ("hdf5", "h5", "hdf"):
            overwrite = bool(kwargs.pop("overwrite", False))
            mode = kwargs.pop("mode", None)
            layout = normalize_layout(kwargs.pop("layout", "gwpy"))
            used: set[str] = set()
            keymap: dict[str, str] = {}
            order: list[str] = []
            with ensure_hdf5_file(target, mode=mode, overwrite=overwrite) as h5f:
                for key, fs in self.items():
                    safe = safe_hdf5_key(str(key))
                    name = unique_hdf5_key(safe, used=used)
                    if layout == LAYOUT_DATASET:
                        fs.write(h5f, format="hdf5", path=name)
                    else:
                        grp = h5f.create_group(name)
                        fs.write(grp, format="hdf5", path="data")
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
        from astropy.io import registry

        return registry.write(self, target, *args, **kwargs)


class FrequencySeriesBaseList(PlotMixin, list[_FS]):
    """List container for `FrequencySeries` objects with type enforcement."""

    EntryClass = FrequencySeries

    def __init__(self, *items: _FS | Iterable[_FS]):
        super().__init__()
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            for item in items[0]:
                self.append(item)
        else:
            for item in items:
                self.append(item)  # type: ignore[arg-type]

    @property
    def segments(self):
        """Frequency spans of each element (xspan)."""
        from gwpy.segments import SegmentList

        return SegmentList([item.xspan for item in self])

    def _validate(self, item: Any, *, op: str) -> None:
        if not isinstance(item, self.EntryClass):
            raise TypeError(
                f"Cannot {op} type '{type(item).__name__}' to {type(self).__name__}"
            )

    def append(self, item: _FS):  # type: ignore[override]
        self._validate(item, op="append")
        super().append(item)
        return self

    def extend(self, items: Iterable[_FS]) -> None:  # type: ignore[override]
        validated = self.__class__(*items)
        super().extend(validated)

    def insert(self, index: int, item: _FS) -> None:  # type: ignore[override]
        self._validate(item, op="insert")
        super().insert(index, item)

    def __setitem__(self, index, item) -> None:  # type: ignore[override]
        if isinstance(index, slice):
            validated = self.__class__(*item)
            super().__setitem__(index, validated)
            return
        self._validate(item, op="set")
        super().__setitem__(index, item)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return self.__class__(*super().__getitem__(index))
        return super().__getitem__(index)

    def plot(self, **kwargs: Any) -> Any:
        """Plot this collection using gwexpy Plot."""
        from gwexpy.interop._registry import ConverterRegistry

        Plot = ConverterRegistry.get_constructor("Plot")
        return Plot(self, **kwargs)

    def copy(self) -> FrequencySeriesBaseList[_FS]:
        return self.__class__(*(item.copy() for item in self))

    def plot_all(self, *args: Any, **kwargs: Any):
        """Alias for plot(). Plots all series."""
        return self.plot(*args, **kwargs)

    @classmethod
    def read(cls, source, *args, **kwargs):
        """
        Read data into a `FrequencySeriesList`.

        Parameters
        ----------
        source : str, file-like
            Source of data, either a file path or a file-like object.
        *args
            Arguments passed to the underlying reader.
        **kwargs
            Keyword arguments passed to the underlying reader.

        Returns
        -------
        FrequencySeriesList
            A new list containing the data read from the source.
        """
        fmt = kwargs.get("format")
        try:
            p = Path(source)
        except TypeError:
            p = None
        if p is not None and p.is_dir() and (fmt in (None, "csv", "txt")):
            from gwexpy.io.collection_dir import read_collection_dir
            from gwexpy.io.utils import apply_unit

            _, items = read_collection_dir(
                p,
                expected_kind=cls.__name__,
                entry_format=fmt,
                reader=lambda path, f: FrequencySeries.read(path, format=f),
            )
            dir_items = []
            for _, v, meta in items:
                unit = meta.get("unit")
                v = apply_unit(v, unit) if unit else v
                dir_items.append(v)
            return cls(dir_items)
        if fmt in ("hdf5", "h5", "hdf"):
            with h5py.File(source, "r") as h5f:
                layout = detect_hdf5_layout(h5f)
                order = read_hdf5_order(h5f) or list(h5f.keys())
                out_items: list[FrequencySeries] = []
                if layout == LAYOUT_DATASET or layout is None:
                    for ds_name in order:
                        try:
                            fs = FrequencySeries.read(h5f, format="hdf5", path=ds_name)
                        except (KeyError, ValueError, TypeError, OSError) as e:
                            logger.debug("Skipping dataset %s: %s", ds_name, e)
                            continue
                        out_items.append(fs)
                    return cls(out_items)
                if layout == LAYOUT_GROUP:
                    for grp_name in order:
                        try:
                            grp = h5f[grp_name]
                            fs = FrequencySeries.read(grp, format="hdf5", path="data")
                        except (KeyError, ValueError, TypeError, OSError):
                            try:
                                fs = FrequencySeries.read(grp, format="hdf5")
                            except (KeyError, ValueError, TypeError, OSError) as e2:
                                logger.debug("Skipping group %s: %s", grp_name, e2)
                                continue
                        out_items.append(fs)
                    return cls(out_items)
        from astropy.io import registry

        return registry.read(cls, source, *args, **kwargs)

    def __reduce_ex__(self, protocol: SupportsIndex):
        return (list, (list(self),))

    def write(self, target, *args, **kwargs):
        from astropy.io import registry

        return registry.write(self, target, *args, **kwargs)


class FrequencySeriesList(ListMapMixin, FrequencySeriesBaseList[FrequencySeries]):
    """List of `FrequencySeries` objects."""

    EntryClass = FrequencySeries

    # ===============================
    # 1. Axis & Edit Operations
    # ===============================

    crop = _make_list_map_method("crop", doc="Crop each FrequencySeries in the list.")
    pad = _make_list_map_method("pad", doc="Pad each FrequencySeries in the list.")
    interpolate = _make_list_map_method(
        "interpolate", doc="Interpolate each FrequencySeries in the list."
    )

    # ===============================
    # 2. Filter & Response
    # ===============================

    zpk = _make_list_map_method(
        "zpk", doc="Apply ZPK filter to each FrequencySeries in the list."
    )
    filter = _make_list_map_method(
        "filter", doc="Apply filter to each FrequencySeries in the list."
    )
    apply_response = _make_list_map_method(
        "apply_response", doc="Apply response to each FrequencySeries in the list."
    )

    # ===============================
    # 3. Analysis & Conversion
    # ===============================

    phase = _make_list_map_method("phase", doc="Compute phase of each FrequencySeries.")

    def angle(self, *args, **kwargs) -> FrequencySeriesList:
        """Alias for phase(). Returns a new FrequencySeriesList."""
        return self.phase(*args, **kwargs)

    degree = _make_list_map_method(
        "degree", doc="Compute phase (in degrees) of each FrequencySeries."
    )
    to_db = _make_list_map_method("to_db", doc="Convert each FrequencySeries to dB.")
    differentiate_time = _make_list_map_method(
        "differentiate_time", doc="Apply time differentiation to each item."
    )
    integrate_time = _make_list_map_method(
        "integrate_time", doc="Apply time integration to each item."
    )
    group_delay = _make_list_map_method(
        "group_delay", doc="Compute group delay of each item."
    )
    smooth = _make_list_map_method("smooth", doc="Smooth each FrequencySeries.")
    rebin = _make_list_map_method(
        "rebin", doc="Rebin each FrequencySeries in the list."
    )

    # ===============================
    # 4. Time Domain Conversion
    # ===============================

    ifft = _make_list_map_method(
        "ifft",
        doc="Compute IFFT of each FrequencySeries. Returns a TimeSeriesList.",
        result_class_path="gwexpy.timeseries.TimeSeriesList",
    )

    # ===============================
    # 5. External Library Interop
    # ===============================

    def to_control_frd(self, *args, **kwargs) -> list:
        """
        Convert each item to control.FRD.
        Returns a list of FRD objects.
        """
        return [fs.to_control_frd(*args, **kwargs) for fs in self]

    def to_torch(self, *args, **kwargs) -> list:
        """
        Convert each item to torch.Tensor.
        Returns a list of Tensors.
        """
        return [fs.to_torch(*args, **kwargs) for fs in self]

    def to_tensorflow(self, *args, **kwargs) -> list:
        """
        Convert each item to tensorflow.Tensor.
        Returns a list of Tensors.
        """
        return [fs.to_tensorflow(*args, **kwargs) for fs in self]

    def to_jax(self, *args, **kwargs) -> list:
        """
        Convert each item to jax.Array.
        Returns a list of Arrays.
        """
        return [fs.to_jax(*args, **kwargs) for fs in self]

    def to_cupy(self, *args, **kwargs) -> list:
        """
        Convert each item to cupy.ndarray.
        Returns a list of Arrays.
        """
        return [fs.to_cupy(*args, **kwargs) for fs in self]

    # ===============================
    # 6. Aggregation
    # ===============================

    def to_pandas(self, **kwargs):
        """
        Convert to pandas.DataFrame.
        Columns are named by channel name or index.
        """
        import pandas as pd

        data = {}
        for i, fs in enumerate(self):
            name = fs.name or f"series_{i}"
            if hasattr(fs, "to_pandas"):
                s = fs.to_pandas(index="frequency", copy=False)
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                elif not isinstance(s, pd.Series):
                    s = pd.Series(fs.value, index=fs.frequencies.value)
            else:
                s = pd.Series(fs.value, index=fs.frequencies.value)
            data[name] = s

        return pd.DataFrame(data)

    def to_xarray(self):
        """
        Convert to xarray.DataArray.
        Concatenates along a new dimension 'channel'.
        """
        import xarray as xr

        # Requires common coords usually, but concat handles it
        das = [fs.to_xarray() for fs in self]

        # We need a new dimension for channel.
        # If das have names, we can use them as coordinates?
        # But DataArrays are concatenated.

        # Check if we should assign channel names
        names = [getattr(fs, "name", f"ch{i}") for i, fs in enumerate(self)]

        # Concat
        return xr.concat(das, dim=xr.DataArray(names, dims="channel", name="channel"))

    def to_tmultigraph(self, name: str | None = None) -> Any:
        """Convert to ROOT TMultiGraph."""
        from gwexpy.interop import to_tmultigraph

        return to_tmultigraph(self, name=name)

    def write(self, target: str, *args: Any, **kwargs: Any) -> Any:
        """Write list to file (HDF5, ROOT, etc.)."""
        fmt = kwargs.get("format")
        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
            from gwexpy.interop.root_ import write_root_file

            return write_root_file(self, target, **kwargs)
        if fmt in ("csv", "txt"):
            from gwexpy.io.collection_dir import write_collection_dir

            overwrite = bool(kwargs.pop("overwrite", False))
            pairs: list[tuple[str, Any]] = []
            for i, fs in enumerate(self):
                key = fs.name or f"series_{i}"
                pairs.append((key, fs))
            return write_collection_dir(
                target,
                kind=type(self).__name__,
                entry_format=str(fmt),
                entries=pairs,
                writer=lambda fs, path, f: fs.write(path, format=f),
                meta_getter=lambda fs: {"unit": str(getattr(fs, "unit", "") or "")},
                overwrite=overwrite,
            )
        if fmt in ("hdf5", "h5", "hdf"):
            overwrite = bool(kwargs.pop("overwrite", False))
            mode = kwargs.pop("mode", None)
            layout = normalize_layout(kwargs.pop("layout", "gwpy"))
            used: set[str] = set()
            order: list[str] = []
            with ensure_hdf5_file(target, mode=mode, overwrite=overwrite) as h5f:
                for i, fs in enumerate(self):
                    key = safe_hdf5_key(str(i))
                    name = unique_hdf5_key(key, used=used)
                    if layout == LAYOUT_DATASET:
                        fs.write(h5f, format="hdf5", path=name)
                    else:
                        grp = h5f.create_group(name)
                        fs.write(grp, format="hdf5", path="data")
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
