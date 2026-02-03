from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, SupportsIndex, TypeVar

import h5py

logger = logging.getLogger(__name__)
import numpy as np
from astropy import units as u

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

from .frequencyseries import FrequencySeries, as_series_dict_class

_FS = TypeVar("_FS", bound=FrequencySeries)


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
        from gwexpy.plot import Plot

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
class FrequencySeriesDict(FrequencySeriesBaseDict[FrequencySeries]):
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

    def pad(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Pad each FrequencySeries in the dict.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.pad(*args, **kwargs)
        return new_dict

    def interpolate(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Interpolate each FrequencySeries in the dict.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.interpolate(*args, **kwargs)
        return new_dict

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

    def zpk(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Apply ZPK filter to each FrequencySeries.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.zpk(*args, **kwargs)
        return new_dict

    def filter(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Apply filter to each FrequencySeries.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.filter(*args, **kwargs)
        return new_dict

    def apply_response(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Apply response to each FrequencySeries.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.apply_response(*args, **kwargs)
        return new_dict

    # ===============================
    # 3. Analysis & Conversion
    # ===============================

    def phase(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Compute phase of each FrequencySeries.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.phase(*args, **kwargs)
        return new_dict

    def angle(self, *args, **kwargs) -> FrequencySeriesDict:
        """Alias for phase(). Returns a new FrequencySeriesDict."""
        return self.phase(*args, **kwargs)

    def degree(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Compute phase (in degrees) of each FrequencySeries.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.degree(*args, **kwargs)
        return new_dict

    def to_db(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Convert each FrequencySeries to dB.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.to_db(*args, **kwargs)
        return new_dict

    def differentiate_time(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Apply time differentiation to each item.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.differentiate_time(*args, **kwargs)
        return new_dict

    def integrate_time(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Apply time integration to each item.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.integrate_time(*args, **kwargs)
        return new_dict

    def group_delay(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Compute group delay of each item.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.group_delay(*args, **kwargs)
        return new_dict

    def smooth(self, *args, **kwargs) -> FrequencySeriesDict:
        """
        Smooth each FrequencySeries.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.smooth(*args, **kwargs)
        return new_dict

    def rebin(self, width: float | u.Quantity) -> FrequencySeriesDict:
        """
        Rebin each FrequencySeries in the dict.
        Returns a new FrequencySeriesDict.
        """
        new_dict = self.__class__()
        for key, fs in self.items():
            new_dict[key] = fs.rebin(width)
        return new_dict

    # ===============================
    # 4. Time Domain Conversion
    # ===============================

    def ifft(self, *args, **kwargs):
        """
        Compute IFFT of each FrequencySeries.
        Returns a TimeSeriesDict.
        """
        from gwexpy.timeseries import TimeSeriesDict

        new_dict = TimeSeriesDict()
        for key, fs in self.items():
            new_dict[key] = fs.ifft(*args, **kwargs)
        return new_dict

    # ===============================
    # 5. External Library Interop
    # ===============================

    def to_control_frd(self, *args, **kwargs) -> dict:
        """
        Convert each item to control.FRD.
        Returns a dict of FRD objects.
        """
        return {key: fs.to_control_frd(*args, **kwargs) for key, fs in self.items()}

    def to_torch(self, *args, **kwargs) -> dict:
        """
        Convert each item to torch.Tensor.
        Returns a dict of Tensors.
        """
        return {key: fs.to_torch(*args, **kwargs) for key, fs in self.items()}

    def to_tensorflow(self, *args, **kwargs) -> dict:
        """
        Convert each item to tensorflow.Tensor.
        Returns a dict of Tensors.
        """
        return {key: fs.to_tensorflow(*args, **kwargs) for key, fs in self.items()}

    def to_jax(self, *args, **kwargs) -> dict:
        """
        Convert each item to jax.Array.
        Returns a dict of Arrays.
        """
        return {key: fs.to_jax(*args, **kwargs) for key, fs in self.items()}

    def to_cupy(self, *args, **kwargs) -> dict:
        """
        Convert each item to cupy.ndarray.
        Returns a dict of Arrays.
        """
        return {key: fs.to_cupy(*args, **kwargs) for key, fs in self.items()}

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


class FrequencySeriesBaseList(list[_FS]):
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

    def copy(self) -> FrequencySeriesBaseList[_FS]:
        return self.__class__(*(item.copy() for item in self))

    def plot(self, **kwargs: Any):
        """Plot all series. Delegates to gwexpy.plot.Plot."""
        from gwexpy.plot import Plot

        return Plot(self, **kwargs)

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


class FrequencySeriesList(FrequencySeriesBaseList[FrequencySeries]):
    """List of `FrequencySeries` objects."""

    EntryClass = FrequencySeries

    # ===============================
    # 1. Axis & Edit Operations
    # ===============================

    def crop(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Crop each FrequencySeries in the list.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.crop(*args, **kwargs))
        return new_list

    def pad(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Pad each FrequencySeries in the list.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.pad(*args, **kwargs))
        return new_list

    def interpolate(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Interpolate each FrequencySeries in the list.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.interpolate(*args, **kwargs))
        return new_list

    # ===============================
    # 2. Filter & Response
    # ===============================

    def zpk(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Apply ZPK filter to each FrequencySeries in the list.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.zpk(*args, **kwargs))
        return new_list

    def filter(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Apply filter to each FrequencySeries in the list.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.filter(*args, **kwargs))
        return new_list

    def apply_response(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Apply response to each FrequencySeries in the list.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.apply_response(*args, **kwargs))
        return new_list

    # ===============================
    # 3. Analysis & Conversion
    # ===============================

    def phase(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Compute phase of each FrequencySeries.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.phase(*args, **kwargs))
        return new_list

    def angle(self, *args, **kwargs) -> FrequencySeriesList:
        """Alias for phase(). Returns a new FrequencySeriesList."""
        return self.phase(*args, **kwargs)

    def degree(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Compute phase (in degrees) of each FrequencySeries.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.degree(*args, **kwargs))
        return new_list

    def to_db(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Convert each FrequencySeries to dB.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.to_db(*args, **kwargs))
        return new_list

    def differentiate_time(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Apply time differentiation to each item.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.differentiate_time(*args, **kwargs))
        return new_list

    def integrate_time(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Apply time integration to each item.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.integrate_time(*args, **kwargs))
        return new_list

    def group_delay(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Compute group delay of each item.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.group_delay(*args, **kwargs))
        return new_list

    def smooth(self, *args, **kwargs) -> FrequencySeriesList:
        """
        Smooth each FrequencySeries.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.smooth(*args, **kwargs))
        return new_list

    def rebin(self, width: float | u.Quantity) -> FrequencySeriesList:
        """
        Rebin each FrequencySeries in the list.
        Returns a new FrequencySeriesList.
        """
        new_list = self.__class__()
        for fs in self:
            list.append(new_list, fs.rebin(width))
        return new_list

    # ===============================
    # 4. Time Domain Conversion
    # ===============================

    def ifft(self, *args, **kwargs):
        """
        Compute IFFT of each FrequencySeries.
        Returns a TimeSeriesList.
        """
        from gwexpy.timeseries import TimeSeriesList

        new_list = TimeSeriesList()
        for fs in self:
            list.append(new_list, fs.ifft(*args, **kwargs))
        return new_list

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
