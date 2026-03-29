from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, SupportsIndex, TypeVar

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

from .histogram import Histogram

_H = TypeVar("_H", bound=Histogram)
logger = logging.getLogger(__name__)


class HistogramBaseDict(OrderedDict[str, _H]):
    """Ordered mapping container for `Histogram` objects."""

    EntryClass = Histogram

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    def __setitem__(self, key: str, value: _H) -> None:
        if not isinstance(value, self.EntryClass):
            raise TypeError(
                f"Cannot set key '{key}' to type '{type(value).__name__}' in {type(self).__name__}"
            )
        super().__setitem__(key, value)

    def setdefault(self, key: str, default: _H | None = None) -> _H:
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

    def copy(self) -> HistogramBaseDict[_H]:
        new = self.__class__()
        for key, val in self.items():
            new[key] = val.copy()
        return new

    def plot(
        self,
        label: str = "key",
        method: str = "plot",
        figsize: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot data."""
        from gwexpy.interop._registry import ConverterRegistry

        Plot = ConverterRegistry.get_constructor("Plot")

        kwargs = dict(kwargs)
        if figsize is not None:
            kwargs.setdefault("figsize", figsize)
        kwargs.update({"label": label, "method": method})

        plot = Plot(self, **kwargs)

        artmap = {"plot": "lines", "scatter": "collections", "step": "lines"}
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

    def plot_all(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for plot(). Plots all histograms in the dict."""
        return self.plot(*args, **kwargs)

    @classmethod
    def read(cls, source: Any, *args: Any, **kwargs: Any) -> HistogramDict:
        """Read data into a HistogramDict."""
        fmt = kwargs.get("format")
        if fmt in ("hdf5", "h5", "hdf"):
            import h5py

            with h5py.File(source, "r") as h5f:
                layout = detect_hdf5_layout(h5f)
                keymap = read_hdf5_keymap(h5f)
                order = read_hdf5_order(h5f)
                keys = order or list(h5f.keys())
                out = cls()
                if layout == LAYOUT_DATASET or layout is None:
                    for ds_name in keys:
                        try:
                            h = Histogram.read(h5f, format="hdf5", path=ds_name)
                        except (KeyError, ValueError, TypeError, OSError) as e:
                            logger.debug("Skipping dataset %s: %s", ds_name, e)
                            continue
                        orig_key = keymap.get(ds_name, ds_name)
                        out[orig_key] = h  # type: ignore[assignment]
                    return out  # type: ignore[return-value]
                if layout == LAYOUT_GROUP:
                    for grp_name in keys:
                        try:
                            grp = h5f[grp_name]
                            h = Histogram.read(grp, format="hdf5", path="data")
                        except (KeyError, ValueError, TypeError, OSError):
                            try:
                                h = Histogram.read(grp, format="hdf5", path=None)
                            except (KeyError, ValueError, TypeError, OSError) as e2:
                                logger.debug("Skipping group %s: %s", grp_name, e2)
                                continue
                        orig_key = keymap.get(grp_name, grp_name)
                        out[orig_key] = h  # type: ignore[assignment]
                    return out  # type: ignore[return-value]
        from astropy.io import registry

        return registry.read(cls, source, *args, **kwargs)  # type: ignore[no-any-return]

    def write(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        """Write dict to file (HDF5, ROOT, etc.)."""
        fmt = kwargs.get("format")

        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
            from gwexpy.interop.root_ import write_root_file

            return write_root_file(self, target, **kwargs)

        if fmt in ("hdf5", "h5", "hdf"):
            overwrite = bool(kwargs.pop("overwrite", False))
            mode = kwargs.pop("mode", None)
            layout = normalize_layout(kwargs.pop("layout", "gwpy"))
            used: set[str] = set()
            keymap: dict[str, str] = {}
            order: list[str] = []
            with ensure_hdf5_file(target, mode=mode, overwrite=overwrite) as h5f:
                for key, h in self.items():
                    safe = safe_hdf5_key(str(key))
                    name = unique_hdf5_key(safe, used=used)
                    if layout == LAYOUT_DATASET:
                        h.write(h5f, format="hdf5", path=name)
                    else:
                        grp = h5f.create_group(name)
                        h.write(grp, format="hdf5", path="data")
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


class HistogramDict(DictMapMixin, HistogramBaseDict[Histogram]):
    """Ordered mapping of `Histogram` objects keyed by label."""

    EntryClass = Histogram

    rebin = _make_dict_map_method("rebin", doc="Rebin each Histogram in the dict. Returns a HistogramDict.")
    fill = _make_dict_map_method("fill", doc="Fill each Histogram in the dict with new data.")
    integral = _make_dict_plain_method(
        "integral", doc="Compute integral of each Histogram in the dict. Returns a dict of Quantities."
    )
    to_density = _make_dict_map_method(
        "to_density", doc="Convert each Histogram to density representation. Returns a HistogramDict."
    )
    mean = _make_dict_plain_method("mean", doc="Compute weighted mean of each Histogram. Returns a dict of Quantities.")
    var = _make_dict_plain_method("var", doc="Compute weighted variance of each Histogram. Returns a dict of Quantities.")
    std = _make_dict_plain_method("std", doc="Compute weighted standard deviation of each Histogram. Returns a dict of Quantities.")
    median = _make_dict_plain_method("median", doc="Compute median of each Histogram. Returns a dict of Quantities.")
    quantile = _make_dict_plain_method("quantile", doc="Compute quantile of each Histogram. Returns a dict of Quantities.")
    min = _make_dict_plain_method("min", doc="Compute lower edge of first non-zero bin for each Histogram. Returns a dict of Quantities.")
    max = _make_dict_plain_method("max", doc="Compute upper edge of last non-zero bin for each Histogram. Returns a dict of Quantities.")


class HistogramBaseList(PlotMixin, list[_H]):
    """List container for `Histogram` objects with type enforcement."""

    EntryClass = Histogram

    def __init__(self, *items: _H | Iterable[_H]):
        super().__init__()
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            for item in items[0]:
                self.append(item)
        else:
            for item in items:
                self.append(item)  # type: ignore[arg-type]

    def _validate(self, item: Any, *, op: str) -> None:
        if not isinstance(item, self.EntryClass):
            raise TypeError(
                f"Cannot {op} type '{type(item).__name__}' to {type(self).__name__}"
            )

    def append(self, item: _H) -> HistogramBaseList[_H]:  # type: ignore[override]
        self._validate(item, op="append")
        super().append(item)
        return self

    def extend(self, items: Iterable[_H]) -> None:
        validated = self.__class__(*items)
        super().extend(validated)

    def insert(self, index: SupportsIndex, item: _H) -> None:
        self._validate(item, op="insert")
        super().insert(index, item)

    def __setitem__(self, index: Any, item: Any) -> None:
        if isinstance(index, slice):
            validated = self.__class__(*item)
            super().__setitem__(index, validated)
            return
        self._validate(item, op="set")
        super().__setitem__(index, item)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return self.__class__(*super().__getitem__(index))
        return super().__getitem__(index)

    def copy(self) -> HistogramBaseList[_H]:
        return self.__class__(*(item.copy() for item in self))

    def plot(self, **kwargs: Any) -> Any:
        """Plot this collection using gwexpy Plot."""
        from gwexpy.interop._registry import ConverterRegistry

        Plot = ConverterRegistry.get_constructor("Plot")
        return Plot(self, **kwargs)

    def plot_all(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for plot(). Plots all histograms."""
        return self.plot(*args, **kwargs)

    @classmethod
    def read(cls, source: Any, *args: Any, **kwargs: Any) -> HistogramBaseList[_H]:
        """Read data into a HistogramList."""
        fmt = kwargs.get("format")
        if fmt in ("hdf5", "h5", "hdf"):
            import h5py

            with h5py.File(source, "r") as h5f:
                layout = detect_hdf5_layout(h5f)
                order = read_hdf5_order(h5f) or list(h5f.keys())
                out_items: list[Histogram] = []
                if layout == LAYOUT_DATASET or layout is None:
                    for ds_name in order:
                        try:
                            h = Histogram.read(h5f, format="hdf5", path=ds_name)
                        except (KeyError, ValueError, TypeError, OSError) as e:
                            logger.debug("Skipping dataset %s: %s", ds_name, e)
                            continue
                        out_items.append(h)
                    return cls(out_items)  # type: ignore[arg-type]
                if layout == LAYOUT_GROUP:
                    for grp_name in order:
                        try:
                            grp = h5f[grp_name]
                            h = Histogram.read(grp, format="hdf5", path="data")
                        except (KeyError, ValueError, TypeError, OSError):
                            try:
                                h = Histogram.read(grp, format="hdf5", path=None)
                            except (KeyError, ValueError, TypeError, OSError) as e2:
                                logger.debug("Skipping group %s: %s", grp_name, e2)
                                continue
                        out_items.append(h)
                    return cls(out_items)  # type: ignore[arg-type]
        from astropy.io import registry

        return registry.read(cls, source, *args, **kwargs)  # type: ignore[no-any-return]

    def write(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        fmt = kwargs.get("format")

        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
            from gwexpy.interop.root_ import write_root_file

            return write_root_file(self, target, **kwargs)

        if fmt in ("hdf5", "h5", "hdf"):
            overwrite = bool(kwargs.pop("overwrite", False))
            mode = kwargs.pop("mode", None)
            layout = normalize_layout(kwargs.pop("layout", "gwpy"))
            used: set[str] = set()
            keymap: dict[str, str] = {}
            order: list[str] = []
            with ensure_hdf5_file(target, mode=mode, overwrite=overwrite) as h5f:
                for i, h in enumerate(self):
                    safe = safe_hdf5_key(str(i))
                    name = unique_hdf5_key(safe, used=used)
                    if layout == LAYOUT_DATASET:
                        h.write(h5f, format="hdf5", path=name)
                    else:
                        grp = h5f.create_group(name)
                        h.write(grp, format="hdf5", path="data")
                    keymap[name] = str(i)
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


class HistogramList(ListMapMixin, HistogramBaseList[Histogram]):
    """List of `Histogram` objects."""

    EntryClass = Histogram

    rebin = _make_list_map_method("rebin", doc="Rebin each Histogram in the list. Returns a HistogramList.")
    fill = _make_list_map_method("fill", doc="Fill each Histogram in the list with new data.")
    to_density = _make_list_map_method(
        "to_density", doc="Convert each Histogram to density representation. Returns a HistogramList."
    )
    mean = _make_list_map_method("mean", doc="Compute weighted mean of each Histogram. Returns a list of Quantities.")
    var = _make_list_map_method("var", doc="Compute weighted variance of each Histogram. Returns a list of Quantities.")
    std = _make_list_map_method("std", doc="Compute weighted standard deviation of each Histogram. Returns a list of Quantities.")
    median = _make_list_map_method("median", doc="Compute median of each Histogram. Returns a list of Quantities.")
    quantile = _make_list_map_method("quantile", doc="Compute quantile of each Histogram. Returns a list of Quantities.")
    min = _make_list_map_method("min", doc="Compute lower edge of first non-zero bin for each Histogram. Returns a list of Quantities.")
    max = _make_list_map_method("max", doc="Compute upper edge of last non-zero bin for each Histogram. Returns a list of Quantities.")
