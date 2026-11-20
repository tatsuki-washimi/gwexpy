"""TimeSeries I/O readers and registration for SPECFEM3D seismograms."""

from __future__ import annotations

import os
from typing import Any

from gwexpy.interop._specfem3d.seismogram_ascii import (
    read_specfem_seismogram_ascii,
    read_specfem_seismograms_ascii,
)
from gwexpy.interop._specfem3d.seismogram_sac import (
    read_specfem_seismogram_sac,
    read_specfem_seismograms_sac,
)
from gwexpy.io.time_selection import apply_time_selection, pop_time_selection

from ._registration import register_timeseries_format

__all__ = [
    "read_specfem3d_sac",
    "read_specfem3d_sac_dict",
    "read_specfem3d_seismogram",
    "read_specfem3d_seismogram_dict",
    "read_specfem3d_sema",
    "read_specfem3d_sema_dict",
    "read_specfem3d_semd",
    "read_specfem3d_semd_dict",
    "read_specfem3d_semv",
    "read_specfem3d_semv_dict",
    "register_specfem3d_timeseries_formats",
]


def _normalise_source(source: Any) -> Any:
    if (
        not isinstance(source, (str, bytes))
        and not hasattr(source, "__fspath__")
        and hasattr(source, "name")
        and isinstance(source.name, (str, os.PathLike))
    ):
        return source.name
    if isinstance(source, (list, tuple)):
        return type(source)(_normalise_source(item) for item in source)
    return source


def _make_single_reader(quantity: str | None = None):
    def _reader(source: Any, *args: Any, **kwargs: Any):
        start, end = pop_time_selection(kwargs)
        if quantity is not None:
            kwargs.setdefault("quantity", quantity)
        ts = read_specfem_seismogram_ascii(_normalise_source(source), **kwargs)
        return apply_time_selection(ts, start, end)

    _reader.__doc__ = "\n    Read a SPECFEM3D ASCII seismogram into a TimeSeries.\n    "
    return _reader


def _make_dict_reader(quantity: str | None = None, ext: str | None = None):
    def _reader(source: Any, *args: Any, **kwargs: Any):
        start, end = pop_time_selection(kwargs)
        if quantity is not None:
            kwargs.setdefault("quantity", quantity)
        if ext is not None:
            kwargs.setdefault("extensions", (f".{ext}",))
        tsd = read_specfem_seismograms_ascii(_normalise_source(source), **kwargs)
        return apply_time_selection(tsd, start, end)

    _reader.__doc__ = (
        "\n    Read SPECFEM3D ASCII seismograms into a TimeSeriesDict.\n    "
    )
    return _reader


read_specfem3d_seismogram = _make_single_reader(None)
read_specfem3d_seismogram_dict = _make_dict_reader(None, None)
read_specfem3d_semd = _make_single_reader("displacement")
read_specfem3d_semv = _make_single_reader("velocity")
read_specfem3d_sema = _make_single_reader("acceleration")
read_specfem3d_semd_dict = _make_dict_reader("displacement", "semd")
read_specfem3d_semv_dict = _make_dict_reader("velocity", "semv")
read_specfem3d_sema_dict = _make_dict_reader("acceleration", "sema")


def read_specfem3d_sac(source: Any, *args: Any, **kwargs: Any):
    """Read a SPECFEM3D SAC seismogram into a ``TimeSeries``."""
    start, end = pop_time_selection(kwargs)
    ts = read_specfem_seismogram_sac(_normalise_source(source), **kwargs)
    return apply_time_selection(ts, start, end)


def read_specfem3d_sac_dict(source: Any, *args: Any, **kwargs: Any):
    """Read SPECFEM3D SAC seismograms into a ``TimeSeriesDict``."""
    start, end = pop_time_selection(kwargs)
    tsd = read_specfem_seismograms_sac(_normalise_source(source), **kwargs)
    return apply_time_selection(tsd, start, end)


def _register_one(
    format_name: str,
    *,
    reader_dict: Any,
    reader_single: Any,
    extension: str | None = None,
    force: bool = True,
) -> None:
    from gwpy.io.registry import default_registry as io_registry

    from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

    already_identified = all(
        (format_name, cls) in io_registry._identifiers
        for cls in (TimeSeriesDict, TimeSeries, TimeSeriesMatrix)
    )
    if already_identified and not force:
        return
    # register_timeseries_format does not forward force= to register_identifier,
    # so omit extension= when the identifier is already registered to stay idempotent.
    register_timeseries_format(
        format_name,
        reader_dict=reader_dict,
        reader_single=reader_single,
        extension=None if already_identified else extension,
        auto_adapt=True,
        force=force,
    )


def register_specfem3d_timeseries_formats(*, force: bool = True) -> None:
    """Register SPECFEM3D seismogram formats on ``TimeSeries``, ``TimeSeriesDict``, and ``TimeSeriesMatrix``."""
    _register_one(
        "semd.specfem3d",
        reader_dict=read_specfem3d_semd_dict,
        reader_single=read_specfem3d_semd,
        extension="semd",
        force=force,
    )
    _register_one(
        "semv.specfem3d",
        reader_dict=read_specfem3d_semv_dict,
        reader_single=read_specfem3d_semv,
        extension="semv",
        force=force,
    )
    _register_one(
        "sema.specfem3d",
        reader_dict=read_specfem3d_sema_dict,
        reader_single=read_specfem3d_sema,
        extension="sema",
        force=force,
    )
    _register_one(
        "specfem3d",
        reader_dict=read_specfem3d_seismogram_dict,
        reader_single=read_specfem3d_seismogram,
        force=force,
    )
    _register_one(
        "sac.specfem3d",
        reader_dict=read_specfem3d_sac_dict,
        reader_single=read_specfem3d_sac,
        force=force,
    )


register_specfem3d_timeseries_formats()
