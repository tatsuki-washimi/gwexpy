"""
I/O registration helpers for reducing boilerplate code.

This module provides utilities to register TimeSeries I/O handlers
with automatic adapter generation for TimeSeriesDict, TimeSeries,
and TimeSeriesMatrix types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from gwpy.io.registry import default_registry as io_registry

if TYPE_CHECKING:
    from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

__all__ = ["register_timeseries_format"]


def register_timeseries_format(
    format_name: str,
    *,
    reader_dict: Callable[..., Any] | None = None,
    reader_single: Callable[..., Any] | None = None,
    reader_matrix: Callable[..., Any] | None = None,
    writer_dict: Callable[..., Any] | None = None,
    writer_single: Callable[..., Any] | None = None,
    writer_matrix: Callable[..., Any] | None = None,
    identifier_dict: Callable[..., bool] | None = None,
    identifier_single: Callable[..., bool] | None = None,
    extension: str | None = None,
    auto_adapt: bool = True,
    force: bool = True,
) -> None:
    """
    Register TimeSeries I/O handlers for a format.

    This helper automatically creates adapters for TimeSeries and TimeSeriesMatrix
    from TimeSeriesDict reader/writer functions when auto_adapt=True, reducing
    boilerplate registration code.

    Parameters
    ----------
    format_name : str
        Format identifier (e.g., "ats", "wav", "gbd").
    reader_dict : callable, optional
        Reader function for TimeSeriesDict.
    reader_single : callable, optional
        Reader function for TimeSeries. If None and auto_adapt=True,
        will be auto-generated from reader_dict.
    reader_matrix : callable, optional
        Reader function for TimeSeriesMatrix. If None and auto_adapt=True,
        will be auto-generated from reader_dict.
    writer_dict : callable, optional
        Writer function for TimeSeriesDict.
    writer_single : callable, optional
        Writer function for TimeSeries. If None and auto_adapt=True,
        will be auto-generated from writer_dict.
    writer_matrix : callable, optional
        Writer function for TimeSeriesMatrix. If None and auto_adapt=True,
        will be auto-generated from writer_dict.
    identifier_dict : callable, optional
        Identifier function for TimeSeriesDict. If None and extension is provided,
        will be auto-generated.
    identifier_single : callable, optional
        Identifier function for TimeSeries. If None, uses identifier_dict.
    extension : str, optional
        File extension (e.g., "ats", "wav"). Used for auto-generating identifiers.
    auto_adapt : bool, optional
        If True, automatically generate missing reader/writer functions.
        Default: True.
    force : bool, optional
        If True, force registration even if format already exists.
        Default: True.

    Examples
    --------
    Simple registration with auto-adaptation:

    >>> register_timeseries_format(
    ...     "gbd",
    ...     reader_dict=read_timeseriesdict_gbd,
    ...     extension="gbd",
    ... )

    Registration with custom single reader:

    >>> register_timeseries_format(
    ...     "ats",
    ...     reader_dict=read_timeseriesdict_ats,
    ...     reader_single=read_timeseries_ats,
    ...     extension="ats",
    ... )

    Registration with writer support:

    >>> register_timeseries_format(
    ...     "wav",
    ...     reader_dict=read_timeseriesdict_wav,
    ...     writer_dict=write_timeseriesdict_wav,
    ...     extension="wav",
    ... )
    """
    # Import here to avoid circular dependencies
    import functools
    from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

    # Register TimeSeriesDict reader
    if reader_dict is not None:
        if not reader_dict.__doc__ or not reader_dict.__doc__.strip():
            reader_dict.__doc__ = f"\n    Read {format_name} data into a TimeSeriesDict.\n    "
        io_registry.register_reader(
            format_name, TimeSeriesDict, reader_dict, force=force
        )

    # Register TimeSeries reader (auto-adapt or use provided)
    if reader_single is None and auto_adapt and reader_dict is not None:
        # Auto-generate: extract first TimeSeries from dict
        @functools.wraps(reader_dict)
        def _adapted_single_reader(*args, **kwargs):
            tsd = reader_dict(*args, **kwargs)
            if not tsd:
                raise ValueError(f"No data found in {format_name} file")
            return tsd[next(iter(tsd.keys()))]
        
        _adapted_single_reader.__doc__ = f"\n    Read {format_name} data into a TimeSeries.\n    "
        reader_single = _adapted_single_reader

    if reader_single is not None:
        if not reader_single.__doc__ or not reader_single.__doc__.strip():
            reader_single.__doc__ = f"\n    Read {format_name} data into a TimeSeries.\n    "
        io_registry.register_reader(
            format_name, TimeSeries, reader_single, force=force
        )

    # Register TimeSeriesMatrix reader (auto-adapt or use provided)
    if reader_matrix is None and auto_adapt and reader_dict is not None:
        # Auto-generate: convert dict to matrix
        @functools.wraps(reader_dict)
        def _adapted_matrix_reader(*args, **kwargs):
            tsd = reader_dict(*args, **kwargs)
            return tsd.to_matrix()
        
        _adapted_matrix_reader.__doc__ = f"\n    Read {format_name} data into a TimeSeriesMatrix.\n    "
        reader_matrix = _adapted_matrix_reader

    if reader_matrix is not None:
        if not reader_matrix.__doc__ or not reader_matrix.__doc__.strip():
            reader_matrix.__doc__ = f"\n    Read {format_name} data into a TimeSeriesMatrix.\n    "
        io_registry.register_reader(
            format_name, TimeSeriesMatrix, reader_matrix, force=force
        )

    # Register TimeSeriesDict writer
    if writer_dict is not None:
        if not writer_dict.__doc__ or not writer_dict.__doc__.strip():
            writer_dict.__doc__ = f"\n    Write TimeSeriesDict to {format_name} format.\n    "
        io_registry.register_writer(
            format_name, TimeSeriesDict, writer_dict, force=force
        )

    # Register TimeSeries writer (auto-adapt or use provided)
    if writer_single is None and auto_adapt and writer_dict is not None:
        # Auto-generate: wrap single series in dict
        @functools.wraps(writer_dict)
        def _adapted_single_writer(ts, target, *args, **kwargs):
            from .. import TimeSeriesDict

            tsd = TimeSeriesDict({ts.name: ts})
            return writer_dict(tsd, target, *args, **kwargs)
        
        _adapted_single_writer.__doc__ = f"\n    Write TimeSeries to {format_name} format.\n    "
        writer_single = _adapted_single_writer

    if writer_single is not None:
        if not writer_single.__doc__ or not writer_single.__doc__.strip():
            writer_single.__doc__ = f"\n    Write TimeSeries to {format_name} format.\n    "
        io_registry.register_writer(
            format_name, TimeSeries, writer_single, force=force
        )

    # Register TimeSeriesMatrix writer (auto-adapt or use provided)
    if writer_matrix is None and auto_adapt and writer_dict is not None:
        # Auto-generate: convert matrix to dict
        @functools.wraps(writer_dict)
        def _adapted_matrix_writer(tsm, target, *args, **kwargs):
            tsd = tsm.to_dict()
            return writer_dict(tsd, target, *args, **kwargs)
        
        _adapted_matrix_writer.__doc__ = f"\n    Write TimeSeriesMatrix to {format_name} format.\n    "
        writer_matrix = _adapted_matrix_writer

    if writer_matrix is not None:
        if not writer_matrix.__doc__ or not writer_matrix.__doc__.strip():
            writer_matrix.__doc__ = f"\n    Write TimeSeriesMatrix to {format_name} format.\n    "
        io_registry.register_writer(
            format_name, TimeSeriesMatrix, writer_matrix, force=force
        )

    # Register identifiers
    if identifier_dict is None and extension is not None:
        # Auto-generate extension-based identifier
        def _extension_identifier(*args, **kwargs):
            # args[1] is the source path in gwpy's identifier signature
            if len(args) < 2:
                return False
            return str(args[1]).lower().endswith(f".{extension}")

        identifier_dict = _extension_identifier

    if identifier_dict is not None:
        io_registry.register_identifier(format_name, TimeSeriesDict, identifier_dict)

    # Use same identifier for TimeSeries unless explicitly provided
    if identifier_single is None:
        identifier_single = identifier_dict

    if identifier_single is not None:
        io_registry.register_identifier(format_name, TimeSeries, identifier_single)
        # TimeSeriesMatrix typically uses the same identifier
        io_registry.register_identifier(format_name, TimeSeriesMatrix, identifier_single)
