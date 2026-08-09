"""TDMS reader for National Instruments files."""

from __future__ import annotations

import datetime
from collections.abc import Iterable

import numpy as np
from astropy import units as u

from gwexpy.io.time_selection import apply_time_selection, pop_time_selection
from gwexpy.io.utils import (
    _coerce_numeric_epoch,
    _consume_timezone_routing_state,
    _is_numeric_epoch,
    _make_timezone_routing_state,
    _reject_timezone_reinterpretation,
    apply_unit,
    datetime_to_gps,
    ensure_datetime,
    ensure_dependency,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format


def _import_nptdms():
    try:
        nptdms = ensure_dependency("nptdms", extra="io")
        return nptdms.TdmsFile
    except ImportError as exc:
        raise ImportError(
            "npTDMS is required for reading TDMS files. "
            "Install with `pip install 'gwexpy[io]'`."
        ) from exc


def read_timeseriesdict_tdms(
    source,
    *,
    channels: Iterable[str] | None = None,
    unit: str | u.Unit | None = None,
    epoch: float | datetime.datetime | None = None,
    timezone=None,
    **kwargs,
) -> TimeSeriesDict:
    """Read a TDMS file into a TimeSeriesDict.

    Parameters
    ----------
    source : str, Path, or list of str/Path
        Path to the TDMS file, or a list of paths.  When a list is
        given, channels found in several files are concatenated along
        the time axis and channels unique to one file are merged in.
    channels : iterable of str, optional
        Channel names to keep.
    unit : str or Unit, optional
        Physical unit override.
    epoch : float or datetime, optional
        Override the start time (GPS seconds or datetime).
        If not provided, uses the timestamp from the TDMS file properties.
    timezone : str or tzinfo, optional
        Localize a naive explicit ``epoch``. Source timestamps are absolute;
        numeric and aware epochs preserve their value and emit a warning.
    **kwargs
        Additional keyword arguments forwarded to the TDMS reader.  ``start``
        and ``end`` are honoured by cropping the result rather than ignored
        (issue #611).

    """
    start, end = pop_time_selection(kwargs)
    timezone_checked, epoch_timezone = _consume_timezone_routing_state(kwargs)
    if not timezone_checked:
        epoch_timezone = _reject_timezone_reinterpretation(
            "tdms",
            timezone,
            epoch,
        )

    multi = expand_multi_source(source)
    if multi is not None:
        return apply_time_selection(
            read_multi_dict(
                read_timeseriesdict_tdms,
                multi,
                "tdms",
                channels=channels,
                unit=unit,
                epoch=epoch,
                timezone=None,
                _timezone_routing_state=_make_timezone_routing_state(epoch_timezone),
                **kwargs,
            ),
            start,
            end,
        )

    TdmsFile = _import_nptdms()
    tdms_file = TdmsFile.read(source)

    # Fallback timing from root
    if "DateTime" in tdms_file.properties:
        dt_root = tdms_file.properties["DateTime"]
        if isinstance(dt_root, (np.datetime64, datetime.datetime)):
            # We will convert it below in the loop if needed
            pass

    tsd = TimeSeriesDict()

    for group in tdms_file.groups():
        for channel in group.channels():
            full_name = f"{group.name}/{channel.name}"

            if channels and full_name not in channels and channel.name not in channels:
                continue

            data = channel.read_data()
            props = channel.properties

            # Timing
            dt = props.get("wf_increment", 1.0)
            if dt == 0 or np.isinf(dt) or np.isnan(dt):
                dt = 1.0  # fallback

            t0 = props.get("wf_start_time", 0.0)
            if (
                t0 == 0.0 or (isinstance(t0, np.datetime64) and np.isnat(t0))
            ) and "DateTime" in tdms_file.properties:
                t0 = tdms_file.properties["DateTime"]

            # Epoch override processing
            if epoch is not None:
                if _is_numeric_epoch(epoch):
                    t0 = _coerce_numeric_epoch(epoch)
                elif isinstance(epoch, datetime.datetime):
                    t0 = datetime_to_gps(ensure_datetime(epoch, tzinfo=epoch_timezone))
                else:
                    raise TypeError(
                        f"epoch must be float or datetime, got {type(epoch)}"
                    )
            # Convert numpy.datetime64 or datetime.datetime to GPS
            elif isinstance(t0, (np.datetime64, datetime.datetime)):
                if isinstance(t0, np.datetime64):
                    if np.isnat(t0):
                        t0 = 0.0
                    else:
                        # Convert to python datetime
                        unix_epoch = np.datetime64("1970-01-01T00:00:00Z")
                        seconds = (t0 - unix_epoch) / np.timedelta64(1, "s")
                        dt_obj = datetime.datetime.fromtimestamp(
                            float(seconds), tz=datetime.UTC
                        )
                        t0 = datetime_to_gps(dt_obj)
                else:  # datetime.datetime
                    if t0.tzinfo is None:
                        t0 = t0.replace(tzinfo=datetime.UTC)
                    t0 = datetime_to_gps(t0)

            ts = TimeSeries(
                data,
                dt=dt,
                t0=t0,
                name=full_name,
                channel=full_name,
            )
            ts = apply_unit(ts, unit)
            tsd[full_name] = ts

    set_provenance(
        tsd,
        {
            "format": "tdms",
            "channels": list(tsd.keys()),
            "epoch_source": "user" if epoch is not None else "tdms_properties",
            "timezone": (str(epoch_timezone) if epoch_timezone is not None else None),
            "unit_source": "override" if unit else "tdms",
        },
    )
    return apply_time_selection(tsd, start, end)


def read_timeseries_tdms(*args, channels=None, **kwargs) -> TimeSeries:
    """Read a TDMS source and return the first selected channel."""
    tsd = read_timeseriesdict_tdms(*args, channels=channels, **kwargs)
    if not tsd:
        raise ValueError("No channels found in TDMS file")
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_tdms(*args, channels=None, **kwargs) -> TimeSeriesMatrix:
    """Read a TDMS source and convert the result to a matrix."""
    tsd = read_timeseriesdict_tdms(*args, channels=channels, **kwargs)
    return tsd.to_matrix()


# -- Registration

register_timeseries_format(
    "tdms",
    reader_dict=read_timeseriesdict_tdms,
    reader_single=read_timeseries_tdms,
    reader_matrix=read_timeseriesmatrix_tdms,
    extension="tdms",
)
