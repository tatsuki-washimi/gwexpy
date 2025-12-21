"""
dttxml reader (Diag GUI XML).

This module supports a minimal XML parser and will use an external
`dttxml_source.txt` parser if it is available next to the project root.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union, Any, List

import numpy as np
from gwpy.io import registry as io_registry

from gwexpy.io.dttxml_common import SUPPORTED_TS, load_dttxml_products
from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    ensure_datetime,
    filter_by_channels,
    parse_timezone,
    set_provenance,
)
from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _build_epoch(value, timezone):
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    tzinfo = parse_timezone(timezone) if timezone else None
    if tzinfo is None:
        from datetime import timezone as _timezone
        tzinfo = _timezone.utc
    return datetime_to_gps(ensure_datetime(value, tzinfo=tzinfo))


def read_timeseriesdict_dttxml(
    source,
    *,
    products=None,
    channels: Optional[Iterable[str]] = None,
    unit=None,
    epoch=None,
    timezone=None,
    pad=np.nan,
    **kwargs,
) -> TimeSeriesDict:
    if products is None:
        raise ValueError("products must be specified for dttxml")
    prod = str(products).upper()
    if prod not in SUPPORTED_TS:
        raise ValueError(f"dttxml products '{prod}' is not a time-series product")

    normalized = load_dttxml_products(source)
    payload = normalized.get(prod, {})
    tsd = TimeSeriesDict()
    for ch, info in payload.items():
        if channels and ch not in channels:
            continue
        epoch_val = epoch if epoch is not None else info.get("epoch")
        gps = _build_epoch(epoch_val, timezone) or 0.0
        dt = info.get("dt") or 1.0
        ts = TimeSeries(
            info.get("data", np.array([])),
            dt=dt,
            t0=gps,
            channel=ch,
            name=ch,
            unit=info.get("unit") or unit,
        )
        ts = apply_unit(ts, unit) if unit else ts
        tsd[ch] = ts
    tsd = TimeSeriesDict(filter_by_channels(tsd, channels))
    set_provenance(
        tsd,
        {
            "format": "dttxml",
            "products": prod,
            "channels": list(channels) if channels else list(tsd.keys()),
            "unit_source": "override" if unit else "file",
        },
    )
    return tsd


def read_timeseries_dttxml(*args, **kwargs) -> TimeSeries:
    tsd = read_timeseriesdict_dttxml(*args, **kwargs)
    if not tsd:
        raise ValueError("No channels found in dttxml")
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_dttxml(*args, **kwargs) -> TimeSeriesMatrix:
    tsd = read_timeseriesdict_dttxml(*args, **kwargs)
    return tsd.to_matrix()


# -- registration
io_registry.register_reader("dttxml", TimeSeriesDict, read_timeseriesdict_dttxml)
io_registry.register_reader("dttxml", TimeSeries, read_timeseries_dttxml)
io_registry.register_reader("dttxml", TimeSeriesMatrix, read_timeseriesmatrix_dttxml)
