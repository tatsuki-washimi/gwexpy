"""
Frequency-domain dttxml reader.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
from gwpy.io.registry import default_registry as io_registry

from gwexpy.io.dttxml_common import (
    SUPPORTED_FREQ,
    SUPPORTED_MATRIX,
    load_dttxml_products,
)
from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    ensure_datetime,
    filter_by_channels,
    parse_timezone,
    set_provenance,
)

from ..collections import FrequencySeriesDict
from ..frequencyseries import FrequencySeries
from ..matrix import FrequencySeriesMatrix


def _build_epoch(value, timezone):
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    tzinfo = parse_timezone(timezone) if timezone else None
    if tzinfo is None:
        tzinfo = UTC
    return datetime_to_gps(ensure_datetime(value, tzinfo=tzinfo))


def read_frequencyseriesdict_dttxml(
    source,
    *,
    products=None,
    channels=None,
    unit=None,
    epoch=None,
    timezone=None,
    native: bool = False,
    **kwargs,
) -> FrequencySeriesDict:
    """Read FrequencySeriesDict from DTT XML file.

    Parameters
    ----------
    native : bool, optional
        If True, use gwexpy's native XML parser instead of the dttxml package.
        Use this to correctly handle complex TF data (subtype 6 phase loss fix).
        Default is False.
    """
    if products is None:
        raise ValueError("products must be specified for dttxml")
    prod = str(products).upper()
    if prod not in SUPPORTED_FREQ:
        raise ValueError(f"dttxml products '{prod}' is not a frequency-series product")

    normalized = load_dttxml_products(source, native=native)
    payload = normalized.get(prod, {})
    fsd = FrequencySeriesDict()
    for ch, info in payload.items():
        if channels and ch not in channels:
            continue
        epoch_val = epoch if epoch is not None else info.get("epoch")
        gps = _build_epoch(epoch_val, timezone)
        freqs = np.asarray(info.get("frequencies") or [])
        df = info.get("df") or (np.diff(freqs)[0] if freqs.size > 1 else None)
        kwargs_fs = {"name": ch, "channel": ch}
        if freqs.size:
            kwargs_fs["frequencies"] = freqs
        elif df is not None:
            kwargs_fs["df"] = df
        fs = FrequencySeries(
            info.get("data", np.array([])),
            unit=info.get("unit") or unit,
            epoch=gps,
            **kwargs_fs,
        )
        fs = apply_unit(fs, unit) if unit else fs
        fsd[ch] = fs
    fsd = FrequencySeriesDict(filter_by_channels(fsd, channels))
    set_provenance(
        fsd,
        {
            "format": "dttxml",
            "products": prod,
            "channels": list(channels) if channels else list(fsd.keys()),
            "unit_source": "override" if unit else "file",
        },
    )
    return fsd


def read_frequencyseriesmatrix_dttxml(
    source,
    *,
    products=None,
    rows=None,
    cols=None,
    pairs=None,
    unit=None,
    epoch=None,
    timezone=None,
    native: bool = False,
    **kwargs,
) -> FrequencySeriesMatrix:
    """Read FrequencySeriesMatrix from DTT XML file.

    Parameters
    ----------
    native : bool, optional
        If True, use gwexpy's native XML parser instead of the dttxml package.
        Use this to correctly handle complex TF data (subtype 6 phase loss fix).
        Default is False.
    """
    if products is None:
        raise ValueError("products must be specified for dttxml")
    prod = str(products).upper()
    if prod not in SUPPORTED_MATRIX:
        raise ValueError(f"dttxml products '{prod}' is not a matrix product")

    normalized = load_dttxml_products(source, native=native)
    payload = normalized.get(prod, {})
    pairs_list = list(payload.keys())
    if pairs:
        pairs_list = [p for p in pairs_list if p in set(pairs)]
    row_labels = rows or sorted({p[0] for p in pairs_list if isinstance(p, tuple)})
    col_labels = cols or sorted({p[1] for p in pairs_list if isinstance(p, tuple)})

    # Determine frequency axis from first entry
    freq_axis = None
    df = None
    for info in payload.values():
        freqs = np.asarray(info.get("frequencies") or [])
        if freqs.size:
            freq_axis = freqs
            break
        if info.get("df"):
            df = info["df"]
            break
    if freq_axis is None:
        freq_axis = np.arange(len(next(iter(payload.values())).get("data", []))) * (
            df or 1.0
        )
    nfreq = len(freq_axis)
    matrix = np.full((len(row_labels), len(col_labels), nfreq), np.nan, dtype=float)
    meta_unit = unit

    for (row, col), info in payload.items():
        if row_labels and row not in row_labels:
            continue
        if col_labels and col not in col_labels:
            continue
        i = row_labels.index(row)
        j = col_labels.index(col)
        vector = np.asarray(info.get("data", np.zeros(nfreq, dtype=float)))
        if vector.size != nfreq:
            if vector.size and not freq_axis.size:
                freq_axis = np.arange(vector.size)
                matrix = np.full(
                    (len(row_labels), len(col_labels), vector.size), np.nan, dtype=float
                )
                nfreq = vector.size
            vector = np.resize(vector, nfreq)
        matrix[i, j] = vector
        if meta_unit is None:
            meta_unit = info.get("unit")

    fsm = FrequencySeriesMatrix(
        matrix,
        frequencies=freq_axis,
        rows=row_labels,
        cols=col_labels,
        unit=meta_unit,
        epoch=_build_epoch(epoch, timezone),
    )
    if unit:
        fsm = apply_unit(fsm, unit)
    set_provenance(
        fsm,
        {
            "format": "dttxml",
            "products": prod,
            "rows": row_labels,
            "cols": col_labels,
            "unit_source": "override" if unit else "file",
        },
    )
    return fsm


# -- registration
io_registry.register_reader(
    "dttxml", FrequencySeriesDict, read_frequencyseriesdict_dttxml, force=True
)
io_registry.register_reader(
    "dttxml", FrequencySeriesMatrix, read_frequencyseriesmatrix_dttxml, force=True
)


def _adapt_frequencyseries(*args, **kwargs):
    fsd = read_frequencyseriesdict_dttxml(*args, **kwargs)
    if not fsd:
        raise ValueError("No channels found in dttxml")
    return fsd[next(iter(fsd.keys()))]


io_registry.register_reader(
    "dttxml", FrequencySeries, _adapt_frequencyseries, force=True
)

io_registry.register_identifier(
    "dttxml", FrequencySeries, lambda *args, **kwargs: str(args[1]).endswith(".xml")
)
io_registry.register_identifier(
    "dttxml", FrequencySeriesDict, lambda *args, **kwargs: str(args[1]).endswith(".xml")
)
