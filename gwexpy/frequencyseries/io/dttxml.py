"""Frequency-domain DTT XML reader."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC
from os import PathLike, fspath

import numpy as np
from gwpy.io.registry import default_registry as io_registry

from gwexpy.io.dttxml_common import (
    SUPPORTED_FREQ,
    SUPPORTED_MATRIX,
    load_dttxml_products,
)
from gwexpy.io.utils import (
    _coerce_numeric_epoch,
    _is_numeric_epoch,
    _reject_timezone_reinterpretation,
    apply_unit,
    datetime_to_gps,
    ensure_datetime,
    filter_by_channels,
    set_provenance,
)

from ..collections import FrequencySeriesDict
from ..frequencyseries import FrequencySeries
from ..matrix import FrequencySeriesMatrix

_DTTXML_FORMATS = ("xml.diaggui", "dttxml")


def _looks_like_dttxml(source) -> bool:
    if not isinstance(source, (str, PathLike)):
        return False
    source_text = str(fspath(source)).lower()
    return source_text.endswith(".xml") or source_text.endswith(".xml.gz")


def _build_epoch(value, timezone):
    if value is None:
        _reject_timezone_reinterpretation("xml.diaggui", timezone, None)
        return None
    tzinfo = _reject_timezone_reinterpretation(
        "xml.diaggui",
        timezone,
        value,
    )
    if _is_numeric_epoch(value):
        return _coerce_numeric_epoch(value)
    if tzinfo is None:
        tzinfo = UTC
    return datetime_to_gps(ensure_datetime(value, tzinfo=tzinfo))


def _frequencies_from_info(info, data_length):
    """Return an entry's explicit frequency axis or reconstruct it from f0/df."""
    raw_frequencies = info.get("frequencies")
    if raw_frequencies is not None:
        frequencies = np.asarray(raw_frequencies)
        if frequencies.size:
            return frequencies

    df = info.get("df")
    if df is None:
        return np.asarray([])
    f0 = info.get("f0")
    if f0 is None:
        f0 = 0.0
    return np.asarray(f0 + np.arange(data_length) * df)


def _frequency_payload_for_reader(payload):
    """Adapt the public loader's two backend shapes at the reader boundary."""
    normalized = {}
    for key, value in payload.items():
        if isinstance(value, FrequencySeries):
            df = getattr(value, "df", None)
            normalized[key] = {
                "data": np.asarray(value.value),
                "frequencies": np.asarray(value.frequencies.value),
                "f0": value.f0.value,
                "df": df.value if df is not None else None,
                "epoch": value.epoch.value if value.epoch is not None else None,
                "unit": value.unit,
            }
        elif isinstance(value, Mapping):
            normalized[key] = dict(value)
        else:
            raise TypeError(f"Unsupported xml.diaggui frequency payload for {key!r}")
    return normalized


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
    source
        Input DTT XML source.
    products
        DTT product name to load.
    channels
        Optional channel filter.
    unit
        Optional unit override.
    epoch
        Optional epoch override.
    timezone
        Time zone used when parsing datetime-like epochs.
    native : bool, optional
        If True, use gwexpy's native XML parser instead of the dttxml package.
        Use this to correctly handle complex TF data (subtype 6 phase loss fix).
        Default is False.
    **kwargs
        Additional unused compatibility arguments.

    """
    del kwargs
    epoch_timezone = _reject_timezone_reinterpretation(
        "xml.diaggui",
        timezone,
        epoch,
    )
    if products is None:
        raise ValueError("products must be specified for xml.diaggui")
    prod = str(products).upper()
    if prod not in SUPPORTED_FREQ:
        raise ValueError(
            f"xml.diaggui products '{prod}' is not a frequency-series product"
        )

    normalized = load_dttxml_products(source, native=native)
    payload = _frequency_payload_for_reader(normalized.get(prod, {}))
    fsd = FrequencySeriesDict()
    for ch, info in payload.items():
        if channels and ch not in channels:
            continue
        epoch_val = epoch if epoch is not None else info.get("epoch")
        gps = _build_epoch(epoch_val, epoch_timezone)
        data = np.asarray(info.get("data", np.array([])))
        freqs = _frequencies_from_info(info, data.size)
        df = info.get("df")
        if df is None and freqs.size > 1:
            df = np.diff(freqs)[0]
        kwargs_fs = {"name": ch, "channel": ch}
        if freqs.size:
            kwargs_fs["frequencies"] = freqs
        else:
            if df is not None:
                kwargs_fs["df"] = df
            f0 = info.get("f0")
            if f0 is not None:
                kwargs_fs["f0"] = f0
        fs = FrequencySeries(
            data,
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
            "format": "xml.diaggui",
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
    source
        Input DTT XML source.
    products
        DTT product name to load.
    rows
        Optional row labels to include.
    cols
        Optional column labels to include.
    pairs
        Optional `(row, col)` pairs to include.
    unit
        Optional unit override.
    epoch
        Optional epoch override.
    timezone
        Time zone used when parsing datetime-like epochs.
    native : bool, optional
        If True, use gwexpy's native XML parser instead of the dttxml package.
        Use this to correctly handle complex TF data (subtype 6 phase loss fix).
        Default is False.
    **kwargs
        Additional unused compatibility arguments.

    """
    del kwargs
    epoch_timezone = _reject_timezone_reinterpretation(
        "xml.diaggui",
        timezone,
        epoch,
    )
    if products is None:
        raise ValueError("products must be specified for xml.diaggui")
    prod = str(products).upper()
    if prod not in SUPPORTED_MATRIX:
        raise ValueError(f"xml.diaggui products '{prod}' is not a matrix product")

    normalized = load_dttxml_products(source, native=native)
    payload = _frequency_payload_for_reader(normalized.get(prod, {}))
    if not payload:
        raise ValueError(f"No matrix pairs found for xml.diaggui product '{prod}'")

    pair_filter = set(pairs) if pairs is not None else None
    row_filter = set(rows) if rows is not None else None
    col_filter = set(cols) if cols is not None else None
    selected_entries = []
    for pair, info in payload.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        row, col = pair
        if pair_filter is not None and pair not in pair_filter:
            continue
        if row_filter is not None and row not in row_filter:
            continue
        if col_filter is not None and col not in col_filter:
            continue
        selected_entries.append((pair, info))

    if not selected_entries:
        raise ValueError(
            f"No matrix pairs found for xml.diaggui product '{prod}' and filters"
        )

    row_labels = (
        list(rows)
        if rows is not None
        else sorted({pair[0] for pair, _ in selected_entries})
    )
    col_labels = (
        list(cols)
        if cols is not None
        else sorted({pair[1] for pair, _ in selected_entries})
    )

    first_pair, first_info = selected_entries[0]
    first_data = np.asarray(first_info.get("data", np.array([])))
    freq_axis = _frequencies_from_info(first_info, first_data.size)
    nfreq = len(freq_axis)
    vectors = []
    meta_unit = unit
    if meta_unit is None:
        meta_unit = next(
            (
                info.get("unit")
                for _, info in selected_entries
                if info.get("unit") is not None
            ),
            None,
        )
    for pair, info in selected_entries:
        vector = np.asarray(info.get("data", np.array([])))
        if vector.ndim != 1:
            raise ValueError(
                f"Matrix data for pair {pair!r} must be one-dimensional; "
                f"got shape {vector.shape}"
            )
        if vector.size != nfreq:
            raise ValueError(
                f"Matrix data for pair {pair!r} has {vector.size} samples, "
                f"but the frequency axis has {nfreq}"
            )
        pair_frequencies = _frequencies_from_info(info, vector.size)
        if pair_frequencies.shape != freq_axis.shape or not np.array_equal(
            pair_frequencies, freq_axis
        ):
            raise ValueError(
                f"Frequency axis for matrix pair {pair!r} does not match "
                f"the axis for pair {first_pair!r}"
            )
        vectors.append((pair, info, vector))

    # Keep parser precision where possible while allowing NaN-filled cells.
    dtype = np.result_type(np.float32, *(vector.dtype for _, _, vector in vectors))
    matrix = np.full((len(row_labels), len(col_labels), nfreq), np.nan, dtype=dtype)
    row_index = {label: index for index, label in enumerate(row_labels)}
    col_index = {label: index for index, label in enumerate(col_labels)}
    for (row, col), _info, vector in vectors:
        matrix[row_index[row], col_index[col]] = vector

    fsm = FrequencySeriesMatrix(
        matrix,
        frequencies=freq_axis,
        rows=row_labels,
        cols=col_labels,
        unit=meta_unit,
        epoch=_build_epoch(
            epoch if epoch is not None else first_info.get("epoch"), epoch_timezone
        ),
    )
    if unit:
        fsm = apply_unit(fsm, unit)
    set_provenance(
        fsm,
        {
            "format": "xml.diaggui",
            "products": prod,
            "rows": row_labels,
            "cols": col_labels,
            "pairs": [pair for pair, _, _ in vectors],
            "unit_source": "override" if unit else "file",
        },
    )
    return fsm


def read_frequencyseries_dttxml(*args, **kwargs) -> FrequencySeries:
    """Read one DTT XML product and return its first channel."""
    fsd = read_frequencyseriesdict_dttxml(*args, **kwargs)
    if len(fsd) == 0:
        raise ValueError("No channels found in xml.diaggui")
    return fsd[next(iter(fsd.keys()))]


# -- registration
for _fmt in _DTTXML_FORMATS:
    io_registry.register_reader(
        _fmt, FrequencySeries, read_frequencyseries_dttxml, force=True
    )
    io_registry.register_reader(
        _fmt, FrequencySeriesDict, read_frequencyseriesdict_dttxml, force=True
    )
    io_registry.register_reader(
        _fmt, FrequencySeriesMatrix, read_frequencyseriesmatrix_dttxml, force=True
    )


io_registry.register_identifier(
    "xml.diaggui",
    FrequencySeries,
    lambda *args, **kwargs: _looks_like_dttxml(args[1] if len(args) > 1 else None),
)
io_registry.register_identifier(
    "xml.diaggui",
    FrequencySeriesDict,
    lambda *args, **kwargs: _looks_like_dttxml(args[1] if len(args) > 1 else None),
)
io_registry.register_identifier(
    "xml.diaggui",
    FrequencySeriesMatrix,
    lambda *args, **kwargs: _looks_like_dttxml(args[1] if len(args) > 1 else None),
)
