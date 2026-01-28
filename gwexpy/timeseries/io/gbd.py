"""
GBD (GRAPHTEC data logger) reader.

This is a cleaned-up port of the legacy `gbd2gwf.py` workflow, adapted for
gwexpy TimeSeries/TimeSeriesDict readers.
"""
from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gwpy.io import registry as io_registry

from gwexpy.io.utils import (
    datetime_to_gps,
    ensure_datetime,
    filter_by_channels,
    parse_timezone,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

HEADER_SIZE_PATTERN = re.compile(r"HeaderSiz[e]?\s*[:=]\s*(\d+)", re.IGNORECASE)


@dataclass
class GBDHeader:
    """Header information extracted from a GBD (GRAPHTEC) file."""

    header_size: int
    start_local: str
    stop_local: str
    dt: float
    dtype: np.dtype
    order: list[str]
    counts: int
    scales: list[float]
    raw_text: str


def read_timeseriesdict_gbd(
    source,
    *,
    timezone=None,
    channels: Iterable[str] | None = None,
    unit=None,
    epoch=None,
    pad=np.nan,
    **kwargs,
) -> TimeSeriesDict:
    """
    Read a GRAPHTEC .GBD file into a TimeSeriesDict.

    Parameters
    ----------
    source : str or file-like
        Path or open file object to the GBD file.
    timezone : str or tzinfo, required
        Local timezone of the logger (IANA name or UTC offset).
    channels : iterable, optional
        Subset of channel names to load.
    unit : str or Unit, optional
        Override the physical unit (default: 'V').
    epoch : float or datetime, optional
        Override the epoch (GPS seconds). If datetime, it will be converted to GPS.
    pad : float, optional
        Padding value (unused, accepted for API symmetry).
    """
    if timezone is None:
        raise ValueError("timezone is required for GBD files")
    tzinfo = parse_timezone(timezone)
    header, data = _read_gbd(source)

    if epoch is not None:
        if isinstance(epoch, (int, float, np.floating)):
            gps_start = float(epoch)
        else:
            gps_start = datetime_to_gps(ensure_datetime(epoch, tzinfo=tzinfo))
        epoch_source = "user"
    else:
        gps_start = datetime_to_gps(ensure_datetime(header.start_local, tzinfo=tzinfo))
        epoch_source = "timezone_start"

    scale_arr = np.asarray(
        header.scales if header.scales else [1.0] * len(header.order)
    )
    if scale_arr.size != len(header.order):
        scale_arr = np.ones(len(header.order))
    data = data * scale_arr

    tsd = TimeSeriesDict()
    for idx, ch in enumerate(header.order):
        ts = TimeSeries(
            data[:, idx],
            t0=gps_start,
            dt=header.dt,
            unit=unit or "V",
            channel=ch,
            name=ch,
        )
        tsd[ch] = ts

    tsd = TimeSeriesDict(filter_by_channels(tsd, channels))
    set_provenance(
        tsd,
        {
            "format": "gbd",
            "timezone": str(timezone),
            "epoch_source": epoch_source,
            "unit_source": "override" if unit else "gbd_default",
            "gap": "pad",
            "pad_value": pad,
            "channels": list(channels) if channels is not None else header.order,
        },
    )
    return tsd


def read_timeseries_gbd(*args, channels=None, **kwargs) -> TimeSeries:
    """
    Adapter to always return a single TimeSeries.
    """
    tsd = read_timeseriesdict_gbd(*args, channels=channels, **kwargs)
    if not tsd:
        raise ValueError("No channels found in GBD file")
    if channels:
        first_key = next(iter(tsd.keys()))
        return tsd[first_key]
    # default: first in file order
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_gbd(*args, channels=None, **kwargs) -> TimeSeriesMatrix:
    """
    Build a TimeSeriesMatrix from a GBD file.
    """
    tsd = read_timeseriesdict_gbd(*args, channels=channels, **kwargs)
    return tsd.to_matrix()


def _read_gbd(source) -> tuple[GBDHeader, np.ndarray]:
    close_after = False
    if isinstance(source, (str, Path)):
        fh = open(source, "rb")
        close_after = True
    else:
        fh = source

    try:
        header_text, header_size = _read_header_text(fh)
        header = _parse_header(header_text, header_size)
        data = _read_data_block(fh, header)
        return header, data
    finally:
        if close_after:
            fh.close()


def _read_header_text(fh: io.BufferedReader) -> tuple[str, int]:
    probe = fh.read(4096)
    text_probe = probe.decode("ascii", errors="ignore")
    match = HEADER_SIZE_PATTERN.search(text_probe)
    if not match:
        raise ValueError("Failed to locate HeaderSiz in GBD file")
    header_size = int(match.group(1))
    fh.seek(0)
    header_bytes = fh.read(header_size)
    header_text = header_bytes.decode("ascii", errors="ignore")
    return header_text, header_size


def _parse_header(header_text: str, header_size: int) -> GBDHeader:
    start = _find_field(header_text, r"Start\s*[:=]\s*([^\r\n]+)")
    stop = _find_field(header_text, r"Stop\s*[:=]\s*([^\r\n]+)", default=start)
    dt_raw = _find_field(header_text, r"Sample\s*[:=]\s*([^\r\n]+)", default="1")
    dt = _parse_sample(dt_raw)
    dtype_raw = _find_field(
        header_text, r"Type\s*[:=]\s*([^\r\n]+)", default="little,float32"
    )
    dtype = _parse_dtype(dtype_raw)
    order_raw = _find_field(header_text, r"Order\s*[:=]\s*([^\r\n]+)", default="")
    order = [c.strip() for c in re.split(r"[,\s]+", order_raw) if c.strip()] or ["CH0"]
    counts_raw = _find_field(header_text, r"Counts\s*[:=]\s*([0-9]+)", default="0")
    counts = int(counts_raw)
    scales = _parse_scales(header_text, order)
    return GBDHeader(
        header_size=header_size,
        start_local=start,
        stop_local=stop,
        dt=dt,
        dtype=dtype,
        order=order,
        counts=counts,
        scales=scales,
        raw_text=header_text,
    )


def _find_field(text: str, pattern: str, default: str | None = None) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if default is not None:
        return default
    raise ValueError(f"Missing required header field matching {pattern}")


def _parse_sample(raw: str) -> float:
    cleaned = raw.strip()
    suffixes = {"ms": 1e-3, "us": 1e-6, "ns": 1e-9, "s": 1.0}
    for suf, scale in suffixes.items():
        if cleaned.lower().endswith(suf):
            try:
                return float(cleaned[: -len(suf)]) * scale
            except ValueError:
                continue
    return float(cleaned)


def _parse_dtype(raw: str) -> np.dtype:
    text = raw.lower()
    endian = "<" if "little" in text else ">" if "big" in text else "<"
    if "float64" in text or "double" in text:
        base = "f8"
    elif "float" in text:
        base = "f4"
    elif "int64" in text or "longlong" in text:
        base = "i8"
    elif "int32" in text or "long" in text:
        base = "i4"
    elif "uint16" in text or "ushort" in text:
        base = "u2"
    elif "int16" in text or "short" in text:
        base = "i2"
    elif "uint8" in text or "uchar" in text or "byte" in text:
        base = "u1"
    else:
        base = "f4"
    return np.dtype(endian + base)


def _parse_scales(header_text: str, order: list[str]) -> list[float]:
    scales = []
    for ch in order:
        pattern = rf"{re.escape(ch)}.*?(?:Range|Scale)\s*[:=]\s*([0-9eE.+-]+)"
        match = re.search(pattern, header_text, re.IGNORECASE | re.DOTALL)
        if match:
            with contextlib.suppress(Exception):
                scales.append(float(match.group(1)))
                continue
        scales.append(1.0)
    return scales


def _read_data_block(fh, header: GBDHeader) -> np.ndarray:
    fh.seek(header.header_size)
    n_channels = len(header.order)
    expected = header.counts * n_channels
    raw = fh.read(expected * header.dtype.itemsize)
    array = np.frombuffer(raw, dtype=header.dtype, count=expected)
    if array.size != expected:
        raise ValueError("GBD data block shorter than expected")
    data = array.reshape((header.counts, n_channels))
    return data.astype(np.float64)


# -- registration

io_registry.register_reader("gbd", TimeSeriesDict, read_timeseriesdict_gbd)
io_registry.register_reader("gbd", TimeSeries, read_timeseries_gbd)
io_registry.register_reader("gbd", TimeSeriesMatrix, read_timeseriesmatrix_gbd)
io_registry.register_identifier(
    "gbd", TimeSeriesDict, lambda *args, **kwargs: str(args[1]).lower().endswith(".gbd")
)
io_registry.register_identifier(
    "gbd", TimeSeries, lambda *args, **kwargs: str(args[1]).lower().endswith(".gbd")
)
