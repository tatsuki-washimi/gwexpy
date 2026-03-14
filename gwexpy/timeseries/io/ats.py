"""
Metronix ATS format reader for gwexpy.
Based on binary header parsing logic provided by user (ats2gwf.py).
"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from gwpy.io.registry import default_registry as io_registry
from gwpy.time import to_gps

from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    ensure_dependency,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from ._registration import register_timeseries_format

_ATS_HEADER_MIN_SIZE = 1024


def _decode_fixed_ascii(raw: bytes) -> str:
    # ATS fixed-size fields are not NUL-terminated (per Metronix programmer notes).
    return raw.decode("utf-8", errors="ignore").strip().strip("\x00").strip()


def _read_ats_header(fh) -> dict[str, Any]:
    """
    Read the ATS binary header.

    Notes
    -----
    Metronix "ATSHeader_80" (and related) fields are stored in Intel byte order
    (little endian). LSB is stored in mV/count.
    """
    header_len_bytes = fh.read(2)
    if len(header_len_bytes) != 2:
        raise ValueError("ATS file too short (missing header length)")

    header_length = int(np.frombuffer(header_len_bytes, dtype="<u2")[0])
    if header_length < _ATS_HEADER_MIN_SIZE:
        raise ValueError(f"ATS header length too small: {header_length}")

    fh.seek(0)
    header_bytes = fh.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("ATS file too short (incomplete header)")

    header_vers = int(np.frombuffer(header_bytes[0x02:0x04], dtype="<i2")[0])
    if header_vers == 1080:
        # CEA/sliced header: data start and per-slice packing are different (see Metronix docs).
        raise NotImplementedError(
            "ATS CEA/sliced header (version 1080) is not supported yet"
        )
    if header_vers not in (80, 81):
        raise ValueError(f"Unsupported ATS header version: {header_vers}")

    ui_samples = int(np.frombuffer(header_bytes[0x04:0x08], dtype="<u4")[0])
    ui_samples64 = int(np.frombuffer(header_bytes[0xF0:0xF8], dtype="<u8")[0])
    total_samples = ui_samples64 if ui_samples == 0xFFFFFFFF else ui_samples

    sample_freq = float(np.frombuffer(header_bytes[0x08:0x0C], dtype="<f4")[0])
    start_time_unix = int(np.frombuffer(header_bytes[0x0C:0x10], dtype="<u4")[0])
    lsb_mV = float(np.frombuffer(header_bytes[0x10:0x18], dtype="<f8")[0])

    adu_serial = int(np.frombuffer(header_bytes[0x20:0x22], dtype="<u2")[0])
    # system / ADC board serial exists at 0x22..0x24 but currently unused
    chan_type = _decode_fixed_ascii(header_bytes[0x26:0x28])
    sensor_type = _decode_fixed_ascii(header_bytes[0x28:0x2E])
    sensor_serial = int(np.frombuffer(header_bytes[0x2E:0x30], dtype="<i2")[0])
    system_type = _decode_fixed_ascii(header_bytes[0x84:0x90])

    bit_indicator = int(np.frombuffer(header_bytes[0xAA:0xAC], dtype="<i2")[0])
    if bit_indicator not in (0, 1):
        raise ValueError(f"Unsupported ATS bit_indicator: {bit_indicator}")

    data_dtype = np.dtype("<i8" if bit_indicator == 1 else "<i4")

    return {
        "header_length": header_length,
        "header_vers": header_vers,
        "total_samples": total_samples,
        "sample_freq": sample_freq,
        "start_time_unix": start_time_unix,
        "lsb_mV": lsb_mV,
        "adu_serial": adu_serial,
        "chan_type": chan_type,
        "sensor_type": sensor_type,
        "sensor_serial": sensor_serial,
        "system_type": system_type,
        "data_dtype": data_dtype,
    }


def read_timeseriesdict_ats(source, **kwargs):
    """
    Read a Metronix ATS file into a TimeSeriesDict.
    """
    ts = read_timeseries_ats(source, **kwargs)
    return TimeSeriesDict({ts.name: ts})


def read_timeseries_ats(
    source: str | Path,
    *,
    unit: str | u.Unit | None = None,
    epoch: float | datetime.datetime | None = None,
    **kwargs,
):
    """
    Read a Metronix ATS file into a TimeSeries.

    Parameters
    ----------
    source : str or Path
        Path to the ATS file, or an open file object.
    unit : str or Unit, optional
        Physical unit override (default: V).
    epoch : float or datetime, optional
        Override the start time (GPS seconds or datetime).
        If not provided, uses the timestamp from the ATS header.
    **kwargs
        Additional keyword arguments.
    """
    if hasattr(source, "read"):
        return _read_timeseries_ats_file(source, unit=unit, epoch=epoch, **kwargs)

    with open(source, mode="rb") as f:
        return _read_timeseries_ats_file(f, unit=unit, epoch=epoch, **kwargs)


def _read_timeseries_ats_file(f, *, unit=None, epoch=None, **kwargs):
    hdr = _read_ats_header(f)

    sample_freq = float(hdr["sample_freq"])
    start_time_unix = int(hdr["start_time_unix"])
    lsb_mV = float(hdr["lsb_mV"])
    header_length = int(hdr["header_length"])
    total_samples = int(hdr["total_samples"])
    data_dtype = hdr["data_dtype"]

    system_type = str(hdr["system_type"])
    adu_serial = int(hdr["adu_serial"])
    channel_type = str(hdr["chan_type"])
    sensor_type = str(hdr["sensor_type"])
    sensor_serial = int(hdr["sensor_serial"])

    # Construct Channel Name (stable, filename-independent)
    chname = f"Metronix_{system_type}_{adu_serial:03}_{channel_type}_{sensor_type}_{sensor_serial:04}"

    # Calculate t0
    if epoch is not None:
        # epoch オーバーライド
        if isinstance(epoch, (int, float)):
            t0 = float(epoch)
        elif isinstance(epoch, datetime.datetime):
            t0 = datetime_to_gps(epoch)
        else:
            raise TypeError(f"epoch must be float or datetime, got {type(epoch)}")
        epoch_source = "user"
    else:
        # ヘッダーから取得（既存の動作）
        dt_obj = datetime.datetime.fromtimestamp(start_time_unix, tz=datetime.UTC)
        t0 = to_gps(dt_obj)
        epoch_source = "ats_header"

    # Read Data
    f.seek(header_length)
    data_raw = np.fromfile(f, dtype=data_dtype, count=total_samples)
    if data_raw.size != total_samples:
        warnings.warn(
            f"ATS data block size mismatch: got {data_raw.size} samples, "
            f"expected {total_samples} (header). Using actual data size.",
            UserWarning,
            stacklevel=2,
        )

    # Scale Data
    # Per Metronix programmer notes: dblLSBMV * counts -> mV (gains already included).
    # Convert to V.
    data_scaled = data_raw.astype(np.float64) * lsb_mV / 1000.0

    ts = TimeSeries(
        data_scaled,
        sample_rate=sample_freq,
        t0=t0,
        name=chname,
        channel=chname,
        unit="V",  # 'Volt'
    )

    # unit 適用
    if unit is not None:
        ts = apply_unit(ts, unit)

    # provenance 記録
    set_provenance(
        ts,
        {
            "format": "ats",
            "epoch_source": epoch_source,
            "unit_source": "override" if unit else "ats_header",
        },
    )

    return ts


def read_timeseries_ats_mth5(source, **kwargs):
    """
    Read Metronix ATS/ATSS file using mth5 library.

    This function uses `mth5.io.metronix.metronix_atss.read_atss`.
    Note that mth5 enforces strict filename conventions (e.g. extension .atss, specific underscores)
    to parse metadata. If your file does not adhere to these conventions, use the default
    `read_timeseries_ats` which parses the binary header directly.
    """
    try:
        mth5 = ensure_dependency("mth5")
        metronix_atss = mth5.io.metronix.metronix_atss
    except ImportError:
        raise ImportError(
            "mth5 library is required to use this reader. Install it via pip."
        )

    # mth5 requires .atss extension?
    # Based on investigation, it checks .suffix in _get_file_type.
    # If source does not end in .atss, we might fail or need to symlink.
    # We will pass it as is and let mth5 handle (or fail).

    # mth5 read_atss returns a ChannelTS object (wrapping xarray)
    # or sometimes directly ChannelTS?
    # read_atss(fn) -> atss_obj.to_channel_ts()

    channel_ts = metronix_atss.read_atss(str(source))

    # Convert ChannelTS (xarray) to TimeSeries
    # channel_ts.ts is the xarray DataArray
    data_array = channel_ts.ts

    # Extract data
    data = data_array.data

    # Extract metadata
    # t0: start time. Obspy/MTH5 usually uses comparison-safe types.
    # xarray time index:
    # data_array.time[0]

    import pandas as pd
    from gwpy.time import to_gps

    # Attempt to get t0 from time index
    if len(data_array.time) > 0:
        # Convert numpy.datetime64/pandas timestamp to GPS
        t0_timestamp = pd.Timestamp(data_array.time.values[0])
        t0 = to_gps(t0_timestamp)
        dt = 1.0 / channel_ts.sample_rate
    else:
        # Fallback or error
        t0 = 0
        dt = 1.0

    name = channel_ts.channel_metadata.id or "mth5_channel"
    unit = channel_ts.channel_metadata.units

    return TimeSeries(data, t0=t0, dt=dt, name=name, unit=unit)


# -- Registration

register_timeseries_format(
    "ats",
    reader_dict=read_timeseriesdict_ats,
    reader_single=read_timeseries_ats,
    extension="ats",
)

# Register mth5 variant
io_registry.register_reader("ats.mth5", TimeSeries, read_timeseries_ats_mth5)
