"""
WIN (NIED) format reader for gwexpy (Fixed implementation based on ObsPy patch).

References:
    - http://www.eri.u-tokyo.ac.jp/people/nakagawa/win/
    - Shigeki NAKAGAWA and Aitaro KATO, "New Module for Reading WIN Format Data in ObsPy",
      Technical Research Report, Earthquake Research Institute, the University of Tokyo, No. 26, pp. 31-36, 2020.
      https://www.eri.u-tokyo.ac.jp/GIHOU/archive/26_031-036.pdf

This module intentionally deviates from ObsPy's legacy WIN reader behavior for the
edge-cases described in the paper above:

- 0.5-byte (4-bit) delta decoding:
  - Fix sign handling for the lower nibble.
  - Skip the unused last nibble when the number of deltas is odd (e.g. even sampling rate).
- 3-byte (24-bit) delta decoding:
  - Apply correct operator precedence and sign-preserving unpack/shift.

We also provide regression tests using a sample WIN file under ``tests/sample-data/gui/``.
"""

from __future__ import annotations

import struct
import warnings

import numpy as np
from gwpy.io import registry as io_registry

try:
    from obspy import Stream, Trace, UTCDateTime

    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def s4(v):
    if v & 0b1000:
        v = -((v - 1) ^ 0xF)
    return v


def _apply_4bit_deltas(output: list[int], sdata: bytes, n_deltas: int) -> None:
    """
    Apply n_deltas 4-bit signed deltas to the last value in ``output``.

    WIN 0.5-byte compression stores two signed 4-bit deltas per byte: upper nibble
    then lower nibble. When the number of deltas is odd (e.g. even sampling rate),
    the last nibble is unused and must be skipped. We therefore decode exactly
    ``n_deltas`` values rather than blindly consuming all nibbles.
    """
    remaining = int(n_deltas)
    for val_byte in sdata:
        # Upper nibble then lower nibble
        for shift in (4, 0):
            if remaining <= 0:
                return
            nib = (val_byte >> shift) & 0b1111
            output.append(output[-1] + s4(nib))
            remaining -= 1


def _read_win_fixed(filename, century="20"):
    """
    Reads a WIN file and returns a Stream object.
    Based on obspy.io.win.core._read_win but with patches applied.
    """
    output: dict[str, list[int]] = {}
    srates: dict[str, int] = {}

    # read win file
    with open(filename, "rb") as fpin:
        fpin.seek(0, 2)
        sz = fpin.tell()
        fpin.seek(0)
        leng = 0
        status0 = 0
        start = 0
        while leng < sz:
            pklen = fpin.read(4)
            if len(pklen) < 4:
                break
            leng = 4
            truelen = struct.unpack(">i", pklen)[0]
            if truelen == 0:
                break
            buff = fpin.read(6)
            leng += 6

            yy = f"{century}{buff[0]:02x}"
            mm = f"{buff[1]:x}"
            dd = f"{buff[2]:x}"
            hh = f"{buff[3]:x}"
            mi = f"{buff[4]:x}"
            sec = f"{buff[5]:x}"

            # Only create UTCDateTime if we actually need it / module exists
            # Logic flow: this function is only called via read_win_file -> registry -> only if HAS_OBSPY

            date = UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi), int(sec))
            if start == 0:
                start = date
            if status0 == 0:
                sdata = None
            while leng < truelen:
                buff = fpin.read(4)
                leng += 4
                flag = f"{buff[0]:02x}"
                chanum = f"{buff[1]:02x}"
                chanum = f"{flag}{chanum}"
                datawide = float(buff[2] >> 4)
                srate = int(buff[3])
                xlen = (srate - 1) * datawide
                if datawide == 0:
                    xlen = srate // 2
                    datawide = 0.5

                idata00 = fpin.read(4)
                leng += 4
                idata22 = struct.unpack(">i", idata00)[0]

                if chanum in output:
                    output[chanum].append(idata22)
                else:
                    output[chanum] = [idata22]
                    srates[chanum] = srate

                xlen = int(xlen)  # ensure int
                sdata = fpin.read(xlen)
                leng += xlen

                if len(sdata) < xlen:
                    fpin.seek(-(xlen - len(sdata)), 1)
                    sdata += fpin.read(xlen - len(sdata))
                    msg = "This shouldn't happen, it's weird..."
                    warnings.warn(msg)

                if datawide == 0.5:
                    # There are (srate - 1) deltas after the first absolute sample.
                    _apply_4bit_deltas(output[chanum], sdata, srate - 1)

                elif datawide == 1:
                    for i in range(int(xlen // datawide)):
                        val = np.frombuffer(sdata[i : i + 1], np.int8)[0]
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                elif datawide == 2:
                    for i in range(int(xlen // datawide)):
                        val = struct.unpack(">h", sdata[2 * i : 2 * (i + 1)])[0]
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                elif datawide == 3:
                    for i in range(int(xlen // datawide)):
                        # 24-bit signed big-endian delta: pad to 32-bit and shift back.
                        # Using signed unpack + arithmetic shift preserves sign.
                        chunk = sdata[3 * i : 3 * (i + 1)]
                        val = struct.unpack(">i", chunk + b"\x00")[0] >> 8
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                elif datawide == 4:
                    for i in range(int(xlen // datawide)):
                        val = struct.unpack(">i", sdata[4 * i : 4 * (i + 1)])[0]
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                else:
                    msg = (
                        f"DATAWIDE is {datawide} but only values of 0.5, 1, 2, 3 or 4 "
                        "are supported."
                    )
                    raise NotImplementedError(msg)

    traces = []
    for chan in output.keys():
        t = Trace(data=np.array(output[chan], dtype=np.int32))
        t.stats.channel = str(chan)
        t.stats.sampling_rate = float(srates[chan])
        t.stats.starttime = start
        traces.append(t)
    return Stream(traces=traces)


def read_win_file(source, **kwargs) -> TimeSeriesDict:
    """
    Read a WIN/WIN32 format file using fixed ObsPy-based reader.
    """
    if not HAS_OBSPY:
        raise ImportError("obspy is required to read WIN format files")

    stream = _read_win_fixed(source, **kwargs)

    # Merge if necessary (simple gap handling)
    stream.merge(method=1, fill_value=np.nan)

    # Convert to TimeSeriesDict
    tsd = TimeSeriesDict()
    for tr in stream:
        # Convert ObsPy trace to TimeSeries
        import datetime

        from gwexpy.io.utils import datetime_to_gps

        # Start time
        dt = tr.stats.starttime.datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        t0 = datetime_to_gps(dt)

        ts = TimeSeries(
            tr.data,
            t0=t0,
            sample_rate=tr.stats.sampling_rate,
            name=tr.id,
            channel=tr.id,
        )
        tsd[tr.id] = ts

    return tsd


# -- Registration
if HAS_OBSPY:
    for fmt in ["win", "win32"]:
        io_registry.register_reader(fmt, TimeSeriesDict, read_win_file, force=True)
        io_registry.register_reader(
            fmt,
            TimeSeries,
            lambda *a, **k: read_win_file(*a, **k)[
                next(iter(read_win_file(*a, **k).keys()))
            ],
            force=True,
        )
        io_registry.register_reader(
            fmt,
            TimeSeriesMatrix,
            lambda *a, **k: read_win_file(*a, **k).to_matrix(),
            force=True,
        )
        io_registry.register_identifier(
            fmt,
            TimeSeriesDict,
            lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"),
        )
        io_registry.register_identifier(
            fmt,
            TimeSeries,
            lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"),
        )
