"""WIN (NIED) format reader for gwexpy (Fixed implementation based on ObsPy patch).

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
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np

from gwexpy.io.utils import (
    _consume_warning_state,
    _make_warning_state,
    _reject_timezone_reinterpretation,
    ensure_dependency,
)
from gwexpy.timeseries.io._multi import expand_multi_source, read_multi_dict
from gwexpy.timeseries.io._registration import register_timeseries_format

try:
    obspy = ensure_dependency("obspy")
    Stream = obspy.Stream
    Trace = obspy.Trace
    UTCDateTime = obspy.UTCDateTime
    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False
    Stream = cast(Any, None)
    Trace = cast(Any, None)
    UTCDateTime = cast(Any, None)

from .. import TimeSeries, TimeSeriesDict

_WIN_UTC_WARNING = "WIN header time is timezone-naive; interpreting as UTC (#632)"


def _record_or_warn_utc_interpretation(marker: list[bool] | None) -> None:
    if marker is None:
        warnings.warn(_WIN_UTC_WARNING, UserWarning, stacklevel=3)
    else:
        marker[0] = True


def s4(v):
    """Convert a 4-bit nibble into a signed integer."""
    if v & 0b1000:
        v = -((v - 1) ^ 0xF)
    return v


def _apply_4bit_deltas(output: list[int], sdata: bytes, n_deltas: int) -> None:
    """Apply n_deltas 4-bit signed deltas to the last value in ``output``.

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


def _read_win_fixed(filename: str | Path, century="20"):
    """Read a WIN file and return a ``Stream`` object.

    Based on obspy.io.win.core._read_win but with patches applied.

    Parameters
    ----------
    filename : str or Path
        Path to the WIN file.
    century : str, optional
        Century prefix (default: "20").

    """
    output: dict[str, list[int]] = {}
    srates: dict[str, int] = {}
    starts: dict[str, Any] = {}
    last_packet_by_channel: dict[str, int] = {}
    packet_times: list[Any] = []

    # Decode each packet from an exact, bounded payload.  Channel lengths must
    # never consume bytes belonging to the following packet.
    with open(filename, "rb") as fpin:
        packet_index = 0
        while True:
            pklen = fpin.read(4)
            if not pklen:
                break
            if len(pklen) != 4:
                raise ValueError("truncated WIN packet length header")

            truelen = struct.unpack(">i", pklen)[0]
            if truelen == 0:
                break
            if truelen < 10:
                raise ValueError(f"invalid WIN packet length: {truelen}")

            payload = fpin.read(truelen - 4)
            if len(payload) != truelen - 4:
                raise ValueError("truncated WIN packet payload")

            timestamp = payload[:6]
            yy = f"{century}{timestamp[0]:02x}"
            mm = f"{timestamp[1]:x}"
            dd = f"{timestamp[2]:x}"
            hh = f"{timestamp[3]:x}"
            mi = f"{timestamp[4]:x}"
            sec = f"{timestamp[5]:x}"
            date = UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi), int(sec))
            packet_times.append(date)

            offset = 6
            packet_channels: set[str] = set()
            while offset < len(payload):
                remaining = len(payload) - offset
                if remaining < 4:
                    raise ValueError("overlong WIN packet payload")

                buff = payload[offset : offset + 4]
                offset += 4
                chanum = f"{buff[0]:02x}{buff[1]:02x}"
                if chanum in packet_channels:
                    raise ValueError(
                        f"WIN channel {chanum} occurs more than once in packet"
                    )
                packet_channels.add(chanum)

                width_code = buff[2] >> 4
                srate = ((buff[2] & 0x0F) << 8) | buff[3]
                if srate == 0:
                    raise ValueError("WIN sample rate must be positive")
                if width_code > 4:
                    msg = (
                        f"DATAWIDE is {float(width_code)} but only values of 0.5, "
                        "1, 2, 3 or 4 are supported."
                    )
                    raise NotImplementedError(msg)

                xlen = srate // 2 if width_code == 0 else (srate - 1) * width_code
                if len(payload) - offset < 4 + xlen:
                    raise ValueError(f"truncated WIN channel {chanum} payload")

                absolute = struct.unpack(">i", payload[offset : offset + 4])[0]
                offset += 4
                sdata = payload[offset : offset + xlen]
                offset += xlen

                if chanum in srates and srates[chanum] != srate:
                    raise ValueError(
                        f"WIN channel {chanum} sample rate changed from "
                        f"{srates[chanum]} to {srate}"
                    )
                if (
                    chanum in last_packet_by_channel
                    and last_packet_by_channel[chanum] != packet_index - 1
                ):
                    raise ValueError(
                        f"WIN channel {chanum} reappears after an internal packet gap"
                    )

                samples = [absolute]
                if width_code == 0:
                    _apply_4bit_deltas(samples, sdata, srate - 1)
                elif width_code == 1:
                    for raw in sdata:
                        delta = np.frombuffer(bytes([raw]), np.int8)[0]
                        samples.append(samples[-1] + delta)
                elif width_code == 2:
                    for i in range(srate - 1):
                        delta = struct.unpack(">h", sdata[2 * i : 2 * (i + 1)])[0]
                        samples.append(samples[-1] + delta)
                elif width_code == 3:
                    for i in range(srate - 1):
                        chunk = sdata[3 * i : 3 * (i + 1)]
                        delta = struct.unpack(">i", chunk + b"\x00")[0] >> 8
                        samples.append(samples[-1] + delta)
                else:
                    for i in range(srate - 1):
                        delta = struct.unpack(">i", sdata[4 * i : 4 * (i + 1)])[0]
                        samples.append(samples[-1] + delta)

                if len(samples) != srate:
                    raise ValueError(
                        f"WIN channel {chanum} decoded sample count {len(samples)} "
                        f"does not match declared sample rate {srate}"
                    )

                if chanum not in output:
                    output[chanum] = []
                    srates[chanum] = srate
                    starts[chanum] = date
                output[chanum].extend(samples)
                last_packet_by_channel[chanum] = packet_index

            packet_index += 1

    for previous, current in zip(packet_times, packet_times[1:], strict=False):
        difference = current - previous
        if difference == 0:
            raise ValueError(f"duplicate WIN packet timestamp: {current}")
        if difference < 0:
            raise ValueError(
                f"backward WIN packet timestamp: {current} follows {previous}"
            )
        if difference != 1:
            raise ValueError(f"gap in WIN packet timestamps: {previous} to {current}")

    traces = []
    for chan in output.keys():
        t = Trace(data=np.array(output[chan], dtype=np.int32))
        t.stats.channel = str(chan)
        t.stats.sampling_rate = float(srates[chan])
        t.stats.starttime = starts[chan]
        traces.append(t)
    return Stream(traces=traces)


def read_win_file(source, **kwargs) -> TimeSeriesDict:
    """Read one or more WIN or WIN32 files with the patched ObsPy-based reader.

    When a list of paths is given, channels found in several files are
    concatenated along the time axis (gaps padded with NaN).
    """
    warning_marker = _consume_warning_state(
        kwargs,
        "_utc_warning_state",
        "_utc_warning_marker",
    )
    timezone = kwargs.pop("timezone", None)
    kwargs.pop("epoch", None)
    _reject_timezone_reinterpretation("win", timezone, None)

    if not HAS_OBSPY:
        raise ImportError("obspy is required to read WIN format files")

    multi = expand_multi_source(source)
    if multi is not None:
        top_level_marker = [False]
        result = read_multi_dict(
            partial(
                read_win_file,
                _utc_warning_state=_make_warning_state(top_level_marker),
            ),
            multi,
            "win",
            **kwargs,
        )
        if top_level_marker[0]:
            _record_or_warn_utc_interpretation(warning_marker)
        return result

    _record_or_warn_utc_interpretation(warning_marker)

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
            dt = dt.replace(tzinfo=datetime.UTC)
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
        register_timeseries_format(
            fmt,
            reader_dict=read_win_file,
            extension=fmt,
        )
