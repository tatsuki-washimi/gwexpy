"""
WIN (NIED) format reader for gwexpy (Fixed implementation based on ObsPy patch).

References:
    - http://www.eri.u-tokyo.ac.jp/people/nakagawa/win/
    - Shigeki NAKAGAWA and Aitaro KATO, "New Module for Reading WIN Format Data in ObsPy",
      Technical Research Report, Earthquake Research Institute, the University of Tokyo, No. 26, pp. 31-36, 2020.
      https://www.eri.u-tokyo.ac.jp/GIHOU/archive/26_031-036.pdf
"""

from __future__ import annotations

import struct
import warnings
import numpy as np
from obspy import Stream, Trace, UTCDateTime
from gwpy.io import registry as io_registry

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def s4(v):
    if v & 0b1000:
        v = -((v - 1) ^ 0xf)
    return v


def _read_win_fixed(filename, century="20"):
    """
    Reads a WIN file and returns a Stream object.
    Based on obspy.io.win.core._read_win but with patches applied.
    """
    output = {}
    srates = {}

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
            truelen = struct.unpack('>i', pklen)[0]
            if truelen == 0:
                break
            buff = fpin.read(6)
            leng += 6

            yy = "%s%02x" % (century, buff[0])
            mm = "%x" % buff[1]
            dd = "%x" % buff[2]
            hh = "%x" % buff[3]
            mi = "%x" % buff[4]
            sec = "%x" % buff[5]

            date = UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi),
                               int(sec))
            if start == 0:
                start = date
            if status0 == 0:
                sdata = None
            while leng < truelen:
                buff = fpin.read(4)
                leng += 4
                flag = '%02x' % buff[0]
                chanum = '%02x' % buff[1]
                chanum = "%02s%02s" % (flag, chanum)
                datawide = int('%x' % (buff[2] >> 4))
                srate = buff[3]
                xlen = (srate - 1) * datawide
                if datawide == 0:
                    xlen = srate // 2
                    datawide = 0.5

                idata00 = fpin.read(4)
                leng += 4
                idata22 = struct.unpack('>i', idata00)[0]

                if chanum in output:
                    output[chanum].append(idata22)
                else:
                    output[chanum] = [idata22, ]
                    srates[chanum] = srate
                
                xlen = int(xlen) # ensure int
                sdata = fpin.read(xlen)
                leng += xlen

                if len(sdata) < xlen:
                    fpin.seek(-(xlen - len(sdata)), 1)
                    sdata += fpin.read(xlen - len(sdata))
                    msg = "This shouldn't happen, it's weird..."
                    warnings.warn(msg)

                if datawide == 0.5:
                    for i in range(xlen):
                        val_byte = sdata[i] # int in Py3
                        
                        # Upper 4 bits
                        v_upper = (val_byte & 0b11110000) >> 4
                        idata2 = output[chanum][-1] + s4(v_upper)
                        output[chanum].append(idata2)
                        
                        if i == (xlen - 1):
                            break
                            
                        # Lower 4 bits
                        v_lower = val_byte & 0b00001111
                        idata2 = idata2 + s4(v_lower)
                        output[chanum].append(idata2)
                        
                elif datawide == 1:
                    for i in range(int(xlen // datawide)):
                        val = np.frombuffer(sdata[i:i + 1], np.int8)[0]
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                elif datawide == 2:
                    for i in range(int(xlen // datawide)):
                        val = struct.unpack('>h', sdata[2 * i:2 * (i + 1)])[0]
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                elif datawide == 3:
                    for i in range(int(xlen // datawide)):
                        # PATCH: Add parenthesis and sign extension for 3-byte int
                        chunk = sdata[3 * i:3 * (i + 1)] + b'\x00' # Pad to 4 bytes? No, b' ' is 0x20. 
                        # Wait, original code uses b' '. 
                        # struct.unpack('>i', chunk + b' ') ? No chunk is 3 bytes.
                        # Original: from_buffer(sdata[...] + b' ', '>i')[0] >> 8
                        # Assuming Big Endian 3 bytes -> Pad at END with something, unpack as 4 byte int, shift right 8.
                        # b' ' is 0x20. If we pad with 0x00 it changes nothing for >> 8 if positive?
                        # The 'chunk' variable was unused, directly unpack.
                        val_tmp = struct.unpack('>i', sdata[3 * i:3 * (i + 1)] + b' ')[0]
                        val = val_tmp >> 8
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                elif datawide == 4:
                    for i in range(int(xlen // datawide)):
                        val = struct.unpack('>i', sdata[4 * i:4 * (i + 1)])[0]
                        idata2 = output[chanum][-1] + val
                        output[chanum].append(idata2)
                else:
                    msg = ("DATAWIDE is %s but only values of 0.5, 1, 2, 3 or 4 "
                           "are supported.") % datawide
                    raise NotImplementedError(msg)

    traces = []
    for i in output.keys():
        t = Trace(data=np.array(output[i], dtype=np.int32))
        t.stats.channel = str(i)
        t.stats.sampling_rate = float(srates[i])
        t.stats.starttime = start
        traces.append(t)
    return Stream(traces=traces)


def read_win_file(source, **kwargs) -> TimeSeriesDict:
    """
    Read a WIN/WIN32 format file using fixed ObsPy-based reader.
    """
    stream = _read_win_fixed(source, **kwargs)
    
    # Merge if necessary (simple gap handling)
    stream.merge(method=1, fill_value=np.nan)

    # Convert to TimeSeriesDict
    tsd = TimeSeriesDict()
    for tr in stream:
        # Convert ObsPy trace to TimeSeries
        from gwexpy.io.utils import datetime_to_gps
        import datetime
        
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
for fmt in ["win", "win32"]:
    io_registry.register_reader(fmt, TimeSeriesDict, read_win_file, force=True)
    io_registry.register_reader(fmt, TimeSeries, lambda *a, **k: read_win_file(*a, **k)[next(iter(read_win_file(*a, **k).keys()))], force=True)
    io_registry.register_reader(fmt, TimeSeriesMatrix, lambda *a, **k: read_win_file(*a, **k).to_matrix(), force=True)
    io_registry.register_identifier(
        fmt, TimeSeriesDict,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"))
    io_registry.register_identifier(
        fmt, TimeSeries,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"))
