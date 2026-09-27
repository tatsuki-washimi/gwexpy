"""Read-only public I/O probes; all generated files stay in .audit_tmp."""

import importlib.metadata as metadata
import json
import shutil
import sqlite3
import struct
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


ROOT = Path(__file__).resolve().parent
FIX = ROOT / "fixtures"
FIX.mkdir(exist_ok=True)


def emit(case, payload):
    print(json.dumps({"case": case, **payload}, default=str, allow_nan=True, sort_keys=True))


def versions():
    import gwexpy

    names = ("gwpy", "obspy", "numpy", "scipy", "pydub", "tinytag")
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    emit("VERSIONS", {"python": sys.version, "executable": sys.executable, "gwexpy": gwexpy.__version__, "source": gwexpy.__file__, "deps": found, "ffmpeg": shutil.which("ffmpeg")})


def summarize_dict(value):
    return {key: {"values": np.asarray(ts.value).tolist(), "shape": ts.shape, "dtype": str(ts.dtype), "t0": float(ts.t0.value), "dt": float(ts.dt.value), "unit": str(ts.unit), "name": ts.name, "channel": str(ts.channel)} for key, ts in value.items()}


def outcome(call):
    try:
        result = call()
        return {"result": summarize_dict(result), "provenance": getattr(result, "_gwexpy_io", None), "attrs": getattr(result, "attrs", None)}
    except Exception as exc:
        return {"exception": type(exc).__name__, "message": str(exc)}


def outcome_with_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = outcome(call)
    return {**result, "warnings": [str(w.message) for w in caught]}


def obspy_dup():
    try:
        import obspy
    except ImportError:
        emit("OBSPY-DUP-001", {"blocked": "NO_DEPENDENCY"})
        return

    def trace(data, when, rate=4):
        tr = obspy.Trace(data=np.asarray(data, dtype=np.int32))
        tr.stats.network = "XX"
        tr.stats.station = "TEST"
        tr.stats.location = ""
        tr.stats.channel = "BHZ"
        tr.stats.starttime = obspy.UTCDateTime(2024, 1, 1) + when
        tr.stats.sampling_rate = rate
        return tr

    orig_merge = obspy.Stream.merge
    for label, second_time, second_data, rate in (
        ("contiguous", 1.0, [5, 6, 7, 8], 4),
        ("gap", 1.5, [5, 6, 7, 8], 4),
        ("overlap_conflict", 0.5, [50, 60, 70, 80], 4),
        ("different_rate", 1.0, [5, 6, 7, 8], 8),
    ):
        first, second = trace([1, 2, 3, 4], 0), trace(second_data, second_time, rate)
        paths = [FIX / f"dup_{label}_{index}.mseed" for index in (0, 1)]
        first.write(str(paths[0]), format="MSEED")
        second.write(str(paths[1]), format="MSEED")
        native = obspy.read(str(paths[0]), format="MSEED") + obspy.read(str(paths[1]), format="MSEED")
        native_before = [(tr.id, str(tr.data.dtype), tr.stats.sampling_rate, tr.data.tolist()) for tr in native]
        try:
            for tr in native:
                tr.data = tr.data.astype(float)
            native.merge(method=1, fill_value=np.nan)
            native_result = [(tr.id, str(tr.data.dtype), tr.stats.sampling_rate, np.asarray(tr.data).tolist()) for tr in native]
        except Exception as exc:
            native_result = {"exception": type(exc).__name__, "message": str(exc)}
        merges = []

        def observed_merge(stream, *args, **kwargs):
            before = [(tr.id, str(tr.data.dtype), tr.stats.sampling_rate, len(tr.data)) for tr in stream]
            try:
                result = orig_merge(stream, *args, **kwargs)
                merges.append({"before": before, "after": [(tr.id, str(tr.data.dtype), tr.stats.sampling_rate, len(tr.data)) for tr in stream], "kwargs": kwargs})
                return result
            except Exception as exc:
                merges.append({"before": before, "exception": type(exc).__name__, "message": str(exc), "kwargs": kwargs})
                raise

        with patch.object(obspy.Stream, "merge", observed_merge):
            public = outcome(lambda: TimeSeriesDict.read([str(p) for p in paths], format="mseed"))
        emit("OBSPY-DUP-001", {"variant": label, "fixtures": [str(p) for p in paths], "native_before": native_before, "native_merge": native_result, "public": public, "public_merges": merges})


def sdb_kw():
    path = FIX / "epoch.sdb"
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL, outHumidity REAL)")
        conn.executemany("INSERT INTO archive VALUES (?, ?, ?)", [(1700000000, 70, 50), (1700000300, 71, 51), (1700000600, 72, 52)])
    from astropy.time import Time
    from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

    reads = {
        "direct_default": outcome(lambda: read_timeseriesdict_sdb(path)),
        "direct_epoch": outcome(lambda: read_timeseriesdict_sdb(path, epoch=999.0)),
        "registry_default": outcome(lambda: TimeSeriesDict.read(path, format="sdb")),
        "registry_epoch": outcome(lambda: TimeSeriesDict.read(path, format="sdb", epoch=999.0)),
        "registry_auto_epoch": outcome(lambda: TimeSeriesDict.read(path, epoch=999.0)),
    }
    emit("SDB-KW-001", {"fixture": str(path), "native_sqlite": [(1700000000, 70, 50), (1700000300, 71, 51), (1700000600, 72, 52)], "external_expected_gps_t0": Time(1700000000, format="unix").gps, "reads": reads})


def win_kw():
    path = FIX / "epoch.win"
    # One complete WIN packet: 2024-01-02 03:04:05, channel 0001, 2 Hz,
    # 1-byte signed delta, samples [10, 11].
    payload = bytes.fromhex("24010203040500011002") + struct.pack(">i", 10) + bytes([1])
    path.write_bytes(struct.pack(">i", len(payload) + 4) + payload)
    from gwexpy.timeseries.io.win import read_win_file

    def read(call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = outcome(call)
        return {**result, "warnings": [str(w.message) for w in caught]}

    emit("WIN-KW-001", {"fixture": str(path), "fixture_hex": path.read_bytes().hex(), "reads": {
        "direct_default": read(lambda: read_win_file(path)),
        "direct_epoch": read(lambda: read_win_file(path, epoch=999.0)),
        "registry_default": read(lambda: TimeSeriesDict.read(path, format="win")),
        "registry_epoch": read(lambda: TimeSeriesDict.read(path, format="win", epoch=999.0)),
        "registry_auto_epoch": read(lambda: TimeSeriesDict.read(path, epoch=999.0)),
    }})


def audio_controls():
    path = FIX / "stereo.wav"
    # Asymmetric channels and non-round sample count make order/rate visible.
    pcm = np.array([[1000, -2000], [2000, -1000], [3000, 0], [4000, 1000], [5000, 2000]], dtype=np.int16)
    wavfile.write(path, 8000, pcm)
    rate, native = wavfile.read(path)
    emit("WAV-CONTROL-001", {"fixture": str(path), "native_rate": rate, "native_shape": native.shape, "native_dtype": str(native.dtype), "native_values": native.tolist(), "reads": {
        "dict_default": outcome(lambda: TimeSeriesDict.read(path, format="wav")),
        "dict_auto": outcome(lambda: TimeSeriesDict.read(path)),
        "dict_epoch": outcome(lambda: TimeSeriesDict.read(path, format="wav", epoch=1234.0)),
        "dict_extract_metadata": outcome_with_warnings(lambda: TimeSeriesDict.read(path, format="wav", extract_metadata=True)),
        "direct_extract_metadata": outcome_with_warnings(lambda: __import__("gwexpy.timeseries.io.wav", fromlist=["read_timeseriesdict_wav"]).read_timeseriesdict_wav(path, extract_metadata=True)),
        "single": outcome(lambda: {"single": TimeSeries.read(path, format="wav")}),
    }})

    flac = FIX / "stereo.flac"
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(str(path)).export(str(flac), format="flac", tags={"title": "D audit title"})
        native_seg = AudioSegment.from_file(str(flac), format="flac")
        raw = np.array(native_seg.get_array_of_samples()).reshape(-1, native_seg.channels)
        native_info = {"rate": native_seg.frame_rate, "channels": native_seg.channels, "width": native_seg.sample_width, "raw_shape": raw.shape, "raw_values": raw.tolist()}
    except Exception as exc:
        native_info = {"exception": type(exc).__name__, "message": str(exc)}
    reads = {
        "dict_default": outcome(lambda: TimeSeriesDict.read(flac, format="flac")),
        "dict_auto": outcome(lambda: TimeSeriesDict.read(flac)),
        "dict_epoch": outcome(lambda: TimeSeriesDict.read(flac, format="flac", epoch=1234.0)),
        "dict_extract_metadata": outcome_with_warnings(lambda: TimeSeriesDict.read(flac, format="flac", extract_metadata=True)),
        "direct_extract_metadata": outcome_with_warnings(lambda: __import__("gwexpy.timeseries.io.audio", fromlist=["read_timeseriesdict_audio"]).read_timeseriesdict_audio(flac, format_hint="flac", extract_metadata=True)),
        "single": outcome(lambda: {"single": TimeSeries.read(flac, format="flac")}),
    }
    from gwexpy.io.utils import extract_audio_metadata
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed_metadata = extract_audio_metadata(flac)
    emit("AUDIO-CONTROL-001", {"fixture": str(flac), "native": native_info, "parsed_metadata": parsed_metadata, "parsed_metadata_warnings": [str(w.message) for w in caught], "reads": reads})


if __name__ == "__main__":
    versions()
    obspy_dup()
    sdb_kw()
    win_kw()
    audio_controls()
