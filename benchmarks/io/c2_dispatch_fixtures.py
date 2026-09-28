"""Deterministic fixtures and observation helpers for the C2 read dispatch lane.

Run ``python benchmarks/io/c2_dispatch_fixtures.py OUTPUT_DIR`` to create the
fixture set and a JSON manifest. ObsPy fixtures are included only when ObsPy
is installed. This module deliberately records behavior; it is not a timing
harness and does not freeze a baseline by itself.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import warnings
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fixtures(directory: str | Path) -> dict[str, Any]:
    """Write small deterministic multi-channel fixtures and hash manifest."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "channels.csv": ("0.0,1.0,10.0\n0.1,2.0,20.0\n0.2,3.0,30.0\n").encode("ascii"),
        "headered.csv": b"time,alpha,beta\n0.0,1.0,10.0\n",
        # The unselected second value is malformed while ch1 remains valid.
        "unselected_malformed.csv": (
            "0.0,1.0,10.0\n0.1,2.0,NOT_A_NUMBER\n0.2,3.0,30.0\n"
        ).encode("ascii"),
        # Selected payload fault for the same channel order and row position.
        "selected_malformed.csv": ("0.0,1.0,10.0\n0.1,BAD,20.0\n0.2,3.0,30.0\n").encode(
            "ascii"
        ),
    }
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = root / name
        path.write_bytes(payload)
        paths[name] = path

    large_csv = root / "channels_large.csv"
    with large_csv.open("w", encoding="ascii", newline="\n") as output:
        for index in range(65536):
            output.write(f"{index / 10:.1f},{index % 1024},{(index * 7) % 4096}\n")
    paths[large_csv.name] = large_csv

    # Deterministic stereo PCM. scipy's WAV reader returns an interleaved
    # (frames, channels) array, so channel selection cannot avoid backend decode.
    wav_path = root / "stereo.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(10)
        for left, right in ((100, 1000), (200, 2000), (300, 3000)):
            output.writeframesraw(left.to_bytes(2, "little", signed=True))
            output.writeframesraw(right.to_bytes(2, "little", signed=True))
    paths[wav_path.name] = wav_path
    truncated_wav = root / "stereo_truncated.wav"
    truncated_wav.write_bytes(wav_path.read_bytes()[:-3])
    paths[truncated_wav.name] = truncated_wav

    large_wav = root / "stereo_large.wav"
    with wave.open(str(large_wav), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(1024)
        payload = bytearray()
        for index in range(1_048_576):
            payload.extend(struct.pack("<hh", index % 1000, (index * 7) % 30000))
        output.writeframes(payload)
    paths[large_wav.name] = large_wav

    optional: dict[str, str] = {}
    try:
        import numpy as np
        import obspy

        stream = obspy.Stream()
        for channel, data in (("BHZ", [1, 2, 3]), ("BHN", [10, 20, 30])):
            trace = obspy.Trace(data=np.asarray(data, dtype=np.int32))
            trace.stats.network = "XX"
            trace.stats.station = "FIX"
            trace.stats.location = "00"
            trace.stats.channel = channel
            trace.stats.sampling_rate = 10.0
            stream.append(trace)
        mseed = root / "two_channels.mseed"
        stream.write(str(mseed), format="MSEED", encoding="STEIM2")
        paths[mseed.name] = mseed
        large_stream = obspy.Stream()
        for channel, offset in (("BHZ", 0), ("BHN", 10000)):
            trace = obspy.Trace(data=np.arange(131072, dtype=np.int32) + offset)
            trace.stats.network = "XX"
            trace.stats.station = "FIX"
            trace.stats.location = "00"
            trace.stats.channel = channel
            trace.stats.sampling_rate = 1024.0
            large_stream.append(trace)
        large_mseed = root / "two_channels_large.mseed"
        large_stream.write(str(large_mseed), format="MSEED", encoding="STEIM2")
        paths[large_mseed.name] = large_mseed
        broken_mseed = root / "malformed.mseed"
        broken_mseed.write_bytes(b"\x00" * 128)
        paths[broken_mseed.name] = broken_mseed
        optional["obspy"] = "available"
    except ImportError:
        optional["obspy"] = "unavailable; miniSEED fixture omitted"

    cases = [
        {
            "name": "public_single_all",
            "file": "channels.csv",
            "call": "TimeSeries.read(path, format='csv')",
            "selection": None,
        },
        {
            "name": "public_single_first_channel",
            "file": "channels.csv",
            "call": "TimeSeries.read(path, format='csv')",
            "selection": None,
            "expected": "first value column",
        },
        {
            "name": "direct_csv_all_channels",
            "file": "channels.csv",
            "call": "read_timeseriesdict_csv(path)",
            "selection": None,
            "expected": "ch1,ch2 in file order",
        },
        {
            "name": "requested_valid",
            "file": "channels.csv",
            "call": "TimeSeriesDict.read(path, format='csv', channels=['ch1'])",
            "selection": "ch1",
        },
        {
            "name": "unselected_valid",
            "file": "channels.csv",
            "call": "TimeSeriesDict.read(path, format='csv', channels=['ch1'])",
            "selection": "ch1; ch2 is unselected",
        },
        {
            "name": "selected_malformed",
            "file": "selected_malformed.csv",
            "call": "TimeSeriesDict.read(path, format='csv', channels=['ch1'])",
            "selection": "ch1",
        },
        {
            "name": "unselected_malformed",
            "file": "unselected_malformed.csv",
            "call": "TimeSeriesDict.read(path, format='csv', channels=['ch1'])",
            "selection": "ch1; ch2 is unselected",
        },
        {
            "name": "headered_error",
            "file": "headered.csv",
            "call": "TimeSeries.read(path, format='csv')",
            "selection": None,
        },
        {
            "name": "csv_single_selected",
            "file": "channels.csv",
            "call": "TimeSeries.read(path, format='csv', channels=['ch2'])",
            "selection": "ch2",
        },
        {
            "name": "csv_single_unselected_malformed",
            "file": "unselected_malformed.csv",
            "call": "TimeSeries.read(path, format='csv', channels=['ch1'])",
            "selection": "ch1; ch2 is malformed",
        },
        {
            "name": "csv_single_selected_malformed",
            "file": "selected_malformed.csv",
            "call": "TimeSeries.read(path, format='csv', channels=['ch1'])",
            "selection": "ch1 is malformed",
        },
        {
            "name": "large_csv_selected",
            "file": "channels_large.csv",
            "call": "TimeSeries.read(path, format='csv', channels=['ch1'])",
            "selection": "ch1",
        },
        {
            "name": "generic_adapter_selected",
            "file": "channels.csv",
            "call": "TimeSeries.read(path, format='c2synthetic', channels=['second'])",
            "selection": "second",
            "note": "isolated synthetic dict reader counts exact requested-channel backend calls",
        },
        {
            "name": "generic_adapter_first",
            "file": "channels.csv",
            "call": "TimeSeries.read(path, format='c2synthetic')",
            "selection": None,
            "note": "isolated synthetic dict reader returns first then second in stable order",
        },
        {
            "name": "large_wav_selected",
            "file": "stereo_large.wav",
            "call": "TimeSeries.read(path, format='wav', channels=['channel_1'])",
            "selection": "channel_1",
            "backend_selection_supported": False,
        },
        {
            "name": "wav_interleaved_selection",
            "file": "stereo.wav",
            "call": "TimeSeries.read(path, format='wav', channels=['channel_1'])",
            "selection": "channel_1",
            "backend_selection_supported": False,
            "note": "scipy.io.wavfile.read returns all interleaved channels; assert output filtering only, never zero backend reads for channel_0",
        },
        {
            "name": "wav_no_selection_first",
            "file": "stereo.wav",
            "call": "TimeSeries.read(path, format='wav')",
            "selection": None,
            "expected": "channel_0 / first file channel",
        },
        {
            "name": "wav_dict_selected",
            "file": "stereo.wav",
            "call": "TimeSeriesDict.read(path, format='wav', channels=['channel_1'])",
            "selection": "channel_1",
            "backend_selection_supported": False,
        },
        {
            "name": "wav_direct_selected",
            "file": "stereo.wav",
            "call": "read_timeseriesdict_wav(path, channels=['channel_1'])",
            "selection": "channel_1",
            "backend_selection_supported": False,
        },
        {
            "name": "wav_truncated_selected",
            "file": "stereo_truncated.wav",
            "call": "TimeSeries.read(path, format='wav', channels=['channel_1'])",
            "selection": "channel_1",
            "backend_selection_supported": False,
            "fault": "RIFF declares three stereo frames; file is truncated by three bytes",
        },
        {
            "name": "wav_truncated_no_selection",
            "file": "stereo_truncated.wav",
            "call": "TimeSeries.read(path, format='wav')",
            "selection": None,
            "backend_selection_supported": False,
            "fault": "same truncated RIFF source without a selector",
        },
    ]
    if "two_channels.mseed" in paths:
        cases.extend(
            [
                {
                    "name": "obspy_selected",
                    "file": "two_channels.mseed",
                    "call": "TimeSeries.read(path, format='mseed', channels=['XX.FIX.00.BHN'])",
                    "selection": "XX.FIX.00.BHN",
                },
                {
                    "name": "obspy_no_selection_first",
                    "file": "two_channels.mseed",
                    "call": "TimeSeries.read(path, format='mseed')",
                    "selection": None,
                    "expected": "first stream trace",
                },
                {
                    "name": "direct_obspy_reader_selected",
                    "file": "two_channels.mseed",
                    "call": "read_miniseed_timeseriesdict(path, channels=['XX.FIX.00.BHN'])",
                    "selection": "XX.FIX.00.BHN",
                },
                {
                    "name": "obspy_selected_malformed_payload",
                    "file": "malformed.mseed",
                    "call": "TimeSeries.read(path, format='mseed', channels=['XX.FIX.00.BHN'])",
                    "selection": "XX.FIX.00.BHN",
                },
                {
                    "name": "obspy_unselected_malformed_payload",
                    "file": "malformed.mseed",
                    "call": "TimeSeries.read(path, format='mseed', channels=['XX.FIX.00.BHZ'])",
                    "selection": "XX.FIX.00.BHZ (not present in malformed stream)",
                    "note": "ObsPy reads/parses the source stream before GWexpy channel filtering; preserve its exact warning/error output for both selections",
                },
                {
                    "name": "large_obspy_selected",
                    "file": "two_channels_large.mseed",
                    "call": "TimeSeries.read(path, format='mseed', channels=['XX.FIX.00.BHN'])",
                    "selection": "XX.FIX.00.BHN",
                },
                {
                    "name": "obspy_dependency_missing",
                    "file": "two_channels.mseed",
                    "call": "TimeSeries.read(path, format='mseed') with seismic.ensure_dependency raising ImportError",
                    "selection": None,
                    "note": "deterministic synthetic optional-dependency failure in isolated worker",
                },
            ]
        )
    manifest = {
        "schema": "gwexpy-c2-dispatch-fixtures-v1",
        "purpose": "B-C2 characterization inputs; not a frozen baseline",
        "optional_backends": optional,
        "cases": cases,
        "files": {
            name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for name, path in sorted(paths.items())
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _array_fingerprint(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(getattr(value, "value", value))
    contiguous = np.ascontiguousarray(array)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(contiguous.view("uint8")).hexdigest(),
    }
    for attribute in ("t0", "dt", "unit", "channel", "name"):
        item = getattr(value, attribute, None)
        if item is not None:
            result[attribute] = str(item)
    return result


def correctness_fingerprint(value: Any) -> dict[str, Any]:
    """Return a stable JSON-compatible fingerprint for Series/Dict results."""
    if hasattr(value, "items"):
        return {
            "channels": [
                [str(key), _array_fingerprint(item)] for key, item in value.items()
            ]
        }
    return _array_fingerprint(value)


def capture_route(call: Callable[[], Any]) -> dict[str, Any]:
    """Capture result fingerprint or exact warning/error category and message."""
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        try:
            result = call()
        except BaseException as exc:
            return {
                "status": "error",
                "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "error_message": str(exc),
                "cause_type": (
                    f"{type(exc.__cause__).__module__}.{type(exc.__cause__).__qualname__}"
                    if exc.__cause__
                    else None
                ),
                "cause_message": str(exc.__cause__) if exc.__cause__ else None,
                "warnings": [_warning_record(item) for item in observed],
            }
    return {
        "status": "ok",
        "fingerprint": correctness_fingerprint(result),
        "warnings": [_warning_record(item) for item in observed],
    }


def _warning_record(item: warnings.WarningMessage) -> dict[str, str]:
    return {
        "category": f"{item.category.__module__}.{item.category.__qualname__}",
        "message": str(item.message),
    }


@contextmanager
def spy_backend_reads(
    owner: Any, attribute: str
) -> Iterator[list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    """Spy on a backend read callable without changing its result or exceptions.

    Patch the narrow backend symbol actually called by the route, for example
    ``gwexpy.timeseries.io.netcdf4_.Dataset``'s variable getter adapter. Counts
    are per invocation with original args/kwargs retained for audit. Define a
    backend-specific wrapper at the call site when the backend API is an object
    method; this helper intentionally does not guess those differing APIs.
    """
    original = getattr(owner, attribute)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    setattr(owner, attribute, spy)
    try:
        yield calls
    finally:
        setattr(owner, attribute, original)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python c2_dispatch_fixtures.py OUTPUT_DIR")
    print(json.dumps(write_fixtures(sys.argv[1]), indent=2, sort_keys=True))
