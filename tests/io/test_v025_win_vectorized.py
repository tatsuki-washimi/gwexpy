"""WIN bulk-decoder behavior and structure contracts for v0.2.5."""

from __future__ import annotations

import inspect
import linecache
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import gwexpy.timeseries.io.win as win_io


def _record(width: int, absolute: int, deltas: tuple[int, ...]) -> bytes:
    """Encode one channel directly from WIN wire fields."""
    rate = len(deltas) + 1
    header = bytes((0x12, 0x34, (width << 4) | (rate >> 8), rate & 0xFF))
    if width == 0:
        nibbles = [delta & 0xF for delta in deltas]
        if len(nibbles) % 2:
            nibbles.append(0x7)  # Unused nibble must not become a sample.
        packed = bytes(
            (nibbles[index] << 4) | nibbles[index + 1]
            for index in range(0, len(nibbles), 2)
        )
    else:
        packed = b"".join(delta.to_bytes(width, "big", signed=True) for delta in deltas)
    return header + struct.pack(">i", absolute) + packed


def _packet(record: bytes) -> bytes:
    """Wrap a channel in a deterministic one-second WIN packet."""
    body = bytes((0x26, 0x01, 0x02, 0x03, 0x04, 0x05)) + record
    return struct.pack(">i", len(body) + 4) + body


@pytest.mark.parametrize(
    ("width", "absolute", "deltas"),
    [
        (0, 100, (7, -8, -1, 0, 1)),
        (1, 37, tuple(1 if index % 2 == 0 else -1 for index in range(4094))),
        (2, 100, (-32768, 32767, -1, 1)),
        (3, 100, (-8388608, 8388607, -1, 1)),
        (4, 2147483647, (2147483647, 1, -2147483648, -1)),
        (4, -2147483648, (-1,)),
        (4, -2147483648, ()),
    ],
)
def test_win_bulk_decode_exact_values_without_python_sample_append(
    tmp_path: Path, width: int, absolute: int, deltas: tuple[int, ...]
) -> None:
    pytest.importorskip("obspy")
    path = tmp_path / "bulk.win"
    path.write_bytes(_packet(_record(width, absolute, deltas)))
    win_source = str(inspect.getsourcefile(win_io._read_win_fixed))
    append_hits = 0

    def trace(frame, event, arg):
        nonlocal append_hits
        if event == "line" and frame.f_code.co_filename == win_source:
            line = linecache.getline(win_source, frame.f_lineno)
            if "samples.append(" in line or "output.append(" in line:
                append_hits += 1
        return trace

    sys.settrace(trace)
    try:
        stream = win_io._read_win_fixed(path)
    finally:
        sys.settrace(None)

    expected = [absolute]
    for delta in deltas:
        expected.append(expected[-1] + delta)
    assert len(stream) == 1
    trace = stream[0]
    assert trace.stats.channel == "1234"
    assert trace.stats.sampling_rate == float(len(expected))
    assert trace.data.dtype == np.dtype("int64")
    np.testing.assert_array_equal(trace.data, expected)
    assert append_hits == 0


@pytest.mark.parametrize(
    ("absolute", "delta"),
    [(2**63 - 1, 1), (-(2**63), -1)],
)
def test_win_cumsum_falls_back_before_int64_overflow(absolute: int, delta: int) -> None:
    accumulator = getattr(win_io, "_cumsum_win_samples", None)
    assert callable(accumulator)
    result = accumulator(absolute, np.array([delta], dtype=np.int64), 1)
    assert result == [absolute, absolute + delta]
