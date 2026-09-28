"""Deterministic B-F5 WIN fixtures and baseline contract metadata.

The fixtures are generated from the wire format, not through the decoder under
measurement. ``materialize`` writes them and returns a manifest containing the
SHA-256 of every byte stream and an exact, compact expected-output recipe.

Primary route: call ``warm_cpu_call(path, decoder)`` once before timing, then
time repeated calls to ``decoder(path)``. The pre-read warms the OS page cache;
the decoder's file open and packet parsing remain inside the timed call, as
required by ``_read_win_fixed(path)``'s path-based contract. Do not time fixture
construction or the pre-read.

Structural test design for F5: parse/inspect the optimized decoder and assert
the sample reconstruction path uses NumPy bulk operations (for example
``frombuffer``/``cumsum``/vectorized arithmetic) and has no Python loop whose
body appends one decoded sample at a time. Exercise every width and compare
the full arrays from this manifest against B1. The current B1 reader is expected
to fail that structural assertion; this module only defines its test contract.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

ORIGIN = (2026, 1, 2, 3, 4, 5)
WIDTHS = {0: "0.5-byte", 1: "1-byte", 2: "2-byte", 3: "3-byte", 4: "4-byte"}
PUBLIC_WARNING = {
    "category": "builtins.UserWarning",
    "message": "WIN header time is timezone-naive; interpreting as UTC (#632)",
}


def _bcd(value: int) -> int:
    return int(f"{value:02d}", 16)


def _packet(*records: bytes, second: int = ORIGIN[5]) -> bytes:
    year, month, day, hour, minute, _ = ORIGIN
    body = bytes(map(_bcd, (year % 100, month, day, hour, minute, second)))
    body += b"".join(records)
    return struct.pack(">i", len(body) + 4) + body


def _record(
    width: int,
    rate: int,
    absolute: int,
    deltas: tuple[int, ...],
    channel: int,
) -> bytes:
    if len(deltas) != rate - 1:
        raise ValueError("delta count must equal rate - 1")
    head = bytes((0x12, channel, (width << 4) | ((rate >> 8) & 15), rate & 255))
    if width == 0:
        nibbles = [delta & 15 for delta in deltas]
        if len(nibbles) & 1:
            nibbles.append(0)
        encoded = bytes(
            (nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles), 2)
        )
    else:
        encoded = b"".join(
            delta.to_bytes(width, "big", signed=True) for delta in deltas
        )
    return head + struct.pack(">i", absolute) + encoded


def _samples(absolute: int, deltas: tuple[int, ...]) -> list[int]:
    result = [absolute]
    for delta in deltas:
        result.append(result[-1] + delta)
    return result


def _case(
    name: str,
    payload: bytes,
    *,
    channel: str | None = None,
    rate: int | None = None,
    samples: list[int] | None = None,
    warning: dict[str, str] | None = PUBLIC_WARNING,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": payload,
        "expected": {
            "channel": channel,
            "sampling_rate": rate,
            "dtype": "int64" if samples is not None else None,
            "samples": samples,
            "warning": warning,
            "error": error,
        },
    }


def fixture_cases() -> list[dict[str, Any]]:
    """Return fresh deterministic valid and malformed wire fixtures."""
    cases = []
    # Cover every DATAWIDE code, including a high-rate nibble stream and rate 1.
    for width, rate, channel in ((0, 5, 1), (1, 5, 2), (2, 5, 3), (3, 5, 4), (4, 5, 5)):
        deltas = tuple((1, -2, 3, -1)[: rate - 1])
        absolute = 100 + width
        samples = _samples(absolute, deltas)
        cases.append(
            _case(
                f"width-{WIDTHS[width].replace('.', '-')}",
                _packet(_record(width, rate, absolute, deltas, channel)),
                channel=f"12{channel:02d}",
                rate=rate,
                samples=samples,
            )
        )

    rate = 4095
    deltas = tuple(1 if i % 2 == 0 else -1 for i in range(rate - 1))
    cases.append(
        _case(
            "width-1-rate-4095",
            _packet(_record(1, rate, 37, deltas, 6)),
            channel="1206",
            rate=rate,
            samples=_samples(37, deltas),
        )
    )
    cases.append(
        _case(
            "rate-one-width-4",
            _packet(_record(4, 1, -2_147_483_648, (), 7)),
            channel="1207",
            rate=1,
            samples=[-2_147_483_648],
        )
    )
    for label, absolute, delta in (
        ("positive-int32-overflow", 2_147_483_647, 1),
        ("negative-int32-overflow", -2_147_483_648, -1),
    ):
        cases.append(
            _case(
                label,
                _packet(_record(4, 2, absolute, (delta,), 8 if delta > 0 else 9)),
                channel="1208" if delta > 0 else "1209",
                rate=2,
                samples=[absolute, absolute + delta],
            )
        )

    # Unsupported width code 5 and representative malformed/truncated records.
    unsupported = bytes((0x12, 0x0A, 0x50, 0x02)) + struct.pack(">i", 0)
    cases.append(
        _case(
            "unsupported-width",
            _packet(unsupported),
            error={
                "category": "builtins.NotImplementedError",
                "message": "DATAWIDE is 5.0 but only values of 0.5, 1, 2, 3 or 4 are supported.",
            },
        )
    )
    valid_record = _record(1, 3, 10, (1, 1), 11)
    truncated_packet = _packet(valid_record)[:-1]
    cases.append(
        _case(
            "truncated-packet",
            truncated_packet,
            error={
                "category": "builtins.ValueError",
                "message": "truncated WIN packet payload",
            },
        )
    )
    truncated_channel = _packet(
        bytes((0x12, 0x0C, 0x10, 0x03)) + struct.pack(">i", 0) + b"\x01"
    )
    cases.append(
        _case(
            "truncated-channel",
            truncated_channel,
            error={
                "category": "builtins.ValueError",
                "message": "truncated WIN channel 120c payload",
            },
        )
    )
    return cases


def materialize(directory: str | Path) -> dict[str, Any]:
    """Write fixture files and return a JSON-serializable hashed manifest."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in fixture_cases():
        path = root / f"{case['name']}.win"
        path.write_bytes(case.pop("bytes"))
        expected = case["expected"]
        manifest_cases.append(
            {
                "name": case["name"],
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "expected": expected,
            }
        )
    return {
        "format": "B-F5 WIN baseline fixture manifest v1",
        "source": "deterministic local wire-format generator",
        "cases": manifest_cases,
        "primary_route": {
            "metric": "warm decoder CPU time",
            "prewarm": "read full fixture bytes once outside timed region",
            "timed_call": "decoder(path), including normal path open and packet parse",
        },
        "structural_contract": (
            "optimized sample reconstruction uses NumPy bulk operations; no Python "
            "per-sample accumulation/list append in a width decode loop"
        ),
    }


def write_manifest(directory: str | Path) -> Path:
    """Materialize fixtures and write their manifest into the same directory."""
    root = Path(directory)
    manifest = materialize(root)
    target = root / "manifest.json"
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def warm_cpu_call(path: str | Path, decoder: Callable[[str | Path], Any]) -> Any:
    """Prime file cache outside timing, then perform one ordinary decoder call."""
    with open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    return decoder(path)
