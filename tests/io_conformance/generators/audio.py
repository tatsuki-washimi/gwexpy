"""Deterministic audio generator used by the IO conformance harness."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("tone.wav", "manifest.json")

_SAMPLE_RATE = 8_000
_DURATION_SECONDS = 0.05
_FREQUENCY_HZ = 440.0
_AMPLITUDE = 0.2


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a tiny deterministic sine-wave WAV fixture into *output_dir*."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / "tone.wav"
    manifest_path = output_dir / "manifest.json"

    sample_count = int(_SAMPLE_RATE * _DURATION_SECONDS)
    frames = bytearray()
    for index in range(sample_count):
        sample = int(
            32767
            * _AMPLITUDE
            * math.sin(2.0 * math.pi * _FREQUENCY_HZ * index / _SAMPLE_RATE)
        )
        frames.extend(struct.pack("<h", sample))

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(bytes(frames))

    manifest_path.write_text(
        json.dumps(
            {
                "duration_seconds": _DURATION_SECONDS,
                "files": list(GENERATED_FILES),
                "generator": "audio",
                "sample_rate_hz": _SAMPLE_RATE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "manifest": manifest_path,
        "wav": wav_path,
    }
