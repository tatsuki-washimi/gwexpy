"""Command-line entry point: ``python -m gwexpy.time``.

Runs GWpy's ``gwpy.time`` command-line tool unchanged.
"""

from __future__ import annotations

from gwpy.time.__main__ import (
    datetime,
    main,
    tconvert,
    tz,
)

__all__ = [
    "datetime",
    "main",
    "tconvert",
    "tz",
]

if __name__ == "__main__":  # pragma: no cover
    main()
