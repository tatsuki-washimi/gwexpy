"""Domain-neutral coupling result interchange helpers."""

from __future__ import annotations

from .segment import (
    SCHEMA_NAME,
    from_json_envelope,
    from_result,
    from_results,
    to_astropy,
    to_json_envelope,
    to_pandas,
    validate,
)

__all__ = [
    "SCHEMA_NAME",
    "from_json_envelope",
    "from_result",
    "from_results",
    "to_astropy",
    "to_json_envelope",
    "to_pandas",
    "validate",
]
