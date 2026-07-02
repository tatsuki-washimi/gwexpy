from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .contract import ScenarioName, ScenarioStatus

__all__ = ["summarize_blocking_rows"]


def summarize_blocking_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, int | str]:
    """Summarize blocking scenario rows for CI reporting."""

    blocking_rows = [
        row for row in rows if row.get("scenario") == ScenarioName.BLOCKING.value
    ]
    blocking_blocked = sum(
        row.get("status") == ScenarioStatus.BLOCKED.value for row in blocking_rows
    )
    blocking_total = len(blocking_rows)
    return {
        "blocking_blocked": blocking_blocked,
        "blocking_total": blocking_total,
        "blocking_display": f"{blocking_blocked} blocked / {blocking_total} total",
    }
