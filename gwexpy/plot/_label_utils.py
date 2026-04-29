"""Shared label-formatting helpers for plot modules."""

from __future__ import annotations

from typing import Any


def format_unit_text(unit: Any) -> str | None:
    """Return a displayable unit string, or None for unitless values."""
    if unit is None:
        return None

    try:
        unit_text = unit.to_string()
    except (AttributeError, ValueError):
        unit_text = str(unit)

    if not unit_text:
        return None
    return unit_text


def format_label_with_unit(label: Any, unit: Any) -> str | None:
    """Format an optional label with an optional unit."""
    unit_text = format_unit_text(unit)
    label_text = str(label) if label else ""

    if label_text and unit_text:
        return f"{label_text} [{unit_text}]"
    if label_text:
        return label_text
    if unit_text:
        return f"[{unit_text}]"
    return None


def format_quantity_label(label: str, unit: Any) -> str:
    """Format a value label without empty brackets for unitless data."""
    unit_text = format_unit_text(unit)
    if unit_text is None:
        return label
    return f"{label} [{unit_text}]"


def format_axis_label(name: Any, unit: Any) -> str:
    """Format an axis name with a unit when one is displayable."""
    unit_text = format_unit_text(unit)
    if unit_text:
        return f"{name} [{unit_text}]"
    return str(name)
