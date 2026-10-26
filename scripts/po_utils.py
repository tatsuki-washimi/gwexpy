"""Small standard-library reader for gettext PO catalogs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_FIELD = re.compile(r"^(msgctxt|msgid_plural|msgid|msgstr(?:\[(\d+)\])?)\s+(.*)$")


def read_po(path: Path) -> list[dict[str, object]]:
    """Read PO entries needed by repository checks without a Babel dependency."""
    entries: list[dict[str, object]] = []
    entry: dict[str, object] = {}
    plural: dict[int, str] = {}
    active: tuple[str, int | None] | None = None
    obsolete = False

    def append_value(field: str, index: int | None, value: str) -> None:
        if field == "plural":
            assert index is not None
            plural[index] = plural.get(index, "") + value
            return
        current = entry.get(field)
        entry[field] = str(current or "") + value

    def flush() -> None:
        nonlocal entry, plural, active, obsolete
        if entry and not obsolete:
            entry["plural"] = [plural[index] for index in sorted(plural)]
            entry["flags"] = set(entry.get("flags", set()))
            entries.append(entry)
        entry = {}
        plural = {}
        active = None
        obsolete = False

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#~"):
            obsolete = True
            active = None
            continue
        if line.startswith("#,"):
            entry.setdefault("flags", set()).update(
                flag.strip() for flag in line[3:].split(",") if flag.strip()
            )
            continue
        if line.startswith("#"):
            continue

        match = _FIELD.match(line)
        if match:
            name, plural_index, literal = match.groups()
            value = ast.literal_eval(literal)
            if not isinstance(value, str):
                raise ValueError(f"Expected a quoted string in {path}:{line_number}")
            if name.startswith("msgstr["):
                field = "plural"
                index = int(plural_index)
            else:
                field = "context" if name == "msgctxt" else name
                index = None
            entry.setdefault("context", None)
            entry.setdefault("msgid", "")
            entry.setdefault("msgid_plural", None)
            entry.setdefault("msgstr", "")
            entry.setdefault("line", line_number if field == "msgstr" else 0)
            if field in {"msgstr", "plural"}:
                entry["line"] = line_number
            append_value(field, index, value)
            active = (field, index)
            continue

        if line.startswith('"') and active is not None:
            value = ast.literal_eval(line)
            if not isinstance(value, str):
                raise ValueError(f"Expected a quoted string in {path}:{line_number}")
            append_value(*active, value)
            continue

        raise ValueError(f"Unrecognized PO syntax in {path}:{line_number}: {line}")

    flush()
    return entries
