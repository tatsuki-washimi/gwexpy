"""Artifactless EN/JA quickstart qualification smoke."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _first_python_block(document: str) -> str:
    match = re.search(r"```python\n(.*?)```", document, flags=re.DOTALL)
    assert match, "quickstart must contain an executable Python fenced block"
    return match.group(1).replace(".show()", "")


def test_en_ja_quickstarts_are_present_and_first_example_executes() -> None:
    english = (ROOT / "docs/web/en/user_guide/quickstart.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs/web/ja/user_guide/quickstart.md").read_text(encoding="utf-8")
    assert "# Quickstart" in english and "3-line Quickstart" in english
    assert "# クイックスタート" in japanese and "3行で最初の図" in japanese
    namespace: dict[str, object] = {}
    exec(_first_python_block(english), namespace)
    assert namespace["ts"].__class__.__name__ == "TimeSeries"
