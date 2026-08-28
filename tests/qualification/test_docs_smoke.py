"""Installed-code and live-site contracts for both quickstart languages."""

from __future__ import annotations

import os
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUICKSTARTS = {
    "en": ROOT / "docs/web/en/user_guide/quickstart.md",
    "ja": ROOT / "docs/web/ja/user_guide/quickstart.md",
}
LIVE_URLS = {
    "en": "https://tatsuki-washimi.github.io/gwexpy/docs/tutorials/quickstart.html",
    "ja": "https://tatsuki-washimi.github.io/gwexpy/docs/ja/tutorials/quickstart.html",
}
MAX_LIVE_HTML = 2 * 1024 * 1024

pytestmark = pytest.mark.skipif(
    os.environ.get("GWEXPY_POST_RELEASE_QUALIFICATION") != "1",
    reason="post-release qualification is opt-in",
)


def _python_blocks(document: str) -> list[str]:
    return re.findall(r"^```python\s*\n(.*?)^```\s*$", document, flags=re.M | re.S)


def test_every_en_ja_quickstart_python_block_executes_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "show", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)
    for language, path in QUICKSTARTS.items():
        document = path.read_text(encoding="utf-8")
        blocks = _python_blocks(document)
        assert len(blocks) == 2, f"unexpected {language} Python block count"
        for index, source in enumerate(blocks):
            namespace: dict[str, object] = {
                "__name__": f"quickstart_{language}_{index}"
            }
            exec(compile(source, f"{path.name}:{index + 1}", "exec"), namespace)
            if index == 0:
                assert namespace["ts"].__class__.__name__ == "TimeSeries"
            else:
                assert namespace["tsd"].__class__.__name__ == "TimeSeriesDict"
                assert namespace["csd"].__class__.__name__ == "FrequencySeries"


class _PageContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_lang: str | None = None
        self.links: list[dict[str, str]] = []
        self.headings: list[str] = []
        self._heading: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "html":
            self.html_lang = values.get("lang")
        elif tag == "link":
            self.links.append(values)
        elif tag in {"h1", "h2"}:
            self._heading = []

    def handle_data(self, data: str) -> None:
        if self._heading is not None:
            self._heading.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2"} and self._heading is not None:
            self.headings.append(" ".join("".join(self._heading).split()))
            self._heading = None


@pytest.mark.parametrize("language", ("en", "ja"))
def test_live_quickstart_language_canonical_and_hreflang(language: str) -> None:
    url = LIVE_URLS[language]
    request = urllib.request.Request(
        url, headers={"User-Agent": "gwexpy-post-release-qualification/1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        assert response.geturl() == url
        payload = response.read(MAX_LIVE_HTML + 1)
        assert len(payload) <= MAX_LIVE_HTML
        document = payload.decode("utf-8")

    parser = _PageContractParser()
    parser.feed(document)
    assert parser.html_lang is not None
    assert parser.html_lang.casefold().startswith(language)
    expected_heading = "Quickstart" if language == "en" else "クイックスタート"
    assert any(expected_heading in heading for heading in parser.headings)

    canonical = [
        link.get("href")
        for link in parser.links
        if link.get("rel", "").casefold() == "canonical"
    ]
    assert canonical == [url]
    alternates = {
        link.get("hreflang"): link.get("href")
        for link in parser.links
        if link.get("rel", "").casefold() == "alternate"
        and link.get("hreflang") in {"en", "ja", "x-default"}
    }
    assert set(alternates) == {"en", "ja", "x-default"}
    assert alternates["en"] == LIVE_URLS["en"]
    assert alternates["ja"] == LIVE_URLS["ja"]
    assert alternates["x-default"] == LIVE_URLS["en"]
