#!/usr/bin/env python3
"""Check public docs navigation, downloadable examples, plots, and build identity."""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

ENTRY_PAGES = (
    "index.html",
    "search.html",
    "tutorials/index.html",
    "tutorials/getting_started.html",
    "tutorials/quickstart.html",
    "tutorials/first_analysis.html",
    "tutorials/commissioner.html",
    "tutorials/scientific_python.html",
    "tutorials/intro_timeseries.html",
    "how-to/io_formats.html",
    "how-to/interop.html",
    "reference/index.html",
    "reference/io_capabilities.html",
    "reference/interop_capabilities.html",
    "about/index.html",
    "about/documentation_version.html",
    "about/developer.html",
    "about/known_limitations.html",
    "how-to/case-studies/index.html",
)


LEGACY_ANCHORS = {
    "tutorials/intro_timeseries.html": (
        "environment-setup",
        "signal-processing-and-demodulation",
        "spectral-analysis-and-correlation",
        "hilbert-huang-transform-hht",
        "statistics-and-preprocessing",
        "resampling-and-reindexing",
        "function-fitting",
        "interoperability",
        "next-steps",
    ),
    "how-to/io_formats.html": tuple(
        "io-formats-" + lang + "-" + suffix
        for lang in ("en", "ja")
        for suffix in (
            "top",
            "quick",
            "basic",
            "a",
            "b",
            "c",
            "d",
            "dev",
            "supported-classes",
        )
    ),
    "how-to/interop.html": tuple(
        "interop-en-" + suffix
        for suffix in (
            "how-to-read",
            "foundation-layer",
            "status-labels",
            "storage-conversion",
            "analysis-conversion",
            "ml-conversion",
            "domain-conversion",
        )
    ),
}


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        key = "href" if tag == "a" else "src" if tag == "img" else None
        if key and values.get(key):
            self.links.append(str(values[key]))


def check(root: Path, expected_revision: str | None = None) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    parsed: dict[Path, Links] = {}

    def parse(path: Path) -> Links:
        if path not in parsed:
            result = Links()
            result.feed(path.read_text(encoding="utf-8"))
            parsed[path] = result
        return parsed[path]

    for language in ("", "ja/"):
        info_path = root / language / "build-info.json"
        if not info_path.exists():
            errors.append(f"Missing {info_path}")
            continue
        info = json.loads(info_path.read_text())
        if expected_revision and info["source_revision"] != expected_revision:
            errors.append(f"Wrong source revision in {info_path}")
        for name in ENTRY_PAGES:
            page = root / language / name
            if not page.exists():
                errors.append(f"Missing page: {language}{name}")
                continue
            text = page.read_text(encoding="utf-8")
            if (
                "gwexpy-build-status" not in text
                or info["source_revision"][:8] not in text
            ):
                errors.append(f"Missing build identity: {language}{name}")
            for anchor in LEGACY_ANCHORS.get(name, ()):
                if anchor == "io-formats-ja-supported-classes":
                    continue
                if anchor not in parse(page).ids:
                    errors.append(f"Missing legacy anchor: {language}{name}#{anchor}")
            counterpart = ("" if language else "ja/") + name
            expected_switch = info["language_baseurl"] + counterpart
            if expected_switch not in parse(page).links:
                errors.append(f"Missing language switch: {language}{name}")
            for href in parse(page).links:
                url = urlsplit(href)
                if href.startswith(info["language_baseurl"]):
                    href = "/gwexpy/docs/" + href.removeprefix(info["language_baseurl"])
                    url = urlsplit(href)
                elif url.scheme or url.netloc:
                    continue
                path = unquote(url.path)
                if path.startswith("/gwexpy/docs/"):
                    target = root / path.removeprefix("/gwexpy/docs/")
                elif path.startswith("/"):
                    target = root / path.lstrip("/")
                else:
                    target = (page.parent / path).resolve() if path else page
                if target.is_dir():
                    target /= "index.html"
                if not target.exists():
                    errors.append(f"{language}{name}: missing {href}")
                elif url.fragment and target.suffix == ".html":
                    if unquote(url.fragment) not in parse(target).ids:
                        errors.append(f"{language}{name}: missing anchor {href}")
        cases = [
            name.removesuffix(".ipynb")
            for name in info.get("notebook_sources", {})
            if name.startswith("how-to/case-studies/")
        ]
        if not cases:
            errors.append(f"Missing canonical case inventory: {language}")
        for notebook in cases:
            evidence = info.get("notebook_execution", {}).get(notebook, {})
            case_page = root / language / (notebook + ".html")
            if not evidence.get("succeeded") or evidence.get("runtime") is None:
                errors.append(f"Missing execution evidence: {language}{notebook}")
            elif "gwexpy-case-status" not in case_page.read_text(encoding="utf-8"):
                errors.append(f"Missing case conditions: {language}{notebook}")
        image = root / language / "_static/images/quickstart-asd.png"
        if not image.exists() or image.stat().st_size < 1000:
            errors.append(f"Missing Quickstart plot: {language}")
    return errors


def check_remote(base_url: str, expected_revision: str) -> list[str]:
    """Read back deployed identity and the introductory figure in both languages."""
    errors = []
    for language in ("", "ja/"):
        prefix = base_url.rstrip("/") + "/" + language
        try:

            def fetch(name: str) -> bytes:
                request = Request(
                    prefix + name + "?revision=" + expected_revision,
                    headers={"Cache-Control": "no-cache"},
                )
                with urlopen(request, timeout=20) as response:
                    return response.read()

            info = json.loads(fetch("build-info.json"))
            if info.get("source_revision") != expected_revision or info.get("dirty"):
                errors.append(
                    f"{prefix}: deployed revision is not the clean expected commit"
                )
            for name in ("index.html", "tutorials/quickstart.html"):
                page = fetch(name).decode("utf-8")
                if (
                    "gwexpy-build-status" not in page
                    or expected_revision[:8] not in page
                ):
                    errors.append(f"{prefix}{name}: missing expected build identity")
            figure = fetch("_static/images/quickstart-asd.png")
            if not figure.startswith(b"\x89PNG\r\n\x1a\n") or len(figure) < 1000:
                errors.append(f"{prefix}: missing Quickstart figure")
        except (URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_root", type=Path, nargs="?")
    parser.add_argument("--url", help="Read back a deployed site instead of local HTML")
    parser.add_argument("--expected-revision")
    args = parser.parse_args()
    if args.url:
        if not args.expected_revision:
            parser.error("--url requires --expected-revision")
        failures = check_remote(args.url, args.expected_revision)
    else:
        if not args.html_root:
            parser.error("provide html_root or --url")
        failures = check(args.html_root, args.expected_revision)
    if failures:
        raise SystemExit("\n".join(failures))
    print("Public docs: requested EN/JA checks passed")
