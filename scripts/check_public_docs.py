#!/usr/bin/env python3
"""Check public docs navigation, downloadable examples, plots, and build identity."""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urljoin, urlsplit
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
    "how-to/index.html",
    "how-to/monitoring/index.html",
    "how-to/calibration/index.html",
    "how-to/monitoring/long_term_trend.html",
    "how-to/monitoring/event_catalog_timeseries.html",
    "how-to/monitoring/chunked_long_data.html",
    "how-to/spectral/resonance_discovery_q.html",
    "how-to/control/control_frd_roundtrip.html",
    "how-to/calibration/calibration_units_contract.html",
    "how-to/interop/root_to_python_migration.html",
)

AUDIENCE_ROUTES = (
    "tutorials/first_analysis.html",
    "tutorials/first_analysis.html#for-gw-experimentalists",
    "tutorials/commissioner.html",
    "tutorials/scientific_python.html",
    "explanation/gwexpy_for_gwpy_users.html",
    "how-to/index.html",
)

# Nine logical workflow pages published by the monitoring / spectral /
# control / calibration / interop program (2 landing + 7 tutorial pages).
WORKFLOW_PAGES = (
    "how-to/monitoring/index.html",
    "how-to/monitoring/long_term_trend.html",
    "how-to/monitoring/event_catalog_timeseries.html",
    "how-to/monitoring/chunked_long_data.html",
    "how-to/spectral/resonance_discovery_q.html",
    "how-to/control/control_frd_roundtrip.html",
    "how-to/calibration/index.html",
    "how-to/calibration/calibration_units_contract.html",
    "how-to/interop/root_to_python_migration.html",
)

# Tutorial workflow pages that must expose a downloadable notebook.
WORKFLOW_NOTEBOOK_PAGES = {
    "how-to/monitoring/long_term_trend.html": "long_term_trend.ipynb",
    "how-to/monitoring/event_catalog_timeseries.html": "event_catalog_timeseries.ipynb",
    "how-to/monitoring/chunked_long_data.html": "chunked_long_data.ipynb",
    "how-to/spectral/resonance_discovery_q.html": "resonance_discovery_q.ipynb",
    "how-to/control/control_frd_roundtrip.html": "control_frd_roundtrip.ipynb",
    "how-to/calibration/calibration_units_contract.html": "calibration_units_contract.ipynb",
    "how-to/interop/root_to_python_migration.html": "root_to_python_migration.ipynb",
}


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


def has_japanese_text(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text)


def is_notebook_payload(data: bytes) -> bool:
    """Return True when downloaded bytes look like a Jupyter notebook."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("cells"), list)


def workflow_page_errors(
    *,
    page_html: str,
    language: str,
    expected_revision: str,
    counterpart_url: str,
) -> list[str]:
    """Pure content checks for one rendered workflow page (EN or JA).

    Each page links only its counterpart language (the theme renders a
    single switch link), so only the counterpart URL is required here.
    """
    errors: list[str] = []
    if "gwexpy-build-status" not in page_html or expected_revision[:8] not in page_html:
        errors.append("missing expected build identity")
    parsed = Links()
    parsed.feed(page_html)
    if counterpart_url not in parsed.links:
        errors.append("missing language switch link")
    if language == "ja/" and not has_japanese_text(page_html):
        errors.append("JA page lacks Japanese translation")
    return errors


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

        # Check workflow pages and JA translation presence
        for wp in WORKFLOW_PAGES:
            wp_file = root / language / wp
            if not wp_file.exists() or wp_file.stat().st_size < 500:
                errors.append(f"Missing workflow HTML page: {language}{wp}")
            elif language == "ja/":
                content = wp_file.read_text(encoding="utf-8")
                if not has_japanese_text(content):
                    errors.append(f"JA workflow page lacks Japanese translation: {wp}")
        for wp in WORKFLOW_NOTEBOOK_PAGES:
            wp_file = root / language / wp
            if not wp_file.exists():
                continue
            ipynb_hrefs = [h for h in parse(wp_file).links if ".ipynb" in h]
            if not ipynb_hrefs:
                errors.append(f"Missing notebook download link: {language}{wp}")
    return errors


def check_remote(base_url: str, expected_revision: str) -> list[str]:
    """Read back identity, audience routes, figure, and downloads in both languages."""
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

            def fetch_url(url: str) -> bytes:
                request = Request(
                    url + "?revision=" + expected_revision,
                    headers={"Cache-Control": "no-cache"},
                )
                with urlopen(request, timeout=20) as response:
                    return response.read()

            info = json.loads(fetch("build-info.json"))
            if info.get("source_revision") != expected_revision or info.get("dirty"):
                errors.append(
                    f"{prefix}: deployed revision is not the clean expected commit"
                )
            pages = dict.fromkeys(
                ("index.html", "tutorials/quickstart.html")
                + tuple(urlsplit(route).path for route in AUDIENCE_ROUTES)
            )
            for name in pages:
                page = fetch(name).decode("utf-8")
                if (
                    "gwexpy-build-status" not in page
                    or expected_revision[:8] not in page
                ):
                    errors.append(f"{prefix}{name}: missing expected build identity")
                parsed = Links()
                parsed.feed(page)
                if name == "index.html":
                    for route in AUDIENCE_ROUTES:
                        if route not in parsed.links:
                            errors.append(f"{prefix}: missing audience route {route}")
                for route in AUDIENCE_ROUTES:
                    target = urlsplit(route)
                    if target.path == name and target.fragment:
                        if target.fragment not in parsed.ids:
                            errors.append(f"{prefix}{route}: missing audience anchor")
            figure = fetch("_static/images/quickstart-asd.png")
            if not figure.startswith(b"\x89PNG\r\n\x1a\n") or len(figure) < 1000:
                errors.append(f"{prefix}: missing Quickstart figure")
            # Read back the nine logical workflow pages: build identity,
            # language switch, rendered figures, and notebook downloads.
            base_en = base_url.rstrip("/") + "/"
            base_ja = base_url.rstrip("/") + "/ja/"
            for name in WORKFLOW_PAGES:
                try:
                    page = fetch(name).decode("utf-8")
                except (URLError, TimeoutError, ValueError) as exc:
                    errors.append(f"{prefix}{name}: {exc}")
                    continue
                for err in workflow_page_errors(
                    page_html=page,
                    language=language,
                    expected_revision=expected_revision,
                    counterpart_url=(base_ja if language == "" else base_en) + name,
                ):
                    errors.append(f"{prefix}{name}: {err}")
                parsed = Links()
                parsed.feed(page)
                page_url = prefix + name
                for src in sorted(set(parsed.links)):
                    if urlsplit(src).scheme or urlsplit(src).netloc:
                        continue
                    if (
                        not src.lower()
                        .split("?")[0]
                        .endswith((".png", ".svg", ".jpg", ".jpeg"))
                    ):
                        continue
                    try:
                        image = fetch_url(urljoin(page_url, src))
                    except (URLError, TimeoutError, ValueError) as exc:
                        errors.append(f"{prefix}{name}: image {src}: {exc}")
                        continue
                    if not image:
                        errors.append(f"{prefix}{name}: empty image {src}")
                if name in WORKFLOW_NOTEBOOK_PAGES:
                    ipynb_hrefs = sorted({h for h in parsed.links if ".ipynb" in h})
                    if not ipynb_hrefs:
                        errors.append(f"{prefix}{name}: missing notebook download link")
                        continue
                    try:
                        notebook = fetch_url(urljoin(page_url, ipynb_hrefs[0]))
                    except (URLError, TimeoutError, ValueError) as exc:
                        errors.append(f"{prefix}{name}: notebook download: {exc}")
                        continue
                    if not is_notebook_payload(notebook):
                        errors.append(
                            f"{prefix}{name}: notebook download is not a notebook"
                        )
            source = Path(__file__).resolve().parents[1] / "docs_redesign"
            for filename in ("quickstart.py", "commissioner.py", "commissioner.xml"):
                path = "_static/downloads/" + filename
                if fetch(path) != (source / path).read_bytes():
                    errors.append(f"{prefix}{path}: download differs from source")
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
