#!/usr/bin/env python3
"""Check public docs navigation, downloadable examples, plots, and build identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlsplit
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
    "about/changelog.html",
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
CHANGELOG_PAGE = "about/changelog.html"

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

DOCS_ARTIFACT_SCHEMA = "gwexpy-docs-pages-artifact-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_READBACK_MAX_ATTEMPTS = 4
MANIFEST_READBACK_RETRY_DELAY_SECONDS = 1.0
ARTIFACT_REQUIRED_FILES = frozenset(
    {"index.html", "build-info.json", "ja/index.html", "ja/build-info.json"}
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
        self.anchor_links: list[str] = []
        self.images: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "a" and values.get("href"):
            href = str(values["href"])
            self.links.append(href)
            self.anchor_links.append(href)
        elif tag == "img" and values.get("src"):
            src = str(values["src"])
            self.links.append(src)
            self.images.append(src)


class Headings(HTMLParser):
    """Collect rendered heading levels and text for release-section checks."""

    def __init__(self) -> None:
        super().__init__()
        self.items: list[tuple[int, str]] = []
        self._level: int | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self._level = int(tag[1])
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._level is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._level is not None and tag == f"h{self._level}":
            self.items.append((self._level, " ".join("".join(self._text).split())))
            self._level = None
            self._text = []


def has_japanese_text(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text)


def is_notebook_payload(data: bytes) -> bool:
    """Return True when downloaded bytes have a minimal nbformat structure."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    if (
        not isinstance(payload.get("cells"), list)
        or not payload["cells"]
        or not isinstance(payload.get("metadata"), dict)
        or type(payload.get("nbformat")) is not int
        or payload["nbformat"] < 4
        or type(payload.get("nbformat_minor")) is not int
    ):
        return False
    for cell in payload["cells"]:
        if not isinstance(cell, dict):
            return False
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            return False
        if not isinstance(cell.get("metadata"), dict):
            return False
        source = cell.get("source")
        if isinstance(source, str):
            pass
        elif isinstance(source, list) and all(isinstance(line, str) for line in source):
            pass
        else:
            return False
        if cell["cell_type"] == "code":
            if not isinstance(cell.get("outputs"), list):
                return False
            execution_count = cell.get("execution_count")
            if execution_count is not None and type(execution_count) is not int:
                return False
    return True


def expected_notebook_hrefs(links: list[str], expected_filename: str) -> list[str]:
    """Select download links whose URL path has the expected notebook filename."""
    return sorted(
        {
            href
            for href in links
            if Path(unquote(urlsplit(href).path)).name == expected_filename
        }
    )


def is_analysis_figure_href(href: str) -> bool:
    """Return whether a rendered image is a Sphinx notebook analysis figure."""
    path = urlsplit(href).path.lower()
    if not (path.startswith("_images/") or "/_images/" in path):
        return False
    filename = Path(unquote(path)).name
    suffix = Path(filename).suffix
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        return False
    stem = Path(filename).stem
    return not any(token in stem for token in ("logo", "icon", "favicon"))


def is_image_payload(data: bytes, content_type: str, href: str) -> bool:
    """Validate an image response by media type, signature, and URL suffix."""
    if not data:
        return False
    stripped = data.lstrip().lower()
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"}:
        return False
    if stripped.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        return False
    suffix = Path(urlsplit(href).path).suffix.lower()
    if suffix == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if suffix == ".webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if suffix == ".svg":
        return b"<svg" in stripped
    return media_type.startswith("image/")


def workflow_page_errors(
    *,
    page_html: str,
    language: str,
    expected_revision: str,
    counterpart_url: str,
    page_name: str | None = None,
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
    if page_name in WORKFLOW_NOTEBOOK_PAGES and not any(
        is_analysis_figure_href(src) for src in parsed.images
    ):
        errors.append("missing analysis figure")
    return errors


def changelog_page_errors(
    page_html: str, language: str, latest_release: str | None
) -> list[str]:
    """Require the latest release details and language-specific activity SVG."""
    errors: list[str] = []
    if not latest_release:
        return ["missing latest release metadata"]
    parsed = Links()
    parsed.feed(page_html)
    image_names = {Path(unquote(urlsplit(src).path)).name for src in parsed.images}
    suffix = "-ja" if language == "ja/" else ""
    other_suffix = "" if language == "ja/" else "-ja"
    expected_svg = f"development-activity-v{latest_release}{suffix}.svg"
    other_svg = f"development-activity-v{latest_release}{other_suffix}.svg"
    if expected_svg not in image_names:
        errors.append(f"missing language-specific activity SVG {expected_svg}")
    if other_svg in image_names:
        errors.append("wrong language activity SVG")
    headings = Headings()
    headings.feed(page_html)
    release_heading = f"[{latest_release}]"
    latest_index = next(
        (
            index
            for index, (level, text) in enumerate(headings.items)
            if level == 2 and release_heading in text
        ),
        None,
    )
    if latest_index is None:
        errors.append(f"missing detailed v{latest_release} release history")
    else:
        next_release_index = next(
            (
                index
                for index in range(latest_index + 1, len(headings.items))
                if headings.items[index][0] == 2
            ),
            len(headings.items),
        )
        if not any(
            level == 3
            for level, _text in headings.items[latest_index + 1 : next_release_index]
        ):
            errors.append(f"missing detailed v{latest_release} release history")
    if language == "ja/" and not has_japanese_text(page_html):
        errors.append("JA changelog lacks Japanese translation")
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
            if name == CHANGELOG_PAGE:
                for error in changelog_page_errors(
                    text, language, info.get("latest_release")
                ):
                    errors.append(f"{language}{name}: {error}")
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
        public_notebooks = info.get("public_notebooks")
        if public_notebooks is None:
            # Read build metadata emitted before public notebooks became their
            # own source of executable cells.
            public_notebooks = info.get("notebook_sources", {}).keys()
        cases = [
            name.removesuffix(".ipynb")
            for name in public_notebooks
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
            else:
                content = wp_file.read_text(encoding="utf-8")
                if language == "ja/" and not has_japanese_text(content):
                    errors.append(f"JA workflow page lacks Japanese translation: {wp}")
                if wp in WORKFLOW_NOTEBOOK_PAGES and not any(
                    is_analysis_figure_href(src) for src in parse(wp_file).images
                ):
                    errors.append(f"missing analysis figure: {language}{wp}")
        for wp in WORKFLOW_NOTEBOOK_PAGES:
            wp_file = root / language / wp
            if not wp_file.exists():
                continue
            expected_filename = WORKFLOW_NOTEBOOK_PAGES[wp]
            ipynb_hrefs = expected_notebook_hrefs(
                parse(wp_file).anchor_links, expected_filename
            )
            if not ipynb_hrefs:
                errors.append(
                    f"Missing notebook download link: {language}{wp} "
                    f"(expected {expected_filename})"
                )
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

            def fetch_url(url: str) -> tuple[bytes, str]:
                request = Request(
                    url + "?revision=" + expected_revision,
                    headers={"Cache-Control": "no-cache"},
                )
                with urlopen(request, timeout=20) as response:
                    headers = getattr(response, "headers", None)
                    content_type = (
                        str(headers.get("Content-Type", "")) if headers else ""
                    )
                    return response.read(), content_type

            info = json.loads(fetch("build-info.json"))
            if info.get("source_revision") != expected_revision or info.get("dirty"):
                errors.append(
                    f"{prefix}: deployed revision is not the clean expected commit"
                )
            pages = dict.fromkeys(
                ("index.html", "tutorials/quickstart.html", CHANGELOG_PAGE)
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
                if name == CHANGELOG_PAGE:
                    for error in changelog_page_errors(
                        page, language, info.get("latest_release")
                    ):
                        errors.append(f"{prefix}{name}: {error}")
                    counterpart = (
                        base_url.rstrip("/")
                        + ("/ja/" if language == "" else "/")
                        + CHANGELOG_PAGE
                    )
                    if counterpart not in parsed.links:
                        errors.append(f"{prefix}{name}: missing language switch link")
                    expected_svg = (
                        f"development-activity-v{info.get('latest_release')}-ja.svg"
                        if language == "ja/"
                        else f"development-activity-v{info.get('latest_release')}.svg"
                    )
                    chart_sources = [
                        src
                        for src in parsed.images
                        if Path(unquote(urlsplit(src).path)).name == expected_svg
                    ]
                    if chart_sources:
                        try:
                            image, content_type = fetch_url(
                                urljoin(prefix + name, chart_sources[0])
                            )
                        except (URLError, TimeoutError, ValueError) as exc:
                            errors.append(
                                f"{prefix}{name}: activity SVG readback failed: {exc}"
                            )
                        else:
                            if not is_image_payload(
                                image, content_type, chart_sources[0]
                            ):
                                errors.append(f"{prefix}{name}: invalid activity SVG")
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
                    page_name=name,
                ):
                    errors.append(f"{prefix}{name}: {err}")
                parsed = Links()
                parsed.feed(page)
                page_url = prefix + name
                for src in sorted(set(parsed.images)):
                    if urlsplit(src).scheme or urlsplit(src).netloc:
                        continue
                    if (
                        not src.lower()
                        .split("?")[0]
                        .endswith((".png", ".svg", ".jpg", ".jpeg", ".webp"))
                    ):
                        continue
                    try:
                        image, content_type = fetch_url(urljoin(page_url, src))
                    except (URLError, TimeoutError, ValueError) as exc:
                        errors.append(f"{prefix}{name}: image {src}: {exc}")
                        continue
                    if not is_image_payload(image, content_type, src):
                        errors.append(f"{prefix}{name}: invalid image {src}")
                if name in WORKFLOW_NOTEBOOK_PAGES:
                    expected_filename = WORKFLOW_NOTEBOOK_PAGES[name]
                    ipynb_hrefs = expected_notebook_hrefs(
                        parsed.anchor_links, expected_filename
                    )
                    if not ipynb_hrefs:
                        errors.append(
                            f"{prefix}{name}: missing notebook download link "
                            f"(expected {expected_filename})"
                        )
                        continue
                    try:
                        notebook, _ = fetch_url(urljoin(page_url, ipynb_hrefs[0]))
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


def check_remote_manifest(
    base_url: str, expected_revision: str, manifest: dict
) -> list[str]:
    """Compare every deployed EN/JA site file with the verified build manifest."""
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "source_revision", "files"}
        or manifest.get("schema") != DOCS_ARTIFACT_SCHEMA
        or manifest.get("source_revision") != expected_revision
        or not isinstance(manifest.get("files"), dict)
    ):
        return ["expected artifact manifest is malformed or bound to another revision"]

    files = manifest["files"]
    if not ARTIFACT_REQUIRED_FILES.issubset(files):
        missing = sorted(ARTIFACT_REQUIRED_FILES - set(files))
        return [f"expected artifact manifest is missing required files: {missing}"]
    for path, digest in files.items():
        if (
            not isinstance(path, str)
            or path.startswith("/")
            or "\\" in path
            or "?" in path
            or "#" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            return ["expected artifact manifest contains an unsafe relative path"]
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            return [f"expected artifact manifest has an invalid SHA-256 for {path}"]

    root_url = base_url.rstrip("/") + "/"

    def verify_file(item: tuple[str, str]) -> tuple[str, str | None]:
        path, expected_digest = item
        request = Request(
            root_url + quote(path, safe="/") + "?revision=" + expected_revision,
            headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity"},
        )
        for attempt in range(1, MANIFEST_READBACK_MAX_ATTEMPTS + 1):
            try:
                with urlopen(request, timeout=20) as response:
                    digest = hashlib.sha256()
                    while chunk := response.read(1024 * 64):
                        digest.update(chunk)
            except HTTPError as exc:
                retryable = exc.code in {408, 425, 429} or 500 <= exc.code <= 599
                exc.close()
                if not retryable or attempt == MANIFEST_READBACK_MAX_ATTEMPTS:
                    return path, f"{path}: deployed artifact readback failed: {exc}"
                time.sleep(MANIFEST_READBACK_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
                continue
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                return path, f"{path}: deployed artifact readback failed: {exc}"
            if digest.hexdigest() == expected_digest:
                return path, None
            # A Pages edge can briefly serve the prior file after a deploy.
            # Retry against the same digest; stale or altered bytes never pass.
            if attempt == MANIFEST_READBACK_MAX_ATTEMPTS:
                return path, f"{path}: deployed bytes differ from the verified artifact"
            time.sleep(MANIFEST_READBACK_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
        return path, f"{path}: deployed bytes differ from the verified artifact"

    with ThreadPoolExecutor(max_workers=8) as executor:
        failures = [
            error
            for _path, error in executor.map(verify_file, files.items())
            if error is not None
        ]
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_root", type=Path, nargs="?")
    parser.add_argument("--url", help="Read back a deployed site instead of local HTML")
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Compare every deployed file with --expected-manifest without route checks",
    )
    args = parser.parse_args()
    if args.url:
        if not args.expected_revision:
            parser.error("--url requires --expected-revision")
        if args.manifest_only:
            if not args.expected_manifest:
                parser.error("--manifest-only requires --expected-manifest")
            try:
                manifest = json.loads(
                    args.expected_manifest.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"cannot read expected artifact manifest: {exc}")
            failures = check_remote_manifest(args.url, args.expected_revision, manifest)
        else:
            failures = check_remote(args.url, args.expected_revision)
    else:
        if args.manifest_only or args.expected_manifest:
            parser.error("--expected-manifest and --manifest-only require --url")
        if not args.html_root:
            parser.error("provide html_root or --url")
        failures = check(args.html_root, args.expected_revision)
    if failures:
        raise SystemExit("\n".join(failures))
    if args.manifest_only:
        print("Public docs: all EN/JA artifact files match the verified manifest")
    else:
        print("Public docs: requested EN/JA checks passed")
