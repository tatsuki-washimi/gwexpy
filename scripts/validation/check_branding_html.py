#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"
DEFAULT_PAGES = [
    ROOT / "docs" / "_build" / "html" / "index.html",
    ROOT / "docs" / "_build" / "html" / "web" / "en" / "index.html",
    ROOT / "docs" / "_build" / "html" / "web" / "ja" / "index.html",
]
REQUIRED_META_KEYS = [
    "og:title",
    "og:description",
    "og:type",
    "og:url",
    "og:image",
    "twitter:card",
    "twitter:title",
    "twitter:description",
    "twitter:image",
]


class BrandingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.links: list[dict[str, str]] = []
        self.images: list[dict[str, str]] = []
        self.anchors: list[dict[str, str]] = []
        self._current_anchor_text: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value for name, value in attrs if value is not None}
        tag = tag.lower()
        attr_map["_tag"] = tag

        if tag == "meta":
            key = attr_map.get("property") or attr_map.get("name")
            content = attr_map.get("content")
            if key and content:
                self.meta[key] = content
            return

        if tag == "link":
            self.links.append(attr_map)
            return

        if tag == "img":
            self.images.append(attr_map)
            return

        if tag == "a":
            self.anchors.append(attr_map)
            self._current_anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._current_anchor_text is not None:
            self._current_anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_anchor_text is not None and self.anchors:
            self.anchors[-1]["text"] = "".join(self._current_anchor_text).strip()
            self._current_anchor_text = None


def _load_conf_module(name: str):
    spec = importlib.util.spec_from_file_location(name, CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _has_rendered_asset_reference(
    parser: BrandingParser,
    *,
    expected_asset: str,
    tag_names: set[str],
    attr_name: str,
) -> bool:
    expected_name = Path(expected_asset).name
    for attrs in parser.links + parser.images + parser.anchors:
        if attrs.get("_tag") not in tag_names:
            continue
        ref = attrs.get(attr_name)
        if not ref:
            continue
        parsed = urlparse(ref)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if Path(parsed.path).name == expected_name:
            return True
    return False


def _validate_page(
    page_path: Path,
    *,
    expected_logo: str,
    expected_favicon: str,
    expected_image: str,
    expected_twitter_card: str,
    expected_baseurl: str,
) -> dict[str, object]:
    parser = BrandingParser()
    parser.feed(page_path.read_text(encoding="utf-8"))

    errors = []
    missing = [key for key in REQUIRED_META_KEYS if not parser.meta.get(key)]
    if missing:
        errors.append(f"missing meta tags: {missing}")

    if parser.meta.get("og:image") != expected_image:
        errors.append(f'og:image expected "{expected_image}", got "{parser.meta.get("og:image")}"')
    if parser.meta.get("twitter:image") != expected_image:
        errors.append(
            f'twitter:image expected "{expected_image}", got "{parser.meta.get("twitter:image")}"'
        )
    if parser.meta.get("twitter:card") != expected_twitter_card:
        errors.append(
            f'twitter:card expected "{expected_twitter_card}", got "{parser.meta.get("twitter:card")}"'
        )
    if not parser.meta.get("og:url", "").startswith(expected_baseurl):
        errors.append(
            f'og:url expected to start with "{expected_baseurl}", got "{parser.meta.get("og:url")}"'
        )

    # Sphinx rewrites the source branding paths into the rendered asset filenames,
    # so we validate the final built HTML by filename.
    if not _has_rendered_asset_reference(
        parser,
        expected_asset=expected_favicon,
        tag_names={"link"},
        attr_name="href",
    ):
        errors.append(f"missing favicon reference to {expected_favicon}")
    if not _has_rendered_asset_reference(
        parser,
        expected_asset=expected_logo,
        tag_names={"img"},
        attr_name="src",
    ):
        errors.append(f"missing logo reference to {expected_logo}")

    return {
        "page": str(page_path),
        "meta": {key: parser.meta.get(key) for key in REQUIRED_META_KEYS},
        "references": {
            "logo": expected_logo,
            "favicon": expected_favicon,
        },
        "errors": errors,
    }


def _brand_social_image_is_expected(image_url: str) -> bool:
    return urlparse(image_url).path.endswith("/_static/branding/og-card.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate built HTML pages for GWexpy branding references."
    )
    parser.add_argument("pages", nargs="*", help="Built HTML page paths to validate")
    args = parser.parse_args()

    conf = _load_conf_module("gwexpy_docs_conf_branding_validation")
    expected_logo = conf.html_logo
    expected_favicon = conf.html_favicon
    expected_image = conf.html_context["og_image"]
    expected_twitter_card = conf.html_context["twitter_card"]
    expected_baseurl = conf.html_baseurl

    if not _brand_social_image_is_expected(expected_image):
        print(
            json.dumps(
                {
                    "error": "docs/conf.py does not point og:image at the branding social card",
                    "expected_suffix": "/_static/branding/og-card.png",
                    "actual": expected_image,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    page_paths = [Path(page) for page in args.pages] if args.pages else DEFAULT_PAGES
    results = [
        _validate_page(
            page,
            expected_logo=expected_logo,
            expected_favicon=expected_favicon,
            expected_image=expected_image,
            expected_twitter_card=expected_twitter_card,
            expected_baseurl=expected_baseurl,
        )
        for page in page_paths
    ]
    failures = [result for result in results if result["errors"]]

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
