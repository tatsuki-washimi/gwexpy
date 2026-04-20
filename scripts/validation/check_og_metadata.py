#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"
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


class MetaTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return

        attr_map = {name.lower(): value for name, value in attrs if value is not None}
        key = attr_map.get("property") or attr_map.get("name")
        content = attr_map.get("content")
        if key and content:
            self.meta[key] = content


def _load_conf_module(name: str):
    spec = importlib.util.spec_from_file_location(name, CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _validate_page(
    page_path: Path,
    *,
    expected_image: str,
    expected_twitter_card: str,
    expected_baseurl: str,
) -> dict[str, object]:
    parser = MetaTagParser()
    parser.feed(page_path.read_text(encoding="utf-8"))
    meta = parser.meta

    errors = []
    missing = [key for key in REQUIRED_META_KEYS if not meta.get(key)]
    if missing:
        errors.append(f"missing meta tags: {missing}")

    if meta.get("og:image") != expected_image:
        errors.append(f'og:image expected "{expected_image}", got "{meta.get("og:image")}"')
    if meta.get("twitter:image") != expected_image:
        errors.append(
            f'twitter:image expected "{expected_image}", got "{meta.get("twitter:image")}"'
        )
    if meta.get("twitter:card") != expected_twitter_card:
        errors.append(
            f'twitter:card expected "{expected_twitter_card}", got "{meta.get("twitter:card")}"'
        )
    if not meta.get("og:url", "").startswith(expected_baseurl):
        errors.append(
            f'og:url expected to start with "{expected_baseurl}", got "{meta.get("og:url")}"'
        )

    return {
        "page": str(page_path),
        "meta": {key: meta.get(key) for key in REQUIRED_META_KEYS},
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate rendered OGP/Twitter metadata.")
    parser.add_argument("pages", nargs="+", help="Built HTML page paths to validate")
    args = parser.parse_args()

    conf = _load_conf_module("gwexpy_docs_conf_og_validation")
    expected_image = conf.html_context["og_image"]
    expected_twitter_card = conf.html_context["twitter_card"]
    expected_baseurl = conf.html_baseurl

    results = [
        _validate_page(
            Path(page),
            expected_image=expected_image,
            expected_twitter_card=expected_twitter_card,
            expected_baseurl=expected_baseurl,
        )
        for page in args.pages
    ]
    failures = [result for result in results if result["errors"]]

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
