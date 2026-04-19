#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from intersphinx_registry import INTERSPHINX_DIR, REMOTE_INVENTORY_SOURCES


def fetch_inventory(name: str, inventory_url: str, timeout: int = 30) -> bytes:
    response = requests.get(
        inventory_url,
        timeout=timeout,
        headers={"User-Agent": "GWexpy-Intersphinx-Refresh"},
    )
    response.raise_for_status()
    return response.content


def refresh_inventories(
    *,
    destination_dir: Path = INTERSPHINX_DIR,
    sources: dict[str, tuple[str, str]] = REMOTE_INVENTORY_SOURCES,
    fetch_inventory=fetch_inventory,
    check_only: bool = False,
) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    if not check_only:
        destination_dir.mkdir(parents=True, exist_ok=True)

    for name, (base_url, inventory_url) in sources.items():
        data = fetch_inventory(name, inventory_url)
        manifest[name] = {
            "base_url": base_url,
            "inventory_url": inventory_url,
        }
        if check_only:
            continue
        (destination_dir / f"{name}.inv").write_bytes(data)

    if not check_only:
        (destination_dir / "sources.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="download inventories into docs/_intersphinx",
    )
    mode.add_argument(
        "--check-upstream",
        action="store_true",
        help="verify that all upstream inventories are reachable without writing files",
    )
    args = parser.parse_args(argv)

    check_only = args.check_upstream
    manifest = refresh_inventories(check_only=check_only)
    action = "checked" if check_only else "updated"
    print(f"{action} {len(manifest)} intersphinx inventories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
