#!/usr/bin/env python3
"""Generate the docs branding asset bundle.

This script keeps the public branding copies in sync with the source assets in
``docs/logo`` and renders a deterministic Open Graph preview card.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "docs" / "logo"
OUTPUT_DIR = ROOT / "docs" / "_static" / "branding"

SOURCE_ASSETS = ("logo.svg", "logo.png", "icon.svg", "icon.png")
CARD_SIZE = (1200, 630)
CARD_BACKGROUND = (255, 255, 255)
CARD_INSET_X = 96
CARD_INSET_Y = 132


def copy_source_assets(source_dir: Path, output_dir: Path) -> None:
    """Copy the public logo assets into the docs branding directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_ASSETS:
        target = output_dir / name
        shutil.copy2(source_dir / name, target)
        os.chmod(target, 0o644)


def build_og_card(source_logo: Path, output_path: Path) -> None:
    """Render a centered social preview card from the logo PNG."""
    with Image.open(source_logo) as source:
        logo = source.convert("RGBA")

    max_size = (CARD_SIZE[0] - 2 * CARD_INSET_X, CARD_SIZE[1] - 2 * CARD_INSET_Y)
    logo = ImageOps.contain(logo, max_size, method=Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", CARD_SIZE, CARD_BACKGROUND)
    offset = (
        (CARD_SIZE[0] - logo.width) // 2,
        (CARD_SIZE[1] - logo.height) // 2,
    )
    canvas.paste(logo, offset, logo.getchannel("A"))
    canvas.save(output_path, format="PNG", compress_level=9, optimize=False)
    os.chmod(output_path, 0o644)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    source_dir: Path = args.source_dir
    output_dir: Path = args.output_dir

    copy_source_assets(source_dir, output_dir)
    build_og_card(source_dir / "logo.png", output_dir / "og-card.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
