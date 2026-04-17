from __future__ import annotations

from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
INTERSPHINX_DIR = DOCS_DIR / "_intersphinx"

REMOTE_INVENTORY_SOURCES: dict[str, tuple[str, str]] = {
    "python": ("https://docs.python.org/3", "https://docs.python.org/3/objects.inv"),
    "numpy": ("https://numpy.org/doc/stable", "https://numpy.org/doc/stable/objects.inv"),
    "scipy": ("https://docs.scipy.org/doc/scipy", "https://docs.scipy.org/doc/scipy/objects.inv"),
    "astropy": ("https://docs.astropy.org/en/stable", "https://docs.astropy.org/en/stable/objects.inv"),
    "matplotlib": ("https://matplotlib.org/stable", "https://matplotlib.org/stable/objects.inv"),
    "gwpy": ("https://gwpy.readthedocs.io/en/stable/", "https://gwpy.readthedocs.io/en/stable/objects.inv"),
}


def inventory_path(name: str, intersphinx_dir: Path = INTERSPHINX_DIR) -> str | None:
    path = intersphinx_dir / f"{name}.inv"
    return str(path) if path.exists() else None


def build_intersphinx_mapping(
    *,
    intersphinx_dir: Path = INTERSPHINX_DIR,
    prefer_remote: bool = False,
) -> dict[str, tuple[str, str | None]]:
    mapping: dict[str, tuple[str, str | None]] = {}
    for name, (base_url, remote_inventory) in REMOTE_INVENTORY_SOURCES.items():
        local_inventory = None if prefer_remote else inventory_path(name, intersphinx_dir)
        mapping[name] = (base_url, local_inventory or remote_inventory)
    return mapping
