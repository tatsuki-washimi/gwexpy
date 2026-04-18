import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "update_intersphinx_inventories.py"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_refresh_inventories_writes_inventory_files_and_sources_manifest(tmp_path):
    module = _load_module("gwexpy_update_inventories_write")
    sources = {
        "python": ("https://docs.python.org/3", "https://docs.python.org/3/objects.inv"),
        "gwpy": ("https://gwpy.readthedocs.io/en/stable/", "https://gwpy.readthedocs.io/en/stable/objects.inv"),
    }

    def fake_fetch(name, inventory_url):
        return f"{name}:{inventory_url}".encode()

    manifest = module.refresh_inventories(
        destination_dir=tmp_path,
        sources=sources,
        fetch_inventory=fake_fetch,
        check_only=False,
    )

    assert (tmp_path / "python.inv").read_bytes() == b"python:https://docs.python.org/3/objects.inv"
    assert (tmp_path / "gwpy.inv").read_bytes() == b"gwpy:https://gwpy.readthedocs.io/en/stable/objects.inv"
    assert manifest["python"]["base_url"] == "https://docs.python.org/3"
    assert json.loads((tmp_path / "sources.json").read_text()) == manifest


def test_refresh_inventories_check_only_does_not_write_files(tmp_path):
    module = _load_module("gwexpy_update_inventories_check")
    calls = []

    def fake_fetch(name, inventory_url):
        calls.append((name, inventory_url))
        return b"unused"

    module.refresh_inventories(
        destination_dir=tmp_path,
        sources={
            "numpy": ("https://numpy.org/doc/stable", "https://numpy.org/doc/stable/objects.inv"),
        },
        fetch_inventory=fake_fetch,
        check_only=True,
    )

    assert calls == [("numpy", "https://numpy.org/doc/stable/objects.inv")]
    assert not any(tmp_path.iterdir())
