import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "docs" / "intersphinx_registry.py"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_intersphinx_mapping_prefers_local_inventory(tmp_path):
    module = _load_module("gwexpy_intersphinx_registry_local")
    local_inventory = tmp_path / "python.inv"
    local_inventory.write_bytes(b"placeholder")

    mapping = module.build_intersphinx_mapping(
        intersphinx_dir=tmp_path,
        prefer_remote=False,
    )

    assert mapping["python"] == (
        "https://docs.python.org/3",
        str(local_inventory),
    )
    assert mapping["gwpy"] == (
        "https://gwpy.readthedocs.io/en/stable/",
        "https://gwpy.readthedocs.io/en/stable/objects.inv",
    )


def test_build_intersphinx_mapping_can_force_remote_even_if_local_exists(tmp_path):
    module = _load_module("gwexpy_intersphinx_registry_remote")
    local_inventory = tmp_path / "gwpy.inv"
    local_inventory.write_bytes(b"placeholder")

    mapping = module.build_intersphinx_mapping(
        intersphinx_dir=tmp_path,
        prefer_remote=True,
    )

    assert mapping["gwpy"] == (
        "https://gwpy.readthedocs.io/en/stable/",
        "https://gwpy.readthedocs.io/en/stable/objects.inv",
    )
