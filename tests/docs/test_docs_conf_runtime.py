import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"


def _load_conf_module(name: str):
    spec = importlib.util.spec_from_file_location(name, CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_build_defaults_disable_notebook_execution_and_exclude_ipynb(
    monkeypatch,
):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("NBS_EXECUTE", raising=False)

    conf = _load_conf_module("gwexpy_docs_conf_local")

    assert conf.nbsphinx_execute == "never"
    assert "**/*.ipynb" in conf.exclude_patterns
    assert "nbsphinx" not in conf.extensions


def test_explicit_notebook_build_keeps_nbsphinx_when_pandoc_exists(monkeypatch):
    monkeypatch.setenv("NBS_EXECUTE", "always")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None)

    conf = _load_conf_module("gwexpy_docs_conf_with_pandoc")

    assert conf.nbsphinx_execute == "always"
    assert "nbsphinx" in conf.extensions
    assert "**/*.ipynb" not in conf.exclude_patterns
