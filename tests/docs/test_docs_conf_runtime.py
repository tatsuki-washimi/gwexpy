import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"
DOCS_PR_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "docs-pr.yml"
DOCS_PAGES_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "docs-pages.yml"


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
    monkeypatch.setattr("shutil.which", lambda name: None)

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


def test_github_actions_defaults_to_failing_on_notebook_errors(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NBS_EXECUTE", raising=False)
    monkeypatch.delenv("NBS_ALLOW_ERRORS", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None)

    conf = _load_conf_module("gwexpy_docs_conf_github_actions")

    assert conf.nbsphinx_allow_errors is False


def test_matplotlib_fonts_prefer_japanese_capable_sans_serif_stack(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("NBS_ALLOW_ERRORS", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None)

    conf = _load_conf_module("gwexpy_docs_conf_fonts")

    assert "sans-serif" in conf.mpl.rcParams["font.family"]
    assert any(
        font_name in conf.mpl.rcParams["font.sans-serif"]
        for font_name in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic")
    )
    matplotlibrc = Path(conf.os.environ["MPLCONFIGDIR"]) / "matplotlibrc"
    content = matplotlibrc.read_text(encoding="utf-8")
    assert "backend: Agg" in content
    assert "font.sans-serif: Noto Sans CJK JP, IPAexGothic, IPAGothic, DejaVu Sans" in content
    assert "axes.unicode_minus: False" in content


def test_local_intersphinx_inventories_are_preferred_by_default(monkeypatch):
    monkeypatch.delenv("INTERSPHINX_USE_REMOTE", raising=False)

    conf = _load_conf_module("gwexpy_docs_conf_intersphinx_local")

    python_inventory = conf.intersphinx_mapping["python"][1]
    gwpy_inventory = conf.intersphinx_mapping["gwpy"][1]

    assert python_inventory is not None
    assert gwpy_inventory is not None
    assert python_inventory.endswith("docs/_intersphinx/python.inv")
    assert gwpy_inventory.endswith("docs/_intersphinx/gwpy.inv")


def _load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_docs_pr_workflow_executes_notebooks_by_default_in_github_actions():
    workflow = _load_workflow(DOCS_PR_WORKFLOW_PATH)
    env = workflow["jobs"]["docs-pr"]["env"]
    assert env["NBS_EXECUTE"] == "never"
    assert env["NBS_ALLOW_ERRORS"] == "0"


def test_docs_pages_workflow_builds_docs_redesign_without_nbsphinx():
    workflow = _load_workflow(DOCS_PAGES_WORKFLOW_PATH)
    job = workflow["jobs"]["publish-pages"]

    # docs_redesign ships pre-executed notebook outputs (myst-nb,
    # nb_execution_mode="off"); the old nbsphinx-execution toggles the
    # previous docs/ build used no longer apply to this deploy target.
    assert "NBS_EXECUTE" not in job.get("env", {})
    assert "NBS_ALLOW_ERRORS" not in job.get("env", {})

    steps_by_name = {step["name"]: step for step in job["steps"] if "name" in step}
    en_build = steps_by_name["Build EN HTML"]
    ja_build = steps_by_name["Build JA HTML"]

    assert en_build["working-directory"] == "docs_redesign"
    assert ja_build["working-directory"] == "docs_redesign"
    assert "nbsphinx" not in en_build["run"]
    assert "nbsphinx" not in ja_build["run"]
    assert "-D language=ja" in ja_build["run"]
    assert "GWEXPY_DOCS_BASEURL" in en_build["env"]
    assert "GWEXPY_DOCS_BASEURL" in ja_build["env"]
