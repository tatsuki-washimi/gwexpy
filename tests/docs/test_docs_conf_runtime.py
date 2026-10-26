import importlib.util
from pathlib import Path

import yaml

from scripts.ci.docs_publish_guard import DOCS_BUILD_INPUTS

ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"
REDESIGN_CONF_PATH = ROOT / "docs_redesign" / "conf.py"
DOCS_PR_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "docs-pr.yml"
DOCS_PAGES_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "docs-pages.yml"
DOCS_PREVIEW_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "docs-redesign-preview.yml"
)


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


def test_docs_redesign_executes_clean_notebooks_in_an_untracked_cache(monkeypatch):
    monkeypatch.delenv("GWEXPY_NB_EXECUTION_MODE", raising=False)
    spec = importlib.util.spec_from_file_location(
        "gwexpy_docs_redesign_conf", REDESIGN_CONF_PATH
    )
    conf = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(conf)

    assert conf.nb_execution_mode == "cache"
    cache = Path(conf.nb_execution_cache_path)
    assert cache.parent == REDESIGN_CONF_PATH.parent / "_build/jupyter-cache"
    assert len(cache.name) == 16
    assert conf._execution_environment["python"]
    assert conf._execution_environment["runtime_source_sha256"]
    assert conf._execution_environment["repo_data_sha256"]
    assert conf._execution_environment["conf_execution_sha256"]
    assert conf._execution_environment["conf_execution_settings"]
    assert conf._execution_environment["env_execution"]
    # Git SHA is tracked separately in _build_identity for HTML footers, not in execution cache fingerprint
    assert conf._build_identity["source_revision"]
    assert "source_revision" not in conf._execution_environment
    assert conf.nb_execution_timeout == 600
    assert conf.nb_execution_allow_errors is False
    assert conf.nb_execution_raise_on_error is True


def test_docs_redesign_cache_fingerprint_invalidation_contract(monkeypatch, tmp_path):
    """Verify each fingerprint input independently in an isolated copy."""
    import json
    import runpy
    import shutil

    import gwexpy

    writes: list[Path] = []
    original_write_text = Path.write_text

    def record_write_text(path, *args, **kwargs):
        writes.append(path.resolve())
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_write_text)

    isolated_root = tmp_path / "fingerprint-checkout"
    isolated_package = isolated_root / "gwexpy"
    shutil.copytree(ROOT / "gwexpy", isolated_package)

    isolated_docs = isolated_root / "docs_redesign"
    isolated_docs.mkdir(parents=True)
    isolated_conf = isolated_docs / "conf.py"
    shutil.copy2(REDESIGN_CONF_PATH, isolated_conf)
    shutil.copy2(ROOT / "docs_redesign" / "release_status.json", isolated_docs)
    identity_path = isolated_docs / "_build_identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "source_revision": "0" * 40,
                "source_ref": "fingerprint-test",
                "dirty": False,
            }
        ),
        encoding="utf-8",
    )

    for relative in (
        "tests/data",
        "tests/fixtures/data",
        "tests/sample-data",
        "docs/_static/samples",
        "docs_redesign/_static/samples",
    ):
        source = ROOT / relative
        if source.is_dir():
            destination = isolated_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
    isolated_samples = isolated_root / "docs" / "_static" / "samples"
    isolated_samples.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(gwexpy, "__file__", str(isolated_package / "__init__.py"))
    base_env = runpy.run_path(str(isolated_conf))
    base_fingerprint = base_env["_execution_fingerprint"]

    # 1. Changing only the recorded Git SHA must NOT change the execution fingerprint.
    assert "source_revision" not in base_env["_execution_environment"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["source_revision"] = "1111222233334444555566667777888899990000"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    env_git_changed = runpy.run_path(str(isolated_conf))
    assert (
        env_git_changed["_build_identity"]["source_revision"]
        == "1111222233334444555566667777888899990000"
    )
    assert env_git_changed["_execution_fingerprint"] == base_fingerprint

    # 2. Modifying Python source invalidates the runtime source hash and fingerprint.
    probe_source = isolated_package / "_test_fingerprint_probe.py"
    probe_source.write_text("# probe for fingerprint invalidation\n", encoding="utf-8")
    env_source_changed = runpy.run_path(str(isolated_conf))
    assert (
        env_source_changed["_execution_environment"]["runtime_source_sha256"]
        != base_env["_execution_environment"]["runtime_source_sha256"]
    )
    assert env_source_changed["_execution_fingerprint"] != base_fingerprint
    probe_source.unlink()

    # 3. Modifying sample data invalidates the data hash and fingerprint.
    probe_data = isolated_samples / "_test_fingerprint_probe.csv"
    probe_data.write_text("x,y\n1,2\n", encoding="utf-8")
    env_data_changed = runpy.run_path(str(isolated_conf))
    assert (
        env_data_changed["_execution_environment"]["repo_data_sha256"]
        != base_env["_execution_environment"]["repo_data_sha256"]
    )
    assert env_data_changed["_execution_fingerprint"] != base_fingerprint
    probe_data.unlink()

    # 4. Changing execution settings invalidates their hash and fingerprint.
    with monkeypatch.context() as env_patch:
        env_patch.setenv("GWEXPY_NB_TIMEOUT", "999")
        env_setting_changed = runpy.run_path(str(isolated_conf))
    assert env_setting_changed["nb_execution_timeout"] == 999
    assert (
        env_setting_changed["_conf_execution_settings"]["nb_execution_timeout"] == 999
    )
    assert (
        env_setting_changed["_execution_environment"]["conf_execution_sha256"]
        != base_env["_execution_environment"]["conf_execution_sha256"]
    )
    assert env_setting_changed["_execution_fingerprint"] != base_fingerprint

    # 5. Changing execution environment invalidates the environment and fingerprint.
    base_execution_env = base_env["_execution_environment"]["env_execution"]
    previous_openblas_value = base_execution_env.get("OPENBLAS_NUM_THREADS")
    changed_openblas_value = "4" if previous_openblas_value != "4" else "3"
    with monkeypatch.context() as env_patch:
        env_patch.setenv("OPENBLAS_NUM_THREADS", changed_openblas_value)
        env_env_changed = runpy.run_path(str(isolated_conf))
    assert (
        env_env_changed["_execution_environment"]["env_execution"][
            "OPENBLAS_NUM_THREADS"
        ]
        == changed_openblas_value
    )
    assert (
        env_env_changed["_execution_environment"]["env_execution"][
            "OPENBLAS_NUM_THREADS"
        ]
        != previous_openblas_value
    )
    assert env_env_changed["_execution_fingerprint"] != base_fingerprint

    # Any files written by this test, including probes, must stay under tmp_path.
    assert writes
    assert all(path.is_relative_to(tmp_path.resolve()) for path in writes)


def test_explicit_notebook_build_keeps_nbsphinx_when_pandoc_exists(monkeypatch):
    monkeypatch.setenv("NBS_EXECUTE", "always")
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None
    )

    conf = _load_conf_module("gwexpy_docs_conf_with_pandoc")

    assert conf.nbsphinx_execute == "always"
    assert "nbsphinx" in conf.extensions
    assert "**/*.ipynb" not in conf.exclude_patterns


def test_github_actions_defaults_to_failing_on_notebook_errors(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("NBS_EXECUTE", raising=False)
    monkeypatch.delenv("NBS_ALLOW_ERRORS", raising=False)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None
    )

    conf = _load_conf_module("gwexpy_docs_conf_github_actions")

    assert conf.nbsphinx_allow_errors is False


def test_matplotlib_fonts_prefer_japanese_capable_sans_serif_stack(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("NBS_ALLOW_ERRORS", raising=False)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None
    )

    conf = _load_conf_module("gwexpy_docs_conf_fonts")

    assert "sans-serif" in conf.mpl.rcParams["font.family"]
    assert any(
        font_name in conf.mpl.rcParams["font.sans-serif"]
        for font_name in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic")
    )
    matplotlibrc = Path(conf.os.environ["MPLCONFIGDIR"]) / "matplotlibrc"
    content = matplotlibrc.read_text(encoding="utf-8")
    assert "backend: Agg" in content
    assert (
        "font.sans-serif: Noto Sans CJK JP, IPAexGothic, IPAGothic, DejaVu Sans"
        in content
    )
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
    pull_request = workflow[True]["pull_request"]
    assert set(pull_request["branches"]) >= {"main", "maint/0.2"}
    assert "paths" not in pull_request
    assert "paths-ignore" not in pull_request
    env = workflow["jobs"]["docs-pr"]["env"]
    assert env["NBS_EXECUTE"] == "never"
    assert env["NBS_ALLOW_ERRORS"] == "0"


def test_docs_redesign_pr_workflow_stages_the_canonical_changelog():
    workflow = _load_workflow(DOCS_PR_WORKFLOW_PATH)
    steps_by_name = {
        step["name"]: step
        for step in workflow["jobs"]["docs-redesign-pr"]["steps"]
        if "name" in step
    }
    prepare = steps_by_name["Prepare isolated docs_redesign source"]

    assert 'mkdir -p "${docs_root}"' in prepare["run"]
    assert 'python scripts/prepare_public_docs.py "${docs_src}"' in prepare["run"]


def test_docs_pages_workflow_builds_docs_redesign_with_executed_notebook_outputs():
    workflow = _load_workflow(DOCS_PAGES_WORKFLOW_PATH)
    assert workflow[True]["push"]["paths"] == list(DOCS_BUILD_INPUTS)
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "gh-pages-publish",
        "queue": "max",
    }

    build = workflow["jobs"]["build-pages"]
    publish = workflow["jobs"]["publish-pages"]
    assert build["permissions"] == {"contents": "read"}
    assert publish["permissions"] == {"actions": "read", "contents": "write"}
    assert publish["needs"] == "build-pages"
    assert publish["if"] == "github.ref == 'refs/heads/main'"

    # MyST-NB executes clean notebook sources in an isolated runner copy; the
    # old nbsphinx toggles for the previous docs/ tree do not apply here.
    assert "NBS_EXECUTE" not in build.get("env", {})
    assert "NBS_ALLOW_ERRORS" not in build.get("env", {})

    steps_by_name = {step["name"]: step for step in build["steps"] if "name" in step}
    prepare = steps_by_name["Prepare isolated docs_redesign source"]
    provision = steps_by_name["Provision docs_redesign build environment"]
    en_build = steps_by_name["Build EN HTML"]
    ja_build = steps_by_name["Build JA HTML"]
    package = steps_by_name["Package and hash the exact site tree"]

    assert 'python scripts/prepare_public_docs.py "${docs_src}"' in prepare["run"]
    assert 'mkdir -p "${docs_root}"' in prepare["run"]
    assert 'python scripts/prepare_public_docs.py "${docs_src}"' in prepare["run"]
    assert 'python -m pip install -e ".[all]"' in provision["run"]
    assert "prepare_docs_redesign.outputs.docs_src" in en_build["run"]
    assert "prepare_docs_redesign.outputs.docs_src" in ja_build["run"]
    assert "nbsphinx" not in en_build["run"]
    assert "nbsphinx" not in ja_build["run"]
    assert "-D language=ja" in ja_build["run"]
    assert "GWEXPY_DOCS_BASEURL" in en_build["env"]
    assert "GWEXPY_DOCS_BASEURL" in ja_build["env"]
    assert not any("GITHUB_TOKEN" in str(step) for step in build["steps"])
    assert not any("rsync" in str(step) for step in build["steps"])
    assert '--expected-revision "${GITHUB_SHA}"' in package["run"]

    publish_steps = {step["name"]: step for step in publish["steps"] if "name" in step}
    download = publish_steps["Download exact same-run Docs artifact"]
    verify = publish_steps["Verify exact EN and JA artifact contents"]
    order = publish_steps["Reject stale or unapproved source ordering"]
    readback_hashes = publish_steps["Verify published EN and JA artifact hashes"]
    prepare_publish = publish_steps["Prepare the gh-pages update"]
    gh_pages_checkout = publish_steps["Checkout gh-pages for the isolated publisher"]
    refresh_before_publish = publish_steps["Refresh main before Docs publication"]
    refresh_before_push = publish_steps["Recheck main before gh-pages push"]
    assert "artifact-ids" in download["with"]
    assert download["with"]["digest-mismatch"] == "error"
    assert "docs_publish_guard.py verify-artifact" in verify["run"]
    assert "docs_publish_guard.py check-order" in order["run"]
    assert "--preview-workflow" not in order["run"]
    assert "--expected-manifest" in readback_hashes["run"]
    assert "--manifest-only" in readback_hashes["run"]
    assert "--main-ref refs/remotes/origin/main" in order["run"]
    assert "fetch-depth" in gh_pages_checkout["with"]
    assert gh_pages_checkout["with"]["fetch-depth"] == 0
    assert "git fetch --no-tags origin" in refresh_before_publish["run"]
    assert "git fetch --no-tags origin" in refresh_before_push["run"]
    assert "docs_publish_guard.py check-order" in refresh_before_push["run"]
    assert "rsync -a --delete" in prepare_publish["run"]
    assert "artifact-digest" in str(build["outputs"])

    publish_names = [step["name"] for step in publish["steps"] if "name" in step]
    assert publish_names.index(
        "Verify exact EN and JA artifact contents"
    ) < publish_names.index("Checkout gh-pages for the isolated publisher")
    assert (
        publish_names.index("Refresh main before Docs publication")
        < publish_names.index("Reject stale or unapproved source ordering")
        < publish_names.index("Prepare the gh-pages update")
    )
    assert (
        publish_names.index("Prepare the gh-pages update")
        < publish_names.index("Recheck main before gh-pages push")
        < publish_names.index("Publish to gh-pages")
    )


def test_docs_preview_workflow_is_manual_dispatch_only():
    """No push trigger may publish to gh-pages from an unprotected branch."""
    workflow = _load_workflow(DOCS_PREVIEW_WORKFLOW_PATH)

    assert "push" not in workflow[True]
    assert "workflow_dispatch" in workflow[True]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "gh-pages-publish",
        "queue": "max",
    }
    build = workflow["jobs"]["build-preview"]
    publish = workflow["jobs"]["publish-preview"]
    assert build["permissions"] == {"contents": "read"}
    assert publish["permissions"] == {"actions": "read", "contents": "write"}
    assert publish["needs"] == "build-preview"
    assert publish["if"] == "github.ref == 'refs/heads/main'"
    assert not any("GITHUB_TOKEN" in str(step) for step in build["steps"])
    assert not any("rsync" in str(step) for step in build["steps"])

    steps_by_name = {step["name"]: step for step in build["steps"] if "name" in step}
    prepare = steps_by_name["Prepare isolated docs_redesign source"]
    assert 'mkdir -p "${docs_root}"' in prepare["run"]
    assert 'python scripts/prepare_public_docs.py "${docs_src}"' in prepare["run"]

    publish_steps = {step["name"]: step for step in publish["steps"] if "name" in step}
    download = publish_steps["Download exact same-run preview artifact"]
    verify = publish_steps["Verify exact EN and JA artifact contents"]
    order = publish_steps["Reject stale or unapproved source ordering"]
    readback_hashes = publish_steps["Verify published EN and JA preview hashes"]
    update = publish_steps["Prepare preview update"]
    refresh_before_publish = publish_steps["Refresh main before preview publication"]
    refresh_before_push = publish_steps["Recheck main before preview push"]
    gh_pages_checkout = publish_steps["Checkout gh-pages for the isolated publisher"]
    assert "artifact-ids" in download["with"]
    assert download["with"]["digest-mismatch"] == "error"
    assert "docs_publish_guard.py verify-artifact" in verify["run"]
    assert "docs_publish_guard.py check-order" in order["run"]
    assert "--preview-workflow" in order["run"]
    assert "--preview-workflow" in refresh_before_push["run"]
    assert "--expected-manifest" in readback_hashes["run"]
    assert "--manifest-only" in readback_hashes["run"]
    assert "--main-ref refs/remotes/origin/main" in order["run"]
    assert gh_pages_checkout["with"]["fetch-depth"] == 0
    assert "git fetch --no-tags origin" in refresh_before_publish["run"]
    assert "git fetch --no-tags origin" in refresh_before_push["run"]
    assert "docs_publish_guard.py check-order" in refresh_before_push["run"]
    assert "rsync -a --delete" in update["run"]
    publish_names = [step["name"] for step in publish["steps"] if "name" in step]
    assert (
        publish_names.index("Refresh main before preview publication")
        < publish_names.index("Reject stale or unapproved source ordering")
        < publish_names.index("Prepare preview update")
    )
    assert (
        publish_names.index("Prepare preview update")
        < publish_names.index("Recheck main before preview push")
        < publish_names.index("Publish preview to gh-pages")
    )
