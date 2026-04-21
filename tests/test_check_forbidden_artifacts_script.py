from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_forbidden_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location("check_forbidden_artifacts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_forbidden_keeps_hidden_prefixes(tmp_path: Path):
    module = load_script_module()
    artifact_path = tmp_path / ".pytest_cache" / "CACHEDIR.TAG"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Signature: 8a477f597d28d172789f06886806bc55", encoding="utf-8")

    matches = module._find_forbidden([str(artifact_path.relative_to(tmp_path))])

    assert matches == [".pytest_cache/CACHEDIR.TAG"]
