from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "bootstrap_trusted_pip.py"


def _load_bootstrap_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trusted_pip_bootstrap", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_uses_hash_pinned_wheel_then_checks_vendored_certifi(
    monkeypatch: Any,
) -> None:
    module = _load_bootstrap_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []
    requirements_contents: list[str] = []

    def record_run(command: list[str], **kwargs: Any) -> None:
        if "-r" in command:
            requirements_contents.append(
                Path(command[command.index("-r") + 1]).read_text(encoding="utf-8")
            )
        calls.append((command, kwargs))

    monkeypatch.setattr(module.subprocess, "run", record_run)
    module.bootstrap("/venv/bin/python")

    assert len(calls) == 2
    install, verify = calls
    command, kwargs = install
    assert command[:4] == ["/venv/bin/python", "-m", "pip", "install"]
    assert "--require-hashes" in command
    assert "--no-deps" in command
    assert "--index-url" in command
    assert "https://pypi.org/simple" in command
    assert "-r" in command
    requirements_file = Path(command[command.index("-r") + 1])
    assert requirements_contents == [
        "pip==26.2.2 "
        "--hash=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70\n"
    ]
    assert kwargs["check"] is True
    assert kwargs["env"]["PIP_CONFIG_FILE"]
    assert not requirements_file.exists()

    verify_command, verify_kwargs = verify
    assert verify_command[:2] == ["/venv/bin/python", "-c"]
    assert "pip.__version__" in verify_command[2]
    assert "certifi.__version__" in verify_command[2]
    assert "2026.10.19" in verify_command[2]
    assert verify_kwargs["check"] is True
    assert not requirements_file.exists()
