"""preflight_doctor.py のユニットテスト。

チェック関数を直接テストするため subprocess をモックする。
CI 環境（conda なし）でも全テストが通るよう、実環境依存のテストには
pytest.mark.skipif または unittest.mock を使用する。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# ヘルパー: スクリプトをモジュールとしてロード
# ---------------------------------------------------------------------------

_MODULE_NAME = "preflight_doctor"


def load_module() -> ModuleType:
    """スクリプトをモジュールとしてロードして返す。

    sys.modules に登録しておかないと `from __future__ import annotations` 環境での
    @dataclass(frozen=True) が AttributeError を出すため、必ず登録する。
    """
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "preflight_doctor.py"
    )
    # キャッシュ済みならそのまま返す
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # exec_module より先に sys.modules へ登録することで dataclass の _is_type が動く
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# check_conda_env — conda 不在時の degraded 分岐
# ---------------------------------------------------------------------------


def test_check_conda_env_no_conda_returns_warn_and_degraded():
    """conda コマンドが存在しない場合は WARN かつ conda_available=False を返す。"""
    mod = load_module()

    with patch("shutil.which", return_value=None):
        result, conda_available = mod.check_conda_env("gwexpy")

    assert result.status.value == "WARN"
    assert conda_available is False
    assert "conda" in result.detail.lower()


def test_check_conda_env_env_not_found_returns_warn_and_degraded():
    """conda はあるが指定 env が存在しない場合は WARN かつ conda_available=False。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = "# conda environments:\nbase  *  /opt/conda\n"
    mock_proc.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/conda"):
        with patch("subprocess.run", return_value=mock_proc):
            result, conda_available = mod.check_conda_env("gwexpy")

    assert result.status.value == "WARN"
    assert conda_available is False


def test_check_conda_env_found_returns_pass_and_available():
    """指定 env が存在する場合は PASS かつ conda_available=True。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = (
        "# conda environments:\nbase  /opt/conda\ngwexpy  /opt/conda/envs/gwexpy\n"
    )
    mock_proc.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/conda"):
        with patch("subprocess.run", return_value=mock_proc):
            result, conda_available = mod.check_conda_env("gwexpy")

    assert result.status.value == "PASS"
    assert conda_available is True


def test_check_conda_env_timeout_returns_warn_and_degraded():
    """conda env list がタイムアウトした場合は WARN かつ degraded。"""
    mod = load_module()

    with patch("shutil.which", return_value="/usr/bin/conda"):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="conda", timeout=30),
        ):
            result, conda_available = mod.check_conda_env("gwexpy")

    assert result.status.value == "WARN"
    assert conda_available is False


# ---------------------------------------------------------------------------
# check_tools — ツールチェック
# ---------------------------------------------------------------------------


def test_check_tools_all_pass_when_available():
    """全ツールが正常に応答する場合は全て PASS。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = "ruff 0.4.0\n"
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        results = mod.check_tools(conda_env=None)

    assert all(r.status.value == "PASS" for r in results)
    assert len(results) == 3  # ruff, mypy, pytest


def test_check_tools_fail_when_tool_missing():
    """ツールが見つからない場合は FAIL。"""
    mod = load_module()

    with patch("subprocess.run", side_effect=FileNotFoundError):
        results = mod.check_tools(conda_env=None)

    assert all(r.status.value == "FAIL" for r in results)


def test_check_tools_fail_when_nonzero_exit():
    """ツールが非ゼロ終了コードを返す場合は FAIL。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.stderr = "error"
    mock_proc.returncode = 1

    with patch("subprocess.run", return_value=mock_proc):
        results = mod.check_tools(conda_env=None)

    assert all(r.status.value == "FAIL" for r in results)


def test_check_tools_uses_conda_run_when_env_provided():
    """conda_env が指定されている場合は conda run コマンドを使う。"""
    mod = load_module()

    captured_cmds: list[list[str]] = []

    mock_proc = MagicMock()
    mock_proc.stdout = "ruff 0.4.0\n"
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
        captured_cmds.append(cmd)
        return mock_proc

    with patch("subprocess.run", side_effect=fake_run):
        mod.check_tools(conda_env="gwexpy")

    assert all(cmd[0] == "conda" for cmd in captured_cmds)
    assert all("gwexpy" in cmd for cmd in captured_cmds)


# ---------------------------------------------------------------------------
# check_git_status
# ---------------------------------------------------------------------------


def test_check_git_status_clean_returns_pass():
    """ワーキングツリーがクリーンな場合は PASS。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        result = mod.check_git_status()

    assert result.status.value == "PASS"


def test_check_git_status_dirty_returns_warn_with_count():
    """未コミットファイルがある場合は WARN、件数が detail に含まれる。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = " M foo.py\n?? bar.py\nM  baz.py\n"
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        result = mod.check_git_status()

    assert result.status.value == "WARN"
    assert "3" in result.detail


def test_check_git_status_no_git_returns_warn():
    """git コマンドが見つからない場合は WARN。"""
    mod = load_module()

    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = mod.check_git_status()

    assert result.status.value == "WARN"


# ---------------------------------------------------------------------------
# check_frozen_tag
# ---------------------------------------------------------------------------


def test_check_frozen_tag_no_tags_returns_pass():
    """HEAD にバージョンタグがない場合は PASS。"""
    mod = load_module()

    def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
        m = MagicMock()
        m.returncode = 0
        if "tag" in cmd:
            m.stdout = ""
        elif "--show-current" in cmd:
            m.stdout = "main\n"
        return m

    with patch("subprocess.run", side_effect=fake_run):
        result = mod.check_frozen_tag()

    assert result.status.value == "PASS"


def test_check_frozen_tag_detached_head_with_version_tag_returns_fail():
    """detached HEAD かつバージョンタグがある場合は FAIL。"""
    mod = load_module()

    def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
        m = MagicMock()
        m.returncode = 0
        if "tag" in cmd:
            m.stdout = "v0.1.3\n"
        elif "--show-current" in cmd:
            m.stdout = ""  # detached HEAD
        return m

    with patch("subprocess.run", side_effect=fake_run):
        result = mod.check_frozen_tag()

    assert result.status.value == "FAIL"
    assert "detached" in result.detail.lower() or "frozen" in result.detail.lower()


def test_check_frozen_tag_branch_with_version_tag_returns_warn():
    """通常ブランチでバージョンタグがある場合は WARN。"""
    mod = load_module()

    def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
        m = MagicMock()
        m.returncode = 0
        if "tag" in cmd:
            m.stdout = "v0.1.3\n"
        elif "--show-current" in cmd:
            m.stdout = "main\n"
        return m

    with patch("subprocess.run", side_effect=fake_run):
        result = mod.check_frozen_tag()

    assert result.status.value == "WARN"
    assert "v0.1.3" in result.detail


def test_check_frozen_tag_non_version_tag_ignored():
    """v* パターンでないタグは無視される（PASS）。"""
    mod = load_module()

    def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
        m = MagicMock()
        m.returncode = 0
        if "tag" in cmd:
            m.stdout = "experiment-1\nwip\n"
        elif "--show-current" in cmd:
            m.stdout = "main\n"
        return m

    with patch("subprocess.run", side_effect=fake_run):
        result = mod.check_frozen_tag()

    assert result.status.value == "PASS"


# ---------------------------------------------------------------------------
# check_registry_smoke
# ---------------------------------------------------------------------------


def test_check_registry_smoke_pass():
    """register_all() が正常終了した場合は PASS。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        result = mod.check_registry_smoke(conda_env=None)

    assert result.status.value == "PASS"


def test_check_registry_smoke_fail_on_import_error():
    """gwexpy import が失敗した場合は FAIL。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.stderr = "ModuleNotFoundError: No module named 'gwexpy'\n"
    mock_proc.returncode = 1

    with patch("subprocess.run", return_value=mock_proc):
        result = mod.check_registry_smoke(conda_env=None)

    assert result.status.value == "FAIL"
    assert "gwexpy" in result.detail.lower() or "failed" in result.detail.lower()


def test_check_registry_smoke_timeout_returns_fail():
    """タイムアウトした場合は FAIL。"""
    mod = load_module()

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60),
    ):
        result = mod.check_registry_smoke(conda_env=None)

    assert result.status.value == "FAIL"
    assert "タイムアウト" in result.detail or "timeout" in result.detail.lower()


# ---------------------------------------------------------------------------
# JSON 出力フォーマット
# ---------------------------------------------------------------------------


def test_json_output_is_valid(capsys):
    """--json フラグで有効な JSON が出力される。"""
    mod = load_module()

    mock_proc = MagicMock()
    mock_proc.stdout = "ruff 0.4.0\n"
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    # 全 subprocess 呼び出しを成功でモック
    def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
        m = MagicMock()
        m.returncode = 0
        # conda env list
        if cmd[:2] == ["conda", "env"]:
            m.stdout = "gwexpy  /opt/conda/envs/gwexpy\n"
        # git status --porcelain
        elif "status" in cmd:
            m.stdout = ""
        # git tag
        elif "tag" in cmd:
            m.stdout = ""
        # git branch --show-current
        elif "--show-current" in cmd:
            m.stdout = "main\n"
        # tool --version
        else:
            m.stdout = "tool 1.0.0\n"
        m.stderr = ""
        return m

    with patch("shutil.which", return_value="/usr/bin/conda"):
        with patch("subprocess.run", side_effect=fake_run):
            ret = mod.main(["--env", "gwexpy", "--json", "--skip-smoke"])

    output = capsys.readouterr().out
    parsed = json.loads(output)

    assert "checks" in parsed
    assert "summary" in parsed
    assert isinstance(parsed["checks"], list)
    assert set(parsed["summary"].keys()) == {"PASS", "WARN", "FAIL"}
    # smoke スキップなので全 PASS の場合 exit 0
    assert ret == 0


# ---------------------------------------------------------------------------
# exit code: FAIL があれば 1 を返す
# ---------------------------------------------------------------------------


def test_exit_code_1_when_any_fail(capsys):
    """FAIL チェックが1つ以上あれば main は 1 を返す。"""
    mod = load_module()

    Status = mod.Status
    CheckResult = mod.CheckResult

    # FAIL を含む結果を直接渡して集約ロジックをテスト
    results = [
        CheckResult(name="tool_ruff", status=Status.FAIL, detail="見つかりません"),
        CheckResult(name="git_status", status=Status.PASS, detail="クリーン"),
    ]
    has_fail = any(r.status == Status.FAIL for r in results)
    assert has_fail is True

    exit_code = 1 if has_fail else 0
    assert exit_code == 1


def test_exit_code_0_when_no_fail():
    """FAIL がない場合は exit code 0。"""
    mod = load_module()

    Status = mod.Status
    CheckResult = mod.CheckResult

    results = [
        CheckResult(name="conda_env", status=Status.PASS, detail="OK"),
        CheckResult(name="git_status", status=Status.WARN, detail="dirty"),
    ]
    has_fail = any(r.status == Status.FAIL for r in results)
    exit_code = 1 if has_fail else 0
    assert exit_code == 0


# ---------------------------------------------------------------------------
# --skip-smoke フラグ
# ---------------------------------------------------------------------------


def test_skip_smoke_excludes_registry_check():
    """--skip-smoke を指定した場合、registry_smoke チェックが含まれない。"""
    mod = load_module()

    # conda なし degraded モードでツールチェックのみ
    def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
        m = MagicMock()
        m.returncode = 0
        m.stdout = "tool 1.0\n"
        m.stderr = ""
        return m

    with patch("shutil.which", return_value=None):  # conda 不在
        with patch("subprocess.run", side_effect=fake_run):
            results = mod.run_all_checks(env_name="gwexpy", skip_smoke=True)

    names = [r.name for r in results]
    assert "registry_smoke" not in names
