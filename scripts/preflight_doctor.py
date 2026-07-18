#!/usr/bin/env python3
"""Preflight Doctor — セッション開始前の環境チェックスクリプト。

各チェックは PASS / WARN / FAIL を返す純粋関数として分離されており、
単体テストが容易な構造になっている。

Usage:
    python scripts/preflight_doctor.py [--env gwexpy] [--skip-smoke] [--json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_conda_env(env_name: str) -> tuple[CheckResult, bool]:
    """conda コマンドの存在と指定 env の存在を確認する。

    Returns:
        (CheckResult, conda_available): conda_available が False の場合、
        以降のツールチェックは degraded モード（現行インタープリタ）で実行する。
    """
    conda_path = shutil.which("conda")
    if conda_path is None:
        return (
            CheckResult(
                name="conda_env",
                status=Status.WARN,
                detail="conda コマンドが見つかりません。以降のツールチェックは現行インタープリタで実行します（degraded モード）",
            ),
            False,
        )

    try:
        proc = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return (
            CheckResult(
                name="conda_env",
                status=Status.WARN,
                detail="conda env list がタイムアウトしました（30s）。degraded モードで続行します",
            ),
            False,
        )
    except FileNotFoundError:
        return (
            CheckResult(
                name="conda_env",
                status=Status.WARN,
                detail="conda 実行に失敗しました。degraded モードで続行します",
            ),
            False,
        )

    # env 名が一覧に含まれるか確認（"name" または "/path/name" 形式どちらにも対応）
    found = any(
        line.split()[0] == env_name
        or (line and line.split()[0].endswith(f"/{env_name}"))
        for line in proc.stdout.splitlines()
        if line and not line.startswith("#")
    )

    if not found:
        return (
            CheckResult(
                name="conda_env",
                status=Status.WARN,
                detail=f"conda 環境 '{env_name}' が見つかりません。degraded モードで続行します",
            ),
            False,
        )

    return (
        CheckResult(
            name="conda_env",
            status=Status.PASS,
            detail=f"conda 環境 '{env_name}' が存在します",
        ),
        True,
    )


def _run_tool_version(
    tool: str,
    *,
    conda_env: str | None,
    timeout: int = 30,
) -> CheckResult:
    """指定ツールの --version を実行してチェック結果を返す。

    conda_env が None の場合は現行インタープリタ（degraded モード）で実行する。
    """
    if conda_env is not None:
        cmd = ["conda", "run", "-n", conda_env, tool, "--version"]
    else:
        cmd = [tool, "--version"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return CheckResult(
            name=f"tool_{tool}",
            status=Status.FAIL,
            detail=f"'{tool}' コマンドが見つかりません",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name=f"tool_{tool}",
            status=Status.FAIL,
            detail=f"'{tool} --version' がタイムアウトしました（{timeout}s）",
        )

    if proc.returncode != 0:
        return CheckResult(
            name=f"tool_{tool}",
            status=Status.FAIL,
            detail=f"'{tool} --version' が返却コード {proc.returncode} で失敗しました",
        )

    version_line = (
        (proc.stdout + proc.stderr).strip().splitlines()[0]
        if (proc.stdout + proc.stderr).strip()
        else "(バージョン不明)"
    )
    return CheckResult(
        name=f"tool_{tool}",
        status=Status.PASS,
        detail=version_line,
    )


def check_tools(
    conda_env: str | None,
) -> list[CheckResult]:
    """ruff / mypy / pytest の存在をチェックする。"""
    tools = ["ruff", "mypy", "pytest"]
    return [_run_tool_version(t, conda_env=conda_env) for t in tools]


def check_git_status() -> CheckResult:
    """git status --porcelain で dirty ファイル数を確認する。"""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return CheckResult(
            name="git_status",
            status=Status.WARN,
            detail="git コマンドが見つかりません",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="git_status",
            status=Status.WARN,
            detail="git status がタイムアウトしました（30s）",
        )

    if proc.returncode != 0:
        return CheckResult(
            name="git_status",
            status=Status.WARN,
            detail=f"git status が返却コード {proc.returncode} で失敗しました",
        )

    dirty_files = [line for line in proc.stdout.splitlines() if line.strip()]
    if dirty_files:
        return CheckResult(
            name="git_status",
            status=Status.WARN,
            detail=f"未コミットの変更が {len(dirty_files)} ファイルあります",
        )

    return CheckResult(
        name="git_status",
        status=Status.PASS,
        detail="ワーキングツリーはクリーンです",
    )


def check_frozen_tag() -> CheckResult:
    """HEAD にバージョンタグが付いている場合の frozen tag 検知。

    - detached HEAD + バージョンタグ → FAIL（frozen バージョン上での作業）
    - 通常ブランチ + バージョンタグ → WARN（タグ付きコミット上での作業）
    - タグなし → PASS
    """
    # HEAD のタグを取得
    try:
        tag_proc = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return CheckResult(
            name="frozen_tag",
            status=Status.WARN,
            detail="git コマンドが見つかりません",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="frozen_tag",
            status=Status.WARN,
            detail="git tag がタイムアウトしました（30s）",
        )

    if tag_proc.returncode != 0:
        return CheckResult(
            name="frozen_tag",
            status=Status.WARN,
            detail=f"git tag の実行に失敗しました（返却コード {tag_proc.returncode}）",
        )

    version_tags = [
        t.strip() for t in tag_proc.stdout.splitlines() if t.strip().startswith("v")
    ]

    if not version_tags:
        return CheckResult(
            name="frozen_tag",
            status=Status.PASS,
            detail="HEAD にバージョンタグはありません",
        )

    # 現在ブランチを確認（detached HEAD の場合は空文字列）
    try:
        branch_proc = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        current_branch = branch_proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        current_branch = ""

    tags_str = ", ".join(version_tags)

    if not current_branch:
        # detached HEAD — frozen バージョン上での作業
        return CheckResult(
            name="frozen_tag",
            status=Status.FAIL,
            detail=f"detached HEAD かつバージョンタグ ({tags_str}) が付いています。frozen バージョン上での作業は禁止されています",
        )

    # 通常ブランチでもタグ付きコミットは WARN
    return CheckResult(
        name="frozen_tag",
        status=Status.WARN,
        detail=f"HEAD にバージョンタグ ({tags_str}) が付いています（ブランチ: {current_branch}）。タグ付きコミット上での直接作業に注意してください",
    )


def check_registry_smoke(
    conda_env: str | None,
    timeout: int = 60,
) -> CheckResult:
    """gwexpy.register_all() が正常に実行できるか確認する。"""
    code = "import gwexpy; gwexpy.register_all()"

    if conda_env is not None:
        cmd = ["conda", "run", "-n", conda_env, "python", "-c", code]
    else:
        cmd = [sys.executable, "-c", code]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return CheckResult(
            name="registry_smoke",
            status=Status.FAIL,
            detail="python インタープリタが見つかりません",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="registry_smoke",
            status=Status.FAIL,
            detail=f"gwexpy.register_all() がタイムアウトしました（{timeout}s）",
        )

    if proc.returncode != 0:
        err_lines = proc.stderr.strip().splitlines()
        err_summary = err_lines[-1] if err_lines else "(エラー詳細なし)"
        return CheckResult(
            name="registry_smoke",
            status=Status.FAIL,
            detail=f"gwexpy.register_all() が失敗しました: {err_summary}",
        )

    return CheckResult(
        name="registry_smoke",
        status=Status.PASS,
        detail="gwexpy.register_all() が正常に実行されました",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_checks(
    env_name: str,
    skip_smoke: bool,
) -> list[CheckResult]:
    """全チェックを実行して CheckResult のリストを返す。"""
    results: list[CheckResult] = []

    # 1. conda 環境チェック
    conda_result, conda_available = check_conda_env(env_name)
    results.append(conda_result)

    # conda が使えない場合は degraded モード（conda_env=None）
    effective_conda = env_name if conda_available else None

    # 2. ツールチェック
    results.extend(check_tools(effective_conda))

    # 3. git 状態チェック
    results.append(check_git_status())

    # 4. frozen tag チェック
    results.append(check_frozen_tag())

    # 5. registry smoke チェック
    if not skip_smoke:
        results.append(check_registry_smoke(effective_conda))

    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_results_text(results: list[CheckResult]) -> str:
    """テキスト形式の出力文字列を返す。"""
    lines: list[str] = []
    for r in results:
        lines.append(f"[{r.status.value}] {r.name}: {r.detail}")

    # サマリ行
    counts = {s: sum(1 for r in results if r.status == s) for s in Status}
    summary = "  ".join(f"{s.value}: {counts[s]}" for s in Status)
    lines.append(f"\nSummary — {summary}")

    return "\n".join(lines)


def format_results_json(results: list[CheckResult]) -> str:
    """JSON 形式の出力文字列を返す。"""
    counts = {s.value: sum(1 for r in results if r.status == s) for s in Status}
    payload = {
        "checks": [
            {"name": r.name, "status": r.status.value, "detail": r.detail}
            for r in results
        ],
        "summary": counts,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="gwexpy",
        metavar="ENV_NAME",
        help="使用する conda 環境名（デフォルト: gwexpy）",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        default=False,
        help="gwexpy.register_all() の smoke チェックをスキップする",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="output_json",
        help="machine-readable な JSON 形式で出力する",
    )
    args = parser.parse_args(argv)

    results = run_all_checks(env_name=args.env, skip_smoke=args.skip_smoke)

    if args.output_json:
        print(format_results_json(results))
    else:
        print(format_results_text(results))

    # FAIL が1つでもあれば exit 1
    has_fail = any(r.status == Status.FAIL for r in results)
    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
