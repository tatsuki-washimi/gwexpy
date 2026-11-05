#!/usr/bin/env python3
"""Run existing release validators for a phase and report their exit codes.

Adds no new pass/fail rules; delegates to existing CLI scripts only.
Steps without required inputs are reported as ``not_evaluated`` (not success).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Phase = str
Disposition = Literal["executed", "not_evaluated"]

PHASES: tuple[Phase, ...] = (
    "preparation",
    "source",
    "candidate",
    "prepublish",
    "postrelease",
)


@dataclass(frozen=True)
class PreflightInputs:
    repo_root: Path
    release_tag: str
    review_evidence: Path | None = None
    reviewed_commit: str | None = None
    release_ref: str | None = None
    repository: str | None = None
    human_approval_evidence: Path | None = None
    go_comment_id: int | None = None


def _tool_root(fallback: Path) -> Path:
    installed = Path(__file__).resolve().parents[2]
    if (installed / "scripts/ci/validate_release_review_evidence.py").is_file():
        return installed
    return fallback.resolve()


def _run(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)


def _executed(name: str, argv: list[str], proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "name": name,
        "disposition": "executed",
        "argv": argv,
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-500:],
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def _load_validate_release_module(tool_root: Path) -> Any:
    path = tool_root / "scripts" / "validate_release.py"
    spec = importlib.util.spec_from_file_location("validate_release_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("validate_release.py is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_validate_s_to_r(
    repo_root: Path,
    tool_root: Path,
    reviewed_commit: str,
    release_ref: str,
    expected_tag: str,
) -> dict[str, Any]:
    name = "validate_s_to_r"
    label = [
        "validate_s_to_r",
        f"reviewed={reviewed_commit}",
        f"release_ref={release_ref}",
        f"tag={expected_tag}",
    ]
    try:
        module = _load_validate_release_module(tool_root)
        module.validate_s_to_r(
            repo_root,
            reviewed_commit,
            release_ref,
            expected_tag=expected_tag,
        )
    except Exception as exc:
        proc = subprocess.CompletedProcess(label, 1, "", str(exc))
        return _executed(name, label, proc)
    proc = subprocess.CompletedProcess(label, 0, "validate_s_to_r=ok", "")
    return _executed(name, label, proc)


def _not_evaluated(name: str, *, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "disposition": "not_evaluated",
        "reason": reason,
        "exit_code": None,
    }


def _steps(inputs: PreflightInputs, phase: Phase) -> list[dict[str, Any]]:
    py = sys.executable
    root = inputs.repo_root.resolve()
    tools = _tool_root(root)
    results: list[dict[str, Any]] = []

    if phase == "preparation":
        argv = [
            py,
            str(tools / "scripts/ci/release_contract.py"),
            "--review-evidence-path",
            inputs.release_tag,
        ]
        results.append(_executed("release_contract_review_evidence_path", argv, _run(argv, cwd=root)))
        return results

    if phase == "source":
        if inputs.review_evidence is None:
            results.append(
                _not_evaluated(
                    "validate_release_review_evidence",
                    reason="missing --review-evidence (validator not run)",
                )
            )
            return results
        reviewed_commit = inputs.reviewed_commit or inputs.release_ref
        argv = [
            py,
            str(tools / "scripts/ci/validate_release_review_evidence.py"),
            "--evidence",
            str(inputs.review_evidence),
            "--expected-tag",
            inputs.release_tag,
            "--repo-root",
            str(root),
            "--phase",
            "source-review",
        ]
        if reviewed_commit:
            argv.extend(["--reviewed-commit", reviewed_commit])
        results.append(_executed("validate_release_review_evidence", argv, _run(argv, cwd=root)))
        if inputs.release_ref and reviewed_commit and inputs.release_ref != reviewed_commit:
            results.append(
                _run_validate_s_to_r(
                    root,
                    tools,
                    reviewed_commit,
                    inputs.release_ref,
                    inputs.release_tag,
                )
            )
        elif inputs.release_ref:
            results.append(
                _not_evaluated(
                    "validate_s_to_r",
                    reason="release_ref equals reviewed commit; S-to-R delta check skipped",
                )
            )
        else:
            results.append(
                _not_evaluated(
                    "validate_s_to_r",
                    reason="missing --release-ref (S-to-R binding not evaluated)",
                )
            )
        return results

    if phase == "candidate":
        version = inputs.release_tag.removeprefix("v")
        argv = [
            py,
            str(tools / "scripts/ci/release_promotion.py"),
            "profile",
            "--version",
            version,
        ]
        results.append(_executed("release_promotion_profile", argv, _run(argv, cwd=root)))
        return results

    if phase == "prepublish":
        missing: list[str] = []
        if inputs.repository is None:
            missing.append("--repository")
        if inputs.human_approval_evidence is None:
            missing.append("--human-approval-evidence")
        if missing:
            results.append(
                _not_evaluated(
                    "verify_release_human_approval",
                    reason=f"missing {', '.join(missing)} (approval validator not run)",
                )
            )
        else:
            argv = [
                py,
                str(tools / "scripts/ci/verify_release_human_approval.py"),
                "--repo-root",
                str(root),
                "--expected-tag",
                inputs.release_tag,
                "--repository",
                inputs.repository,
                "--evidence",
                str(inputs.human_approval_evidence),
            ]
            results.append(_executed("verify_release_human_approval", argv, _run(argv, cwd=root)))
        if inputs.go_comment_id is None or inputs.repository is None:
            results.append(
                _not_evaluated(
                    "verify_release_go",
                    reason="missing --go-comment-id and/or --repository (GO validator not run)",
                )
            )
        else:
            results.append(
                _not_evaluated(
                    "verify_release_go",
                    reason="verify_release_go has no stable CLI entrypoint; use release_promotion or dedicated script with full inputs",
                )
            )
        return results

    if phase == "postrelease":
        results.append(
            _not_evaluated(
                "validate_github_release_readback",
                reason="missing readback artifact paths (postrelease validator not run)",
            )
        )
        return results

    raise ValueError(f"unknown phase: {phase}")


def run_preflight(inputs: PreflightInputs, phase: Phase) -> dict[str, object]:
    results = _steps(inputs, phase)
    executed = [item for item in results if item["disposition"] == "executed"]
    not_evaluated = [item for item in results if item["disposition"] == "not_evaluated"]
    executed_failures = [
        item for item in executed if int(item["exit_code"]) != 0  # type: ignore[arg-type]
    ]
    return {
        "schema": "gwexpy-release-preflight-v2",
        "release_tag": inputs.release_tag,
        "phase": phase,
        "results": results,
        "summary": {
            "executed": len(executed),
            "not_evaluated": len(not_evaluated),
            "executed_failures": len(executed_failures),
        },
    }


def exit_code_for_payload(payload: dict[str, object]) -> int:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return 1
    failures = summary.get("executed_failures", 0)
    return 1 if isinstance(failures, int) and failures > 0 else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--review-evidence", type=Path)
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--release-ref")
    parser.add_argument("--repository")
    parser.add_argument("--human-approval-evidence", type=Path)
    parser.add_argument("--go-comment-id", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    inputs = PreflightInputs(
        repo_root=args.repo_root,
        release_tag=args.release_tag,
        review_evidence=args.review_evidence,
        reviewed_commit=args.reviewed_commit,
        release_ref=args.release_ref,
        repository=args.repository,
        human_approval_evidence=args.human_approval_evidence,
        go_comment_id=args.go_comment_id,
    )
    payload = run_preflight(inputs, args.phase)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in payload["results"]:
            assert isinstance(item, dict)
            if item["disposition"] == "executed":
                print(f"{item['name']}: executed exit={item['exit_code']}")
            else:
                print(f"{item['name']}: not_evaluated ({item['reason']})")
    return exit_code_for_payload(payload)


if __name__ == "__main__":
    raise SystemExit(main())
