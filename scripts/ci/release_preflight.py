#!/usr/bin/env python3
"""Run existing release validators for a phase and report their exit codes.

Adds no new pass/fail rules; delegates to existing CLI scripts only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

Phase = str

PHASES: tuple[Phase, ...] = (
    "preparation",
    "source",
    "candidate",
    "prepublish",
    "postrelease",
)


def _commands(
    repo_root: Path, release_tag: str, phase: Phase
) -> list[tuple[str, list[str]]]:
    py = sys.executable
    root = repo_root.resolve()
    rel = [
        (
            "release_contract_review_evidence_path",
            [
                py,
                str(root / "scripts/ci/release_contract.py"),
                "--review-evidence-path",
                release_tag,
            ],
        ),
        (
            "check_release_metadata",
            [py, str(root / "scripts/check_release_metadata.py")],
        ),
    ]
    if phase == "preparation":
        return rel[:1]
    if phase == "source":
        return [
            (
                "validate_release_review_evidence",
                [
                    py,
                    str(root / "scripts/ci/validate_release_review_evidence.py"),
                    "--help",
                ],
            ),
        ]
    if phase == "candidate":
        version = release_tag.removeprefix("v")
        return [
            (
                "release_promotion_profile",
                [
                    py,
                    str(root / "scripts/ci/release_promotion.py"),
                    "profile",
                    "--version",
                    version,
                ],
            ),
        ]
    if phase == "prepublish":
        return [
            (
                "verify_release_human_approval_help",
                [
                    py,
                    str(root / "scripts/ci/verify_release_human_approval.py"),
                    "--help",
                ],
            ),
            (
                "verify_release_go_help",
                [py, str(root / "scripts/ci/verify_release_go.py"), "--help"],
            ),
        ]
    if phase == "postrelease":
        return [
            (
                "validate_github_release_readback_help",
                [
                    py,
                    str(root / "scripts/ci/validate_github_release_readback.py"),
                    "--help",
                ],
            ),
        ]
    raise ValueError(f"unknown phase: {phase}")


def run_preflight(repo_root: Path, release_tag: str, phase: Phase) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for name, argv in _commands(repo_root, release_tag, phase):
        proc = subprocess.run(argv, cwd=repo_root, text=True, capture_output=True, check=False)
        results.append(
            {
                "name": name,
                "argv": argv,
                "exit_code": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-500:],
                "stderr_tail": (proc.stderr or "")[-500:],
            }
        )
    return {
        "schema": "gwexpy-release-preflight-v1",
        "release_tag": release_tag,
        "phase": phase,
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_preflight(args.repo_root, args.release_tag, args.phase)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in payload["results"]:
            assert isinstance(item, dict)
            print(f"{item['name']}: exit={item['exit_code']}")
    worst = max(int(item["exit_code"]) for item in payload["results"])  # type: ignore[index]
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
