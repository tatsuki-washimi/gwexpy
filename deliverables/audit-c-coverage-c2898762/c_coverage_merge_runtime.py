#!/usr/bin/env python3
"""Attach runtime probe outcomes to ledger rows (identifier substring match)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    base = Path(__file__).resolve().parent
    ledger = [
        json.loads(l) for l in base.joinpath("c-coverage-c2898762.jsonl").read_text().splitlines()
    ]
    probes = [
        json.loads(l)
        for l in base.joinpath("c-coverage-runtime-probes.jsonl").read_text().splitlines()
    ]

    for p in probes:
        needle = p.get("target_identifier") or p.get("identifier", "")
        for rec in ledger:
            if rec["identifier"] != needle:
                continue
            rec["runtime_status"] = (
                "executed_this_pass"
                if p.get("execution_status") == "executed"
                else "not_run"
            )
            rec.setdefault("evidence", {})["runtime_probe"] = p
            if p.get("related_issue"):
                rec["related_issue"] = p["related_issue"]
            break

    summary = json.loads(base.joinpath("c-coverage-c2898762-summary.json").read_text())
    stats_path = base / "c-coverage-probe-stats.json"
    probe_stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    summary["runtime_probes"] = probe_stats or {
        "probes_total": len(probes),
        "executed": sum(1 for p in probes if p.get("execution_status") == "executed"),
        "not_run": sum(1 for p in probes if p.get("execution_status") == "not_run"),
    }
    summary["ledger_runtime_status_after_merge"] = dict(
        __import__("collections").Counter(r["runtime_status"] for r in ledger)
    )

    base.joinpath("c-coverage-c2898762.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ledger), encoding="utf-8"
    )
    base.joinpath("c-coverage-c2898762-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["runtime_probes"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
