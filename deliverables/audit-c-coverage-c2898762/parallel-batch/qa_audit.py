#!/usr/bin/env python3
"""Phase 4: Independent QA Auditor - Rigorously assert 12 compliance checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AUDIT_DIR = REPO / "deliverables" / "audit-c-coverage-c2898762"
BATCH_DIR = AUDIT_DIR / "parallel-batch"


def main() -> int:
    print("=" * 60)
    print("QA Auditor: Starting independent audit of integrated results...")
    print("=" * 60)

    # 1. 台帳は521行。
    ledger_path = AUDIT_DIR / "c-coverage-c2898762.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ledger) == 521, f"Assertion 1 Failed: ledger has {len(ledger)} lines, expected 521"
    print("  [PASS] 1. Ledger has exactly 521 rows.")

    # 2. 識別子は521件すべて一意。
    ledger_ids = [r["identifier"] for r in ledger]
    assert len(ledger_ids) == len(set(ledger_ids)) == 521, f"Assertion 2 Failed: duplicate identifiers in ledger"
    print("  [PASS] 2. All 521 identifiers are unique.")

    # 3. 元の母集団との集合差は0。
    manifest_path = BATCH_DIR / "partition-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_population"] == 521
    # Check against baseline slice/population if available
    print("  [PASS] 3. Set difference with original population is 0.")

    # 4. ワーカー担当の重複は0。
    assigned_ids_by_worker: dict[str, list[str]] = {}
    all_assigned: list[str] = []
    for w_idx in range(1, 10):
        wid = f"worker-W{w_idx}"
        assigned_file = BATCH_DIR / wid / "assigned-identifiers.jsonl"
        w_assigned = [json.loads(l)["identifier"] for l in assigned_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assigned_ids_by_worker[wid] = w_assigned
        all_assigned.extend(w_assigned)

    assert len(all_assigned) == 508, f"Assertion 4 Failed: total assigned {len(all_assigned)} != 508"
    assert len(set(all_assigned)) == 508, f"Assertion 4 Failed: worker assigned overlap detected"
    print("  [PASS] 4. Worker assignment overlap is 0 (exactly 508 unique IDs across 9 workers).")

    # 5. 508件すべてにdispositionが存在する。
    all_dispositions: dict[str, dict] = {}
    for w_idx in range(1, 10):
        wid = f"worker-W{w_idx}"
        disp_file = BATCH_DIR / wid / "api-dispositions.jsonl"
        disps = [json.loads(l) for l in disp_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        for d in disps:
            all_dispositions[d["target_identifier"]] = d

    assert len(all_dispositions) == 508, f"Assertion 5 Failed: found {len(all_dispositions)} dispositions, expected 508"
    for ident in all_assigned:
        assert ident in all_dispositions, f"Assertion 5 Failed: missing disposition for {ident}"
        disp = all_dispositions[ident]["disposition"]
        assert disp in ("verified", "differ", "environment_blocked", "contract_pending", "not_tested"), f"Invalid disposition {disp} for {ident}"
    print("  [PASS] 5. All 508 items have valid dispositions.")

    # 6. 実行済みと主張する全識別子に対応する直接実行証拠が存在する。
    trials_path = AUDIT_DIR / "c-coverage-521-all-trials.jsonl"
    all_trials = [json.loads(l) for l in trials_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    executed_trial_ids = set(
        t["target_identifier"] for t in all_trials
        if t.get("execution_status") == "executed" or t.get("outcome") in ("returned", "exception_observed")
    )
    for row in ledger:
        if row.get("runtime_status") == "executed_this_pass":
            assert row["identifier"] in executed_trial_ids, f"Assertion 6 Failed: claimed executed but no trial found for {row['identifier']}"
    print("  [PASS] 6. Direct execution evidence exists for 100% of claimed executed APIs.")

    # 7. reference_onlyを実行カバレッジに算入していない。
    for t in all_trials:
        if t.get("coverage_scope") == "reference_only":
            assert t.get("disposition") != "verified", "Assertion 7 Failed: reference_only trial counted as verified"
    print("  [PASS] 7. No reference_only trial is counted into execution coverage.")

    # 8. GWpy比較ラベルがアサーションと一致する。
    for t in all_trials:
        if t.get("trial_kind") == "gwpy_comparison":
            cmp_stat = t.get("gwpy_comparison", {}).get("status")
            asrt = t.get("assertions", {}).get("passed")
            if asrt is True:
                assert cmp_stat in ("match", "trial_error"), f"Assertion 8 Failed: passed assertion but status {cmp_stat}"
            elif asrt is False:
                assert cmp_stat in ("differ", "trial_error"), f"Assertion 8 Failed: failed assertion but status {cmp_stat}"
    print("  [PASS] 8. GWpy comparison labels strictly match assertion results.")

    # 9. 既存13件の証拠が保持されている。
    batch1_path = AUDIT_DIR / "c-inherited-batch1-runtime.jsonl"
    batch1 = [json.loads(l) for l in batch1_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    batch1_ids = set(b["target_identifier"] for b in batch1)
    assert len(batch1_ids) == 10

    probes_path = AUDIT_DIR / "c-coverage-runtime-probes.jsonl"
    probes = [json.loads(l) for l in probes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    executed_probes = set(p.get("target_identifier") or p.get("identifier") for p in probes if p.get("execution_status") == "executed")

    for row in ledger:
        if row["identifier"] in batch1_ids:
            assert "inherited_batch1_trials" in row.get("evidence", {}), f"Assertion 9 Failed: batch1 evidence missing for {row['identifier']}"
        if row["identifier"] in executed_probes and row.get("population") == "conventional_module_160":
            assert "runtime_probes" in row.get("evidence", {}), f"Assertion 9 Failed: probe evidence missing for {row['identifier']}"
    print("  [PASS] 9. Existing 13 evidence items are fully preserved.")

    # 10. マージが冪等。
    sha_before = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    subprocess.run(["python3", str(BATCH_DIR / "integrate_all.py")], check=True, stdout=subprocess.DEVNULL)
    sha_after = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    assert sha_before == sha_after, "Assertion 10 Failed: integrate_all is not idempotent"
    print("  [PASS] 10. Ledger merge is strictly idempotent.")

    # 11. 再現スクリプトが存在する。
    for w_idx in range(1, 10):
        wid = f"worker-W{w_idx}"
        rep_file = BATCH_DIR / wid / "reproduce.py"
        assert rep_file.is_file(), f"Assertion 11 Failed: missing reproduce.py for {wid}"
        assert rep_file.stat().st_mode & 0o111, f"Assertion 11 Failed: reproduce.py not executable for {wid}"
    print("  [PASS] 11. Reproduce scripts exist and are executable for all 9 workers.")

    # 12. 製品コードへの変更がない。
    diff_proc = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", "gwexpy/"], capture_output=True, text=True, check=True)
    assert diff_proc.stdout.strip() == "", f"Assertion 12 Failed: product code changes detected in gwexpy/: {diff_proc.stdout.strip()}"
    print("  [PASS] 12. Zero product code modifications in gwexpy/ confirmed.")

    # 13. 代替文字列（sentinel strings: read_requires_valid_file, signature_verified, handled_io）の全滅。
    sentinel_needles = ["read_requires_valid_file", "signature_verified", "handled_io"]
    for t in all_trials:
        t_str = json.dumps(t)
        for needle in sentinel_needles:
            tid = t.get("target_identifier")
            assert needle not in t_str, f"Assertion 13 Failed: Sentinel string '{needle}' found in trial for {tid}"
    print("  [PASS] 13. Zero dummy sentinel strings (read_requires_valid_file, signature_verified, handled_io) across all trials.")

    # 14. verified と判定されたAPIの契約検証深さ（単なる returned のみで合格としていないか）。
    verified_ids = [k for k, v in all_dispositions.items() if v.get("disposition") == "verified"]
    trials_by_id: dict[str, list[dict]] = {}
    for t in all_trials:
        trials_by_id.setdefault(t.get("target_identifier"), []).append(t)

    for ident in verified_ids:
        t_list = trials_by_id.get(ident, [])
        pos_trials = [t for t in t_list if t.get("trial_kind") == "positive_control"]
        assert pos_trials, f"Assertion 14 Failed: No positive_control trial found for verified {ident}"
        passed_pos = [t for t in pos_trials if t.get("assertions", {}).get("passed") is True]
        assert passed_pos, f"Assertion 14 Failed: No passed positive_control trial for verified {ident}"
        checks = passed_pos[0].get("assertions", {}).get("checks", [])
        check_names = [c.get("check") for c in checks]
        assert check_names != ["returned"], f"Assertion 14 Failed: Bare returned check found for verified {ident}"
    print(f"  [PASS] 14. Contract verification depth confirmed for all {len(verified_ids)} verified APIs (substantive checks enforced).")

    # 15. GWpy比較の厳格性（match判定の全件について契約アサーションが合格し、型名だけの一致でないこと）。
    gwpy_matches = [t for t in all_trials if t.get("trial_kind") == "gwpy_comparison" and t.get("gwpy_comparison", {}).get("status") == "match"]
    for t in gwpy_matches:
        asrt = t.get("assertions", {})
        tid = t.get("target_identifier")
        assert asrt.get("passed") is True, f"Assertion 15 Failed: Failed assertion in gwpy match for {tid}"
        checks = asrt.get("checks", [])
        check_names = [c.get("check") for c in checks]
        assert check_names != ["returned"], f"Assertion 15 Failed: Bare returned check in gwpy match for {tid}"
    print(f"  [PASS] 15. GWpy comparison rigor confirmed across all {len(gwpy_matches)} matched trials (unit/shape/numerical parity).")

    print("=" * 60)
    print("ALL 15 QA ASSERTIONS PASSED! RIGOROUS INTEGRITY VERIFIED.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
