#!/usr/bin/env python3
"""Run all 9 workers in parallel using multiprocessing ProcessPoolExecutor."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BATCH_DIR = REPO / "deliverables" / "audit-c-coverage-c2898762" / "parallel-batch"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BATCH_DIR))

from run_worker import run_worker

WORKER_IDS = [f"worker-W{i}" for i in range(1, 10)]


def worker_task(worker_id: str) -> tuple[str, int, dict]:
    code = run_worker(worker_id)
    summary_path = BATCH_DIR / worker_id / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return worker_id, code, summary


def main() -> int:
    t0 = time.time()
    print("=" * 60)
    print("Starting 9-worker parallel execution for 508 APIs...")
    print(f"Workers: {', '.join(WORKER_IDS)}")
    print("=" * 60)

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(worker_task, wid): wid for wid in WORKER_IDS}
        for future in concurrent.futures.as_completed(futures):
            wid = futures[future]
            try:
                worker_id, code, summary = future.result()
                results[worker_id] = summary
                actual_calls = summary.get("execution_status", {}).get("actual_call", 0)
                print(f"--> [{worker_id}] Finished with exit code {code}: {actual_calls} actual calls, {summary.get('verified_api_count')} verified ({summary.get('duration_seconds')}s)")
            except Exception as e:
                print(f"--> [{wid}] CRASHED with exception: {e}", file=sys.stderr)
                results[wid] = {"error": str(e), "exit_code": 1}

    total_time = round(time.time() - t0, 2)
    print("=" * 60)
    print(f"All 9 workers finished in {total_time}s!")
    print("=" * 60)

    # Combine worker summaries
    total_assigned = sum(s.get("assigned_count", 0) for s in results.values())
    total_trials = sum(s.get("trials_total", 0) for s in results.values())

    total_actual_call = sum(s.get("execution_status", {}).get("actual_call", 0) for s in results.values())
    total_attr_access = sum(s.get("execution_status", {}).get("attribute_access", 0) for s in results.values())
    total_smoke_only = sum(s.get("execution_status", {}).get("smoke_only", 0) for s in results.values())
    total_contract_pending_exec = sum(s.get("execution_status", {}).get("contract_pending", 0) for s in results.values())
    total_env_blocked = sum(s.get("environment_blocked_count", 0) for s in results.values())

    total_verified = sum(s.get("verified_api_count", 0) for s in results.values())
    total_differ = sum(s.get("differ_api_count", 0) for s in results.values())
    total_contract_pending = sum(s.get("contract_pending_count", 0) for s in results.values())
    total_not_tested = sum(s.get("not_tested_count", 0) for s in results.values())

    gwpy_num_val_par = sum(s.get("gwpy_comparison", {}).get("numeric_value_parity", 0) for s in results.values())
    gwpy_meta_prop_par = sum(s.get("gwpy_comparison", {}).get("metadata_property_parity", 0) for s in results.values())
    gwpy_val_par = gwpy_num_val_par + gwpy_meta_prop_par
    gwpy_exc_par = sum(s.get("gwpy_comparison", {}).get("exception_parity", 0) for s in results.values())
    gwpy_diff = sum(s.get("gwpy_comparison", {}).get("differ", 0) for s in results.values())
    gwpy_err = sum(s.get("gwpy_comparison", {}).get("trial_error", 0) for s in results.values())
    gwpy_incomp = sum(s.get("gwpy_comparison", {}).get("incomparable", 0) for s in results.values())

    combined_summary = {
        "batch": "508_parallel_batch",
        "worker_count": len(WORKER_IDS),
        "total_assigned": total_assigned,
        "total_trials": total_trials,
        "execution_status": {
            "actual_call": total_actual_call,
            "attribute_access": total_attr_access,
            "smoke_only": total_smoke_only,
            "contract_pending": total_contract_pending_exec,
            "environment_blocked": total_env_blocked,
        },
        "reachability_status": {
            k: sum(s.get("reachability_status", {}).get(k, 0) for s in results.values())
            for k in ["reached_and_returned", "reached_with_observed_exception", "attribute_accessed", "smoke_signature_only", "reachability_unresolved", "fixture_setup_failed", "environment_blocked"]
        },
        "total_verified": total_verified,
        "total_differ": total_differ,
        "total_contract_pending": total_contract_pending,
        "total_environment_blocked": total_env_blocked,
        "total_not_tested": total_not_tested,
        "gwpy_comparison": {
            "numeric_value_parity": gwpy_num_val_par,
            "metadata_property_parity": gwpy_meta_prop_par,
            "value_parity": gwpy_val_par,
            "exception_parity": gwpy_exc_par,
            "differ": gwpy_diff,
            "trial_error": gwpy_err,
            "incomparable": gwpy_incomp,
        },
        "comparison_modes": {
            k: sum(s.get("comparison_modes", {}).get(k, 0) for s in results.values())
            for k in ["numeric_allclose", "collection_elementwise_numeric", "swig_lal_object_numeric", "metadata_equality", "exception_equality", "incomparable", "not_applicable"]
        },
        "duration_seconds": total_time,
        "worker_results": results,
    }

    out_file = BATCH_DIR / "parallel-workers-summary.json"
    out_file.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
