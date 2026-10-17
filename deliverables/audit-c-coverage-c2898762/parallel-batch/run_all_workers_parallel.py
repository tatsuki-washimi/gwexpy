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
                print(f"--> [{worker_id}] Finished with exit code {code}: {summary.get('executed_api_count')}/{summary.get('assigned_count')} executed, {summary.get('verified_api_count')} verified ({summary.get('duration_seconds')}s)")
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
    total_executed = sum(s.get("executed_api_count", 0) for s in results.values())
    total_verified = sum(s.get("verified_api_count", 0) for s in results.values())
    total_differ = sum(s.get("differ_api_count", 0) for s in results.values())
    total_env_blocked = sum(s.get("environment_blocked_count", 0) for s in results.values())
    total_contract_pending = sum(s.get("contract_pending_count", 0) for s in results.values())
    total_not_tested = sum(s.get("not_tested_count", 0) for s in results.values())

    combined_summary = {
        "batch": "508_parallel_batch",
        "worker_count": len(WORKER_IDS),
        "total_assigned": total_assigned,
        "total_trials": total_trials,
        "total_executed": total_executed,
        "total_verified": total_verified,
        "total_differ": total_differ,
        "total_environment_blocked": total_env_blocked,
        "total_contract_pending": total_contract_pending,
        "total_not_tested": total_not_tested,
        "duration_seconds": total_time,
        "worker_results": results,
    }

    out_file = BATCH_DIR / "parallel-workers-summary.json"
    out_file.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
