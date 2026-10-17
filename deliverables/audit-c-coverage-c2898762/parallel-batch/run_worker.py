#!/usr/bin/env python3
"""Execute all assigned APIs for a specific worker and produce deliverables."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BATCH_DIR = REPO / "deliverables" / "audit-c-coverage-c2898762" / "parallel-batch"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BATCH_DIR))

try:
    from .worker_dispatch import execute_api_trials
except ImportError:
    from worker_dispatch import execute_api_trials


def run_worker(worker_id: str) -> int:
    t0 = time.time()
    folder_name = worker_id if worker_id.startswith("worker-") else f"worker-{worker_id}"
    worker_dir = BATCH_DIR / folder_name
    if not worker_dir.exists():
        print(f"Error: {worker_dir} does not exist", file=sys.stderr)
        return 1

    assigned_file = worker_dir / "assigned-identifiers.jsonl"
    assigned_rows = [json.loads(line) for line in assigned_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[{worker_id}] Starting execution of {len(assigned_rows)} assigned APIs...")

    all_trials: list[dict[str, Any]] = []
    all_dispositions: list[dict[str, Any]] = []

    for idx, rec in enumerate(assigned_rows, 1):
        ident = rec["identifier"]
        try:
            trials, disposition = execute_api_trials(worker_id, rec)
        except Exception as e:
            # Fallback error record using run_trial
            try:
                from .worker_engine import run_trial
            except ImportError:
                from worker_engine import run_trial
            trial = run_trial(
                worker_id=worker_id,
                target_identifier=ident,
                trial_kind="positive_control",
                input_description=f"Fallback execution of {ident}",
                prov_obj=None,
                fn=lambda: None,
                execution_status="environment_blocked",
                not_tested_reason=f"Unhandled error during dispatch: {e}",
            )
            trials = [trial]
            disposition = {
                "target_identifier": ident,
                "worker_id": worker_id,
                "execution_status": "environment_blocked",
                "verification_status": "environment_blocked",
                "disposition": "environment_blocked",
                "gwpy_comparison": "not_applicable",
                "trials_count": 1,
                "positive_control_status": "failed",
                "abnormal_input_status": "not_run",
                "gwpy_comparison_status": "not_applicable",
                "not_tested_reason": f"Unhandled error during dispatch: {e}",
                "coverage_scope": "direct",
                "notes": f"Unhandled error: {e}",
            }
        all_trials.extend(trials)
        all_dispositions.append(disposition)

    # Write runtime-trials.jsonl
    trials_file = worker_dir / "runtime-trials.jsonl"
    trials_file.write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in all_trials),
        encoding="utf-8",
    )

    # Write api-dispositions.jsonl
    disp_file = worker_dir / "api-dispositions.jsonl"
    disp_file.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in all_dispositions),
        encoding="utf-8",
    )

    # Compute summary metrics
    duration_s = round(time.time() - t0, 2)
    actual_call_count = sum(1 for d in all_dispositions if d["execution_status"] == "actual_call")
    attribute_access_count = sum(1 for d in all_dispositions if d["execution_status"] == "attribute_access")
    smoke_only_count = sum(1 for d in all_dispositions if d["execution_status"] == "smoke_only")
    env_blocked_count = sum(1 for d in all_dispositions if d["execution_status"] == "environment_blocked")
    executed_count = actual_call_count + attribute_access_count + smoke_only_count

    verified_count = sum(1 for d in all_dispositions if d["verification_status"] == "verified")
    differ_count = sum(1 for d in all_dispositions if d["verification_status"] == "differ")
    contract_pending_count = sum(1 for d in all_dispositions if d["verification_status"] == "contract_pending")
    not_tested_count = sum(1 for d in all_dispositions if d["verification_status"] == "not_run")

    gwpy_value_parity = sum(1 for d in all_dispositions if d.get("gwpy_comparison") == "value_parity")
    gwpy_exception_parity = sum(1 for d in all_dispositions if d.get("gwpy_comparison") == "exception_parity")
    gwpy_differ = sum(1 for d in all_dispositions if d.get("gwpy_comparison") == "differ")
    gwpy_trial_error = sum(1 for d in all_dispositions if d.get("gwpy_comparison") == "trial_error")
    gwpy_incomparable = sum(1 for d in all_dispositions if d.get("gwpy_comparison") in ("incomparable", "not_applicable"))

    summary = {
        "worker_id": worker_id,
        "assigned_count": len(assigned_rows),
        "trials_total": len(all_trials),
        "execution_status": {
            "actual_call": actual_call_count,
            "attribute_access": attribute_access_count,
            "smoke_only": smoke_only_count,
            "environment_blocked": env_blocked_count,
        },
        "verified_api_count": verified_count,
        "differ_api_count": differ_count,
        "contract_pending_count": contract_pending_count,
        "environment_blocked_count": env_blocked_count,
        "not_tested_count": not_tested_count,
        "gwpy_comparison": {
            "value_parity": gwpy_value_parity,
            "exception_parity": gwpy_exception_parity,
            "differ": gwpy_differ,
            "trial_error": gwpy_trial_error,
            "incomparable": gwpy_incomparable,
        },
        "duration_seconds": duration_s,
    }

    summary_file = worker_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Generate reproduce.py
    reproduce_code = f"""#!/usr/bin/env python3
\"\"\"Reproduce script for {worker_id} ({len(assigned_rows)} APIs).\"\"\"

import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
BATCH_DIR = REPO / "deliverables" / "audit-c-coverage-c2898762" / "parallel-batch"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BATCH_DIR))

from run_worker import run_worker

if __name__ == "__main__":
    raise SystemExit(run_worker("{worker_id}"))
"""
    reproduce_file = worker_dir / "reproduce.py"
    reproduce_file.write_text(reproduce_code, encoding="utf-8")
    reproduce_file.chmod(0o755)

    # Generate README.md
    readme_content = f"""# Worker {worker_id} Audit Deliverable

- **Assigned Count**: {len(assigned_rows)} APIs
- **Trials Run**: {len(all_trials)}
- **Actual Calls**: {actual_call_count}
- **Attribute Access / Weak**: {attribute_access_count}
- **Smoke Only**: {smoke_only_count}
- **Environment Blocked**: {env_blocked_count}
- **Verified APIs**: {verified_count}
- **Differ APIs**: {differ_count}
- **Contract Pending APIs**: {contract_pending_count}
- **GWpy Value Parity**: {gwpy_value_parity}
- **GWpy Exception Parity**: {gwpy_exception_parity}
- **GWpy Differs**: {gwpy_differ}
- **Incomparable / Extension**: {gwpy_incomparable}
- **Execution Duration**: {duration_s}s

## Deliverable Files
- `assigned-identifiers.jsonl`: Original input manifest ({len(assigned_rows)} lines)
- `runtime-trials.jsonl`: Machine-readable trial evidence ({len(all_trials)} lines)
- `api-dispositions.jsonl`: Final status and dispositions per API ({len(all_dispositions)} lines)
- `summary.json`: Aggregated metrics
- `reproduce.py`: Standalone execution script
- `README.md`: This report
"""
    readme_file = worker_dir / "README.md"
    readme_file.write_text(readme_content, encoding="utf-8")

    print(f"[{worker_id}] Done: {executed_count}/{len(assigned_rows)} executed, {verified_count} verified, {len(all_trials)} trials ({duration_s}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run specific worker batch")
    parser.add_argument("worker_id", help="Worker ID (e.g. worker-W1)")
    args = parser.parse_args()
    return run_worker(args.worker_id)


if __name__ == "__main__":
    raise SystemExit(main())
