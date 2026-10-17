#!/usr/bin/env python3
"""Phase 1: Parent Agent setup and manifest partitioning for 9 parallel workers."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = REPO_ROOT / "deliverables" / "audit-c-coverage-c2898762"
BATCH_DIR = AUDIT_DIR / "parallel-batch"


def main() -> int:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    manifest_dir = BATCH_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = AUDIT_DIR / "c-coverage-c2898762.jsonl"
    batch1_path = AUDIT_DIR / "c-inherited-batch1-runtime.jsonl"

    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ledger) == 521, f"Expected 521 rows in ledger, got {len(ledger)}"

    batch1 = [json.loads(line) for line in batch1_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    batch1_ids = set(r["target_identifier"] for r in batch1)
    ledger_executed = set(r["identifier"] for r in ledger if r.get("runtime_status") == "executed_this_pass")
    existing_13 = batch1_ids | ledger_executed
    assert len(existing_13) == 13, f"Expected 13 existing evidence items, got {len(existing_13)}"

    remaining_508 = [r for r in ledger if r["identifier"] not in existing_13]
    assert len(remaining_508) == 508, f"Expected 508 remaining items, got {len(remaining_508)}"

    # 1. Types inherited APIs: 121 -> W1 (61) + W2 (60)
    types_rows = sorted(
        [r for r in remaining_508 if r.get("population") == "inherited_api_361" and r["identifier"].split(".")[1] == "types"],
        key=lambda r: r["identifier"],
    )
    assert len(types_rows) == 121, f"Expected 121 types rows, got {len(types_rows)}"
    w1_rows = types_rows[:61]
    w2_rows = types_rows[61:]
    assert len(w1_rows) == 61
    assert len(w2_rows) == 60

    # 2. Timeseries inherited APIs: 75 -> W3
    w3_rows = sorted(
        [r for r in remaining_508 if r.get("population") == "inherited_api_361" and r["identifier"].split(".")[1] == "timeseries"],
        key=lambda r: r["identifier"],
    )
    assert len(w3_rows) == 75, f"Expected 75 timeseries rows, got {len(w3_rows)}"

    # 3. FrequencySeries inherited APIs: 72 -> W4
    w4_rows = sorted(
        [r for r in remaining_508 if r.get("population") == "inherited_api_361" and r["identifier"].split(".")[1] == "frequencyseries"],
        key=lambda r: r["identifier"],
    )
    assert len(w4_rows) == 72, f"Expected 72 frequencyseries rows, got {len(w4_rows)}"

    # 4. Spectrogram inherited APIs: 49 -> W5
    w5_rows = sorted(
        [r for r in remaining_508 if r.get("population") == "inherited_api_361" and r["identifier"].split(".")[1] == "spectrogram"],
        key=lambda r: r["identifier"],
    )
    assert len(w5_rows) == 49, f"Expected 49 spectrogram rows, got {len(w5_rows)}"

    # 5. Fields (16) + Plot (18) + GUI internal (9): 43 -> W6
    fields_rows = [r for r in remaining_508 if r.get("population") == "inherited_api_361" and r["identifier"].split(".")[1] == "fields"]
    plot_rows = [r for r in remaining_508 if r.get("population") == "inherited_api_361" and r["identifier"].split(".")[1] == "plot"]
    gui_rows = [r for r in remaining_508 if r.get("api_classification") == "module_internal_helper"]
    assert len(fields_rows) == 16, f"Expected 16 fields, got {len(fields_rows)}"
    assert len(plot_rows) == 18, f"Expected 18 plot, got {len(plot_rows)}"
    assert len(gui_rows) == 9, f"Expected 9 gui, got {len(gui_rows)}"
    w6_rows = sorted(fields_rows + plot_rows + gui_rows, key=lambda r: r["identifier"])
    assert len(w6_rows) == 43, f"Expected 43 w6 rows, got {len(w6_rows)}"

    # 6. Conventional I/O: 83 -> W7 (42) + W8 (41)
    io_rows = sorted(
        [r for r in remaining_508 if r.get("api_classification") == "conventional_io_helper"],
        key=lambda r: r["identifier"],
    )
    assert len(io_rows) == 83, f"Expected 83 io rows, got {len(io_rows)}"
    w7_rows = io_rows[:42]
    w8_rows = io_rows[42:]
    assert len(w7_rows) == 42
    assert len(w8_rows) == 41

    # 7. General module functions (55) + fitting (10): 65 -> W9
    gen_func_rows = [r for r in remaining_508 if r.get("api_classification") == "conventional_module_function"]
    fit_rows = [r for r in remaining_508 if r.get("api_classification") == "conventional_fitting_helper"]
    assert len(gen_func_rows) == 55, f"Expected 55 gen_func, got {len(gen_func_rows)}"
    assert len(fit_rows) == 10, f"Expected 10 fitting, got {len(fit_rows)}"
    w9_rows = sorted(gen_func_rows + fit_rows, key=lambda r: r["identifier"])
    assert len(w9_rows) == 65, f"Expected 65 w9 rows, got {len(w9_rows)}"

    worker_map = {
        "worker-W1": w1_rows,
        "worker-W2": w2_rows,
        "worker-W3": w3_rows,
        "worker-W4": w4_rows,
        "worker-W5": w5_rows,
        "worker-W6": w6_rows,
        "worker-W7": w7_rows,
        "worker-W8": w8_rows,
        "worker-W9": w9_rows,
    }

    # Verify partition properties
    all_assigned_ids: list[str] = []
    seen_ids: set[str] = set()
    for worker_id, rows in worker_map.items():
        worker_ids = [r["identifier"] for r in rows]
        # Check internal uniqueness
        assert len(worker_ids) == len(set(worker_ids)), f"Duplicate IDs inside {worker_id}"
        # Check disjointness with prior workers
        intersection = seen_ids.intersection(worker_ids)
        assert not intersection, f"Overlap detected between workers: {intersection}"
        seen_ids.update(worker_ids)
        all_assigned_ids.extend(worker_ids)

    # Check union matches remaining_508
    remaining_508_ids = set(r["identifier"] for r in remaining_508)
    assert seen_ids == remaining_508_ids, "Union of assigned IDs does not match remaining 508!"
    assert len(all_assigned_ids) == 508, f"Total assigned count {len(all_assigned_ids)} != 508"

    # Write manifests and initialize worker directories
    partition_manifest = {
        "total_population": len(ledger),
        "existing_evidence_count": len(existing_13),
        "remaining_count": len(remaining_508),
        "workers": {},
    }

    for worker_id, rows in worker_map.items():
        worker_dir = BATCH_DIR / worker_id
        worker_dir.mkdir(parents=True, exist_ok=True)

        # Write to manifests/
        manifest_file = manifest_dir / f"{worker_id}-assigned.jsonl"
        manifest_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

        # Write to worker directory
        worker_assigned_file = worker_dir / "assigned-identifiers.jsonl"
        worker_assigned_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

        partition_manifest["workers"][worker_id] = {
            "assigned_count": len(rows),
            "manifest_file": str(manifest_file.relative_to(REPO_ROOT)),
            "worker_assigned_file": str(worker_assigned_file.relative_to(REPO_ROOT)),
            "first_identifier": rows[0]["identifier"],
            "last_identifier": rows[-1]["identifier"],
        }
        print(f"[{worker_id}] Initialized with {len(rows)} assigned identifiers.")

    summary_file = BATCH_DIR / "partition-manifest.json"
    summary_file.write_text(json.dumps(partition_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nPhase 1 completed successfully! All 508 IDs partitioned and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
