# C audit coverage ledger (521 items) — `c2898762ca68`

**Not product QA completion.** Static classification of 361 `pass2_not_evaluated` inherited APIs + 160 conventional module functions, plus 19 runtime probes (521-backlog conventional APIs).

## Reproduce

```bash
cd /path/to/gwexpy
PYTHONPATH=. python3 deliverables/audit-c-coverage-c2898762/c_coverage_build_ledger.py
PYTHONPATH=. python3 deliverables/audit-c-coverage-c2898762/c_coverage_runtime_probes.py
python3 deliverables/audit-c-coverage-c2898762/c_coverage_merge_runtime.py
PYTHONPATH=. python3 deliverables/audit-c-coverage-c2898762/c_coverage_build_inherited_priority_slice.py
PYTHONPATH=. python3 deliverables/audit-c-coverage-c2898762/c_inherited_batch1_runtime.py
```

## Outputs

| File | Role |
| --- | --- |
| `c-coverage-c2898762.jsonl` | 521-row ledger |
| `c-coverage-c2898762-summary.json` | Aggregates + probe summary |
| `c-coverage-runtime-probes.jsonl` | 19 runtime probes (13 executed + 6 deferred) |
| `c-inherited-priority-slice-c2898762.jsonl` | First 16-ID inherited API priority slice (planning only) |
| `c-inherited-priority-slice-c2898762-summary.json` | Slice aggregates |
| `c_coverage_build_ledger.py` | Static classifier |
| `c_coverage_runtime_probes.py` | Runtime probes |
| `c_coverage_merge_runtime.py` | Attach probe notes to ledger |
| `c_coverage_build_inherited_priority_slice.py` | Build inherited priority slice from ledger |
| `c-inherited-batch1-runtime.jsonl` | Inherited API batch 1 runtime trials (30 rows) |
| `c-inherited-batch1-runtime-summary.json` | Batch 1 aggregates |
| `2026-10-17-c-inherited-batch1-runtime.md` | Batch 1 short report |
| `c_inherited_batch1_runtime.py` | Execute batch 1 trials |
| `c-coverage-521-all-trials.jsonl` | Full trial records across all 521 APIs (1,572 trials) |
| `c-coverage-521-integrated-summary.json` | Integrated audit summary across all 521 APIs |
| `2026-10-17-c-coverage-521-integrated-report.md` | Integrated audit report for 521 APIs |
| `c-coverage-521-evidence-bundle.zip` | Standalone evidence bundle containing ledger, trials, and 9-worker runs |

## Inputs

`deliverables/audit-appendix-artifacts-20261017/{pass1-population.json,pass2-api-status.jsonl,pass2-module-functions.txt}`

Baseline pin: `c2898762ca68e20a1a241fffb3c3262b0a27290b`. Runtime uses checkout HEAD (recorded in summary).

---

## Parallel Execution & Audit Provenance Notes

1. **Commit Provenance (`source_commit`)**:
   - The 1,522 runtime trial records in `c-coverage-521-all-trials.jsonl` record `source_commit: f7265d425619...` because the parallel worker test harness was executed on the local working tree before creating the final wrapper commit `ffb3529bea79...`.
   - The public repository commit providing the complete fixed harness, comparator, ledger, and QA scripts is `ffb3529bea79c21e0bd8eed88b513ab40d53e2bd` (on branch `audit/c-coverage-508-parallel`).
   - Checking out `ffb3529bea79...` and running `python3 deliverables/audit-c-coverage-c2898762/parallel-batch/run_all_workers_parallel.py` reproduces the trial records.

2. **Issue C Reclassification Breakdown**:
   - From the earlier 16 APIs categorized as `collection_elementwise_numeric`, exactly 13 non-array attribute comparisons were reclassified to `metadata_equality` (`metadata_property_parity`):
     - `xspan`: 5
     - `span`: 3
     - `yspan`: 2
     - `band`: 1
     - `segments`: 1
     - `get_axes`: 1
     - Total: 13
   - Exactly 3 true collection APIs remain in `collection_elementwise_numeric`:
     - `TimeSeriesDict.copy`
     - `TimeSeriesList.copy`
     - `TimeSeriesList.coalesce`

3. **Scope of `verified` (246 APIs)**:
   - The 246 `verified` APIs represent APIs that passed concrete contract validation checks configured in the test harness (such as finite-value checks, unit consistency, shape/dimension integrity, and return-type contracts).
   - This status certifies contract compliance against configured assertions, distinct from full end-to-end scientific or physical correctness across all possible input domains (which remains subject to ongoing domain reviews).

4. **Distinction between `disposition=differ` (44) and GWpy Parity `differ=0`**:
   - `disposition=differ` (44 items): Discrepancies observed during internal GWexpy contract validation (e.g. methods inherited from 1D series that fail on 2D arrays, or unimplemented optional capabilities).
   - GWpy Parity `differ=0` (0 items): Unintended divergence between GWexpy and GWpy on identical inputs has been completely eliminated (false positives resolved via `visual_artist_data` / `visual_comparison_pending` and explicit SWIG/PyCBC comparators).

