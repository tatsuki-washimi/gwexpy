# C audit coverage ledger (521 items) — `c2898762ca68`

**Not product QA completion.** Static classification of 361 `pass2_not_evaluated` inherited APIs + 160 conventional module functions, plus 16 limited runtime probes.

## Reproduce

```bash
cd /path/to/gwexpy
PYTHONPATH=. python3 deliverables/audit-c-coverage-c2898762/c_coverage_build_ledger.py
PYTHONPATH=. python3 deliverables/audit-c-coverage-c2898762/c_coverage_runtime_probes.py
python3 deliverables/audit-c-coverage-c2898762/c_coverage_merge_runtime.py
```

## Outputs

| File | Role |
| --- | --- |
| `c-coverage-c2898762.jsonl` | 521-row ledger |
| `c-coverage-c2898762-summary.json` | Aggregates + probe summary |
| `c-coverage-runtime-probes.jsonl` | 16 probe rows (10 executed, 6 deferred) |
| `c_coverage_build_ledger.py` | Static classifier |
| `c_coverage_runtime_probes.py` | Runtime probes |
| `c_coverage_merge_runtime.py` | Attach probe notes to ledger |

## Inputs

`deliverables/audit-appendix-artifacts-20261017/{pass1-population.json,pass2-api-status.jsonl,pass2-module-functions.txt}`

Baseline pin: `c2898762ca68e20a1a241fffb3c3262b0a27290b`. Runtime uses checkout HEAD (recorded in summary).
