# Inherited API batch 1 — runtime verification

Pin: `c2898762ca68e20a1a241fffb3c3262b0a27290b` · Source commit: `03df898055bc7e269ffa803fbd4924bb07571cb7`

## Scope

10 planned inherited APIs (30 core trials + optional reference rows).

**Not merged** into `c-coverage-c2898762.jsonl` in this step.

## Aggregates

| Metric | Value |
| --- | ---: |
| Planned APIs | 10 |
| Executed unique APIs | 10 |
| Positive assertions passed | 10 / 10 |
| Abnormal paths observed | 7 / 10 |
| GWpy comparison | {"match": 9, "incomparable": 1} |

## Interpretation

- `override_unit` trials distinguish metadata unit assignment from physical value rescaling.
- `copy` trials record buffer sharing; no pre-judged correct deep-copy contract.
- Unified I/O trials confirm format registration before round-trip claims.

Artifacts: `c-inherited-batch1-runtime.jsonl`, `c-inherited-batch1-runtime-summary.json`.
