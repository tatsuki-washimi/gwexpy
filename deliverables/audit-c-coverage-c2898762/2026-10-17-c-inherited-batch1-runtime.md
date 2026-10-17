# Inherited API batch 1 — runtime verification

Pin: `c2898762ca68e20a1a241fffb3c3262b0a27290b` · Source commit: `f1a2ff845df2927e73423c9323130b8fd85e1983`

## Scope

10 planned inherited APIs (30 trials: positive / abnormal / GWpy compare each).

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
