# F3 small-input regression diagnostic

This append-only diagnostic used candidate source
`96f6c72f11ea318e8fc9f1ccddc32e29fc300d05` and wheel SHA-256
`22ed237cbaa158bae08267d158eb9bf91e8bb0fd0ea835ef39447ed1efb690c0`.
It used the same frozen harness and controller as `candidate-v1`, but the
frozen small fixture (`many_valid.xml`, 100,838 bytes). Each independent warm
batch contains nine samples per arm in `ABBA · BAAB · ABBA · BAAB · AB` order.
Raw samples, median/MAD, audit identity, and artifact hashes are retained in
`batch-1` and `batch-2`.

| Batch | B1 wall median (MAD) | Candidate wall median (MAD) | Regression | Noise-aware limit |
| --- | ---: | ---: | ---: | ---: |
| 1 | 718,187 (58,280) ns | 1,022,075 (52,975) ns | 42.31% | 39.89% |
| 2 | 754,561 (26,380) ns | 1,119,018 (76,687) ns | 48.30% | 31.05% |

Both batches exceed `max(10%, 3*(MAD_B1/median_B1 + MAD_candidate/median_candidate))`.
This candidate fails the small-input non-regression gate. The large-input
primary PSS result in `candidate-v1` remains a diagnostic for this source;
it is not sufficient for integration. A revised source and wheel need fresh
small-input and large-input release measurements.

`manifest.json` records the exact sample hashes and both gate calculations.
