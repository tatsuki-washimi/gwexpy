# SDB snapshot candidate v2 evidence (`4ccd8648`)

**Status: HOLD.** This append-only package does not alter the prior candidate evidence. Candidate v2 fixes the INT64 timestamp median mismatch, but independent low-noise window batches confirm a warm regression. Do not integrate this source candidate.

## Provenance

- Source commit `4ccd8648668b966119d97e3c677be422a2d0a3ce`; `sdb.py` SHA-256 `565e5756e20a02c0e2e6255fd41aadbd3cb14d413687755f8e263012c4b6f83c`; test SHA-256 `31930b0aede259281863e9c7e8ff31e38b375b962c868a318c6630d164cdcb5a`.
- Candidate wheel SHA-256 `a71b1a48945d02e094d4068b4c752b812c5164b835c73cb6a893769a069e2fa1`; Python/build/setuptools/wheel `3.12.12/1.3.0/80.9.0/0.45.1`, `wheel-no-deps` install.
- B1 source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, wheel SHA-256 `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- WAL harness freeze `02cb6b415a6c3c2cbad0a4ce82e506037409e360`, digest `ef827e3e1c3fce41575fc369aa473720e2b2200d8710e2034a88530bafa49bbd`.
- Structural/timing harness freeze `6f0a3d4ea`, digest `e8422a63694d54ba4c8d349f8fb2da7f5df3cda63387d2e036c1af4cba338c72`.

## INT64 cadence parity

Fixture rows are SQLite int64 `[-2^63, 1, 2]`; exact deltas are `9223372036854775809` and `1`. B1 and the installed candidate-v2 wheel both raise `ValueError: SDB timestamp gap at index 2: expected cadence 9223372036854775809, got 1`, with identical captured warning lists. Candidate-084f139 raised at index 1 using rounded cadence `9223372036854775808`. The fixture, wheel results, and prior candidate result are retained in `extreme-timestamp/`.

## WAL matrix

Each route has five samples per arm. All 14 current B1 captures match frozen baseline-v2 B1 raw-B results; candidate-v2 captures are stable. Four public outcomes match and ten differ under the approved broad concurrent snapshot exception. Exact warnings, errors, fingerprints, trigger events, and raw hashes are recorded in `wal-comparison.json` and the raw folders.

| Public parity | Cases |
|---|---|
| Equal | `unselected_value/selected`, `unselected_malformed/selected`, `usunits/selected`, `usunits/all` |
| Different | `selected_value/selected`, `selected_value/all`, `unselected_value/all`, `selected_malformed/selected`, `selected_malformed/all`, `unselected_malformed/all`, `timestamp/selected`, `timestamp/all`, `schema/selected`, `schema/all` |

## Structure and warm timing

The structural probe shows the payload DataFrame materialization reduced from 4096 rows to 512 rows. This counter excludes SQLite timestamp validation and warning scans, both still whole-source O(N).

Regression threshold is `max(10%, 3 × (MAD_A/median_A + MAD_B/median_B))` per independent five-sample batch.

| Window batch | B1 wall median (MAD) | Candidate wall median (MAD) | Change | Threshold | Gate |
|---|---:|---:|---:|---:|---|
| `f2_sdb_window` | 8.395 ms (0.035) | 10.349 ms (0.129) | +23.3% | 10.0% | exceeded |
| `f2_sdb_window-batch2` | 10.274 ms (1.079) | 12.142 ms (0.883) | +18.2% | 53.3% | inconclusive/within |
| `f2_sdb_window-batch3` | 8.420 ms (0.081) | 10.326 ms (0.044) | +22.6% | 10.0% | exceeded |

Window batches 1 and 3 exceed the gate at low noise (+23.3% and +22.6%; each has a 10.0% threshold). Batch 2 is inconclusive (+18.2%, with a 53.3% threshold due to high noise). Two independent low-noise batches confirm an approximately 23% warm window regression. Candidate v2 remains HOLD. Selected/all summaries and all raw samples are retained.

## Verification and omitted captures

- Focused tests: 72 passed. Ruff check/format and MyPy passed.
- The INT64 public result matches B1 on the installed candidate wheel.
- No complete cold/PSS matrix was collected after the warm HOLD gate was confirmed. A selected cold route completed before batch 3 but is excluded because matching all/window captures were not taken. No wall-time or PSS improvement is claimed.

Full provenance and metrics are in `evidence-manifest.json`; the 14-case signatures and exact warning/error differences are in `wal-comparison.json`. `artifact-sha256.json` hashes every retained capture artifact.
