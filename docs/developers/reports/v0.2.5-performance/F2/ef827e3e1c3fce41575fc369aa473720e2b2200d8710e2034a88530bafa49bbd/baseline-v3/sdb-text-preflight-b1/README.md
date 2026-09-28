# B1 SDB TEXT preflight characterization

This package freezes B1 public behavior before the v6 SDB runtime edit. It
uses the B1 wheel in `/tmp/gwexpy-v025-b-env-b`, source SHA
`1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, wheel SHA
`473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`, and
installed `sdb.py` SHA
`714a7320810755c7479e5c35dc5e0344ba513fc01989b89586d77a4d6618fa6f`.
`probe_b1.py` checks the installed source digest before collecting outcomes.

Each fixture has three timestamped rows. The requested interval selects row 1;
rows 0 and 2 are outside the interval. `fixture-manifest.json` records every
fixture digest, SQLite storage class, input value, timestamp, and selected row.
`b1-public-signatures.json` records the public values, dtype, epoch, warnings,
or exception class/message. The result was collected with Python 3.12.12 and
NumPy 1.26.4 using the public SDB reader.

## Frozen behavior

| Case | B1 result relevant to range validation |
| --- | --- |
| Canonical integer TEXT (`60`, `61`, `62`) | No warning; selected value becomes `16.11111111111111 deg_C`. |
| Decimal or surrounding whitespace TEXT | B1 accepts both without a warning. These forms remain outside the v6 fast-path whitelist. |
| Mixed INTEGER/REAL/NULL storage | No warning; selected value becomes `16.38888888888889 deg_C`. |
| Invalid TEXT in selected or out-of-window row | One `UserWarning` for the whole source; the selected invalid value becomes NaN, while an out-of-window invalid value does not alter the selected sample. |
| Invalid out-of-window TEXT plus irregular timestamps | The payload `UserWarning` is emitted before `ValueError: SDB timestamp gap at index 1: expected cadence 301, got 300`. |
| TEXT at int64/uint64 edges | No warning; selected result is float64 `1.0248191152060862e+19`, reflecting B1's conversion path. |
| TEXT at the float64 integer boundary | `9007199254740991` returns exactly; `9007199254740993` becomes `9007199254740992.0` after B1's float64 conversion. |
| BLOB in selected or out-of-window row | One whole-source non-numeric `UserWarning`; selected BLOB becomes NaN. |
| Out-of-window `barometer=1e308` TEXT | `over=warn` returns the selected sample and emits `RuntimeWarning: overflow encountered in multiply`; `over=raise` raises `FloatingPointError` with that message. |
| Selected `barometer=1e308` TEXT | Same overflow warning/error, with `+Inf` as the selected result when warnings are enabled. |
| Out-of-window or selected `windSpeed=1e-320` TEXT | `under=warn` emits `RuntimeWarning: underflow encountered in multiply`; `under=raise` raises the corresponding `FloatingPointError`. Default NumPy underflow policy is ignored. |

## v6 fast-path qualification to evaluate

The B1 results support testing a deliberately narrow TEXT fast path for
canonical ASCII integer tokens only: `-?(0|[1-9][0-9]*)`, parsed as an exact
integer with absolute value strictly less than `2**53`. The fixture confirms
the largest admitted positive integer (`9007199254740991`) is returned exactly;
the immediately larger tested value takes a lossy float64 round trip. Do not
accept a leading plus, leading zeroes, whitespace, decimal point, exponent,
BLOB, or malformed text on this route. Existing SQLite INTEGER/REAL/NULL
handling remains separate. Run converted float64 values through selected unit
conversion in bounded chunks under `np.errstate(all="raise")`; any
floating-point event or unqualified token must use the B1 full-payload route so
B1 owns warnings, exceptions, and ordering. This is a characterization-based
design gate, not evidence that the candidate implementation is correct; the
candidate must still be compared against the frozen B1 signatures and WAL
matrix.

The canonical frozen 4096-row structural fixture stores `outTemp` as TEXT for
all rows, but its values are canonical small integers. Candidate v5 correctly
falls back on that fixture because its TEXT-unit guard rejects all TEXT. The
v5 structural raw is retained separately under `candidate-v5-f61d8796`; its
4096/4096 materialized rows are a HOLD result. The v6 work must use the same
frozen fixture and harness if the narrow whitelist is implemented.

## Reproduction

From this directory, recreate fixture files with:

```bash
python generate_fixtures.py
/tmp/gwexpy-v025-b-env-b/bin/python -I probe_b1.py fixtures
```

The probe is a one-time B1 correctness characterization, not a modification of
the frozen campaign harness. Artifact hashes are listed in
`artifact-sha256.json`.
