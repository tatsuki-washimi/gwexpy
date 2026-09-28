# Candidate v7 disposition and independent review

This is an append-only disposition note. It does not edit the frozen raw captures, manifests, or earlier HOLD summaries.

## Independent review decision

The independent reviewer finds the selected no-window slowdown **unconfirmed under the frozen noise-aware gate**. The paired ABBA observations are mixed, and the evidence package is valid. Candidate v7 is accepted for integration on the structural and correctness gates. Do not claim a wall-time or PSS improvement.

The selected full-range warm medians were slower in two batches (wall: +10.44%, +12.62%; CPU: +10.29%, +12.97%), but neither batch exceeded `max(10%, 3 × summed relative MAD)`. The nearest-paired ABBA wall deltas contain 2/9 slower observations in batch 1 and 5/9 in batch 2, with paired medians −3.91% and +1.18%. This supports the reviewer's unconfirmed classification. The prior performance summary's HOLD wording is retained as historical; this review disposition supersedes that provisional HOLD.

The source contains an extra ordered timestamp cursor scan when no time bounds are supplied, before the full payload DataFrame query. This is a concrete source-level cost hypothesis, but its contribution was not isolated or separately measured. Do not treat it as a confirmed cause. No v8 runtime change is authorized by this note.

## Structural and correctness basis

- The frozen structure probe reports DataFrame payload materialization of 4096 rows for B0 and 512 requested rows for candidate v7 in all five samples. This counter is the payload DataFrame row count; it excludes SQLite validation cursor traversal and bounded TEXT preflight chunks. Full-source validation remains O(N).
- B1 public static-source parity is 20/20 across the frozen TEXT baseline v3/v4 cases, 12/12 frozen static fault cases, and the negative-zero bitwise supplement.
- WAL captures are stable in all 14 cases. Ten public differences are confined to the user-selected broad concurrent-WAL snapshot exception (selected/unselected payload, malformed values, timestamp, and schema under a committed concurrent write); four cases are equal, including both `usUnits` schedules. Record this scope for S2 disclosure and human scientific/data-model review.

## Arm-label clarification

The frozen harness uses arm label `B0` for the old-R baseline source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c` and wheel `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`. This is a harness label for the old-R/B1-source comparison arm; it does **not** mean the released GWexpy v0.2.4 B0 package. The candidate arm is candidate-v7 source `74cba639fb71234a1e7f326a597557488d39db5c`, wheel `692b8c48a7f8439b2a5d82cf0cb9c1e262711fbce7822c7ca590c230afab65c0`. Preserve the raw arm labels in captures; use these explicit source identities when describing results.

## Evidence and source commits

The append-only artifact indexes are:

- WAL semantic comparison: `2adc909c25f7b7882578947b850553575fea00bdf4ac0765f8ca4bcf1107e71e`
- Structural capture: `ace134bb35415524667a6a8c09c203773b889d67eb2b38d01a8ae71a0a97b19d`
- Timing and memory captures: `d92253ca26168ec55f5335e28f9ea4d29d39e4318a941b3670b8120b10de94ab`

The complete source/test history from the integration base is, in order:

1. `3d84a3cbbd91bf2d8ebf9444077a6378875ed707` — pin SDB validation and payload snapshot
2. `9ae724f1c47de345adad6486efa57f8cf43267b3` — push down exact SDB payload windows
3. `084f139fea9e564f3dc0eff1a2218f7c23f1438c` — batch SDB payload warning validation
4. `4ccd8648668b966119d97e3c677be422a2d0a3ce` — preserve exact SDB cadence median
5. `221d3fce7349b35b26b0a50614d42a5790a7ea5f` — combine bounded SDB validation and warning scan
6. `45435514c5565e36e65a16b879fbe393553a2c4b` — match fresh crop indexing and empty windows
7. `f61d8796e2367c8cbd1f7f596edda6eeadfd5f67` — preserve unit conversion parity in bounded reads
8. `9fa2ed5db68901b2d925484f2276537de2f26dd6` — bound canonical integer TEXT window scans
9. `74cba639fb71234a1e7f326a597557488d39db5c` — preserve negative-zero TEXT parity

Each source/test commit touches only `gwexpy/timeseries/io/sdb.py` and `tests/io/test_v025_sdb_snapshot.py`. Evidence-only commits remain separate and append-only.
