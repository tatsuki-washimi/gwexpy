# B1 mixed TEXT chunk-boundary characterization

This append-only supplement closes the 256-row mixed-storage coverage gap
before the v6 SDB runtime edit. It uses the same verified B1 baseline source
`1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, wheel
`473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`, and
installed `sdb.py` digest
`714a7320810755c7479e5c35dc5e0344ba513fc01989b89586d77a4d6618fa6f`.
The copied installation audit is `baseline-install-audit-B1.json`.

The fixture generator creates three 600-row `outTemp` tables with no type
affinity. Values cross cursor chunks at rows 255/256 and 511/512. The primary
case mixes canonical integer TEXT, SQLite REAL, and NULL in one unit-conversion
column while the requested crop spans rows 254–259. The invalid-token case
places `not-a-number` at row 256, outside a later requested crop. The newline
case selects a TEXT token with a trailing newline at row 255. All fixture
digests and SQLite storage-class counts are in `fixture-manifest-v4.json`.

## Frozen B1 public outcomes

- Mixed TEXT/REAL/NULL across the chunk boundary returns six float64 samples,
  no warning, and selected values
  `[23.333333333333332, 37.916666666666664, NaN, 25.0,
  25.555555555555554, 26.11111111111111]`.
- An out-of-window invalid TEXT token at row 256 emits one whole-source
  `UserWarning` and still returns the selected later four samples.
- A selected `61\n` token is accepted by B1 as numeric, emits no warning, and
  returns `16.11111111111111 deg_C` at that sample.

The public signatures, including exact selected time values and exception or
warning details, are in `b1-public-signatures-v4.json`. The custom public probe
SHA is `315cdeb462933fc2ed421e8bac8083bd854d19a1bc80b2978ce81458aea40a9b`; the
fixture-manifest SHA is
`4992086e5d267dcd52e0b00977d21cc5b65f3f8d92f4fdf78d1c6581b3f882f9`.

The v6 scanner should use `re.fullmatch` for the canonical integer token rule.
The trailing-newline fixture is numeric under B1, but remains unqualified and
must fall back before the scanner emits warnings. Likewise, if any token in a
chunk is unqualified, the scanner must leave warning emission to the B1
full-payload route; it must not emit a partial count before fallback.

Reproduce from this directory with:

```bash
python generate_fixtures.py
/tmp/gwexpy-v025-b-env-b/bin/python -I probe_b1.py fixtures b1-public-signatures-v4.json
```

Artifact digests are recorded in `artifact-sha256.json`.
