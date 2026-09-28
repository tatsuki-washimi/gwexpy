# Candidate v7 negative-zero wheel parity

This is an append-only follow-up to the candidate-discovered mismatch captured
under `candidate-v6-initial-9fa2ed5db/negative-zero-review/`. It records the
exact corrected source wheel and reruns the same public probe against frozen B1
and the installed candidate v7 wheel.

For mixed `outHumidity` storage `('-0', 1.5, '2')`, bounded to row 0, B1 and
candidate v7 both return float64 `-0.0`, raw bytes
`0000000000000080`, sign bit true, and no warnings. The candidate fallback is
verified by the committed regression test, which also checks the full-payload
query shape.

## Exact candidate install

See `install-audit.json` for source commit, source/wheel/installed module
hashes, installation location, file count, Python version, and package version.
The v7 wheel SHA-256 is `692b8c48a7f8439b2a5d82cf0cb9c1e262711fbce7822c7ca590c230afab65c0`.

## Artifacts

- `probe.py`: reproduces the public B1/candidate read and captures raw float64 bytes.
- `negative-zero-mixed.sdb`: fixture used for the recorded run.
- `b1-public.json` and `candidate-v7-public.json`: raw public results.
- `install-audit.json`: exact wheel installation audit.
