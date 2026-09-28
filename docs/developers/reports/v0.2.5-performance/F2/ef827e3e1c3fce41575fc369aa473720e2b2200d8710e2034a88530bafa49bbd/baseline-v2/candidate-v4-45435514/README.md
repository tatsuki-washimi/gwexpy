# SDB candidate v4 partial evidence (HOLD)

This append-only package records candidate source commit `45435514c5565e36e65a16b879fbe393553a2c4b`. Static-source correctness completed 12/12 cases. WAL public correctness completed 11/14 cases; capture stopped after independent review found a unit-conversion parity defect. The remaining three cases were not run. Structure, warm timing, cold, and PSS were not run for v4.

`review-findings/public-red-results.json` records fixed-wheel B1/v4 public outcomes for the unit-overflow, `dx != source_dt`, and SQLite exclusive-upper int64 findings. It also records a separate underflow finding from independent review that still needs characterization. Reproduction scripts and the three SDB-only fixtures are stored alongside it.

The candidate is HOLD. No v4 structural or performance claim is made. Existing v2/v3 evidence and manifests are untouched.
