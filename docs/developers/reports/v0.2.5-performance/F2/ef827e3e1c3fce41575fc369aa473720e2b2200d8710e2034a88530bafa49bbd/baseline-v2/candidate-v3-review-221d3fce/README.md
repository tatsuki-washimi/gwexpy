# Candidate v3 crop/non-finite review reproduction

This append-only package records read-only public behavior comparisons between the frozen B1 wheel and candidate-v3 wheel. The oracle performs a fresh full SDB read and applies the public time-selection crop before materializing `.times`; it verifies `_xindex` is absent at that point.

`comparison-summary.json` records the per-case output, axis, t0, warning, and SQL materialization differences. `raw/` preserves both wheel captures, and `characterize.py` is the exact capture script. Candidate v3 remains HOLD; these findings motivate candidate v4.
