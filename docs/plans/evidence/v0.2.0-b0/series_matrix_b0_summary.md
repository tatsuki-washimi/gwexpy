# SeriesMatrix B0 benchmark summary

Captured: 2026-08-25.  This is the compact tracked record for issue #676; the
raw JSON capture is intentionally not tracked.  The 318-cell B0 contract
candidate associated with the earlier review was rejected by physics/data-model
review.  The later 390-cell candidate was also superseded after final review
found incomplete nested metadata/attrs and exponent-boundary coverage.  The
subsequent 453-cell candidate was superseded after review found missing
supported SpectrogramMatrix slice and axis/epoch independence coverage.  The
later 454-cell candidate was superseded after review found incomplete
equivalent-selector and 3-D batch-slice coverage.  The current executable B0
ledger's later 460-cell candidate was superseded after review found missing
scalar Spectrogram epoch and negative integer sample-index coverage.  The
472-cell candidate was superseded after review required scalar extraction to
reject conflicting matrix-epoch and explicit-time authorities and to preserve
the complete selected-cell `MetaData` payload. The 474-cell candidate was
superseded after review found that scalar extraction overwrote a user value at
the reserved metadata carrier. The current executable B0 ledger contains
exactly 478 cells. The 478-cell candidate was superseded after review found
that its reserved-carrier collision refusals omitted two supported 4-D negative
row/column selectors. The current executable B0 ledger contains exactly 480
cells; this historical benchmark capture does not validate the semantics of
any rejected candidate or measure the later contract fixes.

- fixed SHA: `6a13900672900551ccaf1b18fe78b9ce6f062e29`
- recorded raw capture SHA-256: `5fcdab552f8c910812335e81dfb4e0f170543f69ba78f18a63784191a80bf3b5`
- environment: Python 3.11.14, NumPy 1.26.4, Astropy 6.1.7, GWpy 4.0.1,
  GWexpy 0.1.14, Linux 6.17.0-1032-oem-x86_64
- protocol: 3 warm-ups, 7 independent child processes, at least 250 ms per
  measured batch, up to 3 attempts, and a 5% MAD/median stability threshold
- result: all six operations were stable; the recorded timing stability gate
  was adoptable.  RSS is not an adoption gate; the current runner records
  absolute child-process peak RSS only as a non-gating diagnostic.
- approval: this B0 data-model contract requires explicit human D21/data-model
  sign-off before merge or release; AI review is advisory and cannot provide it

| Operation | Median seconds | MAD seconds |
| --- | ---: | ---: |
| `asarray` | 2.066594178784892e-7 | 1.086915376514269e-9 |
| `construct` | 1.4639424601041264e-4 | 4.061744675253455e-6 |
| `copy` | 2.0250991489544873e-4 | 1.214538253193839e-6 |
| `multiply` | 3.8455615102895885e-4 | 9.653476611376597e-6 |
| `quantity_left_multiply` | 4.6380028833408615e-4 | 1.0054703968747345e-5 |
| `slice` | 4.827991784892895e-5 | 2.3718520064808647e-6 |

## protocol reproduction

Run the protocol from the immutable recorded SHA, keeping its raw output
outside version control:

```bash
conda run -n gwexpy python scripts/benchmarks/series_matrix_benchmark.py \
  --capture-b0 --repo-root . --origin-ref 6a13900672900551ccaf1b18fe78b9ce6f062e29 \
  --output /tmp/series_matrix_b0_capture.json
sha256sum /tmp/series_matrix_b0_capture.json
```

The command creates and removes an exact temporary detached worktree under the
system temporary directory.  It reproduces the protocol, not the prior raw
capture or its digest: the raw capture was deleted, so its recorded SHA-256
cannot be independently verified.  Before a future adoption review, the raw
artifact must be retained and made available for designated reviewer inspection.
Do not add `series_matrix_b0.json` or a B1 raw JSON file to the repository.
