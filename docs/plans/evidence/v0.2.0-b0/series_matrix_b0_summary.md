# SeriesMatrix B0 benchmark summary

Captured: 2026-08-25.  This is the compact tracked record for issue #676; the
raw JSON capture is intentionally not tracked.

- fixed SHA: `6a13900672900551ccaf1b18fe78b9ce6f062e29`
- raw capture SHA-256: `5fcdab552f8c910812335e81dfb4e0f170543f69ba78f18a63784191a80bf3b5`
- environment: Python 3.11.14, NumPy 1.26.4, Astropy 6.1.7, GWpy 4.0.1,
  GWexpy 0.1.14, Linux 6.17.0-1032-oem-x86_64
- protocol: 3 warm-ups, 7 independent child processes, at least 250 ms per
  measured batch, up to 3 attempts, and a 5% MAD/median stability threshold
- result: all six operations were stable; the stability gate was adoptable
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

## reproduction

Run the capture from this worktree, keeping its raw output outside version
control:

```bash
conda run -n gwexpy python scripts/benchmarks/series_matrix_benchmark.py \
  --capture-b0 --repo-root . --origin-ref origin/main \
  --output /tmp/series_matrix_b0_capture.json
sha256sum /tmp/series_matrix_b0_capture.json
```

The command creates and removes an exact temporary detached worktree under the
system temporary directory.  Verify the displayed SHA-256, retain raw data
only as an external artifact when needed, and do not add
`series_matrix_b0.json` or a B1 raw JSON file to the repository.
