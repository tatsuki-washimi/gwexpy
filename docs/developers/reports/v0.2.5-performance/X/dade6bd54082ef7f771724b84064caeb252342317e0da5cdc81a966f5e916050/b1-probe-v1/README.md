# B-X B1 wheel probe (diagnostic only)

This is a single-read characterization of the frozen old-R/B1 wheel against the prepared B-X fixtures. It validates the structural spy and captures warning/error behavior for future TDD. It is **not B-X `BASELINE_FROZEN`**, a wall/CPU/PSS sample, or a performance claim. The final integrated pre-X source SHA is not yet fixed.

The installed B1 source is `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, wheel SHA256 `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`. The fixture manifest SHA256 is `dade6bd54082ef7f771724b84064caeb252342317e0da5cdc81a966f5e916050`; each worker verified its selected file SHA before reading. The exact harness hash, dependency versions, raw file hashes, and results appear in `manifest.json` and the raw JSON files.

| Case | B1 outcome | Structural full-payload `astype` calls |
| --- | --- | ---: |
| ATS int32, 4,194,304 samples | float64 return | 1 |
| ATS int64, 4,194,304 samples | float64 return | 1 |
| Homogeneous NetCDF, 4×4×262,144 | float64 return | 32 (2 per cell) |
| ATS overflow, `seterr=warn` | float64 return; `overflow encountered in multiply` RuntimeWarning | — |
| ATS overflow, `seterr=raise` | `FloatingPointError` with the same message | — |
| NetCDF NaN/±Inf | float64 return | — |
| NetCDF int64 extrema | int64 return | — |
| NetCDF object strings | `TypeError` from `isnan` | — |

The NetCDF reads also emitted the B1 environment's GMT loading and NumPy binary-compatibility RuntimeWarnings. The JSON captures their exact category, message, and order. All captured process stderr files are empty. The structure counter observes only `ndarray.astype` C-calls in the named ATS scale or NetCDF matrix `lossless` function. It does not count all NumPy allocations, nor does it estimate PSS.
