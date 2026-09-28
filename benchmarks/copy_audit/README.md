# B-X/#586 copy-audit preparation

This directory is intentionally outside `benchmarks/io/`. Its fixture generator and standalone installed-wheel harness do not alter any frozen F-lane helper digest. The X lane remains **UNBASELINED** until all F lanes are disposed, the final integrated pre-X source SHA is fixed, the generated fixture bytes are hashed, and B0/B1/pre-X captures are committed append-only. No runtime code is changed by this preparation.

## Frozen inputs to prepare

Run `bx_fixtures.py OUTPUT` once in a new directory during an assigned host slot. It creates:

| Scenario | Purpose | Decoded payload |
| --- | --- | ---: |
| `ats32` | Native ATS int32 payload; baseline public output remains float64 after scaling | 4,194,304 × 4 bytes (16 MiB) |
| `ats64` | Native ATS int64 payload, including alternating signs and values above int32 range | 4,194,304 × 8 bytes (32 MiB) |
| `matrix` | Homogeneous v2 NetCDF matrix, 4 × 4 cells with one float64 dtype, unit, and axis | 4 × 4 × 262,144 × 8 bytes (32 MiB) |
| `ats_overflow` | Small int64/large-scale fixture for `np.seterr=warn` and `raise` error order | 16 samples |
| `matrix_nan_inf` | Homogeneous float64 cells with NaN, +Inf, -Inf, and signed zero | 2 × 2 × 8 samples |
| `matrix_int64_extrema` | Homogeneous int64 cells with min/max and values beyond 2⁵³ | 2 × 2 × 8 samples |
| `matrix_object_strings` | Encoded object/string cells to characterize the existing warning/error route | 2 × 2 × 8 samples |

The generator writes an exclusive new directory and records its own SHA256, each file SHA256, content shape/dtype, and raw ATS/cell value hashes where stable. Commit the small manifest and verification report to an append-only B-X evidence directory; keep the approximately 90 MiB binary fixture files outside Git. Retain the exact generated files for local captures. On another host, regenerate into a new directory and verify every file against the committed hashes before any wheel comparison. Do not modify the local files after the first baseline read.

## Public and structural capture

`bx_measure.py capture` audits every installed GWexpy wheel member against wheel bytes. Each sample checks the fixture file hash and records values SHA256, dtype, shape, exact `t0`/`dt` hex, unit, matrix keys/units or ATS channel/provenance, warning category/message/order, and error type/message. Numeric SHA uses a direct buffer for C-contiguous arrays and bounded C-order chunks otherwise, avoiding an output-sized fingerprint copy. Preview floats use exact hex strings, including stable `nan`, infinity, and signed-zero spellings. Capture the small edge fixtures with both `--numpy-seterr warn` and `raise` where relevant. `ats_overflow` specifically protects the scale operation's warning or exception timing and message; ATS primary values SHA256 protects full float64 bit patterns. `matrix_nan_inf`, `matrix_int64_extrema`, and `matrix_object_strings` constrain any same-dtype shortcut to cases with B1 public parity.

Structure mode uses `sys.setprofile` to count full-payload NumPy `ndarray.astype` C-calls only inside the native ATS scaling function or NetCDF matrix `lossless()` validator. B1 is expected to make one such ATS call per large read and two per homogeneous matrix cell; these are **hypotheses until baseline capture**. The spy is in a separate process and rechecks its public fingerprint. This counter covers these specific copy sites; it is not an allocation, RSS, or PSS claim. An implementation may choose another safe route, but the same frozen spy and wheel comparison must then be reviewed before its gate is used.

## Uninstrumented resource measurements

The controller uses separate `wall` and `pss` worker processes, with nine samples per arm in `ABBA · BAAB · ABBA · BAAB · AB` order for a release comparison. Wall mode reads once to warm process and file state, then records wall and parent CPU nanoseconds for the second read call; fingerprint hashing follows both clocks. PSS mode completes fixture verification, imports, and the wheel audit before a `READY` marker. The controller sends `GO`, then samples Linux `max_t Σ PSS_process(t)` of parent and live descendants at the same **10 ms** instant until the worker reports `READ_HELD`. The worker retains the read result and waits for `STOP`, so fingerprint hashing occurs after the sampled interval. An explicit `--sample-ms 1` retry is reserved for inadequate peak observation and is labeled separately. PSS samples are lower bounds; never sum independent process peaks. All samples must have the same public result and warning sequence. Small edge cases are correctness-only, and timing/PSS have no claim until a frozen pre-X baseline and exact candidate wheel exist.

Capture phases are `historical` (B0/B1), `prex` (B1/pre-X), and `candidate` (pre-X/X). The full integrated pre-X SHA is required in every phase. In `prex`, arm B must be that SHA; in `candidate`, arm A must be that SHA. Use equal Python/dependency versions and wheel-no-deps installs. Save the raw output directory unchanged and append a separate interpretation package; never rewrite a frozen raw manifest.
