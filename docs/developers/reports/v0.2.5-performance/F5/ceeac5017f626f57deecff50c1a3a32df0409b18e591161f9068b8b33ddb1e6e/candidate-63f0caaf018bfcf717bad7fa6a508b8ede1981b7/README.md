# F5 WIN NumPy candidate evidence

Source commit: `63f0caaf018bfcf717bad7fa6a508b8ede1981b7`, based on
`77bea5032f36c7eb2eb09168403c328ba03cbe15`. This candidate implements
the NumPy portion of #518 only. Its wheel SHA-256 is
`06dd14eebae496d1e1a3f6f27853fa642f17170ca4c6b3a47251ec06dcdd60ae`.
All 369 wheel package files match the source commit byte for byte. The wheel
was built with the B1 build command/toolchain and installed with
`pip install --no-deps --no-index` in an isolated `venv --system-site-packages`
under the same Python and dependency versions as B1. Each harness capture
audits import path, installed wheel bytes, and distribution versions.

## Behavior and structure

The exact frozen B-F5 harness digest is
`ceeac5017f626f57deecff50c1a3a32df0409b18e591161f9068b8b33ddb1e6e`.
`correctness/fingerprint.json` reports B1/candidate equality on all 12
independently generated wire fixtures. Both `_read_win_fixed(path)` and the
public `read_win_file(path)` have exact full value bytes, dtype, channel,
sampling rate, timestamp, ordered public keys, warnings, and error
category/message. Cases include all DATAWIDE codes, sample rates 1 and 4095,
both signs of int32 overflow, and the three malformed records.

`structure/raw-*.json` records 4,116 B1 Python sample-append line hits across
valid fixtures and **zero** candidate hits. The candidate source retains
`_apply_4bit_deltas` for existing internal callers; the WIN reader no longer
calls it. Its only `samples.append` in the new path is the conservative
fallback for a bound above int64. This branch cannot be reached by a valid WIN
channel: `|absolute| ≤ 2^31`, `n ≤ 4094`, and `|delta| ≤ 2^31` imply
`|absolute| + n·max|delta| ≤ 4095·2^31 < 2^63`. A direct test verifies the
fallback preserves Python integer accumulation at both int64 boundaries.

The decoder vectorizes nibble sign extension, signed 8/16/24/32-bit delta
decoding, and int64 cumulative summation. Existing packet length, topology,
timestamp, channel, and warning validation remains in the same order. The
decoded-count test was updated to inject a short array at the new bulk
`_decode_win_deltas` seam; its expected `ValueError` contract is unchanged.

## Performance

The primary route is the frozen 4095-sample, 1-byte decoder call. Nine warm
samples per arm were interleaved in `ABBA BAAB ABBA BAAB AB`; cold and Linux
sampled-memory runs used five samples per arm. Captures were separate and used
an assigned quiet-host slot. `summary.json` in each route contains every raw
sample, median, and MAD.

| Metric | B1 median (MAD) | F5 median (MAD) | Result |
| --- | ---: | ---: | --- |
| Warm decoder CPU | 1,789,846 ns (121,073) | 264,166 ns (22,805) | 85.24% lower; primary gate PASS |
| Warm decoder wall | 1,792,159 ns (121,158) | 267,270 ns (23,073) | Supporting |
| Cold controller wall | 2,890,698,689 ns (7,813,205) | 2,871,229,534 ns (49,399,914) | Supporting |
| Peak tree PSS | 298,430 KiB (153) | 298,508 KiB (300) | Supporting; no improvement claim |
| Peak tree RSS | 311,772 KiB (156) | 311,864 KiB (416) | Supporting; no improvement claim |

For the primary CPU metric, `improvement = 0.8524` and
`noise = MAD_B1/median_B1 + MAD_F5/median_F5 = 0.1540`. The predeclared
threshold `max(0.05, 2×noise)` is `0.3079`, so the primary gate passes. The
sampled PSS/RSS peaks are lower bounds at 10 ms resolution. Cold startup shows
no small-file regression beyond the plan's `max(10%, 3×noise)` rule.

## Checks and limits

- Focused WIN tests: 48 passed, with one expected UTC warning.
- Ruff check and Ruff format check: passed for the changed Python files.
- MyPy: passed for `gwexpy/timeseries/io/win.py`.
- `git diff --check`: passed on source commit.
- Candidate correctness and structural captures: passed.
- No full repository suite, physics check, or public API/dependency/schema
  change is involved in this internal decoder optimization.

Only the 1-byte 4095-sample route carries the performance claim. All widths
carry exact correctness and structural coverage. The source commit remains
the candidate wheel identity; this evidence commit adds reports only.
