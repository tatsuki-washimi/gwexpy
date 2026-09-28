# B-F5 WIN decoder baseline v1

Status: baseline evidence collected before any F5 runtime edit. This directory
is append-only after its freeze commit. The candidate must use the exact
`benchmarks/io/*.py` bytes from that commit and compare against B1.

## Identity and fixture contract

- Harness digest: `ceeac5017f626f57deecff50c1a3a32df0409b18e591161f9068b8b33ddb1e6e`.
- WIN fixture manifest SHA-256: `db9e32a69f36df9f68f29ef186114fb7be310f7cd80bba50ca3b8243c4cb02b9`.
- B0 published v0.2.4 wheel SHA-256: `34afe8188c753cd9da0b182a5d2a88cce2f7ec36633730fe0bee15e2506df56d`; source tag SHA `522e52a082925da4dd37966d82a7616bdd2a5248`.
- B1 old-R wheel SHA-256: `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`; old-R source SHA `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`.
- B1 wheel was built with Python 3.12.12, `build` 1.3.0, `setuptools` 80.9.0, and `wheel` 0.45.1 using `python -m build --wheel --no-isolation`. Both wheels were installed with `pip install --no-deps --no-index` in `venv --system-site-packages` environments using the same Python and dependency versions. The pre-sample audit verified all 369 installed package files per arm against the nominated wheel, imported from each arm's `site-packages`, and checked interpreter/version identity. An independent `git show old-R:<path>` comparison matched all 369 B1 wheel package files byte for byte to old R.
- The fixture helper entered this worktree at SHA-256 `0e6b4f1235a872c3f7cbbebf91ad521ee4df4b65268b8a84fa752473fcc5bad0`. Before freeze, its expected warning/error metadata was corrected to the observed exact B1 public behavior, including the UTC warning on malformed reads; it was also formatted and linted. All 12 wire files retain independent generator recipes and per-file hashes in the WIN fixture manifest. The final helper SHA is recorded in `baseline-manifest.json`.

## Preflight

`correctness/fingerprint.json` is B0/B1 equal across all 12 cases. The
controller also checked both arms against the generator's expected full integer
values, dtype, channel, sampling rate, exact warning category/message, and
error category/message. It captures `_read_win_fixed(path)` and public
`read_win_file(path)` separately. Valid cases cover DATAWIDE 0.5/1/2/3/4,
rate 1, rate 4095, and both signs of int32 overflow. Faults cover unsupported
width, truncated packet, and truncated channel. The public route emits
`builtins.UserWarning`: `WIN header time is timezone-naive; interpreting as UTC
(#632)` on valid and malformed inputs; the decoder route emits no warning.

`structure/raw-B.json` records five old-R `samples.append`/`output.append`
source sites and 4,116 traced Python per-sample accumulation line hits across
valid fixtures, including 4,094 for the 4095-sample primary case. B0 has the
same source hash and counts. Candidate structural acceptance still requires a
source/AST review that each optimized width reconstructs samples by NumPy bulk
operations with no Python per-sample accumulation loop. This baseline trace is
an old-R characterization, not a candidate structural pass.

## Quiet-host measurements

The predeclared primary route is `_read_win_fixed(path)` on the 4095-sample
1-byte fixture. Warm service processes pre-read the file and run one decoder
call before interleaved timing. CPU time covers the decoder call, including its
normal file open and packet parse; fixture construction and pre-read are
outside the timed call. Nine samples per arm use `ABBA BAAB ABBA BAAB AB`.
Cold timing and Linux sampled memory use five samples per arm in the five-sample
prefix of that order. Structure, timing, and memory ran separately.

| Metric | B0 median (MAD) | B1 median (MAD) | Role |
| --- | ---: | ---: | --- |
| Warm decoder CPU | 1,652,314 ns (48,466) | 1,600,621 ns (51,029) | Primary |
| Warm decoder wall | 1,655,242 ns (48,302) | 1,603,389 ns (50,844) | Supporting |
| Cold controller wall | 2,708,359,847 ns (8,912,626) | 2,682,944,830 ns (24,081,277) | Supporting |
| Peak tree PSS | 298,452 KiB (479) | 298,378 KiB (258) | Supporting; sampled lower bound |
| Peak tree RSS | 311,760 KiB (336) | 311,676 KiB (108) | Supporting; sampled lower bound |

Each route's `summary.json` contains the unrounded raw samples, median, and MAD;
individual `sample-*.json` files retain wheel audit and memory trace where
applicable. The B0/B1 warm CPU median difference is about 3.1%, while the
combined relative MAD is about 6.1%. These are historical baseline
observations, not a performance improvement claim. The F5 candidate's primary
claim must use the plan's `max(5%, 2×noise)` threshold against B1, preserve
all correctness fingerprints, and pass the structural gate.

## Audit and limits

Actions: read the approved work plan and repository instructions; inspected
old-R WIN decoding; repaired fixture metadata before freeze; generated fixtures;
ran B0/B1 wheel and dependency audits, full correctness and structural probes;
ran separate quiet-host warm CPU, cold, and Linux memory captures; checked
Ruff and formatting on the two new Python files; checked `git diff --check`.
No runtime code or tests were changed. Full pytest, MyPy, and physics checks
were omitted because this commit adds only a benchmark harness and evidence.
The 4095-sample 1-byte case is the primary timing route; other widths are
covered by exact correctness and structure, not separate timing claims. The
sampled PSS/RSS peak is a lower bound at 10 ms resolution.
