# v0.2.5 I/O baseline harness

`run.py` is independent of the package wheel under test. It creates deterministic
CSV fixtures, verifies their hashes before sampling, and invokes each installed
wheel in a fresh `python -I` process. It rejects an import outside that
interpreter's prefix, a version mismatch, an installed file differing from the
nominated wheel, or a dependency-version mismatch between arms. Full installed
file verification runs before samples; sample workers check import identity.

## Prepare the two arms

Use the same Python, build frontend/backend versions, dependency environment,
and `pip install --no-deps --no-index` mode for both wheels. B0 is the published
v0.2.4 wheel. B1 must be built from old R
`1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, with a clean runtime tree.
Record the build command, toolchain versions, SHA-256 of each source/wheel, and
the exact install commands with the evidence. A candidate wheel should be built
with the same toolchain as B1. Do not install the source checkout in either
benchmark interpreter.

Example:

```sh
python benchmarks/io/run.py fixtures /tmp/v025-fixtures
# For F2, generate CSV plus package-independent SDB/TDMS fixtures:
python benchmarks/io/run.py fixtures /tmp/v025-f2-fixtures --with-formats
# For C2, add deterministic CSV, WAV, and optional miniSEED dispatch fixtures:
python benchmarks/io/run.py fixtures /tmp/v025-c2-fixtures --with-c2
python -m build --wheel --no-isolation --outdir /tmp/v025-wheels
python -m pip download --no-deps --only-binary=:all: gwexpy==0.2.4 -d /tmp/v025-wheels
python -m venv --system-site-packages /tmp/v025-b0
python -m venv --system-site-packages /tmp/v025-b1
/tmp/v025-b0/bin/python -m pip install --no-deps --no-index /tmp/v025-wheels/gwexpy-0.2.4-py3-none-any.whl
/tmp/v025-b1/bin/python -m pip install --no-deps --no-index /tmp/v025-wheels/gwexpy-0.2.5-py3-none-any.whl
```

System site packages are suitable only when their versions are controlled and
the audit confirms identical distributions. Pass the two interpreter paths,
wheel paths, versions, source SHAs, and B0/B1 labels to every `capture` call.
The output directory must be new. Place final reviewed evidence under
`docs/developers/reports/v0.2.5-performance/<lane>/<harness-digest>/baseline-v1/`.
Corrections go in `baseline-v2/`; never edit an existing evidence directory.
For a later C1 candidate comparison, run the **C1 freeze commit's exact
`benchmarks/io/*.py` bytes** from a separate checkout. Later F2 or other lane
edits may change this directory and its harness digest. S2 must compare the
C1 baseline git blob hashes with that frozen commit before accepting a
candidate comparison.
Use the analogous C2 freeze commit's exact `benchmarks/io/*.py` bytes for a
later C2 candidate comparison; F2's earlier harness digest is immutable.

```sh
python benchmarks/io/run.py capture \
  --fixtures /tmp/v025-fixtures --output /tmp/v025-c1-structure \
  --lane C1 --scenario c1_many --mode structure --samples 5 \
  --python-a /tmp/v025-b0/bin/python --wheel-a /tmp/v025-wheels/gwexpy-0.2.4-py3-none-any.whl \
  --version-a 0.2.4 --source-sha-a 522e52a082925da4dd37966d82a7616bdd2a5248 --label-a B0 \
  --python-b /tmp/v025-b1/bin/python --wheel-b /tmp/v025-wheels/gwexpy-0.2.5-py3-none-any.whl \
  --version-b 0.2.5 --source-sha-b 1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c --label-b B1
```

## Runs and interpretation

- `correctness` captures exact value and axis byte hashes, selected samples,
  dtype, units, metadata, ordered collection keys, warning category/message,
  and exception type/message. Compare a candidate with **B1**; B0 is historical.
- `structure` instruments `TimeSeries.append` for regular C1 merge routes.
  `c1_merge_64` preconstructs 64 contiguous 4096-sample float64 segments
  before timing and calls `_multi.read_multi_dict` with a trivial reader; it is
  C1's primary merge-only warm and structural benchmark. The 12-file
  `c1_many` CSV route is end-to-end supporting evidence.
  `append_result_sample_bytes` counts result-array sizes at observed
  calls; it is a B1 proxy, not a claim about allocator traffic. The candidate
  C1 gate additionally requires a structural placement/copy probe at the
  implementation's final output write: each input segment's sample bytes may
  be placed at most once, so the predeclared bound is the sum of input segment
  `nbytes` (64 × 4096 × 8 = 2,097,152 bytes for `c1_merge_64`; 12 × 2048 ×
  8 = 196,608 bytes for `c1_many`). The append spy alone
  does not establish this bound; the C1 implementation test must instrument
  its placement path or supply source-audited equivalent evidence. Irregular
  C1 routes are correctness checks and do not carry the zero-append gate.
  The C1 fast path excludes series requiring `_xindex` fallback; a candidate
  implementation test must show they still use the unchanged append path.
- `timing --temperature cold` measures controller spawn through worker exit,
  including fresh-process import/backend setup. `--temperature warm` keeps one
  process per arm, warms each once, then interleaves timed function-level
  repeats between the two processes. The worker's `cpu_ns` covers the route
  call in both modes. Each arm has five baseline or nine release
  samples in `ABBA BAAB ABBA BAAB AB` prefix order.
- `memory` samples Linux parent and child PSS/RSS together from process launch
  through exit. The tree peak is the maximum of simultaneous sums, never a sum
  of individual peaks. Each trace includes monotonic timestamps and PIDs. It
  is a sampled lower bound. For a worker pool route, rerun memory only with
  `--sample-ms 1` if the expected worker count was not observed.
- `fixture_counts` are declared generator facts. They are **not** parser
  validation/materialization counters. A lane must add exact structural probes
  for those counters before claiming the corresponding structural gate.

Routes `f2_1a` and `f2_1b` call `FrequencySeries.read` with explicit `csv`
format and auto detection respectively, on the **same nonuniform-axis 4096-row
file**. Keep their fingerprints and wall samples separate. `f2_2` reads all
eight channels of an 8192-row CSV; `f2_3` selects `ch1` from a 16-channel,
65,536-row CSV. `f2_writer` calls the internal enhanced CSV writer directly,
because the public single-series CSV registration selects GWpy's writer.
The `f2_fault_*` cases cover selected/unselected bad values, timestamp errors,
row width, and first-malformed-row order under both one-channel and all-channel
reads. `c1_*` cases call the format reader directly with a list so they reach
`_multi.read_multi_dict`, the function changed in C1. `c1_public_many` records
the separate GWpy registry list merge behavior. C1 cases cover input order,
gaps, overlap, one file, and empty input. Synthetic direct `_multi` cases add
channel order, integer/NaN padding, unit conversion, nanosecond GPS placement,
and first-source provenance; they generate series from deterministic constants
inside the isolated wheel process.

For F2, `f2_2` and `f2_3` structural runs inspect the enhanced parser's
successful-return locals. They count validated tokens, Python row values,
NumPy matrix values, and materialized columns exactly for the B1 parser;
`csv_parser_local_probe_covered` must be true. A candidate that replaces these
locals needs an equivalent structural probe bound to its parser before any
materialization claim is accepted. Route ③'s declared input has 65,536 rows,
17 columns, and 1,114,112 tokens; its target output is one selected value
column plus time. Routes ①a/①b stay separate and must have identical full
fingerprints, including the exact nonuniform frequency array.

`f2_writer` calls the enhanced internal writer with 524,288 preconstructed
float64 samples. Its structural run supplies a file-like counting sink that
records the exact number and size of writes without retaining output bytes;
the correctness run writes a real file and hashes every byte. Timing and
memory use the real path. A full-output-buffer count of zero and lower Linux
peak RSS are both required for an internal enhanced-writer improvement claim.
The write-call counter observes only the string passed to `write()`; a
candidate could still retain all rows and emit smaller writes. A candidate
also needs an allocation probe at its row-buffer construction path, or a
source-audited equivalent, to show that output-sized Python buffers are not
retained. No public `TimeSeries.write(format="csv")` speed or memory claim
follows from this route, because that registration uses GWpy's native writer.

F2 SDB fixtures cover valid selected/all/window reads, malformed whole-source
`usUnits` and timestamp rows outside the window, selected and unselected bad
payloads, and a WAL update scheduled at the payload SELECT. The structural
DataFrame probe records B1 fetched payload rows separately from the metadata
validation scan. A candidate that stops using `pandas.read_sql_query` needs
an equivalent query/fetch counter. The WAL case records the trigger SQL,
before/after values, public result, warnings, and logs. A same-snapshot SDB
optimization is **HOLD** if it changes B1's concurrent-update result until a
separate W0 behavior exception and disclosure are approved.

F2 TDMS fixtures cover selected/all reads, selected and unselected invalid
increments, and truncated selected/unselected raw payloads. Structural runs
count `TdmsChannel.read_data()` by selected versus unselected channel.
Correctness fingerprints include Python warnings and backend logging records
with exact logger, level, and message. The truncated-payload fixture may be
detected at file open by npTDMS, before per-channel selection; interpret its
captured B1 behavior directly rather than assuming a channel-local fault.

For C2, `--with-c2` adds numeric-only CSV files, interleaved stereo WAV, and
miniSEED when ObsPy is available. Headered CSV is a separate B1 error case:
the public native CSV reader does not treat its first row as a header. Public
single-series and direct multi-series calls are fingerprinted separately,
including no-selection first-channel order, requested/unselected malformed
values, a three-byte-truncated stereo WAV under selected/no-selection reads,
and a deterministic simulated missing-ObsPy dependency. The larger
65,536-row CSV, 1,048,576-frame WAV, and 131,072-sample-per-channel miniSEED
fixtures support warm/cold and Linux RSS/PSS measurements.

C2 structural probes count the actual SciPy WAV backend call and its returned
interleaved values, then count selected and unselected `TimeSeries`
constructions. SciPy WAV cannot read only one interleaved channel, so a zero
unselected backend-read gate does not apply; removing the unselected series
construction is a separate candidate gate. The ObsPy probe counts full-stream
backend calls/returned traces and trace-to-series conversions. B1 already
converts no unselected trace for selected reads, so that zero baseline cannot
support a selection improvement claim. The large CSV probe exposes B1 enhanced
parser materialization as context for the F2-owned parser; C2 does not claim
to optimize that parser. A synthetic registered dict-to-single adapter probe
shows that B1 already forwards an explicit `channels=['second']` selector to
its backend (zero unselected reads); without selection it reads both entries
and returns the first in stable order. This synthetic probe characterizes
dispatch behavior, not a disk I/O speed claim. Candidate implementations that
replace these call sites need equivalent exact spies before any avoided-work
claim.

Structural, timing, and memory modes are separate processes. Do not treat a
baseline as frozen until each required B1 scenario has a public fingerprint,
raw samples, fixture and harness hashes, installed-wheel audit, reviewed fault
matrix, and an append-only committed manifest. The `capture` manifest always
uses `UNBASELINED`; the release owner records freeze status after review and
commit. No performance claim follows from the baseline alone.

### SDB concurrent WAL fault matrix (baseline-v2)

`sdb_wal_v2.py` independently reproduces the SDB payload-query race after
source metadata validation. It captures seven committed WAL mutation classes
(selected and unselected values, selected and unselected malformed payloads,
timestamp, `usUnits`, and schema) with both selected and all-column public
reads. Each route has five B0 and five B1 samples in ABBA order. The runner
records the exact trigger and writer SQL, commit, before/after row and schema,
public fingerprint, warnings, logs, and errors. It asserts that the metadata
query precedes the update and that WAL commit succeeds. The fixture is copied
into a fresh scratch database for every worker. This route is a correctness
matrix; use the F2 baseline-v1 harness for static-source timing, memory, and
fetched-row structure. The v1 evidence and harness bytes remain immutable.

The baseline-v2 evidence directory contains its own fixture manifest and exact
SQLite source bytes. Candidate comparisons must run the frozen `*.py` harness
from the baseline-v2 freeze commit. If candidate SQL changes, review an
equivalent trigger before comparing fingerprints. An approved concurrency
exception still requires separate technical review, disclosure, and human S2
scientific/data-model approval.

## F1 bounded range-read baseline

`f1_range_fixtures.py` creates 102 public `TimeSeriesDict.read` cases across
plain HDF5, NDScope HDF5, NetCDF4, and Zarr. The 4096-sample fixture is the
correctness oracle; a separate 524288-sample fixture is used only for four
short-window structural, warm, cold, and Linux PSS/RSS routes. Both generators
record every file hash and binary64 boundary. Correctness includes ±1 ULP,
half-sample, one-sided, partial/disjoint windows, `pad=`, and selected versus
out-of-range corrupt chunks. B1's public fingerprint is authoritative; an
unbounded read followed by `crop` has not been promoted to an oracle.

`f1_range_run.py` audits each installed wheel and matched dependency versions.
It captures all 102 cases with exact result/metadata/array hashes, warning and
log records, and error type/message/cause. The structural probe records one
full 524288-element materialization per B1 short-window route, via h5py
Dataset, xarray DataArray `values`, or Zarr Array access. The xarray counter
measures full variable materialization, not exact NetCDF backend bytes; a
candidate claiming fetched-byte bounds needs an equivalent backend probe.
Warm timing uses one process per arm, one warm-up, then five interleaved
function calls. Cold and Linux PSS use fresh workers, five samples per arm.

The #611 exception is limited to a completely disjoint entry in a plain-HDF5
mixed-channel read after the parent read succeeds. It does not apply to a
single-channel parent coverage error or another format. B1 raises on
out-of-range corrupt chunks in all four formats. A candidate that skips that
error changes public behavior. The F1 runtime push-down claim is **HOLD**:
preserving those errors requires reading the damaged out-of-range chunks,
which conflicts with the predeclared backend-fetched-bytes bound to selected
chunks. A full scan using bounded memory also fails that fetched-byte gate.
Neither #589 nor #611 authorizes a broader exception. This baseline does not
authorize F1 runtime work. Use the exact frozen `benchmarks/io/*.py` bytes
from the F1 baseline commit for any later comparison.

## F5 WIN decoder baseline

`f5_win_run.py` uses the same B0/B1 wheel audit and dependency identity checks
as `run.py`. It independently constructs 12 byte-level WIN fixtures with all
five DATAWIDE codes, sample rates 1 and 4095, both int32 overflow directions,
and unsupported-width/truncated-packet/truncated-channel faults. The WIN
fixture manifest records each file hash and exact expected values. Correctness
captures both `_read_win_fixed(path)` and public `read_win_file(path)` routes,
including full value bytes, dtype, channel, rate, UTC warning, and exception
category/message. The controller rejects a B0/B1 mismatch or deviation from
the independent wire recipe.

```sh
python benchmarks/io/f5_win_run.py fixtures /tmp/v025-f5-fixtures
python benchmarks/io/f5_win_run.py capture \
  --fixtures /tmp/v025-f5-fixtures --output /tmp/v025-f5-correctness \
  --mode correctness \
  --python-a /tmp/v025-b0/bin/python --wheel-a /tmp/v025-wheels/gwexpy-0.2.4-py3-none-any.whl \
  --version-a 0.2.4 --source-sha-a 522e52a082925da4dd37966d82a7616bdd2a5248 --label-a B0 \
  --python-b /tmp/v025-b1/bin/python --wheel-b /tmp/v025-wheels/gwexpy-0.2.5-py3-none-any.whl \
  --version-b 0.2.5 --source-sha-b 1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c --label-b B1
```

Use the same arm arguments with `--mode structure`, `--mode timing
--temperature warm`, `--mode timing --temperature cold`, and `--mode memory`.
The structural mode traces source lines that append one reconstructed sample;
its old-R count characterizes work, while the candidate needs an independent
source/AST assertion that no width reconstructs samples with a Python
per-sample accumulation loop. The timing and memory route uses the 4095-sample
1-byte fixture. Warm mode pre-reads the file and performs one decoder warm-up
outside the measured call. CPU time of the decoder call is the primary metric;
cold startup and sampled Linux PSS/RSS are supporting evidence. Use five
samples per arm in `ABBA BAAB AB` order for baseline smoke, with timing and
memory runs on a quiet host. Freeze the harness bytes and baseline evidence
before editing WIN runtime code.
