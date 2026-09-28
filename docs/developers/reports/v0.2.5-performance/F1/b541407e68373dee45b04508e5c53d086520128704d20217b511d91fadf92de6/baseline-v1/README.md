# B-F1 range-read baseline-v1

This is the frozen **behavioral and resource baseline** for issue #584. It
does not authorize F1 runtime push-down. The B1 oracle is the public
`TimeSeriesDict.read(source, format=..., start=..., end=...)` call from the
clean old-R wheel. `full_read.crop(...)` is not treated as an oracle.

The independent fixture helper generates 102 public cases over plain HDF5,
NDScope HDF5, NetCDF4, and Zarr. The correctness fixture has 4096 float32
samples, `t0=12345.125` and non-binary-exact `dt=0.1`; it covers exact and
±1 ULP bounds, half-sample and one-sided windows, one-sample results,
partial/disjoint windows, `pad=`, and damaged selected/out-of-range chunks.
A second 524288-sample fixture (256 samples/chunk) is used only for the
four short-intersecting performance routes. Both fixture sets were generated
in two independent temporary directories and matched byte for byte.

All 102 public B0/B1 correctness fingerprints agree. They retain dtype,
ordered keys, sample bytes, timing axes, metadata, warnings, logs, and exact
error type/message/cause. In the plain-HDF5 #611 mixed case, B1 returns
`covered` then `disjoint`: the first has two samples and the second is a
zero-length entry. A disjoint single-series parent request can instead raise
a coverage `ValueError`; `pad=0` is a separate public result. Other formats
use their own B1 disjoint behavior.

Every format raises on a damaged chunk even when that chunk lies outside the
requested window: HDF5/NDScope raise an HDF5 `OSError`, NetCDF raises
`RuntimeError: NetCDF: HDF error`, and Zarr raises a Zstd decompression
`RuntimeError`. The raw fingerprints are authoritative for exact messages.
This is an explicit **HOLD boundary**. Retaining those B1 errors requires
reading the out-of-range damaged chunks, while the predeclared F1 gate
requires backend fetched bytes to be bounded by selected chunks. A full scan
using bounded memory still fails that fetched-byte gate. Neither #589 nor
#611 authorizes a wider exception. No F1 runtime change or push-down claim
follows from this baseline.

Separate structural runs show one 524288-element full materialization in
each B1 short-window route. The h5py and Zarr spies wrap array access;
NetCDF's spy wraps xarray `DataArray.values`, so its count describes eager
variable materialization, not exact backend fetch bytes. The large-fixture
structural result fingerprint equals the corresponding 4096-row public
correctness fingerprint for all five samples in each arm. Five-sample warm
timing keeps one worker per arm, warms once, then interleaves function calls
in ABBA order. Cold and Linux tree-PSS runs use fresh workers. Sampled PSS
is a peak estimate. Timing and memory are supporting evidence only while
the F1 runtime claim is HOLD.

B0 uses the published v0.2.4 wheel and B1 uses the old-R wheel. Both were
installed into fresh matched `--system-site-packages` venvs with `--no-deps`;
Zarr 3.1.5, numcodecs 0.16.5, donfig 0.8.1.post1, and google-crc32c 1.8.0
were installed from the same downloaded wheels into both venvs using
`--ignore-installed --no-deps --no-index`. The runner uses `python -I`, audits
every installed gwexpy wheel member before capture, and records identical
dependency versions and per-arm in-venv import paths. Exact wheel, optional
wheel, fixture, harness, and raw-sample hashes are in the manifest.

Use the exact `benchmarks/io/*.py` bytes from the commit containing this
baseline for any later comparison. Later lane changes are not interchangeable
with this harness.
