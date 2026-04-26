# I/O And Interop Registry Contract Audit

Date: 2026-04-26
Issue: #248, "Audit I/O and interop registry contracts against public docs"
Mode: audit-first only; no runtime behavior changes in this pass.

## Constraints

- `.agent/AGENTS.md` was reviewed. The relevant requirements are registry
  bootstrap awareness, clear optional dependency behavior, and public docs that
  match supported APIs.
- This document records docs-vs-code gaps and follow-up candidates. It does not
  change readers, writers, converters, dependency hints, contract JSON, or public
  docs.
- PR #257 / Issue #251 has merged and is treated as the baseline optional-backend
  fix set. This audit separates gaps already covered there from remaining work.
- PR #262 / Issue #250 has merged and is treated as the broader public-docs drift
  baseline.

## Inputs Reviewed

- Issue #248 body.
- Issue #251 and PR #257 optional-backend / packaging context.
- Issue #250 and PR #262 public-docs audit context.
- `docs/developers/contracts/public_io_contract.json`
- `docs/developers/contracts/public_io_contract.md`
- `docs/developers/contracts/public_interop_contract.json`
- `docs/developers/contracts/public_interop_contract.md`
- `gwexpy/io/`
- `gwexpy/timeseries/io/`
- `gwexpy/frequencyseries/io/`
- `gwexpy/spectrogram/io/`
- `gwexpy/interop/`
- `docs/web/en/user_guide/io_formats.md`
- `docs/web/en/user_guide/interop.md`
- `docs/web/en/user_guide/tutorials/intro_interop.ipynb`
- `docs/web/en/user_guide/installation.md`
- `tests/io/`
- `tests/interop/`
- `scripts/ci/run_gate.py`

## Existing Contract Strengths

- `public_io_contract.json` already distinguishes `public_api`, `direct_api`,
  and broader `registry_api`.
- `public_interop_contract.json` checks `gwexpy.interop.__all__`, interop guide
  rows, and reference index exposure.
- `tests/io/test_io_contract.py`,
  `tests/io/test_io_docs_contract_sync.py`,
  `tests/interop/test_interop_contract.py`, and
  `tests/interop/test_interop_docs_contract_sync.py` prevent a large class of
  docs/registry/name drift.
- The remaining weakness is not the existence of contracts, but the lack of a
  full docs-vs-code optional-dependency matrix.

## Missing Matrix Fields

The current contract files mostly cover names and public surfaces. They do not
centrally encode:

- owning optional package,
- owning gwexpy extra,
- unavailable-dependency error message,
- whether install guidance should be bare package install or
  `pip install 'gwexpy[extra]'`,
- whether a missing dependency causes a registered reader to raise `ImportError`
  or prevents registration entirely,
- whether registry-only behavior is intentionally unpublished.

## Direct I/O Findings

| Format / family | Current support | Optional dependency behavior | Audit note |
| --- | --- | --- | --- |
| `hdf5` | Broad TimeSeries, FrequencySeries, Spectrogram, Histogram, and EventTable family through GWpy/direct helpers | mostly core HDF5 stack | Well covered; docs correctly recommend explicit `format="hdf5"`. |
| `hdf.ndscope` | Registry exposes several adapters, while public contract is TimeSeriesDict-focused | HDF5 stack | Public-vs-registry boundary is encoded. |
| `xml.diaggui` / `dttxml` | TimeSeriesDict public path plus frequency-domain registry readers | `dttxml` optional for some parser paths; native parser also exists | Public docs mostly show TimeSeriesDict. Frequency-domain support should be explicitly public or registry-only. |
| `mseed` / `sac` / `gse2` / `knet` | TimeSeries, TimeSeriesDict, and TimeSeriesMatrix registry read support | ObsPy | PR #257 covers clearer `gwexpy[seismic]` hints for this family. Public tables can still read broader than the contract. |
| `win` / `win32` | Registered only if ObsPy imports successfully | conditional registration, not a reader-level install hint | Docs list WIN/WIN32 as read-only but do not say registration depends on ObsPy. This is not fully covered by #257. |
| `ats` / `ats.mth5` | Native ATS read plus optional MTH5 path | `mth5` for `ats.mth5` | Direct ATS docs match; `ats.mth5` install guidance remains more generic than the `seismic` extra policy. |
| `tdms` | TimeSeries, TimeSeriesDict, and TimeSeriesMatrix read | `nptdms` | PR #257 covers the improved optional-dependency hint. |
| `nc` / `netcdf4` | Direct TimeSeries, Dict, Matrix read/write and interop bridge | `xarray` and usually `netCDF4` | PR #257 adds the extra and install hint. |
| `zarr` | Direct TimeSeries, Dict, Matrix read/write and interop bridge | `zarr` | PR #257 adds the extra and install hint. |
| `wav` | TimeSeries, Dict, Matrix read; TimeSeries write | SciPy WAV stack | Docs correctly separate WAV from compressed audio. |
| `mp3` / `flac` / `ogg` / `m4a` | Audio registry adapters | `pydub`, often external ffmpeg | PR #257 covers core install hints. `extract_audio_metadata()` still has stale `tinytag`-style guidance. |
| planned stubs | placeholder readers fail intentionally | none | Docs list planned stubs; this is aligned. |

## Interop Findings

- `gwexpy.interop.__all__` is contract-covered, which is a strong guard against
  undocumented public namespace drift.
- The interop guide publishes many bridges as public, but only some rows include
  install guidance.
- PR #257 fixed central optional-helper issues, including phantom extras and
  import-name versus package-name hints for the covered dependency set.
- Remaining long-tail public interop targets still need an install policy:
  `ROOT`, `specutils`, `pyspeckit`, `quantities`, `simpeg`, `meshio`, `metpy`,
  `wrf`, `harmonica`, `exudyn`, `openseespy`, `openEMS`/`h5py`, `meep`/`h5py`,
  `emg3d`, `sdynpy`, `pyuff`, `pyOMA`, `multitaper`, and `mtspec`.
- `intro_interop.ipynb` uses a broad `%pip install -q "gwexpy[all]" ...` setup
  pattern, but the notebook demonstrates many packages that are not guaranteed
  to be installed by `all`. PR #257 adds netcdf4 and zarr to `all`, but not the
  broader long-tail public interop set.

## Already Covered By PR #257

Avoid duplicating these optional-backend changes in #248 follow-ups:

- add `netcdf4` and `zarr` extras and include them in `all`,
- fix `_optional.py` phantom extras and import-name/package-name hints,
- fix `ensure_dependency()` extra hint format to
  `pip install 'gwexpy[extra]'`,
- improve missing-dependency hints for `pydub`, `obspy`, `nptdms`, `zarr`, and
  `xarray`,
- add optional-dependency tests for those guard paths,
- update installation docs for `netcdf4`, `zarr`, and `gui` not being included
  in `all`.

## Test Gaps

- No single machine-readable docs-vs-code matrix records
  `format/converter -> owner -> public API -> registry API -> optional
  dependency -> extra -> unavailable behavior`.
- `public_io_contract.json` has no `optional_dependencies`, `extras`, or
  `unavailable_behavior` fields.
- `public_interop_contract.json` tracks public namespace and guide rows, but not
  dependency names, extras, or missing-package behavior.
- `tests/io/test_io_contract.py` can allow a canonical reader to be absent by
  continuing when `get_reader()` returns `None`; this can hide conditional
  registration gaps such as `win`/`win32`.
- `tests/io/test_io_docs_contract_sync.py` checks selected snippets rather than
  every public table row and support cell in `io_formats.md`.
- `tests/interop/test_interop_docs_contract_sync.py` verifies status/API/reference
  rows, but not install guidance or optional dependency text.
- Optional backend tests are split across markers and many `pytest.importorskip`
  calls. Skipped tests do not by themselves verify clear user-facing errors.

## Recommended Next PRs

1. Contract schema PR: add `optional_dependencies`, `extras`, and
   `unavailable_behavior` fields to both public contract JSON files, then extend
   docs-sync tests to assert those fields are represented in user docs or
   installation docs.
2. Direct I/O follow-up after #257: cover `win` conditional registration,
   `ats.mth5` install guidance, and stale `tinytag` install text in audio
   metadata helpers.
3. Interop optional-dependency matrix PR: decide which long-tail public interop
   targets get declared extras, which remain bare-install only, and which should
   stay public without being in `gwexpy[all]`.
4. Registry contract test PR: public registry entries should not silently
   disappear when optional dependencies are unavailable unless conditional
   registration is encoded explicitly.
5. Interop docs/notebook PR: update `intro_interop.ipynb` and/or `interop.md` so
   `gwexpy[all]` does not imply every demonstrated public converter is installed.
6. Frequency-domain DTTXML decision PR: decide whether FrequencySeries
   `xml.diaggui` support is public. If yes, publish it in `io_formats.md` and
   contract JSON; if no, mark it registry-only/internal.

## Dependencies

- #251 / #257 should remain the baseline for optional-backend packaging and
  install-hint behavior.
- #250 / #262 provides general public-docs drift governance, but #248 needs a
  more specific I/O/interop support matrix.
- Prior I/O/docs contract work from #235 / #236 / #237 is already reflected in
  the contract JSON and test structure and should be preserved.

## Audit Notes

- This file records source/docs/test inspection and current contract risks only.
- No runtime code, tests, registry behavior, dependency declarations, or public
  docs changed in this pass.
- Follow-up PRs should separate schema/test additions from behavior changes so
  optional-dependency policy remains reviewable.
