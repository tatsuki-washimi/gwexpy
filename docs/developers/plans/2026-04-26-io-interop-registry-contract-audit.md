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

All repository paths listed above were present in the checked tree. This audit
does not add a CI guard for future path-list drift.

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
- Optional dependency behavior is no longer an untracked policy area for the
  high-value paths covered by PR #257: declared extras, bare-package fallbacks,
  and missing-dependency `ImportError` behavior are represented across the
  package extras, helper maps, contract notes, user docs, and targeted tests.
- The remaining weakness is not the existence of optional-dependency policy,
  but that some details are still split across code, contract prose, docs, and
  tests rather than normalized into one machine-readable field set.

## Current Matrix Coverage And Residual Gaps

The current contract matrix already records the public/registry/direct boundary
for I/O and the public namespace/reference/docs boundary for interop. Optional
dependency and install behavior is also represented for the current baseline:

- package extras in `pyproject.toml` include `seismic`, `io`, `audio`,
  `netcdf4`, `zarr`, `control`, `gw`, and `all`;
- `gwexpy.interop._optional` distinguishes declared-extra and bare-package
  dependency hints plus extras included in `gwexpy[all]`; source-only and
  HDF5-backed bridge behavior is represented by contract notes and bridge
  implementations rather than by one central optional-helper policy;
- direct I/O import guards now emit `gwexpy[extra]` hints for the covered
  optional backends, and tests cover those user-facing errors;
- `public_io_contract.md` and contract notes record important unavailable
  behavior for TDMS and `ats.mth5`;
- docs-sync tests cover the current public I/O and interop row boundaries.

Remaining gaps are narrower:

- the machine-readable JSON schemas still rely on notes/prose for dependency
  behavior instead of normalized fields, so future drift is easier to introduce
  than for names and public APIs;
- direct I/O docs-sync tests assert selected support rows and snippets, not
  every optional-dependency phrase in `io_formats.md` and `installation.md`;
- interop docs-sync tests validate status/API/reference rows, but do not yet
  verify that tutorial install cells and guide notes avoid implying broader
  `gwexpy[all]` coverage than the extras actually provide;
- audio metadata install text still has stale `tinytag` guidance even though
  `tinytag` is now part of the `audio` extra.

## Direct I/O Findings

| Format / family | Current support | Optional dependency behavior | Audit note |
| --- | --- | --- | --- |
| `hdf5` | Broad TimeSeries, FrequencySeries, Spectrogram, Histogram, and EventTable family through GWpy/direct helpers | mostly core HDF5 stack | Well covered; docs correctly recommend explicit `format="hdf5"`. |
| `hdf.ndscope` | Registry exposes several adapters, while public contract is TimeSeriesDict-focused | HDF5 stack | Public-vs-registry boundary is encoded. |
| `xml.diaggui` / `dttxml` | TimeSeriesDict public path plus frequency-domain registry readers | `dttxml` optional for some parser paths; native parser also exists | Public docs mostly show TimeSeriesDict. Frequency-domain support should be explicitly public or registry-only. |
| `mseed` / `sac` / `gse2` / `knet` | TimeSeries, TimeSeriesDict, and TimeSeriesMatrix registry read support | ObsPy | PR #257 covers clearer `gwexpy[seismic]` hints for this family. Public tables can still read broader than the contract. |
| `win` / `win32` | Registered only if ObsPy imports successfully | conditional registration through the ObsPy-backed implementation | Current contract/docs/tests encode the public read-only, collection-first boundary and `win32` alias. The conditional registration detail is covered enough for this audit unless maintainers later want a normalized machine-readable dependency field. |
| `ats` / `ats.mth5` | Native ATS read plus optional MTH5 path | `mth5`, included in the `seismic` extra | Current contract/docs/tests encode the narrow `ats.mth5` public path and missing-`mth5` `ImportError` behavior. No separate docs-only follow-up is needed for `mth5`/`seismic` in this audit. |
| `tdms` | TimeSeries, TimeSeriesDict, and TimeSeriesMatrix read | `nptdms` | PR #257 covers the improved optional-dependency hint. |
| `nc` / `netcdf4` | Direct TimeSeries, Dict, Matrix read/write and interop bridge | `xarray` and usually `netCDF4` | PR #257 adds the extra and install hint. |
| `zarr` | Direct TimeSeries, Dict, Matrix read/write and interop bridge | `zarr` | PR #257 adds the extra and install hint. |
| `wav` | TimeSeries, Dict, Matrix read; TimeSeries write | SciPy WAV stack | Docs correctly separate WAV from compressed audio. |
| `mp3` / `flac` / `ogg` / `m4a` | Audio registry adapters | `pydub`, often external ffmpeg | PR #257 covers core install hints. `extract_audio_metadata()` still has stale `tinytag`-style guidance. |
| planned stubs | placeholder readers fail intentionally | none | Docs list planned stubs; this is aligned. |

## Interop Findings

- `gwexpy.interop.__all__` is contract-covered, which is a strong guard against
  undocumented public namespace drift.
- The interop guide publishes many bridges as public, and the current
  contract/helper/implementation baseline now distinguishes the major install
  categories instead of leaving lower-traffic bridges policy-free:
  - declared-extra dependencies such as `xarray`, `zarr`, `obspy`, `mth5`,
    `control`, `pycbc`, `gwinc`, `finesse`, and `pydub` are covered by
    optional-helper extra hints;
  - bare-package dependencies such as `specutils`, `pyspeckit`, `quantities`,
    `simpeg`, `meshio`, `metpy`, `wrf`, `harmonica`, `exudyn`, `openseespy`,
    `emg3d`, `sdynpy`, `pyuff`, `pyOMA`, `multitaper`, and `mtspec` are covered
    by optional-helper package hints;
  - source-output bridges where the source application is not required at
    runtime, such as `openEMS`/`h5py` and `meep`/`h5py`, are represented by the
    bridge implementations and contract notes rather than optional-helper
    install policy;
  - base-install behavior for bridges backed by core dependencies or already
    installed scientific stack components.
- PR #257 fixed central optional-helper issues, including phantom extras,
  declared-extra hints, bare-package hints, and import-name versus package-name
  hints for the covered dependency set.
- `intro_interop.ipynb` uses a broad `%pip install -q "gwexpy[all]" ...` setup
  pattern, but the notebook demonstrates many packages that are not guaranteed
  to be installed by `all`. PR #257 adds netcdf4 and zarr to `all`, but not the
  broader public interop examples.

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

- No single machine-readable docs-vs-code matrix normalizes
  `format/converter -> owner -> public API -> registry API -> optional
  dependency -> extra/bare install -> unavailable behavior`, even though the
  current policy is represented across existing sources.
- The JSON contract files encode dependency behavior through notes and
  surrounding contract docs rather than structured fields, so tests cannot
  exhaustively compare dependency/install guidance the way they compare API
  names and public rows.
- `tests/io/test_io_contract.py` can allow a canonical reader to be absent by
  continuing when `get_reader()` returns `None`; this remains acceptable for the
  current optional-backend baseline but would need a structured policy if the
  project wants to make conditional registration a first-class contract field.
- `tests/io/test_io_docs_contract_sync.py` checks selected snippets rather than
  every public table row and support cell in `io_formats.md`.
- `tests/interop/test_interop_docs_contract_sync.py` verifies
  status/API/reference rows, but not tutorial bootstrap cells or every optional
  dependency note.
- Optional backend tests are split across markers and many `pytest.importorskip`
  calls. Skipped tests do not by themselves verify clear user-facing errors.

## Recommended Next PRs

1. Audio metadata docs follow-up: update `extract_audio_metadata()` docstring and
   warning text so `tinytag` points to `pip install 'gwexpy[audio]'` or
   `pip install "gwexpy[all]"` where appropriate, instead of stale bare-package
   / "coming soon" guidance.
2. Interop docs/notebook PR: update `intro_interop.ipynb` and/or `interop.md` so
   `gwexpy[all]` does not imply every demonstrated public converter is installed.
3. Optional policy normalization spike: only if maintainers want stronger
   machine-readable enforcement, prototype narrow structured fields for the
   remaining dependency/install/unavailable-behavior cases already represented
   in prose and tests.
4. Frequency-domain DTTXML decision PR: decide whether FrequencySeries
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
