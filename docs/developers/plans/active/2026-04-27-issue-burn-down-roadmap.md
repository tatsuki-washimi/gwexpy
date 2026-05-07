# Issue Burn-Down Roadmap For Contract Audits And Release Readiness

Date: 2026-04-27
Issues: #269-#294 plus follow-up routing issues #346 and #347
Mode: planning document only; no runtime behavior changes in this pass.

> **For agentic workers:** Use this document as the coordinating plan for the
> post-audit issue wave. Prefer small contract-inventory and regression-test PRs
> before behavior-changing PRs. Physics-facing behavior changes require explicit
> human review.

## Goal

Close the current contract-audit and release-readiness issue wave without
creating large, cross-cutting PRs. The working strategy is to stabilize the
low-level data contracts first, then audit public surfaces, then address
numerical and analysis behavior, then synchronize plot/docs behavior, and only
then publish package releases.

## Issue Inventory

### Foundation And Data Model

- #269 Follow up SeriesMatrix invariant hardening
- #270 Define and test family-specific `to_matrix` contracts
- #271 Add core metadata round-trip contract tests
- #289 Audit collection API ordering mapping and metadata contracts
- #291 Audit low-level types metadata and axis foundation contracts
- #292 Audit FrequencySeries and Spectrogram standalone class contracts

### Individual Public APIs

- #276 Audit SegmentTable and table contract semantics
- #279 Audit time GPS and leap-second conversion contracts
- #280 Audit CLI command contracts and failure behavior
- #281 Audit detector channel proxy and unit alias contracts
- #287 Audit field data model axis unit and algebra contracts
- #290 Audit histogram and segments public API contracts

### Numerical, Statistical, And Analysis Surfaces

- #273 Add numerical algorithm contract regression tests
- #277 Audit fitting API contracts and metadata propagation
- #278 Audit noise model PSD ASD and optional backend contracts
- #282 Audit astro range units and compatibility contracts
- #284 Audit analysis workflow contracts for Bruco coupling and response
- #285 Audit time-frequency and space-transform contracts
- #286 Audit statistics correlation and numerical primitive contracts
- #288 Audit preprocessing pipeline decomposition and forecasting contracts
- #346 Track preprocessing runtime policies after #288

### Plot, GUI, Docs, And I/O Residuals

- #272 Finish small I/O and interop audit follow-ups
- #274 Harden GUI and plot metadata contracts (post-release GUI track)
- #275 Add tutorial and public-doc drift prevention guards
- #283 Audit extended plot helper and visual contract surface
- #347 Track residual plot-helper policies after #283

### Release Roadmaps

- #293 Prepare PyPI release roadmap and release gates
- #294 Prepare conda-forge release roadmap and feedstock onboarding

## Operating Principles

- Start with contract inventory and tests. Do not mix broad behavior changes into
  the first PR for an issue unless the fix is narrow and low-risk.
- Mark intentional metadata loss explicitly. The distinction between
  value-oriented exports and metadata-preserving paths should be visible in docs
  and tests.
- Keep PRs aligned to module ownership. Avoid PRs that span foundation types,
  numerical algorithms, GUI, and release docs at the same time.
- Treat unit, axis, metadata, statistical interpretation, and physics-facing
  default changes as review-sensitive.
- Prefer current behavior baselines over aspirational behavior. If a current
  behavior is undesirable but relied upon, document it first and open a
  behavior-change follow-up.
- Keep release tasks blocked until release-critical audit decisions are closed
  or explicitly deferred as known limitations.
- Current release policy update (2026-04-29): #278 and #284 are
  release-blocking until runtime corrections pass CI and receive human
  physics/statistics review. They are not accepted as first-release known
  limitations.

## Recommended Burn-Down Waves

### Wave 1: Foundation Contracts

**Issues:** #291 -> #269 -> #270 / #271 -> #289 / #292

**Purpose:** Establish the low-level contracts that downstream public APIs,
plotting, I/O, analysis, and release documentation will depend on.

**Suggested PR order:**

1. #291: Inventory and test `gwexpy.types`, axis helpers, metadata containers,
   and creator/validation helpers.
2. #269: Harden SeriesMatrix construction, slicing, math, and axis invariants.
3. #270: Define family-specific `to_matrix()` behavior after the SeriesMatrix
   invariant decisions are clear.
4. #271: Add round-trip tests that distinguish metadata-preserving persistence
   from value-only exports.
5. #289: Audit collection ordering, key, map/delegation, and container-return
   contracts.
6. #292: Audit standalone `FrequencySeries`, `Spectrogram`, and
   `BifrequencyMap` class behavior.

**Completion criteria:**

- Foundation behavior has regression tests or documented intentional gaps.
- Matrix and collection metadata policies are explicit enough for downstream
  plot, analysis, and I/O work.
- Known behavior changes are split into follow-up PRs when they are not
  required to close the audit issue.

#### Wave 1 Batch 1 Completion Report

Status as of 2026-04-27: the first autonomous Wave 1 batch has been completed
and merged. All PR heads were green in GitHub CI before merge.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #291 | #295 `[AGENT:validation] Add types foundation contract audit coverage` | 2026-04-27 11:32 JST | `ae31d88` | Added low-level types, axis, metadata-container, and `as_series` contract coverage. Runtime behavior unchanged. Known `MetaDataMatrix` CSV blank channel/unit xfail remains documented. |
| #270 | #296 `[AGENT:docs] Define to_matrix family contracts` | 2026-04-27 11:32 JST | `1f59c6e` | Documented and tested family-specific `to_matrix()` contracts for `TimeSeries`, `FrequencySeries`, and `Spectrogram`. Also classified the GWOSC segment fetch test as network-only so non-network CI gates stay deterministic. Runtime behavior unchanged. |
| #269 | #298 `[AGENT:validation] Enforce SeriesMatrix matmul xindex equality` | 2026-04-27 11:33 JST | `acf87e7` | Hardened `SeriesMatrix.__matmul__` so same sample length is no longer enough when sample-axis coordinates differ. Near-equal numeric axes use `rtol=1e-9, atol=0.0`; one-sided missing axes still raise. Physics-sensitive behavior was reviewed before merge. This closed the matrix-multiplication slice, not every #269 follow-up. |
| #271 | #297 `[AGENT:validation] Preserve SeriesMatrix round-trip metadata` | 2026-04-27 12:19 JST | `5eab91a` | Preserved `SeriesMatrix` metadata through pickle, added metadata round-trip tests, filtered non-picklable `attrs` entries per active pickle protocol, and preserved subclass pickle reducers such as `SpectrogramMatrix.__reduce__`. |

Net outcome:

- The first four foundation issues targeted for overnight execution are closed
  through merged PRs.
- Contract docs now distinguish metadata-preserving persistence from value-only
  exports.
- `SeriesMatrix` math now enforces sample-axis coordinate compatibility instead
  of only sample-axis length.
- Pickle round trips preserve stable matrix metadata while avoiding runtime
  callback/logger objects in `attrs` from breaking persistence.
- Wave 1 Batch 1 left #289 and #292 as the remaining Lane A targets; those
  were closed by the Batch 2 follow-up below.

##### SeriesMatrix xindex tolerance residual

#269 remains open for the post-#298 tolerance decision: the merged matmul slice
prevents silent multiplication on mismatched sample axes, while a later deep
dive identified that exact `xindex` equality can reject physically acceptable
sub-sample timestamp jitter. No runtime tolerance policy changes in this
roadmap update. Any move from exact equality to a tolerance such as a
sample-period-relative `np.allclose` policy needs physics review before code or
contract tests make that behavior normative.

#### Wave 1 Batch 2 Completion Report

Status as of 2026-04-27: the second autonomous Wave 1 batch has been completed
and merged. All PR heads were green in GitHub CI before merge.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #289 | #300 `[AGENT:validation] Audit collection API contracts` | 2026-04-27 16:29 JST | `2d35528` | Documented and tested collection ordering, delegation, mutability, metadata, and manifest contracts across shared mixins and `TimeSeries`, `FrequencySeries`, `Spectrogram`, and `Histogram` collections. Runtime behavior unchanged. Known follow-ups include FrequencySeries axis tightening, TimeSeries metadata preservation, and HistogramList scalar/stat return policy. |
| #292 | #299 `[AGENT:validation] Audit standalone series contracts` | 2026-04-27 16:29 JST | `8ec299a` | Documented and tested standalone `FrequencySeries`, `Spectrogram`, and `BifrequencyMap` contracts for construction, indexing, view/copy behavior, metadata preservation, helper return classes, rebinning, and current projection behavior. Runtime behavior unchanged. Known follow-ups include GWpy-vs-gwexpy return-class decisions, BifrequencyMap unit conversion, and channel/epoch propagation decisions. |

Net outcome:

- Wave 1 foundation issues #291, #270, #271, #289, and #292 are now closed
  through merged PRs. #269 has a merged matmul invariant slice and remains open
  for the xindex tolerance policy decision above.
- Foundation contracts now cover low-level types, SeriesMatrix invariants,
  family-specific `to_matrix()` behavior, metadata persistence, collection API
  behavior, and standalone frequency/spectrogram class behavior.
- Behavior-changing follow-ups remain intentionally deferred where they would
  alter axis validation, metadata propagation, or public return classes.
- Lane A no longer blocks Wave 2 public-surface audit work.

### Wave 2: Individual Public API Surfaces

**Issues:** #272, #279, #280, #281, #276, #290, #287

**Purpose:** Close relatively well-bounded public API surfaces once the
foundation contracts are stable.

**Suggested PR groups:**

- I/O residuals: #272
- Time and detector surfaces: #279, #281
- CLI contract: #280
- Table and segment-facing APIs: #276, #290
- Field data model: #287

**Completion criteria:**

- Each public surface has explicit construction, metadata, error, optional
  dependency, and export/import behavior where applicable.
- CLI behavior has exit-code, stdout/stderr, and help-text coverage.
- Field, table, histogram, segment, detector, and time APIs do not rely on
  implicit or undocumented passthrough behavior for public contracts.

#### Wave 2 Batch 1 Completion Report

Status as of 2026-04-27: the first Wave 2 public-surface batch has been
completed and merged. The PR head was green in GitHub CI before merge and was
reviewed for contract wording before merge.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #272 | #302 `[AGENT:docs] Finish I/O interop follow-ups` | 2026-04-27 18:58 JST | `122ad2f` | Updated I/O interop docs and contract records for audio extras, DTTXML boundaries, and WAV metadata fallback behavior. |

Net outcome:

- #272 is closed through PR #302.
- I/O residual documentation, install guidance, and public contract JSON now
  agree with current implementation behavior.
- `gwexpy[all]` is documented as a convenience bundle for declared extras, not
  as an exhaustive installer for every public interop backend.
- Public DTTXML direct I/O remains
  `TimeSeriesDict.read(..., format="xml.diaggui", products=...)`;
  frequency-domain DTTXML direct shims and registry adapters are
  implementation-only.
- The WAV metadata contract now matches the existing warning-and-skip behavior
  when `tinytag` is missing.
- Wave 2 Batch 1 left #279, #280, #281, #276, #290, and #287 for the next
  public-surface batch.

#### Wave 2 Batch 2 Completion Report

Status as of 2026-04-28: the second Wave 2 public-surface batch has been
merged. All PR heads were clean and green in GitHub checks before merge.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #281 | #317 `[AGENT:validation] Add detector proxy contract baseline` | 2026-04-28 15:14 JST | `b0443db` | Added docs/test-only detector package, channel proxy, and unit alias contract coverage. Runtime behavior unchanged. |
| #279 | #318 `[AGENT:validation] Add time conversion contract baseline` | 2026-04-28 15:14 JST | `4572590` | Added docs/test-only GPS/UTC conversion, scalar/vector return-type, timezone-policy, and interop helper contract coverage. Runtime behavior unchanged. |
| #280 | #319 `[AGENT:validation] Add CLI command contract baseline` | 2026-04-28 15:14 JST | `e10172b` | Added docs/test-only CLI placeholder, stdout/stderr, exit-code, version/help, and GWpy re-export boundary contract coverage. Runtime behavior unchanged. |
| #290 | #320 `[AGENT:validation] Add histogram and segments contract baseline` | 2026-04-28 15:14 JST | `c5aa144` | Added docs/test-only `Histogram` and `gwexpy.segments` public contract coverage for units, bin geometry, metadata, statistics, rebinning, collections, and proxy boundaries. Runtime behavior unchanged. |
| #276 | #321 `[AGENT:validation] Add SegmentTable contract baseline` | 2026-04-28 15:15 JST | `88dd783` | Added docs/test-only `SegmentTable`, `RowProxy`, `SegmentCell`, lazy payload/cache, copy/materialize, and structural plotting contract coverage. Runtime behavior unchanged. |
| #287 | #325 `[AGENT:validation] Add field algebra contract baseline` | 2026-04-28 15:15 JST | `e3ac1e9` | Added a docs/test-only field-algebra baseline plus xfail-guarded coordinate/domain mismatch behavior. This intentionally did not close #287 because broader field data-model audit and physics-facing runtime guard decisions remain. |

Net outcome:

- #276, #279, #280, #281, and #290 are closed through merged PRs.
- #287 remains open as an explicit field data-model residual. The merged
  baseline records the current aligned-arithmetic behavior and defers runtime
  coordinate/domain guards for human physics review.
- #278 is not a Wave 2 blocker. PR #313 already merged a docs/test-only noise
  baseline, but #278 is now a 0.1.1 release blocker until the PSD/ASD runtime
  correction for `transient_gaussian_noise(psd=...)` passes CI and human
  physics/statistics review.
- Wave 2 public-surface work is complete for the issue set that can be closed
  through docs/test-only audits without changing runtime field algebra.

#### Behavior Decision Queue

Status as of 2026-04-28: human review comments on PRs #314, #315, #316,
#322, and #328 identify design decisions that should not block docs/test-only
contract PRs. Resolve these before runtime behavior PRs, or before release if
they become public-contract commitments.

| Priority | Source PR | Owning Issue | Decision Area | Where To Decide | When To Decide | Review Required |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | #328 | #291 / #287 | `Array4D` dimension-dropping return classes, `AxisApiMixin.sel(method=...)`, downstream smoke-test ordering | #291 follow-up design note cross-linked to #287 | Before low-level axis or `ScalarField` runtime hardening PRs | Human review; physics/data-model review if `ScalarField` semantics change |
| 2 | #315 / #345 | #346 | `inverse_transform()` return type, `Inf` missing-data policy, `TimeSeriesMatrix.to_list()` flattening, PCA/ICA inverse fallback semantics | #346 preprocessing runtime policy follow-up | Before preprocessing return-type or metadata behavior PRs | Human API/metadata review; statistical review for missing-data policy |
| 3 | #314 | #284 | Bruco Welch DC/Nyquist scaling, `GWEXPY_BRUCO_BLOCK_BYTES` deterministic CI policy, correction priority | 2026-04-29 #278/#284 release-blocker runtime review | Before 0.1.1 release | Human physics/statistics review |
| 4 | #322 / #341 / #344 | #347 / #274 | Residual GauCh colorbar/Rayleigh coverage, ancillary plot helper public-stability policy, GUI metadata/colorbar payload integration | #347 for plot helpers, #274 for GUI metadata | Before additional plot helper API commitments or GUI metadata work | Human UX/API review; physics review if labels imply unit semantics |
| 5 | #316 | #282 | Astro pure re-export vs thin wrapper, optional `inspiral_range` CI, physical type/unit assertions | #282 follow-up design note | Before wrapper implementation or optional-deps CI expansion | Human physics/API review for wrapper behavior; CI review for optional deps |

Operating rule:

- Merge docs/test-only PRs when CI is green and review comments only identify
  follow-up design decisions.
- Do not mix the decisions above into the audit PRs.
- Each runtime PR should cite the corresponding decision record and mark human
  physics, statistics, API, or CI review where required.

### Wave 3: Numerical And Analysis Contracts

**Issues:** #273 -> #285 / #286 -> #277 / #278 / #284 / #288 / #282

**Purpose:** Stabilize numerical and statistical contracts while preserving a
clear review boundary for physics-sensitive changes.

**Suggested PR order:**

1. #273: Add broad numerical algorithm contract tests and identify behavior
   changes that need physics review.
2. #285: Audit time-frequency and space-transform axis, unit, and regularity
   contracts.
3. #286: Audit statistics, correlation, and numerical primitive contracts.
4. #277, #278, #284, #288, #282: Work through fitting, noise, analysis
   workflows, preprocessing/decomposition, and astro range as focused tracks.

**Completion criteria:**

- Reference tests exist for PSD/ASD/CSD/coherence, FFT-related axes, statistical
  bounds, correlation outputs, and optional dependency behavior.
- Physics-sensitive changes are isolated from docs/test-only contract PRs.
- #278 and #284 are complete only when the runtime corrections are merged,
  issue comments record the verification, and human physics/statistics review
  signs off on the release-facing behavior.
- Known numerical limitations are either fixed with tests or recorded as release
  notes / follow-up issues.

#### Wave 3 Batch 1 Completion Report

Status as of 2026-04-27: the first Wave 3 contract-test batch has been
completed and merged. All PR heads were green in GitHub CI before merge.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #273 | #304 `[AGENT:validation] Add Wave 3 numerical contract baseline` | 2026-04-27 19:45 JST | `cce2b5a` | Added the numerical contract baseline for TimeSeries FFT axes, matrix spectral behavior, normalization, interpolation, and fitting/error-surface defaults. Runtime behavior unchanged. Known non-second transient FFT behavior remains a strict xfail. |
| #285 | #305 `[AGENT:validation] Add transform axis contract coverage` | 2026-04-27 19:56 JST | `ad31432` | Added the first transform-axis slice for transient FFT, regular-sampling failure behavior, matrix FFT/PSD/ASD, CSD/coherence, and matrix spectrogram contracts. Runtime behavior unchanged. #285 remains open for follow-up transform slices. |
| #286 | #306 `[AGENT:validation] Add numerical primitive contract coverage` | 2026-04-27 20:07 JST | `6684e85` | Added statistical bound and numerical primitive contracts for correlation/coherence-style values, weighted means, error handling, tolerance behavior, and stable shape/unit expectations. Runtime behavior unchanged. #286 remains open for follow-up slices. |

Net outcome:

- Wave 3 now has a reusable numerical baseline before behavior-changing work.
- #285 and #286 have initial docs/test-only coverage but remain open because
  both issues intentionally require smaller follow-up slices.
- The planned follow-up slices after Batch 1 were special transforms,
  Q-transform, spectrogram cleaning, and field/space transform workflows.
  #307-#309 now cover the first three; field/space transform workflows remain
  before the issue is closed.

#### Wave 3 #285 Follow-Up Progress

Status as of 2026-04-27: #285 remains open and is being reduced through
docs/test-only transform slices.

| Slice | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| Special transforms | #307 `[AGENT:validation] Add special transform contract coverage` | 2026-04-27 20:27 JST | `4a8de43` | Added DCT, cepstrum, Laplace, STLT, CWT, EMD, and HHT contract coverage for axes, units, metadata, scaling, and optional dependency behavior. Runtime behavior unchanged. |
| Q-transform | #308 `[AGENT:validation] Add Q-transform contract coverage` | 2026-04-27 20:42 JST | `ba20d70` | Added direct TimeSeries Q-transform passthrough contracts, TimeSeriesList/Dict/Matrix container contracts, and documented current metadata loss plus the irregular-sampling gap. Runtime behavior unchanged. |
| Spectrogram cleaning | #309 `[AGENT:validation] Add spectrogram cleaning contract coverage` | 2026-04-27 | `f47717b7` | Added `Spectrogram.clean()` threshold, fill-mode, line-removal, rolling-median, combined-mask, metadata-preservation, and source-immutability contracts. Runtime behavior unchanged; gate details are recorded in `audit-manifest-285-spectrogram-cleaning.yaml`. |
| Field/space transforms | #310 `[AGENT:validation] Add field space transform contract coverage` | Not merged | N/A | Adds `ScalarField` space transform, `FieldList`/`FieldDict`, and inherited `VectorField`/`TensorField` workflow contracts for public axes, domains, wavelength units, source immutability, and current wrapper reconstruction behavior. Runtime behavior unchanged. |

Remaining #285 work after the field/space transform slice lands:

- Field-space behavior changes that require runtime decisions, including gwpy
  `name`/`channel` preservation, non-default `VectorField` basis preservation,
  and explicit `TensorField.rank` preservation across inherited wrappers.
- PyEMD-dependent HHT numerical contracts, only after optional dependency and
  physics-review expectations are explicit.

#### Wave 3 #288 Contract Closure

Status as of 2026-04-29: #288 has been reduced from an audit issue to a
runtime policy follow-up. The remaining policy decisions now live in #346.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #288 | #345 `[AGENT:validation] Close preprocessing contract residuals` | 2026-04-29 12:44 JST | `8edede55` | Added docs/test-only coverage for pipeline inverse errors, unsupported `StandardizeTransform.inverse_transform()` inputs, current non-finite behavior, matrix/list flattening, multivariate standardization restoration, and PCA/ICA inverse reconstruction contracts. Runtime behavior unchanged. |

Net outcome:

- #288 is closed as an audit/baseline issue.
- #346 tracks the remaining runtime/statistical/API decisions: preprocessing
  container persistence, non-finite policy, matrix structure preservation,
  PCA/ICA inverse fallback semantics, and ARIMA time-standard validation or
  documentation.
- Runtime PRs against #346 should remain focused and require human
  API/metadata review, plus statistical review when non-finite or decomposition
  semantics change.

### Wave 4: Plot, GUI, And Public Docs Synchronization

**Issues:** #275 -> #283 -> #274

**Purpose:** Synchronize user-visible plot, GUI, and documentation behavior after
the data and metadata contracts are clear.

**Suggested PR order:**

1. #275: Add public-doc drift guards and update stale tutorial/notebook patterns.
2. #283: Audit extended plot helpers and add headless structural assertions.
3. #274: Harden GUI and core plot metadata behavior against the settled
   metadata contracts as a post-release GUI stabilization track.

**Completion criteria:**

- Public docs no longer teach stale API patterns.
- Plot labels, colorbars, legends, title defaults, and unitless behavior have
  headless tests where stable.
- GUI payload metadata gaps are tracked as post-release stabilization work
  before any metadata-rich GUI behavior is introduced.

#### Wave 4 Progress Report

Status as of 2026-04-29: the public-doc, plot-helper baseline, and pre-PyPI
plot-helper runtime hardening slices have landed. The remaining Wave 4 work is
GUI-facing metadata/payload behavior plus residual plot-helper policy work
tracked separately in #347.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #283 | #322 `[AGENT:validation] Add plot helper contract baseline` | 2026-04-28 18:53 JST | `6bcc887` | Added docs/test-only plot-helper contracts for field plots, dashboards, pair plots, labels, colorbars, and optional plotting surfaces. Runtime behavior unchanged. |
| #275 | #323 `[AGENT:docs] Guard public tutorial colorbar mappables` | 2026-04-28 19:57 JST | `0147000` | Replaced brittle public tutorial colorbar mappable lookups with explicit mappables, restored the GPS-aware spectrogram x-axis in the GBD tutorial, and added notebook drift guards. Runtime package behavior unchanged. |
| #283 / #274 | #341 `[AGENT:plot] Harden plot helper contracts before PyPI` | 2026-04-29 10:31 JST | `a69a537` | Rejected incompatible `PairPlot` inputs instead of silently aligning/truncating, avoided empty/unitless dashboard labels, and made `plot_summary` tolerate unitless spectrogram colorbars. |
| #283 | #344 `[AGENT:plot] Harden plot helper labels and FieldPlot colorbar access` | 2026-04-29 13:39 JST | `bbb71d3` | Centralized plot label/unit formatting, accepted dynamic scientific label wording, exposed the public `FieldPlot.last_field_colorbar` API, preserved explicit `label=""`, and added an `[Unreleased]` changelog note. |

Net outcome:

- #275 remains open for residual tutorial/public-doc drift work, including
  deferred source-policy cleanup and any remaining notebook guard follow-ups.
- #283 is closed as the plot-helper audit/runtime-closure issue. #347 now tracks
  remaining plot-helper policy follow-ups such as GauCh colorbar handles,
  Rayleigh structural coverage, ancillary helper public-stability decisions,
  optional backend install hints, and visual-regression policy.
- #274 remains open for GUI payload metadata, renderer labels, and colorbar
  behavior that affects GUI-facing contracts, but it is not release-blocking for
  the first PyPI package because the GUI app is outside the first supported
  PyPI surface.

### Wave 5: Release Readiness

**Issues:** #293 -> #294

**Purpose:** Publish only after the audit wave is resolved or explicit known
limitations are accepted.

**Suggested PR order:**

1. #293: Fix release metadata checks, update install docs, build artifacts,
   verify PyPI Trusted Publishing, and publish the PyPI release.
2. #294: Prepare conda-forge staged-recipes submission from the stable source
   release, then validate the created feedstock and fresh conda install.

**Completion criteria:**

- `python -m build` and `twine check dist/*` pass.
- Release metadata is consistent across `_version.py`, `CITATION.cff`,
  `.zenodo.json`, and `CHANGELOG.md`.
- Public docs accurately distinguish source install, PyPI install, and conda
  install states.
- PyPI release is verified in a fresh environment before conda-forge onboarding.

#### Wave 5 Progress Report

Status as of 2026-05-07: the `gwexpy==0.1.2` hotfix lane is prepared but not
yet published or fresh-environment smoke-tested. The conda-forge staged-recipes
PR for #294 is open, mergeable, and CI-green for the old `0.1.1` sdist, but it
has not yet been reviewed or merged, and `conda-forge/gwexpy-feedstock` does
not exist yet.

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #293 | #326 `[AGENT:release] Harden release metadata version parsing` | 2026-04-28 20:00 JST | `f8cff53` | Hardened the release metadata checker so malformed or unparseable top-level `CITATION.cff` version metadata fails closed. Release tooling only; runtime package behavior unchanged. |
| #293 | #332 `[AGENT:release] Harden PyPI release gates` | 2026-04-28 20:42 JST | `0a710a1` | Added fail-closed required-file and release-date metadata checks, aligned current `0.1.1` release dates, removed bytecode/internal trees from release artifacts, added artifact hygiene checks, and added fresh-venv wheel smoke before PyPI publication. |
| #294 | #327 `[AGENT:release] Record conda-forge onboarding roadmap` | 2026-04-28 19:57 JST | `4bf66c0` | Added the conda-forge staged-recipes roadmap, later corrected the v1 recipe model to use `build.python.entry_points`, deferred the optional GUI entry point from the first core recipe, and added noarch recipe test guidance. Docs/test-only. |
| #293 | #351 `[AGENT:release] Fix release-blocking physics/statistics issues` | 2026-04-29 JST | `3ced86d` | Resolved the release-blocking #278/#284 physics/statistics issues before publication. |
| #293 | #352 `[AGENT:release] Exclude GUI and sample data from first PyPI artifacts` | 2026-04-29 JST | `a45c7b4` | Removed GUI, Sphinx utility, sample data, and tests from first PyPI artifacts; updated artifact hygiene checks and release docs. |
| #294 | conda-forge/staged-recipes#33169 `Add gwexpy` | Open as of 2026-04-30 | `487d1cd` in fork | The staged-recipes PR was submitted from `gwexpy==0.1.1` and must be refreshed to `gwexpy==0.1.2` before merge. Previous local `conda-smithy` / `rattler-build` and remote CI evidence applies to the old sdist only. |

Net outcome:

- #293 release execution for PyPI `0.1.2` is pending: Trusted Publishing is the
  expected release path, but `v0.1.2` has not been tagged/published and the
  fresh PyPI install smoke test has not run yet.
- #278 and #284 were resolved before the PyPI publication path proceeded.
- #274 is explicitly deferred as a post-release GUI stabilization track. The
  first PyPI package supports the Python library surface, not the experimental
  GUI app; `gwexpy.gui` is not included in first-release artifacts, no
  `gwexpy.gui` console script is installed, and no `gui` PyPI extra is
  published for the first PyPI release.
- #294 remains open for refreshing staged-recipes to `gwexpy==0.1.2`,
  conda-forge review/merge, feedstock creation, first
  package publication, and a fresh `conda install -c conda-forge gwexpy` smoke
  test. The external `conda-forge/staged-recipes` submission is already open at
  <https://github.com/conda-forge/staged-recipes/pull/33169>.
- The accepted release-gate policy is fail-closed: if `CITATION.cff` exists but
  its top-level `version` cannot be parsed by the supported subset, the release
  metadata check must fail rather than warn and continue.

## Parallel Work Lanes

### Lane A: Core Contract

Issue order: #291 -> #269 -> #270 / #271 -> #289 / #292

This lane owns low-level types, metadata, SeriesMatrix, conversion, collection,
and standalone frequency/spectrogram semantics. It should run ahead of plot,
GUI, and release work.

### Lane B: Public Surface

Issues: #272, #279, #280, #281, #276, #290, #287

This lane can run in parallel with Lane A when the target does not depend on
unsettled SeriesMatrix or metadata decisions. Keep each PR module-scoped.

### Lane C: Numerical And Analysis

Issue order: #273 -> #285 / #286 -> #277 / #278 / #284 / #288 / #282

This lane should begin with docs/test-only baselines. Runtime behavior changes
should be split and marked for human review when they affect physics or
statistical interpretation.

### Lane D: Docs And Visual Surfaces

Issue order: #275 -> #283 -> #274

This lane can start early for public-doc drift guards, but plot/GUI metadata
changes should wait for Lane A metadata decisions.

### Lane E: Release

Issue order: #293 -> #294

This lane should remain planning-only until the release-critical parts of Lanes
A-D are complete or explicitly deferred.

## PR Shape

Each issue should usually close through one of these PR types:

1. **Contract inventory PR**
   - Adds or updates developer docs and documents current behavior.
   - Adds minimal tests for current behavior where gaps are obvious.
   - Opens follow-up issues for behavior changes rather than mixing them in.
2. **Regression test PR**
   - Adds focused tests for axes, units, metadata, optional dependencies, and
     error behavior.
   - Does not change runtime behavior except narrow test-enabling fixes.
3. **Behavior hardening PR**
   - Changes runtime behavior after the contract is accepted.
   - Includes regression tests and release notes when user-visible.
   - Flags physics/statistical review when applicable.
4. **Release readiness PR**
   - Updates metadata, changelog, install docs, and release workflow checks.
   - Verifies build artifacts and fresh-environment smoke tests.

## Verification Gates

Minimum checks for most docs/test-only audit PRs:

```bash
rtk git diff --check
rtk pytest <focused test paths> -q
```

Additional checks for runtime behavior PRs:

```bash
rtk ruff check gwexpy/ tests/
rtk mypy gwexpy/
rtk pytest tests/ -q
```

Additional checks for docs-facing PRs:

```bash
rtk conda run -n gwexpy python -m sphinx -b html -D nbsphinx_execute=never docs /tmp/gwexpy-docs-html
```

Additional checks for release PRs:

```bash
rtk python scripts/check_release_metadata.py
rtk python -m build
rtk python scripts/check_release_artifacts.py dist
rtk twine check dist/*
```

## Release Blocking Policy

Before closing #293, decide for every remaining audit issue whether it is:

- **Release-blocking:** must be closed before PyPI publication.
- **Known limitation:** documented in release notes and acceptable for the
  target release.
- **Post-release follow-up:** not part of the public contract for the target
  release, but tracked with a clear owner and scope.

Current classification:

- #274 is a **post-release follow-up**. The first PyPI release excludes the
  experimental GUI app from the supported public surface, built artifacts, and
  PyPI extras metadata.

Conda-forge onboarding (#294) now targets the PyPI `0.1.2` sdist after
publication, but the open staged-recipes PR still needs a refresh from its
original `0.1.1` source. Keep #294 open until the staged-recipes PR merges,
the feedstock is created, and a fresh conda install from the `conda-forge`
channel passes.

## Success Criteria

- #269-#292 are closed, or any remaining items are explicitly classified as
  known limitations or post-release follow-ups.
- Release docs and metadata no longer contradict the actual distribution state.
- PyPI publication has a reproducible build path and fresh-environment smoke
  proof.
- Conda-forge submission has a reviewed recipe plan, an open staged-recipes PR,
  and does not promise pip extras that conda packaging cannot represent.
