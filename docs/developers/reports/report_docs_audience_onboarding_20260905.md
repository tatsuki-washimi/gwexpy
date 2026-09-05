# Docs audience onboarding implementation — 2026-09-05

The implementation follows the approved audience-to-reproducible-analysis plan in `docs_redesign`, using English sources and Japanese gettext catalogues. The existing worktree was left intact; this change was developed on `docs/audience-onboarding`, based on `d433b2ecb6e938d86c30615d54c7f20fcf97cde7`.

## Delivered behavior

- The homepage and existing Getting Started URL lead to six backgrounds, including a standalone **Commissioner** route in both languages. A compact introduction explains units, timestamps, channel names, and reproducible analysis before the routes.
- Downloadable Quickstart and Commissioner scripts generate deterministic synthetic channels. The first saves an ASD figure; the second round-trips ndscope HDF5, crops a copy, compares ASD/coherence, and saves data, figures, settings, and runtime package versions. A separately labelled, synthetic DiagGUI XML time-series sample exercises the optional `dttxml` path. Virgo dataDisplay is a workflow analogy, without a claim of native format support.
- The first-analysis and Scientific Python lessons explain enough Python/metadata/spectral concepts to finish locally. SciPy and GWexpy ASD results are checked numerically. TimeSeries basics now focuses on creation, metadata, selection, and plotting, linking to advanced material. Noise-model synthesis is introduced before optional detector/seismic examples.
- I/O and interoperability catalogues live under Reference; procedure pages retain old URLs and EN/JA anchors. The GWpy guide links to current capabilities, build identity, and limitations. Developer documentation is separate from user onboarding.
- Every document and search page shows development status, imported package version, source revision, introductory target release, and limitations. `build-info.json` records build and notebook execution evidence. The released and introductory target versions are centrally defined in `release_status.json`; the target introductory release is 0.2.2.
- Case pages show audience, dependencies, synthetic-data origin, measured execution duration, package version, and the distinction between execution and physical validation.
- Public notebook execution code is prepared from 59 canonical notebooks after reconciling existing public fixes. Public English prose/gettext identities remain in `docs_redesign`; generated outputs stay outside the checkout. Preparation rejects different code-cell counts. Runtime source, Python, dependency versions, and revision identify the execution cache.
- CI checks the Markdown examples with the development package and an isolated base-dependency installation of the named introductory release. It builds EN/JA, checks entry paths and legacy anchors, regenerates the shared figure, retains failure diagnostics, and reads expected HTML/commit/figure back after production deployment.

## Notebook defects exposed and corrected

The original noise-budget failure combined projected ASDs with incompatible witness units. GWpy-compatible transfer estimates retain their numerical contract; the examples explicitly attach native output/input units before projection.

Wiener and advanced BrUCo examples now use matched mean averaging for diagonal PSD and off-diagonal CSD entries. With `Cij = <conj(X_i) X_j>`, the row filter is `(Cyx @ inv(Cxx)).conj()`. Explicit output/input units make the FFT projection compatible with the target. No extra FFT normalization factor is applied.

The ROOT histogram example now uses the public `Histogram.from_root` class method. Global warning suppression and unexpected-error swallowing were removed, including an ARIMA trial loop that previously counted failed fits as nondetections. Optional integrations retain explicit `ImportError` messages; expected short-chain MCMC autocorrelation errors remain narrowly handled.

## Physics review scope

**Human physics review is required before merging these tutorial interpretation changes.** No library runtime or public API is changed, and GWpy compatibility behavior remains intact.

Automated review found known injected Wiener gains accurate to a maximum relative error of `4.85e-12`, exact injected waveform reconstruction to `3.16e-13`, and correct epoch, cadence, length, and target unit. For the original noisy example, the 100–130 Hz reconstruction correlation was `0.99675`; band RMS fell from `6.52e-23` to `2.20e-23`. A delayed-signal test distinguishes the corrected conjugation from the old formula. Regression tests execute the actual notebook estimation cells with known gain/phase and mixed native witness units.

A quadrature budget assumes independent sources. The documentation does not require the estimated sum to stay below the measured ASD at every bin, and successful execution is not presented as independent physical validation.

## Verification

See [the audit manifest](audit_docs_audience_onboarding_20260905.json) for commands, final results, and changed files. Local diagnostics and disposable builds are under `/tmp/gwexpy-docs-onboarding/`; Japanese extraction and rendered-link evidence are under `/tmp/gwexpy-ja-integration/`.

- Ruff passed for the library, tests, modified configuration and new scripts. Mypy passed for 396 library files and the three new scripts.
- The full suite initially reported 12,867 passed, 7 failed, 307 skipped and 6 expected failures. All seven failures referenced moved documentation catalogues or the replaced hard-coded installation version. After updating those references, the final affected docs/I/O contract run passed all 208 tests, with 4 skips, including the new numerical regression tests. The full suite was not repeated as one invocation after these fixes.
- All 59 prepared notebooks executed successfully; both language builds include those successful executions. The final render reused the matching execution cache. The final post-merge EN and JA renders each emitted two existing duplicate API object-index warnings. The legacy docs strict HTML build passed.
- The rendered checker passed 19 entry pages per language, retained legacy fragments, local links, language counterparts, the shared PNG and execution metadata for all 24 case studies.
- The introductory examples passed with both the development environment and an isolated base-dependency GWexpy 0.2.2 installation. The optional XML sample passed where `dttxml` was installed; it was explicitly skipped in the base-only environment.
- Desktop (1440 px) and mobile (390 px) browser checks passed for both languages, including all six routes and the literal Commissioner name, with no horizontal overflow. Terminology, documentation synchronization and forbidden-artifact checks passed.

External linkcheck is **not clean**: 238 working links, 14 redirects, 126 unchecked targets, 1 ignored target, 7 broken targets and 13 timeouts. Existing GitLab destinations returned 403; historical GitHub release links and the GWpy citation destination returned 404; an upstream Astropy documentation link failed TLS verification; SciPy destinations timed out. Exact destinations are recorded in the manifest. These external results are separate from the successful local navigation checks and are not suppressed by the new validation code.

The main verification environment uses Python 3.11.14, GWexpy 0.2.3, GWpy 4.0.2, NumPy 1.26.4, SciPy 1.12.0, Astropy 6.1.7, Matplotlib 3.10.8, Sphinx 8.2.3, and MyST-NB 1.4.0. Clean release testing installs GWexpy 0.2.2 in an independent venv.

BLAS thread counts are set to one for docs jobs to prevent small seasonal-model fits from oversubscribing large machines. `python -m sphinx` binds Sphinx to the selected environment; a user-level `sphinx-build` executable can otherwise select a different Python.

## Integration with updated main

Main advanced to `547332db3` while the implementation was in progress. The merge preserves its API documentation presentation hooks, release records and line-ending qualification changes. In the three overlapping notebooks, the canonical/public fixes in this branch retain native-unit correctness and additionally test matching spectral averaging and complex conjugation. No runtime files differ from the updated main branch. The post-merge docs, I/O contract and affected release test run passed **305 tests**, with **4 skips**. Both language builds and the rendered entry-path checks passed again after the merge.

## Remaining external actions

Production publication requires this change to reach the main-branch Pages workflow after human review. The new post-publication readback gate must run against that deployed commit; local build success alone is not recorded as a completed deployment. The 30-second route-selection and 10–15-minute learning targets remain usability goals, not measured user-study results.

## PR review follow-up

The subsequent CI/base comparison and physics consistency review are recorded in [the PR #713 verification follow-up](pr713_validation/README.md). It distinguishes the corrected Actionlint/Matplotlib inventory failures from the additional, preexisting PR Fast failures and the pending release-review binding. Existing release approvals are unchanged.
