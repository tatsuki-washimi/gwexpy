# Tutorial and Notebook API Drift Audit

Date: 2026-04-26
Issue: #250
Scope: public tutorial and reference docs only, inventory-first.

## Constraints

- Do not modify review-pending PRs #256, #257, #258, or #259.
- Keep this pass audit-first / inventory-only.
- Avoid public docs edits in this pass except for future minimal fixes proposed
  below.
- Do not re-execute notebooks or refresh notebook outputs in this pass.

## Inputs Reviewed

- `.agent/AGENTS.md`
- `docs/developers/plans/2026-04-18-docs-next-phase-implementation-plan.md`
- `docs/developers/plans/BULLETIN_BOARD.md`
- `tests/docs/test_tutorial_notebook_quality.py`
- `tests/docs/test_docs_notebooks.py`
- `scripts/check_docs_sync.py`

`.agent/AGENTS.md` refers to `.agent/docs/SKILL.md` and related local skill
files, but no `.agent/*/SKILL.md` files were present in this checkout.

## Safe Checks Run

All checks were run from an Issue #250-only worktree:
`/tmp/gwexpy-issue-250-docs-audit`.

```text
python scripts/check_docs_sync.py
```

Result: passed. The script reported three JA-only and three EN-only public docs
paths, and confirmed heading counts are synchronized for 157 paired files.

JA-only paths:

- `user_guide/gwexpy_for_gwpy_users_ja.md`
- `user_guide/gwpy_added_api_index_ja.md`
- `user_guide/tutorials/time_frequency_analysis_comparison.md`

EN-only paths:

- `user_guide/gwexpy_for_gwpy_users_en.md`
- `user_guide/gwpy_added_api_index_en.md`
- `user_guide/migration_0.1.1.md`

```text
pytest -q tests/docs/test_tutorial_notebook_quality.py
```

Result: `38 passed`.

```text
pytest -q tests/docs/test_docs_notebooks.py
```

Result: `4 skipped`. This is expected locally because notebook execution is
disabled unless `GITHUB_ACTIONS=true` or `GWEXPY_RUN_NOTEBOOK_TESTS=1`.

## Inventory Summary

Target scopes:

- `docs/web/en/user_guide/tutorials/`
- `docs/web/ja/user_guide/tutorials/`
- `docs/web/en/reference/`
- `docs/web/ja/reference/`
- `tests/docs/test_tutorial_notebook_quality.py`
- `tests/docs/test_docs_notebooks.py`

Counts:

- Tutorial target files (`.ipynb`, `.md`, `.rst`, direct tutorial dirs): 82
- Reference docs (`.md`, `.rst`, recursive reference dirs): 244
- EN notebooks in tutorial scope: 59

Reference file sets are paired between EN and JA for the checked `.md` / `.rst`
paths. Tutorial file sets are intentionally asymmetric: EN owns the notebook
files, while the JA tutorial directory mostly contains index and Markdown pages.
Many EN notebooks contain both `lang-en` and `lang-ja` cells, but the JA tutorial
index still describes local `.ipynb` download paths under
`docs/web/ja/user_guide/tutorials/`. That source policy should be clarified
before copying notebooks or changing paths.

## Internal-Only Link Audit

Search terms:

- `docs/developers`
- `docs_internal`
- `API_MAPPING.md`
- `API_MAPPING`
- `developers/plans`

Result: no hits in the public tutorial/reference target scopes.

Recommendation: add a small regression test that asserts public docs do not link
to internal-only planning or mapping surfaces.

## API Drift and Stale Pattern Candidates

### Stale mappable / plot object patterns

These notebooks still use fragile `plt.gca()` image/collection lookup patterns
or older plot-wrapper assumptions that are not covered by the current quality
test allowlist:

- `docs/web/en/user_guide/tutorials/advanced_correlation.ipynb`
  - Uses `plt.gca().get_images()[-1]` / `plt.gca().collections[-1]` for a
    colorbar mappable.
- `docs/web/en/user_guide/tutorials/case_gbd_format.ipynb`
  - Uses `plot = sg.plot()`.
  - Uses `plt.gca().get_images()[-1]` / `plt.gca().collections[-1]` for a
    colorbar mappable.
- `docs/web/en/user_guide/tutorials/time_frequency_analysis_comparison.ipynb`
  - Uses the same `plt.gca()` image/collection fallback pattern in two cells.

These should be fixed in a focused notebook API drift PR, ideally with an
extension to `tests/docs/test_tutorial_notebook_quality.py` so the pattern does
not reappear.

### Potential stale status labels

- `docs/web/en/user_guide/tutorials/field_scalar_intro_outputs.md`
  - Contains: `Status: Main outputs are embedded; three visualization methods
    remain marked as coming soon`.
  - This should be verified against the current tutorial outputs before editing.

Reference pages also contain broad stability labels such as `Stable` and
`Experimental` under fields, GUI, and CLI reference pages. These were not changed
in this pass because they require release-policy confirmation, not just text
cleanup.

## EN / JA Structure Findings

- Reference `.md` / `.rst` structure is paired for the audited file set.
- Tutorial structure differs by design: EN has executable notebooks; JA mostly
  has indexes and Markdown pages.
- The JA tutorial index presents `.ipynb` downloads as if notebook files live
  under `docs/web/ja/user_guide/tutorials/`. In this checkout, those notebook
  files are not present there.
- The current `scripts/check_docs_sync.py` intentionally skips
  `user_guide/tutorials/` and `reference/` heading synchronization checks, so it
  does not detect tutorial/reference content drift inside this issue scope.

## Notebook Execution Classification

Classification is heuristic and based on imports, file/data access, and existing
metadata. It is intended to decide test strategy, not to assert user-facing
readiness.

Summary for 59 EN notebooks:

- Smoke-testable: 26
- Optional dependency required: 15
- Network or external data required: 18
- Docs-only narrative notebooks: 0

| Notebook | Classification | Notes |
| --- | --- | --- |
| `advanced_arima.ipynb` | Optional dependency required | `analysis`, `ci-heavy` |
| `advanced_bruco.ipynb` | Network or external data required | real-data placeholder, `ci-heavy` |
| `advanced_control_basics.ipynb` | Smoke-testable | `ci-heavy` |
| `advanced_control_discretization.ipynb` | Optional dependency required | `control`, `ci-heavy` |
| `advanced_control_modeling.ipynb` | Optional dependency required | `control`, `ci-heavy` |
| `advanced_correlation.ipynb` | Optional dependency required | `analysis`, `fitting`, stale mappable pattern, `ci-heavy` |
| `advanced_coupling.ipynb` | Smoke-testable |  |
| `advanced_decomposition.ipynb` | Smoke-testable | `ci-heavy` |
| `advanced_field_analysis.ipynb` | Smoke-testable |  |
| `advanced_fitting.ipynb` | Optional dependency required | `fitting`, `ci-heavy` |
| `advanced_hht.ipynb` | Optional dependency required | `analysis` |
| `advanced_modal_analysis.ipynb` | Smoke-testable |  |
| `advanced_peak_detection.ipynb` | Smoke-testable |  |
| `advanced_peak_tracking.ipynb` | Smoke-testable | `ci-heavy` |
| `advanced_spectrogram_processing.ipynb` | Smoke-testable | `ci-heavy` |
| `case_active_damping.ipynb` | Optional dependency required | `control` |
| `case_arima_burst_search.ipynb` | Network or external data required | `analysis` |
| `case_bootstrap_gls_fitting.ipynb` | Network or external data required | `fitting`, `ci-heavy` |
| `case_bruco_advanced.ipynb` | Smoke-testable | real-data placeholder |
| `case_bruco_ica_denoising.ipynb` | Smoke-testable | real-data placeholder |
| `case_calibration_pipeline.ipynb` | Smoke-testable | real-data placeholder |
| `case_coupling_analysis.ipynb` | Network or external data required | `ci-heavy` |
| `case_dttxml_calibration.ipynb` | Network or external data required | real-data placeholder |
| `case_finesse_optics.ipynb` | Smoke-testable |  |
| `case_gbd_format.ipynb` | Network or external data required | stale mappable and plot-wrapper patterns, `ci-heavy` |
| `case_glitch_analysis.ipynb` | Optional dependency required | `control`, real-data placeholder |
| `case_hdf5_provenance.ipynb` | Smoke-testable | real-data placeholder |
| `case_lockin_detection.ipynb` | Smoke-testable | `ci-heavy` |
| `case_ml_preprocessing.ipynb` | Optional dependency required | external package, real-data placeholder |
| `case_noise_budget.ipynb` | Network or external data required |  |
| `case_physics_validation.ipynb` | Smoke-testable | real-data placeholder |
| `case_pycbc_search.ipynb` | Network or external data required | `gw`, real-data placeholder |
| `case_schumann_resonance.ipynb` | Smoke-testable | real-data placeholder |
| `case_segment_analysis.ipynb` | Network or external data required |  |
| `case_seismic_obspy.ipynb` | Network or external data required | `seismic`, real-data placeholder |
| `case_signal_extraction.ipynb` | Smoke-testable | `ci-heavy` |
| `case_transfer_function.ipynb` | Smoke-testable | `ci-heavy` |
| `case_violin_mode.ipynb` | Smoke-testable |  |
| `case_wiener_filter.ipynb` | Network or external data required | `ci-heavy` |
| `field_advanced_workflow.ipynb` | Optional dependency required | `fitting` |
| `field_scalar_intro.ipynb` | Network or external data required |  |
| `intro_fitting.ipynb` | Network or external data required | `fitting` |
| `intro_frequencyseries.ipynb` | Optional dependency required | `control`, `netcdf4` |
| `intro_histogram.ipynb` | Network or external data required | external package |
| `intro_interop.ipynb` | Network or external data required | `external`, `netcdf4`, `seismic`, `zarr`, `ci-heavy` |
| `intro_mapplotting.ipynb` | Smoke-testable |  |
| `intro_noise.ipynb` | Smoke-testable |  |
| `intro_plotting.ipynb` | Optional dependency required | `fitting`, `ci-heavy` |
| `intro_segment_table.ipynb` | Smoke-testable |  |
| `intro_spectrogram.ipynb` | Network or external data required |  |
| `intro_table.ipynb` | Network or external data required | sample file dependency |
| `intro_timeseries.ipynb` | Network or external data required | `analysis`, `external`, `fitting`, `seismic`, `ci-heavy` |
| `matrix_frequencyseries.ipynb` | Smoke-testable | `ci-heavy` |
| `matrix_spectrogram.ipynb` | Optional dependency required | external package, `ci-heavy` |
| `matrix_timeseries.ipynb` | Optional dependency required | external package, `ci-heavy` |
| `rayleigh_gauch_tutorial.ipynb` | Smoke-testable | `ci-heavy` |
| `segment_asd_pipeline.ipynb` | Smoke-testable |  |
| `segment_visualization.ipynb` | Smoke-testable |  |
| `time_frequency_analysis_comparison.ipynb` | Optional dependency required | `analysis`, stale mappable pattern |

## Test Coverage Observations

- `tests/docs/test_tutorial_notebook_quality.py` already catches several
  concrete regressions, including local path leakage, selected plotting API drift,
  selected notebook language drift, selected HHT/fitting mappable drift, and a
  fixed list of `ci-heavy` tags.
- `tests/docs/test_docs_notebooks.py` is safe locally because notebook execution
  skips unless explicitly enabled.
- Current checks do not encode a central classification of notebooks by smoke
  testability, optional dependency, external data, or narrative-only status.
- Several non-smoke notebooks are not currently covered by the fixed `ci-heavy`
  metadata list. That may be intentional, but it should be made explicit.

## Prioritized Small PR Candidates

1. Notebook plotting API drift guard
   - Fix stale `plt.gca().get_images()` / `collections[-1]` mappable patterns in
     `advanced_correlation.ipynb`, `case_gbd_format.ipynb`, and
     `time_frequency_analysis_comparison.ipynb`.
   - Add a regression assertion to `tests/docs/test_tutorial_notebook_quality.py`.

2. Notebook execution classification metadata
   - Add a small manifest or test fixture that records notebook classifications:
     smoke-testable, optional dependency required, network/external data
     required, or docs-only narrative.
   - Update tests to validate that non-smoke notebooks declare why they are not
     part of default local execution.

3. JA tutorial notebook source-policy clarification
   - Clarify whether JA pages should link to EN-owned bilingual notebooks, use
     generated JA wrappers, or receive copied notebook files.
   - Make only the minimal index wording/path fix after that policy is confirmed.

4. Public-doc internal-link guard
   - Add a targeted test that fails when public tutorial/reference docs mention
     `docs/developers/**`, `docs_internal/**`, or `API_MAPPING.md`.
   - This is currently green, so it is a low-risk prevention PR.

5. Stability/status label review
   - Verify `field_scalar_intro_outputs.md` "coming soon" text against current
     rendered outputs.
   - Review reference `Stable` / `Experimental` labels for fields, GUI, and CLI
     pages against the current release policy before editing.

