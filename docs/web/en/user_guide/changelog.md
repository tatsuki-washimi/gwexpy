# Changelog

Notable changes to the GWexpy project.

## [0.1.2] - 2026-05-08
### Targeted Narrow Hotfix Scope
- Narrow compatibility fixes for GWpy4 public I/O proxy imports and GWF list/dict read behavior.
- Targeted auto-identify/read-path fixes for histogram HDF5, ATS/MTH5, audio, seismic, SegmentTable span CSV, and FrequencySeries DTT XML flows.
- Includes only a minimal #369 landing/demo import hunk for this integration track.

### Publication Status
- `v0.1.2` metadata and release notes are finalized, but publication is not complete yet.
- Git tag creation, PyPI publication, Zenodo publication, fresh release-smoke reruns, and conda-forge refresh remain deferred release-day follow-up work.

## [0.1.1] - 2026-04-28
### Added
- Added public contract baselines for `SegmentTable`, histogram/segments helpers, detector/time conversion surfaces, CLI behavior, and plot-helper documentation/tests.
- Added release-gate checks for metadata consistency, artifact hygiene, and fresh-environment wheel smoke before PyPI publication.

### Changed
- Clarified installation guidance around optional extras, source installs, and future release-channel transitions.

### Known Limitations And Follow-Ups
- PyPI publication is still a human-controlled final step for issue #293. Public install docs intentionally remain GitHub/source-based until the first PyPI release is actually published; switch them to `pip install gwexpy` only after publication and post-publish smoke succeed.
- `conda-forge` packaging is not published yet. Issue #294 remains the follow-up for the staged-recipes submission and fresh conda-environment smoke tests.
- Open numerical and analysis audit follow-ups remain for noise contracts (#278), astro range assumptions and unit handling (#282), Bruco/coupling/response workflow semantics (#284), and preprocessing/decomposition/forecasting contracts (#288). Current docs/test baselines record present behavior, but these surfaces still have deferred policy decisions.
- GUI and visual-surface follow-ups remain open for payload metadata, labels, colorbars, plot-helper semantics, and residual public-doc drift (#274, #275, #283). The GUI should still be treated as experimental.
- Local validation follow-up #335 tracks an intermittent one-process `pytest tests/ -q` exit 139. Split-suite validation passed and remains the current evidence base, but the single-process crash is not yet explained.

## [0.1.0] - 2026-04-08
### Added
- Redesigned documentation site (Task 1 & 2).
- Stabilized `SeriesMatrix` classes.
- Enhanced multi-dimensional data support via `ScalarField`.

### Fixed
- Added `CITATION.cff` for easier citation.
- Reorganized dependencies and improved OS-specific installation guides.

## Past Versions

For more details, please visit [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases).
