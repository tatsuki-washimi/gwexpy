# Changelog

Notable changes to the GWexpy project.

## [0.1.4] - 2026-05-14
### Added
- `to_gps()` now has opt-in `dtype=` output modes. The default remains
  GWpy-compatible, while `dtype=float` / `dtype="float"` return plain numeric
  seconds and `dtype="quantity"` returns seconds quantities for direct `.times`
  comparisons.

### Documentation
- README, docs hub pages, roadmap, troubleshooting pages, and footer links now
  point lightweight bug reports and feature requests to the public feedback
  form. Security reports remain directed to the repository security policy.

### Tests
- Added NetCDF fixture coverage requiring an explicit time coordinate (#393).
- Added GWF regression coverage for multi-channel list-source reads and padded
  gap reads with `parallel > 1`.

## [0.1.3] - 2026-05-12
### Fixed
- Fixed multi-file GWF reads for `TimeSeries` and `TimeSeriesDict` inputs.
- Fixed ndscope HDF5 auto-detection for `TimeSeriesMatrix.read()`.
- Aligned public I/O docs and contract metadata with current autodetection
  behavior.
- Preserved original frequency column values in the FrequencySeries CSV fast
  path.
- Improved zarr 3 matrix round-trip coverage and removed timeout-prone fixture
  behavior.
- Treated PyGMT installations without a loadable GMT shared library as an
  unavailable optional backend instead of failing at import time.

### Known Issues
- Some bundled NetCDF fixture paths can fail the TimeSeries reader
  time-coordinate contract (#393). Generated NetCDF round-trip coverage still
  passes; files should expose an explicit time coordinate.

## [0.1.2] - 2026-05-08
### Targeted Narrow Hotfix Scope
- Narrow compatibility fixes for GWpy4 public I/O proxy imports and GWF list/dict read behavior.
- Targeted auto-identify/read-path fixes for histogram HDF5, ATS/MTH5, audio, seismic, SegmentTable span CSV, and FrequencySeries DTT XML flows.
- Includes only a minimal #369 landing/demo import hunk for this integration track.

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
