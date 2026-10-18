# ADR: Move GUI application ownership out of GWexpy

**Status:** Proposed (pending #645 review)
**Date:** 2026-10-18
**Issue:** [#645](https://github.com/tatsuki-washimi/gwexpy/issues/645)
**Source extraction:** `61fc5dc7453ae7165dc711f13951625552eca88d` (2026-07-11)

## Decision

GWexpy is the signal-processing and scientific-data analysis **library**.
It neither ships nor maintains an embedded GUI implementation. The historical
experimental GUI has been extracted to the standalone
[pyaggui repository](https://github.com/tatsuki-washimi/pyaggui).
**GWexpy Studio** is a different GUI application in a separate repository;
this ADR does not assume it shares code or a release process with pyaggui.

Application-specific UI, Qt workers, tests, release tooling and GUI support
belong in the corresponding application's own repository. Applications may
consume supported *public* GWexpy APIs. GWexpy must not depend on a GUI
application or on `pyaggui` itself. Any need for a reusable private library
helper must first be reviewed as a GWexpy public API change.

## Migration and safeguards

- Remove old `gwexpy/gui` and GUI-only tests/scripts/configuration.
- Preserve historical PRs, release notes, audit manifests and Git history.
- Keep short redirects at the old English/Japanese user-guide paths and the
  published docs GUI how-to location for existing inbound links.
- Remove autodoc module and API references for the absent `gwexpy.gui`.
- Enforce a repository-boundary check in the usual hygiene CI.
- Do not delete the old source until changes after the 2026-07-11 split
  have been triaged. The post-extraction corrections include the
  NumPy-truth-value fix (#795), GBD/SQLite extension mapping and FFL support
  description. They are proposed for porting in
  [pyaggui PR #2](https://github.com/tatsuki-washimi/pyaggui/pull/2).
- Prior to merging #645's deletion, confirm that pyaggui PR #2 has landed,
  its tests/CI passed, and application owners accepted the migration. The
  independent GWexpy Studio repository is not modified by this change.

## Not included

DTTXML product readers, native frame I/O, SciPy/Matplotlib plotting helpers,
core analysis modules, scientific-data compatibility and historic documents
remain GWexpy responsibilities. A filename mentioning DiagGUI or a Qt-era
record is not by itself evidence that a core module should be removed.
