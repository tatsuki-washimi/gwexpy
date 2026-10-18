# ADR-2026-10-18: GUI applications are external to GWexpy

**Status:** Accepted repository-ownership boundary (Issue [#645](https://github.com/tatsuki-washimi/gwexpy/issues/645)); implementation and cross-repository validation tracked in separate PRs.

## Context

GWexpy is an analysis library. Previously, a prototype application under
`gwexpy/gui/` was retained in source checkouts despite exclusion from wheels.
That created duplicate ownership of application code, tests and Qt tooling.
The `pyaggui` extraction started from
[`61fc5dc7453ae7165dc711f13951625552eca88d`](https://github.com/tatsuki-washimi/gwexpy/commit/61fc5dc7453ae7165dc711f13951625552eca88d)
on **2026-07-11**; a second, independent application, **GWexpy Studio**, is
developed in another repository.

## Decision

- GWexpy only owns the analytical Python library, scientific types, I/O,
  numerical algorithms, and programmatic plotting such as `gwexpy.plot`.
- GWexpy contains and distributes **no GUI application**, Qt-specific test
  harness, GUI-only dependencies, or legacy `gwexpy.gui` module.
- `pyaggui` and GWexpy Studio each own their own application code,
  tests, documentation, dependency constraints and release processes.
- GUI applications may depend on **public** GWexpy APIs. GWexpy must not
  depend on those applications, their private APIs, or Qt-only components.
- No source re-homing inside GWexpy (e.g. `legacy_gui/`). The removed
  implementation remains reconstructible in Git history; DTT/GUI technical
  reports are explicitly labelled historical instead of silently deleted.

This supersedes the earlier decision to keep the GUI as an experimental,
development-only feature inside the GWexpy source tree.

## Migration and guardrails

Legacy GUI entry points are removed, not replaced with compatibility stubs.
The EN/JA user guides preserve the inbound URLs but point to external
projects; GUI autodoc pages are removed. The
`scripts/check_repository_boundaries.py` PR Fast gate detects reintroduction
of the original source and test trees, GUI-only extras, import statements
and obsolete commands in live GUI migration documentation.

## Cross-repository handover

All 32 tracked application files, 21 GUI tests and two design documents
had corresponding files in the `pyaggui` tree when audited against
GWexpy `10311d8dcebd8938522a4e07d78fb8dfa73b3145` and pyaggui
`cd715f72ed9ce75dd25fe0743bf3dd8f05c3601b`. The old end-to-end
GUI smoke test corresponds to `pyaggui/tests/gui/test_gui_smoke.py`.
The branch [pyaggui PR #1](https://github.com/tatsuki-washimi/pyaggui/pull/1)
ports **post-extraction** changes: #765-C1 NumPy normalization with five tests,
SDB format mapping, and FFL documentation. The original `tests/nds/`
Qt-worker live probes were specific to the old GUI; they are retired from
GWexpy in favor of a **new, non-GUI NDS backend connectivity/discovery test**.
No claim is made that every obsolete GUI test has identical coverage in the
independent application.

**Handover gate:** pyaggui PR #1 must be independently reviewed and merged
(or an equivalent documented transfer accepted) before this removal PR is
merged to `main`. The GWexpy change is independently subject to lint, typing,
all required CI tests, package build and published-docs checks.

The GUI projects and GWexpy have independent issue/release processes.
Deletion here is not authorization to publish or label any version v0.2.6.
