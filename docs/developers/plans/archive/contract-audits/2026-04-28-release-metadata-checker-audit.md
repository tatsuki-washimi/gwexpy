# Release Metadata Checker Audit

Date: 2026-04-28
Issue: #293, "Prepare PyPI release roadmap and release gates"
Branch: `codex/wave4-293-release-metadata-parser`
Mode: focused release-gate hardening; no release publication.

## Scope For This Slice

This slice fixes and tests the local release metadata checker used by the
larger #293 release-readiness roadmap. It is limited to:

- parsing the top-level `version:` scalar from `CITATION.cff`;
- preserving the existing dependency-free checker implementation;
- adding focused script tests for the CFF parser and a metadata consistency
  smoke path;
- recording audit evidence for this release-gate check.

This slice does not change package version metadata, create release artifacts,
publish to PyPI/TestPyPI, tag a release, or alter physics/runtime library
behavior.

## Bug Fixed

`scripts/check_release_metadata.py` previously parsed `CITATION.cff` with a
multiline regular expression whose capture was not bounded to a single line.
For an unquoted CFF value such as:

```yaml
version: 0.1.1
date-released: 2026-04-28
url: https://example.invalid/gwexpy
```

the captured version could include the following `date-released` and `url`
fields. That caused false metadata mismatches even when the actual CFF version
matched `gwexpy/_version.py`.

The parser now scans `CITATION.cff` line by line and returns only the scalar
value from a top-level `version:` line. It handles unquoted values, single- and
double-quoted values, and inline comments after unquoted values without adding
PyYAML or any other dependency.

## Current Checker Behavior

- `gwexpy/_version.py` remains the required source for the detected version.
- Missing `CITATION.cff` still emits a warning and returns `None`.
- Missing `.zenodo.json` still emits a warning and returns `None`.
- Missing `CHANGELOG.md` still emits a warning and does not fail the checker.
- When present, `CITATION.cff`, `.zenodo.json`, and `CHANGELOG.md` are checked
  against the detected Python version using the existing consistency policy.
- The CFF parser only considers top-level `version:` lines and ignores nested
  fields such as `preferred-citation: ... version:`.

## Validation

Focused tests cover:

- the regression where `version`, `date-released`, and `url` appear together;
- single-quoted and double-quoted CFF versions;
- unquoted CFF versions with inline comments;
- nested/non-top-level `version:` fields;
- the existing missing-file warning path;
- a full `main()` smoke check with temporary `_version.py`, `CITATION.cff`,
  `.zenodo.json`, and `CHANGELOG.md` files.

Commands used for this slice are recorded in
`audit-manifest-293-release-metadata.yaml`.

## Deferred Full Release Roadmap Items

The following #293 items remain outside this focused parser slice:

- decide and apply the target release version and release date;
- synchronize `gwexpy/_version.py`, `CITATION.cff`, `.zenodo.json`, and
  `CHANGELOG.md` for the actual release;
- convert `[Unreleased]` changelog entries into a dated release section;
- update public installation docs and README distribution messaging;
- build wheel/sdist artifacts and run `twine check`;
- configure or verify PyPI Trusted Publishing;
- perform TestPyPI or equivalent dry-run publication checks;
- tag a release, create a GitHub release, or publish to PyPI;
- perform fresh-environment installation smoke tests from published artifacts;
- start the conda-forge onboarding work tracked after #293.
