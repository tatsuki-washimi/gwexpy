# Work Report: Phase 1 qualification handover

**Date**: 2026-08-29 17:17:04 JST
**Status**: ⏳ In progress — implementation is paused for a new-session handover
**Branch / worktree**: `test/v020-post-release-qualification` / `/home/washimi/.config/superpowers/worktrees/gwexpy/v020-post-release-qualification`
**Publication state**: No tag, push, GitHub Release, PyPI/TestPyPI upload, conda-forge update, Zenodo action, or post-release dispatch has been performed.

## Summary

Phase 1 post-release qualification infrastructure and a substantial P0 HDF5 exact-time implementation have been committed in an isolated worktree.  The original v0.2.0 P0 defect is reproducibly repaired on its direct round-trip path, but independent final review found further P0 data-integrity and compatibility gaps.  Therefore P0 is **not approved**, candidate qualification must not begin, and the approved public-release stop boundary remains in force.

## Committed work

The branch is 21 commits ahead of `origin/main`.  The qualification foundation includes the immutable v0.2.0 claims ledger, strict artifact/evidence harness, public-contract suite, and a manual post-release workflow gate.  The major commits are:

- `872d826f5` through `ccd824fc3`: published-release qualification harness, report aggregation hardening, claim contracts, and workflow support.
- `166e9c419`: 19-cell public-contract manifest and contract tests.
- `45b80aa02`: post-release qualification workflow.
- `1e3f7c2d6`, `d42bf74c0`, `105730003`, `c6ddcb89c`, `3bca8d7b9`, `dfbd76415`, `4db5ffc83`: P0 HDF5 exact epoch/sidecar implementation and defensive transaction/path/link tests.

P0 production and test changes are limited to:

- `gwexpy/timeseries/io/hdf5.py`
- `gwexpy/timeseries/io/__init__.py`
- `tests/timeseries/test_hdf5_exact_t0.py`

## Verified before the final review

The following were run on the then-current P0 head.  They demonstrate direct-path progress but do **not** override the final review findings below.

- Initial v0.2.0 reproduction: exact epoch `1234567890123456789` was read back as `1234567890123456955` (+166 ns) and without the exact sidecar authority.
- P0 regression suite: `117 passed`.
- Qualification exact node and existing HDF5 compatibility selectors: `46 passed` in the combined focused run.
- `ruff check gwexpy tests`: passed.
- `mypy gwexpy`: passed for 395 source files.
- Earlier I/O gates: `io-contract` 1321 passed / 24 skipped; `io-conformance` 71 passed / 7 skipped; `io-optional` 71 passed / 2 skipped.

`ruff check .` has one pre-existing, unchanged failure at `docs_redesign/conf.py:242` (`D103` from commit `52a31a44dd`).  Do not treat it as a P0 source change without an explicit scope decision.

## Final review: release-blocking P0 findings

The final independent spec and quality reviews are **not approved**.  These are all reproducible locally and must be handled before P0 can be declared fixed.

1. **Critical — sidecar is not bound to dataset identity.**
   - Reading an exact dataset through group hard-link, dataset hard-link, or internal soft-link aliases can lose exact authority and reintroduce +166 ns quantization.
   - A GWpy-only `append=True, overwrite=True` replacement can leave an old sidecar attached to new data: returned values/native `t0` belong to the replacement, while `t0_gps_ns` remains the previous exact epoch.
   - Next implementation should bind sidecar entries to a private dataset token/marker and native epoch fingerprint, resolve aliases by token, and fail closed for absent, stale, or mutually inconsistent entries.

2. **Important — reload breaks registered HDF5 I/O.**
   - Reloading `gwexpy.timeseries.io.hdf5` resets module-local base handlers while the registry marker causes early return; the next write raises a registration `RuntimeError`.
   - Preserve/recover the native handlers without recursive wrapping and add reload/import-order round-trip tests.

3. **Important — safe GWpy path forms regress.**
   - Inline operations reject absolute HDF5 paths and valid UTF-8 byte paths that GWpy accepts.
   - Canonicalize only the sidecar key; preserve native path semantics for pathname, File, Group, and file-like read/write paths.

4. **Important — transaction cleanup failure can break exception atomicity.**
   - If post-success rollback-link cleanup fails, the wrapper raises but leaves new dataset/sidecar visible and recovery state behind.
   - Cleanup must be included in transactional success, otherwise rollback or raise an explicit unrecoverable rollback error with recovery location.

5. **Important — current preflight/file-like strategy has non-bounded RAM overhead.**
   - The wrapper writes the complete dataset once in a core-driver preflight and then writes it again.  In a separate-process 64 MB float64 probe, peak RSS was roughly 527,496 KiB versus GWpy native 312,400 KiB.
   - File-like transactions additionally duplicate entire HDF5 buffers.  Remove data-writing preflight where the existing rollback transaction suffices; use chunked disk-backed temporary storage for file-like work.

The existing path/link hardening is still valuable: raw `..`, invalid UTF-8, `ExternalLink`, stale sidecar, and group-alias write paths now have regression coverage.  It must be retained while the dataset-identity design is corrected.

## P1 bootstrap status

No P1 production change has started.  A read-only design audit found that the minimal P1 implementation must cover more than root `__init__`:

- lazy root exports while preserving `__all__`, `dir`, type identity, and `coupling`;
- lazy `interop/__init__.py`, otherwise importing its registry eagerly registers constructors;
- separate constructor/I/O/full-bootstrap state with promotion from `register_all(False)` to `register_all()`;
- removal of eager I/O import from time/frequency packages and Spectrogram provenance registration;
- on-demand hooks for all public read/write boundaries without replacing `UnifiedReadWriteMethod` descriptors;
- an interop registry fallback that loads the owner package for known constructor names.

Historical commit `329a72978` provides a partial lazy-root shape but must not be cherry-picked blindly: it lacks current `coupling`, leaves eager interop behavior, and changes descriptor semantics.

## Candidate qualification and release metadata

Not started.  Do not update version/CITATION/CHANGELOG/release metadata, build candidate artifacts, dispatch candidate or post-release workflows, or request human sign-off until P0 and P1 are both approved.

## Handover Instructions for the Next Model

1. Work only in the existing clean worktree and retain the branch history; do not reset, squash, or amend commits.
2. Treat the final P0 findings above as the current blocking requirements.  Begin with a design review for a dataset identity marker/token plus sidecar epoch binding before editing code.  Add RED regressions for all alias forms and GWpy-only overwrite first.
3. Preserve the current fail-closed external-link/raw-path/transaction coverage.  Do not broaden scope to new formats or API changes.
4. After P0 implementation, obtain a fresh spec review and a separate code-quality review before starting P1.
5. Then implement P1 from the design checklist using clean-subprocess contract tests; update opposite eager-registration tests rather than deleting coverage.
6. Only after P0/P1, physics/maintainer sign-off, candidate metadata, candidate build, and all 19 candidate cells are green may the status become `READY FOR RELEASE — awaiting explicit publication approval`.
7. Even then, do not create/push tags, create releases, upload artifacts, update conda-forge/Zenodo, dispatch post-release qualification, or begin Phase 2 without a new explicit user authorization.

## Handover prompt

```text
Resume the Phase 1 v0.2.1 pre-publication qualification work in
/home/washimi/.config/superpowers/worktrees/gwexpy/v020-post-release-qualification
on branch test/v020-post-release-qualification. Read
docs/developers/reports/report_Phase1QualificationHandover_20260829_171704.md
first. P0 is NOT approved: fix the dataset-identity-bound HDF5 sidecar design
and the listed compatibility/atomicity/memory findings with TDD and independent
spec plus quality review. Do not perform any public release action.
```
