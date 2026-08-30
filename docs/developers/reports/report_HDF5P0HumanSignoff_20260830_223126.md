# HDF5 Exact Epoch P0 Human Sign-off

## Decision

HDF5 Exact Epoch P0 is approved and closed.

The approval applies only to commit
`ae79fb065ae4d5d712c59e49a2afa5c20fd815de` on branch
`test/v020-post-release-qualification`.

## Basis

- The exact-time qualification reproduces with a 0 ns error.
- The focused HDF5 suite passed 799 tests.
- The surrounding compatibility selector passed 54 tests.
- The repository qualification selector passed 2,814 tests, with 72 skipped and
  3 expected failures.
- The fifth fresh specification review reported no Critical, Important, or Minor
  findings.
- The code-quality re-review reported no Critical or Important findings.
- Changed-file static checks, repository MyPy, and the branch diff check passed.

## Accepted follow-up

The code-quality re-review recorded one non-blocking Minor gap: no tracked
public regression test injects an arbitrary exception from a custom
`warnings.warn` hook after a committed file-like write.

An isolated public-write check confirmed the committed state remains successful.
Track the regression test as normal maintenance; it does not reopen P0.

## Freeze boundary

Do not modify the frozen P0 baseline while bootstrap P1 is under development.

If a later source or test change touches this scope, rerun the P0 diff
confirmation before treating the approval as applicable.

## Out of scope

This sign-off does not approve bootstrap P1, the full v0.2.1 qualification,
release dispatch, merge, push, tagging, publication, or release.
