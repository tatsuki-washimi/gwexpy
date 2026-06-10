# Parallel Worktrees Rules

Applies when multiple agents or developers share the same repository simultaneously
(parallel worktree sessions, multi-agent tmux setups, concurrent CI runs, etc.).

---

## (a) Not-Alone Principle

**You are not alone in the codebase; do not revert, overwrite, or clean up changes
you did not make.**

- If you encounter unexpected changes (unstaged edits, unknown commits, foreign
  branches), **stop, report, and escalate** — do not resolve them silently.
- Treat unrecognised changes as owned by someone else until confirmed otherwise.

## (b) Write-Scope Declaration

Before starting any work, explicitly declare your write scope:

```
Write-scope: src/gwexpy/io/formats/  gwexpy/timeseries.py
```

- Keep write scopes **disjoint** across parallel workers.
- If two workers need the same file, **serialise** the work — do not run concurrently.
- Scope must be declared in your opening message or plan document.

## (c) Read-Only Roles

Agents assigned as **reviewer / auditor / explorer** are read-only by default.

- Read-only roles must not write, commit, or delete files unless explicitly
  reassigned by the coordinator or user.
- If you are unsure of your role, assume read-only and ask before editing.

## (d) Version-Branch Mapping

- **Never edit release branches or tagged versions** (e.g., `v0.1.2` is frozen).
- All development work happens on `main` or a dedicated feature branch.
- **Before starting work**, confirm the current branch is within your assigned scope:

```bash
git branch --show-current   # must match your task's target branch
git tag --points-at HEAD     # must be empty (not a frozen tag)
```

## (e) Conflict Escalation

When you detect an out-of-scope change or merge conflict:

1. **Stop work immediately** — do not attempt to fix or revert.
2. Report to the coordinator with: affected files, nature of the conflict, your
   current branch and last commit SHA.
3. Wait for coordination before resuming.

### Completion Report (required)

Every worker must include the following in their final report:

- [ ] **Touched files**: list of files created, modified, or deleted
- [ ] **Verification performed**: tests run, lint checks, manual inspection
- [ ] **Unresolved conflicts**: list any open issues or skipped items
- [ ] **Assumptions**: any scope assumptions that affected the work

---

## Checklist (before starting work)

- [ ] Write-scope declared and confirmed disjoint from other workers
- [ ] Current branch matches target (not a release tag or frozen branch)
- [ ] Role confirmed (worker vs. read-only reviewer)
- [ ] No unexpected unstaged changes in write-scope files

See also: `.harness/rules/common/harness-editing.md` for absolute-path and
portability constraints.
