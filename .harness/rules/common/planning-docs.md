# GWexpy Planning Document Rules

This rule governs `docs/developers/plans/` and all planning-related documents in the repository.
It prevents two classes of real errors observed in Claude sessions: ambiguous status expressions and unverified license claims.

## Status Vocabulary (Strict)

Use **only** the following status values. Each word carries a precise meaning:

| Status | Meaning |
|---|---|
| `planned` | Discussed and decided, but no actual files have been changed yet. |
| `in-progress` | Implementation has started; some files changed, but not fully verified. |
| `completed` | All target files changed **and** the verification command has passed. |

**Never write `completed` (実施済み) unless you can cite a concrete verification command that ran successfully.**

When the distinction is uncertain, append a qualifier:

- `(計画中)` — the change is only written in the plan
- `(実行予定)` — scheduled but not yet started

## Status Heading Template

Every plan section that tracks implementation state must include a status line:

```
Status: planned | in-progress | completed (verified: <command>)
```

Examples:

```
Status: planned
Status: in-progress
Status: completed (verified: pytest tests/io/test_dttxml_common.py -q)
```

Omitting the `verified:` field when claiming `completed` is a rule violation.

## Critique Appendix Retention

Planning documents often include an `## Appendix` section with critique notes, reviewer feedback, or records of factual errors caught during review.

- Do **not** delete or rewrite Appendix content when correcting the main body.
- Fix the main body text, then add a note such as `> 本文修正済み — 元の誤記録は Appendix を参照` to explain the discrepancy.
- Retaining error traces in the Appendix is intentional: they help future readers understand what was corrected and why.

## Review Triggers

- Adding or updating any file under `docs/developers/plans/`.
- Changing status of a task from `in-progress` to `completed`.
- Any plan that references external package licenses (see also `optional-dependencies.md`).
