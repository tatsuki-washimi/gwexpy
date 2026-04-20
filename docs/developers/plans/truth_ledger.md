# Truth Ledger (Phase 7)

Last-updated: 2026-04-20

## Purpose

This ledger records findings where the repository state and Git history show an item is already implemented, but the integrated audit report still marks it as stale, pending, or unresolved.

Phase 7 starts from this ledger so that workers can update the audit-facing documents from a shared source of truth instead of re-investigating the same evidence.

## Scope

- Included: items that are already implemented in the repo, but whose status in `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md` is stale.
- Excluded: items that are still genuinely open, or only partially improved without enough evidence to call them implemented.

## Implemented But Audit Is Stale

| ID | Audit report status | Truth status | Primary evidence | Notes |
|---|---|---|---|---|
| `15-1` | `❌未修正` | Implemented | `docs/web/en/examples/index.rst`, `docs/web/ja/examples/index.rst`, `239aac72` | `examples/index` already contains thumbnail-backed featured cards in both languages. |
| `15-3` | `要確認（ハブ導線側のため今回判定保留）` | Implemented | `docs/web/{en,ja}/index.rst`, `docs/web/{en,ja}/examples/index.rst`, `b06d3ed6` | Homepage teaser vs canonical gallery roles are already documented in both languages and also re-approved in the A0 ledger. |
| `17-17` | `❌未修正` | Implemented | `docs/_templates/layout.html`, `docs/conf.py`, `tests/docs/test_og_metadata.py`, `3e1498fa` | OGP and Twitter metadata tags are emitted from the shared layout and backed by docs tests. |
| `17-18` | `❌未修正` | Implemented | `docs/_static/external-links.js`, `docs/conf.py`, `tests/docs/test_external_links_js.py`, `1052183f` | Cross-origin HTTP(S) links are decorated with `_blank` and `rel="noopener noreferrer"` via shared JS registered in Sphinx. |
| `17-19` | `❌未修正` | Implemented | `docs/_static/custom.css`, `docs_internal/analysis/webpage/evidence/17-19_responsive_table_probe_result.md`, `6bd4445e`, `17bc1bae` | Table overflow containment and responsive probe evidence are already present; A0 ledger also marks this implemented. |
| `17-24` | `❌未修正` | Implemented | `docs/web/en/user_guide/verification_and_quality.md`, `docs/web/ja/user_guide/verification_and_quality.md`, `b06d3ed6`, `17bc1bae` | Public coverage entry points are already exposed through the verification guide and linked Codecov/README surfaces. |

## Explicit Non-Inclusions

| ID | Current truth | Reason not listed above |
|---|---|---|
| `17-21` | Needs revalidation | CI now exposes doctest execution paths, but the stronger claim "sample code is guaranteed to work on the latest version" is still not fully supported by the current public evidence. |

## Cross-Reference

- Audit report: `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md` (`reference-only` historical audit surface; update only when Phase 7 status sync is intentionally performed)
- Rebaseline ledger: `docs_internal/analysis/webpage/A0_再基準化台帳.md` (`reference-only` historical classification record)
- Phase 7 kickoff log: `docs/developers/plans/BULLETIN_BOARD.md`

## Immediate Use In Phase 7

Workers updating audit-facing documents should treat this ledger as the Phase 7 baseline:

1. Close or reword the stale audit entries listed above.
2. Keep `17-21` separate until its evidence threshold is explicitly re-approved.
3. When a status changes, update the integrated audit report and this ledger in the same change.
