# Documentation Audit and Reorganization Plan

This plan outlines the systematic audit, categorization, and reorganization of documents within the `docs_internal/` directories. Each file has been analyzed based on its content rather than its name.

## Proposed Categorization

Based on the audit, the documents have been classified into four categories:

*   **Category 1 (Active/Incomplete)**: Remaining as is for ongoing work.
*   **Category 2 (Completed/Archivable)**: Finished tasks or historical logs to be moved to `docs_internal/archive/`.
*   **Category 3 (Useful Assets)**: Technical specifications, research reports, or theoretical notes to be moved to `docs_internal/tech_notes/` for permanent reference.
*   **Category 4 (Incorrect/Irrelevant)**: To be deleted (None identified).

### docs_internal/plans/

| File | Category | Proposed Action |
|---|---|---|
| `2026-03-26-extra-lib-review-fixes.md` | 2 | Move to `archive/plans/` |
| `GWexpy_restructure_plan_20260203.md` | 2 | Move to `archive/plans/` |
| `NumericalHardening_plan_20260203_180637.md` | 2 | Move to `archive/plans/` |
| `SegmentTable.md` | 3 | Move to `tech_notes/specs/` |
| `SegmentTable_docs_plan.md` | 2 | Move to `archive/plans/` |
| `SegmentTable_implementation_plan.md` | 2 | Move to `archive/plans/` |
| `extra_lib.md` | 3 | Move to `tech_notes/research/` |
| `fix-dependences.md` | 2 | Move to `archive/plans/` |
| `handover_to_claude.md` | 2 | Move to `archive/plans/` |
| `io_improvements.md` | 2 | Move to `archive/plans/` |
| `io_improvements_implementation.md` | 2 | Move to `archive/plans/` |
| `numerical_hardening_plan.md` | 2 | Move to `archive/plans/` |
| `plan_TutorialVerification_20260223.md` | 2 | Move to `archive/plans/` |
| `plan_documentation_improvements_20260223_135347.md` | 2 | Move to `archive/plans/` |
| `plan_gwpy_usability_alignment_20260303.md` | 2 | Move to `archive/plans/` |
| `plan_integrated_work_post_codex_20260127_181313.md` | 2 | Move to `archive/plans/` |
| `plan_io_interop_gwpy_20260202.md` | 2 | Move to `archive/plans/` |
| `plan_release_v0.1.0b1_20260130.md` | 2 | Move to `archive/plans/` |
| `plan_timeseries_mock_leak_fix_20260328.md` | 2 | Move to `archive/plans/` |
| `rayleigh_statistics_plan.md` | 2 | Move to `archive/plans/` | ※内容未確認。Rayleigh 統計機能が未実装の場合は Category 1 への変更を検討してください。 |
| `step_11_integration_testing_gemini.md` | 2 | Move to `archive/plans/` |

> **Note**: `implementation_plan_doc_audit_20260328.md`（本ファイル自身）はこのリストに含まれていません。進行中の作業ドキュメントとして Category 1 (Active) 扱いとし、移動・アーカイブは行いません。

### docs_internal/prompts/

| File | Category | Proposed Action |
|---|---|---|
| All files | 2 | Move to `archive/prompts/` |

### docs_internal/reports/

| File | Category | Proposed Action |
|---|---|---|
| `GWpy4_deep-research-report.md` | 3 | Move to `tech_notes/research/` |
| `api_unification_spectral_params_20260206.md` | 2 | Move to `archive/reports/` |
| `deepclean_preprocessing_proposal_20260204.md` | 2 | Move to `archive/reports/` |
| `documentation_audit_api_unification_20260206.md` | 2 | Move to `archive/reports/` |
| `hht_implementation_notes_20260204.md` | 3 | Move to `tech_notes/implementation/` |
| `hht_implementation_review_20260204.md` | 2 | Move to `archive/reports/` |
| `hht_research_summary_20260204.md` | 3 | Move to `tech_notes/research/` |
| `noise_budget_technical_report_20260204.md` | 3 | Move to `tech_notes/reports/` |
| `noise_subtraction_technical_report_20260204.md` | 3 | Move to `tech_notes/reports/` |
| `notebook_outputs_sync_report_20260214.md` | 2 | Move to `archive/reports/` |
| `report_DocsRestructure_20260203_150914.md` | 2 | Move to `archive/reports/` |
| `report_Documentation_Improvements_20260223.md` | 2 | Move to `archive/reports/` |
| `report_GWpy4_Migration_20260223.md` | 2 | Move to `archive/reports/` |
| `report_GWpy4_Migration_Test_Fixes_20260223.md` | 2 | Move to `archive/reports/` |
| `report_NumericalHardening_20260203_180637.md` | 2 | Move to `archive/reports/` |
| `report_Release_Prep_0.1.0b2_20260223.md` | 2 | Move to `archive/reports/` |
| `report_Ruff_and_minepy_Fixes_20260223.md` | 2 | Move to `archive/reports/` |
| `report_integrated_completion_20260127.md` | 2 | Move to `archive/reports/` |
| `stat_info_technical_report_20260204.md` | 3 | Move to `tech_notes/reports/` |
| `stlt_cleanup_recommendation_20260204.md` | 2 | Move to `archive/reports/` |
| `stlt_implementation_notes_20260204.md` | 3 | Move to `tech_notes/implementation/` |
| `stlt_technical_report_20260204.md` | 3 | Move to `tech_notes/reports/` |
| `test_coverage_improvement_20260328.md` | 2 | Move to `archive/reports/` |
| `test_coverage_report.md` | 3 | Move to `tech_notes/reports/` |
| `walkthrough_ja.md` | 2 | Move to `archive/reports/` |
| `work_report_phase3_20260131.md` | 2 | Move to `archive/reports/` |

### docs_internal/reviews/

| File | Category | Proposed Action |
|---|---|---|
| All files | 2 | Move to `archive/reviews/` |

## User Review Required

> [!IMPORTANT]
> - All plans and reports seem completed, so no "Category 1 (Active)" items were identified in these folders.
> - ~~Category 3 (Useful Assets) contains research on library interoperability, mathematical/physical theory (HHT, STL-T, Noise Budget), and class specifications. Are you comfortable moving these to `docs_internal/tech_notes/`?~~
>   **→ 確認済み・同意**: Category 3 (Useful Assets) を `docs_internal/tech_notes/` へ移動することが承認されました（2026-03-28）。

## Proposed Execution

1. Prepare target directories:
   - **既存（確認のみ）**: `docs_internal/archive/plans/`, `docs_internal/archive/reports/`, `docs_internal/archive/reviews/` はすでに存在します。新規作成は不要です。
   - **新規作成が必要**: `docs_internal/archive/prompts/`, `docs_internal/tech_notes/research/`, `docs_internal/tech_notes/specs/`, `docs_internal/tech_notes/implementation/`, `docs_internal/tech_notes/reports/`
2. Move files as categorized.
3. Verify directory structure.

## Open Questions

- Should we keep the original directory names within `tech_notes` (e.g., `tech_notes/plans/`, `tech_notes/reports/`) or organize by topic (as proposed above)?

## Verification Plan

### Manual Verification
- Confirm all files are moved correctly.
- Ensure no broken internal documentation links (though internal documents are rarely cross-linked).
