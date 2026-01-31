# Handover Instructions for Next Model

**Task**: Agent Skills Refactoring
**Current Status**: Complete (47 → 30 skills)
**Reference Plan**: `.agent/skills/REFACTOR_PLAN.md`

---

## ✅ Completed: Agent Skills Refactoring

### Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Skills | 47 | 30 | -36% |
| Structured Skills (with reference/) | 0 | 10 | +10 |

### Phase 1: Skill Consolidation

**Batch 1** (11 → 4):
- `finalize_work` ← wrap_up, wrap_up_gwexpy, git_commit
- `run_tests` ← test_code, test_gui, test_notebooks
- `fix_errors` ← fix_mypy, fix_legacy_python_compatibility, fix_notebook
- `manage_docs` ← build_docs, sync_docs

**Batch 2** (6 → 2):
- `suggest_next` ← suggest_model, suggest_skill
- `archive_work` (updated) ← archive_plan, conversation_report

**Batch 3** (13 → 4):
- `lint_check` ← lint, check_deps
- `organize_project` ← organize, ignore
- `analyze_external` ← analyze_code, multimedia_analysis, office_document_analysis, search_web_research
- `refactor_code` ← refactor_nb, refactor_skills

**Renames/Merges** (3 → 2):
- `manage_gui` ← manage_gui_architecture (rename)
- `verify_physics` ← check_physics (rename)
- `visualize_fields` (updated) ← debug_axes merged in

### Phase 2: Domain Separation

Restructured with `reference/` directories:
- `prep_release/` → reference/versioning.md, changelog.md, build.md, testpypi.md
- `add_type/` → reference/array.md, series.md, field.md

---

## 📂 Current Skill List (30 skills)

### ワークフロー管理 (3)
`finalize_work`, `handover_session`, `prep_release`

### 開発・コーディング (5)
`add_type`, `extend_gwpy`, `fix_errors`, `manage_field_metadata`, `manage_gui`

### 品質保証・テスト (3)
`run_tests`, `lint_check`, `profile`

### 科学・物理検証 (3)
`verify_physics`, `calc_bode`, `visualize_fields`

### ドキュメント (3)
`manage_docs`, `make_notebook`, `compare_methods`

### プロジェクト管理 (9)
`setup_plan`, `suggest_next`, `estimate_effort`, `archive_work`, `collaborative_design`, `review_repo`, `learn_skill`, `list_skills`, `recover_quota`

### ユーティリティ (4)
`organize_project`, `analyze_external`, `refactor_code`, `presentation_management`

---

## 📂 Key Files

- `.agent/skills/REFACTOR_PLAN.md` (Master plan)
- `.agent/skills/README.md` (Skill catalog)
