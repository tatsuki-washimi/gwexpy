# Agent Skills 整備計画

**作成日**: 2026-01-31  
**ステータス**: Phase 1 完了 / Phase 2 完了

---

## 概要

gwexpy の Agent Skills を整備し、トークン効率とメンテナンス性を向上させる。

### 目標

- スキル数: 47 → 20-25 (約50%削減)
- 起動時トークン: ~4,700 → ~2,000 (約57%削減)
- Progressive Disclosure の徹底

---

## Phase 1: スキル統合

### 1.1 ワークフロー系の統合

**統合前**:

- `wrap_up` (26行)
- `wrap_up_gwexpy` (37行)
- `git_commit` (39行)

**統合後**: `finalize_work` (1スキル)

- 作業完了時の一連のフロー（検証、テスト、リント、ドキュメント更新、コミット）を一括実行
- モード選択: `--quick` (git_commit相当), `--full` (wrap_up_gwexpy相当)

### 1.2 テスト系の統合

**統合前**:

- `test_code` (36行)
- `test_gui` (91行)
- `test_notebooks` (26行)

**統合後**: `run_tests` (1スキル)

- pytest, GUI テスト, ノートブック実行を統合
- ドメイン分離: `reference/gui.md`, `reference/notebooks.md` に詳細を移動

### 1.3 エラー修正系の統合

**統合前**:

- `fix_mypy` (静的解析エラー)
- `fix_legacy_python_compatibility` (Python 3.9互換性)
- `fix_notebook` (ノートブックエラー)

**統合後**: `fix_errors` (1スキル)

- エラータイプ別のガイダンスを提供
- ドメイン分離: `reference/mypy.md`, `reference/python39.md`, `reference/notebooks.md`

### 1.4 ドキュメント系の統合

**統合前**:

- `build_docs` (Sphinx ビルド)
- `sync_docs` (docstring 同期)

**統合後**: `manage_docs` (1スキル)

- ビルドと同期を統合

### 1.5 提案系の統合

**統合前**:

- `suggest_model` (モデル提案)
- `suggest_skill` (スキル提案)

**統合後**: `suggest_next` (1スキル)

- 次のアクション（モデル・スキル）を提案

---

## Phase 2: ドメイン分離

### 2.1 prep_release の分離

**現状**: 単一ファイルにすべてのリリース手順

**改善後**:

```
prep_release/
├── SKILL.md (概要とナビゲーション)
└── reference/
    ├── build.md (パッケージビルド)
    ├── testpypi.md (TestPyPI 公開)
    └── production.md (本番 PyPI 公開)
```

### 2.2 add_type の分離

**改善後**:

```
add_type/
├── SKILL.md (概要)
└── reference/
    ├── array.md (Array 型の追加)
    ├── series.md (Series 型の追加)
    └── field.md (Field 型の追加)
```

---

## Phase 3: カタログ整備

### 3.1 index.md の作成

`.agent/skills/README.md`:

- クイックスタート
- カテゴリ別一覧
- 使用頻度別ガイド

### 3.2 命名規則の統一

**パターン**: `動詞_対象物`

- `finalize_work`
- `run_tests`
- `fix_errors`
- `manage_docs`

---

## 統合後のスキル一覧（予定）

### ワークフロー管理 (3)

| 新スキル名         | 説明                                   | 統合元                              |
| ------------------ | -------------------------------------- | ----------------------------------- |
| `finalize_work`    | 作業完了時の検証・整理・コミット       | wrap_up, wrap_up_gwexpy, git_commit |
| `prep_release`     | リリース準備（パッケージビルド・公開） | -                                   |
| `handover_session` | セッション引継ぎ                       | -                                   |

### 開発・コーディング (5)

| 新スキル名      | 説明                     | 統合元                                                  |
| --------------- | ------------------------ | ------------------------------------------------------- |
| `add_type`      | 新しい配列型の実装       | -                                                       |
| `extend_gwpy`   | GWpy クラスの安全な継承  | -                                                       |
| `fix_errors`    | 各種エラーの修正         | fix_mypy, fix_legacy_python_compatibility, fix_notebook |
| `refactor_code` | コードのリファクタリング | refactor_nb, refactor_skills                            |
| `manage_gui`    | GUI アーキテクチャの管理 | manage_gui_architecture                                 |

### 品質保証・テスト (3)

| 新スキル名     | 説明                 | 統合元                              |
| -------------- | -------------------- | ----------------------------------- |
| `run_tests`    | テストスイートの実行 | test_code, test_gui, test_notebooks |
| `lint_check`   | コード品質チェック   | lint, check_deps                    |
| `profile_code` | パフォーマンス分析   | profile                             |

### 科学・物理検証 (3)

| 新スキル名         | 説明                     | 統合元        |
| ------------------ | ------------------------ | ------------- |
| `verify_physics`   | 物理・数学の正当性検証   | check_physics |
| `calc_bode`        | Bode Plot 計算           | -             |
| `visualize_fields` | フィールドデータの可視化 | -             |

### ドキュメント (3)

| 新スキル名        | 説明                           | 統合元                |
| ----------------- | ------------------------------ | --------------------- |
| `manage_docs`     | ドキュメントの管理             | build_docs, sync_docs |
| `make_notebook`   | チュートリアルノートブック作成 | -                     |
| `compare_methods` | 手法比較ドキュメント作成       | -                     |

### プロジェクト管理 (6)

| 新スキル名        | 説明               | 統合元                                          |
| ----------------- | ------------------ | ----------------------------------------------- |
| `setup_plan`      | 作業計画の策定     | -                                               |
| `suggest_next`    | 次のアクション提案 | suggest_model, suggest_skill                    |
| `estimate_effort` | 工数見積もり       | -                                               |
| `archive_work`    | 作業報告書の作成   | archive_work, archive_plan, conversation_report |
| `review_repo`     | リポジトリレビュー | -                                               |
| `list_skills`     | スキル一覧表示     | -                                               |

### ユーティリティ (3)

| 新スキル名         | 説明                           | 統合元                                                                           |
| ------------------ | ------------------------------ | -------------------------------------------------------------------------------- |
| `organize_project` | プロジェクト構造の整理         | organize, ignore                                                                 |
| `analyze_external` | 外部コード・ドキュメントの分析 | analyze_code, multimedia_analysis, office_document_analysis, search_web_research |
| `manage_metadata`  | メタデータ・フィールドの管理   | manage_field_metadata                                                            |

---

## 削除予定スキル

以下のスキルは統合後に削除:

- `wrap_up` → `finalize_work` に統合
- `wrap_up_gwexpy` → `finalize_work` に統合
- `git_commit` → `finalize_work` に統合
- `test_code` → `run_tests` に統合
- `test_gui` → `run_tests` に統合
- `test_notebooks` → `run_tests` に統合
- `fix_mypy` → `fix_errors` に統合
- `fix_legacy_python_compatibility` → `fix_errors` に統合
- `fix_notebook` → `fix_errors` に統合
- `build_docs` → `manage_docs` に統合
- `sync_docs` → `manage_docs` に統合
- `suggest_model` → `suggest_next` に統合
- `suggest_skill` → `suggest_next` に統合
- `refactor_nb` → `refactor_code` に統合
- `ignore` → `organize_project` に統合
- `analyze_code` → `analyze_external` に統合
- `multimedia_analysis` → `analyze_external` に統合
- `office_document_analysis` → `analyze_external` に統合
- `search_web_research` → `analyze_external` に統合
- `archive_plan` → `archive_work` に統合
- `conversation_report` → `archive_work` に統合
- `check_deps` → `lint_check` に統合
- `debug_axes` → `visualize_fields` に統合
- `collaborative_design` → 維持または `setup_plan` に統合
- `learn_skill` → 維持
- `recover_quota` → 維持または削除検討

---

## 実装順序

1. **finalize_work** の作成（wrap_up系の統合）
2. **run_tests** の作成（test系の統合）
3. **fix_errors** の作成（fix系の統合）
4. **manage_docs** の作成（docs系の統合）
5. **suggest_next** の作成（suggest系の統合）
6. 古いスキルの削除
7. **README.md** の作成（カタログ）
