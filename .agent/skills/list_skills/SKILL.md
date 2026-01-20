---
name: list_skills
description: 登録されているスキル一覧をカテゴリー別に分類して表示する
---

# List Skills

このスキルは、プロジェクトで利用可能なすべてのエージェントスキルをスキャンし、目的別に分類して表示します。

## Instructions

1. **スキャン**:
    * `.agent/skills/` ディレクトリ内の全サブディレクトリを確認します。
    * 各ディレクトリの `SKILL.md` から `name` と `description` (YAML frontmatter) を読み取ります。

2. **カテゴリー分類**:
    以下のカテゴリーに従ってスキルを分類します：
    * **1. 開発・実装**: `add_type`, `extend_gwpy`, `refactor_nb`
    * **2. 解析・リサーチ**: `analyze_code`, `compare_methods`, `profile`
    * **3. 品質保証・テスト**: `lint`, `test_code`, `test_notebooks`, `test_gui`, `review_repo`, `check_deps`, `fix_notebook`, `fix_notebook_local`
    * **4. ドキュメント**: `build_docs`, `sync_docs`, `make_notebook`
    * **5. ワークフロー**: `git_commit`, `ignore`, `organize`, `prep_release`, `wrap_up`, `estimate_effort`
    * **6. サイエンス**: `check_physics`, `calc_bode`, `debug_axes`
    * **7. メタ**: `list_skills`, `suggest_skill`, `suggest_model`, `learn_skill`, `recover_quota`, `refactor_skills`

3. **表示**:
    * カテゴリーごとに見出しを作成し、各スキルをテーブル形式またはリスト形式で表示します。
    * フォーマット例:

      ### [カテゴリー名]

        | スキル名 | 説明 |
        | :--- | :--- |
        | `skill_name` | 説明文 |
