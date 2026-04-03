---
name: evidence-pack
description: 作業完了時に、使用 skill / commands / tests / review / changed files をまとめる監査証跡ワークフロー。
trigger: manual
---

# Evidence Pack & Audit Manifest

作業が完了したら、以下の手順で監査証跡（Audit Manifest）を作成し、PR の説明欄に添付してください。

## When to Use
- 機能実装が完了し、PR を作成する直前
- 技術的負債の解消作業が完了した時
- 大規模なリファクタリングの後

## Manifest Generation

以下の項目を収集して整理します。

1. **変更ファイル一覧 (Changes)**
   ```bash
   git diff --name-only main
   ```

2. **実行したテスト (Testing)**
   - `pytest` の結果 (PASS/FAIL)
   - GUI テストの結果
   - 手動検証のステップ

3. **使用したツール・スキル (Skills & Tools)**
   - 例: `setup_plan`, `verify_physics`, `finalize_work`

4. **物理/技術レビュー (Reviews)**
   - `physics-reviewer` エージェントの判定
   - `exception-auditor` の監査結果

## Audit Manifest Template

PR の冒頭に以下のテンプレートをコピー＆ペーストして記入してください。

```markdown
## Audit Manifest

- **Task**: [タスク名 / Issue 番号]
- **Status**: [Completed / Blocked]
- **Files Modified**:
  - [file1]
  - [file2]
- **Verification**:
  - [x] pytest PASS
  - [x] ruff/mypy clean
  - [ ] physics review (N/A)
- **Skills Used**: [setup_plan, ...]
- **Known Gaps**: [もしあれば記述]
```

## Storage
生成された要約は PR の説明欄に記録します。詳細な実行ログ（`stdout.log` など）が必要な場合は、`docs_internal/work_logs/` に一時的に保存することを検討してください。
