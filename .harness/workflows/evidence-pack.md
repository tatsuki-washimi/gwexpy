---
name: evidence-pack
description: 作業完了時に、使用 skill / commands / tests / review / changed files をまとめる監査証跡ワークフロー。
trigger: manual
---

# Evidence Pack & Audit Manifest

作業が完了したら、以下の手順で監査証跡（Audit Manifest）を作成し、PR の説明欄に添付してください。

## 雛形の自動生成

スクリプトを使って Audit Manifest の雛形を自動生成できます。

```bash
python scripts/generate_evidence_pack.py \
    --base main \
    --task "タスク名 / Issue 番号" \
    --tests "pytest tests/io PASS" \
    --tests "ruff check PASS" \
    --skills "setup_plan,verify_physics" \
    | tee /tmp/evidence.md
```

生成後は `/tmp/evidence.md` を開き、以下の手動項目を追記してください。

- `<!-- TODO: ... -->` プレースホルダーを実際の内容で置き換える
- **Physics Review**: `verify_physics` / `check_physics` スキルの判定結果
- **Known Gaps**: 将来の課題・制限事項

> **フォールバック**: スクリプトが実行できない場合は、下記の「Manifest Generation」手順に従って手動で作成してください。

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
