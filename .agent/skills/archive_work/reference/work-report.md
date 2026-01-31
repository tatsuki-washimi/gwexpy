# Work Report Mode

作業完了時の実績レポート生成。

## Instructions

### 1. Create Work Report

タスク実施内容の詳細サマリーを作成：

- **変更ファイル**: 追加・修正したファイルとその概要
- **テスト実行**: 実行したテスト・結果
- **解決内容**: 修正したバグ、実装した機能、リファクタリング等
- **パフォーマンス**: 改善メトリクス（速度向上、トークン削減等）

### 2. Collect Metadata

以下の情報を収集：

- **使用モデル**: Claude Opus 4.5、Gemini 3 Flash等
- **実行時間**: 推定/実際の時間
- **リソース消費**: トークン使用量、処理時間等

### 3. Save Report

```
docs/developers/reports/report_<TaskName>_<TIMESTAMP>.md
```

例：`report_AgentSkillsRefactoring_20260131_150230.md`

### 4. Notify User

保存先を明示的に通知。

## Report Structure (Markdown)

```markdown
# Work Report: <Task Name>

**Date**: YYYY-MM-DD HH:MM:SS
**Model(s)**: Claude Opus 4.5, Gemini 3 Flash
**Status**: ✅ Completed / ⏳ In Progress

---

## Summary

[1-3文の概要]

## Changes

### Files Added/Modified
- `path/to/file1.py`: [What changed]
- `path/to/file2.md`: [What changed]

## Test Results

- Unit tests: ✅ All passed
- GUI tests: ✅ All passed
- Notebooks: ✅ All executed

## Resolutions

### Bugs Fixed
- [Bug 1]: [Solution]

### Features Added
- [Feature 1]: [Details]

### Refactoring
- [Refactor 1]: [Details]

## Performance Impact

- Token usage: Reduced by 20%
- Execution time: -15%

## Next Steps

[推奨される次のアクション]
```

## Knowledge Extraction

レポート作成時に以下も検討：

- **再利用可能なパターン**: learn_skill で新スキル化するか？
- **ユニークな設計哲学**: refactor_skills で既存スキルを更新するか？
- **注意点**: 将来の参考になるベストプラクティス

詳細は `learn_skill` / `refactor_skills` スキルを参照。
