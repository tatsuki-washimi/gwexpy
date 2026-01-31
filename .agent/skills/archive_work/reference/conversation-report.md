# Conversation Report Mode

会話全体のレポートを生成・保存。

## Instructions

### 1. Collect Context

以下の情報を収集：

- **全作業の要約**: 会話内で実施したすべてのタスク
- **達成事項**: 完了したアイテム、実装した機能
- **ブロック項目**: 未解決の問題、延期した作業
- **現在の状態**: プロジェクトの進捗状況

### 2. Generate Timestamp

ローカル時刻でタイムスタンプを生成：

```bash
date "+%Y%m%d_%H%M%S"
```

例：`20260131_150230`

### 3. Structure Report

会話内容を構造化：

- **期間**: 会話開始時刻から終了時刻
- **セッション**: 使用したモデル、セッション情報
- **タスク**: タスク一覧と進捗
- **アウトプット**: 生成されたファイル・成果物

### 4. Save Report

保存先：

```
docs/developers/reviews/conversation_report_<TIMESTAMP>.md
```

例：`conversation_report_20260131_150230.md`

### 5. Confirm Save

ユーザーに保存先を通知。

## Report Structure

```markdown
# Conversation Work Report

**Date**: YYYY-MM-DD HH:MM:SS
**Session**: [Session ID or Model]
**Duration**: HH:MM (estimated)

---

## Summary

[会話の概要を1-3文で記述]

## Accomplishments

### Completed Tasks

- ✅ Task 1: [Description]
- ✅ Task 2: [Description]
- ✅ Task 3: [Description]

### Files Created/Modified

```
.agent/skills/suggest_next/
├── SKILL.md
└── reference/
    ├── models.md
    └── skills.md

.agent/skills/archive_work/
├── SKILL.md (updated)
└── reference/
    ├── work-report.md
    ├── plan-archive.md
    └── conversation-report.md
```

### Key Achievements

- [Achievement 1]
- [Achievement 2]
- [Achievement 3]

## Current Status

### In Progress

- ⏳ [Task in progress]

### Blocked / Deferred

- ⚠️ [Blocked task]: [Reason]
- 📋 [Deferred task]: [Reason]

## Project State

### Changes Summary

- **Files**: [Number] created, [Number] modified
- **Skills**: [Number] created, [Number] updated
- **Tests**: [Pass/Fail status]
- **Build**: [Build status]

### Next Steps

[推奨される次のアクション]

## References

- Parent task: [Parent task link]
- Related docs: [Related documents]
- Issue tracker: [Links to related issues]

---

*Generated automatically by `archive_work --conversation` skill*
```

## Best Practices

### Content Guidelines

- **正確性**: 実際に実施した作業のみを記述
- **完全性**: すべての達成事項をカバー
- **簡潔性**: 冗長な説明は避ける

### File Management

- **バージョン管理**: `docs/developers/reviews/` をgitで追跡
- **保存形式**: Markdown（テキストベース）
- **命名**: タイムスタンプを含めた統一形式

### Use Cases

1. **セッション終了時**: 会話全体のログを記録
2. **長期プロジェクト**: 複数セッションの進捗追跡
3. **ハンドオーバー**: 次のセッション/モデルへの引継ぎ情報

## Integration with Other Skills

- **`finalize_work --full`**: work-report の後に使用
- **`archive_work --plan`**: 計画アーカイブ（並行実行可）
- **`handover_session`**: 会話ログを次セッションへ引継ぎ
