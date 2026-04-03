---
name: archive_work
description: 作業完了後にタイムスタンプをつけて報告書を作成・保存し、実装計画・会話ログも統合的に管理する
---

# Archive Work: Work Reports, Plans & Conversation Logs

作業完了後の統合的なアーカイビング・スキル。以下の3つのモードをサポートします：

1. **Work Report** (デフォルト) - 作業実績のドキュメント化
2. **Plan Archive** - 実装計画の保存
3. **Conversation Report** - 会話全体のログ記録

## Quick Usage

```bash
/archive_work                      # Work report mode (default)
/archive_work --plan              # Archive implementation plan
/archive_work --conversation      # Archive conversation log
```

## Modes

### 1. Work Report (Default)

作業完了時の実績レポートを生成します。

- **Content**: 変更ファイル、テスト実行、解決バグ、パフォーマンス改善等
- **Save Path**: `docs/developers/reports/report_<TaskName>_<TIMESTAMP>.md`
- **Metadata**: 使用モデル、実行時間

詳細：[reference/work-report.md](reference/work-report.md)

### 2. Plan Archive

実装計画（implementation_plan.md）を日本語で保存します。

- **Content**: 計画内容の翻訳、モデル選定・リソース管理戦略の追加
- **Save Path**: `docs/developers/plans/<description>_plan_<TIMESTAMP>.md`
- **Metadata**: 推奨モデル、クォータ管理戦略

詳細：[reference/plan-archive.md](reference/plan-archive.md)

### 3. Conversation Report

会話全体のレポートを生成します。

- **Content**: 会話内の全作業の要約、達成事項、ブロック項目
- **Save Path**: `docs/developers/reviews/conversation_report_<TIMESTAMP>.md`
- **Metadata**: タイムスタンプ、進捗状況

詳細：[reference/conversation-report.md](reference/conversation-report.md)

## Workflow Example

```
1. 作業実施
2. /archive_work              # 実績レポート作成
3. /archive_work --plan      # 計画を正式に保存（必要に応じて）
4. /archive_work --conversation  # 会話ログ保存（セッション終了時）
```
