# Plan Archive Mode

実装計画を正式な計画書として保存。

## Instructions

### 1. Locate Implementation Plan

現在の `implementation_plan.md` （英語）を検索・読込：

```bash
find . -name "implementation_plan.md" -type f
```

### 2. Collect Metadata

以下の情報を収集：

- **推奨モデル**: `suggest_next` で実施した場合のモデル選定理由
- **クォータ戦略**: `recover_quota` で策定したリソース管理戦略
  - Context compression の方法
  - Batching 戦略
  - Token 効率化方法

### 3. Translate & Format

英語の `implementation_plan.md` を日本語に翻訳：

- 正確性：専門用語は英語のまま保持する場合がある
- 構造：元の構造を維持
- 追加セクション：日本語版特有の説明

### 4. Add Resource Section

新セクション「使用モデルとリソース最適化」を追加：

```markdown
## 使用モデルとリソース最適化

### 推奨LLMモデル

**第一選択**: Claude Opus 4.5 (Thinking)
- 理由: 複雑な物理・数学検証が必要
- 推定トークン: ~5,000-8,000
- 実行時間: 10-20分

### リソース管理戦略

**Context Compression**:
- 前回の会話サマリーを活用
- 不要なコメントを削除

**Batching**:
- 複数の小タスクをまとめて実行

**Token 効率化**:
- 短い変数名を使用
- リポジトリサマリーの活用
```

### 5. Determine Save Path

タイムスタンプ生成（ローカル時刻）：

```bash
date +%Y%m%d_%H%M%S
```

保存先：

```
docs/developers/plans/<description>_plan_<TIMESTAMP>.md
```

例：`SkillsRefactoring_plan_20260131_150230.md`

### 6. Save & Notify

- ファイル保存
- 保存先を明示的に通知

## Document Structure

```markdown
# [Goal] 計画書

**作成日**: YYYY-MM-DD HH:MM:SS
**状態**: 計画中 / 実施中 / 完了

---

## 概要

[概要をここに記述]

## 目標

[目標をここに記述]

## フェーズ

[フェーズ分割をここに記述]

## 実装スケジュール

[スケジュールをここに記述]

## 使用モデルとリソース最適化

[上記で作成したセクション]

## リファレンス

[関連ドキュメント・リンク]
```

## Tips

- **翻訳品質**: 技術的正確性を優先
- **メタデータ**: suggest_next と recover_quota 実施後が最適
- **バージョン管理**: git で `docs/developers/plans/` をトラック
