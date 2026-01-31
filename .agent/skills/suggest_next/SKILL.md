---
name: suggest_next
description: 次のタスクの内容（コーディング、リサーチ、リファクタリングなど）に応じて、最適なLLMモデルとスキルを提案する
---

# Suggest Next: Models & Skills

次のアクション（最適なLLMモデルと使用すべきスキル）を提案します。

## Quick Usage

```bash
/suggest_next
```

## What It Does

タスクの性質や現在のプロジェクト状況を分析し、以下を提案します：

1. **最適なLLMモデル**（Gemini、Claude、GPT等）
2. **推奨スキル**（Agent Skills から最適な選択肢）
3. **実装理由**と利点

## How It Works

1. **タスク分類**: 主な目的（実装、分析、テスト、ドキュメント等）を判定
2. **リソース判定**: 現在のクォータやプロジェクト状況を考慮
3. **提案**: 最適なモデル名とスキルを、利点と一緒に提示

## See Also

- [Models & Environments](reference/models.md) - Available LLM models and recommended usage
- [Skills Catalog](reference/skills.md) - Complete skill list and selection guide
