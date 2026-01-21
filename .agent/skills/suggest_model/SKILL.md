---
name: suggest_model
description: 次のタスクの内容（コーディング、リサーチ、リファクタリングなど）に応じて、最適なLLMモデルを提案する
---

# Suggest AI Development Tool & Model

タスクの性質に基づき、利用可能な開発環境（Google Antigravity, OpenAI Codex, Gemini CLI, VS Code）から最適なモデルとツールの組み合わせを提案します。

## 1. 利用可能なモデルと環境 (2026年時点)

### Google Antigravity (Googlw One AI Pro plan)

* **Gemini 3 Pro (High/Low)**: 高精度・深い推論・超大規模コンテキスト。レポジトリ全体の解析や複雑な設計に最適。
* **Gemini 3 Flash**: 高速応答。小規模な修正やバッチ処理に最適。
* **Claude Sonnet 4.5 (Thinking有無)**: 推論と生成のバランスに優れた汎用モデル。「Thinking」版は複雑な論理構築に。
* **Claude Opus 4.5 (Thinking)**: 最高峰の分析力。物理数学的な厳密性や難解なバグ修正に最適。
* **GPT-OSS 120B (Medium)**: オープンソースベースの大規模モデル。

### OpenAI Codex (Team plan / CLI / IDE拡張)

* **GPT-5.2 / GPT-5.2-Codex**: 最新のエンジニアリング特化。大規模リファクタリングやツール連携に。
* **GPT-5.1-Codex-Max**: 長時間自律実行・大規模タスク持続。コンテキスト圧縮機能を備えたエンドツーエンド遂行用。
* **GPT-5.1-Codex-Mini**: 小規模・高速な自律修正用。

### Gemini CLI (無料トライアル)

* **Gemini 2.5 Pro / Flash / Flash Lite**: CLI経由の軽量タスクやクイックな調査用。

### VS Code (GitHub Copilot / Ollama)

* **GitHub Copilot**: リアルタイム補完。
* **Ollama (llama3.1:8b, qwen2.5:7b, gemma3:4b等)**: ローカル実行、機密性の高い実験、オフライン作業用。

## 2. 推奨される使い分け

1. **大規模構造変更・物理数学実装**: `Claude Opus 4.5 (Thinking)` または `Gemini 3 Pro (High)`
2. **長時間・複雑な自律エンドツーエンド遂行**: `GPT-5.1-Codex-Max`
3. **高速コーディング・ドキュメント作成**: `Gemini 3 Flash` または `Claude Sonnet 4.5`
4. **ローカルでの機密試作**: `Ollama (qwen2.5:7b / llama3.1:8b)`

## 3. 手順

1. **タスクカテゴリの特定**: 分析、実装、テスト、ドキュメントのいずれが主目的か。
2. **リソース状況の検討**: クオータ（High/Medium/Low）を考慮し、最適な環境を選択。
3. **提案**: 上記の実在するモデル名称と環境を、具体的なメリットと共に提示します。
