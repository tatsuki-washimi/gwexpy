---
name: task-routing
description: 依頼内容から、使うべき workflow / agent / skill / rule を最初に案内する軽量ワークフロー。
trigger: manual
---

# Task Routing Guide

GWexpy のハーネス（`.harness/`）が提供する機能を最大限に活用するため、タスクの性質に応じて適切な入口を選択してください。

## Routing Table

| タスクの種類 | 推奨ワークフロー / エージェント | 関連スキル |
| :--- | :--- | :--- |
| **新規機能開発** | `feature-development` | `/setup_plan`, `/finalize_work` |
| **バグ修正・リファクタ** | `feature-development` | `/fix_errors`, `/refactor_code` |
| **数値ロジック追加** | `numerical-audit` | `/verify_physics` |
| **技術的負債の解消** | `technical-debt` | `exception-auditor` |
| **ドキュメント・チュートリアル更新** | `docs-sync` | `/make_notebook` |
| **リリース準備** | `release` | `/prep_release`, `metadata-checker` |
| **依存関係・パッケージ変更** | `optional-deps-reviewer` | `pyproject.toml` 監査 |

## Primary Entry Points

1. **まず /session-start を実行**: 環境が正しいか確認します。
2. **次に /setup_plan を実行**: タスクを分解し、`.harness/rules/` に基づいた設計を行います。
3. **作業中にエージェントを呼ぶ**: 物理検証なら `physics-reviewer`、コード品質なら `gwexpy-linter` を都度実行します。
4. **完了前に /evidence-pack を実行**: 監査証跡をまとめ、PR に添付します。

## Decision Flow

- **「何をしていいかわからない」** -> この `task-routing` を再度読み、`.harness/skills/README.md` のクイックルーティング表を見てください。
- **「物理的に正しいか不安」** -> `physics-reviewer` エージェントに相談し、`verify_physics` スキルを実行してください。
- **「リリースしても大丈夫か」** -> `release` ワークフローの `Pre-Release Gates` を全てチェックしてください。

## Examples

- **「新しいフィルタ関数を追加したい」**
  - 入口: `feature-development`
  - 必須チェック: `numerical-audit` (数値スケール妥当性)
- **「古いコードを最新の API に合わせたい」**
  - 入口: `technical-debt`
  - 補助: `catalog_legacy` (移行元コードの検索)
