---
name: phase1_scale_invariance
description: "(deprecated) 数値スケール妥当性の検証・修正。numeric-scale-checker agent に統合されました。"
---

# Phase 1 Scale Invariance — Deprecated

この skill は **numeric-scale-checker agent** に統合されました。

GW strain スケール（~1e-21）に対する数値定数（`eps/tol/atol/rtol`）の妥当性検証・スケール不変性テストは以下を使用してください:

- **canonical**: `.harness/agents/numeric-scale-checker.md`

## 移行手順

以前このスキルを呼び出していた箇所を `numeric-scale-checker` agent に置き換えてください。
agent はこの skill の全チェックリスト・修正パターン・出力フォーマットを引き継いでいます。
