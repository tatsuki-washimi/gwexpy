# [Goal] Numerical Hardening Plan (Scale Invariance)

**作成日**: 2026-02-03 18:06:37 (Archived)
**状態**: 完了

---

## 概要

`gwexpy` における低振幅データ（$10^{-21}$）の数値的不安定性を解消するための包括的計画。ASTによる静的解析で見つかった160件以上のリスク（Death Floats, Exception Swallowing等）に対処する。

## 目標

1. **Scale Invariance**: 入力データが $1$ でも $10^{-21}$ でも、相対的に同等の解析結果が得られること。
2. **Silent Failure の根絶**: 例外の握りつぶしや、不正なデフォルト値による「見かけ上の成功」を排除すること。

## フェーズ

1. **Phase 0 (Unsilencing)**: 例外処理の適正化（Opus担当）
2. **Phase 1 (Core)**: `gwexpy.numerics` モジュールの構築（Opus担当）
3. **Phase 2 (Algo)**: アルゴリズムごとの数値安定化（Codex担当）
4. **Phase 3 (UI)**: 表示・フォーマットの適正化（GPT-5.1/Sonnet担当）

## 実装スケジュール (実績)

- Day 1: 監査完了、計画策定、Phase 0/1 完了
- Day 1 (cont.): Phase 2/3 完了、検証テスト完了

## 使用モデルとリソース最適化

### 推奨LLMモデル

- **Audit/Planning**: Gemini 2.0 Pro (高速・長文脈)
- **Phase 0/1**: Claude Opus 4.5 (高信頼性・推論)
- **Phase 2**: GPT-5.2 Codex (数学・コーディング)
- **Phase 3**: GPT-5.1 Codex Max (大量コンテキスト・フォーマット) / Sonnet 4.5 (GUI)

### リソース管理戦略

- **Gemini CLI活用**: 検証コード生成などの定型タスクをCLIにオフロードし、メインエージェントのトークンを節約。
- **並行プロンプト**: Phase 0/1 と Phase 2/3 を並行可能なようにプロンプト設計し、開発時間を短縮。

## 参照

- [Original Plan](file:///home/washimi/work/gwexpy/docs/developers/plans/numerical_hardening_plan.md)
- [Audit Report](file:///home/washimi/work/gwexpy/docs/developers/analysis/step1_2_summary.md)
