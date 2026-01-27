# 残存作業リストと統合計画 (Remaining Tasks & Integrated Plan)

これまでの作業（例外処理の一部修正、CI最適化の一部実施）を踏まえ、残っている作業を整理し、今後の実行計画を提示します。

- 最終更新: 2026-01-26
- 担当: Antigravity

## 1. 完了済み/進行中のタスク (Status Report)

### ✅ 一部完了: CI最適化 & テスト修正

- `qtbot.waitForWindowShown` → `qtbot.waitExposed` への置換が主要なテストファイルで完了。
- `pyproject.toml` への警告抑制設定（scitokens, dateparser, numpy等）が追加済み。
- _検証待ち_: 実際のCI環境での「完全なグリーン」状態の確認。

### ⏳ 未着手: 例外処理と型安全性 (plan_exception_and_type_safety_20260126.md)

- `gwexpy/gui/nds/util.py` 等への例外処理修正は未反映。
- 型安全性の向上（`type: ignore` 削減、`SeriesMatrix` の型整理）は手つかずの状態。

## 2. 残存作業リスト (Task List)

### A. 型安全性とコード品質 (優先度: 高)

1.  **Mixinの型定義強化**: `TimeSeriesMatrixCoreMixin` 等での未定義属性（`_dx`, `meta`）の解消。
2.  **`super()` 呼び出しの適正化**: GWpyオーバライド箇所での `safe-super` 無視コメントの削減。
3.  **MyPy設定の厳格化**: コンフィグ除外リストから修正済みモジュールを削除し、チェックを有効化。
4.  **例外処理の修正**: `gps_now`, `extract_xml_channels`, `online_stop` の例外ハンドリング改善。

### B. テスト環境の最終調整 (優先度: 中)

1.  **GUI系テストの安定化**: 描画待ち時間やリトライ処理の微調整（Flaky対策）。
2.  **カバレッジ分析**: クリティカルパスのカバレッジ測定と、不足箇所の特定。
3.  **CI動作確認**: すべての変更を統合した状態でのCIパス確認。

## 3. 統合実行計画

既存の2つの計画書を統合的に進める以下の手順を推奨します。

### Step 1: 型安全性と例外処理の実装 (担当: Antigravity / Claude Sonnet 4.5)

- **ターゲット**: `gwexpy/types/` および `gwexpy/gui/nds/`
- **アクション**:
  - `plan_exception_and_type_safety_20260126.md` の Phase 1 & 2 を実行。
  - コードベースの静的解析エラー（MyPy）を削減。

### Step 2: テストの実行とカバレッジ計測 (担当: Antigravity / Gemini 3 Flash)

- **ターゲット**: 全体 (`tests/`)
- **アクション**:
  - `pytest --cov` を実行し、Step 1 の変更が既存機能を壊していないか確認。
  - `logs/` を確認し、新たな警告が出ていないかチェック。

### Step 3: 最終仕上げ (担当: Antigravity)

- **アクション**:
  - `archive_work` を実行して変更履歴を記録。
  - `pyproject.toml` の最終調整（バージョン番号や不要な依存の削除など）。

## 4. 次のアクション

ユーザー様の承認が得られ次第、**Step 1（型安全性と例外処理）** の作業を開始します。

_この統合計画で進めてよろしいでしょうか？_
