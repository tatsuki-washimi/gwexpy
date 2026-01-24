# 作業計画書: 型チェックカバレッジの拡大 (Phase 2: Spectrogram)

**作成日時**: 2026-01-24 18:25
**タスク**: `gwexpy.spectrogram` パッケージの MyPy 除外を解除し、型安全性を高める

---

## 1. 目的とゴール

Phase 1 に続き、`gwexpy.spectrogram` パッケージの MyPy 除外設定を解除します。`FrequencySeries` と同様に、Mixin クラスにおける `self` の型定義を明確化し、潜在的な属性エラーを未然に防ぎます。

**ゴール**:

- `gwexpy.spectrogram` パッケージ全体の MyPy チェックをパスさせる。
- Mixin クラス (`SpectrogramMatrixAnalysisMixin` 等) に Protocol を導入し、インターフェースを明示する。
- 既存テストをパスさせる。

---

## 2. 詳細ロードマップ

### フェーズ 1: Mixin クラスの型定義改善 (10分)

- `gwexpy/spectrogram/matrix_analysis.py` に `_SpectrogramMatrixLike` Protocol を定義し、`self` 型アノテーションを追加。
- `gwexpy/spectrogram/matrix_core.py` (もしあれば) も同様に対応。

### フェーズ 2: 設定変更と検証 (5分)

- `pyproject.toml` から `gwexpy.spectrogram.*` を削除。
- `mypy` を実行し、新たなエラーが発生しないか確認。

### フェーズ 3: テスト実行 (5分)

- `pytest tests/spectrogram/` を実行し、リグレッションがないことを確認。

---

## 3. テスト・検証計画

- **MyPy**: `mypy -p gwexpy.spectrogram --ignore-missing-imports --check-untyped-defs`
- **Pytest**: `pytest tests/spectrogram/`

---

## 4. 推奨モデル

- **Claudes Sonnet 4.5 (or Gemini 3 Pro)**
  - Mixin の型定義には引き続き高度なモデルを使用し、整合性を保つ。

---

## 5. 承認確認

この計画で進めてよろしいでしょうか？
