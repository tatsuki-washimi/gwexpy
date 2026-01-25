# 作業計画書: QA検証 (Lint & Test) - 2026/01/24

## 1. 目的とゴール

最近実施した `normalization.py` のリント修正および `VectorField`/`TensorField` の実装と修正が、既存のコードベースに悪影響を与えていないことを確認し、コード品質を担保する。

- **ゴール**:
  - `gwexpy/fields/normalization.py` のリントエラーが解消されていること。
  - 関連するテストスイート（特に normalization と fields）がすべてパスすること。

## 2. 詳細ロードマップ

### フェーズ 1: 静的解析 (Lint)

- `lint` スキルを実行し、`normalization.py` および `test_normalization.py` の警告が解消されたか確認。
- 他の関連ファイルに波及した警告がないかチェック。

### フェーズ 2: ユニットテスト実行 (Verification)

- `test_code` スキルを使用し、以下のテストを実行：
  - `tests/fields/test_normalization.py`
  - `tests/fields/test_vectorfield.py` (存在する場合)
  - `tests/fields/test_tensorfield.py` (存在する場合)
  - その他 `tests/fields/` 配下の関連テスト
- 失敗した場合は、デバッグと修正を行う。

### フェーズ 3: 作業報告とクリーンアップ

- `archive_work` スキルを使用して、検証結果をレポートとして保存。
- 修正が完了していれば Git へのコミットを検討。

## 3. テスト・検証計画

- **ツール**: `pytest` (via `test_code`), `ruff`, `mypy` (via `lint`)
- **対象ディレクトリ**: `gwexpy/fields/`, `tests/fields/`

## 4. 推奨モデル、スキル、工数見積もり

- **推奨モデル**: `Gemini 3 Flash` (高速なイテレーションに最適)
- **推奨スキル**: `lint`, `test_code`, `archive_work`
- **推定合計時間**: 10 分
- **推定クォータ消費**: Low 〜 Medium
