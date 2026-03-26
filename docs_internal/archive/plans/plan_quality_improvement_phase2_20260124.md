# 実装計画：コード品質の更なる向上（MyPy拡充・精密型定義・例外監査）

## 1. 目的とゴール

本計画は、`gwexpy` のコードベースを「プロフェッショナルな静的解析と堅牢なエラー処理」を備えた状態に引き上げることを目的とします。

### ゴール

- **MyPy 拡充**: `analysis`, `io`, `fields` モジュールの `ignore_errors = true` を解除し、型安全にする。
- **精密型定義**: `xindex` や `metadata` に `Protocol` や `TypeAlias` を導入し、`Any` を排除する。
- **例外監査**: リポジトリ全体に残る「広すぎる例外捕捉（`except Exception:`）」を特定し、適切なロギングまたは具体的例外への置換を行う。

## 2. 詳細ロードマップ (フェーズ別)

### フェーズ 1: 精密型定義の基盤構築 (Precision Typing)

- `gwexpy/types/typing.py` (新規) を作成し、共有プロトコルを定義する。
  - `XIndex`: `__getitem__`, `__len__`, `unit` 等を持つプロトコル。
  - `ArrayLike`: numpy 配列互換のインタフェース。
- `SeriesMatrix` 関連の Mixin 群、および `ScalarField` にこれらの型を適用する。
- `metadata` の `MetaDataDict` に型ヒントを追加し、要素が `MetaData` オブジェクトであることを明示する。

### フェーズ 2: MyPy カバー範囲の拡大 (Coverage expansion)

- **Step 1: `fields` モジュール**: `gwexpy/fields/*.py` の型チェックを有効化。`ScalarField` の 4D 構造維持ロジックに型ヒントを付与。
- **Step 2: `analysis` モジュール**: `gwexpy/analysis/*.py` のチェックを有効化。Scipy/Numpy 連携部分の型不整合を解消。
- **Step 3: `io` モジュール**: `gwexpy/io/*.py` および各クラスの `io` Mixin を対象とする。オプション依存関係（`h5py`, `zarr` 等）の型宣言を整理。

### フェーズ 3: 広域例外捕捉の監査と修正 (Exception Audit)

- `grep` を用いて `except Exception:` および `except:` を全検索。
- 以下の順序で監査・修正を実施：
  - **コア信号処理**: `timeseries/_signal.py` 等。失敗時に `NaN` を返す場合は `logger.debug` を必須化。
  - **データローダー**: `gui/loaders/` 等。ファイル形式エラーなどの具体的例外を捕捉するように修正。
  - **GUI スレッド**: `gui/nds/` 等。クラッシュ防止のための広域捕捉は残しつつ、`logger.exception` でトレースを確実に記録。

## 3. テスト・検証計画

- **静的解析**: 各フェーズ完了ごとに `mypy .` および `ruff check .` がパスすることを確認。
- **既存テストの維持**: `pytest tests/types` をはじめとする全テストスイートのパスを確認（退行バグの防止）。
- **ロギング実証**: 重大な例外捕捉箇所に対し、意図的にエラーを起こしてログが出力されるか（ユニットテスト内で）確認。

## 4. 推奨モデル、スキル、工数見積もり

### 推奨モデル

- **プロトコル設計・全体整合性**: `Claude Opus 4.5 (Thinking)` または `Gemini 3 Pro (High)`
- **一括リファクタリング・MyPy修正**: `GPT-5.2-Codex` (最新エンジニアリングへの適応力)
- **軽量な修正・ドキュメント**: `Gemini 3 Flash`

### 推奨スキル

- `fix_mypy`: 静的解析エラーの効率的解消
- `check_physics`: 物理単位と型の整合性検証
- `manage_field_metadata`: `fields` モジュールの 4D 構造管理
- `learn_skill`: 新しく設計した型パターンを記録

### 工数見積もり

- **推定合計時間**: 約 90 - 120 分 (AI 自己完結時間換算では 20 - 30 分)
- **推定クォータ消費**: **High** (広範なファイルへの `multi_replace_file_content` 実行を伴うため)
- **ブレイクダウン**:
  - フェーズ 1: 30分 (高度な設計判断が必要)
  - フェーズ 2: 45分 (ファイル数が多い)
  - フェーズ 3: 30分 (検索と個別判断の繰り返し)

## 5. 懸念事項

- **外部ライブラリの型定義**: `gwpy`, `astropy`, `scipy` の一部に型定義が不完全な箇所があり、`stub` の作成や `type: ignore` が不可避となる可能性がある。
- **循環参照**: 型定義を共通化する際、インポートの循環が発生しやすいため、`from __future__ import annotations` と `TYPE_CHECKING` を慎重に使い分ける必要がある。
