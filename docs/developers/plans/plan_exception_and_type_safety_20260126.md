# 実装計画: 例外処理の厳格化と型安全性の向上 (Refactor Exception Handling & Enhance Type Safety)

この計画書は、`gwexpy` コードベースにおける広域例外処理の見直しと、静的解析（MyPy）における型無視（`type: ignore`）の削減を目的としています。

- ステータス: 提案中
- 最終更新: 2026-01-26
- 担当: Antigravity

## 1. 目的とゴール

### 例外処理の最適化

- 予期しないエラーを「黙殺（pass）」せず、適切なログ出力または特定例外の捕捉に置き換える。
- 致命的なパースエラーやスレッドエラーを可視化し、デバッグ性を向上させる。

### 型安全性の向上

- `type: ignore` コメントを削減し、プロパティ定義やキャスト、ジェネリクスを用いて型整合性を解決する。
- 属性未定義（`attr-defined`）のエラーを、クラス構造の見直しにより解消する。
- `pyproject.toml` での MyPy 除外ディレクトリを削減する。

## 2. 詳細ロードマップ

### Phase 1: 例外処理のリファクタリング

#### Step 1.1: GUI/NDS 関連の修正

- `gwexpy/gui/nds/util.py:gps_now`: `except Exception:` を `ImportError` に限定し、フォールバック発生を `logger.info` で記録。
- `gwexpy/gui/nds/cache.py:online_stop`:
  - 重複定義の削除。
  - シグナル切断時の `except Exception: pass` を、予期せぬエラーの場合のみ `logger.debug` または `logger.warning` で記録。

#### Step 1.2: I/O 関連の修正

- `gwexpy/io/dttxml_common.py:extract_xml_channels`: `except Exception:` を廃止。`xml.etree.ElementTree.ParseError` 等を想定。エラー時は `warnings.warn` 出力、または重要な場合は例外を送出。

#### Step 1.3: 全体的な広域例外の洗い出しと修正

- `grep` を用いて GUI 以外のモジュールで `except Exception:` を検索。
- 特定のライブラリ（`nds2`, `astropy`, `scipy`）に依存する箇所での例外を適切に限定。

### Phase 2: 型安全性の向上と MyPy 修正

#### Step 2.1: 属性定義の追加 (`attr-defined` 解消)

- `TimeSeriesMatrixCoreMixin` 等で、ベースクラスや Mixin に `_dx`, `meta`, `shape` などの属性・プロパティを適切に宣言（または `cast` の活用範囲を整理）。

#### Step 2.2: super() 呼び出しの型不整合 (`safe-super` 解消)

- `_spectral_fourier.py` におけるオーバライド時の `type: ignore[safe-super]` を、`Protocol` または明示的な型アノテーションを用いた `super()` 経由の呼び出し形式へ変更。

#### Step 2.3: `TimeSeriesMatrix` クラス定義の整理

- 複雑な Mixin 継承構造を見直し、MyPy が認識可能な順序・形式に整理することでクラス定義トップの `type: ignore` を削減。

#### Step 2.4: `pyproject.toml` 設定の更新

- MyPy の `exclude` リストから `spectrogram/` 等を順次取り除き、エラーが出ないことを確認。

## 3. 検証・テスト計画

- **単体テスト**: `pytest tests/gui` および `tests/types` を実行し、リファクタリングによる壊れがないか確認。
- **静的解析**: `ruff` および `mypy --package gwexpy` を実行し、エラー数・警告数が削減されていることを確認。
- **結合試験**: 実際に DTT XML の読み込みや NDS 接続（モック）を行い、例外ログが正しく出力されるか確認。

## 4. 推奨モデル・スキル・工数見積もり

### 推奨モデル

- **実装・修正**: `Claude Sonnet 4.5` (論理的な型整合性の構築に優れるため)
- **全体スキャン・計画実行**: `Gemini 3 Flash`

### 推奨スキル

- `fix_mypy`: 型エラー解決パターンの適用
- `lint`: Ruff/MyPy による即時検証
- `test_code`: 動作確認
- `check_physics`: 信号処理メソッド変更時の数学的整合性確認

### 工数見積もり

| 内容               | 予定時間 | トークン消費 (想定) |
| :----------------- | :------- | :------------------ |
| Phase 1 (例外処理) | 4h       | Medium              |
| Phase 2 (型安全性) | 8h       | High                |
| 検証・最終調整     | 2h       | Low                 |
| **合計**           | **14h**  | **--**              |

---

_この計画書を承認いただけますか？承認後、Phase 1 より作業を開始します。_
