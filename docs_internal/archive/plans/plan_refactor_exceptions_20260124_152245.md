# 作業計画書: 広範な例外処理のリファクタリング

**作成日時**: 2026-01-24 15:22:45
**タスクファイル**: `docs/developers/plans/task_refactor_exceptions_20260124_152151.md`

---

## 1. 目的とゴール

リポジトリレビューで特定された `except Exception:` の使用箇所を、より具体的な例外型またはログ出力を伴う形にリファクタリングし、コードの堅牢性とデバッグ性を向上させます。

**対象ファイル**:

- `gwexpy/spectral/estimation.py` (4箇所)
- `gwexpy/analysis/bruco.py` (3箇所)
- `gwexpy/types/seriesmatrix_base.py` (1箇所)

**ゴール**:

- 各 `except Exception:` ブロックにおいて、本来捕捉すべき例外を特定する。
- 可能な限り具体的な例外型 (`ValueError`, `TypeError`, `AttributeError` など) に置き換える。
- 広範な例外捕捉が必要な場合はログ出力 (`logging.exception()` または `warnings.warn()`) を追加する。

---

## 2. 詳細ロードマップ

### フェーズ 1: コンテキスト分析と例外特定 (10分)

各ファイルの該当箇所を調査し、例外が発生しうる条件を文脈から特定します。

**gwexpy/spectral/estimation.py**:

- Line 128: `value.to("s")` の単位変換失敗 → `AttributeError`, `u.UnitConversionError`
- Line 199: `get_window(window, nperseg)` の生成失敗 → `ValueError` (無効なウィンドウ名)
- Line 239: `stride.value` や `resolution.value` へのアクセス失敗 → `AttributeError`
- Line 407: Numba JIT 関数の実行失敗 → `numba.errors.*` (非常に広範なため、ログ付き `Exception` のまま維持する可能性あり)

**gwexpy/analysis/bruco.py**:

- Line 837, 1253, 1365: 外部プロセスやファイルI/O、複雑な計算における予期しない失敗箇所を特定。

**gwexpy/types/seriesmatrix_base.py**:

- Line 406: データ変換や属性アクセスの失敗時の安全なフォールバック。

### フェーズ 2: リファクタリング実装 (15分)

特定した例外型に基づき、各箇所を修正します。

**実装方針**:

1. **具体的な例外が明確な場合**: `except (ValueError, AttributeError):` のように指定する。
2. **複数の予期しない例外が発生しうる場合**: `except Exception as e:` とし、`logging.warning(f"Unexpected error: {e}")` を追加。
3. **Numba など外部ライブラリの失敗**: ログ付きの広範な捕捉を維持し、フォールバック処理 (Pure Python版) を実行。

### フェーズ 3: テストと検証 (5分)

修正箇所に関連するテストを実行し、リグレッションがないことを確認します。

**実行コマンド**:

```bash
pytest tests/spectral/ tests/analysis/ tests/types/ -v
ruff check gwexpy/spectral/estimation.py gwexpy/analysis/bruco.py gwexpy/types/seriesmatrix_base.py
```

---

## 3. テスト・検証計画

- **ユニットテスト**: 各対象ファイルに対応するテストディレクトリを実行
- **リント**: `ruff check` で構文エラーがないことを確認
- **物理検証**: 必要に応じて `check_physics` スキルを使用 (今回は例外処理の修正が主なため不要と判断)

---

## 4. 推奨モデル、スキル、工数見積もり

### 推奨モデル

- **第一候補: Claude Sonnet 4.5 (現在使用中)**
  - **理由**: コンテキストの推論とコード修正のバランスが優れている。
- **代替案: Gemini 3 Pro (High)**
  - **理由**: 依存関係が深い場合の広範囲なコンテキスト検索が必要な際に有効。

### 推奨スキル

- `view_file`: 対象ファイルの該当箇所を詳細に確認
- `multi_replace_file_content`: 複数箇所の例外処理を一括修正
- `test_code`: 関連テストの実行と検証
- `lint`: 静的解析による品質確認

### 推定時間とクォータ

| 項目                        | 時間       | 備考                     |
| :-------------------------- | :--------- | :----------------------- |
| **合計時間**                | 25 〜 35分 |                          |
| フェーズ1: コンテキスト分析 | 10分       | 各例外ブロックの文脈読解 |
| フェーズ2: 実装             | 15分       | 8箇所の修正              |
| フェーズ3: テストと検証     | 5分        | pytest + ruff            |
| **推定クォータ消費**        | Medium     | 複数ファイルの読込と解析 |

**懸念点**:

- Numba やその他外部ライブラリの例外が再現困難なエッジケース（特定のハードウェアや環境依存）である場合、安全側に倒してログ付き `Exception` を残す判断が必要になる可能性があります。

---

## 5. 承認確認

この計画で作業を進めてよろしいでしょうか？承認いただければ、フェーズ1から順次実行します。
