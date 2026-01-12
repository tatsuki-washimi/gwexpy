# コードレビュー報告書

**日付:** 2026-01-07
**対象:** gwexpy プロジェクト全体 (gui/ ディレクトリを除く)

---

## 1. エグゼクティブサマリー

gwexpy プロジェクトの総合的なコードレビューを実施しました。発見された問題は主に以下のカテゴリに分類されます:

- **未使用インポート (F401):** 28件
- **未使用変数 (F841):** 4件
- **空白行のホワイトスペース (W293/W291):** 約200件
- **重複関数定義:** 2件
- **テストファイルの問題:** 1件
- **互換性問題:** 1件

**全体的な評価:** コードベースは概ね良好な状態ですが、リファクタリング過程での未使用インポートの残留と、スタイル上の軽微な問題が散見されます。

---

## 2. 発見された問題の詳細

### 2.1 未使用インポート (F401)

| ファイル | 未使用のインポート |
|---------|-------------------|
| `frequencyseries/frequencyseries.py:14` | `scipy.signal` |
| `frequencyseries/matrix_core.py:5` | `numpy` |
| `frequencyseries/matrix_core.py:7` | `gwexpy.types.seriesmatrix.SeriesMatrix` |
| `plot/plot.py:61` | `gwexpy.frequencyseries.FrequencySeriesMatrix` |
| `plot/plot.py:72` | `gwexpy.timeseries.TimeSeriesMatrix` |
| `spectrogram/matrix.py:323` | `gwexpy.types.seriesmatrix_validation._expand_key` |
| `spectrogram/matrix.py:442` | `.spectrogram.Spectrogram` |
| `spectrogram/spectrogram.py:6` | `astropy.units` |
| `spectrogram/spectrogram.py:195-196` | `SpectrogramMatrix`, `SpectrogramList`, `SpectrogramDict` |
| `timeseries/matrix.py:3` | `typing.Optional` |
| `timeseries/matrix_core.py:10` | `gwexpy.types.seriesmatrix.SeriesMatrix` |
| `timeseries/matrix_interop.py:5-6` | `TimeSeries`, `TimeSeriesDict`, `TimeSeriesList` |
| `types/metadata.py:17` | `RegularityMixin`, `InteropMixin` |
| `types/series_matrix_analysis.py:6` | `MetaDataMatrix` |
| `types/series_matrix_core.py:3` | `Optional`, `Union` |
| `types/series_matrix_indexing.py:3` | `Any` |
| `types/series_matrix_io.py:5` | `Any`, `Optional` |
| `types/seriesmatrix_base.py:6` | `gwpy.types.index.Index` |

### 2.2 未使用変数 (F841)

| ファイル | 変数名 | 行番号 |
|---------|-------|--------|
| `plot/plot.py` | `kwargs_orig` | 56 |
| `plot/plot.py` | `units_consistent` | 251 |
| `timeseries/matrix_spectral.py` | `df` | 34 |

### 2.3 重複/冗長関数定義

**問題1:** `to_pandas_frequencyseries` と `from_pandas_frequencyseries`

- `interop/frequency.py` (lines 9-32)
- `interop/pandas_.py` (lines 139-182)

両ファイルに同名の関数が存在します。`interop/__init__.py` は `frequency.py` から優先的にインポートしており、`pandas_.py` の実装は使用されていません。

### 2.4 テストファイルの問題

**ファイル:** `tests/utils/test_enum.py`

**問題:** `gwpy.utils.tests.test_enum` から `test_numpy_type_enum` をインポートしようとしていますが、この関数は現在のgwpyバージョンに存在しません。

```python
# 現在のコード (エラー)
from gwpy.utils.tests.test_enum import (
    TestNumpyTypeEnum,
    test_numpy_type_enum,  # 存在しない
)
```

### 2.5 空白行のホワイトスペース (W293/W291)

以下のファイルに多数の空白行ホワイトスペースが存在:

- `analysis/bruco.py` - 約60箇所
- `analysis/coupling.py` - 約5箇所
- `frequencyseries/frequencyseries.py` - 2箇所
- `frequencyseries/matrix.py` - 2箇所
- `plot/defaults.py` - 約10箇所
- `plot/plot.py` - 数箇所

### 2.6 非効率な実装

**問題なし** - 主要なアルゴリズムは効率的に実装されています。

### 2.7 矛盾・バグ

**現時点で致命的なバグは発見されていません。**

---

## 3. テスト結果サマリー

```
=== テスト実行結果 ===
- Passed: 1253
- Failed: 49 (主にGWF I/O関連 - 外部依存)
- Errors: 27 (主にLAL/PyCBC依存テスト)
- Skipped: 41
- XFailed: 3
- XPassed: 1
```

失敗やエラーの大部分は、オプショナルな外部依存パッケージ(LAL, PyCBC, framel等)に関連しており、gwexpyコア機能には影響ありません。

---

## 4. 優先度分類

### 高優先度 (即時修正必要)

1. `tests/utils/test_enum.py` のインポートエラー

### 中優先度 (品質向上)

2. 未使用インポートの削除
2. 未使用変数の削除/活用
3. 重複関数定義の整理

### 低優先度 (スタイル改善)

5. 空白行ホワイトスペースの修正 (ruff --fix で自動修正可能)

---

## 5. 推奨事項

1. **ruff --fix の活用:** 空白行のホワイトスペースは `ruff check --fix` で自動修正可能
2. **CI/CD への統合:** pre-commit フックでruff checkを強制することを推奨
3. **定期的なコードレビュー:** 月次でのコードレビューを推奨
