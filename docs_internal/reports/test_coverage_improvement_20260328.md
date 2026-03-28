# テストカバレッジ改善レポート

**作成日**: 2026-03-28
**対象セッション**: コンテキスト圧縮を挟む2セッション（2026-03-28）

---

## 概要

本レポートは、`gwexpy` プロジェクトに対して実施したテストカバレッジ改善作業の全記録です。
未テストモジュールを中心に新規テストファイルを追加し、合計 **5,687 テスト** を収集できる状態にしました。

---

## 完了した作業

### セッション前半（コンテキスト圧縮前）

| コミット | テスト数 | 対象モジュール |
|---------|---------|--------------|
| `cfd76597` | 14 | `types/array.py` — Array クラス |
| `ef0d0f36` | 46 | `io/utils.py` (65% → 86%) |
| `956d4f98` | ~ | `timeseries/hurst.py` (67% → 100%) |
| `32bfde27` | 60 | `timeseries/decomposition.py` (68% → 95%) |
| `2cf85be3` | 40 | `signal/preprocessing/imputation.py` (69% → 99%) |
| `009dd006` | 50 | `frequencyseries/frequencyseries.py` (69% → ~84%) |
| `b61f00b0` | 56 | `spectrogram/collections.py` (66% → 71%) |
| `67f3e940` | 103 | `timeseries/pipeline.py` |
| `860ceaee` | ~ | `interop/mne_.py` (25% → 89%) |
| `4ebae203` | ~ | `interop/root_.py`, `plot/_init_helpers.py` |
| `9494299e` | ~ | `spectrogram/collections.py` (52% → 77%) |
| `9e008c71` | 112 | `fitting/core.py` (28% → 69%) |
| `a1f0779c` | ~ | `analysis/coupling.py` (20% → 28%) |
| `6956db36` | 19 | `fitting/gls.py` |
| `4c34171c` | 17 | `analysis/response.py` |
| `8f3eead5` | 57 | `analysis/bruco.py` |
| `3b86808f` | 15 | `io/dttxml_common.py` |
| `f1f1ede2` | 23 | `io/collection_dir.py` |
| `4c96ac86` | 29 | `io/hdf5_collection.py` |
| `b4ef1c7b` | ~ | `fitting/highlevel.py` |
| `63b018f0` | ~ | `analysis/coupling.py` — plot_cf テスト |
| `0e7cb3b2` | ~ | `io/pickle_compat.py` — フォールバックパス |

### セッション前半：interop モジュール

| コミット | テスト数 | 対象モジュール |
|---------|---------|--------------|
| `88984cad` | 18 | `interop/_time.py` |
| `0b4d05c8` | 15 | `interop/errors.py`, `interop/_optional.py` |
| `db6dd203` | 18 | `interop/frequency.py` |
| `eb1e5644` | 3 | `analysis/coupling.py` — CouplingResult.plot() |
| `18145677` | 12 | `interop/_registry.py` |
| `9bbdf94c` | 20 | `interop/base.py`, `interop/json_.py` |
| `cabb5499` | 14 | `interop/astropy_.py` |
| `a4219672` | ~ | `interop/pandas_.py` 拡張 |
| `77bfd6a1` | ~ | `interop/xarray_.py` 拡張 |

### セッション後半（コンテキスト圧縮後）

| コミット | テスト数 | 対象モジュール |
|---------|---------|--------------|
| `a57ea100` | 15 | `interop/sqlite_.py` |
| `9922093c` | 8 | `interop/dask_.py` |
| `60e0113b` | 7 | `interop/zarr_.py` |
| `b48ac920` | 9 | `interop/hdf5_.py` |
| `06fc9f93` | 44 | `plot/defaults.py` — determine_* ヘルパー群 |
| `19dca21a` | 11 | `interop/cupy_.py` |
| `04a660a7` | 19 | `interop/control_.py` |
| `b36991c5` | 13 | `interop/torch_dataset.py` |
| `a725514b` | 11 | `interop/quantities_.py` |
| `b00227bf` | 11 | `interop/polars_.py` |
| `95f02c81` | 12 | `interop/pydub_.py` |
| `6f1cd46e` | 12 | `interop/torch_.py`, `interop/tensorflow_.py` |
| `be2b9c18` | 8 | `interop/specutils_.py` |
| `3a4107bd` | 5 | `interop/pyspeckit_.py` |
| `7a568d42` | 16 | `noise/obspy_.py` |
| `7b56da3f` | 11 | `noise/gwinc_.py` |

---

## テスト追加のアプローチ

### オプション外部ライブラリのモック戦略

インストールされていないオプション依存（`torch`, `tensorflow`, `polars`, `cupy`, `control`, `pydub`, `specutils`, `quantities`, `polars`, `mth5` 等）は `unittest.mock.patch` + `sys.modules` でモックして、ロジックをテスト済みです。

```python
with patch.dict(sys.modules, {"torch": fake_torch_mod}):
    from gwexpy.interop.torch_ import to_torch
    result = to_torch(ts)
```

### pytest.importorskip による条件付きスキップ

実際のライブラリが必要なテスト（`dask`, `zarr`, `h5py`, `xarray` 等）は `pytest.importorskip()` を使用し、ライブラリが存在しない環境では自動スキップになります。

### matplotlib Agg バックエンド

プロット系テスト（`analysis/coupling.py` の `plot()` など）では、GUI セグフォルトを防ぐためファイル先頭に `matplotlib.use("Agg")` を必須設定しています。

---

## 既知の残存問題

### pre-existing テスト失敗（本セッションとは無関係）

| テストファイル | 失敗数 | 原因 |
|--------------|--------|------|
| `tests/interop/test_interop_obspy.py::TestFromObspy::test_trace_to_ts` | 1 | `TypeError: Unsupported conversion: Trace -> TimeSeries`（実装バグ） |
| `tests/signal/test_imputation.py::TestImpute::test_fill_value_ffill_non_nan` | 1 | 実装バグ |
| `tests/timeseries/test_matrix_analysis.py` | ~10 | `sklearn` 未インストールによる `ImportError` |
| `tests/timeseries/test_pipeline.py::TestImputeTransform::test_transform_list` | 1 | 実装バグ |

---

## 残りの作業候補

### 高優先度（実装バグ修正）

- [ ] `gwexpy/interop/obspy_.py` の `from_obspy` 実装バグ修正（`Trace -> TimeSeries` 変換未実装）
- [ ] `signal/test_imputation.py` の `ffill` バグ調査・修正

### 中優先度（カバレッジ向上）

以下のモジュールは tests ファイルが存在するが、カバレッジが低い可能性があります：

| モジュール | 備考 |
|----------|------|
| `fields/scalar.py` | 2,230行、大規模 |
| `fields/signal.py` | 1,529行、大規模 |
| `timeseries/core.py` | 中核クラス、多機能 |
| `spectrogram/matrix.py` | 行列スペクトログラム |
| `frequencyseries/matrix.py` | 行列周波数シリーズ |

### 低優先度（外部ライブラリ依存）

以下はライブラリが未インストールのため実テストが困難（モックによる基本テストは済み）：

- `interop/mth5_.py` (`mth5` 未インストール)
- `interop/netcdf4_.py` (`netCDF4` 未インストール)
- `interop/polars_.py` の `from_polars_dataframe` 詳細 (`polars` 未インストール)

---

## テスト実行コマンド

```bash
# 全テスト（GUI除く）
conda run -n gwexpy python -m pytest tests/ --ignore=tests/gui/ -q

# 特定モジュールグループ
conda run -n gwexpy python -m pytest tests/interop/ -q
conda run -n gwexpy python -m pytest tests/noise/ -q
conda run -n gwexpy python -m pytest tests/plot/ -q

# 既知の失敗を除外して実行
conda run -n gwexpy python -m pytest tests/ \
  --ignore=tests/gui/ \
  --ignore=tests/interop/test_interop_obspy.py \
  --ignore=tests/signal/test_imputation.py \
  -q
```

---

## 総括

本作業セッションで新規追加したテストファイル数：**30+** ファイル
追加テスト総数（概算）：**700+** テスト
テスト収集総数：**5,687** テスト（セッション開始時より大幅増加）

モックを活用することで、外部ライブラリが未インストールの環境でもロジックの正確性を検証できる堅牢なテスト基盤を構築しました。
