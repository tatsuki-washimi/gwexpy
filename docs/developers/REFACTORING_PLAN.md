# TimeSeries Refactoring Plan

## 目的
`timeseries.py` (3147行) を保守性・可読性向上のため、物理的な処理内容に基づいて複数のモジュールに分割する。

## 分割方針

### 原則
1. **物理的な処理内容で分ける** - 機能的な凝集度を最大化
2. **循環参照を避ける** - 依存関係は一方向に保つ
3. **後方互換性を維持** - ユーザーコードの変更不要
4. **段階的移行** - 各モジュールを独立してテスト可能

### モジュール構成

```
gwexpy/timeseries/
├── __init__.py              # 公開API（TimeSeries等をエクスポート）
├── timeseries.py            # 【削除予定】現行の巨大ファイル
├── timeseries_backup.py     # バックアップ
├── _core.py                 # ★ コアクラス定義・基本操作
├── _resampling.py           # ★ リサンプリング関連
├── _signal.py               # ★ 信号処理（Hilbert, 位相等）
├── _spectral.py             # ★ スペクトル変換（FFT, CWT等）
├── _analysis.py             # ★ 統計・分析（impute, rolling等）
├── _interop.py              # ★ 相互運用性（pandas, torch等）
└── tests/
```

## 詳細分割計画

### 1. `_core.py` - コアクラス定義・基本操作
**目的**: `TimeSeries` クラスの骨格と、最も基本的な操作

**含むメソッド**:
- クラス定義、`__init__`（gwpy.TimeSeries を継承）
- プロパティ: `is_regular`, `_check_regular`
- 基本操作: `tail()`, `crop()`, `append()`
- ユーティリティ: `find_peaks()`

**依存**: numpy, astropy, gwpy のみ

**推定行数**: ~300行

---

### 2. `_resampling.py` - リサンプリング・時間軸操作
**目的**: 時間軸の再編成に関する全処理

**含むメソッド**:
- `asfreq(rule, method, fill_value, ...)`
- `resample(rate, *args, **kwargs)`
- `_resample_time_bin(...)` (内部ヘルパー)
- `stlt(stride, window, ...)` (Short-Time Laplace Transform)
  - **現状の注意**: 現在の実装は STFT の振幅外積（3D表示）のみを行っており、数学的なラプラス変換（$\sigma$ による複素周波数）は未実装。将来的な課題として保留。

**依存**: `_core.TimeSeries`, pandas-like time rules

**推定行数**: ~500行

---

### 3. `_signal.py` - 信号処理
**目的**: 時間領域での信号変換・解析

**含むメソッド**:
- Hilbert変換系:
  - `analytic_signal()`, `hilbert()`, `envelope()`
- 位相・周波数系:
  - `instantaneous_phase()`, `unwrap_phase()`, `instantaneous_frequency()`
- 時間領域ミキシング:
  - `_build_phase_series()` (内部)
  - `mix_down()`, `baseband()`, `lock_in()`
- 高度な解析:
  - `hilbert_analysis()`, `transfer_function()`, `xcorr()`

**依存**: `_core.TimeSeries`, scipy.signal

**推定行数**: ~600行

---

### 4. `_spectral.py` - スペクトル変換
**目的**: 周波数領域への変換処理

**含むメソッド**:
- FFT系:
  - `fft()`, `rfft()`, `psd()`, `asd()`, `csd()`, `coherence()`
- その他の変換:
  - `dct()`, `laplace()`, `cwt()`, `cepstrum()`
- 時間周波数解析:
  - `emd()`, `hht()`
- ヘルパー:
  - `_prepare_data_for_transform()`

**依存**: `_core.TimeSeries`, scipy.fft, pywt

**推定行数**: ~800行

---

### 5. `_analysis.py` - 統計・分析
**目的**: 統計処理、前処理、時系列モデリング

**含むメソッド**:
- 前処理:
  - `impute()`, `standardize()`
- 時系列モデル:
  - `fit_arima()`, `hurst()`, `local_hurst()`
- ローリング統計:
  - `rolling_mean()`, `rolling_std()`, `rolling_median()`, `rolling_min()`, `rolling_max()`

**依存**: `_core.TimeSeries`, preprocess.py, arima.py, hurst.py, rolling.py

**推定行数**: ~300行

---

### 6. `_interop.py` - 相互運用性
**目的**: 外部ライブラリとの相互変換

**含むメソッド**:
- pandas: `to_pandas()`, `from_pandas()`
- xarray: `to_xarray()`, `from_xarray()`
- HDF5: `to_hdf5_dataset()`, `from_hdf5_dataset()`
- ObsPy: `to_obspy_trace()`, `from_obspy_trace()`
- SQLite: `to_sqlite()`, `from_sqlite()`
- PyTorch: `to_torch()`, `from_torch()`
- TensorFlow: `to_tf()`, `from_tf()`
- Dask: `to_dask()`, `from_dask()`
- Zarr: `to_zarr()`, `from_zarr()`
- NetCDF4: `to_netcdf4()`, `from_netcdf4()`
- JAX: `to_jax()`, `from_jax()`
- CuPy: `to_cupy()`, `from_cupy()`
- librosa: `to_librosa()`
- pydub: `to_pydub()`, `from_pydub()`
- astropy: `to_astropy_timeseries()`, `from_astropy_timeseries()`
- MNE: `to_mne_rawarray()`, `to_mne_raw()`

**依存**: `_core.TimeSeries`, 各外部ライブラリ (optional)

**推定行数**: ~600行

---

## 実装手順

### Phase 1: Core の分離 (最優先)
1. `_core.py` を作成
   - `TimeSeries` クラス定義（空のスタブメソッド含む）
   - `is_regular`, `_check_regular` 実装
   - `tail`, `crop`, `append` を移行
2. `__init__.py` で `_core.TimeSeries` を `TimeSeries` としてエクスポート
3. 既存テストが通ることを確認

### Phase 2: Spectral の分離
1. `_spectral.py` を作成
2. スペクトル系メソッドを `_core.TimeSeries` に Mixin として追加
3. テスト実行

### Phase 3: Signal & Resampling の分離
1. `_signal.py`, `_resampling.py` を作成
2. 各メソッドを Mixin化

### Phase 4: Analysis & Interop の分離
1. `_analysis.py`, `_interop.py` を作成
2. 全メソッド移行完了

### Phase 5: クリーンアップ
1. 元の `timeseries.py` を削除
2. `timeseries_backup.py` も削除（リリース前）
3. ドキュメント更新

---

## Mixin パターン

各モジュールは、以下のように Mixin として実装:

```python
# _core.py
class TimeSeriesCore(gwpy.timeseries.TimeSeries):
    \"\"\"Core TimeSeries with basic operations.\"\"\"
    def tail(self, n=5): ...
    def crop(self, start, end): ...

# _spectral.py
class TimeSeriesSpectralMixin:
    \"\"\"Spectral transform methods.\"\"\"
def fft(self, nfft=None, ...): ...
    def cwt(self, ...): ...

# __init__.py
from ._core import TimeSeriesCore
from ._spectral import TimeSeriesSpectralMixin
from ._signal import TimeSeriesSignalMixin
# ... other mixins

class TimeSeries(
    TimeSeriesCore,
    TimeSeriesSpectralMixin,
    TimeSeriesSignalMixin,
    # ... other mixins
):
    \"\"\"
    Extended TimeSeries with all gwexpy functionality.
    \"\"\"
    pass
```

---

## 注意事項

1. **循環参照の回避**:
   - Mixin は `_core.TimeSeries` を直接インポートしない
   - 型ヒントには `TYPE_CHECKING` を活用

2. **テストの整合性**:
   - 各 Phase で既存テストが100%通ることを確認
   - 新しいテストは各モジュールごとに作成可能

3. **IDE サポート**:
   - `.pyi` スタブファイルも更新（必要に応じて）

---

## メトリクス目標

| 項目 | 現状 | 目標 |
|------|------|------|
| 最大ファイル行数 | 3147 | < 800 |
| モジュール数 | 1 | 6 |
| Cyclomatic complexity | 不明 | 各関数 < 15 |

---

## 関連 Issue / PR

- Issue #XXX: Refactor timeseries.py into logical modules
- PR #YYY: Phase 1 - Core separation

---

## 承認

- [ ] プロジェクトリード承認
- [ ] コードレビュー完了
- [ ] 全テスト通過
- [ ] ドキュメント更新完了
