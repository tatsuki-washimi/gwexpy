# TimeSeries

**継承元:** `gwpy.timeseries.TimeSeries`

すべての gwexpy 機能を備えた拡張 TimeSeries。

## 主な拡張機能 (Key Extensions)

### 統計と相関

- **`correlation(other, method="pearson", ...)`**
  他の TimeSeries との相関を計算します。
  手法: `"pearson"`, `"kendall"`, `"mic"`, `"distance"`.
- **`partial_correlation(other, controls=None, ...)`**
  第三変数の影響を除いた偏相関を計算します。
- **`fastmi(other, grid_size=128)`**
  FastMI (FFTベース) 推定器を用いて相互情報量を計算します。
- **`granger_causality(other, maxlag=5)`**
  時系列間の因果関係（Granger Causality）を検定します。

### 信号処理

- **`hilbert()` / `envelope()`**
  解析信号とその振幅包絡線を計算します。
- **`mix_down(f0)`**
  特定の搬送周波数で信号を復調します。
- **`fft(mode="steady"|"transient", ...)`**
  ゼロパディングやウィンドウ管理のオプションを備えた拡張FFT。

### モデリングと前処理

- **`arima(order=(p,d,q))`**
  ARIMA 時系列モデルを適合します。
- **`impute(method="interpolate")`**
  データ内の欠損値 (NaN) を処理します。
- **`standardize(method="zscore")`**
  平均 0、分散 1 になるようにデータを再スケーリングします。

## 使用例

```python
from gwexpy.timeseries import TimeSeries
ts = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)

# 非線形相関を計算
mic_score = ts.correlation(other_ts, method="mic")

# 標準化して包絡線を計算
env = ts.standardize().envelope()
```

## Pickle / shelve の可搬性

> [!WARNING]
> 信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。

gwexpy の pickle は可搬性を優先しており、unpickle 時に **GWpy 型**を返す設計です
（読み込み側に gwexpy が無くても、gwpy があれば復元できます）。

## 全メソッド一覧

| カテゴリ | メソッド |
|---|---|
| **スペクトル** | `fft`, `psd`, `asd`, `spectrogram`, `q_transform`, `cwt`, `cepstrum` |
| **信号処理** | `filter`, `bandpass`, `highpass`, `lowpass`, `notch`, `resample`, `detrend`, `whiten`, `taper` |
| **解析** | `find_peaks`, `instantaneous_phase`, `rolling_mean` |
| **相互運用** | `to_pandas`, `to_torch`, `to_tensorflow`, `to_xarray` |
| **入出力** | `read`, `write`, `get`, `fetch` |
