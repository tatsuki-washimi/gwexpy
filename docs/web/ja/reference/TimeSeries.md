# TimeSeries

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`TimeSeries` は単一の等間隔サンプリング時系列を扱うための基本クラスで、GWexpy の信号処理・モデリング・相互運用拡張を含みます。

## 代表的なシグネチャ

```python
TimeSeries(data, unit=None, t0=None, dt=None, sample_rate=None, times=None, ...)
TimeSeries.fft(fftlength=None, overlap=0, window="hann", ...)
```

## 最小例

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

ts = TimeSeries(np.random.randn(1024), sample_rate=1024, unit="strain")
psd = ts.psd(fftlength=1.0)
```

## 関連理論

- [Physics Models](../user_guide/physics_models.md)
- {ref}`トランジェント FFT の検証 <validated-ja-transient-fft>` - トランジェント FFT モードの振幅規約と前提条件
- {ref}`ARIMA 予測時刻の検証 <validated-ja-arima-forecast>` - 予測延長時の GPS 時刻前提
- {ref}`MCMC / GLS 尤度の検証 <validated-ja-mcmc-gls>` - 時系列データをフィットに渡す際の尤度前提
- [FFT_Conventions](FFT_Conventions.md)
- [前提条件と規約](../user_guide/prerequisites_and_conventions.md)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [Signal Extraction](../user_guide/tutorials/case_signal_extraction.ipynb)
- [Advanced ARIMA](../user_guide/tutorials/advanced_arima.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** [`gwpy.timeseries.TimeSeries`](https://gwpy.readthedocs.io/en/latest/reference/gwpy.timeseries.TimeSeries/)

すべての gwexpy 機能を備えた拡張 TimeSeries。

## 物理コンテキスト

`TimeSeries` は「単一チャネルの時間領域信号」を表します。重力波ひずみ `strain`、地面速度、電圧、制御誤差信号、マイク出力のように、**各サンプルが一つの時刻に対応する量**を保持する場合の入口です。

- **時間軸の意味**: `t0`, `dt`, `sample_rate`, `times` は解析窓の物理時刻を決めます。特に `fetch()` / `fetch_open_data()` の戻り値は GPS 時刻を前提に後段のセグメント解析やイベント同期解析へ渡されます。
- **単位の意味**: `unit` は単なる飾りではなく、フィルタ・微積分・フィッティング・外部変換時の解釈に効きます。`strain`、`m/s`、`V` などを明示しておくと、周波数領域へ変換した後の `1/Hz` 系の量とも整合を取りやすくなります。
- **等間隔サンプリング前提**: `TimeSeries` は等間隔サンプル列を想定します。不規則サンプリングやイベント表は `SegmentTable` / table 系、複数チャネル同時解析は `TimeSeriesMatrix` / `TimeSeriesDict` を使う方が自然です。

## 解析上の注意点

### FFT・PSD に入る前

`fft()`, `psd()`, `asd()`, `spectrogram()` は時間領域データを周波数領域へ写像します。このとき重要なのは、配列値そのものよりも「**窓長・オーバーラップ・窓関数・平均化単位**」です。

- 定常雑音の代表量を見たいなら `psd()` / `asd()` を使う
- 短いバーストや chirp を追いたいなら `spectrogram()` / `q_transform()` を使う
- 振幅規約や transient mode の扱いは [FFT_Conventions](FFT_Conventions.md) と検証済みアルゴリズム側の根拠を参照する

### 前処理の意味

`detrend()`, `highpass()`, `whiten()`, `standardize()`, `impute()` は、単に見た目を整える操作ではなく、**どの物理成分を保持し、どの系統誤差を落とすか**を決める操作です。

- `detrend()` / `highpass()` は低周波ドリフトを除きたいときに使う
- `whiten()` は検出・時刻周波数可視化・相関解析の前に広帯域比較をしやすくする
- `impute()` は欠損を埋めるが、埋めた区間を物理信号そのものとして解釈してはいけない

### どこで誤読しやすいか

1. サンプル値だけを見て `sample_rate` や `t0` を無視する
2. 前処理後の系列を「元の物理量そのもの」とみなす
3. `fft()` 結果の単位や振幅規約を確認せずに別手法と比較する
4. 単一チャネル系列に多チャネル因果や空間構造の意味を持たせすぎる

## どのページへ進むか

- 時間領域から周波数領域への規約確認: [FFT_Conventions](FFT_Conventions.md)
- 観測データ取得や direct I/O: [I/O Formats](../user_guide/io_formats.md)
- GWpy からの移行観点: [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- 時間周波数解析の比較: [時間-周波数解析: 手法比較ガイド](../user_guide/tutorials/time_frequency_comparison.md)
- ARIMA や予測系: [Advanced ARIMA](../user_guide/tutorials/advanced_arima.ipynb)

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

:::{admonition} warning
:class: warning

信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。
:::

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
