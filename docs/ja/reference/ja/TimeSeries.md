# TimeSeries

**継承元:** TimeSeriesInteropMixin, TimeSeriesAnalysisMixin, TimeSeriesResamplingMixin, TimeSeriesSignalMixin, SignalAnalysisMixin, TimeSeriesSpectralMixin, StatisticsMixin, FittingMixin, PhaseMethodsMixin, RegularityMixin, _LegacyTimeSeries (gwpy.timeseries.TimeSeries)

すべての gwexpy 機能を備えた拡張 TimeSeries。

このクラスは複数のモジュールからの機能を統合します：
- コア操作: is_regular, _check_regular, tail, crop, append, find_peaks
- スペクトル変換: fft, psd, cwt, laplace など
- 信号処理: hilbert, mix_down, xcorr など
- 解析: impute, standardize, rolling_* など
- 相互運用性: to_pandas, to_torch, to_xarray など

gwpy.timeseries.TimeSeries から継承し、完全な互換性を持ちます。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `dt` | サンプル間隔 |
| `t0` | 開始時刻 (GPS エポック) |
| `times` | 時間配列 |
| `sample_rate` | サンプリングレート |
| `duration` | 継続時間 |
| `channel` | データチャンネル |
| `name` | データセット名 |
| `unit` | 物理単位 |

## スペクトル変換

| メソッド | 説明 |
|---------|------|
| `fft()` | 高速フーリエ変換 |
| `psd()` / `asd()` | パワー/振幅スペクトル密度 |
| `spectrogram()` / `spectrogram2()` | スペクトログラム |
| `q_transform()` | Q 変換 |
| `cwt()` | 連続ウェーブレット変換 |
| `cepstrum()` | ケプストラム解析 |

## 信号処理

| メソッド | 説明 |
|---------|------|
| `filter()` / `bandpass()` / `highpass()` / `lowpass()` / `notch()` | フィルタリング |
| `resample()` / `decimate()` | リサンプリング |
| `detrend()` | トレンド除去 |
| `whiten()` | ホワイトニング |
| `taper()` | テーパー処理 |
| `hilbert()` / `analytic_signal()` | 解析信号 |
| `heterodyne()` / `baseband()` / `mix_down()` / `lock_in()` | ヘテロダイン処理 |
| `xcorr()` | 相互相関 |
| `transfer_function()` / `coherence()` / `csd()` | 伝達関数/コヒーレンス/CSD |

## 解析

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `rms()` | 統計量 |
| `rolling_mean()` / `rolling_std()` 等 | ローリング統計 |
| `find_peaks()` | ピーク検出 |
| `instantaneous_phase()` / `instantaneous_frequency()` | 瞬時位相/周波数 |
| `envelope()` | 包絡線 |
| `impute()` | 欠損値補完 |
| `standardize()` | 標準化 |

## 相互運用性

| メソッド | 説明 |
|---------|------|
| `to_pandas()` / `from_pandas()` | pandas 変換 |
| `to_torch()` / `to_tensorflow()` / `to_jax()` | ML フレームワーク変換 |
| `to_xarray()` / `to_polars()` | xarray/polars 変換 |
| `to_control()` / `from_control()` | python-control 変換 |
| `to_pint()` | Pint 変換 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | 時間範囲でクロップ（あらゆる時刻形式対応） |
| `append()` / `prepend()` | データ接続 |
| `shift()` | 時間シフト |
| `pad()` | パディング |
| `gate()` / `mask()` | ゲート/マスク処理 |

## 入出力

| メソッド | 説明 |
|---------|------|
| `read()` / `write()` | ファイル入出力 |
| `get()` / `fetch()` / `find()` | データ取得 (NDS, GWF) |
