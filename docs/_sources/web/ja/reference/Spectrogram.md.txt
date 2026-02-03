# Spectrogram

**継承元:** SpectrogramAnalysisMixin, SpectrogramInteropMixin, SignalAnalysisMixin, PhaseMethodsMixin, RegularityMixin, BaseSpectrogram (gwpy.spectrogram.Spectrogram)

追加の相互運用メソッドを備えた gwpy.spectrogram.Spectrogram の拡張。

## Pickle / shelve の可搬性

> [!WARNING]
> 信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。

gwexpy の pickle は可搬性を優先しており、unpickle 時に **GWpy 型**を返す設計です
（読み込み側に gwexpy が無くても、gwpy があれば復元できます）。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `dt` | 時間間隔 |
| `t0` | 開始時刻 |
| `times` | 時間配列 |
| `df` | 周波数間隔 |
| `f0` | 開始周波数 |
| `frequencies` | 周波数配列 |
| `channel` | データチャンネル |
| `name` | データセット名 |
| `unit` | 物理単位 |

## スペクトル解析

| メソッド | 説明 |
|---------|------|
| `percentile()` | パーセンタイル計算 |
| `bootstrap_asd()` | ブートストラップ ASD 推定 |
| `rebin()` | 時間/周波数リビン |
| `interpolate()` | 補間 |

## 位相解析

| メソッド | 説明 |
|---------|------|
| `phase()` | 位相計算 |
| `angle()` | phase() のエイリアス |
| `degree()` | 位相（度）、単位 'deg' を設定 |
| `radian()` | 位相（ラジアン）、単位 'rad' を設定 |
| `unwrap_phase()` | 位相アンラップ |

## 統計

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `median()` | 統計量 |
| `abs()` | 絶対値 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | 時間範囲でクロップ |
| `crop_frequencies()` | 周波数範囲でクロップ |

## 相互運用性

| メソッド | 説明 |
|---------|------|
| `to_pandas()` | pandas 変換 |
| `to_timeseries_list()` | 周波数ビンごとに TimeSeriesList へ変換 |
| `to_frequencyseries_list()` | 時刻ビンごとに FrequencySeriesList へ変換 |
| `to_torch()` / `to_tensorflow()` / `to_jax()` / `to_cupy()` | ML フレームワーク変換 |
| `to_xarray()` | xarray 変換 |

## 入出力

| メソッド | 説明 |
|---------|------|
| `read()` / `write()` | ファイル入出力 |

## 可視化

| メソッド | 説明 |
|---------|------|
| `plot()` | スペクトログラムプロット |
