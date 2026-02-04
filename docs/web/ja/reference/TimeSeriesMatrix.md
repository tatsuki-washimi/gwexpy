# TimeSeriesMatrix

**継承元:** PhaseMethodsMixin, TimeSeriesMatrixCoreMixin, TimeSeriesMatrixAnalysisMixin, TimeSeriesMatrixSpectralMixin, TimeSeriesMatrixInteropMixin, SeriesMatrix

**共通の時間軸を共有する複数の TimeSeries オブジェクト用の2Dマトリックスコンテナ。**

このクラスは、各要素が TimeSeries データストリームに対応する2次元配列（行 x 列）を表します。**マトリックス内のすべての要素は同じ時間配列を共有します**（同じ `t0`, `dt`, サンプル数）。グリッド構造で整理された多変量時系列のように振る舞います。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `dt` | 時間間隔 |
| `t0` | 開始時刻 |
| `times` | 時間配列 |
| `sample_rate` | サンプリングレート |
| `duration` | 継続時間 |
| `N_samples` | サンプル数 |
| `channel_names` | 要素名のフラットなリスト |
| `channels` | 各要素のチャンネル識別子の2D配列 |

## スペクトル変換

| メソッド | 説明 |
|---------|------|
| `fft()` | 各要素の FFT を計算。FrequencySeriesMatrix を返す |
| `psd()` | 各要素の PSD を計算。FrequencySeriesMatrix を返す |
| `asd()` | 各要素の ASD を計算。FrequencySeriesMatrix を返す |
| `q_transform()` | 各要素の Q 変換を計算。SpectrogramMatrix を返す |

## 信号処理

| メソッド | 説明 |
|---------|------|
| `filter()` / `bandpass()` / `highpass()` / `lowpass()` / `notch()` | 要素ごとのフィルタ適用 |
| `resample()` | 要素ごとのリサンプル |
| `detrend()` | 要素ごとのトレンド除去 |
| `lock_in()` | 要素ごとのロックイン増幅 |

## 統計・解析

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `median()` | 統計量計算 |
| `rolling_mean()` / `rolling_std()` / `rolling_max()` / `rolling_min()` | ローリング統計 |
| `pca()` / `ica()` | PCA/ICA 分解 |
| `impute()` | 欠損値の補完 |
| `correlation_vector()` | ターゲット時系列と全チャンネルの相関（`method="pearson"` は高速なベクトル化パスあり） |
| `partial_correlation_matrix()` | 全チャンネルの偏相関行列（precision から計算、shrinkage/eps 対応） |
| `coherence()` / `csd()` | コヒーレンス/クロススペクトル計算 |

## 線形代数

| メソッド | 説明 |
|---------|------|
| `det()` | 各サンプル点での行列式 |
| `inv()` | 各サンプル点での逆行列 |
| `trace()` | 対角要素の和 |
| `schur()` | シュア補行列 |
| `diagonal()` | 対角要素の抽出 |

## 変換・相互運用

| メソッド | 説明 |
|---------|------|
| `to_pandas()` | pandas DataFrame に変換 |
| `to_dict()` / `to_list()` | TimeSeriesDict / TimeSeriesList に変換 |
| `to_torch()` / `to_tensorflow()` / `to_jax()` / `to_cupy()` | ML フレームワークへ変換 |
| `from_neo()` | neo.AnalogSignal から作成 |

## 入出力

| メソッド | 説明 |
|---------|------|
| `read()` | ファイルからマトリックスを読み込む |
| `write()` | マトリックスをファイルに書き込む |
| `to_hdf5()` / `to_zarr()` | HDF5/Zarr 形式で保存 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | GPS 開始/終了時刻でクロップ。gwexpy.time.to_gps がサポートする形式を使用可能 |
| `append()` / `prepend()` | サンプル軸に沿って別のマトリックスを追加 |
| `interpolate()` | 新しいサンプル軸に補間 |
| `pad()` | サンプル軸に沿ってパディング |
