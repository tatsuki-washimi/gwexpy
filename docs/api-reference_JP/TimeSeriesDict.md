# TimeSeriesDict

**継承元:** TimeSeriesDict

`TimeSeries` オブジェクトをチャンネル名をキーとして保持するマッピングコンテナです。

## メソッド

### `abs`, `angle`

各要素の絶対値と位相角。

### `append`

各要素の末尾にデータを追加します（インプレース）。

### `asd`

各要素の ASD（振幅スペクトル密度）を計算し、FrequencySeriesDict を返します。

### `bandpass`, `highpass`, `lowpass`, `notch`, `zpk`

各要素に指定されたフィルタを適用します。

### `crop`

各要素を特定の時間範囲で切り抜きます。

### `detrend`, `whiten`

各要素のトレンド除去およびホワイトニング。

### `fft`

各要素の FFT を計算し、FrequencySeriesDict を返します。

### `impute`

欠損値（NaN）の補完。

### `pca`

チャンネル間で PCA 分解を実行します。

### `plot`, `step`

プロットまたは階段グラフを作成します。チャンネル名がデフォルトの凡例（label）として使用されます。

### `psd`, `spectrogram`, `q_transform`

各要素の PSD、スペクトログラム、または Q 変換を計算し、対応する Dict クラス（FrequencySeriesDict, SpectrogramDict）を返します。

### `read`

```python
read(source, *args, **kwargs)
```

指定されたソースから複数のチャンネルを読み込み TimeSeriesDict を返します。

### `resample`

各要素を再サンプリングします。インプレース操作です。

### `rms`, `std`

各要素の RMS または標準偏差を計算し、pandas.Series を返します。

### `rolling_mean`, `rolling_median`, `rolling_std`, `rolling_min`, `rolling_max`

各要素に移動窓計算（ローリング統計）を適用します。

### `to_matrix`

この辞書を TimeSeriesMatrix に変換します（時間アライメントが自動的に行われます）。

### `to_pandas`, `to_polars`

各ライブラリの DataFrame に変換します。

### `write`

TimeSeriesDict をファイル（GWF, HDF5 など）に書き出します。
