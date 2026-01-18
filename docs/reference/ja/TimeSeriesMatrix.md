# TimeSeriesMatrix

**継承元:** SeriesMatrix

共通の時間軸を共有する複数の `TimeSeries` オブジェクトを保持する **2次元行列コンテナ** です。

このクラスは、各要素が `TimeSeries` データストリームに対応する 2次元配列（行 x 列）を表します。重要な点として、**行列内のすべての要素は同一の時間配列（同じ `t0`, `dt`, サンプル数）を共有** します。これは、グリッド構造に配置された多変量時系列データのように振る舞います。

`dt`, `t0`, `times` などのエイリアスを提供し、FFT を介して `FrequencySeriesMatrix` を構築できます。

## メソッド

### `MetaDataMatrix`

各要素のメタデータ。

### `N_samples`

サンプル数。

### `T`

転置ビュー。

### `abs`, `angle`

絶対値と位相角。

### `append`, `append_exact`

行列の追加。

### `asd`

```python
asd(self, **kwargs: 'Any') -> 'Any'
```

各要素の ASD（振幅スペクトル密度）を計算し、FrequencySeriesMatrix を返します。

### `auto_coherence`, `coherence`, `csd`

TimeSeries の各メソッドへの要素ごとの委譲。

### `bandpass`, `detrend`, `highpass`, `lowpass`, `taper`, `whiten`

TimeSeries プリプロセス・フィルタメソッドへの要素ごとの委譲。

### `channels`

チャンネル名の 2D 配列。

### `crop`

```python
crop(self, start: 'Any' = None, end: 'Any' = None, copy: 'bool' = False) -> "'TimeSeriesMatrix'"
```

指定された GPS 開始時刻と終了時刻で行列を切り抜きます。

### `StandardScaler` (他 `*Scaler`)

スケーリング（標準化）を行うための変換器クラスです。

### `pca`

主成分分析 (Principal Component Analysis) を行い、低次元の成分を抽出します。

### `ica`

独立成分分析 (Independent Component Analysis) を行い、信号源を分離します。

### `diagonal`

対角要素の抽出。

### `diff`

離散差分。

### `dt`, `t0`, `times`

時間間隔、開始時刻、時間配列のエイリアス。

### `fft`

```python
fft(self) -> "'FrequencySeriesMatrix'"
```

各要素の FFT を計算し、FrequencySeriesMatrix を返します。

### `filter`

フィルタの適用。

### `histogram`

各要素のヒストグラムを計算します。

### `resample`

指定されたサンプリング周波数に再サンプリングします。

### `spectrogram`

```python
spectrogram(self, stride: 'Any' = 1, fftlength: 'Optional[Any]' = None, overlap: 'Optional[Any]' = None, window: 'str' = 'hann', **kwargs: 'Any') -> 'Any'
```

各要素のスペクトログラムを計算し、SpectrogramMatrix を返します。

### `standardize`

```python
standardize(self, *, axis: 'str' = 'time', method: 'str' = 'zscore', ddof: 'int' = 0, **kwargs: 'Any') -> 'Any'
```

行列を標準化します。

### `to_dict`, `to_list`

TimeSeriesDict または TimeSeriesList への変換。

### `to_mne`, `to_mne`

mne.io.RawArray への変換。

### `to_neo`

neo.AnalogSignal への変換。

### `to_pandas`

pandas DataFrame への変換。

### `to_torch`

PyTorch テンソルへの変換（形状を維持）。

### `whiten_channels`

```python
whiten_channels(self, *, method: 'str' = 'pca', eps: 'float' = 1e-12, n_components: 'Optional[int]' = None, return_model: 'bool' = True) -> 'Any'
```

行列（チャンネル/コンポーネント）をホワイトニング（白色化）します。PCA などを使用します。

### `write`, `read`, `to_hdf5`

ファイル入出力。

### `x0`, `xindex`, `xspan`, `xunit`

x 軸（時間軸）のプロパティ。
