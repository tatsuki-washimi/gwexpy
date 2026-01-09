# TimeSeriesList

**継承元:** UserList

`TimeSeries` オブジェクトのリストです。

## メソッド

### `__init__`

```python
__init__(self, initlist=None)
```

TimeSeriesList を初期化します。

### `asd`

各要素の ASD（振幅スペクトル密度）を計算し、FrequencySeriesList を返します。

### `bandpass`, `highpass`, `lowpass`

各要素にフィルタを適用します。

### `crop`

各要素を特定の時間範囲で切り抜きます。

### `detrend`, `whiten`

各要素のトレンド除去およびホワイトニング。

### `fft`

各要素の FFT を計算し、FrequencySeriesList を返します。

### `impute`

```python
impute(self, *, method: 'str' = 'linear', fill_value: 'Any' = 0.0, overlap: 'bool' = True, **kwargs: 'Any') -> 'Any'
```

欠損値（NaN）を指定された手法で補完します。

### `join`

```python
join(self, pad=0.0, gap='raise')
```

リスト内の TimeSeries を 1 つの TimeSeries に結合します。

### `plot`

すべてのシリーズを垂直に並べてプロットします。

### `read`

ファイルから TimeSeries のリストを読み込みます。

### `resample`

各要素を再サンプリングします。

### `spectrogram`

各要素のスペクトログラムを計算し、SpectrogramList を返します。

### `to_matrix`

リストを TimeSeriesMatrix に変換します。

### `to_tmultigraph`

ROOT の TMultiGraph オブジェクトに変換します。

### `write`

リストをファイル（HDF5 など）に書き出します。
