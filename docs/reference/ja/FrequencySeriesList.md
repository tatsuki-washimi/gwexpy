# FrequencySeriesList

**継承元:** FrequencySeriesBaseList

`FrequencySeries` オブジェクトのリスト。

## メソッド

### `angle`

```python
angle(self, *args, **kwargs) -> 'FrequencySeriesList'
```

phase() のエイリアス。各 FrequencySeries の位相を計算し、新しい FrequencySeriesList を返します。

### `apply_response`

```python
apply_response(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各 FrequencySeries にレスポンスを適用します。新しい FrequencySeriesList を返します。

### `crop`

```python
crop(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各 FrequencySeries をクロップします。新しい FrequencySeriesList を返します。

### `degree`

```python
degree(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各 FrequencySeries の位相（度単位）を計算し、新しい FrequencySeriesList を返します。

### `differentiate_time`

```python
differentiate_time(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムに時間微分を適用します。新しい FrequencySeriesList を返します。

### `filter`

```python
filter(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムにフィルタを適用します。新しい FrequencySeriesList を返します。

### `ifft`

```python
ifft(self, *args, **kwargs)
```

各 FrequencySeries の逆FFT（IFFT）を計算します。TimeSeriesList を返します。

### `integrate_time`

```python
integrate_time(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムに時間積分を適用します。新しい FrequencySeriesList を返します。

### `interpolate`

```python
interpolate(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムを補間します。新しい FrequencySeriesList を返します。

### `pad`

```python
pad(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムをパディングします。新しい FrequencySeriesList を返します。

### `phase`

```python
phase(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムの位相を計算します。新しい FrequencySeriesList を返します。

### `plot`

```python
plot(self, **kwargs)
```

すべてのシリーズをプロットします。`gwpy.plot.Plot` に委譲されます。

### `plot_all`

```python
plot_all(self, *args, **kwargs)
```

plot() のエイリアス。すべてのシリーズをプロットします。

### `to_db`

```python
to_db(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムをデシベル（dB）単位に変換します。新しい FrequencySeriesList を返します。

### `write`

```python
write(self, target: str, *args, **kwargs) -> 'Any'
```

リストをファイル（HDF5, ROOT等）に書き出します。

### `to_tmultigraph`

ROOT の TMultiGraph オブジェクトに変換します。

### `zpk`

```python
zpk(self, *args, **kwargs) -> 'FrequencySeriesList'
```

各アイテムに ZPK フィルタを適用します。新しい FrequencySeriesList を返します。
