# FrequencySeriesDict

**継承元:** FrequencySeriesBaseDict

ラベル（チャンネル名など）をキーとした `FrequencySeries` オブジェクトの順序付きマッピング。

## メソッド

### `angle`

```python
angle(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

phase() のエイリアス。各 FrequencySeries の位相を計算し、新しい FrequencySeriesDict を返します。

### `apply_response`

```python
apply_response(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries にレスポンスを適用します。新しい FrequencySeriesDict を返します。

### `crop`

```python
crop(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries をクロップします。インプレース操作（GWpy互換）で、selfを返します。

### `degree`

```python
degree(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries の位相（度単位）を計算し、新しい FrequencySeriesDict を返します。

### `differentiate_time`

```python
differentiate_time(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各アイテムに時間微分を適用します。新しい FrequencySeriesDict を返します。

### `filter`

```python
filter(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries にフィルタを適用します。新しい FrequencySeriesDict を返します。

### `group_delay`

```python
group_delay(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各アイテムの群遅延を計算します。新しい FrequencySeriesDict を返します。

### `ifft`

```python
ifft(self, *args, **kwargs)
```

各 FrequencySeries の逆FFT（IFFT）を計算します。TimeSeriesDict を返します。

### `integrate_time`

```python
integrate_time(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各アイテムに時間積分を適用します。新しい FrequencySeriesDict を返します。

### `interpolate`

```python
interpolate(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries を補間します。新しい FrequencySeriesDict を返します。

### `pad`

```python
pad(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries をパディングします。新しい FrequencySeriesDict を返します。

### `phase`

```python
phase(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries の位相を計算します。新しい FrequencySeriesDict を返します。

### `plot`

```python
plot(self, label='key', method='plot', figsize=None, **kwargs)
```

辞書内のすべてのシリーズをプロットします。

### `plot_all`

```python
plot_all(self, *args, **kwargs)
```

plot() のエイリアス。辞書内のすべてのシリーズをプロットします。

### `smooth`

```python
smooth(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries を平滑化します。新しい FrequencySeriesDict を返します。

### `to_db`

```python
to_db(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries をデシベル（dB）単位に変換します。新しい FrequencySeriesDict を返します。

### `write`

```python
write(self, target: str, *args, **kwargs) -> 'Any'
```

辞書をファイル（HDF5, ROOT等）に書き出します。

### `to_tmultigraph`

ROOT の TMultiGraph オブジェクトに変換します。

### `zpk`

```python
zpk(self, *args, **kwargs) -> 'FrequencySeriesDict'
```

各 FrequencySeries に ZPK フィルタを適用します。新しい FrequencySeriesDict を返します。
