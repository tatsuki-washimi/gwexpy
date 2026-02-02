# FrequencySeriesDict

**継承元:** FrequencySeriesBaseDict

ラベルをキーとする `FrequencySeries` オブジェクトの順序付きマッピング。

## メソッド

### `__init__`

```python
__init__(self, *args: 'Any', **kwargs: 'Any')
```

self を初期化します。

*(OrderedDict から継承)*

### `EntryClass`

```python
EntryClass(data, unit=None, f0=None, df=None, frequencies=None, name=None, epoch=None, channel=None, **kwargs)
```

互換性と将来の拡張のための gwpy の FrequencySeries の軽量ラッパー。

### `angle`

```python
angle(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

phase() のエイリアス。新しい FrequencySeriesDict を返します。

### `crop`

```python
crop(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

辞書内の各 FrequencySeries をクロップします。その場で操作（GWpy 互換）。self を返します。

### `degree`

```python
degree(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries の位相（度単位）を計算します。新しい FrequencySeriesDict を返します。

### `differentiate_time` / `integrate_time`

周波数領域での時間微分/積分を各アイテムに適用します。

### `filter`

```python
filter(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries にフィルタを適用します。新しい FrequencySeriesDict を返します。

### `group_delay`

```python
group_delay(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各アイテムの群遅延を計算します。

### `ifft`

```python
ifft(self, *args, **kwargs)
```

各 FrequencySeries の IFFT を計算します。TimeSeriesDict を返します。

### `interpolate`

```python
interpolate(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

辞書内の各 FrequencySeries を補間します。

### `phase`

```python
phase(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries の位相を計算します。

### `plot`

```python
plot(self, label: 'str' = 'key', method: 'str' = 'plot', figsize: 'Optional[Any]' = None, **kwargs: 'Any')
```

データをプロットします。

パラメータ
----------
label : str, optional
    ラベル付け方法: ``'key'``（辞書キーを使用）または ``'name'``（各アイテムの name 属性を使用）
method : str, optional
    :class:`~gwpy.plot.Plot` の呼び出しメソッド。デフォルト: ``'plot'``

### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

FrequencySeriesDict をファイルに書き込みます。

HDF5 出力では `layout` を指定できます（デフォルトは GWpy 互換の dataset-per-entry）。

```python
fsd.write("out.h5", format="hdf5")               # GWpy互換（既定）
fsd.write("out.h5", format="hdf5", layout="group")  # 旧形式（group-per-entry）
```

HDF5 のデータセット名（GWpy の `path=` 用）:
- キーは HDF5 で安全な名前にサニタイズされます（例: `H1:ASD` -> `H1_ASD`）。
- サニタイズ後の名前が衝突する場合、`__1` のようなサフィックスが付与されます。
- 元のキーはファイル属性に保存され、gwexpy の `read()` は元キーを復元します。

> [!WARNING]
> 信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。

pickle 可搬性メモ: gwexpy の `FrequencySeriesDict` は unpickle 時に builtins の `dict` を返します
（中身は GWpy の `FrequencySeries`、読み込み側に gwexpy は不要です）。

### `smooth`

```python
smooth(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries を平滑化します。

### `to_db`

```python
to_db(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries を dB に変換します。

### `to_matrix`

```python
to_matrix(self)
```

この FrequencySeriesDict を FrequencySeriesMatrix (Nx1) に変換します。

### `to_pandas` / `to_xarray`

pandas.DataFrame / xarray.Dataset に変換します。キーは列/データ変数になります。

### `to_cupy` / `to_jax` / `to_tensorflow` / `to_torch`

各アイテムを対応するフレームワークのテンソル/配列に変換します。

### `write`

```python
write(self, target: 'str', *args: 'Any', **kwargs: 'Any') -> 'Any'
```

辞書をファイル（HDF5, ROOT など）に書き込みます。

### `zpk`

```python
zpk(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries に ZPK フィルタを適用します。
