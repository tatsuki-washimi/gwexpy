# SpectrogramList

**継承元:** UserList

`Spectrogram` オブジェクトのリスト。
TimeSeriesList と同様ですが、2D の Spectrogram を対象としています。

.. note::
   Spectrogram オブジェクトはメモリ上で非常に大きくなる可能性があります。
   ディープコピーを避けるため、可能な限り `inplace=True` を使用してください。

## メソッド

### `__init__`

```python
__init__(self, initlist=None)
```

初期化。

*( `MutableSequence` から継承)*

### `append`

```python
append(self, item)
```

リストの末尾にアイテムを追加します。

*( `MutableSequence` から継承)*

### `crop`

```python
crop(self, t0, t1, inplace=False)
```

各スペクトログラムをクロップします。

### `crop_frequencies`

```python
crop_frequencies(self, f0, f1, inplace=False)
```

周波数軸でクロップします。

### `extend`

```python
extend(self, other)
```

反復可能オブジェクトの要素を追加してリストを拡張します。

*( `MutableSequence` から継承)*

### `interpolate`

```python
interpolate(self, dt, df, inplace=False)
```

各スペクトログラムを補間します。

### `plot`

```python
plot(self, **kwargs)
```

すべてのスペクトログラムを垂直方向に並べてプロットします。

### `read`

```python
read(self, source, *args, **kwargs)
```

HDF5からスペクトログラムをリストに読み込みます。

### `rebin`

```python
rebin(self, dt, df, inplace=False)
```

各スペクトログラムをリビンします。

### `to_cupy`

```python
to_cupy(self, dtype=None)
```

各スペクトログラムを CuPy 配列に変換します。リストを返します。

### `to_matrix`

```python
to_matrix(self)
```

SpectrogramMatrix (N, Time, Freq) に変換します。

### `to_torch`

```python
to_torch(self, device=None, dtype=None)
```

各スペクトログラムを PyTorch テンソルに変換します。リストを返します。

### `write`

```python
write(self, target, *args, **kwargs)
```

リストをファイルに書き出します。
