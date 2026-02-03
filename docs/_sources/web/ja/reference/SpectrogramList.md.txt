# SpectrogramList

**継承元:** PhaseMethodsMixin, UserList

Spectrogram オブジェクトのリスト。
参考: TimeSeriesList に似ていますが、2D Spectrogram 用です。

.. note::
   Spectrogram オブジェクトはメモリを大量に消費する可能性があります。
   ディープコピーを避けるため、可能な限り `inplace=True` を使用してください。

## メソッド

### `__init__`

```python
__init__(self, initlist=None)
```

self を初期化します。正確なシグネチャは help(type(self)) を参照してください。

*(PhaseMethodsMixin から継承)*

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

`phase(unwrap=unwrap, deg=deg)` のエイリアス。

### `bootstrap_asd`

```python
bootstrap_asd(self, *args, **kwargs)
```

リスト内の各スペクトログラムからロバスト ASD を推定します（FrequencySeriesList を返します）。

### `crop`

```python
crop(self, t0, t1, inplace=False)
```

各スペクトログラムをクロップします。

### `crop_frequencies`

```python
crop_frequencies(self, f0, f1, inplace=False)
```

周波数をクロップします。

### `degree`

```python
degree(self, unwrap: 'bool' = False) -> "'SpectrogramList'"
```

各スペクトログラムの位相（度単位）を計算します。

### `interpolate`

```python
interpolate(self, dt, df, inplace=False)
```

各スペクトログラムを補間します。

### `phase`

```python
phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

データの位相を計算します。

パラメータ
----------
unwrap : `bool`, optional
    `True` の場合、不連続性を除去するために位相をアンラップします。デフォルトは `False`。
deg : `bool`, optional
    `True` の場合、位相を度で返します。デフォルトは `False`（ラジアン）。

戻り値
-------
`Series` or `Matrix` or `Collection`
    データの位相。

### `plot`

```python
plot(self, **kwargs)
```

すべてのスペクトログラムを縦に積み重ねてプロットします。

### `plot_summary`

```python
plot_summary(self, **kwargs)
```

リストをスペクトログラムとパーセンタイルサマリーを並べてプロットします。

### `write`

```python
write(self, target, *args, **kwargs)
```

SpectrogramList をファイルに書き込みます。

HDF5 出力では `layout` を指定できます（デフォルトは GWpy 互換の dataset-per-entry）。

```python
sgl.write("out.h5", format="hdf5")               # GWpy互換（既定）
sgl.write("out.h5", format="hdf5", layout="group")  # 旧形式（group-per-entry）
```

> [!WARNING]
> 信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。

pickle 可搬性メモ: gwexpy の `SpectrogramList` は unpickle 時に builtins の `list` を返します
（中身は GWpy の `Spectrogram`、読み込み側に gwexpy は不要です）。

### `radian`

```python
radian(self, unwrap: 'bool' = False) -> "'SpectrogramList'"
```

各スペクトログラムの位相（ラジアン単位）を計算します。

### `rebin`

```python
rebin(self, dt, df, inplace=False)
```

各スペクトログラムをリビンします。

### `to_matrix`

```python
to_matrix(self)
```

SpectrogramMatrix (N, Time, Freq) に変換します。

### `to_cupy` / `to_dask` / `to_jax` / `to_tensorflow` / `to_torch`

各アイテムを対応するフレームワークのテンソル/配列に変換します。リストを返します。

### `write`

```python
write(self, target, *args, **kwargs)
```

リストをファイルに書き込みます。
