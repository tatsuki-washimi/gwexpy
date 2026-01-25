# SpectrogramDict

**継承元:** PhaseMethodsMixin, UserDict

Spectrogram オブジェクトの辞書。

.. note::
   Spectrogram オブジェクトはメモリを大量に消費する可能性があります。
   可能な限り `inplace=True` を使用してコンテナをその場で更新してください。

## メソッド

### `__init__`

```python
__init__(self, dict=None, **kwargs)
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

辞書内の各スペクトログラムからロバスト ASD を推定します（FrequencySeriesDict を返します）。

### `crop`

```python
crop(self, t0, t1, inplace=False)
```

各スペクトログラムを時間方向でクロップします。

パラメータ
----------
t0, t1 : float
    開始時刻と終了時刻。
inplace : bool, optional
    True の場合、その場で変更します。

戻り値
-------
SpectrogramDict

### `crop_frequencies`

```python
crop_frequencies(self, f0, f1, inplace=False)
```

各スペクトログラムを周波数方向でクロップします。

パラメータ
----------
f0, f1 : float or Quantity
    開始周波数と終了周波数。
inplace : bool, optional
    True の場合、その場で変更します。

戻り値
-------
SpectrogramDict

### `degree`

```python
degree(self, unwrap: 'bool' = False) -> "'SpectrogramDict'"
```

各スペクトログラムの位相（度単位）を計算します。

### `interpolate`

```python
interpolate(self, dt, df, inplace=False)
```

各スペクトログラムを新しい解像度に補間します。

パラメータ
----------
dt : float
    新しい時間解像度。
df : float
    新しい周波数解像度。
inplace : bool, optional
    True の場合、その場で変更します。

戻り値
-------
SpectrogramDict

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

辞書をスペクトログラムとパーセンタイルサマリーを並べてプロットします。

### `radian`

```python
radian(self, unwrap: 'bool' = False) -> "'SpectrogramDict'"
```

各スペクトログラムの位相（ラジアン単位）を計算します。

### `rebin`

```python
rebin(self, dt, df, inplace=False)
```

各スペクトログラムを新しい時間/周波数解像度にリビンします。

パラメータ
----------
dt : float
    新しい時間ビンサイズ。
df : float
    新しい周波数ビンサイズ。
inplace : bool, optional
    True の場合、その場で変更します。

戻り値
-------
SpectrogramDict

### `to_matrix`

```python
to_matrix(self)
```

SpectrogramMatrix に変換します。

戻り値
-------
SpectrogramMatrix
    (N, Time, Freq) の3D配列。

### `to_cupy` / `to_dask` / `to_jax` / `to_tensorflow` / `to_torch`

各アイテムを対応するフレームワークのテンソル/配列に変換します。辞書を返します。

### `write`

```python
write(self, target, *args, **kwargs)
```

辞書をファイルに書き込みます。
