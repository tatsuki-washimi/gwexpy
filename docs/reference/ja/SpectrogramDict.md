# SpectrogramDict

**継承元:** UserDict

`Spectrogram` オブジェクトのディクショナリ。

.. note::
   Spectrogram オブジェクトはメモリ上で非常に大きくなる可能性があります。
   ディープコピーを避けるため、可能な限り `inplace=True` を使用してコンテナを直接更新してください。

## メソッド

### `__init__`

```python
__init__(self, dict=None, **kwargs)
```

初期化。

*( `MutableMapping` から継承)*

### `crop`

```python
crop(self, t0, t1, inplace=False)
```

各スペクトログラムを時間軸でクロップします。

**パラメータ:**
- **t0, t1** : float
    開始時間と終了時間。
- **inplace** : bool, オプション
    Trueの場合、インプレースで変更します。

**戻り値:**
- **SpectrogramDict**

### `crop_frequencies`

```python
crop_frequencies(self, f0, f1, inplace=False)
```

各スペクトログラムを周波数軸でクロップします。

**パラメータ:**
- **f0, f1** : float または Quantity
    開始周波数と終了周波数。
- **inplace** : bool, オプション
    Trueの場合、インプレースで変更します。

**戻り値:**
- **SpectrogramDict**

### `interpolate`

```python
interpolate(self, dt, df, inplace=False)
```

各スペクトログラムを新しい解像度に補間します。

**パラメータ:**
- **dt** : float
    新しい時間解像度。
- **df** : float
    新しい周波数解像度。
- **inplace** : bool, オプション
    Trueの場合、インプレースで変更します。

**戻り値:**
- **SpectrogramDict**

### `plot`

```python
plot(self, **kwargs)
```

すべてのスペクトログラムを垂直方向に並べてプロットします。

### `read`

```python
read(self, source, *args, **kwargs)
```

HDF5ファイルからディクショナリを読み込みます。ファイルのキーがディクショナリのキーになります。

### `rebin`

```python
rebin(self, dt, df, inplace=False)
```

各スペクトログラムを新しい時間/周波数解像度にリビン（再ビン化）します。

**パラメータ:**
- **dt** : float
    新しい時間ビンサイズ。
- **df** : float
    新しい周波数ビンサイズ。
- **inplace** : bool, オプション
    Trueの場合、インプレースで変更します。

**戻り値:**
- **SpectrogramDict**

### `to_cupy`

```python
to_cupy(self, dtype=None)
```

CuPy配列のディクショナリに変換します。

### `to_matrix`

```python
to_matrix(self)
```

SpectrogramMatrix に変換します。

**戻り値:**
- **SpectrogramMatrix**
    (N, Time, Freq) の 3D 配列。

### `to_torch`

```python
to_torch(self, device=None, dtype=None)
```

PyTorch テンソルのディクショナリに変換します。

### `update`

```python
update(self, other=None, **kwargs)
```

マッピングや反復可能オブジェクトからディクショナリを更新します。

*( `UserDict` から継承)*

### `write`

```python
write(self, target, *args, **kwargs)
```

ディクショナリをファイルに書き出します。
