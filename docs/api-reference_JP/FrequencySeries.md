# FrequencySeries

**継承元:** FrequencySeries

GWpy の FrequencySeries を軽量にラップしたもので、互換性の維持と将来の拡張を目的としています。

## メソッド

### `DictClass`

```python
DictClass(*args: 'Any', **kwargs: 'Any')
```

ラベルをキーとした `FrequencySeries` オブジェクトの順序付きマッピング。

### `abs`

```python
abs(self, axis=None, **kwargs)
```

要素ごとの絶対値を計算します。複素入力 `a + ib` の場合、絶対値は `sqrt(a^2 + b^2)` となります。

### `angle`

```python
angle(self, unwrap: 'bool' = False) -> "'FrequencySeries'"
```

`phase(unwrap=unwrap)` のエイリアス。

### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

現在のシリーズの末尾に別のシリーズを接続します。

**パラメータ:**
- **other** : `Series`
    接続する同一型の別のシリーズ。
- **inplace** : bool, オプション
    インプレースで操作を行うか（デフォルトは True）。
- **pad** : float, オプション
    不連続なシリーズを埋めるための値。
- **gap** : str, オプション
    ギャップの処理方法：'raise'（エラー）、'ignore'（単純結合）、'pad'（ゼロ埋め）。

### `channel`

データに関連付けられた観測チャンネル。

### `copy`

```python
copy(self, order='C')
```

配列のコピーを返します。

### `crop`

```python
crop(self, start=None, end=None, copy=False)
```

指定された x 軸の範囲にシリーズを切り詰めます。

### `degree`

```python
degree(self, unwrap: 'bool' = False) -> "'FrequencySeries'"
```

この FrequencySeries の位相を度（degrees）単位で計算します。

### `diff`

```python
diff(self, n=1, axis=-1)
```

指定した軸方向に n 次の離散差分を計算します。

### `differentiate_time`

```python
differentiate_time(self) -> 'Any'
```

周波数ドメインで時間微分を適用します。
(2 * pi * i * f) を乗算します。変位 -> 速度 -> 加速度の変換に使用されます。

### `dx`

x 軸のサンプル間隔。

### `epoch`

データに関連付けられた GPS エポック。

### `filter`

```python
filter(self, *filt, **kwargs)
```

この `FrequencySeries` にフィルタを適用します。

**パラメータ:**
- **filt**: FIR係数、SOS係数、(分子, 分母)、(零点, 極, ゲイン) など。

### `filterba`

```python
filterba(self, *args, **kwargs)
```

この FrequencySeries に [b, a] フィルタを適用します。
*(gwpy から継承)*

### `find_peaks`

```python
find_peaks(self, threshold: 'Optional[float]' = None, method: 'str' = 'amplitude', **kwargs: 'Any') -> 'Any'
```

シリーズ内のピークを検索します。`scipy.signal.find_peaks` のラッパーです。

### `fit`

```python
fit(self, model: 'str', p0: 'Optional[dict[str, float]]' = None, method: 'str' = 'leastsq', **kwargs: 'Any') -> 'Any'
```

シリーズをモデル関数にフィットさせます。

**パラメータ:**
- **model**: モデル名 ('gaussian', 'power_law' 等) または関数。
- **p0**: パラメータ初期値の辞書。
- **method**: 'leastsq' (デフォルト) または 'mcmc'。

### `from_cupy`

```python
from_cupy(array: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```

CuPy 配列から FrequencySeries を作成します。

### `from_hdf5_dataset`

```python
from_hdf5_dataset(group: 'Any', path: 'str') -> 'Any'
```

HDF5 データセットから FrequencySeries を読み込みます。

### `from_jax`

```python
from_jax(array: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```

JAX 配列から FrequencySeries を作成します。

### `from_pandas`

```python
from_pandas(series: 'Any', **kwargs: 'Any') -> 'Any'
```

pandas.Series から FrequencySeries を作成します。

### `from_root`

```python
from_root(obj: 'Any', return_error: 'bool' = False, **kwargs: 'Any') -> 'Any'
```

ROOT の TGraph または TH1 から FrequencySeries を作成します。

### `from_tf`

```python
from_tf(tensor: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```

tensorflow.Tensor から FrequencySeries を作成します。

### `from_torch`

```python
from_torch(tensor: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```

torch.Tensor から FrequencySeries を作成します。

### `from_xarray`

```python
from_xarray(da: 'Any', **kwargs: 'Any') -> 'Any'
```

xarray.DataArray から FrequencySeries を作成します。

### `group_delay`

```python
group_delay(self) -> 'Any'
```

シリーズの群遅延を計算します。
群遅延は -d(phase)/d(omega) として定義されます。秒単位の群遅延を返します。

### `idct`

```python
idct(self, type: 'int' = 2, norm: 'str' = 'ortho', *, n: 'Optional[int]' = None) -> 'Any'
```

逆離散コサイン変換 (IDCT) を計算して TimeSeries を返します。

### `ifft`

```python
ifft(self, *, mode: 'str' = 'auto', trim: 'bool' = True, original_n: 'Optional[int]' = None, pad_left: 'Optional[int]' = None, pad_right: 'Optional[int]' = None, **kwargs: 'Any') -> 'Any'
```

逆FFTを計算して gwexpy の TimeSeries を返します。過渡応答（transient mode）の復元をサポートしています。

### `inject`

```python
inject(self, other)
```

共通する x 軸の値に沿って、互換性のある 2 つのシリーズを加算します。

### `integrate_time`

```python
integrate_time(self) -> 'Any'
```

周波数ドメインで時間積分を適用します。
(2 * pi * i * f) で除算します。加速度 -> 速度 -> 変位の変換に使用されます。

### `interpolate`

```python
interpolate(self, df)
```

指定された周波数解像度 (df) にシリーズを補間します。

### `is_regular`

この FrequencySeries が等間隔な周波数グリッドを持っている場合に True を返します。

### `median`

```python
median(self, axis=None, **kwargs)
```

指定された軸方向にメディアン（中央値）を計算します。

### `name`

データセットの名称。

### `pad`

```python
pad(self, pad_width, **kwargs)
```

シリーズを新しいサイズにパディングします。

### `phase`

```python
phase(self, unwrap: 'bool' = False) -> "'FrequencySeries'"
```

この FrequencySeries の位相を計算します（ラジアン単位）。

### `plot`

```python
plot(self, xscale='log', **kwargs)
```

データをプロットします。

### `quadrature_sum`

```python
quadrature_sum(self, other: 'Any') -> 'Any'
```

2つのシリーズの振幅の二乗和平方根 `sqrt(self^2 + other^2)` を計算します。

### `read`

```python
read(source, *args, **kwargs)
```

ファイルから FrequencySeries を読み込みます。

### `shift`

```python
shift(self, delta)
```

x 軸方向に `delta` だけシリーズをシフトします。インプレースで動作します。

### `smooth`

```python
smooth(self, width: 'Any', method: 'str' = 'amplitude') -> 'Any'
```

周波数シリーズを平滑化します。

**パラメータ:**
- **width**: 平滑化ウィンドウのサンプル数。
- **method**: 'amplitude' (振幅), 'power' (パワー), 'complex' (複素数), 'db' (デシベル)。

### `to_cupy`

```python
to_cupy(self, dtype: 'Any' = None) -> 'Any'
```

CuPy 配列に変換します。

### `to_db`

```python
to_db(self, ref: 'Any' = 1.0, amplitude: 'bool' = True) -> "'FrequencySeries'"
```

シリーズをデシベル (dB) 単位に変換します。

### `to_hdf5_dataset`

```python
to_hdf5_dataset(self, group: 'Any', path: 'str', *, overwrite: 'bool' = False, compression: 'Optional[str]' = None, compression_opts: 'Any' = None) -> 'Any'
```

HDF5 データセットに書き出します。

### `to_jax`

```python
to_jax(self, dtype: 'Any' = None) -> 'Any'
```

JAX 配列に変換します。

### `to_pandas`

```python
to_pandas(self, index: 'str' = 'frequency', *, name: 'Optional[str]' = None, copy: 'bool' = False) -> 'Any'
```

pandas.Series に変換します。

### `to_polars`

```python
to_polars(self, name: 'Optional[str]' = None, as_dataframe: 'bool' = True, index_column: 'str' = 'frequency') -> 'Any'
```

polars オブジェクトに変換します。

### `to_tf`

```python
to_tf(self, dtype: 'Any' = None) -> 'Any'
```

tensorflow.Tensor に変換します。

### `to_tgraph`

```python
to_tgraph(self, error: 'Optional[Any]' = None) -> 'Any'
```

ROOT の TGraph または TGraphErrors に変換します。

### `to_th1d`

```python
to_th1d(self, error: 'Optional[Any]' = None) -> 'Any'
```

ROOT の TH1D に変換します。

### `to_torch`

```python
to_torch(self, device: 'Optional[str]' = None, dtype: 'Any' = None, requires_grad: 'bool' = False, copy: 'bool' = False) -> 'Any'
```

PyTorch テンソルに変換します。

### `to_xarray`

```python
to_xarray(self, freq_coord: 'str' = 'Hz') -> 'Any'
```

xarray.DataArray に変換します。

### `unit`

データの物理単位。

### `update`

```python
update(self, other, inplace=True)
```

新しいデータを末尾に追加し、同量のデータを先頭から削除して、シリーズを更新します。

### `value_at`

```python
value_at(self, x)
```

指定した `xindex` の値におけるシリーズの値を返します。

### `write`

```python
write(self, target, *args, **kwargs)
```

FrequencySeries をファイルに書き出します。

### `x0`

最初のデータ点の x 座標。

### `xindex`

x 軸上のデータ位置の配列。

### `xspan`

データがカバーする x 軸の範囲 [low, high)。

### `xunit`

x 軸インデックスの単位。

### `zip`

```python
zip(self)
```

`xindex` と `value` 配列をスタックして、2次元の numpy 配列を返します。

### `zpk`

```python
zpk(self, zeros, poles, gain, analog=True)
```

零点・極・ゲイン（ZPK）フィルタを適用します。
