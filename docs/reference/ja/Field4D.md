# Field4D

**継承元:** Array4D, AxisApiMixin, StatisticalMethodsMixin, Array

時空間物理場 (4D Field) クラス群。時間・周波数軸 (axis 0) と空間3軸 (axis 1-3) を持ち、ドメイン遷移やFFT操作を提供します。

**主な特徴**: 全てのスライシング操作において常に 4D 構造を維持します。整数インデックスを指定した場合でも、該当する次元は削除されず、長さ1の軸として保持されます（Field4D を返します）。

## メソッド

### `fft_time`

```python
fft_time(self, nfft=None)
```

時間軸（第0軸）に沿って FFT を計算します。

GWpy の ``TimeSeries.fft()`` と同様の正規化（rfft / nfft、および DC 成分以外の 2 倍補正）を適用します。

引数
----------
nfft : int, optional
    FFT の長さ。None の場合、第0軸の長さが使用されます。

戻り値
-------
Field4D
    ``axis0_domain='frequency'`` に遷移した新しい Field4D。

### `ifft_time`

```python
ifft_time(self, nout=None)
```

周波数軸（第0軸）に沿って逆 FFT を計算します。

``fft_time()`` の逆操作（GWpy の ``FrequencySeries.ifft()`` 相当）を適用します。

引数
----------
nout : int, optional
    出力時間シリーズの長さ。None の場合、``(n_freq - 1) * 2`` として計算されます。

戻り値
-------
Field4D
    ``axis0_domain='time'`` に戻った新しい Field4D。

### `fft_space`

```python
fft_space(self, axes=None, n=None)
```

空間軸に沿って FFT を計算します。

符号付き両側 FFT (numpy.fft.fftn) を使用し、角波数 (k = 2π·fftfreq) を生成します。

引数
----------
axes : iterable of str, optional
    変換対象の軸名（例: ['x', 'y']）。None の場合、'real' ドメインのすべての空間軸を変換します。
n : tuple of int, optional
    各軸の FFT 長さ。

戻り値
-------
Field4D
    指定された軸が 'k' ドメインに遷移した新しい Field4D。

### `ifft_space`

```python
ifft_space(self, axes=None, n=None)
```

k空間軸に沿って逆 FFT を計算します。

引数
----------
axes : iterable of str, optional
    変換対象の軸名（例: ['kx', 'ky']）。None の場合、'k' ドメインのすべての空間軸を変換します。
n : tuple of int, optional
    各軸の出力長さ。

戻り値
-------
Field4D
    指定された軸が 'real' ドメインに戻った新しい Field4D。

### `wavelength`

```python
wavelength(self, axis)
```

波数軸から波長を計算します。

引数
----------
axis : str or int
    kドメインの軸名またはインデックス。

戻り値
-------
`~astropy.units.Quantity`
    波長値 (λ = 2π / |k|)。k=0 の場合は inf を返します。

## プロパティ

| プロパティ | 説明 |
|-----------|------|
| `axis0_domain` | 第0軸のドメイン: 'time' または 'frequency' |
| `space_domains` | 空間軸の名前とドメイン ('real' または 'k') の対応 |
| `axis_names` | 全軸の名前 |
| `unit` | データの物理単位 |
| `axes` | 各次元の AxisDescriptor のタプル |

## スライシングの挙動

Field4D は `__getitem__` をオーバーライドしており、常に Field4D を返します。
ある軸に対して整数インデックスを指定した場合、その軸は削除されず、長さ1の軸として残ります。

```python
>>> field.shape
(100, 32, 32, 32)
>>> field[0].shape
(1, 32, 32, 32)
```
