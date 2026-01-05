# SpectrogramMatrix

**継承元:** PhaseMethodsMixin, SpectrogramMatrixCoreMixin, SpectrogramMatrixAnalysisMixin, SeriesMatrix

Spectrogram データ（時間-周波数マップ）のための評価行列。

このクラスは、以下の構造を持つ Spectrogram のコレクションを表します：
- 3D: (Batch, Time, Frequency)
- 4D: (Row, Col, Time, Frequency)

SeriesMatrix を継承しており、強力なインデクシング、メタデータ管理、および解析機能（スライス、補間、統計）を提供します。

## メソッド

### `MetaDataMatrix`

要素ごとのメタデータを含む行列。

### `N_samples`

x軸（時間軸）に沿ったサンプル数。

### `T`

行列の転置（行と列を入れ替え）。

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

`phase(unwrap=unwrap, deg=deg)` のエイリアス。

### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

別の行列をサンプル軸（時間軸）に沿って結合します。

### `append_exact`

```python
append_exact(self, other, inplace=False, pad=None, gap=None, tol=3.814697265625e-06)
```

厳密な連続性チェックを行い、別の行列を結合します。

### `astype`

```python
astype(self, dtype, copy=True)
```

行列データを指定された型にキャストします。

### `channel_names`

全要素名のフラットなリスト。

### `channels`

各行列要素のチャンネル識別子の2次元配列。

### `col_index`

```python
col_index(self, key)
```

列キーの整数インデックスを取得します。

*(Inherited from `SeriesMatrix`)*

### `col_keys`

```python
col_keys(self)
```

全ての列のキー（ラベル）を取得します。

*(Inherited from `SeriesMatrix`)*

### `conj`

```python
conj(self)
```

行列の複素共役。

### `copy`

```python
copy(self, order='C')
```

この行列のディープコピーを作成します。

### `crop`

```python
crop(self, start=None, end=None, copy=False)
```

サンプル軸（時間軸）に沿って指定範囲で行列を切り抜きます。

### `degree`

```python
degree(self, unwrap: 'bool' = False) -> 'Any'
```

行列の位相を度数法（degree）で計算します。

### `det`

```python
det(self)
```

各サンプル点における行列式を計算します。

### `df`

_ドキュメントはありません。_

### `diagonal`

```python
diagonal(self, output: 'str' = 'list')
```

行列から対角要素を抽出します。

### `dict_class`

```python
dict_class(dict=None, **kwargs)
```

Spectrogram オブジェクトの辞書。

.. note::
   Spectrogram オブジェクトはメモリを大量に消費する可能性があります。
   可能な限り `inplace=True` を使用して、コンテナをインプレースで更新してください。

### `diff`

```python
diff(self, n=1, axis=None)
```

サンプル軸に沿って n 次の離散差分を計算します。

### `dt`

時間間隔 (dx)。

### `duration`

サンプルがカバーする期間。

### `dx`

x軸（時間軸）上のサンプル間のステップサイズ。

### `f0`

_ドキュメントはありません。_

### `frequencies`

周波数配列 (yindex)。

### `get_index`

```python
get_index(self, key_row: 'Any', key_col: 'Any') -> 'tuple[int, int]'
```

指定されたキーに対する (行, 列) の整数インデックスを取得します。

### `imag`

```python
imag(self)
```

行列の虚部。

### `interpolate`

```python
interpolate(self, xindex, **kwargs)
```

新しいサンプル軸へ行列を補間します。

### `inv`

```python
inv(self, swap_rowcol: 'bool' = True)
```

各サンプル点における逆行列を計算します。

### `is_compatible`

```python
is_compatible(self, other: 'Any') -> 'bool'
```

別の SpectrogramMatrix/オブジェクトとの互換性をチェックします。
データ形状（時間軸）とメタデータ形状（Batch/Col）の間の不一致によるループ範囲の問題を回避するため、`SeriesMatrix.is_compatible` をオーバーライドしています。

### `is_compatible_exact`

```python
is_compatible_exact(self, other: 'Any') -> 'bool'
```

別の行列との厳密な互換性をチェックします。

### `is_contiguous`

```python
is_contiguous(self, other: 'Any', tol: 'float' = 3.814697265625e-06) -> 'int'
```

この行列が別の行列と連続しているかチェックします。

### `is_contiguous_exact`

```python
is_contiguous_exact(self, other: 'Any', tol: 'float' = 3.814697265625e-06) -> 'int'
```

厳密な形状マッチングで連続性をチェックします。

### `is_regular`

このシリーズが等間隔グリッドを持つ場合 True を返します。

### `keys`

```python
keys(self) -> 'tuple[tuple[Any, ...], tuple[Any, ...]]'
```

行と列の両方のキーを取得します。

### `list_class`

```python
list_class(initlist=None)
```

Spectrogram オブジェクトのリスト。
参照: TimeSeriesList に似ていますが、2D Spectrogram 用です。

.. note::
   Spectrogram オブジェクトはメモリを大量に消費する可能性があります。
   ディープコピーを避けるため、可能な限り `inplace=True` を使用してください。

### `loc`

値に直接アクセスするためのラベルベースのインデクサ。

### `max`

```python
max(self, axis=None, out=None, keepdims=False, initial=None, where=True, ignore_nan=True)
```

指定された軸に沿った最大値を返します。

完全なドキュメントについては `numpy.amax` を参照してください。

See Also
--------
numpy.amax : equivalent function

*(Inherited from `ndarray`)*

### `mean`

```python
mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, ignore_nan=True)
```

指定された軸に沿った配列要素の平均を返します。

完全なドキュメントについては `numpy.mean` を参照してください。

See Also
--------
numpy.mean : equivalent function

*(Inherited from `ndarray`)*

### `median`

```python
median(self, axis=None, out=None, overwrite_input=False, keepdims=False, ignore_nan=True)
```

_ドキュメントはありません。_

### `min`

```python
min(self, axis=None, out=None, keepdims=False, initial=None, where=True, ignore_nan=True)
```

指定された軸に沿った最小値を返します。

完全なドキュメントについては `numpy.amin` を参照してください。

See Also
--------
numpy.amin : equivalent function

*(Inherited from `ndarray`)*

### `names`

各行列要素の名前の2次元配列。1Dの場合は channel_names のエイリアス。

### `pad`

```python
pad(self, pad_width, **kwargs)
```

サンプル軸に沿って行列をパディング（埋め合わせ）します。

### `phase`

```python
phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

データの位相を計算します。

Parameters
----------
unwrap : `bool`, optional
    `True` の場合、不連続性を除去するために位相をアンラップします。
    デフォルトは `False` です。
deg : `bool`, optional
    `True` の場合、位相を度（degrees）で返します。
    デフォルトは `False`（ラジアン）です。
**kwargs
    基礎となる計算に渡される追加引数。

Returns
-------
`Series` of `Matrix` or `Collection`
    データの位相。

### `plot`

```python
plot(self, **kwargs)
```

gwexpy.plot.Plot を使用して行列データをプロットします。

### `plot_summary`

```python
plot_summary(self, **kwargs)
```

行列をサイドバイサイドのスペクトログラムとパーセンタイルサマリーとしてプロットします。

### `prepend`

```python
prepend(self, other, inplace=True, pad=None, gap=None, resize=True)
```

別の行列をサンプル軸の先頭に結合します。

### `prepend_exact`

```python
prepend_exact(self, other, inplace=False, pad=None, gap=None, tol=3.814697265625e-06)
```

厳密な連続性チェックを行い、別の行列を先頭に結合します。

### `radian`

```python
radian(self, unwrap: 'bool' = False) -> 'Any'
```

行列の位相をラジアンで計算します。

### `read`

```python
read(source, format=None, **kwargs)
```

ファイルから SeriesMatrix を読み込みます。

Parameters
----------
source : str or path-like
    読み込むファイルへのパス。
format : str, optional
    ファイルフォーマット。None の場合、拡張子から推測されます。
**kwargs
    リーダーに渡される追加引数。

Returns
-------
SeriesMatrix
    読み込まれた行列。

利用可能な組み込みフォーマットは以下の通りです：

======== ==== ===== =============
 Format  Read Write Auto-identify
======== ==== ===== =============
     ats  Yes    No            No
  dttxml  Yes    No            No
     gbd  Yes    No            No
    gse2  Yes    No            No
    knet  Yes    No            No
      li  Yes    No            No
     lsf  Yes    No            No
     mem  Yes    No            No
miniseed  Yes    No            No
     orf  Yes    No            No
     sac  Yes    No            No
     sdb  Yes    No            No
  sqlite  Yes    No            No
 sqlite3  Yes    No            No
 taffmat  Yes    No            No
    tdms  Yes    No            No
     wav  Yes    No            No
     wdf  Yes    No            No
     win  Yes    No            No
   win32  Yes    No            No
     wvf  Yes    No            No
======== ==== ===== =============

### `real`

```python
real(self)
```

行列の実部。

### `reshape`

```python
reshape(self, shape, order='C')
```

行列の次元を変更します。

### `rms`

```python
rms(self, axis=None, keepdims=False, ignore_nan=True)
```

_ドキュメントはありません。_

### `row_index`

```python
row_index(self, key)
```

行キーの整数インデックスを取得します。

*(Inherited from `SeriesMatrix`)*

### `row_keys`

```python
row_keys(self)
```

全ての行のキー（ラベル）を取得します。

*(Inherited from `SeriesMatrix`)*

### `schur`

```python
schur(self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None)
```

ブロック行列のシューア補完を計算します。

### `series_class`

```python
series_class(data, unit=None, t0=None, dt=None, f0=None, df=None, times=None, frequencies=None, name=None, channel=None, **kwargs)
```

追加の相互運用メソッドで gwpy.spectrogram.Spectrogram を拡張します。

### `shape3D`

行列の形状を 3-タプル (n_rows, n_cols, n_samples) として返します。
4D行列（スペクトログラム）の場合、最後の次元は周波数と見なされるため、n_samples は _x_axis_index によって決定されます。

*(Inherited from `SeriesMatrix`)*

### `shift`

```python
shift(self, delta)
```

サンプル軸を定数オフセットだけシフトします。

### `std`

```python
std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, ignore_nan=True)
```

指定された軸に沿った配列要素の標準偏差を返します。

完全なドキュメントについては `numpy.std` を参照してください。

See Also
--------
numpy.std : equivalent function

*(Inherited from `ndarray`)*

### `step`

```python
step(self, where: 'str' = 'post', **kwargs: 'Any') -> 'Any'
```

行列をステップ関数としてプロットします。

### `submatrix`

```python
submatrix(self, row_keys, col_keys)
```

特定の行と列を選択して部分行列を抽出します。

### `t0`

開始時刻 (x0)。

### `times`

時刻配列 (xindex)。

### `to_cupy`

```python
to_cupy(self, dtype=None) -> Any
```

CuPy Array に変換します。

### `to_dask`

```python
to_dask(self, chunks='auto') -> Any
```

Dask Array に変換します。

### `to_dict`

```python
to_dict(self)
```

SpectrogramDict に変換します。

### `to_dict_flat`

```python
to_dict_flat(self) -> 'dict[str, Series]'
```

行列を名前から Series へのフラットな辞書に変換します。

### `to_hdf5`

```python
to_hdf5(self, filepath, **kwargs)
```

行列を HDF5 ファイルに書き込みます。

### `to_jax`

```python
to_jax(self) -> Any
```

JAX Array に変換します。

### `to_list`

```python
to_list(self)
```

SpectrogramList に変換します。

### `to_pandas`

```python
to_pandas(self, format='wide')
```

行列を pandas DataFrame に変換します。

### `to_series_1Dlist`

```python
to_series_1Dlist(self)
```

行列を Spectrogram オブジェクトのフラットな 1D リストに変換します。

### `to_series_2Dlist`

```python
to_series_2Dlist(self)
```

行列を Spectrogram オブジェクトの 2D ネストリストに変換します。

### `to_tensorflow`

```python
to_tensorflow(self, dtype: Any = None) -> Any
```

tensorflow.Tensor に変換します。

### `to_torch`

```python
to_torch(self, device: Optional[str] = None, dtype: Any = None, requires_grad: bool = False, copy: bool = False) -> Any
```

torch.Tensor に変換します。

### `to_zarr`

```python
to_zarr(self, store, path=None, **kwargs) -> Any
```

Zarr ストレージに保存します。

### `trace`

```python
trace(self)
```

行列のトレース（対角要素の和）を計算します。

### `transpose`

```python
transpose(self, *axes)
```

行と列を転置します。サンプル軸は 2 のまま維持します。

### `units`

各行列要素の単位の2次元配列。

### `update`

```python
update(self, other, inplace=True, pad=None, gap=None)
```

サイズ変更なしで行列に追加して更新します（ローリングバッファスタイル）。

### `value`

データ値の基礎となる numpy 配列。

### `value_at`

```python
value_at(self, x)
```

特定の x 軸位置における行列値を取得します。

### `var`

```python
var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, ignore_nan=True)
```

指定された軸に沿った配列要素の分散を返します。

完全なドキュメントについては `numpy.var` を参照してください。

See Also
--------
numpy.var : equivalent function

*(Inherited from `ndarray`)*

### `write`

```python
write(self, target, format=None, **kwargs)
```

行列をファイルに書き込みます。

### `x0`

サンプル軸の開始値。

### `xarray`

サンプル軸の値を返します。

### `xindex`

サンプル軸インデックス配列。

### `xspan`

サンプル軸の全範囲をタプル (start, end) として返します。

### `xunit`

_ドキュメントはありません。_
