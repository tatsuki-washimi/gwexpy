# TimePlaneTransform

**継承元:** object

時間-周波数変換などで生成される、(time, axis1, axis2) 構造を持つ3次元データのコンテナ。

このクラスは Array3D をラップし、セマンティック構造を強制します：
- 軸 0 は「時間」
- 軸 1 と 2 は対称的な空間/周波数次元

## メソッド

### `__init__`

```python
__init__(self, data3d, *, kind='custom', meta=None)
```

TimePlaneTransform を初期化します。

パラメータ
----------
data3d : Array3D or tuple
    基盤となる3Dデータ。
    推奨: `Array3D` インスタンス。
    サポートされるタプル形式: (value, time_axis, axis1, axis2, unit/metadata)。
kind : str, optional
    変換の種類を説明する文字列 (例: "stlt", "bispectrum")。デフォルトは "custom"。
meta : dict, optional
    追加のメタデータ辞書。デフォルトは None (空の辞書として保存)。

### `at_sigma`

```python
at_sigma(self, sigma)
```

特定の sigma インデックス（軸1が sigma の場合）または値で2D平面 (Spectrogram ライク) を抽出します。

これは軸 1 が sigma であることを前提とします。

### `at_time`

```python
at_time(self, t, *, method='nearest')
```

特定の時刻 `t` で Plane2D を抽出します。

パラメータ
----------
t : Quantity or float
    時刻値。float の場合、時間軸の単位であると仮定されます。
method : str, optional
    "nearest" (デフォルト)。将来のバージョンでは補間をサポートする可能性があります。

戻り値
-------
Plane2D

### `axes`

3つの AxisDescriptor を返します: (time, axis1, axis2)。

### `axis1`

最初の対称軸（軸 1）の AxisDescriptor。

### `axis2`

2番目の対称軸（軸 2）の AxisDescriptor。

### `kind`

変換の種類を説明する文字列 (例: 'stlt', 'bispectrum')。

### `meta`

追加のメタデータ辞書。

### `ndim`

次元数 (常に 3)。

### `plane`

```python
plane(self, drop_axis, drop_index, *, axis1=None, axis2=None)
```

1つの軸に沿った特定のインデックスでスライスして2D平面を抽出します。

パラメータ
----------
drop_axis : int or str
    スライスする軸（削除する軸）。
drop_index : int
    `drop_axis` に沿って選択する整数インデックス。
axis1 : str or int, optional
    結果の Plane2D の汎用軸 1。
axis2 : str or int, optional
    結果の Plane2D の汎用軸 2。

戻り値
-------
Plane2D

### `shape`

3Dデータ配列の形状 (time, axis1, axis2)。

### `times`

時間軸（軸 0）の座標配列。

### `to_array3d`

```python
to_array3d(self)
```

基盤となる Array3D オブジェクトを返します（上級者向け）。

### `unit`

データ値の物理単位。

### `value`

numpy 配列としての基盤データ値。
