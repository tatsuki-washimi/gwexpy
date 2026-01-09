# TimePlaneTransform / LaplaceGram

**継承元:** object

時間-周波数変換などの手法によって生成される、(time, axis1, axis2) 構造を持つ 3D データを保持するためのコンテナです。

このクラスは Array3D をラップし、以下のセマンティック構造を強制します：
- 軸 0 : "time" (時間)
- 軸 1, 2 : 対称的な空間または周波数の次元

**LaplaceGram** は STLT (短時間ラプラス変換) 用の特殊な TimePlaneTransform であり、
軸 1 は "sigma" (実部 s)、軸 2 は "frequency" (周波数) となります。

## メソッド

### `__init__`

```python
__init__(self, data3d, *, kind='custom', meta=None)
```

TimePlaneTransform を初期化します。

**パラメータ:**
- **data3d** : Array3D または tuple
    基となる 3D データ。`Array3D` インスタンスを推奨します。
    タプル形式もサポート：(value, time_axis, axis1, axis2, unit/metadata)。
- **kind** : str, オプション
    変換の種類を示す文字列（例: "stlt", "bispectrum"）。デフォルトは "custom"。
- **meta** : dict, オプション
    追加のメタデータディクショナリ。デフォルトは None（空の辞書として保存）。

### `at_sigma`

```python
at_sigma(self, sigma)
```

特定の sigma インデックスまたは値における 2D 平面 (スペクトログラム風) を抽出します。
(軸1が "sigma" である場合、例えば `LaplaceGram` で利用可能)。

**パラメータ:**
- **sigma** : float または int
    Sigma の値またはインデックス。

**戻り値:**
- **Plane2D** (スペクトログラム風)

### `at_time`

```python
at_time(self, t, *, method='nearest')
```

特定の時間 `t` における Plane2D を抽出します。

**パラメータ:**
- **t** : Quantity または float
    時間。float の場合、時間軸の単位であるとみなされます。
- **method** : str, オプション
    "nearest" (デフォルト)。将来のバージョンで補間をサポートする可能性があります。

**戻り値:**
- **Plane2D**

### `axes`

3つの AxisDescriptor (time, axis1, axis2) を返します。

### `axis1`

第1の対称軸 (axis 1) の AxisDescriptor。

### `axis2`

第2の対称軸 (axis 2) の AxisDescriptor。

### `kind`

変換の種類を示す文字列（'stlt', 'bispectrum' など）。

### `meta`

追加のメタデータディクショナリ。

### `ndim`

次元数（常に 3）。

### `plane`

```python
plane(self, drop_axis, drop_index, *, axis1=None, axis2=None)
```

特定の軸に沿ったインデックスでスライスし、2D 平面を抽出します。

**パラメータ:**
- **drop_axis** : int または str
    スライス（削除）する軸。
- **drop_index** : int
    `drop_axis` に沿って選択する整数のインデックス。
- **axis1** : str または int, オプション
    生成される Plane2D の軸1の名称またはインデックス。
- **axis2** : str または int, オプション
    生成される Plane2D の軸2の名称またはインデックス。

**戻り値:**
- **Plane2D**

### `shape`

3D データ配列の形状 (time, axis1, axis2)。

### `times`

時間軸（軸 0）の座標配列。

### `to_array3d`

```python
to_array3d(self)
```

基となる Array3D オブジェクトを返します（高度な用途向け）。

### `unit`

データ値の物理単位。

### `value`

基となるデータ値を numpy 配列として返します。
