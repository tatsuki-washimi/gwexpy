# SpectrogramMatrix

**継承元:** ndarray

スペクトログラムデータのマトリックス（行列）。
形状は通常、スペクトログラムのリストの場合は (N, Time, Frequencies)、スペクトログラムのマトリックスの場合は (N, M, Time, Frequencies) となります。

## 属性
- **times** : array-like (時間軸)
- **frequencies** : array-like (周波数軸)
- **unit** : Unit (単位)
- **name** : str (名称)

## メソッド

### `col_keys`

```python
col_keys(self)
```

列のメタデータキーのリストを返します。

### `mean`

```python
mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True)
```

指定された軸方向に配列要素の平均を返します。
詳細は `numpy.mean` のドキュメントを参照してください。

*( `ndarray` から継承)*

### `plot`

```python
plot(self, **kwargs)
```

マトリックスデータをプロットします。

- 3D (Batch, Time, Freq) の場合、スペクトログラムを垂直方向にリストとしてプロットします。
- 4D (Row, Col, Time, Freq) の場合、スペクトログラムをグリッド（格子）状にプロットします。
- 3D で行/列のメタデータがグリッドを構成する場合は、1列ではなくグリッドとしてプロットされます。

**オプション引数:**
- **monitor**: int または (row, col) - 単一の要素をプロットする場合
- **method**: 'pcolormesh' (デフォルト)
- **separate**: bool (4Dの場合はデフォルトで True)
- **geometry**: tuple (形状に基づいたデフォルト値)
- **yscale**: 'log' (デフォルト)
- **xscale**: 'linear' (デフォルト)

### `row_keys`

```python
row_keys(self)
```

行のメタデータキーのリストを返します。

### `to_cupy`

```python
to_cupy(self, dtype=None)
```

CuPy 配列に変換します。

### `to_torch`

```python
to_torch(self, device=None, dtype=None, requires_grad=False, copy=False)
```

PyTorch テンソルに変換します。
