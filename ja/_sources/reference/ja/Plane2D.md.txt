# Plane2D

**継承元:** Array2D

`Array2D` を拡張し、より具体的な平面データを扱うためのクラスです。通常、TimePlaneTransform から抽出されたデータを保持します。

## メソッド

### `axis1`

第1軸（通常は空間または周波数軸）の AxisDescriptor。

### `axis2`

第2軸（通常は空間または周波数軸）の AxisDescriptor。

### `imshow`, `pcolormesh`

matplotlib を使用したプロット。*(Array2D から継承)*

### `plot`

```python
plot(self, method='imshow', **kwargs)
```

平面データをプロットします。

### `rename_axes`

軸の名前を変更します。

### `swapaxes`

```python
swapaxes(self, axis1, axis2)
```

2つの軸を入れ替えます。AxisDescriptor も適切に更新されます。

### `transpose`

配列の次元を転置します。

### `x0`, `xindex`, `xspan`, `xunit`

x 軸（第1軸）のプロパティ。

### `y0`, `yindex`, `yspan`, `yunit`

y 軸（第2軸）のプロパティ。

*(その他のメソッドは Array2D から継承されます)*
