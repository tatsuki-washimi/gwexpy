# Array2D

**継承元:** AxisApiMixin, Array2D

統合された軸 API を備えた 2次元配列です。

## メソッド

### `T`

転置された配列のビュー。

### `abs`

要素ごとの絶対値。

### `append`

シリーズの末尾に接続します。

### `axes`

各次元の AxisDescriptor オブジェクトのタプル。各記述子には軸名とインデックス値が含まれます。

### `axis`

```python
axis(self, key: Union[int, str]) -> gwexpy.types.axis.AxisDescriptor
```

インデックスまたは名前で軸記述子を取得します。

### `axis_names`

すべての軸の名前のタプル。

### `copy`, `crop`, `diff`

コピー、切り抜き、離散差分。

### `dx`, `dy`

x 軸および y 軸のサンプル間隔。

### `flatten`

配列を 1次元に平坦化したコピーを返します。単位情報のみ保持され、インデックス情報は削除されます。

### `imshow`, `pcolormesh`

matplotlib を使用したプロット。*(gwpy から継承)*

### `plot`

```python
plot(self, method='imshow', **kwargs)
```

2次元データをプロットします。

### `rename_axes`

```python
rename_axes(self, mapping: Dict[str, str], *, inplace=False)
```

軸の名前を変更します。

### `sel`

```python
sel(self, indexers=None, *, method='nearest', **kwargs)
```

軸名と座標値を指定してサンプルを選択します。

### `swapaxes`

軸を入れ替えたビューを返します。

### `transpose`

次元の順序を入れ替えます。

### `unit`

データの物理単位。

### `value_at`

指定した座標 (x, y) における値を返します。

### `write`, `read`

ファイルの入出力。

### `x0`, `xindex`, `xspan`, `xunit`

x 軸のプロパティ。

### `y0`, `yindex`, `yspan`, `yunit`

y 軸のプロパティ。
