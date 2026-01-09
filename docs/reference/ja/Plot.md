# Plot

**継承元:** Plot

`gwpy.plot.Plot` の拡張クラスです。`gwexpy.types.SeriesMatrix` 引数を自動的に個別の `gwpy.types.Series` オブジェクトに展開して処理し、可能な限り行列のレイアウトとメタデータを保持します。

## メソッド

### `__init__`

```python
__init__(self, *args, **kwargs)
```

**パラメータ:**
- **figsize**: float のタプル (width, height) - インチ単位の図の大きさ。
- **dpi**: float - 解像度（dots per inch）。
- **facecolor**: 背景色。
- **edgecolor**: 枠線の色。
- **linewidth**: 枠線の幅。
- **layout**: {'constrained', 'compressed', 'tight', None} - レイアウトメカニズム。ラベルの重なりを防ぐために 'constrained' や 'tight' が推奨されます。

### `add_segments_bar`

```python
add_segments_bar(self, segments, ax=None, height=0.14, pad=0.1, sharex=True, location='bottom', **plotargs)
```

状態情報を示すセグメントバーを図に追加します。デフォルトでは、メインの x 軸の直下（または直上）に表示されます。

**パラメータ:**
- **segments**: `DataQualityFlag` または `SegmentList` - 状態セグメント。
- **location**: 'top' または 'bottom' - バーの配置場所。

### `close`

```python
close(self)
```

プロットを閉じ、メモリを解放します。

### `colorbar`

```python
colorbar(self, mappable=None, cax=None, ax=None, fraction=0.0, use_axesgrid=True, emit=True, **kwargs)
```

現在の `Plot` にカラーバーを追加します。このメソッドは、親 Axes のサイズを変更せずに、その隣に新しい Axes を描画する点が標準の `matplotlib.figure.Figure.colorbar` と異なります。

**パラメータ:**
- **mappable**: カラーマッピングの対象となるデータ。
- **ax**: カラーバーを配置する基準となる Axes。
- **fraction**: カラーバーに使用する元の Axes の割合（デフォルト 0 はサイズ変更なし）。

### `get_axes`

```python
get_axes(self, projection=None)
```

すべての `Axes` を検索します。オプションで投影法（projection）を指定してフィルタリングできます。

### `refresh`

```python
refresh(self)
```

現在の図を更新します。

### `save`

```python
save(self, *args, **kwargs)
```

図をディスクに保存します。`savefig` のエイリアスです。

### `show`

```python
show(self, block=None, warn=True)
```

現在の図を表示します。

**パラメータ:**
- **block**: 図が閉じられるまでブロックするかどうか。
