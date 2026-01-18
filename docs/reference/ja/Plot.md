# Plot

**継承元:** Plot

:class:`gwpy.plot.Plot` の拡張で、:class:`gwexpy.types.SeriesMatrix` 引数を
個々の :class:`gwpy.types.Series` オブジェクトに自動展開し、
可能な限りマトリクスのレイアウトとメタデータを保持します。

## メソッド

### `__init__`

```python
__init__(self, *args, **kwargs)
```

パラメータ
----------
figsize : 2-tuple of floats, default: :rc:`figure.figsize`
    図の寸法 ``(width, height)`` (インチ単位)。

dpi : float, default: :rc:`figure.dpi`
    1インチあたりのドット数。

facecolor : default: :rc:`figure.facecolor`
    図パッチの表面色。

edgecolor : default: :rc:`figure.edgecolor`
    図パッチの境界色。

linewidth : float
    フレームの線幅（図パッチの境界線幅）。

frameon : bool, default: :rc:`figure.frameon`
    ``False`` の場合、図の背景パッチの描画を抑制します。

layout : {'constrained', 'compressed', 'tight', `.LayoutEngine`, None}
    Axes の装飾（ラベル、目盛りなど）の重複を避けるためのプロット要素配置メカニズム。

*(Figure から継承)*

### `add_segments_bar`

```python
add_segments_bar(self, segments, ax=None, height=0.14, pad=0.1, sharex=True, location='bottom', **plotargs)
```

状態情報を示すセグメントバー `Plot` を追加します。

パラメータ
----------
segments : `~gwpy.segments.DataQualityFlag`
    データ品質フラグ、またはこの Plot に関する状態セグメントを示す `SegmentList`

ax : `Axes`, optional
    新しい `Axes` を配置する基準となる特定の `Axes`

height : `float`, optional
    アンカー軸の割合としての新しい軸の高さ

pad : `float`, optional
    新しい軸とアンカー間のパディング（アンカー軸の寸法の割合）

location : `str`, optional
    新しいセグメント軸の位置。デフォルトは ``'bottom'``

### `close`

```python
close(self)
```

プロットを閉じてメモリを解放します。

### `colorbar`

```python
colorbar(self, mappable=None, cax=None, ax=None, fraction=0.0, use_axesgrid=True, emit=True, **kwargs)
```

現在の `Plot` にカラーバーを追加します。

パラメータ
----------
mappable : matplotlib data collection
    色付けをマッピングするコレクション

cax : `~matplotlib.axes.Axes`
    カラーバーを描画する Axes

ax : `~matplotlib.axes.Axes`
    カラーバーを配置する基準となる Axes

fraction : `float`, optional
    カラーバーに使用する元の軸の割合。デフォルト (0) は元の軸をまったくリサイズしません。

use_axesgrid : `bool`
    :mod:`mpl_toolkits.axes_grid1` を使用してカラーバー軸を生成

emit : `bool`, optional
    `True` の場合、`Axes` 上のすべてのマップ可能なオブジェクトをカラーバーと同じ色付けに更新

戻り値
-------
cbar : `~matplotlib.colorbar.Colorbar`
    新しく追加された `Colorbar`

### `get_axes`

```python
get_axes(self, projection=None)
```

すべての `Axes` を検索します（オプションで指定された投影に一致するもの）

パラメータ
----------
projection : `str`
    返す軸タイプの名前

戻り値
-------
axlist : `list` of `~matplotlib.axes.Axes`

### `plot_mmm`

```python
plot_mmm(self, median, min_s, max_s, ax=None, **kwargs)
```

_ドキュメントなし。_

### `refresh`

```python
refresh(self)
```

現在の図を更新します。

### `save`

```python
save(self, *args, **kwargs)
```

図をディスクに保存します。

このメソッドは :meth:`~matplotlib.figure.Figure.savefig` のエイリアスです。

### `show`

```python
show(self, warn=True)
```

図を表示します。
