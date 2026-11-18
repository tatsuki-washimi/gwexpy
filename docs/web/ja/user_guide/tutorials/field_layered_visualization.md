# フィールドAPI: 地形・地図・物理場のレイヤー可視化

このチュートリアルでは、`FieldPlot`（Matplotlib）および `GeoMap`（PyGMT）を用いて、**背景地図タイル**・**DEM 地形（陰影起伏・等高線）**・**スカラー場／ベクトル場**・**検出器マーカー**・**静的背景を保持したアニメーション**を同一図面に重ねて描画する方法を説明します。

## レイヤーモデルと `zorder` 規約

`FieldPlot` の各レイヤーメソッドは以下の既定 `zorder` 規約に従って重なります（すべてキーワード引数で上書き可能です）。

| レイヤー | メソッド | 既定 `zorder` | 既定 `colorbar` |
| :--- | :--- | :--- | :--- |
| 背景地図タイル | `FieldPlot.add_basemap()` | `0.0` | `—` |
| 地形 DEM | `FieldPlot.add_terrain()` | `1.0` | `False` |
| スカラー場 | `FieldPlot.add_scalar()` | `2.0` | `True` |
| ベクトル場 | `FieldPlot.add_vector()` | `3.0` | `—` |
| マーカー・検出器 | `FieldPlot.add_markers()`, `FieldPlot.add_detector()` | `4.0` | `—` |

同一の `Axes` 上に重ねるレイヤー間では `GeoPlotContext` による整合性検証が行われます:

- `angle` ドメイン軸（`lon`/`lat`）と `real`/`local` 直交軸（`x`/`y` [m]）の混在は `ValueError` を送出し、`to_local_cartesian()` での変換を案内します。
- 有効な `crs` の不一致や `local_origin` のズレは `UserWarning` を出力します。
- 地理参照のないプレーンな場に対する `add_basemap()` / `add_detector()` は `ValueError` になります。

## 1. KAGRA 周辺の合成地形と波動場の準備

実測の国土地理院 DEM（`ScalarField.read("dem.xml", format="gsi-dem")` 等）を使う場合は、`dem.to_local_cartesian(origin=(137.31, 36.41))` で局所直交座標（m）へ変換できます。ここではオフラインでも実行できるよう、KAGRA 周辺を模擬した合成 DEM と波動場を作成します。

```python
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.fields import ScalarField, VectorField
from gwexpy.plot.field import FieldPlot

# KAGRA (137.31 E, 36.41 N) を原点とする局所直交格子
nx, ny = 60, 60
x = np.linspace(-1500.0, 1500.0, nx) * u.m
y = np.linspace(-1500.0, 1500.0, ny) * u.m
z0 = np.array([0.0]) * u.m
t_static = np.array([0.0]) * u.s

X, Y = np.meshgrid(x.value, y.value, indexing="ij")
elev = 400.0 + 700.0 * np.exp(-((X - 200.0) ** 2 + (Y + 150.0) ** 2) / (2 * 600.0**2))

dem = ScalarField(
    elev[np.newaxis, :, :, np.newaxis],
    unit=u.m,
    name="elevation",
    axis0=t_static,
    axis1=x,
    axis2=y,
    axis3=z0,
    space_domain="real",
)
dem._gwex_geo = SimpleNamespace(active=False, crs="EPSG:6668", local_origin=(137.31, 36.41))

# 同じ格子上の地震動・波動場の模擬データ
t = np.linspace(0.0, 2.0, 10) * u.s
wave_data = np.sin(
    2.0 * np.pi * (0.5 * t.value[:, None, None, None] - X[None, :, :, None] / 800.0)
) * np.exp(-(X[None, :, :, None] ** 2 + Y[None, :, :, None] ** 2) / (2 * 700.0**2))

wave = ScalarField(
    wave_data,
    unit=u.um / u.s,
    name="ground_velocity",
    axis0=t,
    axis1=x,
    axis2=y,
    axis3=z0,
    space_domain="real",
)
wave._gwex_geo = dem._gwex_geo
```

## 2. `FieldPlot` でレイヤーを順に重ねる

```python
fp = FieldPlot()
fp.add_terrain(dem, style="hillshade")
fp.add_terrain(dem, style="contour", levels=6, colors="black", alpha=0.35, linewidths=0.6)
fp.add_scalar(wave, x="x", y="y", slice_kwargs={"t": 0, "z": 0}, cmap="RdBu_r", alpha=0.6)
fp.add_detector("K1")

# ネットワーク接続時は国土地理院タイル等も背景に重ねられます:
# fp.add_basemap(source="gsi-pale", cache_dir=".tile_cache")

plt.show()
```

複数のスカラー場を重ねる際は `add_scalar(..., colorbar=False)` を指定するとカラーバーの重複を防げます。

## 3. `FieldBase.plot()` のワンコール重ね描き

`ScalarField.plot()` に `terrain=`・`terrain_style=`・`basemap=`・`vector=`・`detectors=` を渡すと、規約の `zorder` 順でまとめて描画できます。

```python
flow = VectorField({"x": wave, "y": wave * 0.5})
fp = wave.plot(
    x="x",
    y="y",
    t=0,
    z=0,
    terrain=dem,
    terrain_style="shaded_relief",
    # basemap="gsi-pale",
    vector=flow,
    detectors=["K1"],
    alpha=0.55,
    cmap="magma",
)
```

## 4. `animate_scalar()` による静的背景付きアニメーション

`FieldPlot.animate_scalar()` は単一の `QuadMesh` とカラーバーのみを初回に生成し、フレームごとに配列（`set_array()`）とタイトルのみを更新するため、背景の地図・地形・マーカーはそのまま保持されます。

```python
fp_anim = FieldPlot()
fp_anim.add_terrain(dem, style="hillshade")
fp_anim.add_detector("K1")
ani = fp_anim.animate_scalar(wave, x="x", y="y", axis="t", interval=150, cmap="RdBu_r", alpha=0.6)
```

> **注意:** `shading="gouraud"` は `QuadMesh.set_array()` と非互換のため、`animate_scalar()` では `ValueError` になります。

## 5. `GeoMap` による広域・測地投影プロット

緯度スパンが約 2° を超える広域図では、`add_basemap()` が WebMercator 歪みに対する `UserWarning` を出して `GeoMap` の利用を案内します。`GeoMap`（`pygmt`・`xarray` が必要）ではリモート地形や地理座標 `ScalarField` を測地投影で描画できます。

```python
from gwexpy.plot import GeoMap

gmap = GeoMap(projection="Mercator", region=[135.0, 140.0, 34.0, 38.0])
gmap.add_relief(resolution="15s", shading=True)  # @earth_relief_15s を取得
# gmap.add_field(angle_dem, cmap="viridis", alpha=0.7)
gmap.plot_detector("K1")
gmap.add_colorbar(label="Elevation [m]")
```

## タイルプロバイダーの利用規約と出典表記

`gwexpy.plot._tiles.PROVIDERS` に登録されているプロバイダー:

- `"gsi-std"`, `"gsi-pale"`, `"gsi-photo"`, `"gsi-relief"` — 国土地理院（地理院タイル）。利用規約に基づき出典表記が必須です（既定で `attribution=True` により自動表示されます）。詳細: https://maps.gsi.go.jp/development/ichiran.html
- `"osm"` — OpenStreetMap 標準タイル。大量の自動取得は利用規約上非推奨のため（https://operations.osmfoundation.org/policies/tiles/）、反復利用時は `cache_dir=` を指定してください。
