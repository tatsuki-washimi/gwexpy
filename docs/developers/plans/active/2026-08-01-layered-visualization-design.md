# 地図+地形+物理場のレイヤー描画 — 設計書

> Last-updated: 2026-08-01 (rev 1 — 初版)
> Reviewer Status: **draft**（ユーザーレビュー待ち）

Status: planned

対象マイルストーン: 未設定（他イシューと合わせて後日判断。実装開始は v0.1.12 リリースおよび `maint/0.1` 分岐後）

---

## Goal

地図タイル（国土地理院タイル等）+ 地形（陰影起伏・等高線）+ 物理場（スカラー場の半透明重ね・ベクトル場の quiver/streamline）+ マーカー（検出器位置等）を、少ないコードで同一図面に重ねて描けるようにする。第一の応用例はニュートンノイズ評価（KAGRA 周辺の地形背景 + 波動場アニメーション）だが、特定用途に限定しない汎用のレイヤー合成 API として設計する。

本設計は地形 DEM 読み込み設計（`2026-07-31-terrain-scalarfield-io-design.md`、GitHub #544–#556）の姉妹編であり、DEM 設計が定めたデータ規約（angle 軸 deg 保持、欠測 = NaN、GeoMetadata、`to_local_cartesian()`）を描画側から消費する。設計会議（2026-07-31 議事録）には描画拡張の議論はなく、本設計はユーザーの追加要望（2026-07-31）に基づく新規拡張である。

---

## 背景 — コードベース調査で確定した事実

設計に先立つ調査（2026-07-31〜08-01）で確定した事実。以下が本設計の形を決めている。

1. **描画 API は 2 系統が併存する。** 系統 A は `FieldPlot(Plot)`（`gwexpy/plot/field.py`、gwpy `Plot` 継承）で、`add_scalar()`（pcolormesh + 自動 colorbar）/ `add_vector()`（quiver / streamline）をビルダー式に積む構造。`FieldBase.plot()`（`gwexpy/fields/base.py:346`）、`FieldBase.animate()`（同 `:398`）、`VectorField.plot()`（magnitude + 白 quiver 重ね書き）がこの上にある。系統 B は `ScalarField.plot_map2d()` 等（`gwexpy/fields/scalar.py:1168` ほか）で素の matplotlib を使い `(fig, ax)` を返す。**レイヤー合成は系統 A のビルダー構造と自然に適合する**ため、本設計は系統 A に一本化する（系統 B は現状維持・スコープ外）。
2. **地図描画は `GeoMap`（PyGMT ラッパ、`gwexpy/plot/geomap.py`、217 行）が既存。** 海岸線・大陸塗り・検出器座標 DB `DETECTORS`（K1/H1/L1/V1/G1、`geomap.py:26-32`、pygmt 不在でもモジュール import 可）を持つが、**grdimage / DEM / 陰影起伏に相当するメソッドはない**。extras は `plotting = ["pygmt"]` 登録済み（`_EXTRA_MAP` に `"pygmt": "plotting"` あり）。
3. **陰影起伏（hillshade）は追加依存ゼロで実装できる。** matplotlib は必須依存（`pyproject.toml:42`）であり、`matplotlib.colors.LightSource` がある。リポジトリ内に hillshade / LightSource の使用は 0 ヒット（完全新規領域）。pillow は matplotlib の必須依存なので、タイル PNG のデコードにも追加インストールは不要。
4. **cartopy / contextily / 3D 描画（mplot3d, plotly, pyvista）はリポジトリ全体で 0 ヒット。** ただし `projection=` kwarg は gwexpy `Plot.__init__`（`gwexpy/plot/plot.py:239`）→ gwpy → `add_subplot` まで素通しできる経路が既存で、上級者が cartopy GeoAxes を自前で渡す道は現状でも塞がれていない。
5. **ScalarField → xarray.DataArray 変換が既存**（`to_xarray_field`、`gwexpy/interop/xarray_.py:335`）。PyGMT の `grdimage` は DataArray を直接受け取れるため、GeoMap に ScalarField を渡す橋は小さい。
6. **確定バグ①: `VectorField.plot(stride=)` は TypeError になる。** `gwexpy/fields/vector.py:363-381` で `stride` が `plot_kwargs` から pop されないまま `add_scalar` → `pcolormesh(stride=...)` に到達する（quiver 側へは別途コピーしている）。修正は pop 1 行 + 回帰テスト。
7. **確定バグ（構造的）②: `FieldBase.animate()` はカラーバーが累積する。** `base.py:467-476` の `update()` が毎フレーム `ax.clear()` + `add_scalar()` を呼び、`add_scalar` は `field.py:108` で必ず `self.colorbar()` を実行する。`ax.clear()` はカラーバー用 Axes を消さないため、フレーム数だけ colorbar Axes が増える。さらに `ax.clear()` 方式は「静的レイヤーを背景に保持する」拡張と根本的に両立しない。
8. **`get_slice()` の自動軸選択は DEM 形状 `(1, nx, ny, 1)` で誤動作する。** `base.py:253-268` は「fixed_coords にない最初の 2 軸」を選ぶため、singleton の t 軸が表示 x 軸に選ばれる。描画側は「サイズ > 1 の軸 2 本」を自動選択する規則で局所回避する（`get_slice` 本体の既定変更はしない）。
9. **テスト慣行**: スモークテスト（アーティスト型 assert + `plt.close`）+ ラベル契約テスト（`tests/plot/test_plot_helper_contracts.py`）。Agg バックエンド強制。画像比較テストなし。`network` マーカー既存（`pyproject.toml`）。`tests/test_geomap.py` に `_require_pygmt_runtime()` skip パターン既存。
10. **ラベル整形の二重実装**: 系統 A は `format_axis_label`（`gwexpy/plot/_label_utils.py`、無次元なら `[]` 省略）、系統 B は f-string 直書き（無次元でも `[]` が出る）。契約テストは系統 A のみ。本設計の新規コードは `format_axis_label` の使用を規約とする。

---

## 設計

### 1. レイヤー合成 API — `FieldPlot` に一本化（新規クラスなし）

`FieldPlot` は「空の Plot に `add_*` を積む」ビルダー型なのでレイヤーモデルと自然に適合する。レイヤーの重なりは matplotlib `zorder` に素直にマップし、既定値の規約を設ける（すべて kwargs で上書き可）:

```
basemap (0.0) < terrain (1.0) < scalar (2.0) < vector (3.0) < markers (4.0)
```

#### 1-1. 既存 `add_scalar` の最小拡張（前提整備）

```python
# gwexpy/plot/field.py
def add_scalar(self, field, x=None, y=None, slice_kwargs=None, *,
               colorbar=True,      # 新設。False で colorbar 生成をスキップ
               **kwargs):          # zorder, alpha 等は従来どおり pcolormesh へ素通し
```

- 既定 `True` で完全後方互換（既存のラベル契約テストは不変）。
- `colorbar=False` 時は `last_field_colorbar` を更新しない。背景・中間レイヤーで必須になる。
- `add_vector` は変更不要（quiver/streamline は既にレイヤーとして重なる）。

#### 1-2. `FieldPlot.add_terrain`（新設・依存ゼロ）

```python
def add_terrain(self, dem, x=None, y=None, *,
                style="hillshade",    # "hillshade" | "shaded_relief" | "contour" | "contourf"
                azdeg=315.0, altdeg=45.0, vert_exag=1.0,
                levels=None,          # contour 系のみ
                cmap=None,            # hillshade: "gray" / shaded_relief: "terrain" 既定
                colorbar=False,       # 背景レイヤーなので既定 False
                slice_kwargs=None, **kwargs):
    """DEM ScalarField を陰影起伏 / 等高線の背景レイヤーとして描画する。

    Returns: AxesImage (hillshade / shaded_relief) | QuadContourSet (contour 系)
    """
```

- `dem` は DEM 設計の規約どおりの `ScalarField`（angle 軸 deg でも local cartesian m でも可）。
- x/y 未指定時は「サイズ > 1 の軸 2 本」を自動選択してから `get_slice` に明示指定で渡す（背景 8 の誤動作回避）。
- `hillshade`: `LightSource(azdeg, altdeg).hillshade(Z, dx=, dy=, vert_exag=)` → グレースケール `imshow(extent=..., origin=...)`。dx/dy は軸間隔から取得。angle 軸のときは dx に `cos(lat_center)` 補正を掛けて縦横の実距離比を概ね保つ（docstring に近似である旨明記）。NaN はマスクして透明化（DEM の欠測 = NaN 規約と整合）。
- `shaded_relief`: `LightSource.shade(Z, cmap=..., blend_mode="overlay")` の RGB 画像。`colorbar=True` 指定時のみ標高カラーバー用の `ScalarMappable` を別途作る。
- `contour` / `contourf`: `ax.contour(x, y, Z.T, levels=...)`。等高線ラベルは返り値に対しユーザーが `ax.clabel()` を呼ぶ（最小 API）。
- lat native 降順軸（DEM 規約）は imshow extent / pcolormesh の降順座標がそのまま扱えるため反転処理は不要。テストで向きを固定する。
- 計算本体（`compute_hillshade(z, dx, dy, ...)`）は visualize_fields スキルの三層構造に従い基盤層 `gwexpy/plot/_geo_utils.py`（新規）へ分離。

#### 1-3. `FieldPlot.add_basemap`（新設・XYZ タイル、依存実質ゼロ）

```python
def add_basemap(self, *, source="gsi-std",    # provider 名 or "{z}/{x}/{y}" URL テンプレート
                zoom=None,                    # None: 軸範囲から自動決定(タイル枚数上限つき)
                extent=None,                  # None: 現在の軸範囲 (lonmin, lonmax, latmin, latmax)
                alpha=1.0, attribution=True,  # 帰属表示テキストを図隅に自動追加
                cache_dir=None, timeout=10.0, **kwargs) -> list[AxesImage]:
```

- タイル基盤は新規 `gwexpy/plot/_tiles.py` に分離:
  - `PROVIDERS`: `"gsi-std"`（標準地図）/ `"gsi-pale"`（淡色）/ `"gsi-photo"`（写真）/ `"gsi-relief"`（色別標高）/ `"osm"`。URL テンプレートと帰属文字列のペア。既定は国土地理院（第一応用が KAGRA 周辺のため）。
  - `deg2tile(lon, lat, z)` / `tile_extent_deg(x, y, z)`: slippy-map 数式の**純関数**（オフライン単体テスト対象）。
  - `fetch_tile(url, *, timeout, cache_dir) -> PIL.Image.Image`: `urllib.request` + User-Agent ヘッダ。**これが唯一のネットワーク接点 = テストのモック点**。
- 描画はタイルごとに `imshow(im, extent=タイルの lon/lat 境界)`（§4 参照）。
- ネットワーク失敗は `URLError` / `OSError` をそのまま伝播（メッセージに provider 名とオフラインの可能性を付記）。新設例外なしの慣行に従う。

#### 1-4. `FieldPlot.add_markers` / `add_detector`（新設）

```python
def add_markers(self, x, y, *, labels=None, marker="o", zorder=4.0, **kwargs)
def add_detector(self, name, *, label=True, zorder=4.0, **kwargs)
```

- `DETECTORS` 辞書を `geomap.py:26` から新規 `gwexpy/plot/_detectors.py` へ移設し、`geomap.py` からは再エクスポート（後方互換・重複 DB 回避）。
- 軸が local cartesian（m）のときは geo コンテキスト（§4）の `local_origin` を使って lon/lat → x/y 変換して打点。origin 不明なら ValueError。

#### 1-5. ワンコール糖衣 — 作る。ただし最後に

DEM 設計会議の段取り原則（「コアメソッド安定後にエイリアス」）を描画でも踏襲する。レイヤーメソッド群（1-1〜1-4）の安定後、最終サブイシューで:

```python
# gwexpy/fields/base.py の plot() に予約 kwarg を追加
field.plot(x="lon", y="lat",
           terrain=dem,               # ScalarField → add_terrain(style=terrain_style)
           terrain_style="hillshade",
           basemap="gsi-pale",        # provider 名 or True
           vector=flow_field,         # VectorField → add_vector(mode="quiver")
           detectors=["K1"],          # → add_detector
           alpha=0.6, cmap="magma")
```

- `plot()` の kwargs 振り分け（`base.py:389-393`）に予約キー 4 つを追加するだけで実装できる。軸名との衝突時は `slices=` 優先の現行規則を踏襲。
- 描画順は規約 zorder に固定。細かい制御が要るユーザーはレイヤーメソッドへ誘導。
- 系統 B（`plot_map2d` 等）には便宜 kwarg を**追加しない**（レイヤー面の一本化）。

### 2. バックエンド戦略 — matplotlib 主軸 + GeoMap 副軸

| 観点 | FieldPlot（matplotlib）= 主軸 | GeoMap（PyGMT）= 広域・出版用 |
| :-- | :-- | :-- |
| 対象領域 | 局所〜県スケール（KAGRA NN が該当） | 広域〜全球、測地投影が要る図 |
| 物理場重ね | pcolormesh / quiver / streamline / 等高線すべて | grdimage（スカラーのみ、v1） |
| アニメーション | FuncAnimation（§5） | 非対応（スコープ外と明記） |
| 地形 | DEM ScalarField + LightSource（ローカルデータ） | GMT リモート earth_relief（ネットワーク） |
| 依存 | 追加ゼロ | 既存 `plotting` extra |

**GeoMap への ScalarField 受け入れ（採用）**:

```python
# gwexpy/plot/geomap.py
def add_relief(self, resolution="01s", *, shading=True, cmap="geo", **kwargs):
    """GMT リモート earth_relief を grdimage で描画（ネットワーク必要）。"""
def add_field(self, field, *, cmap=None, alpha=None, shading=None, **kwargs):
    """angle 軸 ScalarField を to_xarray_field 経由で grdimage 描画。"""
def add_colorbar(self, label=None, **kwargs): ...
```

- `add_field` は `to_xarray_field` で DataArray 化 → geomap.py 内ヘルパ `_field_to_grid()` で `(1, nx, ny, 1)` を 2D squeeze、lat 降順を昇順ソート（データ同時反転）して `(lat, lon)` 次元順に整え → `fig.grdimage(grid=da)`。xarray は既存 optional（netcdf4 extra）なので `require_optional("xarray")` を通す。ヘルパは interop 層に置かない（pygmt 依存を interop に持ち込まない）。
- `space_domain` が angle でない場は ValueError（「GeoMap は地理座標専用。局所直交場は FieldPlot を使うこと」）。

**cartopy 非採用（妥当性確認済み）**: (i) proj/geos のバイナリ依存が重く「必須依存を増やさない」に反する、(ii) 広域測地投影の需要は GeoMap が既に担う、(iii) `projection=` 素通し経路（背景 4）により上級者が自前で cartopy GeoAxes を渡す道は塞がれない（非サポートだがブロックもしない、と docs に明記）。レイヤーメソッドはデータ座標で描くだけにして投影非依存に保つ（`transform=` は kwargs 素通しで通る）。

### 3. 依存戦略 — v1 は新規依存ゼロ

| 機能 | 実装 | 依存 |
| :-- | :-- | :-- |
| hillshade / shaded relief / 等高線 | `matplotlib.colors.LightSource` + contour | **ゼロ**（matplotlib は必須依存） |
| XYZ タイル | `urllib.request` + `PIL.Image`（pillow は matplotlib の必須依存） | **実質ゼロ** |
| 広域地図・地形 | PyGMT（GeoMap） | 既存 `plotting` extra |
| DEM データ自体 | #547/#548 の read | `geospatial` extra |

- **contextily は v1 では不採用**。守備範囲（タイル取得と配置のみ、~100 行の自前実装で足りる）に対して mercantile / xyzservices / requests / geopy / joblib の芋づる依存が過剰。厳密な再投影ワーピングが必要になった時点で follow-up として再評価（§7 リスク 2）。
- **pillow の `pyproject.toml` dependencies への明示追加を推奨**（コメント付き。matplotlib 経由で必ず入るためインストール実体は不変だが、「import するものは宣言する」原則に従う）。ポリシー判断はイシュー内の要確認事項とする。
- したがって **v1 では extras 5 点セットの連動更新は不要**（pygmt は登録済み、新パッケージなし）。

### 4. 座標整合規則

#### geo プロットコンテキスト（基盤層）

`gwexpy/plot/_geo_utils.py` に、Axes 単位の軽量な整合状態を持つ:

```python
@dataclass
class GeoPlotContext:
    kind: Literal["angle", "local", "plain"]   # 表示軸の性質
    crs: str | None                            # 最初の active GeoMetadata から
    local_origin: tuple[float, float] | None   # local のとき
    axis_units: tuple[Unit, Unit]

def resolve_geo_context(ax, field=None, x_name=..., y_name=...) -> GeoPlotContext:
    """最初の geo-aware レイヤーで確定し、以後のレイヤーはこれと照合する。
    実体は ax の属性(例: ax._gwexpy_geo_context)に保持。"""
```

判定は表示 2 軸の `space_domain`（angle/real）と `field.geo`（`_gwex_geo`、#546）から導出。

#### 整合規則表

| 状況 | 挙動 |
| :-- | :-- |
| angle 軸コンテキストに real（m）軸フィールドを重ねる（逆も） | **ValueError**。「`to_local_cartesian()` で揃えるか、DEM 側を deg のまま使う」と誘導（#545 の fft_space ガードと同文体） |
| 両者 active geo で `crs` 文字列不一致 | **UserWarning**（v1 は警告のみ。#555 の「crs 不一致演算は素通り」既知制限と整合） |
| local 同士で `local_origin` 不一致 | **UserWarning**（距離ズレ量を概算表示） |
| geo なしの plain フィールドに `add_basemap` / `add_detector` | **ValueError**(地理参照がない) |
| 単位の deg/rad 混在 | astropy 換算して受理（軸ラベルは最初のレイヤーの単位を維持） |

#### タイル（WebMercator）と deg/m 軸の整合

- **再投影ワーピングはしない**。タイル 1 枚ごとに正しい lon/lat 境界を `extent` に与えて `imshow` する。タイル内部の緯度方向歪みは、局所域（ズーム ≥ 10、緯度スパン ≪ 1°）では表示上無視できる。
- **緯度スパンが閾値（案: 2°）を超えたら UserWarning** で GeoMap（PyGMT）へ誘導。近似の妥当域を仕様として docstring に明記。
- local（m）軸のときは、タイル四隅を #550 と同じ局所平面式（x = R·cos(lat0)·Δlon, y = R·Δlat）で変換して配置。`GeoPlotContext.local_origin` が必要（なければ ValueError で `origin=` 指定を促す）。

### 5. アニメーション統合（既知バグ②の修正を包含）

`ax.clear()` + 毎フレーム `add_scalar` 方式を廃止し、**メッシュ 1 個を作って `set_array()` で更新**する方式へ:

```python
# gwexpy/plot/field.py
def animate_scalar(self, field, x=None, y=None, axis="t", interval=100,
                   slice_kwargs=None, *, colorbar=True, **kwargs) -> FuncAnimation:
    """既存 Axes 上の静的レイヤー(basemap/terrain/markers)を保持したまま、
    field の axis 方向スライスだけを更新するアニメーションを返す。
    メッシュと colorbar は 1 回だけ生成する。"""
```

- `FieldBase.animate`（`base.py:398`）は内部を `animate_scalar` への委譲に置き換え（公開シグネチャ不変・後方互換）。`ax.clear()` が消えるため**カラーバー累積は構造的に解消**し、静的レイヤー保持が自動的に成立する。
- 使用イメージ（静的背景 + 動的波動場）:

```python
fp = FieldPlot()
fp.add_basemap(source="gsi-pale")
fp.add_terrain(dem, style="hillshade")
fp.add_detector("K1")
ani = fp.animate_scalar(wave_field, x="lon", y="lat", axis="t", alpha=0.6)
```

- vmin/vmax 既定は現行（全データ min/max で固定）を踏襲。タイトル更新は `set_title` のみ。
- 制約: `set_array` と `shading="gouraud"` の組み合わせは非互換のため、animate 経路では shading を flat/auto に限定（明示 ValueError）。
- VectorField の動的更新（`quiver.set_UVC`）は follow-up（受け皿のみイシューに記載）。
- 着手時は最初に累積の再現テスト（フレームを 2 回進めて `len(fig.axes)` 不変を assert）を書いて挙動を確定させる。

### 6. 既知バグ・非対称の扱い

| 項目 | 扱い | 理由 |
| :-- | :-- | :-- |
| `VectorField.plot(stride=)` TypeError（確定、`vector.py:363-381`） | **umbrella 外の単独バグイシュー**。`maint/0.1` backport 候補 | 純粋な既存バグ。修正は pop 1 行 + 回帰テストで、v0.2.0 を待つ理由がない |
| animate カラーバー累積（構造的に確定） | **サブイシュー C-6 に包含**。独立修正はしない | 修正の正解が「単一メッシュ更新方式への置換」であり、レイヤー保持アニメーションと同一の変更のため。二度作り直すのは無駄 |
| `add_scalar` が colorbar を強制 | **前提整備サブイシュー C-1** | レイヤー化の直接の前提 |
| ラベル二重実装（系統 B f-string vs `_label_utils.py`） | **別建て低優先イシュー（umbrella 外）**。本拡張の新規コードは `format_axis_label` 使用を規約化 | 描画拡張のブロッカーではない技術的負債 |
| 系統 A/B 返り値非対称（FieldPlot vs (fig, ax)） | **v1 では非対応**。本設計書に「レイヤー面は系統 A に一本化、系統 B は現状維持」の決定のみ記録 | 破壊的変更になり得るため、独立の設計判断として切り離す |

---

## イシュー構成 — 新 Umbrella C（#555 へは追加しない）

別 umbrella とする理由: (i) 対象サブシステムが `gwexpy/plot/` で、#555 は `gwexpy/fields/` + `fields/io/` — レビュー観点・変更ファイルが重ならない。(ii) #555 は既に 11 サブイシューで飽和。(iii) 依存プロファイルが逆（A は rasterio extras、C は依存ゼロ）。ただし umbrella 本文に **#545（angle domain）/ #546（GeoMetadata）への依存を明記**する。C の大半は合成 angle フィールドでテストできるため、**#545 が入った時点で並行着手可能**（#547 の実 DEM reader を待つ必要はない）。

**Umbrella C**: `[Umbrella] Layered map/terrain/field visualization`

| ID | タイトル（要旨） | 依存 |
| :-- | :-- | :-- |
| C-0（外） | plot: `VectorField.plot(stride=)` が pcolormesh に stride を渡し TypeError（バグ修正、maint/0.1 候補） | なし |
| C-x（外） | plot: 軸ラベル整形の二重実装統一（低優先） | なし |
| C-1 | plot: `add_scalar` のレイヤー化基盤（`colorbar=` トグル、zorder 規約） | なし |
| C-2 | plot: geo プロットコンテキスト + レイヤー座標整合検証（`_geo_utils.py`） | #545, #546 |
| C-3 | plot: `FieldPlot.add_terrain`（hillshade / shaded relief / contours） | #545, C-1, C-2 |
| C-4 | plot: `add_markers` / `add_detector`（`DETECTORS` 移設） | C-2 |
| C-5 | plot: 軽量 XYZ タイル basemap（`_tiles.py`、GSI/OSM providers） | C-2 |
| C-6 | plot: アニメーション再設計 — 静的レイヤー + 単一メッシュ更新（colorbar 累積解消を包含） | C-1 |
| C-7 | plot: `GeoMap.add_relief` / `add_field(ScalarField)` / `add_colorbar` | #545 |
| C-8 | fields: `FieldBase.plot` ワンコール糖衣（`terrain=`, `basemap=`, `vector=`, `detectors=`） | C-1〜C-5 |
| C-9 | docs: レイヤー描画チュートリアル（KAGRA 地形 + 波動場ケーススタディ） | C-3〜C-7（データは #547 以降） |

クリティカルパス: **#545 → C-2 → C-3 → C-6**（ニュートンノイズの「地形背景 + 波動場アニメ」到達に必要な最短列）。C-0 / C-1 は即着手可能。C-5 / C-7 は独立に並行可。

---

## テスト計画

既存慣行（スモーク + 契約、Agg 強制、画像比較なし）を踏襲。

- **スモーク**: 新規 `tests/plot/test_field_layers.py`（`test_field_plots.py` と同型: アーティスト型 assert + `plt.close`）。合成 DEM フィクスチャは angle 軸 `ScalarField`（`shape=(1,nx,ny,1)`、`space_domain={"lon":"angle","lat":"angle",...}`）を直接生成（#545 のみに依存し、#547 の reader 不要）。
- **契約**: `tests/plot/test_plot_helper_contracts.py` へ追記 — (i) `colorbar=False` で Axes 数不変、(ii) animate でフレーム進行後も `len(fig.axes)` 不変、(iii) angle 軸ラベル `"lon [deg]"`（`format_axis_label` 経由）、(iv) angle×real 混在 ValueError のメッセージ文言、(v) GeoMap backend エラー契約（既存 `test_geomap_optional_backend_error_contract` と同型）。
- **タイルのネットワーク分離**: `fetch_tile()` を唯一の接点として monkeypatch（合成 256×256 PIL 画像を返す）。タイルインデックス数式（`deg2tile` / `tile_extent_deg`）は既知値（例: KAGRA 座標 → zoom 12 のタイル番号）で純関数テスト。実ネットワークは `@pytest.mark.network` 1 本のみ（GSI タイル 1 枚）。
- **GeoMap**: `_require_pygmt_runtime()` パターンで追記。`add_relief` は GMT リモートデータ取得を伴うため network マーカー。
- **後方互換回帰**: `add_scalar` 既定で colorbar 生成、既存 `test_field_plots.py` が無変更で通ること。

---

## リスク・未解決事項

1. **タイル利用規約・帰属表示**: 地理院タイルは出典表記必須（`attribution=True` 既定で自動描画）。OSM はプログラム的大量取得を非推奨としており、既定 provider を GSI にし、docs に利用規約リンクを掲載。`cache_dir` でアクセス数を抑える。
2. **WebMercator 近似の妥当域**: タイルごとの extent 配置は緯度スパン ~2° 超で歪みが視認され得る。閾値警告 + GeoMap 誘導で運用するが、閾値の妥当性は実装時に実測で確定（未解決）。厳密ワーピングが必要になれば contextily / rasterio.warp を follow-up で再評価。
3. **`set_array` × `shading="gouraud"` 非互換**: animate 経路では flat/auto に限定し明示エラー。
4. **angle 軸上のベクトル場**: m/s 成分の矢を deg 軸に置く場合、矢の長さは表示スケールであり物理的整合はない。docstring で明記し、厳密には `to_local_cartesian()` 後の描画を推奨。
5. **複数スカラーレイヤー時のカラーバーレイアウト**: 2 個以上で図が窮屈になる。v1 は `colorbar=False` の使い分けをチュートリアルで示すに留める（レイアウトマネージャは非スコープ）。
6. **`get_slice` 自動軸選択の一般改善**（singleton 軸を優先除外するか）: 本設計は `add_terrain` 側の局所回避で進めるが、`fill_missing`（#549）と規則を共有すべきかは Umbrella A 側と要すり合わせ（未解決）。
7. **pillow の明示宣言**: インストール実体は不変だが dependencies 行が増えることへのポリシー判断（要ユーザー確認）。
8. **geo メタデータ未搭載フィールドとの併用体験**: #547 以前に C-3 を使うには自前で angle フィールドを組む必要がある。チュートリアル（C-9)は #547 到達後に書く順序で吸収。

---

## Non-Goals

- 3D 描画(mplot3d / plotly / pyvista)— 需要が確認されてから別途設計
- cartopy GeoAxes の公式サポート(`projection=` 素通しは現状維持、非サポート明記)
- 厳密な地図投影ワーピング(contextily / rasterio.warp)— follow-up 再評価
- 系統 B(`plot_map2d` 等)へのレイヤー機能追加・返り値非対称の解消
- GeoMap でのアニメーション
- カラーバーレイアウトマネージャ
- VectorField の動的アニメーション更新(`quiver.set_UVC`)— C-6 の follow-up

---

## 付録 — GitHub イシュー本文ドラフト(英語)

以下は投稿用ドラフト。`#C-N` はプレースホルダで、投稿時に実イシュー番号へ置換する。milestone は指定しない(後日まとめて判断)。

### Umbrella C: `[Umbrella] Layered map/terrain/field visualization`

```markdown
## Goal

Compose basemap tiles + terrain (hillshade/contours) + physical fields
(scalar overlay, vector quiver/streamlines) + markers (detectors) in one
figure with minimal code. Primary use case: Newtonian-noise studies around
KAGRA (terrain background + wavefield animation), but the design is
application-agnostic.

Design document: `docs/developers/plans/active/2026-08-01-layered-visualization-design.md`

## Design summary

- All layering converges on `FieldPlot` (matplotlib); `GeoMap` (PyGMT)
  covers wide-area/geodetic-projection figures. No new figure classes.
- zorder conventions: basemap (0) < terrain (1) < scalar (2) < vector (3)
  < markers (4).
- v1 adds zero new dependencies: hillshade via `matplotlib.colors.LightSource`,
  XYZ tiles via urllib + PIL (pillow ships with matplotlib). cartopy and
  contextily are deliberately not adopted (rationale in the design doc).
- Layer coordinate-consistency checks (angle vs local vs plain, CRS
  mismatch warnings) build on the geospatial DEM infrastructure.

## Relation to the DEM umbrella (#555)

Depends on #545 ("angle" space domain) and #546 (GeoMetadata) only. Most
sub-issues are testable with synthetic angle-axis fields, so work can
proceed in parallel with the DEM readers (#547+) once #545 lands.

## Non-goals

3D plotting, official cartopy support, strict map reprojection/warping,
layer features for the legacy `plot_map2d` family, GeoMap animation,
colorbar layout management, animated vector updates (follow-up of the
animation sub-issue).

## Sub-issues

- [ ] #C-1 plot: add_scalar layering groundwork (colorbar= toggle, zorder conventions)
- [ ] #C-2 plot: geo plot context + layer coordinate-consistency checks
- [ ] #C-3 plot: FieldPlot.add_terrain (hillshade / shaded relief / contours)
- [ ] #C-4 plot: add_markers / add_detector (shared DETECTORS db)
- [ ] #C-5 plot: lightweight XYZ tile basemap (GSI/OSM providers)
- [ ] #C-6 plot: animation rework — static layers + single-mesh updates
- [ ] #C-7 plot: GeoMap.add_relief / add_field(ScalarField) / add_colorbar
- [ ] #C-8 fields: one-call overlay aliases in FieldBase.plot
- [ ] #C-9 docs: layered visualization tutorial (KAGRA terrain + wavefield)

Related bug fixes outside this umbrella: #C-0 (VectorField.plot stride
TypeError), #C-x (axis-label formatting unification).

## Scheduling

Implementation starts after the v0.1.12 release and the `maint/0.1` branch
cut. Critical path: #545 → #C-2 → #C-3 → #C-6. #C-1 can start immediately;
#C-5 / #C-7 are independent.
```

### C-0: `plot: VectorField.plot(stride=) passes stride to pcolormesh (TypeError)`

```markdown
`VectorField.plot()` copies `stride` into the quiver kwargs but does not
pop it from `plot_kwargs` (`gwexpy/fields/vector.py:363-381`), so it also
reaches `add_scalar` → `ax.pcolormesh(stride=...)`, which raises
TypeError. Confirmed by code inspection.

Fix: pop `stride` from the scalar-layer kwargs; add a regression test
`vector_field.plot(stride=2)`.

Candidate for backport to `maint/0.1` once cut (pure bug fix, one line +
test). Found during the layered-visualization design
(`docs/developers/plans/active/2026-08-01-layered-visualization-design.md`).
```

### C-x: `plot: unify axis-label formatting (format_axis_label vs f-strings)`

```markdown
Two label implementations coexist: `gwexpy/plot/_label_utils.py`
`format_axis_label` (omits `[]` for dimensionless; contract-tested) and
raw f-strings in `gwexpy/fields/scalar.py` (`plot_map2d`, `plot_profile`,
`plot_time_space_map`; emit `[]` even when dimensionless; no contract
tests). Migrate the f-string call sites to `format_axis_label` and extend
the label contract tests to cover them.

Low priority; not a blocker for the layered-visualization umbrella #C-U.
New plotting code must use `format_axis_label` (design-doc convention).
```

### C-1: `plot: add_scalar layering groundwork (colorbar= toggle, zorder conventions)`

```markdown
Prerequisite for layered plotting (#C-U).

- Add keyword-only `colorbar=True` to `FieldPlot.add_scalar`
  (`gwexpy/plot/field.py`). `colorbar=False` skips colorbar creation and
  does not update `last_field_colorbar`. Default keeps full backward
  compatibility (existing label-contract tests unchanged).
- Document the layer zorder conventions (basemap 0 < terrain 1 < scalar 2
  < vector 3 < markers 4) in the FieldPlot docstring; all overridable via
  kwargs.

Acceptance: contract test that `colorbar=False` leaves `len(fig.axes)`
unchanged; existing `tests/plot/test_field_plots.py` and
`test_plot_helper_contracts.py` pass unmodified.
```

### C-2: `plot: geo plot context + layer coordinate-consistency checks`

```markdown
Foundation layer for #C-U. New module `gwexpy/plot/_geo_utils.py`:

- `GeoPlotContext` dataclass (kind: "angle"|"local"|"plain", crs,
  local_origin, axis_units) stored per-Axes; `resolve_geo_context()`
  establishes it from the first geo-aware layer and validates subsequent
  layers against it.
- Consistency rules: angle×real mixing → ValueError (message points to
  `to_local_cartesian()`, same style as the #545 fft_space guard); active
  CRS string mismatch → UserWarning (consistent with the #555 known
  limitation); local-origin mismatch → UserWarning with estimated offset;
  `add_basemap`/`add_detector` on plain fields → ValueError; deg/rad
  mixing → accepted via astropy conversion.
- `compute_hillshade(z, dx, dy, azdeg, altdeg, vert_exag)` and the local
  tangent-plane helpers (same formulas as #550) live here too.

Depends on #545 (angle domain), #546 (GeoMetadata). Errors are plain
ValueError/UserWarning (no new exception classes).

Acceptance: per-row tests of the consistency table; pure-function tests
for `compute_hillshade` on a synthetic slope.
```

### C-3: `plot: FieldPlot.add_terrain (hillshade / shaded relief / contours)`

```markdown
Zero-dependency terrain background layer for DEM ScalarFields (#C-U).

`FieldPlot.add_terrain(dem, x=None, y=None, *, style="hillshade",
azdeg=315, altdeg=45, vert_exag=1.0, levels=None, cmap=None,
colorbar=False, slice_kwargs=None, **kwargs)`

- styles: "hillshade" (LightSource.hillshade → grayscale imshow),
  "shaded_relief" (LightSource.shade RGB), "contour"/"contourf".
- Default display axes: the two axes with size > 1 (avoids the
  `get_slice` first-two-axes rule picking the singleton t axis of
  `(1, nx, ny, 1)` DEMs; `get_slice` itself is unchanged).
- NaN → transparent (shared missing-value convention); native descending
  lat axes supported without flipping; on angle axes dx gets a
  cos(lat_center) correction (documented approximation).
- Returns the artist (AxesImage or QuadContourSet); `colorbar=False` by
  default (background layer).

Depends on: #545, #C-1, #C-2.

Acceptance: smoke tests on a synthetic angle-axis DEM (artist types,
NaN transparency, descending-axis orientation fixed by test); no
colorbar axes created by default.
```

### C-4: `plot: add_markers / add_detector (shared DETECTORS db)`

```markdown
Marker layer for #C-U.

- Move the `DETECTORS` coordinate db from `gwexpy/plot/geomap.py:26` to a
  new `gwexpy/plot/_detectors.py`; re-export from geomap.py for backward
  compatibility.
- `FieldPlot.add_markers(x, y, *, labels=None, marker="o", zorder=4.0, **kw)`
  and `add_detector(name, *, label=True, zorder=4.0, **kw)`.
- On local-cartesian axes, detector lon/lat is converted through the geo
  context's `local_origin` (ValueError when absent).

Depends on: #C-2.

Acceptance: backward-compat import test via geomap.py; local-axis
conversion test against the #550 tangent-plane formulas.
```

### C-5: `plot: lightweight XYZ tile basemap (add_basemap, GSI/OSM providers)`

```markdown
Basemap tile layer for #C-U, with no new dependencies (urllib + PIL;
pillow ships with matplotlib).

New `gwexpy/plot/_tiles.py`:
- `PROVIDERS`: gsi-std / gsi-pale / gsi-photo / gsi-relief / osm (URL
  template + attribution string). Default provider: GSI.
- `deg2tile()` / `tile_extent_deg()` slippy-map math as pure functions.
- `fetch_tile(url, *, timeout, cache_dir)` — the single network contact
  point (mock target). Failures propagate URLError/OSError with the
  provider name in the message.

`FieldPlot.add_basemap(*, source="gsi-std", zoom=None, extent=None,
alpha=1.0, attribution=True, cache_dir=None, timeout=10.0, **kwargs)`

- No reprojection warping: each tile is imshow-ed at its lon/lat extent
  (valid for local areas); latitude span above ~2° → UserWarning pointing
  to GeoMap. On local-cartesian axes tile corners go through the #550
  tangent-plane conversion (requires local_origin).
- Attribution text auto-added by default (GSI terms require it); docs
  link the provider terms of use.

Depends on: #C-2.

Acceptance: pure-function tile-math tests with known values (KAGRA coords
→ zoom-12 tile indices); placement tests with a mocked `fetch_tile`
returning synthetic 256×256 images; exactly one `@pytest.mark.network`
test fetching a single GSI tile; attribution and wide-extent warning
tests.
```

### C-6: `plot: animation rework — static layers + single-mesh updates (fixes colorbar accumulation)`

```markdown
Replace the `ax.clear()` + per-frame `add_scalar` animation loop
(`gwexpy/fields/base.py:467-476`) with a single mesh updated via
`set_array()`.

Structural defect being fixed: `add_scalar` unconditionally calls
`self.colorbar()` (`gwexpy/plot/field.py:108`), and `ax.clear()` does not
remove colorbar axes, so every frame adds one. The `ax.clear()` approach
is also fundamentally incompatible with keeping static layers (basemap /
terrain / markers) in the background.

- New `FieldPlot.animate_scalar(field, x=None, y=None, axis="t",
  interval=100, slice_kwargs=None, *, colorbar=True, **kwargs)`: mesh and
  colorbar created once; static layers untouched.
- `FieldBase.animate` delegates to it (public signature unchanged).
- `shading="gouraud"` is rejected on the animate path (incompatible with
  set_array); flat/auto only.
- First implementation step: write the reproduction test (advance two
  frames, assert `len(fig.axes)` unchanged) to pin the defect.
- Animated vector updates (`quiver.set_UVC`) are a follow-up, out of
  scope here.

Depends on: #C-1.

Acceptance: reproduction test passes after rework; static-layer artist
counts unchanged across frames; existing `test_animate` passes; vmin/vmax
fixed-scale behavior preserved.
```

### C-7: `plot: GeoMap.add_relief / add_field(ScalarField) / add_colorbar`

```markdown
Wide-area / geodetic-projection counterpart of the FieldPlot layers
(#C-U). Extends `gwexpy/plot/geomap.py` (PyGMT backend, existing
`plotting` extra):

- `add_relief(resolution="01s", *, shading=True, cmap="geo", **kw)` — GMT
  remote earth_relief via grdimage (network required).
- `add_field(field, *, cmap=None, alpha=None, shading=None, **kw)` —
  angle-axis ScalarField → `to_xarray_field`
  (`gwexpy/interop/xarray_.py:335`) → internal `_field_to_grid()` (squeeze
  `(1,nx,ny,1)` to 2D, sort descending lat ascending with simultaneous
  data flip, dims `(lat, lon)`) → `fig.grdimage(grid=...)`. xarray goes
  through `require_optional("xarray")`. Non-angle fields → ValueError
  ("GeoMap is for geographic coordinates; use FieldPlot for local
  Cartesian fields").
- `add_colorbar(label=None, **kw)`.
- No animation support (documented non-goal).

Depends on: #545 (angle domain).

Acceptance: under the `_require_pygmt_runtime()` skip pattern, grdimage
of a synthetic angle field; value↔coordinate correspondence preserved
after the lat sort; ValueError for real-axis fields; `add_relief` marked
`@pytest.mark.network`.
```

### C-8: `fields: one-call overlay aliases in FieldBase.plot (terrain=, basemap=, vector=, detectors=)`

```markdown
Sugar layer added last, after the layer methods stabilize (staged-alias
principle shared with the DEM umbrella).

`field.plot(x="lon", y="lat", terrain=dem, terrain_style="hillshade",
basemap="gsi-pale", vector=flow, detectors=["K1"], alpha=0.6)`

- Implemented in the kwargs dispatch of `FieldBase.plot`
  (`gwexpy/fields/base.py:389-393`) by reserving four keys; axis-name
  collisions follow the existing `slices=` precedence rule.
- Layer order fixed to the zorder conventions; users needing fine control
  are pointed to the layer methods.
- The legacy `plot_map2d` family does NOT get these kwargs (layering is
  consolidated on FieldPlot).

Depends on: #C-1, #C-2, #C-3, #C-4, #C-5.

Acceptance: the one-call form produces the same artist composition as
manual layer construction; zorder order verified.
```

### C-9: `docs: layered visualization tutorial (KAGRA terrain + wavefield case study)`

```markdown
End-to-end notebook: GSI DEM around KAGRA (user-supplied files; synthetic
fallback for CI) → `add_basemap` + `add_terrain` + detector marker →
overlay a synthetic wavefield (`alpha=`) → `animate_scalar`. GeoMap
wide-area variant (`add_relief` + `add_field`). Wire into
docs/web/{en,ja} tutorials alongside `intro_mapplotting.ipynb`; document
tile-provider terms of use and attribution.

Depends on: #C-3 .. #C-7 (real DEM data path additionally on #547/#548).
Tiles are mocked/bundled for CI (network marker for live fetches).
```
