# 地形データ（DEM）の ScalarField 読み込み — 設計書

> Last-updated: 2026-07-31 (rev 1 — 初版)
> Reviewer Status: **draft**（ユーザーレビュー待ち）

Status: planned

対象テーマ: **Experiment data workflow**（将来テーマ候補、リリース版・時期未割当）

---

## Goal

国土地理院 DEM（JPGIS/GML）およびグローバル標準 GeoTIFF（ASTER GDEM / SRTM 等）の地形標高データを、`ScalarField.read()` で直接読み込めるようにする。第一の応用例はニュートンノイズ評価（KAGRA 周辺地形の空間スペクトル解析）だが、特定用途に限定しない汎用の地理参照付きフィールド基盤として設計する。

本設計書は 2026-07-31 の設計会議（ユーザー × ChatGPT、議事録は維持者ローカル保管）の合意事項を、gwexpy の実装慣行に沿った実装可能な設計へ落としたものである。

### 会議で合意された前提（本設計の拘束条件）

1. API は `ScalarField.read(file_or_list, format=..., ...)`（GWpy の TimeSeries/FrequencySeries API 思想を踏襲）
2. 緯度経度データは**角度のまま軸に保持** → 距離変換メソッドを後実装 → 安定後に読み込み時オプションのエイリアス追加
3. 欠損値は **NaN で読み込む**（読み込み時補間なし）→ `fill_missing()` を後実装 → 安定後にエイリアス
4. 鉛直基準は海抜 0 m（ジオイド）をデフォルトとする。座標回転は読み込み時に行わない（別 umbrella issue）
5. データ規約: `shape=(1, nx, ny, 1)`、`axis0=[0 s]`・`axis3=[0 m]` はダミー軸、値 = 標高、座標はピクセル中心、y 降順は native 保持
6. v1 は直交・等間隔・north-up ラスタ限定。GeoTIFF 書き出しは 2D 交換形式限定
7. 対応フォーマット順: ①国土地理院 DEM（JPGIS/GML）→ ②GeoTIFF（ASTER GDEM / SRTM）
8. 依存は optional extras `geospatial`（必須依存にしない）

---

## 背景 — コードベース調査で確定した設計前提

設計に先立ち実施したコードベース調査（2026-07-31）で確定した事実。以下が本設計の形を決めている。

1. **`ScalarField` に read/write は存在しない。** gwpy の Array4D 系は unified I/O 非対応（gwpy で `read` を持つのは Series/Array2D のみ）。gwexpy 独自コンテナへの後付け前例は `SeriesMatrixIOMixin`（`gwexpy/types/series_matrix_io.py`）。また `docs/developers/contracts/public_io_contract.md` は Field 系を契約対象外と明記しており、`ScalarField.read` の新設は I/O 契約の新領域になる。
2. **緯度経度の deg 軸は現行では構築できない。** `FieldBase._validate_domain_units()`（`gwexpy/fields/base.py:508-540`）は real domain の空間軸に `u.m` 等価単位を要求する（dimensionless のみスキップ）。`space_domain` の許容値も `"real" | "k"` のみ。合意 2「角度のまま保持」の実現には **`space_domain` への第 3 の値 `"angle"` の追加（FieldBase 拡張）が必須**。
3. **`_gwex_*` 属性伝播機構が既存。** `gwexpy/types/array.py:42-45` の `__array_finalize__` が `_gwex_` 接頭辞の属性を slice/view/ufunc 結果へ自動コピーし、`fft_space` 等は `_propagate_gwex_attrs()`（`array.py:78-82`）を明示的に呼ぶ。地理メタデータはこの機構に載せることで `_metadata_slots` を変更せずに全伝播経路をカバーできる。
4. **軸の等間隔（regular）判定が極めて厳格。** `_REGULAR_RTOL = 2.5e-14`（`gwexpy/types/axis.py`）のため、`np.arange(8)*0.1 + 100.0` ですら irregular 判定になる。float64 の表現丸めにより、非正確間隔 × 数百点（SRTM の 3601 点格子など）は局所座標化後でも `fft_space` が失敗しうる。**原点シフトだけでは不十分で、判定許容誤差のパラメータ化が空間 FFT ワークフロー成立の前提条件**になる。
5. **静的場（時間軸 1 点）の前例あり。** `gwexpy/interop/meshio_.py` の `_pack_as_scalar_field`（`axis0=[0.0]` + singleton padding）が標準テンプレート。nt=1 でも `plot` / `fft_space` は動作し、`fft_time` / `psd` は明示拒否される。
6. **国土地理院 GML は stdlib（`xml.etree` + `zipfile`）でパース可能。** 第一目標の GSI reader は追加依存ゼロで実装でき、rasterio の導入は GeoTIFF 対応まで遅延できる。
7. extras 追加時の連動更新箇所: `pyproject.toml` / `gwexpy/interop/_optional.py`（`_EXTRA_MAP`・`_EXTRAS_INCLUDED_IN_ALL`）/ `tests/interop/test_interop_contract.py`（自動照合）/ `docs/web/{en,ja}/user_guide/installation.md`・`io_formats.md` / conda-forge roadmap 表。
8. 既知の関連問題（本設計と独立だが隣接）: `ifft_space` の降順軸再構成がコメントと不一致（実測では昇順になる）、空間軸原点が fft 往復で保存されない、降順軸のテストが 0 件。DEM は y 降順軸を常用するため、これらはサブイシュー #1 および回転 umbrella の検討項目として記録する。

---

## データ規約

reader が返す `ScalarField` の規約。全 reader 共通で `gwexpy/fields/io/core.py` の共有ヘルパが保証する。

| 項目 | 規約 |
| :--- | :--- |
| shape | `(1, nx, ny, 1)` |
| axis0 | `[0.] * u.s`（静的場のダミー軸。meshio 前例に一致） |
| axis3 | `[0.] * u.m`（ダミー鉛直軸。標高は値であって軸ではない） |
| 値 | 標高 [m]（ジオイド基準）。`z_physical = z_stored × scale + offset` を適用 |
| 座標 | ピクセル中心。`AREA_OR_POINT` 等の由来情報は GeoMetadata に保存のみ |
| 地理座標入力 | `axis_names=("t","lon","lat","z")`、`space_domain={"lon":"angle","lat":"angle","z":"real"}`、lon=deg 昇順・lat=deg **native 降順**（北→南） |
| 投影座標入力 | `axis_names=("t","x","y","z")`、axis1/axis2 は m で `"real"` |
| 転置 | rasterio `(band, row, col)` → band 選択 → `.T` → `[newaxis, :, :, newaxis]`。GSI tupleList（`+x-y` 順）も同じ `(y_desc, x_asc)` 中間形を経由 |
| `axis_order="ascending"` | データと軸を**必ず同時に**反転し、GeoMetadata.history に記録。データのみ反転する経路はコード上存在させない |
| 欠損 | NoData（GSI: `-9999.`、GeoTIFF: nodata タグ）→ NaN。整数 DEM は float 昇格。読み込み時補間なし |

---

## モジュール構成

### 新規ファイル

```
gwexpy/fields/geo.py                 # GeoMetadata frozen dataclass + geo_bounds()（stdlib+numpy のみ）
gwexpy/fields/io/__init__.py         # import 副作用で reader 登録（timeseries/io/__init__.py と同型）
gwexpy/fields/io/_registration.py    # register_field_format()（ScalarField 用、default_registry へ登録）
gwexpy/fields/io/core.py             # read_scalarfield() 本体（source 正規化・format 解決・axis_order 適用）
gwexpy/fields/io/gsi_dem.py          # 国土地理院 DEM (JPGIS GML) reader。stdlib のみ。zip・モザイク対応
gwexpy/fields/io/geotiff.py          # GeoTIFF reader/writer。rasterio 条件付き登録（win.py 方式）
```

### 既存ファイルの変更

```
gwexpy/fields/base.py        # space_domain "angle" 追加（deg/rad 等価検証）+ geo プロパティ
gwexpy/fields/scalar.py      # read/write classmethod 接続、fill_missing、to_local_cartesian、
                             #   fft_space/ifft_space の geo 無効化フック + angle 軸拒否メッセージ
gwexpy/fields/signal.py      # freq_space 系に angle 軸ガード
gwexpy/fields/__init__.py    # 末尾で `from . import io as _io`（登録トリガ）
gwexpy/types/axis.py         # is_regular(rtol=, atol_ulps=) パラメータ化 + mean_delta()（既定挙動不変）
pyproject.toml               # geospatial = ["rasterio", "pyproj"]（all には含めない）
gwexpy/interop/_optional.py  # _EXTRA_MAP に rasterio/pyproj → "geospatial"
docs/web/{en,ja}/user_guide/io_formats.md, installation.md   # フォーマット表・extras 表更新
```

配置根拠: reader は「型ごとに `gwexpy/<type>/io/`」という既存慣行（NDScope 等の実績）に従う。GeoMetadata は座標変換基盤（`gwexpy/coordinates`、回転 umbrella の領分）ではなくフィールドの provenance メタデータなので `gwexpy/fields/geo.py` に置く。

---

## 公開 API

### GeoMetadata（`gwexpy/fields/geo.py`）

```python
@dataclass(frozen=True)
class GeoMetadata:
    """ラスタ由来フィールドの地理参照 provenance。

    軸座標配列が唯一の真実源。transform は読み込み時点の provenance として
    凍結し、bounds・解像度が必要な場面では軸から導出する（geo_bounds() 参照）。
    """
    crs: str | None = None            # "EPSG:6668" 等の文字列（rasterio 型は保持しない。pickle 安全）
    transform: tuple[float, ...] | None = None   # GDAL 6 係数。provenance 固定・更新しない
    vertical_datum: str = "geoid"     # 例 "T.P. (JGD2011 vertical)" / "EGM96 geoid"
    nodata: float | None = None       # 元ファイルの NoData 値（データ側は NaN 化済み）
    band: int | None = None           # 読み込んだ band（1 始まり）
    source: tuple[str, ...] = ()      # ファイルパス / GSI メッシュコード
    source_format: str | None = None  # "gsi-dem" | "geotiff"
    active: bool = True               # False = 地理参照は provenance のみ（k 空間・局所座標化後）
    local_origin: tuple[float, float] | None = None   # to_local_cartesian の (lon0, lat0)
    history: tuple[str, ...] = ()     # append-only の処理履歴

    def with_history(self, entry: str) -> "GeoMetadata": ...
    def deactivated(self, reason: str) -> "GeoMetadata": ...


def geo_bounds(field) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) をピクセル中心軸からセル縁へ外挿して返す。"""
```

### FieldBase 拡張（`gwexpy/fields/base.py`）

- `space_domain` の許容値に `"angle"` を追加（str 形式・dict 値の両方）。
- `_validate_domain_units()` に分岐追加: `real → u.m 等価`、`k → 1/u.m 等価`、`angle → u.rad 等価`（deg を含む）。dimensionless はこれまで通りスキップ。
- `geo` プロパティ（getter/setter）。実体は `_gwex_geo` 属性で、`_metadata_slots` は変更しない。

```python
@property
def geo(self) -> GeoMetadata | None:
    """Georeferencing metadata (None when not georeferenced)."""
    return getattr(self, "_gwex_geo", None)
```

### ScalarField の read/write（`gwexpy/fields/scalar.py`、実装は `fields/io/core.py` へ委譲）

```python
@classmethod
def read(cls, source, format: str | None = None, **kwargs) -> "ScalarField":
    """ファイル（単体・zip・複数タイルのリスト）から ScalarField を読み込む。

    format: "gsi-dem" | "geotiff"。None なら identifier（内容 magic → 拡張子）で自動判定。
    """

def write(self, target, format: str | None = None, **kwargs) -> None:
    """v1 は format="geotiff"（2D 交換形式）のみ対応。"""
```

### reader / writer 関数

```python
# gwexpy/fields/io/gsi_dem.py（追加依存ゼロ: xml.etree + zipfile）
def read_scalarfield_gsi_dem(
    source, *,
    axis_order: Literal["native", "ascending"] = "native",
    bounds: tuple[float, float, float, float] | None = None,  # (lonmin, latmin, lonmax, latmax)
    unit="m",                          # GSI は常に m。上書き用
    vertical_datum: str | None = None, # 既定 "T.P. (JGD2011 vertical)"
    dtype="float64",
    name: str | None = None,
) -> ScalarField: ...

def identify_gsi_dem(origin, filepath, fileobj, *args, **kwargs) -> bool:
    """先頭 ~2KB の FGD GML 名前空間検出 → 拡張子 .xml/.zip + "FG-GML-*DEM*" フォールバック。
    例外は握って False（既存 identify 慣行）。"""

# gwexpy/fields/io/geotiff.py（rasterio。未導入時は registry 未登録 = win.py 方式）
def read_scalarfield_geotiff(
    source, *,
    band: int | None = None,           # 複数 band で未指定なら ValueError
    unit=None,                         # 単位タグなし・未指定なら ValueError
    axis_order: Literal["native", "ascending"] = "native",
    bounds=None,                       # rasterio window 読み（部分読み込み）
    vertical_datum: str | None = None, # 既定 "geoid (unspecified)"、VERT_CS があればそれ
    dtype="float64",
    name: str | None = None,
) -> ScalarField: ...

def write_scalarfield_geotiff(
    field, target, *,
    crs: str | None = None,            # 省略時 field.geo.crs（active 必須）
    nodata: float = float("nan"),
    dtype: str = "float32",
    **rasterio_kwargs,
) -> None: ...
```

```python
# gwexpy/fields/io/_registration.py
def register_field_format(
    format_name: str, *,
    reader=None, writer=None,
    magic_identifier=None, extension=None,
    aliases: tuple[str, ...] = (), force: bool = True,
) -> None:
    """ScalarField を対象に gwpy.io.registry.default_registry へ登録する。
    register_timeseries_format() の単一クラス版（dict/matrix アダプタなし）。"""
```

### フィールド側メソッド（`gwexpy/fields/scalar.py`）

```python
def fill_missing(
    self,
    method: Literal["nearest", "linear", "constant"] = "nearest",
    *,
    value: float | None = None,          # method="constant" 用
    axes: tuple[str, str] | None = None, # 既定: サイズ>1 の空間軸 2 本を自動選択
    max_distance=None,                   # nearest の充填距離上限（軸単位）
) -> ScalarField:
    """NaN を空間補間で充填した新フィールドを返す（scipy のみ使用・追加依存なし）。
    geo は active のまま history に追記。全画素 NaN は ValueError。"""

def to_local_cartesian(
    self,
    origin: tuple[float, float] | None = None,   # (lon0, lat0)。None なら格子中心
    *,
    names: tuple[str, str] = ("x", "y"),
    ellipsoid: str = "GRS80",
) -> ScalarField:
    """angle 軸 (lon, lat) を局所平面近似 x=R·cos(lat0)·Δlon, y=R·Δlat [m] へ変換する。
    軸を (arange(n)-i0)*d の形で再合成して正則性を最大化（依存ゼロ）。
    space_domain は real 化、geo は deactivated + local_origin 記録。
    近似誤差 ~ (格子スパン/地球半径)^2 程度。広域は将来の to_projected() を使う。"""

def to_projected(self, target_crs: str, *, resolution=None, method="bilinear") -> ScalarField:
    """将来実装（pyproj による厳密再投影 + 再標本化）。v1 ではシグネチャ予約のみ。"""
```

---

## 設計決定と理由

### A. read の実装方式 — 手書き classmethod + 内部 registry ディスパッチのハイブリッド

`UnifiedReadWriteMethod` の後付け（gwpy/astropy connect 機構）は docstring 書き換え等の複雑さ（`_registration.py` の `_ensure_registry_docstring` ハックが必要になった実績）に見合わない。gwexpy 独自コンテナの前例（`SeriesMatrixIOMixin`）も手書き classmethod である。ただし同 Mixin と違い**自前ディスパッチ表は持たず**、format 解決・読み出しは `default_registry`（`register_field_format` 経由の登録）へ委譲する。これによりフォーマット追加は登録 1 行で済み、将来 `UnifiedReadWriteMethod` へ移行しても呼び出しシグネチャは不変。手書き層の責務は (i) list/zip ソースの正規化、(ii) `format=None` 時の identify と親切なエラー、(iii) 既存 interop docstring（openems/meep）が先取りしている「classmethod 接続」実態との整合、に限る。

public_io_contract は現行の Container Semantic Contract（v0.2.0）では**対象外を維持**し、GeoTIFF write 安定後に契約組み入れの follow-up issue（#10）で監査する。

### B. GSI DEM reader — フォーマット名 "gsi-dem"、モザイクは共通ラティス検証 + NaN 充填

- registry は (format, class) キーなので既存 "xml.diaggui" と衝突しない。canonical 名 "gsi-dem" 一本（エイリアスなし）。GeoTIFF は "geotiff"（identify: TIFF magic `II*\0` / `MM\0*` → 拡張子 .tif/.tiff）。
- モザイク結合手順: 各タイルから (Envelope, 格子数, dlon/dlat, 値配列) を抽出し、
  1. srsName 不一致 → ValueError
  2. 解像度不一致（相対差 > 1e-6）→ ValueError（DEM5A と DEM10B の混在拒否。DEM5A+5B は解像度一致なら許容し種別を `geo.source` に記録）
  3. タイル原点が共通ラティスに乗らない（小数部 > 0.01 px）→ ValueError
  4. 選択タイル群の union bbox を NaN 初期化 → タイル配置。**タイル間の隙間は NaN のまま許容**（欠測=NaN 規約と一貫）。同一メッシュ重複は UserWarning + 後勝ち
- zip 入力（GSI 配布形態）を v1 に含める。ディレクトリ入力は対象外。

### C. angle domain — `space_domain` に第 3 の値 "angle" を追加

変更は次の 3 系統に閉じる: `base.py` の `__new__` バリデーションと `_validate_domain_units`（u.rad 等価検証）、`fft_space`/`ifft_space` のガード（angle 軸は「`to_local_cartesian()` を先に」と誘導するメッセージで拒否）、`signal.py` の k 変換系ガード。将来の `to_projected(target_crs=...)` は「angle → real の domain 遷移 + 再標本化」であり、読み込み時エイリアス（`projected=` オプション）は同メソッド安定後に read kwarg として薄く被せる（合意 2 の段取り通り）。

### D. GeoMetadata の保持・伝播 — `_gwex_geo` 属性 + transform は provenance 固定

`_metadata_slots` には**追加しない**。既存の `_gwex_*` 機構が slice/演算/fft の全伝播経路を既にカバーしており、pickle・repr・既存契約への影響もゼロ。公開面は `FieldBase.geo` プロパティのみ。

transform は「読み込み時点の記録」として凍結し、slice 等での更新は行わない。gwexpy では軸座標配列が唯一の真実源であり、transform の二重管理は不整合バグの温床になるため。bounds が必要な場面は `geo_bounds()` が軸から都度導出する。

`fft_space`/`ifft_space` の末尾（`_propagate_gwex_attrs` 直後）で「geo があれば `deactivated("fft_space")` に置換」する。ifft でも再活性化しない（fft で軸原点情報が失われるため復元不能）。

### E. 段階分割 — 8 PR + follow-up（後述のイシュー構成）

GSI reader（依存ゼロ）が extras 導入前に headline 機能として完結する順序が肝。#1/#2/#3 は並行着手可能で、クリティカルパスは #2 → #3 → #4。

### F. bounds 部分読み込み — v1 に含める

- GeoTIFF: rasterio の window 読みへの素通しで実装コスト極小。ASTER/SRTM（1 タイル 100 MB 級）で実益大。
- GSI: bounds と Envelope の交差判定で XML パース自体をスキップするタイル事前フィルタのみ。サブタイル切り出しはせず、読み込み後のフィールドスライスに委ねる。

---

## 会議合意からの意図的な差分（要レビュー）

| # | 会議での案 | 本設計 | 理由 |
| :--- | :--- | :--- | :--- |
| 1 | x/y slice で transform・bounds を更新（議事録 §11） | transform は provenance 固定、bounds は軸から導出 | gwexpy は軸座標配列が真実源。二重管理は不整合バグの温床 |
| 2 | extras に rioxarray を含める（議事録 §5） | rasterio + pyproj のみ | xarray を経由しない実装のため不要 |
| 3 | —（議事録に無し） | axis regular 判定のパラメータ化（サブイシュー #1）を追加 | 調査で判明した空間 FFT 成立の前提条件（背景 §4） |

---

## メタデータ伝播規則

| 操作 | geo の扱い | active |
| :--- | :--- | :--- |
| slice / 算術 ufunc / copy / pickle / fft_time | 保持 | 不変 |
| fft_space / ifft_space | 保持 + deactivated + history 追記（ifft でも再活性化しない） | False へ |
| fill_missing / 読み込み時 axis_order="ascending" | 保持 + history 追記 | 不変 |
| to_local_cartesian | 保持 + local_origin 記録 + deactivated | False へ |
| to_projected（将来） | 新しい active geo に置換 + 履歴連鎖 | True |
| signal 派生量（psd/coherence 等）/ to_xarray_field | 伝播しない（v1 既知制限として文書化） | — |

算術 ufunc は第 1 フィールドオペランドから継承（座標一致は既存 `__array_ufunc__` が強制済み）。crs 不一致の検査は v1 では行わない（既知制限）。

---

## エラー仕様

新設例外クラスは作らない（リポジトリ慣行: `ValueError` / `ImportError` + `UserWarning`）。

| 条件 | 例外 | メッセージ要点 |
| :--- | :--- | :--- |
| rasterio 未導入で geotiff 指定 | ImportError | `pip install 'gwexpy[geospatial]'`（`ensure_dependency(extra="geospatial")`。未導入時は registry 未登録） |
| format=None で判定不能 | ValueError | format 指定を促す（SeriesMatrixIOMixin と同文型） |
| GeoTIFF: CRS なし | ValueError | v1 は読み込み側 crs 上書き非対応。GIS での付与を案内 |
| GeoTIFF: 複数 band で band 未指定 | ValueError | band 数と `band=` 指定例を提示 |
| GeoTIFF: 単位タグなし・unit 未指定 | ValueError | `unit="m"` 指定例を提示 |
| 回転・せん断格子（transform.b or d ≠ 0） | ValueError | v1 は north-up 直交格子のみ |
| 全画素 NaN（単一・モザイク後） | ValueError | ソース列挙 |
| GSI: FGD GML 構造でない / sequenceRule が "+x-y" 以外 | ValueError | 対応形式の明示 |
| モザイク: srsName / 解像度 / ラティス不整合 | ValueError | 不一致タイルのパスと値を列挙 |
| モザイク: 同一メッシュ重複 | UserWarning | 後勝ちで上書きした旨 |
| angle 軸への fft_space / freq_space 系 | ValueError | 「angle domain。`to_local_cartesian()` を先に」 |
| fill_missing: 全画素 NaN / constant で value 未指定 | ValueError | — |
| write geotiff: geo が None/inactive で crs= も無い | ValueError | provenance のみでは書けない旨 |
| write geotiff: 時間軸長 > 1 等の 4D 非退化 | ValueError | 「2D 交換形式。4D は NetCDF/Zarr を使え」 |

---

## テスト計画

実行: `conda run -n gwexpy pytest`。fixture は**合成データのみ**（GSI 実タイルは利用規約確認が未了のため同梱しない。実タイル 1 枚での手動検証手順を PR に記載する）。

| ファイル | 主要ケース |
| :--- | :--- |
| `tests/types/test_types_axis.py`（追記） | `is_regular(rtol=)` の緩和判定、既定挙動不変、`mean_delta()`、log 軸は緩めても irregular |
| `tests/fields/test_scalarfield_angle_domain.py` | "angle" 受理（str/dict）、deg/rad OK・m は ValueError、`__array_finalize__` 伝播、fft_space 拒否メッセージ、signal 系ガード、既存 "real"/"k" 回帰 |
| `tests/fields/test_scalarfield_geo_metadata.py` | 伝播規則表を行単位で検証（slice 保持 / 演算左継承 / fft で deactivated / ifft 非再活性化 / pickle round-trip / frozen 性） |
| `tests/io/test_gsi_dem_reader.py` | tupleList 解析、startPoint オフセット、-9999→NaN、**ピクセル中心座標の数値一致**、**転置の正しさ**（既知パターンで `data[0,ix,iy,0]` == 期待値）、**lat 軸 native 降順**、zip 展開、モザイク（2×2、隙間 NaN、不整合 ValueError、重複 warning）、bounds タイルフィルタ |
| `tests/io/test_gsi_dem_public_io.py` | `ScalarField.read` 経由: format 自動判定、list 入力、`axis_order="ascending"` で**データと軸の同時反転**（値-座標対応が native と同一）、shape/(1,nx,ny,1)・ダミー軸単位、geo.crs、エラー系 |
| `tests/io/test_geotiff_reader.py` | `importorskip("rasterio")`。地理座標/投影座標の両 fixture、nodata→NaN、band 選択・未指定エラー、単位未指定エラー、回転格子エラー、bounds window、**CRS 文字列保持** |
| `tests/io/test_geotiff_public_io.py` | read→write→read round-trip: **値・座標・y 向きが反転しない**、NaN→nodata→NaN、inactive geo の write 拒否 |
| `tests/fields/test_scalarfield_fill_missing.py` | nearest/linear/constant、max_distance、全 NaN エラー、geo history 追記、非欠測画素の不変性 |
| `tests/fields/test_scalarfield_local_cartesian.py` | 原点既定/指定、既知 2 点間距離の近似精度、軸正則性（緩和判定で fft_space が通る = ニュートンノイズワークフロー疎通）、geo deactivated + local_origin |
| `tests/interop/test_interop_contract.py`（追記） | geospatial extra が pyproject と `_EXTRA_MAP` で整合、`_EXTRAS_INCLUDED_IN_ALL` 非包含 |
| fixture 生成器 | `tests/io_conformance/generators/gsi_dem.py`（合成 GML 文字列） |

会議で挙がった回帰必須項目（ピクセル中心 / 転置 / y 降順 native / NoData→NaN / CRS 保持 / 書き戻し非反転）は上表の太字で網羅。

---

## PR 分割とイシュー構成

Umbrella issue 2 本 + サブイシュー構成。すべて **Experiment data workflow の将来テーマ候補**（milestone 未割当）。実装時期は未定。

### Umbrella A: Geospatial DEM support for ScalarField

| # | サブイシュー | 依存 |
| :--- | :--- | :--- |
| 1 | types: parameterizable axis regularity check (`is_regular` + `fft_space(spacing_rtol=)`) | なし |
| 2 | fields: add "angle" space domain to FieldBase | なし |
| 3 | fields: GeoMetadata provenance dataclass + `FieldBase.geo` | 2 |
| 4 | fields/io: `ScalarField.read` scaffold + GSI DEM (JPGIS GML) reader with mosaic | 2, 3 |
| 5 | build: `geospatial` extra + GeoTIFF reader (ASTER GDEM / SRTM) | 4 |
| 6 | fields: `ScalarField.fill_missing()` | 3 |
| 7 | fields: `ScalarField.to_local_cartesian()` | 1, 2, 3 |
| 8 | fields/io: GeoTIFF writer (2D exchange) + round-trip regressions | 5 |
| 9 | (follow-up) `to_projected(target_crs)` via pyproj + read-time aliases | 5, 6, 7 |
| 10 | (follow-up) public_io_contract への Field direct-I/O 組み入れ監査 | 8 |
| 11 | docs: DEM チュートリアル notebook（GSI → fill_missing → 局所座標化 → 空間 FFT） | 4–8 |

PR は #1〜#8 が 1:1 対応（docs 行更新は各 PR に同梱、#11 のみ独立）。#1/#2/#3 は並行着手可能。クリティカルパス: #2 → #3 → #4。

### Umbrella B: Coordinate frames and rotation for Field classes（概要のみ・詳細設計は別途）

会議 §20–28 で合意された方向性の記録: 受動座標変換（`to_frame`）と再格子化（`regrid_to_frame`）と水平回転ショートカット（`rotate_horizontal`）の分離、`GridGeometry`（origin + 3×3 basis）、`DetectorFrame`、Astropy アダプター（天文座標は独自実装しない）、Scalar/Vector/Tensor の変換則分離（φ'=φ / v'=Rv / T'=RTR^T）。VectorField/TensorField が FieldDict(dict) ベースである点は設計時の考慮事項。

---

## Non-Goals（v1 で行わないこと）

- 座標回転・検出器座標系変換（Umbrella B）
- pyproj による厳密再投影・再標本化（`to_projected` は follow-up #9）
- 回転・せん断格子、曲線座標格子、GCP 地理参照の対応
- lazy loading / Dask 化（現行 Field は eager な ndarray サブクラス。部分読み込みは bounds で対応）
- GeoTIFF への 4D 完全 round-trip（NetCDF/Zarr の役割）
- 読み込み時の自動補間・自動投影（安定後にエイリアスとして追加する合意手順に従う）
- public_io_contract への Field 組み入れ（follow-up #10）
- STL 等の相対座標系ポリゴンデータの読み込み（会議合意: 基準点はオプション引数で指定、デフォルトは重心。フォーマット優先順位で GSI・GeoTIFF の後段としたため v1 スコープ外。着手時に別イシュー化）

---

## リスク・未解決事項

1. **regular 判定の既定値変更の是非**（最重要）: 本設計はオプトイン（`spacing_rtol`）で迂回するが、既定 atol を値 ULP 基準へ変えるコア変更はサブイシュー #1 内で別途議論（既存テストとの整合が必要）。
2. **GSI GML の変種**: DEM5A/5B/10B/10A、旧 JPGIS（非 GML）、sequenceRule 変種。v1 は `+x-y` のみ対応しエラー明示。合成 fixture のみでは不十分な可能性 → 実タイル 1 枚での手動検証手順を PR #4 に記載。
3. **モザイク union bbox のメモリ肥大**（遠隔タイル同時指定時）: v1 は文書警告 + bounds 推奨。
4. **crs 不一致の演算が素通り**（座標一致なら通る）: v1 既知制限として文書化。
5. **geo の永続化**: pickle 以外（将来の Field HDF5/NetCDF/Zarr、to_xarray_field）で geo が失われる。follow-up で attrs エンコードを検討。
6. **vertical_datum の正準文字列**: "T.P. (JGD2011 vertical)" は仮置き。GSI 製品仕様の表記に合わせて確定する。
7. **DEM5A+5B 混在許容**（解像度一致時のみ + geo.source に記録）の是非: 要レビュー。
8. **rasterio の導入コスト**: conda-forge では GDAL 連鎖が重い。`geospatial` を `all` に含めない判断を installation docs に明記。CI レーン追加はサブイシュー #5 で決定。

---

## 付録: イシュー本文ドラフト

投稿前にユーザー承認を得ること。以下は `gh issue create` に渡す本文案。

### Umbrella A

**Title**: `[Umbrella] Geospatial DEM support for ScalarField (future theme)`

```markdown
## Goal

Read terrain elevation data (GSI DEM / GeoTIFF) directly into `ScalarField`,
with georeferencing provenance that survives field operations. Primary use
case: Newtonian-noise studies around KAGRA (spatial spectra of terrain), but
the design is application-agnostic.

Design document: `docs/developers/plans/active/2026-07-31-terrain-scalarfield-io-design.md`

## Data conventions

- `shape=(1, nx, ny, 1)`; `axis0=[0 s]`, `axis3=[0 m]` are dummy axes; values = elevation [m, geoid-referenced]
- Pixel-center coordinates; native axis order preserved (lat typically descending)
- Geographic input keeps degrees on the axes (new `space_domain="angle"`); missing values become NaN (no interpolation at read time)
- Vertical reference: mean sea level (geoid) by default

## Non-goals

Coordinate rotation / detector frames (separate umbrella), pyproj reprojection
(follow-up), rotated/sheared grids, lazy loading, 4D GeoTIFF round-trip,
read-time auto-interpolation/projection (added later as aliases per the agreed
staged plan), public_io_contract inclusion (follow-up audit).

## Sub-issues

- [ ] #1 types: parameterizable axis regularity check
- [ ] #2 fields: "angle" space domain
- [ ] #3 fields: GeoMetadata + `FieldBase.geo`
- [ ] #4 fields/io: `ScalarField.read` + GSI DEM reader (mosaic)
- [ ] #5 build: `geospatial` extra + GeoTIFF reader
- [ ] #6 fields: `fill_missing()`
- [ ] #7 fields: `to_local_cartesian()`
- [ ] #8 fields/io: GeoTIFF writer + round-trip regressions
- [ ] #9 (follow-up) `to_projected()` + read-time aliases
- [ ] #10 (follow-up) public_io_contract audit for Field direct I/O
- [ ] #11 docs: DEM tutorial notebook

## Scheduling

Future theme; milestone unassigned. Issues #1/#2/#3 can proceed in parallel; the critical
path is #2 → #3 → #4.
```

### サブイシュー 1

**Title**: `types: parameterizable axis regularity check (is_regular + fft_space(spacing_rtol=))`

```markdown
`AxisDescriptor.regular` uses `_REGULAR_RTOL=2.5e-14` with a 1-ULP atol on the
spacing, so any float64 arithmetic sequence with a large offset-to-spacing
ratio (e.g. `np.arange(8)*0.1 + 100.0`) or an inexact spacing over hundreds of
points (SRTM 3601-point grids) is judged irregular, and `fft_space` raises.
This blocks the DEM → spatial-FFT workflow even after shifting to a local
origin.

Proposal (backward-compatible, opt-in):
- `AxisDescriptor.is_regular(*, rtol=None, atol_ulps=None)` — parameterized check; `None` keeps current defaults
- `AxisDescriptor.mean_delta()` — representative spacing for the relaxed check
- `ScalarField.fft_space(..., spacing_rtol: float | None = None)` — use the relaxed check and `mean_delta()` when given; default `None` keeps the current strict behaviour

Whether the *default* tolerance should change (value-ULP based instead of
spacing-ULP based) is a separate decision to be discussed in this issue —
existing tests assert the strict behaviour.

Also record here the two adjacent known issues found during design review:
`ifft_space` descending-axis reconstruction contradicts its comment (actual
output is ascending), and the spatial-axis origin is not preserved through an
`fft_space`/`ifft_space` round trip. Zero descending-axis test cases exist.

Design doc: `docs/developers/plans/active/2026-07-31-terrain-scalarfield-io-design.md`
Depends on: none. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 2

**Title**: `fields: add "angle" space domain to FieldBase`

```markdown
`space_domain` currently allows only `"real" | "k"`, and
`_validate_domain_units` requires real-domain spatial axes to be equivalent to
`u.m` — so latitude/longitude axes in degrees cannot be constructed at all.
The agreed DEM design keeps geographic input in degrees at read time, which
requires a third domain value.

Scope:
- Accept `"angle"` in `FieldBase.__new__` (str and per-axis dict forms)
- `_validate_domain_units`: angle → require `u.rad`-equivalent units (deg OK)
- `fft_space`/`ifft_space`: reject angle axes with a message pointing to
  `to_local_cartesian()` (and, later, `to_projected()`)
- `signal.py` freq-space helpers: same guard
- Tests: `tests/fields/test_scalarfield_angle_domain.py` (acceptance, unit
  validation, propagation through `__array_finalize__`, fft rejection,
  regression for existing "real"/"k" paths)

Depends on: none. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 3

**Title**: `fields: GeoMetadata provenance dataclass + FieldBase.geo`

```markdown
Add `gwexpy/fields/geo.py` with a frozen `GeoMetadata` dataclass (crs,
transform, vertical_datum, nodata, band, source, source_format, active,
local_origin, history) and a `FieldBase.geo` property backed by the existing
`_gwex_*` attribute-propagation mechanism (`types/array.py`) — no
`_metadata_slots` change needed.

Key rules (see design doc for the full propagation table):
- Axis coordinate arrays are the single source of truth; `transform` is frozen
  read-time provenance (never updated on slicing); bounds are derived from the
  axes via `geo_bounds()`
- `fft_space`/`ifft_space` replace geo with `deactivated("fft_space")` (no
  re-activation on inverse)
- Arithmetic/slicing/pickle keep geo as-is

Tests: `tests/fields/test_scalarfield_geo_metadata.py` (row-by-row propagation
table, pickle round-trip, frozen-ness).

Depends on: #2. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 4

**Title**: `fields/io: ScalarField.read scaffold + GSI DEM (JPGIS GML) reader with mosaic`

```markdown
First headline feature — zero new dependencies (stdlib `xml.etree` + `zipfile`).

Scope:
- `gwexpy/fields/io/` package: `_registration.py` (`register_field_format()`
  targeting `ScalarField` in gwpy's `default_registry`), `core.py`
  (`read_scalarfield()`: source normalization, format resolution via registry,
  `axis_order` handling), `gsi_dem.py`
- `ScalarField.read(source, format=None, **kwargs)` / `write()` classmethods
  (thin wrappers; hybrid design — see design doc §A)
- GSI reader: FGD GML parsing (`tupleList`, startPoint offset, sequenceRule
  "+x-y" only), NoData -9999 → NaN, zip input, multi-tile mosaic (srsName /
  resolution / lattice-alignment checks; gaps stay NaN; duplicate mesh →
  warning, last wins), `bounds=` tile pre-filter, `axis_order="native"|"ascending"`
- Output conventions: `(1, nx, ny, 1)`, lon/lat in degrees with
  `space_domain="angle"`, pixel-center coordinates, lat native-descending
- Format name: `"gsi-dem"`; content-based identifier with extension fallback
- Tests: `tests/io/test_gsi_dem_reader.py`, `test_gsi_dem_public_io.py`;
  synthetic GML fixture generator in `tests/io_conformance/generators/gsi_dem.py`
  (no real GSI tiles bundled; manual verification steps for one real tile
  documented in the PR)

Depends on: #2, #3. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 5

**Title**: `build: geospatial extra + GeoTIFF reader (ASTER GDEM / SRTM)`

```markdown
Add the `geospatial = ["rasterio", "pyproj"]` extra (NOT included in `all` —
GDAL dependency chain is heavy) and the GeoTIFF reader.

Scope:
- `gwexpy/fields/io/geotiff.py`: conditional registration when rasterio is
  importable (win.py pattern); `ensure_dependency(extra="geospatial")` error
  otherwise
- Reader: band selection (error when multiple bands and `band=` missing), unit
  tag handling (error when unknown and `unit=` missing), scale/offset applied,
  nodata → NaN (integer DEM promoted to float), CRS string preserved in geo,
  rotated/sheared transforms rejected, `bounds=` window reads, geographic
  (degree/angle) and projected (metre/real) inputs
- Extras bookkeeping (5-point set): `pyproject.toml`, `interop/_optional.py`
  `_EXTRA_MAP` (+ not in `_EXTRAS_INCLUDED_IN_ALL`),
  `tests/interop/test_interop_contract.py`, installation/io_formats docs
  (en/ja), conda-forge roadmap table
- Tests: `tests/io/test_geotiff_reader.py` (importorskip)

Depends on: #4. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 6

**Title**: `fields: ScalarField.fill_missing()`

```markdown
Explicit NaN infilling as a field-side method (per the agreed staged plan:
read keeps NaN; interpolation is a separate, recorded step).

- `fill_missing(method="nearest"|"linear"|"constant", *, value=None, axes=None,
  max_distance=None)` — scipy only (already a required dependency)
- Rationale: automatic infilling at read time can silently smooth valleys,
  cliffs and tile seams (KAGRA-area rivers/ponds produce large NoData holes at
  ~300 m elevation)
- geo stays active; history gets `fill_missing(method=..., ...)`
- All-NaN input → ValueError; untouched pixels bit-identical
- Read-time alias (`fill_missing=` kwarg) is deferred until this stabilizes
  (tracked in #9)

Depends on: #3. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 7

**Title**: `fields: ScalarField.to_local_cartesian()`

```markdown
Dependency-free local tangent-plane conversion for angle-domain fields:
x = R·cos(lat0)·Δlon, y = R·Δlat (GRS80 radius), origin defaulting to the grid
center. Axes are re-synthesized as `(arange(n)-i0)*d` to maximize regularity so
that `fft_space` (with `spacing_rtol` from #1) works — this is the enabling
step for the Newtonian-noise spatial-spectrum workflow.

- `space_domain` → "real"; geo → deactivated with `local_origin` recorded
- Approximation error ~ (span/R)²; wide areas should use the future
  `to_projected()` (pyproj, #9)
- Tests: known-distance accuracy, regularity of output axes, fft_space
  passthrough, geo bookkeeping

Depends on: #1, #2, #3. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 8

**Title**: `fields/io: GeoTIFF writer (2D exchange) + round-trip regressions`

```markdown
`ScalarField.write(target, format="geotiff")` /
`write_scalarfield_geotiff()` — 2D exchange format only.

- Requires an *active* geo (or explicit `crs=`); provenance-only geo → ValueError
- Requires degenerate time/z axes (true 2D content); otherwise ValueError with
  a pointer to NetCDF/Zarr for full 4D round-trips
- NaN → nodata on write
- Round-trip regressions: read→write→read preserves values, coordinates and
  y-orientation from both native and ascending inputs; no silent flips

Depends on: #5. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 9（follow-up）

**Title**: `fields: to_projected(target_crs) via pyproj + read-time aliases`

```markdown
Rigorous reprojection/resampling (`to_projected(target_crs, resolution=,
method=)`) using pyproj, then thin read-time aliases per the agreed staged
plan: `ScalarField.read(..., projected="EPSG:...", fill_missing="nearest")`
calling the (by then stabilized) field-side methods. Records source/dest CRS,
resolutions, resampling algorithm and bounds in geo.history.

Depends on: #5, #6, #7. Theme: Experiment data workflow (stretch); milestone unassigned.
```

### サブイシュー 10（follow-up）

**Title**: `contracts: audit Field direct I/O for public_io_contract inclusion`

```markdown
`public_io_contract.md` currently excludes Field classes ("Field classes stay
outside this schema slice until their direct-I/O story is audited separately").
Once gsi-dem/geotiff read-write stabilizes (#8), audit and add Field entries
(formats, optional deps, unavailable_behavior, conformance generators) to the
contract, or record the decision not to.

Depends on: #8. Theme: Experiment data workflow; milestone unassigned.
```

### サブイシュー 11

**Title**: `docs: DEM tutorial notebook (GSI read → fill_missing → local frame → spatial FFT)`

```markdown
End-to-end case study notebook: read GSI tiles around KAGRA (user-supplied
files; synthetic fallback for CI), mosaic, `fill_missing`, `to_local_cartesian`,
`fft_space` spatial spectrum. Wire into docs/web tutorials (en/ja).

Depends on: #4–#8. Theme: Experiment data workflow; milestone unassigned.
```

### Umbrella B

**Title**: `[Umbrella] Coordinate frames and rotation for Field classes`

```markdown
## Goal

A shared spatial-geometry layer so that terrain fields, detector frames and
(eventually) celestial frames use one consistent transformation API.

Direction agreed in the 2026-07-31 design meeting (details to be designed in
this issue before any implementation):

- Separate three operations that "rotate" conflates:
  1. `to_frame(frame)` — passive basis change, no interpolation
  2. `regrid_to_frame(frame, resolution=, method=)` — resampling onto a grid
     aligned with the new frame
  3. `rotate_horizontal(angle, center=, resample=)` — explicit-axis shortcut
  4. `assume_frame(frame)` — metadata-only assignment (dangerous, hence the name)
- `GridGeometry` (origin + 3×3 basis + parent frame) so grid orientation is
  representable without touching the 1-D axis model
- `DetectorFrame.from_axes(...)` / `from_arms(..., orthogonalize=)` — origin +
  basis vectors, not Euler angles (matrix/quaternion as the source of truth)
- Value transformation rules per class: scalar φ'=φ; vector v'=Rv; tensor
  T'=RTRᵀ. Note: VectorField/TensorField are dict-based containers
  (FieldDict), not ndarray subclasses — component rotation returns new dicts
- Celestial/time-dependent frames via an Astropy adapter (ITRS/GCRS/ICRS,
  IERS-aware time scales) — no in-house astronomy
- DEM-specific constraint: height fields stay single-valued only under
  vertical-axis rotations; general 3-D rotations may require a SurfaceMesh
  representation (future `HeightField` subclass candidate)

## Relationship to the DEM umbrella

The DEM reader (Umbrella A) deliberately performs no rotation at read time;
fields carry their native CRS/axes. This umbrella provides the later
transformations.

Future theme (design first); milestone unassigned.
```
