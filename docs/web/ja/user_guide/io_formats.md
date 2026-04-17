<a id="io-formats-ja-top"></a>

# ファイル I/O 対応フォーマットガイド

gwexpy の end-user 向け I/O ガイドです。
このページでは、ユーザーが直接使う `.read()` / `.write()` / `fetch()` 系 API だけを扱います。

`to_*()` / `from_*()` による変換や、xarray / ROOT object / Zarr array との橋渡しは、このページでは扱いません。必要な場合は [他ライブラリ連携チュートリアル](tutorials/intro_interop) や [Interop API リファレンス](../reference/api/interop) を参照してください。

:::{warning}
**セキュリティ警告: Pickle 形式の取り扱い**

**Pickle** (:term:`Pickle`) は便利ですが、信頼できないソースから受け取った Pickle ファイルを読み込むことは危険です。悪意のあるファイルにより、実行環境で任意のコードが実行される可能性があります。

共同研究者との共有や長期保存には、**HDF5**、**GWF**、**Zarr** のような構造化された形式を優先してください。
:::

## まず最初に: 判断ルール

- **まず保存形式を選ぶ**なら、GW 系データは **HDF5**、観測網の既存資産は **MiniSEED / SAC / WIN / ATS**、汎用交換は **CSV / NetCDF4 / Zarr**、ロガー固有データは **GBD / TDMS / SDB / WAV / Audio** を起点に考えると整理しやすいです。**MTH5** については、現時点の public direct-I/O は **`ats.mth5` の単一路だけ** で、汎用の standalone **`format="mth5"`** はまだ公開していません。
- **自動判別でよい**のは、拡張子から reader が一意に決まる場合です。
- **`format=` を明示する**のは、`.xml` のように経路が複数ありうる場合、独自拡張子を使っている場合、または実験データで自動判別が不安な場合です。
- **`timezone` を必ず指定する**のは、ファイル内に UTC/GPS がなくローカル時刻だけを持つ形式です。現時点でユーザーが明示必須なのは **GBD** です。
- **Read only / Write only に注意する**: 表の `○ / ×` は「読めるが書けない」「書けるが読めない」を意味します。
- **Field を安全に保存したい**場合は、このページでは **HDF5** と **Pickle** を基準に考えてください。NetCDF4 / Zarr / ROOT object への橋渡しは interop 側の話です。

## セクション移動

- <a href="#io-formats-ja-quick">クイック判定表</a>
- <a href="#io-formats-ja-basic">基本的な使い方</a>
- <a href="#io-formats-ja-a">A. GW標準</a>
- <a href="#io-formats-ja-b">B. 地震・地球物理観測</a>
- <a href="#io-formats-ja-c">C. 汎用・解析用</a>
- <a href="#io-formats-ja-d">D. 計測機器・ロガー</a>
- <a href="#io-formats-ja-dev">開発者向け補足</a>

<a id="io-formats-ja-quick"></a>

## クイック判定表

| グループ | こういうときに選ぶ | 最初に見る形式 | このページで扱う形式 |
|---|---|---|---|
| **A. GW標準** | GW 系の標準保存、共有、取得経路を使いたい | **HDF5** | GWF, HDF5, ndscope-hdf5, DTTXML, NDS2, GWOSC |
| **B. 地震・地球物理観測** | 既存の地震・電磁気観測フォーマットを読む | **MiniSEED** | MiniSEED, SAC, GSE2, K-NET, WIN / WIN32, ATS, ATS.MTH5（MTH5 standalone は状況注記のみ） |
| **C. 汎用・解析用** | 汎用保存、外部解析、交換をしたい | **CSV / TXT** または **Zarr** | CSV / TXT, NetCDF4, Zarr, Pickle, ROOT |
| **D. 計測機器・ロガー** | ロガーや機材固有の時系列を読む | **GBD** または **TDMS** | GBD, TDMS, SDB / SQLite / SQLite3, WAV, MP3, FLAC, OGG, M4A |

> **補足**: `NDS2` と `GWOSC` はファイル形式ではなく取得経路ですが、GW 系の代表的な入口なので **A. GW標準** に含めています。表では `ネットワーク経由` として扱います。

<a id="io-formats-ja-basic"></a>

## `.read()` / `.write()` / `fetch()` の基本

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

# 拡張子から自動判別
tsd = TimeSeriesDict.read("path/to/data.mseed")

# format を明示
tsd = TimeSeriesDict.read("path/to/data.dat", format="miniseed")

# 書き出し
tsd.write("output.h5", format="hdf5")

# ネットワーク経由
ts = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- `.read()` / `.write()` は gwpy の I/O レジストリを利用します。
- `.xml` は用途が曖昧なので、**DTTXML では `format="dttxml"` を明示**してください。
- `NDS2` と `GWOSC` はファイルではないため、`.read()` ではなく `fetch()` / `fetch_open_data()` を使います。

## 対応クラスの早見表

「単一チャネルなのか、複数チャネルなのか」で入口を迷いやすい形式を先に整理すると、次のようになります。

| 形式 / 系統 | 単一チャネル | 複数チャネル | そのほかの対応 |
|---|---|---|---|
| **GWF / MiniSEED / SAC / GSE2 / K-NET / WIN / WIN32 / ATS / CSV / TXT / SDB / SQLite / SQLite3 / WAV / Audio** | `TimeSeries` | `TimeSeriesDict` | end-user 向け direct I/O の基本形 |
| **NetCDF4 / Zarr / GBD / TDMS** | `TimeSeries` | `TimeSeriesDict`, `TimeSeriesMatrix` | 行列系まで含む direct I/O |
| **HDF5** | `TimeSeries`, `FrequencySeries` など | `TimeSeriesDict` など | `Spectrogram`, `Histogram`, `EventTable`, `Field` まで含む主保存先 |
| **ndscope-hdf5** | - | `TimeSeriesDict` | ndscope 互換スキーマ |
| **DTTXML** | - | `TimeSeriesDict` | `products` 必須 |
| **NDS2 / GWOSC** | `TimeSeries` | - | `fetch()` / `fetch_open_data()` を使う |
| **ATS.MTH5** | `TimeSeries` | - | 一部対応の単一路 |
| **Pickle** | 主要クラス全般 | 主要クラス全般 | 信頼できるデータだけに使用 |
| **ROOT** | `EventTable` | - | EventTable の直 I/O のみ |

- **まず迷ったら** `TimeSeries` と `TimeSeriesDict` を基準に考えてください。
- **`TimeSeriesMatrix` が出てくるのは主に** `NetCDF4`, `Zarr`, `GBD`, `TDMS` です。
- **Series 以外もまとめて保持したい** 場合は、まず **HDF5** か **Pickle** を検討してください。

<a id="io-formats-ja-a"></a>

## A. GW標準

GW 系の標準保存・交換・取得経路です。
迷ったらまず **HDF5**、外部標準との互換が必要なら **GWF**、診断ツール出力なら **DTTXML** を見てください。

| 形式 / 経路 | 読 / 写 | 主な入口 | 向いている用途 | 備考 |
|---|:---:|---|---|---|
| **GWF** (`.gwf`) | ○ / ○ | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | LIGO/KAGRA の標準交換 | 標準形式。gwpy 経由 |
| **HDF5** (`.h5`, `.hdf5`) | ○ / ○ | 各クラスの `.read(..., format="hdf5")`, `.write(..., format="hdf5")` | 長期保存、メタデータ保持 | このページで唯一、Field 系の第一候補 |
| **ndscope-hdf5** (`.h5`, `.hdf5`) | ○ / ○ | `TimeSeriesDict.read(..., format="ndscope-hdf5")`, `.write(..., format="ndscope-hdf5")` | ndscope 互換 | `TimeSeriesDict` 限定 |
| **DTTXML** (`.xml`, `.xml.gz`) | ○ / × | `TimeSeriesDict.read(..., format="dttxml", products="...")` | DTT 出力、診断結果 | `products` 必須 |
| **NDS2** | ○ / × | `TimeSeries.fetch()` | 検出器データサーバ取得 | ネットワーク経由 |
| **GWOSC** | ○ / × | `TimeSeries.fetch_open_data()` | オープンデータ取得 | ネットワーク経由 |

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.h5", format="hdf5")
frame = TimeSeriesDict.read("data.gwf", format="gwf")
dtt = TimeSeriesDict.read("diag.xml", format="dttxml", products="TS")
open_data = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- **HDF5** は安全で構造化しやすく、GW 系で最も無難な保存先です。
- **DTTXML** は `products` によって出力型が変わります。複素 transfer function を扱う場合は、周波数系列側で `native=True` を優先してください。
- **NDS2 / GWOSC** はファイル形式ではないため、ページ中では A に置きつつ備考で `ネットワーク経由` と明示します。

<a id="io-formats-ja-b"></a>

## B. 地震・地球物理観測

地震観測・電磁気観測の既存フォーマットです。
既存資産をまず読めることが重要なグループで、MiniSEED を起点に考えると分かりやすいです。

| 形式 | 読 / 写 | 主な入口 | 向いている用途 | 備考 |
|---|:---:|---|---|---|
| **MiniSEED** (`.mseed`) | ○ / ○ | `TimeSeriesDict.read(..., format="miniseed")`, `.write(..., format="miniseed")` | 地震波形の標準交換 | `gap` でギャップ処理を指定 |
| **SAC** (`.sac`) | ○ / ○ | `TimeSeriesDict.read(..., format="sac")`, `.write(..., format="sac")` | 地震波形解析 | ObsPy 経由 |
| **GSE2** (`.gse2`) | ○ / ○ | `TimeSeriesDict.read(..., format="gse2")`, `.write(..., format="gse2")` | 地震波形交換 | ObsPy 経由 |
| **K-NET** (`.knet`) | ○ / × | `TimeSeriesDict.read(..., format="knet")` | K-NET 強震記録 | 読み込み専用 |
| **WIN / WIN32** (`.win`, `.cnt`) | ○ / × | `TimeSeriesDict.read(..., format="win")`, `TimeSeriesDict.read(..., format="win32")` | 国内 WIN データ | 改良版 parser、読み込み専用 |
| **ATS** (`.ats`) | ○ / × | `TimeSeries.read(..., format="ats")`, `TimeSeriesDict.read(..., format="ats")` | Metronix 観測データ | ネイティブ binary reader |
| **ATS.MTH5** (`format="ats.mth5"`) | ○ / × | `TimeSeries.read(..., format="ats.mth5")` | MTH5 経由の単一路 | 一部対応 |
| **MTH5 standalone** (`.h5`) | 対応中 | 専用 `format="mth5"` は未整備 | 今後の汎用 MTH5 direct I/O | **現時点では public direct-I/O 対応ではありません**。使える direct path は `ats.mth5` のみ |

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.mseed", format="miniseed", gap="pad")
win = TimeSeriesDict.read("data.cnt", format="win32")
ats = TimeSeries.read("data.atss", format="ats.mth5")
```

- **MiniSEED** はギャップがある場合、既定では `NaN` パディングされます。`gap="raise"` で失敗させることもできます。
- **K-NET**, **WIN / WIN32** は表のとおり読み込み専用です。
- **ATS.MTH5** は利用経路が限定される current direct path です。
- **MTH5 standalone** は設計・公開整理中です。**「MTH5 は対応済み」ではなく、「`ats.mth5` のみ一部対応」** と読んでください。

<a id="io-formats-ja-c"></a>

## C. 汎用・解析用

解析ノート、外部解析、交換に向いた形式です。
このグループでは「保存形式として選ぶ話」と「他ライブラリへ変換する話」を混ぜないのが重要です。

| 形式 | 読 / 写 | 主な入口 | 向いている用途 | 備考 |
|---|:---:|---|---|---|
| **CSV / TXT** (`.csv`, `.txt`) | ○ / ○ | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | 軽量な交換、目視確認 | ディレクトリ一括読み込みにも対応 |
| **NetCDF4** (`.nc`) | ○ / ○ | `TimeSeries.read(..., format="netcdf4")`, `TimeSeriesDict.read(..., format="netcdf4")`, `.write(..., format="netcdf4")` | 時系列系の科学データ保存 | direct I/O は TimeSeries 系中心 |
| **Zarr** (`.zarr`) | ○ / ○ | `TimeSeries.read(..., format="zarr")`, `TimeSeriesDict.read(..., format="zarr")`, `.write(..., format="zarr")` | chunked 保存、並列処理 | direct I/O は TimeSeries 系中心 |
| **Pickle** (`.pkl`) | ○ / ○ | 各クラスの `.read()`, `.write()` | Python オブジェクト丸ごと保存 | 信頼できるデータだけに使用 |
| **ROOT** (`.root`) | ○ / ○ | `EventTable.read(..., format="root")`, `EventTable.write(..., format="root")` | EventTable の入出力 | 直 I/O は EventTable のみ |

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.table import EventTable

ascii_data = TimeSeriesDict.read("data.csv")
chunked = TimeSeriesDict.read("data.zarr", format="zarr")
events = EventTable.read("events.root", format="root")
```

- **CSV / TXT** は素朴ですが、共有や確認には依然として有用です。
- **NetCDF4 / Zarr** はこのページでは **TimeSeries 系の direct I/O** としてだけ扱います。Field と xarray の橋渡しは interop 側を見てください。
- **Zarr** では `sample_rate` 欠落時の既定挙動に未解決の API 課題があります。**このページで解決済みとはみなしていません**。
- **ROOT** の object-level 変換はここでは扱いません。I/O ガイドでは EventTable の直 I/O のみ扱います。

<a id="io-formats-ja-d"></a>

## D. 計測機器・ロガー

ロガーや機材固有の時系列です。
時刻の扱い、単位、音声の `t0` など、フォーマットごとの注意点が比較的重要です。

| 形式 | 読 / 写 | 主な入口 | 向いている用途 | 備考 |
|---|:---:|---|---|---|
| **GBD** (`.gbd`) | ○ / × | `TimeSeries.read(..., format="gbd")`, `TimeSeriesDict.read(..., format="gbd")` | GRAPHTEC ロガー | `timezone` 必須 |
| **TDMS** (`.tdms`) | ○ / × | `TimeSeries.read(..., format="tdms")`, `TimeSeriesDict.read(..., format="tdms")` | National Instruments | 読み込み専用 |
| **SDB / SQLite / SQLite3** (`.sdb`, `.sqlite`, `.sqlite3`) | ○ / × | `TimeSeries.read(..., format="sdb" / "sqlite" / "sqlite3")`, `TimeSeriesDict.read(...)` | WeeWX 等の蓄積データ | `sqlite` / `sqlite3` も同系統 |
| **WAV** (`.wav`) | ○ / ○ | `TimeSeries.read(..., format="wav")`, `TimeSeriesDict.read(..., format="wav")`, `.write(..., format="wav")` | 非圧縮音声 | 絶対時刻は保持しない |
| **MP3 / FLAC / OGG / M4A** | ○ / ○ | `TimeSeries.read(..., format="mp3" / "flac" / "ogg" / "m4a")`, `.write(...)` | 圧縮音声 | `pydub`、一部形式は `ffmpeg` が必要 |

```python
from gwexpy.timeseries.collections import TimeSeriesDict

logger = TimeSeriesDict.read("data.gbd", timezone="Asia/Tokyo")
weather = TimeSeriesDict.read("archive.sqlite3", format="sqlite3")
audio = TimeSeriesDict.read("sound.flac", format="flac")
```

- **GBD** は `timezone` を省略できません。
- **SDB / SQLite / SQLite3** は同系統の reader です。公開ページでは 3 つとも明示して混乱を避けます。
- **WAV / Audio** は絶対時刻を持たないため、読み込み時は便宜上 `t0=0.0` として扱います。「絶対時刻がある」という意味ではありません。

<a id="io-formats-ja-dev"></a>

## 開発者向け補足

通常の利用ではこの節は読み飛ばして構いません。
未掲載実装や placeholder をまとめて確認したい場合だけ見てください。

### 設計上は管理するが、公開ページでは主表示しないもの

| 形式 | 現状 | 補足 |
|---|---|---|
| `ndscope-hdf5` | 実装済み（未掲載） | `TimeSeriesDict` 限定の HDF5 スキーマ |
| `SQLite`, `SQLite3` | 実装済み（未掲載） | `SDB` と同系統の alias |
| `ATS.MTH5` | 実装済み（一部対応） | MTH5 経由の current public direct path |
| `MTH5 standalone` | 対応中 | 専用 `format="mth5"` は未整備。public direct-I/O としては未公開 |

### 未実装フォーマット (Stubs)

以下は placeholder reader があるだけで、一般ユーザー向けの実用形式ではありません。`.read()` は未実装として失敗します。

#### 時系列 stub

| 形式 | 状態 |
|---|---|
| `orf` | 対応予定 |
| `mem` | 対応予定 |
| `wvf` | 対応予定 |
| `wdf` | 対応予定 |
| `taffmat` | 対応予定 |
| `lsf` | 対応予定 |
| `li` | 対応予定 |

#### 周波数系列 stub

| 形式 | 状態 |
|---|---|
| `win` | 対応予定 |
| `win32` | 対応予定 |
| `sdb` | 対応予定 |
| `orf` | 対応予定 |
| `mem` | 対応予定 |
| `wvf` | 対応予定 |
| `wdf` | 対応予定 |
| `taffmat` | 対応予定 |
| `lsf` | 対応予定 |
| `li` | 対応予定 |

## 関連ページ

- [他ライブラリ連携チュートリアル](tutorials/intro_interop)
- [Interop API リファレンス](../reference/api/interop)
- [インストールガイド](installation)

## ページ末尾ナビゲーション

- <a href="#io-formats-ja-quick">クイック判定表へ戻る</a>
- <a href="#io-formats-ja-basic">基本的な使い方へ戻る</a>
- <a href="#io-formats-ja-top">トップへ戻る</a>
