---
myst:
  html_meta:
    description: "GWexpy の direct I/O ガイドです。対応フォーマット、読み書きの可否、fetch 系の入口、format= を明示すべき場面を整理します。"
---

<a id="io-formats-ja-top"></a>

# ファイル I/O 対応フォーマットガイド

> **ページ種別:** ガイド

gwexpy の利用者向け I/O ガイドです。
このページでは、ユーザーが直接使う `.read()` / `.write()` / `fetch()` 系 API による読み書き・取得だけを扱います。

`to_*()` / `from_*()` による変換や、xarray / ROOT object / Zarr array との橋渡しは、このページでは扱いません。質問が「このオブジェクトを別ライブラリや別コンテナへどう変換するか」であれば、それは interop 側の話です。必要な場合は [他ライブラリ連携チュートリアル](tutorials/intro_interop.ipynb) や [Interop API リファレンス](../reference/api/interop) を参照してください。

## このページでわかること

| 項目 | 内容 |
| --- | --- |
| **対象読者** | `gwexpy` オブジェクトの直接的な読み書きやネットワーク取得の入口を選びたい利用者 |
| **前提** | `TimeSeries` / `TimeSeriesDict` の基本、拡張子による形式判定、direct I/O と interop の違い |
| **こんなときに読む** | 対応形式を選びたい、`format=` を明示すべき場面を知りたい、読み込み専用かどうかを確認したい |
| **検索キーワード** | file I/O, direct I/O, `read`, `write`, `fetch`, HDF5, GWF, MiniSEED, Zarr, NDS2, GWOSC |

**検索ヒント:** file I/O, direct I/O, `read`, `write`, `fetch`, HDF5, GWF, MiniSEED, Zarr, NDS2, GWOSC

:::{warning}
**セキュリティ警告: Pickle 形式の取り扱い**

**Pickle** (:term:`Pickle`) は便利ですが、信頼できないソースから受け取った Pickle ファイルを読み込むことは危険です。悪意のあるファイルにより、実行環境で任意のコードが実行される可能性があります。

共同研究者との共有や長期保存には、**HDF5**、**GWF**、**Zarr** のような構造化された形式を優先してください。
:::

## まず最初に: 判断ルール

- **まず保存形式を選ぶ**なら、GW 系データは **HDF5**、観測網の既存資産は **MiniSEED / SAC / WIN / ATS**、汎用交換は **CSV / NetCDF4 / Zarr**、ロガー固有データは **GBD / TDMS / SDB / WAV / Audio** を起点に考えると整理しやすいです。**MTH5** については、現時点の public direct-I/O は **`ats.mth5` の単一路だけ** で、汎用の standalone **`format="mth5"`** はまだ公開していません。
- **自動判別でよい**のは、拡張子から reader が一意に決まる場合です。
- 汎用 **HDF5** は **`format="hdf5"` を明示**してください。`.h5` / `.hdf5` は複数の HDF5 系経路で共有されており、クラスによって自動判別の挙動が揃っていません。
- **`format=` を明示する**のは、`.xml` のように経路が複数ありうる場合、独自拡張子を使っている場合、または実験データで自動判別が不安な場合です。
- **`timezone` を必ず指定する**のは、ファイル内に UTC/GPS がなくローカル時刻だけを持つ形式です。現時点でユーザーが明示必須なのは **GBD** です。
- **Read only / Write only に注意する**: 表の `○ / ×` は「読めるが書けない」「書けるが読めない」を意味します。
- Series 以外の direct I/O では、まず **HDF5** の `Spectrogram` / `Histogram` / `EventTable` を基準に考えてください。Field 系の direct `.read()` / `.write()` はまだ監査中で、このページでは安定契約として公開していません。

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

| グループ | まず見る場面 | 最初の形式 | このページで扱う形式 |
|---|---|---|---|
| **A. GW標準** | GW 系の標準保存、共有、取得経路を使いたい | **HDF5** | GWF, HDF5, hdf.ndscope, xml.diaggui, NDS2, GWOSC |
| **B. 地震・地球物理観測** | 既存の地震・電磁気観測フォーマットを読む | **mseed** | mseed, SAC, GSE2, K-NET, WIN / WIN32, ATS, ATS.MTH5（MTH5 standalone は状況注記のみ） |
| **C. 汎用・解析用** | 汎用保存、外部解析、交換をしたい | **CSV / TXT** または **Zarr** | CSV / TXT, NetCDF4, Zarr, ROOT |
| **D. 計測機器・ロガー** | ロガーや機材固有の時系列を読む | **GBD** または **TDMS** | GBD, TDMS, SDB / SQLite / SQLite3, WAV, MP3, FLAC, OGG, M4A |

> **補足**: `NDS2` と `GWOSC` はファイル形式ではなく取得経路ですが、GW 系の代表的な入口なので **A. GW標準** に含めています。表では `ネットワーク経由` として扱います。

<a id="io-formats-ja-basic"></a>

## `.read()` / `.write()` / `fetch()` の基本

- 目的: 形式ごとの細かい違いを見る前に、direct I/O の基本入口を確認する
- 入力: ファイルパス、必要に応じた `format=`、またはネットワーク取得の問い合わせ
- 出力: `TimeSeries`、`TimeSeriesDict` などの direct I/O の返り値

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

# 拡張子から自動判別
tsd = TimeSeriesDict.read("path/to/data.mseed")

# format を明示
tsd = TimeSeriesDict.read("path/to/data.dat", format="mseed")

# 書き出し
tsd.write("output.h5", format="hdf5")

# ネットワーク経由
ts = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- `.read()` / `.write()` は gwpy の I/O レジストリを利用します。
- `.xml` は用途が曖昧なので、**DiagGUI XML では `format="xml.diaggui"` を明示**してください。
- `NDS2` と `GWOSC` はファイルではないため、`.read()` ではなく `fetch()` / `fetch_open_data()` を使います。

(io-formats-ja-supported-classes)=
## 対応クラスの早見表

「単一チャネルなのか、複数チャネルなのか」で入口を迷いやすい形式を先に整理すると、次のようになります。

| 形式 / 系統 | 単一 | 複数 | そのほかの対応 |
|---|---|---|---|
| **GWF / mseed / SAC / GSE2 / K-NET / WIN / WIN32 / ATS / SDB / SQLite / SQLite3 / WAV / Audio** | `TimeSeries` | `TimeSeriesDict` | end-user 向け direct I/O の基本形 |
| **CSV** | `TimeSeries` | `TimeSeriesDict` | `TimeSeriesDict` は manifest 付き collection directory にも対応 |
| **TXT** | `TimeSeries` | `TimeSeriesDict` | 複数チャネルの direct I/O は collection directory を使う |
| **nc / Zarr / GBD / TDMS** | `TimeSeries` | `TimeSeriesDict`, `TimeSeriesMatrix` | 行列系まで含む direct I/O |
| **HDF5** | `TimeSeries`, `FrequencySeries` など | `TimeSeriesDict` など | `Spectrogram`, `Histogram`, `EventTable` を含む主保存先 |
| **hdf.ndscope** | - | `TimeSeriesDict` | ndscope 互換スキーマ。alias: `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` |
| **xml.diaggui** | - | `TimeSeriesDict` | `products` 必須。旧 alias: `dttxml` |
| **NDS2 / GWOSC** | `TimeSeries` | - | `fetch()` / `fetch_open_data()` を使う |
| **ATS.MTH5** | `TimeSeries` | - | 一部対応の単一路 |
| **ROOT** | `EventTable` | - | EventTable の直 I/O のみ |

- **まず迷ったら** `TimeSeries` と `TimeSeriesDict` を基準に考えてください。
- **`TimeSeriesMatrix` が出てくるのは主に** `NetCDF4`, `Zarr`, `GBD`, `TDMS` です。
- **Series 以外もまとめて保持したい** 場合は、まず **HDF5** を検討してください。

## オプション依存関係の対応表

多くの direct I/O 経路は GWexpy の基本インストールで利用できます。次の形式は optional package または optional metadata helper に依存します。

| 形式 / 系統 | オプション依存関係 | GWexpy extra | 未導入時の挙動 |
|---|---|---|---|
| **WAV metadata** | `tinytag` | `audio` | `.read(..., extract_metadata=True)` は警告を出し、metadata を省略します。[インストールガイド](installation.md) の `audio` または `all` extra で追加してください。通常の WAV 読み書きは利用できます。 |
| **MP3 / FLAC / OGG / M4A** | `pydub`, `tinytag` | `audio` | 音声の読み書きは `ImportError` を送出します。一部 codec は外部の `ffmpeg` / `libav` も必要です。 |
| **TDMS** | `nptdms` | `io` | reader は必要な `io` extra の案内付きで `ImportError` を送出します。 |
| **mseed / SAC / GSE2 / K-NET** | `obspy` | `seismic` | 登録済みの reader / writer は必要な `seismic` extra の案内付きで `ImportError` を送出します。 |
| **WIN / WIN32** | `obspy` | `seismic` | 条件付き登録です。ObsPy がない環境では `win` / `win32` の registry entry 自体が存在しない場合があります。 |
| **ATS.MTH5** | `mth5` | `seismic` | reader は必要な `seismic` extra の案内付きで `ImportError` を送出します。 |
| **nc / NetCDF4** | `xarray`, `netCDF4` | `netcdf4` | reader / writer は必要な `netcdf4` extra の案内付きで `ImportError` を送出します。 |
| **Zarr** | `zarr` | `zarr` | reader / writer は必要な `zarr` extra の案内付きで `ImportError` を送出します。 |

<a id="io-formats-ja-a"></a>

## A. GW標準

GW 系の標準保存・交換・取得経路です。
迷ったらまず **HDF5**、外部標準との互換が必要なら **GWF**、診断ツール出力なら **DTTXML** を見てください。

| 形式 / 経路 | 読 / 写 | 主な入口 | 用途 | 備考 |
|---|:---:|---|---|---|
| **GWF** (`.gwf`) | ○ / ○ | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | LIGO/KAGRA の標準交換 | 標準形式。gwpy 経由 |
| **HDF5** (`.h5`, `.hdf5`) | ○ / ○ | 各クラスの `.read(..., format="hdf5")`, `.write(..., format="hdf5")` | 長期保存、メタデータ保持 | `format="hdf5"` の明示を推奨 |
| **hdf.ndscope** (`.h5`, `.hdf5`) | ○ / ○ | `TimeSeriesDict.read(..., format="hdf.ndscope")`, `.write(..., format="hdf.ndscope")` | ndscope 互換 | `TimeSeriesDict` 限定。旧 alias: `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` |
| **xml.diaggui** (`.xml`, `.xml.gz`) | ○ / × | `TimeSeriesDict.read(..., format="xml.diaggui", products="...")` | DiagGUI / DTT 出力 | `products` 必須。旧 alias: `dttxml` |
| **NDS2** | ○ / × | `TimeSeries.fetch()` | 検出器データサーバ取得 | ネットワーク経由 |
| **GWOSC** | ○ / × | `TimeSeries.fetch_open_data()` | オープンデータ取得 | ネットワーク経由 |

- 目的: GW 系で代表的な direct I/O とネットワーク取得の入口を比較する
- 入力: HDF5, GWF, DTTXML, または検出器 / オープンデータ取得の条件
- 出力: `TimeSeries`, `TimeSeriesDict`, または取得した open data

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.h5", format="hdf5")
frame = TimeSeriesDict.read("data.gwf", format="gwf")
merged = TimeSeriesDict.read(["part0.gwf", "part1.gwf"], "H1:STRAIN", pad=float("nan"))
dtt = TimeSeriesDict.read("diag.xml", format="xml.diaggui", products="TS")
open_data = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- **HDF5** は安全で構造化しやすく、GW 系で最も無難な保存先です。
- **GWF** は `TimeSeries` と `TimeSeriesDict` で `.gwf` ファイルの list / tuple 入力に対応します。ファイルは時刻 span 順で結合されます。連続 span はそのまま結合し、ギャップは既定で失敗します。`pad=<値>` または `gap="pad"` で埋められ、`gap="ignore"` では埋めずに連結します。オーバーラップする span は既定または `gap="raise"` で失敗しますが、`gap="ignore"` では span 順に連結し、オーバーラップの連結を許可します。`start` / `end` が実データの外側に伸びる場合も、既定の `gap="raise"` では失敗します。外側区間を埋めるには `pad=<値>` または `gap="pad"` を使ってください。`gap="ignore"` は内部ギャップも外側の `start` / `end` 区間も padding しません。複数ファイル読み込みで channel 名を指定しない場合、自動検出は先頭ファイルを使い、残りのファイルにも互換 channel がある前提です。
- **DTTXML** は `products` によって出力型が変わります。public direct read は `TimeSeriesDict.read(..., format="xml.diaggui", products=...)` に揃えます。
- 周波数領域の DTTXML direct shim と registry adapter は implementation-only で、public direct-I/O contract には含めません。複素 transfer function を扱う高度な内部利用では `native=True` を優先できます。
- **NDS2 / GWOSC** はファイル形式ではないため、ページ中では A に置きつつ備考で `ネットワーク経由` と明示します。

<a id="io-formats-ja-b"></a>

## B. 地震・地球物理観測

地震観測・電磁気観測の既存フォーマットです。
既存資産をまず読めることが重要なグループで、MiniSEED を起点に考えると分かりやすいです。

| 形式 | 読 / 写 | 主な入口 | 用途 | 備考 |
|---|:---:|---|---|---|
| **mseed** (`.mseed`) | ○ / ○ | `TimeSeriesDict.read(..., format="mseed")`, `.write(..., format="mseed")` | 地震波形の標準交換 | `gap` でギャップ処理を指定。旧 alias: `miniseed` |
| **SAC** (`.sac`) | ○ / ○ | `TimeSeriesDict.read(..., format="sac")`, `.write(..., format="sac")` | 地震波形解析 | ObsPy 経由 |
| **GSE2** (`.gse2`) | ○ / ○ | `TimeSeriesDict.read(..., format="gse2")`, `.write(..., format="gse2")` | 地震波形交換 | ObsPy 経由 |
| **K-NET** (`.knet`) | ○ / × | `TimeSeriesDict.read(..., format="knet")` | K-NET 強震記録 | 読み込み専用 |
| **WIN / WIN32** (`.win`, `.cnt`) | ○ / × | `TimeSeriesDict.read(..., format="win")`, `TimeSeriesDict.read(..., format="win32")` | 国内 WIN データ | 改良版 parser、読み込み専用 |
| **ATS** (`.ats`) | ○ / × | `TimeSeries.read(..., format="ats")`, `TimeSeriesDict.read(..., format="ats")` | Metronix 観測データ | ネイティブ binary reader |
| **ATS.MTH5** (`format="ats.mth5"`) | ○ / × | `TimeSeries.read(..., format="ats.mth5")` | MTH5 経由の単一路 | 一部対応 |
| **MTH5 standalone** (`.h5`) | 対応中 | 専用 `format="mth5"` は未整備 | 今後の汎用 MTH5 direct I/O | **現時点では public direct-I/O 対応ではありません**。使える direct path は `ats.mth5` のみ |

- 目的: 地震・地球物理系 reader の違いを、MTH5 を過大評価せずに比較する
- 入力: MiniSEED, WIN/WIN32, または限定公開中の `ats.mth5` 経路
- 出力: reader に応じた `TimeSeries` または `TimeSeriesDict`

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.mseed", format="mseed", gap="pad")
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

| 形式 | 読 / 写 | 主な入口 | 用途 | 備考 |
|---|:---:|---|---|---|
| **CSV** (`.csv`) | ○ / ○ | `TimeSeries.read("data.csv")`, `TimeSeriesDict.read("data.csv")`, `TimeSeriesDict.write(..., format="csv")` | 軽量な交換、目視確認 | `.csv` は自動判定されます。単純な CSV 交換は metadata-light です |
| **TXT** (`.txt`) | ○ / ○ | `TimeSeries.read(..., format="txt")`, `TimeSeriesDict.read(dir, format="txt")`, `TimeSeriesDict.write(dir, format="txt")` | プレーンテキスト交換 | 複数チャネルの direct I/O は collection directory を使う |
| **nc** (`.nc`) | ○ / ○ | `TimeSeries.read(..., format="nc")`, `TimeSeriesDict.read(..., format="nc")`, `TimeSeriesMatrix.read(..., format="nc")`, `.write(..., format="nc")` | 時系列系の科学データ保存 | direct I/O は TimeSeries 系中心。旧 format alias: `netcdf4` |
| **Zarr** (`.zarr`) | ○ / ○ | `TimeSeries.read(..., format="zarr")`, `TimeSeriesDict.read(..., format="zarr")`, `TimeSeriesMatrix.read(..., format="zarr")`, `.write(..., format="zarr")` | chunked 保存、並列処理 | direct I/O は TimeSeries 系中心 |
| **ROOT** (`.root`) | ○ / ○ | `EventTable.read("events.root")`, `EventTable.write(..., format="root")` | EventTable の入出力 | `.root` は自動判定されます。直 I/O は EventTable のみ |

- 目的: 汎用交換向けの direct I/O を、interop 専用の橋渡しと混同せず整理する
- 入力: CSV, Zarr, ROOT などの汎用形式
- 出力: `TimeSeriesDict`, `TimeSeriesMatrix`, または `EventTable`

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.table import EventTable

ascii_data = TimeSeriesDict.read("data.csv")
chunked = TimeSeriesDict.read("data.zarr", format="zarr")
events = EventTable.read("events.root")
```

- **CSV** は素朴ですが、共有や確認には依然として有用です。単純な CSV ファイルは metadata-light と考えてください。`name`、`channel`、`unit` まで保持したい場合は HDF5、GWF、Zarr、NetCDF、または manifest 付き collection directory を使います。
- **TXT** の direct I/O はより限定的で、単一 series は `format="txt"` 明示、複数チャネルは collection directory 前提です。
- **Pickle** の可搬性メモは各クラスの reference に残していますが、このページでは Pickle を public direct `.read()` / `.write()` 形式としては扱いません。
- **NetCDF4 / Zarr** はこのページでは **TimeSeries 系の direct I/O** としてだけ扱います。Field と xarray の橋渡しは interop 側を見てください。NetCDF の `netcdf4` は `nc` の旧 format token alias であり、`.netcdf4` は公開された自動判定 extension alias ではありません。
- **Zarr** の direct I/O では、配列ごとの timing metadata を明示的に要求します。`sample_rate` を優先し、`dt` は fallback として受け付けます。どちらも無い場合は `ValueError` を送出し、legacy store を意図的に救済したい場合だけ `sample_rate_override=...` または `dt_override=...` を指定してください。
- **ROOT** の object-level 変換はここでは扱いません。I/O ガイドでは EventTable の直 I/O のみ扱います。

<a id="io-formats-ja-d"></a>

## D. 計測機器・ロガー

ロガーや機材固有の時系列です。
時刻の扱い、単位、音声の `t0` など、フォーマットごとの注意点が比較的重要です。

| 形式 | 読 / 写 | 主な入口 | 用途 | 備考 |
|---|:---:|---|---|---|
| **GBD** (`.gbd`) | ○ / × | `TimeSeries.read(..., format="gbd", timezone=...)`, `TimeSeriesDict.read(..., format="gbd", timezone=...)`, `TimeSeriesMatrix.read(..., format="gbd", timezone=...)` | GRAPHTEC ロガー | public read では `timezone` 必須 |
| **TDMS** (`.tdms`) | ○ / × | `TimeSeries.read(..., format="tdms")`, `TimeSeriesDict.read(..., format="tdms")`, `TimeSeriesMatrix.read(..., format="tdms")` | National Instruments | 読み込み専用。`nptdms` が必要 |
| **SDB / SQLite / SQLite3** (`.sdb`, `.sqlite`, `.sqlite3`) | ○ / × | `TimeSeries.read(..., format="sdb" / "sqlite" / "sqlite3")`, `TimeSeriesDict.read(..., format="sdb" / "sqlite" / "sqlite3")` | WeeWX 等の蓄積データ | 同系統 reader。public direct I/O は読み込み専用 |
| **WAV** (`.wav`) | ○ / ○ | `TimeSeries.read(..., format="wav")`, `TimeSeriesDict.read(..., format="wav")`, `TimeSeries.write(..., format="wav")` | 非圧縮音声 | public write は単一路のみ。絶対時刻は保持しない |
| **MP3 / FLAC / OGG / M4A** | ○ / ○ | `TimeSeries.read(..., format="mp3" / "flac" / "ogg" / "m4a")`, `TimeSeriesDict.read(..., format=...)`, `.write(...)` | 圧縮音声 | `pydub`、一部形式は `ffmpeg` が必要 |

- 目的: ロガー系・音声系 format で注意すべき条件をまとめる
- 入力: ロガーデータ、SQLite 系アーカイブ、音声ファイル
- 出力: `TimeSeries`, `TimeSeriesDict`, または `TimeSeriesMatrix`

```python
from gwexpy.timeseries.collections import TimeSeriesDict

logger = TimeSeriesDict.read("data.gbd", timezone="Asia/Tokyo")
weather = TimeSeriesDict.read("archive.sqlite3", format="sqlite3")
audio = TimeSeriesDict.read("sound.flac", format="flac")
```

- **GBD** は `timezone` を省略できません。
- **TDMS** は optional dependency の `nptdms` が必要です。
- **MP3 / FLAC / OGG / M4A** は optional dependency の `pydub` が必要で、MP3/M4A は `ffmpeg` も必要になることが多いです。
- **SDB / SQLite / SQLite3** は同系統の reader です。公開ページでは 3 つとも明示して混乱を避けます。
- **WAV / 圧縮音声形式** は絶対時刻を持たないため、読み込み時は便宜上 `t0=0.0` として扱います。「絶対時刻がある」という意味ではありません。

<a id="io-formats-ja-dev"></a>

## 開発者向け補足

通常の利用ではこの節は読み飛ばして構いません。
未掲載実装や placeholder をまとめて確認したい場合だけ見てください。

### 設計上は管理するが、公開ページでは主表示しないもの

| 形式 | 状態 | 補足 |
|---|---|---|
| `hdf.ndscope` | 実装済み（未掲載） | `TimeSeriesDict` 限定の HDF5 スキーマ。旧 alias は `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` |
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

- [他ライブラリ連携チュートリアル](tutorials/intro_interop.ipynb)
- [Interop API リファレンス](../reference/api/interop)
- [検証と品質の見方](verification_and_quality.md)
- [インストールガイド](installation)

## 次に読む

- [Interop / 変換ガイド](interop) で `to_*()` / `from_*()` による橋渡しを確認する
- [GPS 時刻ユーティリティ](time_utilities) で時刻文字列やタイムゾーンの扱いを確認する
- [インストールガイド](installation) で形式ごとの依存関係を確認する

## ページ末尾ナビゲーション

- <a href="#io-formats-ja-quick">クイック判定表へ戻る</a>
- <a href="#io-formats-ja-basic">基本的な使い方へ戻る</a>
- <a href="#io-formats-ja-top">トップへ戻る</a>
