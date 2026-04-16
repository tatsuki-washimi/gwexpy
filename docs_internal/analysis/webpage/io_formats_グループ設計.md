# `io_formats.html` ドメイン別グループ設計（I/O専用版）

作成日: 2026-04-16  
最終更新: 2026-04-17（I/O / interop 分離）

---

## この設計書の責務

この文書は、`user_guide/io_formats.html` を再編するための **end-user I/O 専用設計書** である。
ここでいう「対応」は、ユーザーがドキュメント上で直接案内する `.read()` / `.write()` の可否だけを指す。

以下は **対象外** とし、別紙 [interop / 変換対応設計](./interop_変換対応設計.md) で扱う。

- `to_*` / `from_*` によるオブジェクト変換
- 外部ライブラリのオブジェクトとの相互変換
- ROOT object, xarray object, Zarr array のような橋渡し API
- 対称でない file-bridge helper

> [!IMPORTANT]
> 旧版で混在していた `interop` ベースの対応表現は、この文書から除外した。
> `io_formats` 本文は今後、**「何をどう read/write できるか」だけ** を説明する。

---

## 判定基準

「対応」とみなす条件は次のとおり。

1. エンドユーザーが `Class.read(...)` または `obj.write(...)` として直接使える
2. `gwpy` 継承 I/O、`gwexpy` の I/O レジストリ、または明示的な user-facing writer 経路で裏づけられる
3. `gwexpy.interop` の `to_*` / `from_*` だけで成立するものは含めない

従って、`ROOT`, `NetCDF4`, `Zarr` の Field 系対応や ROOT object 変換は、この文書では「I/O 対応」ではなく `interop` 側へ分離する。

---

## グループ設計の確定内容

`io_formats.html` は、**4グループ** で再編する。

| グループ | 主な形式 | ページ上の役割 | 主対象 |
|---|---|---|---|
| **A: GW標準** | GWF, HDF5, ndscope-hdf5, DTTXML, NDS2, GWOSC | 重力波解析の標準保存・交換・取得経路 | TimeSeries系、HDF5 は Field を含む |
| **B: 地震・地球物理観測** | MiniSEED, SAC, GSE2, K-NET, WIN / WIN32, ATS, ATS.MTH5, MTH5 standalone | 地震・電磁気観測の現場データ | TimeSeries, TimeSeriesDict |
| **C: 汎用・解析用** | CSV / TXT, NetCDF4, Zarr, Pickle, ROOT | 汎用保存、解析連携、交換 | TimeSeries系中心。Pickle は全クラス、ROOT は EventTable 直I/O に限定 |
| **D: 計測機器・ロガー** | GBD, TDMS, SDB / SQLite / SQLite3, WAV, MP3, FLAC, OGG, M4A | 機材固有またはロガー由来の時系列 | TimeSeries, TimeSeriesDict |
> [!NOTE]
> `NDS2` と `GWOSC` は GW 系の代表的な取得経路として **A: GW標準** に含める。ただしファイル形式ではないため、備考欄で `ネットワーク経由` と明記する。
> `ROOT` はこの文書では **EventTable の直I/O** だけを含める。Series / Histogram / Spectrogram の ROOT object 変換は interop 側へ移す。

---

## 公開ページの主表示対象（再編後）

この節は、`io_formats.html` 本文の主表示対象を示す。
エンドユーザーが最初に見るべき一覧であり、全管理対象のうち **すでに使える direct I/O** を中心に並べる。

| 形式 / 経路 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| **GWF** (`.gwf`) | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | 掲載済み | Frame 形式。Series 優先 |
| **HDF5** (`.h5`, `.hdf5`) | 各クラスの `.read(..., format="hdf5")`, `.write(..., format="hdf5")` | 掲載済み | TimeSeries 系、FrequencySeries 系、Spectrogram 系、Histogram、EventTable、Field |
| **ndscope-hdf5** (`.h5`, `.hdf5`) | `TimeSeriesDict.read(..., format="ndscope-hdf5")`, `.write(..., format="ndscope-hdf5")` | 実装済み（未掲載） | ndscope 互換の HDF5 スキーマ。Dict 限定 |
| **DTTXML** (`.xml`) | `TimeSeriesDict.read(..., format="dttxml", products="...")` | 掲載済み | `products` 指定前提。Write なし |
| **NDS2** | `TimeSeries.fetch()` | 掲載済み | ネットワーク経由 |
| **GWOSC** | `TimeSeries.fetch_open_data()` | 掲載済み | ネットワーク経由 |
| **MiniSEED** (`.mseed`) | `TimeSeriesDict.read(..., format="miniseed")`, `.write(..., format="miniseed")` | 掲載済み | ObsPy 経由 |
| **SAC** (`.sac`) | `TimeSeriesDict.read(..., format="sac")`, `.write(..., format="sac")` | 掲載済み | ObsPy 経由 |
| **GSE2** (`.gse2`) | `TimeSeriesDict.read(..., format="gse2")`, `.write(..., format="gse2")` | 掲載済み | ObsPy 経由 |
| **K-NET** (`.knet`) | `TimeSeriesDict.read(..., format="knet")` | 掲載済み | Read only |
| **WIN / WIN32** (`.win`, `.cnt`) | `TimeSeriesDict.read(..., format="win")`, `TimeSeriesDict.read(..., format="win32")` | 掲載済み | Read only。国内 WIN データ向け改良 reader |
| **ATS** (`.ats`) | `TimeSeries.read(..., format="ats")`, `TimeSeriesDict.read(..., format="ats")` | 掲載済み | ネイティブ binary reader |
| **ATS.MTH5** (`format="ats.mth5"`) | `TimeSeries.read(..., format="ats.mth5")` | 実装済み（一部対応） | MTH5 経由の単一路。`.atss` と命名規約依存 |
| **MTH5 standalone** (`.h5`) | 専用 `read(..., format="mth5")` / `write(..., format="mth5")` は未整備 | 対応中 | 現時点の direct path は `ats.mth5` のみ |
| **CSV** (`.csv`) | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | 掲載済み | enhanced CSV reader を含む |
| **TXT** (`.txt`) | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | 掲載済み | enhanced ASCII reader を含む |
| **NetCDF4** (`.nc`) | `TimeSeries.read(..., format="netcdf4")`, `TimeSeriesDict.read(..., format="netcdf4")`, `TimeSeriesMatrix.read(..., format="netcdf4")`, `.write(..., format="netcdf4")` | 掲載済み | TimeSeries 系の direct I/O |
| **Zarr** (`.zarr`) | `TimeSeries.read(..., format="zarr")`, `TimeSeriesDict.read(..., format="zarr")`, `TimeSeriesMatrix.read(..., format="zarr")`, `.write(..., format="zarr")` | 掲載済み | TimeSeries 系の direct I/O |
| **Pickle** (`.pkl`) | 各クラスの `.read()`, `.write()` | 掲載済み | セキュリティ警告が必要 |
| **ROOT** (`.root`) | `EventTable.read(..., format="root")`, `EventTable.write(..., format="root")` | 掲載済み | EventTable 直I/O のみ |
| **GBD** (`.gbd`) | `TimeSeries.read(..., format="gbd")`, `TimeSeriesDict.read(..., format="gbd")`, `TimeSeriesMatrix.read(..., format="gbd")` | 掲載済み | Read only。`timezone` 必須 |
| **TDMS** (`.tdms`) | `TimeSeries.read(..., format="tdms")`, `TimeSeriesDict.read(..., format="tdms")`, `TimeSeriesMatrix.read(..., format="tdms")` | 掲載済み | Read only |
| **SDB** (`.sdb`) | `TimeSeries.read(..., format="sdb")`, `TimeSeriesDict.read(..., format="sdb")` | 掲載済み | Read only |
| **SQLite** (`.sqlite`) | `TimeSeries.read(..., format="sqlite")`, `TimeSeriesDict.read(..., format="sqlite")` | 実装済み（未掲載） | SDB と同じ reader |
| **SQLite3** (`.sqlite3`) | `TimeSeries.read(..., format="sqlite3")`, `TimeSeriesDict.read(..., format="sqlite3")` | 実装済み（未掲載） | SDB と同じ reader |
| **WAV** (`.wav`) | `TimeSeries.read(..., format="wav")`, `TimeSeriesDict.read(..., format="wav")`, `.write(..., format="wav")` | 掲載済み | 絶対時刻は保持しない |
| **MP3** (`.mp3`) | `TimeSeries.read(..., format="mp3")`, `TimeSeriesDict.read(..., format="mp3")`, `.write(..., format="mp3")` | 掲載済み | `pydub` / ffmpeg 前提 |
| **FLAC** (`.flac`) | `TimeSeries.read(..., format="flac")`, `TimeSeriesDict.read(..., format="flac")`, `.write(..., format="flac")` | 掲載済み | `pydub` / ffmpeg 前提 |
| **OGG** (`.ogg`) | `TimeSeries.read(..., format="ogg")`, `TimeSeriesDict.read(..., format="ogg")`, `.write(..., format="ogg")` | 掲載済み | `pydub` / ffmpeg 前提 |
| **M4A** (`.m4a`) | `TimeSeries.read(..., format="m4a")`, `TimeSeriesDict.read(..., format="m4a")`, `.write(..., format="m4a")` | 掲載済み | `pydub` / ffmpeg 前提 |

---

## 全管理対象一覧（状態つき）

### 状態ラベル

- `掲載済み`: 実装があり、現行 `io_formats.html` でも案内している
- `実装済み（未掲載）`: 実装はあるが、現行 `io_formats.html` での案内が不足している
- `実装済み（一部対応）`: 主経路は使えるが、対象クラスや命名条件が限定される
- `対応中`: direct I/O の説明面または format 名整理が未完
- `対応予定`: placeholder reader はあるが、実データ対応は未完

### A. GW標準

| 形式 / 経路 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| GWF | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | 掲載済み | gwpy 継承 |
| HDF5 | 各クラスの `.read(..., format="hdf5")`, `.write(..., format="hdf5")` | 掲載済み | 全体の標準保存形式 |
| ndscope-hdf5 | `TimeSeriesDict.read(..., format="ndscope-hdf5")`, `.write(..., format="ndscope-hdf5")` | 実装済み（未掲載） | Dict 限定 |
| DTTXML | `TimeSeriesDict.read(..., format="dttxml", products="...")` | 掲載済み | Read only |
| NDS2 | `TimeSeries.fetch()` | 掲載済み | ネットワーク経由 |
| GWOSC | `TimeSeries.fetch_open_data()` | 掲載済み | ネットワーク経由 |

### B. 地震・地球物理観測

| 形式 / 経路 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| MiniSEED | `TimeSeriesDict.read(..., format="miniseed")`, `.write(..., format="miniseed")` | 掲載済み | ObsPy 経由 |
| SAC | `TimeSeriesDict.read(..., format="sac")`, `.write(..., format="sac")` | 掲載済み | ObsPy 経由 |
| GSE2 | `TimeSeriesDict.read(..., format="gse2")`, `.write(..., format="gse2")` | 掲載済み | ObsPy 経由 |
| K-NET | `TimeSeriesDict.read(..., format="knet")` | 掲載済み | Read only |
| WIN | `TimeSeriesDict.read(..., format="win")` | 掲載済み | Read only |
| WIN32 | `TimeSeriesDict.read(..., format="win32")` | 掲載済み | Read only |
| ATS | `TimeSeries.read(..., format="ats")`, `TimeSeriesDict.read(..., format="ats")` | 掲載済み | ネイティブ binary reader |
| ATS.MTH5 | `TimeSeries.read(..., format="ats.mth5")` | 実装済み（一部対応） | 単一路のみ |
| MTH5 standalone | 専用 `read(..., format="mth5")` / `write(..., format="mth5")` は未整備 | 対応中 | format 名と対象範囲の整理が必要 |

### C. 汎用・解析用

| 形式 / 経路 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| CSV | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | 掲載済み | enhanced CSV reader を含む |
| TXT | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | 掲載済み | enhanced ASCII reader を含む |
| NetCDF4 | `TimeSeries.read(..., format="netcdf4")`, `TimeSeriesDict.read(..., format="netcdf4")`, `TimeSeriesMatrix.read(..., format="netcdf4")`, `.write(..., format="netcdf4")` | 掲載済み | TimeSeries 系 |
| Zarr | `TimeSeries.read(..., format="zarr")`, `TimeSeriesDict.read(..., format="zarr")`, `TimeSeriesMatrix.read(..., format="zarr")`, `.write(..., format="zarr")` | 掲載済み | TimeSeries 系 |
| Pickle | 各クラスの `.read()`, `.write()` | 掲載済み | セキュリティ警告を必須表示 |
| ROOT | `EventTable.read(..., format="root")`, `EventTable.write(..., format="root")` | 掲載済み | EventTable のみ |

### D. 計測機器・ロガー

| 形式 / 経路 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| GBD | `TimeSeries.read(..., format="gbd")`, `TimeSeriesDict.read(..., format="gbd")`, `TimeSeriesMatrix.read(..., format="gbd")` | 掲載済み | Read only |
| TDMS | `TimeSeries.read(..., format="tdms")`, `TimeSeriesDict.read(..., format="tdms")`, `TimeSeriesMatrix.read(..., format="tdms")` | 掲載済み | Read only |
| SDB | `TimeSeries.read(..., format="sdb")`, `TimeSeriesDict.read(..., format="sdb")` | 掲載済み | Read only |
| SQLite | `TimeSeries.read(..., format="sqlite")`, `TimeSeriesDict.read(..., format="sqlite")` | 実装済み（未掲載） | SDB alias |
| SQLite3 | `TimeSeries.read(..., format="sqlite3")`, `TimeSeriesDict.read(..., format="sqlite3")` | 実装済み（未掲載） | SDB alias |
| WAV | `TimeSeries.read(..., format="wav")`, `TimeSeriesDict.read(..., format="wav")`, `.write(..., format="wav")` | 掲載済み | 絶対時刻なし |
| MP3 | `TimeSeries.read(..., format="mp3")`, `TimeSeriesDict.read(..., format="mp3")`, `.write(..., format="mp3")` | 掲載済み | `pydub` / ffmpeg 前提 |
| FLAC | `TimeSeries.read(..., format="flac")`, `TimeSeriesDict.read(..., format="flac")`, `.write(..., format="flac")` | 掲載済み | `pydub` / ffmpeg 前提 |
| OGG | `TimeSeries.read(..., format="ogg")`, `TimeSeriesDict.read(..., format="ogg")`, `.write(..., format="ogg")` | 掲載済み | `pydub` / ffmpeg 前提 |
| M4A | `TimeSeries.read(..., format="m4a")`, `TimeSeriesDict.read(..., format="m4a")`, `.write(..., format="m4a")` | 掲載済み | `pydub` / ffmpeg 前提 |

### 未実装だが reader entry がある形式

| 形式 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| ORF | `TimeSeries.read(..., format="orf")`, `TimeSeriesDict.read(..., format="orf")`, `TimeSeriesMatrix.read(..., format="orf")` | 対応予定 | placeholder reader |
| MEM | `TimeSeries.read(..., format="mem")`, `TimeSeriesDict.read(..., format="mem")`, `TimeSeriesMatrix.read(..., format="mem")` | 対応予定 | placeholder reader |
| WVF | `TimeSeries.read(..., format="wvf")`, `TimeSeriesDict.read(..., format="wvf")`, `TimeSeriesMatrix.read(..., format="wvf")` | 対応予定 | placeholder reader |
| WDF | `TimeSeries.read(..., format="wdf")`, `TimeSeriesDict.read(..., format="wdf")`, `TimeSeriesMatrix.read(..., format="wdf")` | 対応予定 | placeholder reader |
| TAFFMAT | `TimeSeries.read(..., format="taffmat")`, `TimeSeriesDict.read(..., format="taffmat")`, `TimeSeriesMatrix.read(..., format="taffmat")` | 対応予定 | placeholder reader |
| LSF | `TimeSeries.read(..., format="lsf")`, `TimeSeriesDict.read(..., format="lsf")`, `TimeSeriesMatrix.read(..., format="lsf")` | 対応予定 | placeholder reader |
| LI | `TimeSeries.read(..., format="li")`, `TimeSeriesDict.read(..., format="li")`, `TimeSeriesMatrix.read(..., format="li")` | 対応予定 | placeholder reader |

### 周波数系列側の未実装 placeholder

| 形式 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| WIN | `FrequencySeries.read(..., format="win")`, `FrequencySeriesDict.read(..., format="win")`, `FrequencySeriesMatrix.read(..., format="win")` | 対応予定 | placeholder reader |
| WIN32 | `FrequencySeries.read(..., format="win32")`, `FrequencySeriesDict.read(..., format="win32")`, `FrequencySeriesMatrix.read(..., format="win32")` | 対応予定 | placeholder reader |
| SDB | `FrequencySeries.read(..., format="sdb")`, `FrequencySeriesDict.read(..., format="sdb")`, `FrequencySeriesMatrix.read(..., format="sdb")` | 対応予定 | placeholder reader |
| ORF | `FrequencySeries.read(..., format="orf")`, `FrequencySeriesDict.read(..., format="orf")`, `FrequencySeriesMatrix.read(..., format="orf")` | 対応予定 | placeholder reader |
| MEM | `FrequencySeries.read(..., format="mem")`, `FrequencySeriesDict.read(..., format="mem")`, `FrequencySeriesMatrix.read(..., format="mem")` | 対応予定 | placeholder reader |
| WVF | `FrequencySeries.read(..., format="wvf")`, `FrequencySeriesDict.read(..., format="wvf")`, `FrequencySeriesMatrix.read(..., format="wvf")` | 対応予定 | placeholder reader |
| WDF | `FrequencySeries.read(..., format="wdf")`, `FrequencySeriesDict.read(..., format="wdf")`, `FrequencySeriesMatrix.read(..., format="wdf")` | 対応予定 | placeholder reader |
| TAFFMAT | `FrequencySeries.read(..., format="taffmat")`, `FrequencySeriesDict.read(..., format="taffmat")`, `FrequencySeriesMatrix.read(..., format="taffmat")` | 対応予定 | placeholder reader |
| LSF | `FrequencySeries.read(..., format="lsf")`, `FrequencySeriesDict.read(..., format="lsf")`, `FrequencySeriesMatrix.read(..., format="lsf")` | 対応予定 | placeholder reader |
| LI | `FrequencySeries.read(..., format="li")`, `FrequencySeriesDict.read(..., format="li")`, `FrequencySeriesMatrix.read(..., format="li")` | 対応予定 | placeholder reader |

---

## グループ別の配置方針

### A. GW標準

最初に置く。
「まず何を選ぶか」の判断負荷を下げるため、A の先頭で **GWF / HDF5 / ndscope-hdf5 / DTTXML の違い** を短く比較する。

- **GWF**: 重力波時系列の標準交換形式
- **HDF5**: 汎用保存の第一候補。Field 系もここへ寄せる
- **ndscope-hdf5**: ndscope 互換の HDF5 スキーマ
- **DTTXML**: DTT 出力や診断ツール向け
- **NDS2 / GWOSC**: GW 系の取得経路。備考欄では `ネットワーク経由` と明示する

### B. 地震・地球物理観測

ObsPy 系と国内/電磁気形式をまとめる。
ユーザーにとっては「観測現場の既存形式を読む」区分として認識できればよい。

- MiniSEED / SAC / GSE2 / K-NET
- WIN / WIN32
- ATS
- ATS.MTH5
- MTH5 standalone（対応中）

### C. 汎用・解析用

研究ノート、汎用保存、外部解析との交換で使う形式をまとめる。
ただしこの文書では **direct I/O だけ** を扱う。

- CSV / TXT
- NetCDF4
- Zarr
- Pickle
- ROOT（EventTable の直I/O のみ）

### D. 計測機器・ロガー

機材固有またはロガー由来の時系列形式をまとめる。

- GBD
- TDMS
- SDB / SQLite / SQLite3
- WAV
- MP3 / FLAC / OGG / M4A

---

## `io_formats.html` への実装ルール

ページ本文は次の順序にする。

1. 判断ルール
2. グループ別クイック判定表
3. `read()` / `write()` の基本
4. グループ A-D の詳細
5. 開発者向け補足

また、本文では次を徹底する。

- `interop` 由来の対応表を混入させない
- 「変換できる」と「この形式で read/write できる」を同義にしない
- 開発者向けの stub / 設計レベル情報は末尾へ隔離する
- `Field` 対応は HDF5 と Pickle を中心に説明し、NetCDF4 / Zarr / ROOT は interop 側へ誘導する
- 実装済みだが未掲載の `ndscope-hdf5`, `SQLite`, `SQLite3` は本文へ明示的に追加する
- placeholder reader がある未実装形式は「対応予定」として末尾に隔離する

---

## interop 側へ移す事項

以下は [interop / 変換対応設計](./interop_変換対応設計.md) に移管する。

- ROOT object (`TGraph`, `TH1D`, `TH2D`, `TMultiGraph`) との相互変換
- `from_root()` / `to_tgraph()` / `to_th1d()` / `to_th2d()` / `write_root_file()`
- `from_xarray_field()` / `to_xarray_field()`
- `from_zarr()` / `to_zarr()` の array-level bridge
- `from_netcdf4()` / `to_netcdf4()` の interop helper
- 将来的な object-store / cloud-native bridge の議論

---

## 議論・未解決事項

> [!IMPORTANT]
> **D1: `ndscope-hdf5` 命名の是非**
> 実装名は `ndscope-hdf5`。ドキュメント側だけ `hdf5.ndscope` に揃えるかは別途判断が必要。

> [!IMPORTANT]
> **D2: 表記言語**
> 日本語（英語併記）スタイルを維持する。

> [!IMPORTANT]
> **D3: `ROOT` の公開位置**
> EventTable の直I/Oとして `io_formats` に残すが、Series / Histogram / Spectrogram の ROOT 変換は interop 側で主導する。
