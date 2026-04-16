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

`io_formats.html` は、**4グループ + 1特殊区分** で再編する。

| グループ | 主な形式 | ページ上の役割 | 主対象 |
|---|---|---|---|
| **A: GW標準** | GWF, HDF5, DTTXML | 重力波解析の標準保存・交換形式 | TimeSeries系、HDF5 は Field を含む |
| **B: 地震・地球物理観測** | MiniSEED, SAC, GSE2, K-NET, WIN, ATS / MTH5 | 地震・電磁気観測の現場データ | TimeSeries, TimeSeriesDict |
| **C: 汎用・解析用** | CSV / TXT, NetCDF4, Zarr, Pickle, ROOT | 汎用保存、解析連携、交換 | TimeSeries系中心。Pickle は全クラス、ROOT は EventTable 直I/O に限定 |
| **D: 計測機器・ロガー** | GBD, TDMS, SDB, WAV / Audio | 機材固有またはロガー由来の時系列 | TimeSeries, TimeSeriesDict |
| **特殊: ネットワーク取得** | NDS2, GWOSC | ローカルファイルではなく遠隔取得 | TimeSeries |

> [!NOTE]
> `NDS2` と `GWOSC` は一覧上は近接配置してよいが、ローカルファイル形式とは別区分で扱う。
> `ROOT` はこの文書では **EventTable の直I/O** だけを含める。Series / Histogram / Spectrogram の ROOT object 変換は interop 側へ移す。

---

## 対応状況マップ（I/O専用）

| 形式 / 経路 | 直接 I/O の主対象 | Read | Write | 備考 |
|---|---|:---:|:---:|---|
| **GWF** (`.gwf`) | `TimeSeries`, `TimeSeriesDict` | ○ | ○ | Frame 形式。Series 優先 |
| **HDF5** (`.h5`, `.hdf5`) | `TimeSeries` 系, `Spectrogram`, `Histogram`, `EventTable`, `Field` | ○ | ○ | Field 系の標準保存形式 |
| **DTTXML / LIGO_LW** (`.xml`) | `TimeSeriesDict`, `FrequencySeriesDict`, `FrequencySeriesMatrix`, `Spectrogram` | ○ | × | DTT 診断ツール出力。`products` 指定前提 |
| **NDS2 / GWOSC** | `TimeSeries` | ○ | × | 特殊区分。ネットワーク取得 |
| **MiniSEED / SAC / GSE2 / K-NET** | `TimeSeries`, `TimeSeriesDict` | ○ | 形式により異なる | ObsPy 経由 |
| **WIN** (`.win`) | `TimeSeriesDict` | ○ | × | 独自改良デコーダ |
| **ATS / MTH5** | `TimeSeries`, `TimeSeriesDict` | ○ | 形式により異なる | `ats` と `ats.mth5` を分けて扱う |
| **CSV / TXT** (`.csv`, `.txt`) | `TimeSeries`, `TimeSeriesDict` | ○ | ○ | 汎用テキスト |
| **NetCDF4** (`.nc`) | `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix` | ○ | ○ | xarray ベースの direct I/O。Field は interop 側 |
| **Zarr** (`.zarr`) | `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix` | ○ | ○ | TimeSeries 系の direct I/O。Field は interop 側 |
| **Pickle** (`.pkl`) | 全主要クラス | ○ | ○ | セキュリティ警告が必要 |
| **ROOT** (`.root`) | `EventTable` | ○ | ○ | gwpy table/root 経路。ROOT object 変換は interop 側 |
| **GBD** (`.gbd`) | `TimeSeriesDict` | ○ | × | `timezone` 必須 |
| **TDMS** (`.tdms`) | `TimeSeriesDict` | ○ | × | `npTDMS` 依存 |
| **SDB / SQLite** (`.sdb`, `.sqlite`) | `TimeSeriesDict` | ○ | × | 気象・ロガー DB |
| **WAV** (`.wav`) | `TimeSeriesDict` | ○ | ○ | 絶対時刻は保持しない |
| **Audio** (`.mp3` 等) | `TimeSeriesDict` | ○ | ○ | `pydub` / ffmpeg 前提 |

---

## グループ別の配置方針

### A. GW標準

最初に置く。
「まず何を選ぶか」の判断負荷を下げるため、A の先頭で **GWF / HDF5 / DTTXML の違い** を短く比較する。

- **GWF**: 重力波時系列の標準交換形式
- **HDF5**: 汎用保存の第一候補。Field 系もここへ寄せる
- **DTTXML**: DTT 出力や診断ツール向け

### B. 地震・地球物理観測

ObsPy 系と国内/電磁気形式をまとめる。
ユーザーにとっては「観測現場の既存形式を読む」区分として認識できればよい。

- MiniSEED / SAC / GSE2 / K-NET
- WIN
- ATS / MTH5

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
- SDB / SQLite
- WAV / Audio

### 特殊区分: ネットワーク取得

ローカルファイルと混同させないため、NDS2 / GWOSC は独立区分にする。

---

## `io_formats.html` への実装ルール

ページ本文は次の順序にする。

1. 判断ルール
2. グループ別クイック判定表
3. `read()` / `write()` の基本
4. グループ A-D の詳細
5. 特殊区分（NDS2 / GWOSC）
6. 開発者向け補足

また、本文では次を徹底する。

- `interop` 由来の対応表を混入させない
- 「変換できる」と「この形式で read/write できる」を同義にしない
- 開発者向けの stub / 設計レベル情報は末尾へ隔離する
- `Field` 対応は HDF5 と Pickle を中心に説明し、NetCDF4 / Zarr / ROOT は interop 側へ誘導する

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
