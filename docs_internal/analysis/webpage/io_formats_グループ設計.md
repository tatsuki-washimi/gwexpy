# `io_formats.html` ドメイン別グループ設計（Audit完了版 v2）

作成日: 2026-04-16  
最終更新: 2026-04-16（Field系クラスの対応状況を追加）

---

## 網羅的な対応状況マップ（100% コード整合）

リポジトリ内の I/O レジストリ、継承構造、および `interop` 層を走査した結果に基づき、Field系クラス（ScalarField 等）を含む正確な対応状況を定義する。

| グループ | 形式 / 拡張子 | TimeSeries | FreqSeries | Spectrogram | Histogram | EventTable | **Field** | 備考 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **A: GW標準** | **GWF** (`.gwf`) | ○ | — | — | — | (○) | — | Frame形式はSeries優先 |
| | **HDF5** (`.h5`) | ○ | ○ | ○ | ○ | ○ | **○** | **Fieldの推奨形式** |
| | **DTTXML** (`.xml`) | ○ | ○ | ○ | — | — | — | シリーズ・行列抽出用 |
| | **Network** (NDS2) | ○ | — | — | — | — | — | ネットワーク取得 |
| **B: 地震/観測**| **ObsPy系** | ○ | — | — | — | — | — | 1D時系列優先 |
| | **WIN** (`.win`) | ○ | — | — | — | — | — | 独自改良デコーダ |
| | **ATS / MTH5** | ○ | — | — | — | — | — | 時系列データ用 |
| **C: 汎用/物理**| **ROOT** (`.root`) | ○ | ○ | ○ | ○ | ○ | **○** | interop層で対応 |
| | **CSV / TXT** | ○ | ○ | — | — | ○ | — | プレーンテキスト |
| | **NetCDF4** | ○ | — | — | — | — | (○) | xarray経由で部分対応 |
| | **Zarr** (`.zarr`) | ○ | — | — | — | — | **○** | 多次元配列対応 |
| | **Pickle** (`.pkl`) | ○ | ○ | ○ | ○ | ○ | **○** | 全クラス共通 |
| **D: 計測機** | **GB-D / TDMS** | ○ | — | — | — | — | — | 1Dロガー用 |
| | **Audio** (WAV等) | ○ | — | — | — | — | — | 音声信号解析 |

---

## 詳細グループ設計

### グループ A: 重力波標準形式（GW Standard）

#### A-1: GWF（重力波フレームファイル）`.gwf`
- **対応**: TimeSeries, TimeSeriesDict, (EventTable)

#### A-2: HDF5（GWpy/GWexpy オブジェクト保存）`.h5` / `.hdf5`
**Field系クラスの標準的な保存形式。** `Array4D` のメタデータ（`axis0_domain`, `space_domains` 等）を属性として保持可能。
- **対応**: 全クラス（含む **ScalarField, VectorField, TensorField**）

#### A-3: DTTXML（DTT 診断ツール出力）`.xml`
- **対応**: TimeSeriesDict, FrequencySeriesDict, FrequencySeriesMatrix, Spectrogram

#### A-4: ネットワークデータ取得
NDS2（`TimeSeries.fetch`）および GWOSC（`fetch_open_data`）をサポート。

---

### グループ B: 地震・地球物理観測（Seismology / Geophysics）

#### B-1: 国際・国内標準（ObsPy 連携）
MiniSEED, SAC, GSE2, K-NET, WIN 等。
- **対応**: TimeSeries, TimeSeriesDict

#### B-2: ATS / MTH5
`format="ats.mth5"` により TimeSeries として読み込み可能。

---

### グループ C: 汎用・物理解析形式（General Purpose / High Energy Physics）

#### C-1: ROOT (CERN ROOT) `.root`
`interop` 層により、GWexpy オブジェクトを ROOT の `TGraph`, `TH1D`, `TH2D`, `TMultiGraph` へ（およびその逆へ）変換。Field 系は 2D スライス抽出後に ROOT オブジェクト化するワークフローを想定。
- **対応**: **全主要クラス**

#### C-2: Zarr / NetCDF4
多次元・高速読込に適した形式。`ScalarField` などの `Array4D` 継承クラスとの親和性が高い。
- **対応**: TimeSeries, **Field系**

#### C-3: CSV / TXT / Pickle
- **CSV**: Series, Table。
- **Pickle**: **全クラス**。

---

### グループ D: 計測機器・ロガー専用形式（Instrument Data）

#### D-1: GB-D / TDMS / SDB / Audio
特定機材・音声形式用。主として 1D の `TimeSeries` 読込に使用。

---

## 議論・未解決事項

> [!IMPORTANT]
> **D1: ndscope 命名の是非**
> 実装名は `ndscope-hdf5` だが、ドキュメント表記を `hdf5.ndscope` にして他と揃えるか？

> [!IMPORTANT]
> **D2: 表記言語**
> 日本語（英語併記）スタイルを採用する。
