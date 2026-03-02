# ファイル I/O 対応フォーマットガイド

gwexpy がサポートする全ファイルフォーマットの一覧と、各フォーマットの読み書き方法をまとめたエンドユーザー向けガイドです。
本ページでは内部実装メソッドは記載せず、ユーザーが直接利用するクラスメソッド `.read()` / `.write()` のみを示します。

---

## 対応フォーマット一覧

| フォーマット | 拡張子 | Read | Write | 推奨クラス・メソッド | 外部依存 | 備考 |
|---|---|:---:|:---:|---|---|---|
| GWF | `.gwf` | ○ | ○ | `TimeSeriesDict.read()` / `.write()` | — (gwpy 標準) | gwpy 標準フォーマット |
| HDF5 | `.h5`, `.hdf5` | ○ | ○ | `TimeSeriesDict.read(format="hdf5")` | — (gwpy 標準) | gwpy 標準 |
| LIGO_LW XML (DTTXML) | `.xml`, `.xml.gz` | ○ | × | `TimeSeriesDict.read(format="dttxml")` / `FrequencySeriesDict.read(format="dttxml")` | — | `products` 引数が必須 |
| CSV / TXT | `.csv`, `.txt` | ○ | ○ | `TimeSeriesDict.read(format="csv")` | — (gwpy 標準) | ASCII 形式。ディレクトリ読み込みにも対応 |
| Pickle | `.pkl` | ○ | ○ | `TimeSeries.read(format="pickle")` | — | Python シリアライズ |
| WAV | `.wav` | ○ | ○ | `TimeSeriesDict.read("file.wav")` | scipy | `t0` は常に 0（絶対時刻なし） |
| MiniSEED | `.mseed` | ○ | ○ | `TimeSeriesDict.read(format="miniseed")` | ObsPy | 地震波形フォーマット |
| SAC | `.sac` | ○ | ○ | `TimeSeriesDict.read(format="sac")` | ObsPy | 地震波形フォーマット |
| GSE2 | `.gse2` | ○ | ○ | `TimeSeriesDict.read(format="gse2")` | ObsPy | 地震波形フォーマット |
| KNET | `.knet` | ○ | × | `TimeSeriesDict.read(format="knet")` | ObsPy | K-NET 強震記録 |
| GBD | `.gbd` | ○ | × | `TimeSeriesDict.read("file.gbd", timezone=...)` | — | `timezone` が必須 |
| WIN / WIN32 | `.win`, `.cnt` | ○ | × | `TimeSeriesDict.read(format="win")` | ObsPy | NIED WIN 形式（改良版パーサ） |
| MTH5 | `.h5` | ○ | ○ | `TimeSeries.read(format="ats.mth5")` | mth5 | 磁力計データ（設計上サポート） |
| ATS | `.ats` | ○ | × | `TimeSeriesDict.read("file.ats")` | — | Metronix バイナリパーサ |
| ROOT | `.root` | ○ | ○ | `EventTable.read(format="root")` | — (gwpy 経由) | CERN ROOT テーブル |
| SQLite / SDB | `.sdb`, `.sqlite`, `.db` | ○ | × | `TimeSeriesDict.read("file.sdb")` | — | WeeWX / Davis 気象データ |
| NetCDF4 | `.nc` | ○ | ○ | — | — | 設計上サポート（gwpy/xarray 経由） |
| Zarr | `.zarr` | ○ | ○ | — | — | 設計上サポート（クラウド最適化） |
| Audio (MP3, FLAC 等) | `.mp3`, `.flac` | ○ | ○ | — | pydub | 設計上サポート |
| NDS2 | (ネットワーク) | ○ | × | `TimeSeries.fetch(...)` | nds2-client | ネットワークデータサーバ |
| TDMS | `.tdms` | ○ | × | `TimeSeriesDict.read("file.tdms")` | npTDMS | National Instruments |
| ORF | `.orf` | × | × | — | — | 未実装（stub） |

> **注記**: 「gwpy 標準」と記載したフォーマット（GWF, HDF5, CSV/TXT, Pickle）は gwpy のビルトイン IO 経路で処理されます。gwexpy は gwpy を継承しているため、そのまま利用可能です。
> NetCDF4, Zarr, Audio (MP3/FLAC) は設計表 (`io_support.csv`) に記載されていますが、gwexpy 内に専用の reader 実装はまだありません。

---

## フォーマット別詳細

---

### GBD — GRAPHTEC データロガー `.gbd`

**拡張子**: `.gbd`
**Read/Write**: Read ○ / Write ×
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/logger.gbd", timezone="Asia/Tokyo")
```

**必須引数**:
- `timezone` (str or tzinfo) — ロガーのローカルタイムゾーン（IANA 名、例: `"Asia/Tokyo"` または UTC オフセット）。**必須**。省略すると `ValueError` が発生します。

**主な任意引数**:
- `channels` (iterable[str], optional) — 読み込むチャンネルのリスト。省略時は全チャンネル。
- `digital_channels` (iterable[str], optional) — デジタルチャンネルとして扱うチャンネル名のリスト。省略時は `ALARM`, `ALARMOUT`, `PULSE*`, `LOGIC*` を自動検出。
- `unit` (str or Unit, optional) — 物理単位の上書き。デフォルト `'V'`。
- `epoch` (float or datetime, optional) — エポック（GPS 秒）の上書き。datetime の場合は GPS に変換される。
- `pad` (float, optional) — パディング値。デフォルト `NaN`。

**外部依存**: なし（ネイティブ実装）

**注意**:
- デジタルチャンネル（`ALARM`, `PULSE*` 等）は 0/1 にバイナライズされます。
- ヘッダの `HeaderSiz` フィールドが見つからない場合、`ValueError` が発生します。
- スケールは AMP セクションから自動抽出されます。

**参照実装**: `gwexpy/timeseries/io/gbd.py`

---

### ATS — Metronix `.ats`

**拡張子**: `.ats`
**Read/Write**: Read ○ / Write ×
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.ats")
```

mth5 ライブラリ経由で読む場合（`.atss` ファイル向け）:
```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.read("path/to/data.atss", format="ats.mth5")
```

**必須引数**: なし

**主な任意引数**: なし（バイナリヘッダから自動的にメタデータを取得）

**外部依存**:
- 標準読み込み: なし（ネイティブバイナリパーサ）
- `ats.mth5` フォーマット: `mth5` ライブラリが必要。未インストール時は `ImportError` が発生します。

**注意**:
- ATS ヘッダバージョン 80/81 に対応。CEA/sliced ヘッダ（バージョン 1080）は未対応（`NotImplementedError`）。
- LSB 値（mV/count）からボルト（V）への変換が自動で行われます。
- チャンネル名はヘッダ情報から `Metronix_{system}_{serial}_{type}_{sensor}_{serial}` の形式で自動生成されます。
- `ats.mth5` フォーマットは mth5 のファイル名規約に従う必要があります。規約に合わない場合はデフォルトのバイナリパーサを使用してください。

**参照実装**: `gwexpy/timeseries/io/ats.py`

---

### SDB — WeeWX / Davis 気象ステーション `.sdb`

**拡張子**: `.sdb`, `.sqlite`, `.sqlite3`
**Read/Write**: Read ○ / Write ×
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/weewx.sdb")
```

**必須引数**: なし

**主な任意引数**:
- `table` (str, optional) — 読み込むテーブル名。デフォルト `'archive'`。
- `columns` (list[str], optional) — 読み込む列名のリスト。省略時は既知の気象カラム（`barometer`, `outTemp`, `windSpeed` 等）を自動選択。

**外部依存**: なし（標準ライブラリの `sqlite3` + `pandas` を使用）

**注意**:
- Imperial 単位から SI 単位への自動変換が行われます（例: °F → °C、inHg → hPa、mph → m/s、inch → mm）。
- テーブルには `dateTime` カラムが必須です（UNIX タイムスタンプ）。
- サンプリングレートは `dateTime` の中央値差分から自動推定されます。

**参照実装**: `gwexpy/timeseries/io/sdb.py`

---

### TDMS — National Instruments `.tdms`

**拡張子**: `.tdms`
**Read/Write**: Read ○ / Write ×
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.tdms")
```

**必須引数**: なし

**主な任意引数**:
- `channels` (list[str], optional) — 読み込むチャンネルのリスト。チャンネル名は `"グループ名/チャンネル名"` の形式。省略時は全チャンネル。
- `unit` (str, optional) — 物理単位の上書き。

**外部依存**: `npTDMS` — 未インストール時は `ImportError` が発生し、`pip install nptdms` を案内するメッセージが表示されます。

**注意**:
- チャンネル名は `"グループ名/チャンネル名"` の形式で格納されます。
- `wf_increment`（サンプル間隔）と `wf_start_time`（開始時刻）は TDMS プロパティから取得されます。
- 開始時刻が `numpy.datetime64` や `datetime` の場合、GPS 時刻に自動変換されます。

**参照実装**: `gwexpy/timeseries/io/tdms.py`

---

### WAV — 音声ファイル `.wav`

**拡張子**: `.wav`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/audio.wav")
```

**必須引数**: なし

**主な任意引数**: なし

**外部依存**: `scipy`（`scipy.io.wavfile`）— gwexpy の推奨依存関係に含まれます。

**注意**:
- `t0` は常に `0.0`（GPS 秒）に設定されます。WAV ファイルは絶対時刻を保持しません。
- マルチチャンネルファイルの場合、チャンネル名は `channel_0`, `channel_1`, ... の形式になります。
- モノラルファイルは自動的に 1 チャンネルとして読み込まれます。
- Write は gwpy 標準の WAV writer 経路で対応します。

**参照実装**: `gwexpy/timeseries/io/wav.py`

---

### MiniSEED — 地震波形 `.mseed`

**拡張子**: `.mseed`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

# 読み込み
tsd = TimeSeriesDict.read("path/to/data.mseed", format="miniseed")

# 書き出し
tsd.write("output.mseed", format="miniseed")
```

**必須引数**: なし

**主な任意引数**:
- `channels` (list[str], optional) — 読み込むチャンネル。トレース ID（`NET.STA.LOC.CHA`）またはチャンネルコードで指定。
- `unit` (str, optional) — 物理単位の上書き。
- `epoch` (float or datetime, optional) — エポックの上書き。
- `timezone` (str, optional) — タイムゾーン指定。
- `pad` (float, optional) — ギャップ埋め値。デフォルト `NaN`。
- `gap` (str, optional) — ギャップ処理方法。`"pad"`（デフォルト）または `"raise"`。

**外部依存**: `ObsPy` — 未インストール時は `ImportError` が発生し、`pip install obspy` を案内するメッセージが表示されます。

**注意**:
- ObsPy の `read()` 関数を経由してデータを読み込みます。
- ギャップがある場合はデフォルトで `NaN` パディングされます。`gap="raise"` でエラーにすることも可能です。
- トレースの自動マージ（`merge(method=1)`）が適用されます。

**参照実装**: `gwexpy/timeseries/io/seismic.py`

---

### SAC — 地震波形 `.sac`

**拡張子**: `.sac`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.sac", format="sac")
tsd.write("output.sac", format="sac")
```

**必須引数**: なし

**主な任意引数**: MiniSEED と同一（`channels`, `unit`, `epoch`, `timezone`, `pad`, `gap`）。

**外部依存**: `ObsPy`

**注意**:
- SAC は通常 1 トレース/ファイルです。複数トレースの書き出しは ObsPy の挙動に依存します。

**参照実装**: `gwexpy/timeseries/io/seismic.py`

---

### GSE2 — 地震波形 `.gse2`

**拡張子**: `.gse2`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.gse2", format="gse2")
tsd.write("output.gse2", format="gse2")
```

**必須引数**: なし

**主な任意引数**: MiniSEED と同一（`channels`, `unit`, `epoch`, `timezone`, `pad`, `gap`）。

**外部依存**: `ObsPy`

**参照実装**: `gwexpy/timeseries/io/seismic.py`

---

### KNET — K-NET 強震記録 `.knet`

**拡張子**: `.knet`
**Read/Write**: Read ○ / Write ×
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.knet", format="knet")
```

**必須引数**: なし

**主な任意引数**: MiniSEED と同一（`channels`, `unit`, `epoch`, `timezone`, `pad`, `gap`）。

**外部依存**: `ObsPy`

**注意**:
- 読み込みのみ対応。書き出し（Writer）は登録されていません。

**参照実装**: `gwexpy/timeseries/io/seismic.py`

---

### WIN / WIN32 — NIED WIN 形式 `.win` / `.cnt`

**拡張子**: `.win`, `.cnt`
**Read/Write**: Read ○ / Write ×
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.win", format="win")
# または
tsd = TimeSeriesDict.read("path/to/data.cnt", format="win32")
```

**必須引数**: なし

**主な任意引数**:
- `century` (str, optional) — 年の世紀部分。デフォルト `"20"`。

**外部依存**: `ObsPy` — ObsPy がインストールされていない場合、reader が登録されず `ImportError` が発生します。

**注意**:
- gwexpy 独自の改良版パーサを使用しています。ObsPy 標準の WIN reader と比較して以下の修正が適用されています:
  - 0.5 バイト（4 ビット）デルタデコード: 下位ニブルの符号処理を修正。奇数デルタ数の未使用ニブルをスキップ。
  - 3 バイト（24 ビット）デルタデコード: 演算子優先度と符号保持アンパック/シフトを修正。
- ギャップは `NaN` でマージされます。

**参照実装**: `gwexpy/timeseries/io/win.py`

---

### DTTXML — Diag DTT XML (時系列)

**拡張子**: `.xml`, `.xml.gz`
**Read/Write**: Read ○ / Write ×
**推奨 API（時系列）**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/dtt_output.xml", format="dttxml", products="TS")
```

**推奨 API（周波数系列）**:
```python
from gwexpy.frequencyseries.collections import FrequencySeriesDict

fsd = FrequencySeriesDict.read("path/to/dtt_output.xml", format="dttxml", products="PSD")
```

```python
from gwexpy.frequencyseries.matrix import FrequencySeriesMatrix

fsm = FrequencySeriesMatrix.read("path/to/dtt_output.xml", format="dttxml", products="TF")
```

**必須引数**:
- `products` (str) — 取り出す製品の種類。**必須**。省略すると `ValueError` が発生します。
  - 時系列: `"TS"`
  - 周波数系列: `"PSD"`, `"ASD"`, `"FFT"`
  - マトリクス: `"TF"`, `"STF"`, `"CSD"`, `"COH"`

**主な任意引数**:
- `channels` (iterable[str], optional) — 読み込むチャンネルのリスト。
- `unit` (str, optional) — 物理単位の上書き。
- `epoch` (float or datetime, optional) — エポックの上書き。
- `timezone` (str, optional) — タイムゾーン指定。
- `native` (bool, optional) — `True` にすると gwexpy ネイティブ XML パーサを使用。複素 TF データ（subtype 6 の位相損失修正）に推奨。デフォルト `False`。（FrequencySeriesDict / FrequencySeriesMatrix のみ）
- `rows`, `cols`, `pairs` — マトリクス読み込み時のフィルタリング（FrequencySeriesMatrix のみ）。

**外部依存**: なし（ネイティブ実装）

**注意**:
- 時系列と周波数領域の両方に対応。`products` の値で出力型が決まります。
- 拡張子 `.xml` は自動識別されます（`format="dttxml"` の明示指定は省略可能）。

**参照実装**: `gwexpy/timeseries/io/dttxml.py`, `gwexpy/frequencyseries/io/dttxml.py`

---

### GWF — 重力波フレーム `.gwf`

**拡張子**: `.gwf`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.gwf", format="gwf")
tsd.write("output.gwf", format="gwf")
```

**必須引数**: なし（gwpy 標準の引数に準拠）

**外部依存**: — （gwpy 標準。内部で `python-ldas-tools-framecpp` 等を使用）

**注意**:
- gwpy の標準 IO 経路で処理されます。gwexpy 側に独自の reader/writer 実装はありません。

**参照実装**: gwpy 標準（`gwpy/timeseries/io/gwf.py`）

---

### HDF5 — 汎用科学データ `.h5` / `.hdf5`

**拡張子**: `.h5`, `.hdf5`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.h5", format="hdf5")
tsd.write("output.h5", format="hdf5")
```

**必須引数**: なし

**外部依存**: `h5py`（gwexpy の必須依存関係に含まれます）

**注意**:
- gwexpy の `TimeSeriesDict.read()` は HDF5 ファイルに対して拡張された読み込みロジックを持ちます。
- レイアウト自動検出（`LAYOUT_DATASET` / `LAYOUT_GROUP`）に対応。
- キーマップと並び順の復元に対応。

**参照実装**: gwpy 標準（`gwpy/timeseries/io/hdf5.py`）+ gwexpy 拡張（`gwexpy/timeseries/collections.py`）

---

### CSV / TXT — ASCII テキスト `.csv` / `.txt`

**拡張子**: `.csv`, `.txt`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

# 単一ファイル
tsd = TimeSeriesDict.read("path/to/data.csv", format="csv")

# ディレクトリ内の複数ファイルをまとめて読み込み
tsd = TimeSeriesDict.read("path/to/data_dir/")
```

**必須引数**: なし

**外部依存**: なし（gwpy 標準）

**注意**:
- gwexpy はディレクトリパスを指定すると、ディレクトリ内の CSV/TXT ファイルをまとめて `TimeSeriesDict` として読み込む拡張機能を持ちます。

**参照実装**: gwpy 標準（`gwpy/timeseries/io/ascii.py`）+ gwexpy 拡張（`gwexpy/timeseries/collections.py`）

---

### Pickle — Python シリアライズ `.pkl`

**拡張子**: `.pkl`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.read("path/to/data.pkl", format="pickle")
ts.write("output.pkl", format="pickle")
```

**必須引数**: なし

**外部依存**: なし

**注意**:
- gwpy 標準のシリアライズ経路。信頼できるソースのファイルにのみ使用してください（pickle の安全性に関する一般的な注意事項が適用されます）。

**参照実装**: gwpy 標準

---

### ROOT — CERN ROOT `.root`

**拡張子**: `.root`
**Read/Write**: Read ○ / Write ○
**推奨 API**:
```python
from gwexpy.table import EventTable

table = EventTable.read("path/to/data.root", format="root")
```

**外部依存**: — （gwpy の table/root 経路経由）

**注意**:
- gwexpy の `table/io/root.py` は gwpy の同名モジュールの再エクスポートです。
- 主にイベントテーブル形式のデータに使用されます。

**参照実装**: gwpy 標準（`gwpy/table/io/root.py`）— gwexpy 経由: `gwexpy/table/io/root.py`

---

### NDS2 — ネットワークデータサーバ

**拡張子**: なし（ネットワークプロトコル）
**Read/Write**: Read ○ / Write ×
**推奨 API**:
```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.fetch("channel_name", start, end)
```

**外部依存**: `nds2-client`（Python バインディング）

**注意**:
- ファイルではなくネットワーク経由のデータ取得です。
- gwpy 標準の `fetch()` メソッドを使用します。

**参照実装**: gwpy 標準（`gwpy/timeseries/io/nds2.py`）

---

## 設計上サポート（gwexpy 内に専用実装なし）

以下のフォーマットは設計表（`io_support.csv`）に記載されていますが、gwexpy リポジトリ内に専用の reader/writer 実装は現時点でありません。gwpy 標準経路または外部ライブラリとの連携で将来対応予定です。

| フォーマット | 拡張子 | 設計上の Read/Write | 備考 |
|---|---|:---:|---|
| MTH5 | `.h5` | Read ○ / Write ○ | `mth5` ライブラリ経由。`ats.mth5` で部分対応あり |
| NetCDF4 | `.nc` | Read ○ / Write ○ | xarray 経由で対応予定 |
| Zarr | `.zarr` | Read ○ / Write ○ | クラウド最適化形式。将来対応 |
| Audio (MP3, FLAC 等) | `.mp3`, `.flac` | Read ○ / Write ○ | `pydub` ライブラリ経由で対応予定 |

---

## 未実装（stub）一覧

以下のフォーマットはプレースホルダ（stub）として IO レジストリに登録されています。`.read()` を呼び出すと未実装例外（`UnimplementedIOError` または `NotImplementedError`）が発生します。仕様書やサンプルファイルの提供により、将来の実装が予定されています。

### 時系列 stub（`gwexpy/timeseries/io/stubs.py`）

| フォーマット名 | 登録クラス | 備考 |
|---|---|---|
| `orf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | ORF 形式 |
| `mem` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | MEM 形式 |
| `wvf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | WVF 形式 |
| `wdf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | WDF 形式 |
| `taffmat` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | TAFFMAT 形式 |
| `lsf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | LSF 形式 |
| `li` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | LI 形式 |

### 周波数系列 stub（`gwexpy/frequencyseries/io/stubs.py`）

| フォーマット名 | 登録クラス | 備考 |
|---|---|---|
| `win` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 時系列では実装済みだが周波数領域は未実装 |
| `win32` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 同上 |
| `sdb` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 時系列では実装済みだが周波数領域は未実装 |
| `orf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 未実装 |
| `mem` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 未実装 |
| `wvf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 未実装 |
| `wdf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 未実装 |
| `taffmat` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 未実装 |
| `lsf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 未実装 |
| `li` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | 未実装 |

---

## `.read()` / `.write()` の基本的な使い方

gwexpy のすべてのデータクラスは gwpy の IO レジストリを利用しています。基本的な使い方は以下の通りです。

### 読み込み

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

# 拡張子から自動判別
tsd = TimeSeriesDict.read("path/to/file.gbd", timezone="Asia/Tokyo")

# format を明示的に指定
tsd = TimeSeriesDict.read("path/to/file.dat", format="miniseed")

# 単一チャンネルを読み込む
ts = TimeSeries.read("path/to/file.gbd", timezone="Asia/Tokyo")
```

### 書き出し

```python
# 対応フォーマットへ書き出し
tsd.write("output.mseed", format="miniseed")
tsd.write("output.h5", format="hdf5")
```

### 周波数系列

```python
from gwexpy.frequencyseries.collections import FrequencySeriesDict
from gwexpy.frequencyseries.frequencyseries import FrequencySeries

fsd = FrequencySeriesDict.read("path/to/dtt.xml", format="dttxml", products="PSD")
```

### `format` 引数の動作

- 省略時: ファイルの拡張子から `io_registry.register_identifier(...)` で登録された識別関数を使って自動判別されます。
- 明示指定時: 指定されたフォーマット名に対応する reader/writer が直接呼び出されます。
- 自動判別が失敗する場合（例: `.xml` は DTTXML 以外の XML でも使われるため）は `format` を明示してください。

---

## 参照元ファイル一覧

本ドキュメントの作成にあたり参照した実装ファイルの一覧です。

| モジュールパス | 概要 |
|---|---|
| `gwexpy/timeseries/io/__init__.py` | 時系列 IO モジュールの登録エントリポイント |
| `gwexpy/timeseries/io/gbd.py` | GBD reader。`timezone` は必須 |
| `gwexpy/timeseries/io/ats.py` | ATS reader（バイナリパーサ）。`ats.mth5` variant あり |
| `gwexpy/timeseries/io/sdb.py` | SDB reader（WeeWX SQLite）。`sdb`, `sqlite`, `sqlite3` の 3 フォーマット名で登録 |
| `gwexpy/timeseries/io/tdms.py` | TDMS reader。`npTDMS` に依存 |
| `gwexpy/timeseries/io/wav.py` | WAV reader。`scipy.io.wavfile` ラッパー。`t0=0` 固定 |
| `gwexpy/timeseries/io/seismic.py` | MiniSEED / SAC / GSE2 / KNET reader/writer。ObsPy に依存 |
| `gwexpy/timeseries/io/win.py` | WIN/WIN32 reader。ObsPy 必須。改良版 4bit/24bit デルタデコード |
| `gwexpy/timeseries/io/dttxml.py` | DTTXML 時系列 reader。`products` は必須 |
| `gwexpy/timeseries/io/stubs.py` | 時系列 stub（`orf`, `mem`, `wvf`, `wdf`, `taffmat`, `lsf`, `li`） |
| `gwexpy/timeseries/io/hdf5.py` | HDF5 IO（gwpy 再エクスポート） |
| `gwexpy/timeseries/io/ascii.py` | ASCII IO（gwpy 再エクスポート） |
| `gwexpy/timeseries/io/nds2.py` | NDS2 IO（gwpy 再エクスポート） |
| `gwexpy/timeseries/io/cache.py` | キャッシュ IO（gwpy 再エクスポート） |
| `gwexpy/timeseries/io/losc.py` | GWOSC IO（gwpy 再エクスポート） |
| `gwexpy/timeseries/io/core.py` | コア IO（gwpy 再エクスポート） |
| `gwexpy/frequencyseries/io/__init__.py` | 周波数系列 IO モジュールの登録エントリポイント |
| `gwexpy/frequencyseries/io/dttxml.py` | DTTXML 周波数系列 reader。`products` は必須。`native` オプションあり |
| `gwexpy/frequencyseries/io/stubs.py` | 周波数系列 stub（`win`, `win32`, `sdb`, `orf`, `mem`, `wvf`, `wdf`, `taffmat`, `lsf`, `li`） |
| `gwexpy/frequencyseries/io/hdf5.py` | HDF5 IO（gwpy 再エクスポート） |
| `gwexpy/frequencyseries/io/ascii.py` | ASCII IO（gwpy 再エクスポート） |
| `gwexpy/frequencyseries/io/ligolw.py` | LIGO_LW IO（gwpy 再エクスポート） |
| `gwexpy/table/io/root.py` | ROOT IO（gwpy 再エクスポート） |
| `gwexpy/timeseries/collections.py` | `TimeSeriesDict.read()` 拡張（HDF5 レイアウト検出、ディレクトリ読み込み） |
| `docs/developers/design/design_data/io_support.csv` | 設計データ: 対応フォーマット一覧 |
