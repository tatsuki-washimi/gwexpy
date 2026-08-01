# Virgo データ導線（frame path / dataDisplay ROOT product path）— 設計書

> Last-updated: 2026-08-01 (rev 1 — 初版)
> Reviewer Status: **draft**（ユーザーレビュー待ち）

Status: planned

対象マイルストーン: 未定（Issue A のみ次パッチ候補、他は v0.2.0 以降）

---

## 1. 背景と非目標

Status: planned

### 1.1 経緯

GWADW2026 の発表後、Virgo の協力者から、GWexpy で Virgo データを読む方法について
フィードバックを受けた。要旨は次の 3 点である。

1. Virgo dataDisplay には、LIGO の NDscope に相当する簡便な raw data export 経路がない。
2. 必要に応じて script でデータを抽出し HDF5 に保存する運用がある。
3. 実用的な第一経路は、frame file を直接指定して通常どおり GWpy で開くことである。

（私信のため、発言者と原文は本設計書には記載しない。維持者ローカルに保管。）

加えて、dataDisplay は描画に CERN ROOT を使っており、`TCanvas` / `TH1` / `TGraph` などを
`.root` ファイルとして保存できる。dataDisplay の user manual にも `-o [filename]` による
`.root` 出力、`-noplot` による `dd.root` 生成、`Save Plots` による `dy.root` 保存、
reference plot の `ddref.root` 保存、`export2ascii.C` による ASCII 抽出が記載されている。

したがって「export 機能がない」は、**raw channel data の簡便な export がない**という意味で
あって、**ROOT plot product の export はある**と整理し直すのが正確である。

### 1.2 GWexpy の立ち位置

GWexpy は real-time control system ではなく、control-room / offline / near-online
commissioning analysis のための解析レイヤーである。この切り分けが、以下の非目標を規定する。

### 1.3 非目標（Non-goals）

- dataDisplay GUI の自動操作
- Cm メッセージ / shared memory / dataSender との接続
- Virgo control system / CDS / real-time infrastructure への依存
- real-time control loop への介入
- Virgo site 固有の private path・private channel 命名規約の core へのハードコード
- schema 未確定の HDF5 を「対応済み」と主張すること
- 任意 HDF5 の generic reader（HDF5 は schema-less container であり、schema なしに正しく
  読むことはできない。documented profile のみを支援する）
- `.C` / Cling マクロの無条件実行

---

## 2. 三経路の責務分離

Status: planned

```
主経路（raw data）
  Virgo .gwf / .ffl
      -> GWpy GWF backend        ← frame decoding は GWpy が authoritative
      -> GWexpy TimeSeries / TimeSeriesDict
      -> GWexpy analysis / matrix workflows

副経路（plot product）
  dataDisplay .root
      -> ROOT object 抽出（TCanvas/TPad 走査 → TGraph/TH1/TH2）
      -> GWexpy TimeSeries / FrequencySeries / Histogram / Spectrogram

保留経路
  .C / Cling マクロ           ← security（任意コード実行）
  script 生成 HDF5            ← schema 未入手
```

| Layer | 責務 |
|---|---|
| GWpy | frame file の decode |
| GWexpy | GWpy 互換 container・解析メソッド・matrix workflow |
| Virgo site script | dataDisplay からの抽出（必要な場合） |
| ROOT interop | plot product の数値ペイロード回収 |
| docs | 推奨 workflow と非目標の明示 |

### 2.1 副経路は raw data ではない（重要）

dataDisplay の `.root` 出力は、多くの場合、元 channel data に対して次の処理を適用した
**後段の産物**である。

- resampling / anti-aliasing
- band-pass filtering
- FFT window / overlap / averaging / median
- DC subtraction（noDC）
- trigger selection
- unit conversion
- time / frequency binning

したがって `.root` から取り出せるものは frame file 内の raw channel data と同一とは限らない。
docs でも実装でも、この区別を曖昧にしてはならない。

```
.ffl / .gwf 経路 : raw または標準の Virgo channel data access
dataDisplay .root 経路 : 処理済み・描画済み data product の再利用
```

### 2.2 GWDama HDF5 interop との分離

GWexpy には既に GWDama interop（外部 project との thin interop / reader MVP、code copy
なし）の方針が別途存在する。本設計で扱う「Virgo script 生成 HDF5」はこれとは別物であり、
以下のとおり分離して管理する。

| 項目 | 位置づけ |
|---|---|
| GWpy GWF reader | primary Virgo frame path |
| GWpy/GWexpy HDF5 | 一般的な HDF5 I/O 互換 |
| GWDama HDF5 interop | 外部 package / data model interop |
| Virgo script 生成 HDF5 | site-practical fallback profile（schema 未入手） |
| dataDisplay direct connector | non-goal |

---

## 3. 実測結果

Status: F1〜F3 は completed (verified: `python /tmp/verify_root_dtype.py`)、F4 は planned（未実測）

環境: ROOT 6.36.04（system python 3.12）、gwpy 4.0.1、gwexpy `main` @ `7ded3b0f9`。
検証スクリプトの内容は Appendix B に収録（`/tmp` 上にあるため揮発性。再現時は Appendix B
から復元する）。

> **本文訂正**: 元の設計会話では「ROOT interop は standalone `TGraph`, `TH1D`, `TH2D` の
> object-level import/export が実装済みで、残る gap は主に `TCanvas` 走査」と結論していた。
> 実測の結果この整理は不正確であり、以下 F1・F2 のとおり訂正する。元の記録は Appendix A
> に保持する。

### F1. `TH1F` / `TH2F` は無警告で壊れた値を返す

Status: completed (verified: `python /tmp/verify_root_dtype.py`)

投入値は 1D が `[1, 2, 3, 4]`、2D が `[11, 12, 21, 22]`。

```
TH1F -> [3.20000076e+001 5.34643471e-315 4.00193173e-322 6.90476210e-310]  correct=False
TH1D -> [1. 2. 3. 4.]                                                      correct=True
TH2F -> [5.44486713e-315 4.43176884e-321 0.00000000e+000 5.47212582e-310]  correct=False
TH2D -> [11. 12. 21. 22.]                                                  correct=True
```

**破損値は実行ごとに変動する**（未初期化メモリを読むため）。したがって「特定の壊れた値が
出る」ではなく「非決定的なゴミが返る」と理解すること。回帰テストは値の一致
（`correct=True`）で判定し、破損値そのものを期待値に書かない。

原因は `gwexpy/interop/root_.py` が `obj.GetArray()` の戻り値を常に `dtype=np.float64` で
`np.frombuffer` していること。`TH1F` / `TH2F` の `GetArray()` は `Float_t*`（4 byte）を
返すため、バイト列を誤解釈する。該当箇所:

| 位置 | 内容 |
|---|---|
| `root_.py:312` | TH2 content |
| `root_.py:328` | TH2 sumw2 |
| `root_.py:365` | TH1 content |
| `root_.py:372` | TH1 sumw2 |

例外も警告も出ないため、`TH1F` を読んでいるユーザーは黙って誤った解析結果を得る。
`to_th1d` / `to_th2d` は常に `TH1D` / `TH2D` を生成するので、gwexpy 内の round-trip
テストだけではこの不具合は永久に検出されない。**Virgo とは独立に影響する既存バグ。**

### F2. `TH1` → 非 Histogram クラスは未実装

Status: completed (verified: `python /tmp/verify_root_dtype.py`)

```
TH1D -> FrequencySeries: NotImplementedError: TH1 to non-Histogram class conversion not fully implemented yet.
TH1D -> TimeSeries:      NotImplementedError: TH1 to non-Histogram class conversion not fully implemented yet.
TProfile -> TimeSeries:  NotImplementedError: TH1 to non-Histogram class conversion not fully implemented yet.
```

`root_.py:480`。`from_root` は `is_hist and is_histogram_cls` のときだけ return し、
それ以外は末尾の `raise` に落ちる。dataDisplay の FFT / COHE plot は `TH1` 系で保存される
可能性が高いため、`FrequencySeries` への変換ができないと副経路の中核が成立しない。

docs 側にも既に `docs/web/ja/user_guide/interop.md:172` に「`TH1 -> non-Histogram` は未完」
と記載があり、実測と整合している。

### F3. `TCanvas` / `TFile` は非対応

Status: completed (verified: `python /tmp/verify_root_dtype.py`)

```
TCanvas -> TimeSeries: TypeError: Object Name: c_f3 Title: c_f3 is neither TH1, TH2 nor TGraph
```

`TFile` と `TMultiGraph` は案内メッセージ付きで意図的に拒否している（`root_.py:276-285`、
`"read an individual object from the file first"` / `"convert each TGraph individually"`）。
`TCanvas` は汎用 TypeError。設計会話で指摘された gap の認識は正しい。

### F4. `.ffl` の直接読み出しは gwpy 側に穴がある可能性が高い

Status: planned（**未実測**。以下は静的コードリーディングのみに基づく）

静的コードリーディングによる根拠:

- `gwexpy/io/ffldatafind.py` は `gwpy.io.ffldatafind` の re-export のみ。これは
  **datafind API（`find_urls` 等）であり、`.ffl` パスを `read()` に渡す経路ではない。**
- gwpy `io/utils.py:258` の `file_list()` は `.cache` / `.lcf` / `.ffl` を認識して
  `read_cache()` に回す。
- しかし GWF リーダ本体 `gwpy/timeseries/io/gwf/core.py:144-152` の「read cache file
  up-front」分岐は **`.lcf` と `.cache` のみ**で `.ffl` を含まない。
- その後 `is_cache(source)`（`gwpy/io/cache.py:400`）は str パスも受けて中身をパースするため
  **True を返す** → `find_contiguous("raw.ffl")` → `flatten()`（`gwpy/io/cache.py:562`）が
  文字列を 1 文字ずつ展開してしまう。
- gwpy 本体に `.ffl` を `TimeSeries.read` へ渡すテストは存在しない
  （`gwpy/io/tests/test_cache.py` と `test_ffldatafind.py` のみ）。

GWexpy の `_gwf_io.read_gwf_timeseriesdict` は gwpy の `read_timeseriesdict` を直接呼ぶため、
この挙動をそのまま継承する。

> **注意**: 「GWpy が `.ffl` を読めるので継承クラスでも読める」という前提は、実測で確認する
> まで docs に書かない。確定は Issue B の責務。

### F5. 既存資産と制約

Status: planned（コード・docs の読み取りに基づく事実整理。動作実測は含まない）

- 複数 `.gwf` の list / tuple 読み込み、時刻順マージ、gap / overlap 制御は実装済み
  （`gwexpy/timeseries/_gwf_io.py`、docs `io_formats.md:168`）。
- interop の公開 API は contract test で機械的に縛られている。
  `docs/developers/contracts/public_interop_contract.json` と
  `docs/web/{ja,en}/user_guide/interop.md` の行が一致しないと
  `tests/interop/test_interop_contract.py` / `test_interop_docs_contract_sync.py` が落ちる。
  **公開 interop 関数を追加する作業は必ずこの 3 点セットの更新を含める。**
- conda `gwexpy` 環境には ROOT が入っていない（system python にのみ ROOT 6.36.04）。
  ROOT テストは `pytest.importorskip("ROOT")` で skip される。上記実測は system python で
  実施した。
- `gwexpy/gui/loaders/loaders.py:19` と `gwexpy/gui/README.md:55` は既に `.ffl` を
  サポート形式として宣言している。F4 が未検証のまま残るとこの記述が誇大表示になる。
- Virgo / ffl / dataDisplay / TCanvas に関する既存 issue は open / closed とも無し。

---

## 4. plot type → GWexpy container 対応表（暫定）

Status: planned

> **暫定**。dataDisplay 実サンプルの ROOT class 構成を未確認のため、確定は Issue C 完了後。

| dataDisplay plot | 想定 ROOT object | GWexpy container | 現状 |
|---|---|---|---|
| TIME | `TGraph` / `TGraphErrors` | `TimeSeries` | 実装済み |
| BRMSTIME | `TGraph` / `TH1` | `TimeSeries` | TGraph 経由なら可 |
| FFT | `TH1` / `TGraph` | `FrequencySeries` | **TH1 経路は未実装（F2）** |
| COHE | `TGraph` / `TH1` | `FrequencySeries` | 同上 |
| TRFCT | `TGraph` ペア | `FrequencySeriesDict` 等 | 未検討 |
| 1D-DISTRIB | `TH1` | `Histogram` | 実装済み（ただし F1） |
| FFTTIME | `TH2` | `Spectrogram` | 実装済み（ただし F1） |
| 2D-DISTRIB | `TH2` | 2D histogram 相当 | 未検討 |
| RAWTIME / RAW-IMAGE | 要調査 | 要調査 | 未検討 |

確定に必要な情報（Issue C で収集）:

- ヒストグラムの実クラスが `TH1F` か `TH1D` か（F1 の影響範囲を決める）
- object が `TCanvas` の内部にのみ存在するか、top-level に直置きされるか
- `TCanvas` → `TPad` → primitive の入れ子深さ
- axis title の書式（現行の unit 抽出は正規表現 `\[(.*?)\]` に依存）
- x 軸が GPS 絶対時刻か t0 相対秒か（manual の `export2ascii.C` は GPS と説明）

---

## 5. API 設計案

Status: planned

### 5.1 新規公開 API（案）

```python
def list_root_objects(source) -> list[dict]:
    """ROOT file の read-only インベントリ。

    key 名・ROOT class 名・階層パスを返すのみで、変換は行わない。
    dataDisplay の .root が何を含んでいるかを、変換を試みる前に確認するための入口。
    """

def from_root_canvas(canvas, cls=None) -> dict[str, Any]:
    """TCanvas / TPad を再帰的に辿り、数値 object のみを抽出する。

    cls=None のとき、ROOT class から GWexpy class を推定する。
    plot styling は復元しない。目的は数値ペイロードの回収。
    """
```

### 5.2 設計原則

- **plot styling は復元しない。** 目的は plot に含まれる数値ペイロードを GWpy 互換
  container に戻すことであり、canvas の見た目の再現ではない。
- object の `name` / `title` / axis label は保持する。
- 未知の ROOT metadata は捨てず `meta` に退避する。
- 複数 object を含む場合は dict / list container に格納する。
- 既存 `from_root()` の `TypeError` メッセージを、新 API を案内する文面に更新する。

### 5.3 unit 抽出の制約

現行の unit 抽出は axis title に対する正規表現 `\[(.*?)\]` に依存している
（`root_.py:341-348`, `404-411`, `445-451`）。これは gwexpy 自身の `to_th1d` / `to_tgraph`
が `_get_label()` で `"name [unit]"` 形式を書き出すことを前提とした対称設計であり、
**dataDisplay の axis title がこの規約に従う保証はない。** Issue C で実サンプルの title
書式を確認し、合致しない場合は次のいずれかを選ぶ。

1. unit を `None` のままにし、title 全体を `meta` に退避する
2. dataDisplay 固有の title パターンを追加で認識する（profile 化）
3. 呼び出し側が `unit=` を明示できる引数を追加する

### 5.4 F1 の修正方針（比較）

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| (a) | ROOT class から dtype を判定して `np.frombuffer` に渡す | 最速、コピー無し | ROOT クラス階層に密結合。新しい派生型で再発しうる |
| (b) | `GetBinContent()` ベースの安全な経路にフォールバック | 型に依存せず堅牢 | bin 数が多いと遅い |
| (c) | 入力を `TH1D` / `TH2D` に `Copy` してから読む | 堅牢かつ既存コードの変更が最小 | メモリコピーが増える |

推奨は **(a) を主経路、未知の派生型では (c) にフォールバック**。いずれの案でも
`TProfile` のように `GetArray()` の意味が異なる型は明示的に `TypeError` とする。

---

## 6. `.C` / Cling マクロの方針

Status: planned

ROOT の canvas / histogram は `.C` マクロとして保存できるが、これは data file ではなく
**実行可能な C++ コード**である。自動実行には次の懸念がある。

- 任意コード実行（arbitrary code execution）
- 再現性の欠如
- 外部ライブラリ / 環境依存
- ROOT バージョン依存
- 隠れた副作用

したがって GWexpy core の通常の `.read()` / auto-identify の対象にはしない。
支援する場合の選択肢は次のいずれか。

1. trusted-only / 明示的 opt-in の workflow として隔離する
2. マクロを実行せず、object 定義と数値配列だけを parse する
3. ROOT 側で `.root` object file に変換してから読ませる（推奨の回避策）
4. 支援しない（deferred）

現時点の判断は **4（deferred）**。独立 issue は起票せず、Umbrella の Deferred 節に記録する。

---

## 7. script 生成 HDF5 の方針

Status: planned

§1.1 の「script で抽出して HDF5 に保存する運用」は、dataDisplay の documented native export
ではなく **Virgo user-side の script workflow** と理解する。確認した dataDisplay の
マニュアル類には native HDF5 export schema の記載がない（記載があるのは frame / `.ffl`
入力、ROOT 出力、画像出力、ASCII 抽出、wav 入出力、Cm / shared memory / dataSender）。

したがって現時点の扱い:

```
status: investigation only（representative file / schema 入手待ち）
source: user-side script。dataDisplay の native export ではない
relation to GWDama: 別物（§2.2 参照）
```

schema 入手後に必要な情報: dataset layout、channel 名、epoch / GPS start、
sample rate / time axis、unit、multi-channel 構造、attrs、Virgo 固有 metadata。

入手できても generic HDF5 reader にはせず、まず「Virgo HDF5 profile」として分離する。
unknown attrs は捨てず `meta` に保存し、required / optional field を明確に分ける。
schema version が無ければ GWexpy 側で reader profile version を付ける。

独立 issue は起票せず、Umbrella の Deferred 節に記録する。

---

## 8. Issue 分割と実行順序

Status: planned

| ID | 種別 | 内容 | 優先度 |
|---|---|---|---|
| U | Umbrella | Virgo データ導線の全体管理 | — |
| A | bug | `from_root` が `TH1F`/`TH2F` を float64 として読む（F1） | **P0** |
| B | investigate | `.ffl` を `TimeSeries.read` に直接渡せるか実測（F4） | P1 |
| C | investigate | dataDisplay 実サンプル `.root` の構造インベントリ | P1 |
| D | feat | `TH1` → `TimeSeries`/`FrequencySeries` 変換（F2） | P2 |
| E | feat | `TCanvas`/`TPad` 走査と `TFile` インベントリ（F3） | P2 |
| F | docs | Virgo データアクセスガイド（ja/en） | P3 |

```
Phase 1（依存なし・並行可）
  A  ← Virgo と独立。単独で実装着手可能
  B  ← 合成 fixture のみで完結
  C  ← 実サンプルの持ち込み待ち

Phase 2（Phase 1 の結果待ち）
  D  ← A の dtype 修正が前提
  E  ← C の構造確定が前提
  F  ← B と C の結論が前提
```

### 8.1 各 issue の受け入れ条件（要旨）

**A**: `TH1F` / `TH2F` / `TH1I` の fixture で値が一致する回帰テスト。`TProfile` は明示的に
`TypeError`。gwexpy 内 round-trip だけでは検出できないため、**非 double 型を明示的に
生成するテスト**を必須とする。

**B**: 合成 `.gwf` 2 本 + それを指す合成 `.ffl`（ネスト形式を含む）で、gwpy / gwexpy の
`TimeSeries.read` / `TimeSeriesDict.read` の 4 通り、および `format=` 省略時の挙動を実測。
動く場合は metadata 保持を確認して docs と回帰テストへ。動かない場合は
(i) gwexpy 側で `read_cache()` により path list へ展開、(ii) docs で手動展開を案内、
(iii) upstream gwpy へ issue、の三択を決める。`gwexpy/gui` の `.ffl` 記載の見直しも含む。

**C**: §4 の「確定に必要な情報」をすべて埋め、対応表を暫定から確定へ更新する。
fixture は「実サンプルで構造を確認し、テストは同構造の合成ファイルで行う」を原則とし、
Virgo 固有の channel 名・path をリポジトリにハードコードしない。

**D**: `TH1D` / `TH1F` → `FrequencySeries` / `TimeSeries` の値・xindex・unit・name 保持
テスト、非等間隔 bin ケース、`TProfile` の明示エラー。underflow / overflow は series では
表現できないため、破棄するか `meta` に退避するかを決める。

**E**: contract 3 点セット（`interop/__init__.py` の `__all__`、
`public_interop_contract.json` の `root` エントリ、`interop.md` ja/en の対応行）を同時更新。
`public_interop_contract.md:105` の `implemented_partial` ステータスも見直す。

**F**: 未確定事項を「対応済み」と書かないことが最重要要件。§1.3 の非目標を明記し、
`.root` 経路が加工済み product であることを目立たせる。

---

## 9. docs 構成案（Issue F）

Status: planned

配置: `docs/web/ja/user_guide/virgo_data_access.md` および en 版。
`io_formats.md` / `interop.md` から相互リンクする（既存の切り分け
「ROOT は io_formats では EventTable 直 I/O のみ、object 変換は interop 側」
（`interop.md:213`）を踏襲）。

```
1. スコープ
   - GWexpy は dataDisplay ではない
   - DAQ / real-time control には触らない
   - GWexpy は GWpy 互換 object を受け取る解析レイヤー

2. Virgo のデータ所在
   - frame file: .gwf
   - frame file list: .ffl（/virgoData/ffl の raw / trend / trend100s / rds / spectro）
   - raw / trend / rds の違い

3. GWpy でのフレーム読み出し
   - 単一ファイル
   - ファイル list
   - .ffl（★ Issue B の結論が出るまで書かない）

4. GWexpy container への接続
   - TimeSeries -> GWexpy メソッド
   - TimeSeriesDict -> multi-channel workflow（commissioning の中心例）
   - matrix 変換
   - metadata 保持チェックリスト（t0 / dt / unit / channel / span）

5. dataDisplay .root の再利用
   - ★ 加工済み product であり raw data ではないことを目立たせる
   - 対応表は Issue C 確定後に記載

6. dataDisplay が担うこと
   - channel discovery / 目視確認 / plot 設定 / operational GUI workflow

7. GWexpy がやらないこと（§1.3 の非目標）
```

channel 名の例には次を用いる。

```
V1:ENV_CEB_MIC
V1:Hrec_hoft_20000Hz
V1:Hrec_Range_BNS
V1:ENV_CEB_MIC_rms
V1:ENV_CEB_MIC_50Hz
```

ただし **「Virgo training material 由来の例であり、実際の可用性は GPS 時刻・データ種別・
site 環境に依存する」と明記**する。

---

## 10. 前提と留保

Status: planned

- **実サンプル未参照**: dataDisplay `.root` の実構造（`TH1F` か `TH1D` か、`TCanvas`
  入れ子か）は未確認。実サンプルは別マシンにあり、本設計時点では参照できていない。
  §4 の対応表は暫定であり、Issue C で確定させる。
- **`.ffl` の可否は未確定**: 静的コードリーディングでは gwpy 側に穴がある可能性が高いが、
  実測していない。docs には Issue B の結論が出るまで `.ffl` の使い方を書かない。
- **Issue A は Virgo と独立**: `TH1F` / `TH2F` の破損は既存ユーザーに現時点で影響しうる。
  Virgo 関連 issue の進捗を待たずに実装へ進めてよい。
- **ROOT テスト環境**: conda `gwexpy` 環境に ROOT が無いため、ROOT 関連の回帰テストは
  CI で skip される可能性がある。Issue A の受け入れ条件には、テストが実際に実行される
  環境の明示を含める。

---

## Appendix A. 設計会話の記録（訂正前）

2026-08-01 の設計会話（ユーザー × 外部 LLM、記録は維持者ローカル保管）では、リポジトリ
調査の結果として次のように結論していた。

> **既に standalone `TGraph`, `TH1D`, `TH2D` の object-level import/export は実装されて
> います。** 残っている主要課題は以下です。
> 1. Virgo `.ffl` を GWexpy class に直接渡した実環境確認
> 2. dataDisplay `.root` file の実際の key / canvas / primitive 構造確認
> 3. `TCanvas` / `TPad` traversal helper
> 4. axis title からの unit / semantic inference の検証
> 5. TIME / FFT / BRMS / FFTTIME ごとの class mapping
> 6. dataDisplay sample fixtures と regression tests

同会話の対応状況表では次のように記録されていた。

| 項目 | 会話中の記録 | 実測（§3） |
|---|---|---|
| `TGraph` → series | 実装済み | 実装済み（一致） |
| `TH1D` → Histogram / series | 実装済み | **Histogram のみ。series は `NotImplementedError`（F2）** |
| `TH2D` → Spectrogram | 実装済み | `TH2D` は正常。**`TH2F` は破損（F1）** |
| `TCanvas` traversal | 未実装 | 未実装（一致） |
| `TFile` 自動走査 | 未実装 | 未実装（一致） |
| `.ffl` 専用 parser | 未実装 | 未実装（一致） |
| GWpy backend 経由の `.ffl` | 未検証 | 未検証。**静的には穴がある可能性が高い（F4）** |

この誤差は、会話が GitHub 上のコード検索と commit summary に基づいており、実際にコードを
実行していなかったことによる。本文 §3 は実行結果に基づいて訂正済みである。

## Appendix B. 検証スクリプト

§3 の F1〜F3 は次のスクリプトを `python /tmp/verify_root_dtype.py` として実行した結果である。
PyROOT が必要（conda `gwexpy` 環境には ROOT が無いため system python で実行した）。
`/tmp` 上のファイルは揮発性なので、再現時はここから復元する。

```python
"""Verify F1/F2/F3 of the Virgo dataDisplay interop design doc."""

from __future__ import annotations

import numpy as np
import ROOT

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.histogram import Histogram
from gwexpy.interop import from_root
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries

EXPECTED_1D = [1.0, 2.0, 3.0, 4.0]
EXPECTED_2D = [11.0, 12.0, 21.0, 22.0]


def _fill_1d(hist):
    for i, v in enumerate(EXPECTED_1D):
        hist.SetBinContent(i + 1, v)
    return hist


def _fill_2d(hist):
    for i in (1, 2):
        for j in (1, 2):
            hist.SetBinContent(i, j, float(i * 10 + j))
    return hist


def main() -> None:
    print("== F1: TH1F/TH2F dtype handling ==")
    for cls_name, ctor in (("TH1F", ROOT.TH1F), ("TH1D", ROOT.TH1D)):
        h = _fill_1d(ctor(f"h_{cls_name}", cls_name, 4, 0, 4))
        got = np.asarray(from_root(Histogram, h).values, dtype=float)
        print(f"  {cls_name} -> {got}  correct={np.allclose(got, EXPECTED_1D)}")

    for cls_name, ctor in (("TH2F", ROOT.TH2F), ("TH2D", ROOT.TH2D)):
        h = _fill_2d(ctor(f"h_{cls_name}", cls_name, 2, 0, 2, 2, 0, 2))
        got = np.asarray(from_root(Spectrogram, h).value, dtype=float).ravel()
        print(f"  {cls_name} -> {got}  correct={np.allclose(got, EXPECTED_2D)}")

    print("== F2: TH1 -> non-Histogram class ==")
    hd = _fill_1d(ROOT.TH1D("h_f2", "h_f2", 4, 0, 4))
    for cls in (FrequencySeries, TimeSeries):
        try:
            from_root(cls, hd)
        except Exception as exc:  # noqa: BLE001
            print(f"  TH1D -> {cls.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  TH1D -> {cls.__name__}: converted (no error)")

    prof = ROOT.TProfile("h_prof", "h_prof", 3, 0, 3)
    try:
        from_root(TimeSeries, prof)
    except Exception as exc:  # noqa: BLE001
        print(f"  TProfile -> TimeSeries: {type(exc).__name__}: {exc}")
    else:
        print("  TProfile -> TimeSeries: converted (no error)")

    print("== F3: TCanvas / TFile ==")
    canvas = ROOT.TCanvas("c_f3", "c_f3")
    try:
        from_root(TimeSeries, canvas)
    except Exception as exc:  # noqa: BLE001
        print(f"  TCanvas -> TimeSeries: {type(exc).__name__}: {exc}")
    else:
        print("  TCanvas -> TimeSeries: converted (no error)")


if __name__ == "__main__":
    main()
```

## Appendix C. 参照

- dataDisplay user manual (2022) — frame formatted data の access / management / display、
  Xforms GUI、ROOT 描画、ASCII 抽出、`Read FFL` / `Read Files`、`-o` / `-noplot` オプション
- Virgo training material (2025) — DAQ による frame 収集、`.ffl` による frame file 指定、
  `/virgoData/ffl` の `raw.ffl` / `trend.ffl` / `trend100s.ffl` / `rds.ffl` / `spectro.ffl`、
  channel 命名規約（underscore 区切り、先頭 3 fragment に意味）
- dataDisplay v10r10 update (2021) — 最新 Frame library / Frv library 使用、ROOT v6 未対応
- Virgo 協力者からの私信（2026-07、GWADW2026 発表後フォロー）。原文は維持者ローカル保管
