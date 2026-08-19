# Capability-domain 長期ロードマップ設計(v0.2.0 → v1.0)

> Last-updated: 2026-08-16 (rev 4 — shared-infrastructure triage と release-evidence 補正)

Status: active
Authority: canonical
Audience: maintainer-facing(ビルド済み docs には含まれない — `docs/conf.py` の
`exclude_patterns` が `developers/**` を除外している)

対象: `ROADMAP.md` / `docs_redesign/explanation/roadmap.md` / GitHub milestone・issue triage 規則。
**実装作業は含まない。** 実装(#637 の composition 移行、Field I/O、SegmentTable 等)は別タスクで扱う。

## 1. Goal

外部 AI との議論(2026-08-08〜09、以下「原議論」)で提案された gwexpy の長期構造 —
機能ドメイン分類、v0.2.0 → v1.0 のリリース系列、各ドメインの Minimum/v1.0/Long-term
ゴール、「release statement + headline user stories」方式 — を、3視点レビュー
(アーキテクチャ/批判的/事実監査。物理視点は今回未実施)で検証したうえで、
repo 内の一次文書として固定する。

原議論は repo 外のセッション添付として提供されたため、永続する一次資料としては
扱わない。本文書の §12 Appendix に要旨を転記し、repo 内で自己完結させる。

## 2. Scope / Non-scope

### Scope

1. 本文書(正本)— ドメイン定義・per-domain goals・theme mapping・triage 規則
2. `ROADMAP.md`(正本)— release inclusion scope と Definition of done。本文書は
   その release scope を taxonomy に写像するが、上書きしない
3. 公開 roadmap (`docs_redesign/explanation/roadmap.md`) 同期の方針
4. issue #413(v0.2.0 release notes)は `ROADMAP.md` の release statement / DoD
   から派生させる
5. GitHub milestone/issue/label 同期(Phase 0 scope freeze)の根拠

### Non-scope

- 実装作業全般(#637 の設計検討そのものを含む)
- v0.3.0 以降の milestone 作成・将来 milestone 用 sub-issue の起票
  (`ROADMAP.md` の release policy — 次々minor の milestone は前 minor 出荷まで作らない — に従う)
- v0.1.x patch の triage(#653 が正。本文書はそれを上書きしない)
- レガシー `docs/web` の roadmap 更新(#606 で `docs_redesign` に統合予定)
- PR #488 の merge/close/rebase 判断

## 3. Domain taxonomy(定義)

原議論の最終分類(11 機能ドメイン + 4 横断基盤 + consumer layer)を採用する。ただし
境界判定は原議論の「性格が違う」式の主観的説明ではなく、**機械判定可能な述語**に
書き換える(§9 M1-M4 の指摘への対応)。

| 種別 | # | ドメイン | 主要モジュール対応 | 境界判定 |
|---|---|---|---|---|
| Core data | 1 | Time / Frequency series | `gwexpy/types/` (基底), `gwexpy/timeseries/`, `gwexpy/frequencyseries/`, `gwexpy/spectrogram/` | GWpy から継承した基底型そのものと epoch/dt/df 精度に関わるものはここ。多チャンネル化・matrix 化は 2 へ、I/O は 6 へ |
| Core data | 2 | Multi-channel / Matrix | `gwexpy/types/*matrix*`, `*Dict`, `*List` collections | 1 の型の**コレクション**であり、共通 axes/labels を跨いだ演算・線形代数を持つもの |
| Core data | 3 | Histogram / Distribution | `gwexpy/histogram/` | bin geometry を第一級に持つデータモデル。axis semantics が time/frequency なら 1 または Spectrogram、統計的 binning なら 3 |
| Core data | 4 | Field / Spatial | `gwexpy/fields/` | 空間座標(x/y/z や angle-domain)を第一級に持つデータモデル |
| Core data | 5 | Segment / Experiment Dataset | `gwexpy/table/` (SegmentTable) | 任意の 1〜4 のオブジェクトを cell として保持する実験区間の表 |
| Data access | 6 | Experimental I/O | `gwexpy/io/`, 各パッケージの `io/` サブモジュール | **シグネチャに第三者の in-memory Python オブジェクトが現れない**(バイト列・ファイル・ストリーム → GWexpy オブジェクト)。現れるなら 7 |
| Data access | 7 | Scientific Interoperability | `gwexpy/interop/` | **シグネチャに第三者 in-memory オブジェクトが現れる**(`from_*`/`to_*` によるオブジェクト変換)。同一形式でもファイル読み書きは 6、オブジェクト変換は 7(両方あってよい。例: ROOT ファイル読取は 6、`TH1D` 変換は 7) |
| Analysis | 8 | General Signal / Statistical Analysis | `gwexpy/statistics/`, `gwexpy/signal/` の一部 | **API のシグネチャと docstring を GW 固有の名詞(channel role, injection, coupling, detector geometry)なしで書けるか**。書けるなら 8 |
| Analysis | 9 | Commissioning / Experimental Analysis | `gwexpy/signal/`, `gwexpy/noise/`, `gwexpy/fitting/` の一部 | 上記の逆(GW/検出器固有の語なしに docstring が書けない) |
| Analysis | 10 | Modeling / Forecasting | `gwexpy/noise/` (noise model), 将来の Fisher | **測定データを変換するのではなく、測定されていない値を生成・予測する**(pygwinc カーブ、Fisher forecast)。witness→予測寄与の noise projection は 9/10 の両義があり得るため、個々の issue で明示的に一方へ固定する |
| Presentation | 11 | Scientific Visualization | `gwexpy/plot/` | **描画規約(軸ラベル・単位・スケーリング・レイヤ合成)そのものを変える issue はここ**。各コンテナの `plot()` メソッドは 11 の規約への薄い委譲として扱い、`plot()` に届く metadata の変更は当該データドメイン側に分類する |
| Cross-cutting | X1 | Semantic / Metadata Contract | 全ドメイン横断 | units, axes, dtype, labels, channels, epoch, metadata, arithmetic, ufunc, slicing, copying の契約 |
| Cross-cutting | X2 | Persistence / Provenance / Reproducibility | 全ドメイン横断 | schema version, source files, operation parameters, software version, processing history |
| Cross-cutting | X3 | API Stability / GWpy Compatibility | 全ドメイン横断 | stable/provisional/experimental, deprecation policy, GWpy 追随ポリシー |
| Cross-cutting | X4 | Performance / Scalability | 全ドメイン横断 | parallel, streaming, lazy loading, memory usage, chunking, benchmark |
| Consumer(ドメイン外) | — | GWexpy Studio / pyaggui / CLI / Jupyter | `gwexpy/cli/` 含む | public API のみを消費する側。**未解決事項**: `gwexpy/cli/` は wheel 内かつ `._version` 等の private モジュールを参照しており、「public API のみで完結」という consumer layer の前提を現時点で満たしていない。CLI を core 側の一部として扱うか、private 依存を解消して真の consumer にするかは本文書では未決定とし、Decision log に open item として残す(トラッキング issue: #674、milestone なし) |

**未解決の帰属**: `gwexpy/numerics/`, `gwexpy/detector/`, `gwexpy/astro/`, `gwexpy/utils/`
はいずれの機能ドメインにも一対一で対応しない実装支援コードである。これらは
「ドメインを持たない基盤ユーティリティ」として扱い、triage 時に無理に 1〜11 へ
押し込めない。

**採用前提の検証**: 本分類は 3 回の改訂を経ており(§9 参照)、採用前のストレステストを
経ていない。Phase 0 の wave-1 ラベリング(実 issue 約 50 件への機械的分類)を、
この分類の妥当性検証を兼ねるものとして扱う。単一ドメインに落ちない issue が
目立つ場合、triage rules(§7)を先に見直す。

## 4. Per-domain goals(Minimum / v1.0 / Long-term)

各ドメインの 3 段階ゴールは原議論の記述を踏襲するが、事実誤認が確認された
Histogram(#3)は実装の実態に合わせて修正する。

### 4.1 Domain 1 — Time / Frequency series
- **Minimum**: GWpy から継承した基底型(`TimeSeries`/`FrequencySeries`/`Spectrogram`)の
  epoch・dt/df 精度・単位保持が GWexpy 独自の追加解析・I/O 経路でも壊れない。
- **v1.0**: `_t0_ns` 級の精度追跡(#513)を含め、基底型の semantics が X1 契約の対象として凍結される。
- **Long-term**: GWpy 本体の進化に追随しつつ、GWexpy 固有の実験用途拡張が基底契約を壊さない。

### 4.2 Domain 2 — Multi-channel / Matrix
- **Minimum**: `TimeSeriesDict/List/Matrix` 等の主要 GWpy-style operation がコンテナ全体へ
  適用できる。Matrix は共通 axis・row/column labels・channel metadata・element units・
  四則演算・`@`・transpose/inverse/diagonal を最低限成立させる。
- **v1.0**: 明示的な channel loop なしで `measurement → spectral matrix → MIMO model →
  matrix operation → result` が metadata-aware に完結する。broadcasting・ufunc・
  Quantity・unit propagation・slicing・serialization の semantic contract が固定される。
- **Long-term**: SVD/eigen decomposition、frequency-dependent modal decomposition、
  matrix fitting、MIMO system identification、noise projection、control synthesis
  を第一級機能とする。

### 4.3 Domain 3 — Histogram / Distribution(事実修正版)
現状(v0.1.13 時点)は原議論が想定した「silent downgrade が起きている」状態ではない。
`gwexpy/histogram/histogram.py` は `__array_ufunc__ = None` と `_reject_arithmetic()` に
より **四則演算(`+ - * / **` および reflected/in-place 形)を全面的に `TypeError` で
拒否している**(`CHANGELOG.md` `[0.1.13]` 節に記録、契機は #579・CLOSED 済み)。
したがって v0.2.0 の作業は「契約の明文化」ではなく「不確かさ伝播規則の新規設計」である。

- **Minimum(v0.2.0 で完了する範囲)**: 現状の fail-closed 挙動を #612 の contract matrix に
  明示的 `raises` 行として登録する。bin edges/centers/values/counts/underflow-overflow/
  unit/metadata の所有、`rebin`・基本統計・plotting・serialization・ROOT `TH1D` interop
  (実装済み、`gwexpy/interop/root_.py`)は現状維持。
- **v1.0(将来テーマで設計)**: 演算(`hist * (2*u.s)` 等)を許可するには、まず
  count/weighted/density の区別、per-bin error と covariance の semantics、bin
  非互換時の明示的失敗ルールを設計してから実装する。`normalize()`/`cdf()` の公開 API
  は現時点で存在せず、これも新規設計が必要。
- **Long-term**: `HistogramND`、TH2D の 1D-DISTRIB/2D-DISTRIB 区別(Virgo dataDisplay
  interop 設計書に既出)、covariance-aware fitting との統合。

### 4.4 Domain 4 — Field / Spatial
- **Minimum**: regular-grid spatial data について x/y/z coordinates・spatial unit・
  value unit・component information を保持し、slicing・interpolation・spatial FFT・
  basic plotting・read/write が成立する。
- **v1.0**: `ScalarField → VectorField → TensorField` で共通する semantic contract
  (coordinate system・gradient・divergence・curl・spatial filtering・resampling・
  coordinate transformation)が unit-aware に固定される。座標だけ回して component を
  回さない誤りを許さない(`v' = Rv`, `T' = R T Rᵀ`)。
- **Long-term**: spatio-temporal 解析(`F(x,y,z,t)` / `F(x,y,z,f)`)、
  `sensor array → spatial field → k-space analysis → physical propagation model →
  detector coupling`。

### 4.5 Domain 5 — Segment / Experiment Dataset
- **Minimum**: 1 row = 1 measurement/segment の heterogeneous table として、
  row/column selection・assignment・filter・concat・persistence が成立する。
- **v1.0**: `measurement log → SegmentTable → select → read → compute → store →
  compare → export` が完結する。data + analysis result + experimental condition を
  同一 workflow に保持する。
- **Long-term**: experiment provenance/workflow object 化。source・parameters・
  algorithm・software version・timestamp・dependency を cell/result 単位で追跡できる。

### 4.6 Domain 6 — Experimental I/O
- **Minimum**: 主要な実験形式について `read()`/`write()` を GWpy-style interface に統一し、
  absolute time・sampling rate・unit・channel・instrument metadata を復元する。
- **v1.0**: 「ファイルを読める」ではなく「測定データを意味的に正しく正規化できる」。
  native representation → GWexpy canonical representation の契約を format ごとに持ち、
  canonical HDF5 等は `GWexpy → file → GWexpy` の metadata-preserving round trip を保証する。
- **Long-term**: DAQ・network stream・instrument・database・archive を含む
  experimental data ingestion layer。

### 4.7 Domain 7 — Scientific Interoperability
- **Minimum**: pygwinc・Finesse・ObsPy・python-control・PyTorch 等との converter を提供する。
- **v1.0**: `gwexpy_object → external_object → external analysis → gwexpy_object` の往復で
  axis・unit・channel・labels・metadata を保持する。「converter 数」は KPI にしない。
- **Long-term**: measurement/control/simulation/ML/geophysics/noise model を接続する
  共通 data layer。bridge-not-reimplement 原則(外部ツールを再実装しない、接続する)を
  常に優先する。

### 4.8 Domain 8 — General Signal / Statistical Analysis
- **Minimum**: 頻繁に再実装される汎用解析(moving statistics, interpolation, PCA/ICA,
  Kendall/MIC, AR/ARMA/ARIMA, Hurst, HHT, fitting)を `data.method(...)` として再利用可能にする。
- **v1.0**: `GWexpy object → analysis → GWexpy object/result` の統一 contract。
  SciPy/statsmodels で十分な部分は可能な限り委譲する。
- **Long-term**: 複数 primitive を組み合わせた再利用可能な解析パイプライン。

### 4.9 Domain 9 — Commissioning / Experimental Analysis
- **Minimum**: transfer function・coherence/CSD・Wiener filtering・Bruco・PEM injection・
  noise projection・coupling estimation・modal/control analysis を再利用可能な
  workflow にする。
- **v1.0**: 重力波検出器 commissioning や精密実験で繰り返される解析手順そのものを
  reusable workflow として提供する。
- **Long-term**: `PEM channels → coherence screening → transfer estimation →
  coupling model → noise projection → ranking` のような commissioning workflow を
  primitive の組み合わせとして表現できる。

### 4.10 Domain 10 — Modeling / Forecasting
- **Minimum**: noise model 等、測定ではなく物理モデル/シミュレーションに基づく出力を
  GWexpy の unit/axis/label-aware data model に載せる。
- **v1.0**: 将来の Fisher forecasting 等について、frequency axis・PSD units・
  parameter labels・covariance units を保持した `FisherMatrix` 級の成果物が得られる。
  Fisher 自体は GWexpy の中心ではなく、metadata-aware design の実証例と位置づける。
- **Long-term**: 物理モデル・シミュレーション・noise model・forecast の統合。

### 4.11 Domain 11 — Scientific Visualization
- **Minimum**: 主要 data class に `obj.plot()` を提供し、axis labels・unit・
  time/frequency axis・legend を metadata から自動決定する。
- **v1.0**: single channel・Dict/List・Matrix・Spectrogram・Field・SegmentTable-derived
  result で plotting convention を統一する。plotting API も data API と同様の
  stable contract にする。
- **Long-term**: interactive scientific visualization(linked views、matrix element
  inspection、spatial field slicing、segment selection、interactive channel comparison)。

### 4.12 X1 — Semantic / Metadata Contract
- **Minimum(v0.2.0)**: #612 の宣言的 contract matrix(class × operand × operator ×
  side × in-place)が現行実装に対して green であり、既知の未対応組合せは明示的
  `raises` 行として登録されている(欠落と区別可能)。
- **v1.0**: 全ドメインの演算・スライス・コピーが X1 契約下にあり、モジュール単位の
  stability label(#400、§6 参照)で experimental/provisional/stable が明示される。
- **Long-term**: 新規ドメイン追加時に X1 契約が自動的に要求される開発プロセスが定着する。

### 4.13 X2 — Persistence / Provenance / Reproducibility
- **Minimum**: SegmentTable・Field の on-disk format に `schema_version` と最小限の
  provenance(source files, creation date, gwexpy version)を持たせる。**未知フィールドの
  扱い(保持/破棄/エラー)をプロジェクト単一の規則として、最初のファイルを書く前に決定する**
  (GWDama reader のブロッカーと同根の課題)。
- **v1.0**: 任意の GWexpy object が「どこから来て何をされたか」を追跡できる。
  Monte-Carlo provenance(#508)が copy/slice/serialization を跨いで保持される。
- **Long-term**: 全 I/O 経路で provenance が自動伝播する。

### 4.14 X3 — API Stability / GWpy Compatibility
- **Minimum**: #400 のモジュール単位 stability label(stable/provisional/experimental)を
  定義し、release note で区別する。GWpy-native HDF5 readability golden tests(#402)。
- **v1.0**: 全モジュールが stability label を持ち、deprecation window(リリース数・期間の
  具体的数値)が定義される。Python・依存(numpy/astropy/gwpy)の support window が明文化される。
- **Long-term**: GWpy の pre-release/main に対する定期カナリア CI。GWpy メジャー
  リリース後の追随速度ポリシー。

### 4.15 X4 — Performance / Scalability
- **Minimum(v0.2.0)**: `#637` のデータモデル変更**前**に代表的操作の性能ベースラインを
  取得し、変更後の退行を予算内に収める。v0.2.0 固有の baseline と数値予算は #676、
  共有 benchmark infrastructure は #581 が追跡する。
- **v1.0**: `#580`/`#581` のベンチマーク基盤に基づき、測定駆動で性能改善を優先付けする
  運用が定着する。lazy/distributed execution は「実証された利用要求」が前提。
- **Long-term**: 大規模 SegmentTable ワークフロー(demonstrated usage を前提とした
  lazy/out-of-core)への拡張。

## 5. Domain × release-theme matrix

列はリリーステーマ名(バージョン番号は次の 1 minor のみ確定、それ以降は名前で扱う。
§11 D13 参照)。◎=primary、○=secondary、—=対象外(意図的な選択であり欠落ではない)。
列順は `ROADMAP.md` の Future themes 節の記載順(I/O time and dispatch semantics →
Experiment data workflow → Advanced segment workflows → Spatial geometry and layered
visualization → Mesh-aware fields and solver interoperability → Fisher forecasting and
advanced analysis → API compatibility and stabilization → Later 0.x)に合わせる。

| Domain | v0.2.0 (Container Semantic Contract) | I/O time & dispatch semantics | Experiment data workflow | Advanced segment workflows | Spatial geometry & layered viz | Mesh fields & solver interop | Fisher & advanced analysis | API compatibility & stabilization | Later 0.x (ecosystem/app) | v1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 Time/Frequency | ○ | ○ | — | — | — | — | — | — | — | criteria met |
| 2 Matrix | ◎ | — | — | — | — | — | ○ | ○(#640) | — | criteria met |
| 3 Histogram | ○(raises 登録のみ) | — | — | ○(segment 集約) | ○(plotting) | ○(ROOT interop 強化) | ○(covariance fitting) | — | — | criteria met |
| 4 Field | — | — | ◎ | — | ◎ | ○ | — | — | — | criteria met |
| 5 Segment | — | — | ◎ | ◎ | — | — | — | — | — | criteria met |
| 6 Experimental I/O | ○(golden tests) | ◎ | ○ | — | — | ○ | — | — | ○ | criteria met |
| 7 Interoperability | — | — | — | — | — | ◎ | — | — | ◎ | criteria met |
| 8 General Analysis | — | — | — | — | — | — | ○ | — | — | criteria met |
| 9 Commissioning | — | — | — | — | — | — | — | — | — | criteria met |
| 10 Modeling | — | — | — | — | — | — | ◎ | — | — | criteria met |
| 11 Visualization | — | — | — | ○ | ◎ | — | ○ | — | — | criteria met |
| X1 Semantic contract | ◎ | ◎(明示的 contract 化) | ○ | — | — | — | — | ○ | — | 全域適用 |
| X2 Provenance | ○ | — | ◎ | — | — | — | — | — | — | 全域適用 |
| X3 API stability | ◎ | — | — | — | — | — | — | ◎(#400 継続) | — | 全域適用(stability label 完了) |
| X4 Performance | ◎(ベースライン取得) | — | — | ○(lazy 実証要求) | — | — | — | — | ○ | 全域適用 |

v0.2.0 列は **Committed release** の domain mapping、I/O time & dispatch から
Later 0.x までの future-theme 列は **Directional**、v1.0 列は release theme ではなく
**criteria 列**である(全 domain が Minimum goal を満たし、X1-X4 が全域適用済みで
あることの確認列)。re-scope され得るのは Directional 列であり、Committed scope は
`ROADMAP.md` の release inclusion criteria / DoD に従う。

## 6. Release-theme taxonomy mapping & headline user stories

Release inclusion scope と Definition of done の正本は `ROADMAP.md` であり、本文書の
正本範囲は taxonomy・per-domain goals・theme mapping・triage 規則である。本節は
`ROADMAP.md` の release scope を domain/foundation に写像し、将来テーマの設計根拠を
保持するが、release scope を独立に再定義しない。issue #413 の release statement と
migration text も `ROADMAP.md` から派生させる。各 user story には
**実行可能な受入成果物(named end-to-end test か、グリーンで走る example notebook)**
を紐付ける — 「必要かどうか」の判定を実行可能にするため(§9 M5 への対応)。

### v0.2.0 — Container Semantic Contract

> Every supported operation on a GWexpy container preserves class, unit, axes,
> labels, and metadata, or raises explicitly — never a silent downgrade.

Headline user stories:
1. Container arithmetic follows a declarative, human-reviewed contract matrix
   (class × operand × operator × side × in-place); unsupported cells are explicit
   `raises` entries; the suite asserts its collected-case count.
   **受入成果物**: `tests/types/test_series_matrix_contract_manifest.py`(収集数アサート付き)が green。
2. `np.sqrt(matrix)` is not adopted as a v0.2.0 direct B1 outcome (#637);
   instead, v0.2.0 DoD is satisfied by the documented B0/Phase-A fallback branch
   with explicit failures for unsupported direct ufuncs. Supported operator paths,
   including Quantity-left/right multiplication, retain B0 semantics. B1/
   composition is not adopted for v0.2.0.
   **受入成果物**: `tests/types/test_series_matrix_operator_contract.py` の当該ケースが
   green。B0 契約セルの正本は `tests/types/series_matrix_contract_manifest.py` で、
   その凍結は `tests/types/test_series_matrix_contract_manifest.py` が検証する。
   fallback を選択したのは D21(D10 の decision-date 機構は superseded)。
3. HDF5 written by GWexpy for GWpy-derived containers is readable by a
   GWpy-only process (no `import gwexpy`).
   **受入成果物**: `tests/io/test_hdf5_gwpy_compat.py`(新設)が gwpy-only subprocess で green。

Definition of done の規範的文面は `ROADMAP.md` を参照。本節の taxonomy mapping では、
上記3成果物に加え、#676 が追跡するリファクタ前の性能ベースラインと数値化された
退行予算を X4 gate として扱う。#581 はその測定に利用する共有 benchmark infrastructure
であり、v0.2.0 milestone member ではない。

### I/O time and dispatch semantics(将来テーマ、domain: 6 中心)

> Consistent time interpretation (time zones, numeric time scales and epochs) and
> uniform reader behaviour across supported experiment data formats.

Primary domains: 6, 1, X1。`ROADMAP.md` の記載(2026-08-15 確定)からの逐語転記:

- **Track A — Time interpretation contract**(#632, #634, #636)。Headline user
  story: a user reading data from any supported format gets a value with an
  explicit, documented time reference, never a silently-assumed one. Affected
  readers, and writers where applicable, interpret time through an explicit
  timezone / scale / unit / epoch contract; the implicit legacy GPS-seconds
  interpretation is deprecated.
  **受入成果物**: a cross-format time-interpretation conformance matrix covering
  #632, #634, and #636, including explicit-zone/scale cases and required
  fail-closed cases.
  **Non-goals**: no unrelated format expansion, and no changes to existing
  on-disk time encodings.
- **Track B — Dispatch / reader semantics**(#444 → #616)。Headline user story: a
  user reading across multiple backends gets identical collection-fallback and
  gap/pad behaviour regardless of which reader served the request. First decides
  the collection fallback registry contract for `FrequencySeriesDict`/`List`/
  `Matrix` `.read()`/`.write()` — whether to keep the Astropy registry or
  converge on the GWpy default registry (#444) — then makes gap/pad behaviour
  consistent across supported `TimeSeriesDict.read` backends (#616).
  **受入成果物**: (a) a `FrequencySeriesDict`/`List`/`Matrix` collection
  dispatch/reachability matrix covering the chosen registry, and (b) a
  cross-backend `TimeSeriesDict.read` gap/pad test matrix.
  **Non-goals**: this track does not implement a new registry mechanism, add
  new file formats, or change unrelated backend behaviour.

両 Track は独立に完了する。テーマ完了 = 両 Track が green。A→B の強制依存はない
(Track B の実装順序が #444 → #616 なのは Track B 内部の依存であり、Track A との
依存ではない)。

### Experiment data workflow(次テーマ、番号未確定)

> Read, transform, and persist spatial Field data and per-segment experiment
> records through GWpy-style APIs, without escaping to pandas for metadata-bearing state.

Primary domains: 5, 4, X2。

Headline user stories:
1. A user can `ScalarField.read(...)`/`.write(...)` GSI DEM and GeoTIFF data with
   geospatial metadata round-tripping through canonical HDF5.
   **受入成果物**: `#547`/`#551` の round-trip テスト。
2. A user can complete `scan/read → filter/select → assign → row/column operations
   → persist(HDF5, schema_version + provenance)→ resume` for a SegmentTable
   without leaving GWexpy objects.
   **受入成果物**: #592 の critical path 完了を示す統合テスト。
3. On-disk schema versioning and the unknown-field policy are defined once and
   applied to both Field and SegmentTable persistence.
   **受入成果物**: 設計 issue(§8 参照)のクローズ + 両フォーマットでの適用確認。

### Advanced segment workflows(将来テーマ)

> Aggregate and reduce across many experiment segments without hand-written
> nested loops or an unmeasured memory ceiling.

Primary domains: 5, 3, X4。

Headline user stories:
- A user can `groupby`/`aggregate` across segments into summary Histograms or tables.
- A user can process a SegmentTable larger than memory via a lazy `SegmentFrame`,
  **only once a demonstrated usage requirement exists**(X4 Minimum の前提を継承。
  「数万 segment」のような未検証の数値目標は置かない)。

Design groundwork: [SegmentTable workflow plan](../plans/active/2026-08-01-segmenttable-workflow-design.md)。

### Spatial geometry & layered visualization(将来テーマ)

> Compose terrain, physical fields, and detector geometry into one correctly
> rotated, correctly labeled map.

Primary domains: 4, 11, X1。

Headline user stories:
- A user can rotate a Field into a detector frame with vector/tensor components
  transformed correctly (`v' = Rv`, not just the coordinates).
- A user can overlay DEM terrain, a scalar/vector field, and detector markers on
  one figure with a defined z-order.

制約: `gwexpy/fields/` 変更は physics-reviewer 必須(プロジェクト規約)。担い手が
確保できない場合、このテーマの着手を遅らせる。

Design groundwork: [layered visualization plan](../plans/active/2026-08-01-layered-visualization-design.md)。

### Mesh fields & solver interoperability(将来テーマ)

> Bring simulation output (OpenFOAM, FLOW-3D, SPECFEM3D, SimPEG, …) into the
> same metadata-aware workflow as measured Field data, via explicit interpolation.

Primary domains: 4, 7。

設計判断として明記: mesh topology(vertices/cells/association)の独自実装より先に、
`meshio`/`PyVista` 等の既存ライブラリへの委譲可否を評価する(bridge-not-reimplement
原則、§9 M10 の指摘への対応)。独自 `MeshField` を作るのは、既存ライブラリで
unit/axis/metadata 保持が実現できないと確認された場合に限る。

Headline user stories:
- A user can read a solver output format into a mesh-aware field and explicitly
  interpolate it onto a regular Field for existing GWexpy analysis.

### Fisher forecasting & advanced analysis(将来テーマ)

> Perform labeled, metadata-aware advanced inference/forecasting calculations.

Primary domains: 10, 2。

Headline user stories:
- A user can compute a `FisherMatrix` from spectral models with frequency axis,
  PSD units, and parameter labels intact.

制約: overlap reduction function は physics review gate(`ROADMAP.md` 既定)。
review の担い手が確保できるまで着手しない。

### API compatibility and stabilization(将来テーマ、foundation: X3)

> Continuation of the v0.2.0 "API stability labelling" workstream (#400) that
> establishes and documents the public API surface.

Primary domains: X3, 2。X3/#400 を起点とする**継続テーマ**であり、v0.2.0 の
API stability labelling ワークストリームの再記述ではない(重複ホーム化の防止、
§9 M4/Critic 参照)。`ROADMAP.md` の記載(2026-08-15 確定)からの逐語転記:

Covers auditing GWpy method overrides against the documented compatibility
principle (#639), and #640, which splits into a behavioural-contract issue for
the `ignore_nan` default mismatch between `TimeSeriesMatrix` and
`FrequencySeriesMatrix`, and a documentation-only issue for the
`TimeSeriesDict.append` docstring gap.
**受入成果物**: published compatibility matrix and per-module behaviour
contract document.
**Non-goals**: this theme does not redesign any container's arithmetic
behaviour — see the v0.2.0 theme above for that contract.

### Later 0.x — Ecosystem and application readiness(番号を持たない)

Ecosystem backlog(domain 7 所属項目)の成熟したものを minor に昇格させる段階。
headless application contract(GWexpy Studio/pyaggui/CLI が必要とする source
inspection・format capability introspection・serializable operation parameters・
provenance)と、X4 の測定駆動の性能改善を含む。番号と個数はその時点の release
policy に従って決める。

### v1.0 — Public contract stabilization

§11 D11 の裁定に従い、**主経路はモジュール単位の stability label(#400)の段階昇格**
(experimental → provisional → stable)であり、v1.0 は最後のコア モジュールが
stable になった時点の宣言とする。全 domain 一括の "public contract stabilization"
宣言は補助的な位置づけとし、モジュール単位の進捗が実質的な進捗指標になる。

## 7. Issue triage rules

### ドメイン判定の決定木

1. issue が直すか増やすのは、データ型そのものか(1-5)、読み書きか(6-7)、
   解析手法か(8-10)、描画か(11)、複数ドメインに同時に効く不変条件か(X1-X4)。
2. 6 vs 7: シグネチャに第三者 in-memory オブジェクトが現れるか(§3 参照)。
3. 8 vs 9: GW/検出器固有の名詞なしで docstring が書けるか。
4. 10: 測定を変換するのでなく未測定値を生成するか。
5. 11: 「図のあるべき姿」を変えるか(変えるなら 11、`plot()` に届く metadata を
   変えるだけならデータドメイン側)。
6. tie-break: 最も特化した core domain を主、X を副として両方付与してよい
   (ラベルは複数付与可)。

### milestone 判定

「候補テーマの headline user story のうち、少なくとも 1 つの**受入成果物**が
この issue なしでは green にならないか?」— Yes なら当該テーマの milestone 候補、
No なら backlog。判定対象は user story の文言ではなく §6 で紐付けた受入成果物
そのものであること(文言だけでの判定は恣意性が残るため、§9 M1/M5 の指摘に基づき
成果物ベースに限定する)。

**共有インフラ例外**: 複数テーマで再利用する infrastructure prerequisite は、
(1) release 固有の受入 evidence を所有する milestone member が別に存在し、
(2) その member と prerequisite の依存が `ROADMAP.md` / decision record / issue の
いずれかで明示される場合に限り、milestone 外に置いてよい。v0.2.0 では #676 が
pre-#637 baseline と numeric regression budget を所有し、#581 は共有 benchmark
infrastructure として milestone 外に置く。この例外は release 固有の成果物に owner が
いない状態を許可するものではない。

**降格規則**: リリースが遅延した場合、いずれの headline user story の受入成果物にも
不要な項目から milestone を外す。この規則自体は各 milestone 説明文に明記する。

### ラベル運用

`domain:*` 13 個(§3 の 1〜11 の英語 slug)+ `perf`(X4)+ `provenance`(X2)を新設し、
既存 `contract`(X1)/`compatibility`・`gwpy4`(X3)は description のみ更新する。
実行は Phase 0 の別承認事項とする(本文書は提案のみ)。

## 8. 既存文書との整合表

| 既存文書 | 関係 |
|---|---|
| `docs/developers/plans/active/2026-08-01-roadmap-reorganization-plan.md` | 前身の再編計画。completed のまま不変更。本文書はその後継として v0.3.0 以降・v1.0 の詳細を補う |
| `2026-07-31-terrain-scalarfield-io-design.md` | Domain 4 Minimum の一次設計(Experiment data workflow テーマで参照継続) |
| `2026-08-01-segmenttable-workflow-design.md` | Domain 5 の一次設計(v0.2.0 から Experiment data workflow テーマへ移動後も参照継続) |
| `2026-08-01-layered-visualization-design.md` | Domain 4+11 の一次設計 |
| `2026-08-01-ecosystem-interop-plan.md` | Domain 7 backlog の一次設計。優先順位は非公開のまま維持 |
| `2026-08-01-virgo-datadisplay-interop-design.md` | Domain 3(TH2D 区別)・Domain 7(ROOT/Virgo)の一次設計 |
| GWDama reader の unknown-attributes ブロッカー(`ROADMAP.md` Ecosystem Backlog) | X2 Minimum の schema-policy 設計 issue と同根の課題として関連付ける |
| `docs_redesign/explanation/roadmap.md` | 公開向け要約。バージョン番号・ドメイン番号を出さない制約下で本文書の内容を反映(D3) |
| issue #413 | `ROADMAP.md` の v0.2.0 release statement / DoD から派生する release notes と migration guide |
| issue #400 | X3 Minimum のモジュール単位 stability label。v1.0 の主経路(D11) |
| `ROADMAP.md` `## v0.1.14 — I/O contract and maintenance hardening` 節 | v0.1.14 は 2026-08-15 に released 済み(D16)。本節は per-change 正本 `CHANGELOG.md` `[0.1.14]` の要約であり、#632/#634 の partial mitigation を明記した上で、両issue の現在の帰属先は「I/O time and dispatch semantics」テーマ(本文書 §6、Directional、milestone 未設定)であると上書き宣言している |

## 9. レビュー所見の採用/棄却判定表

2026-08-09 に実施した 3 視点レビュー(アーキテクチャ/批判的/事実監査)の重大・中
所見を、本設計への反映状況で記録する。**物理・統計視点のレビューはユーザー判断で
未実施**であり、Fisher の式・MIMO 行列演算・座標回転の物理的妥当性そのものの
検証は本文書の範囲外として残る(将来の懸念事項)。

| 所見 ID(出典) | 概要 | 判定 |
|---|---|---|
| C1(architect) | contract 凍結リリースと #637 データモデル大改造の同居は矛盾 | **採用**: v0.2.0 を二段分割(D9)、#637 に期日+fallback(D10) |
| C2(architect) | v0.2.0 が過大、繰り越し禁止ポリシーと衝突 | **採用**: 二段分割(D9)で対応 |
| C3(architect) | X1-X4 にリリース行がない | **採用**: §5 matrix に X1-X4 行を追加 |
| C4(architect) | 永続フォーマットの schema/未知フィールド方針が未決 | **採用**: X2 Minimum に明記、設計 issue 起票(§8) |
| M1(architect)/M5(architect) | user story 必要性の「機械判定」が成立しない | **採用**: §7 で受入成果物ベースの判定に変更 |
| M9(architect) | 性能目標が定量化されていない | **部分採用**: v0.2.0 に性能ベースライン取得を追加(X4 Minimum)。具体的な退行予算数値は実装時に決定(未決事項として残す) |
| M10(architect) | contract test matrix の維持コスト・0 件収集の罠 | **採用**: §6 v0.2.0 story 1 に収集数アサートを明記 |
| M11(architect) | GUI/docs 工事を v0.2.0 に混入 | **採用**: milestone 外の独立作業に分離(D9) |
| M7(architect) | 依存グラフが実装の層構造と不一致 | **採用**: §3 の境界判定を実装構造(`gwexpy/types/` が最下層)に合わせて記述 |
| M12(architect) | CLI が consumer layer 分類と実装で矛盾 | **採用(open item化)**: §3 の consumer layer 行に未解決事項として明記。解消は本文書の範囲外 |
| R1(Critic) | #637 は一度失敗済みの未解決課題。fallback なしの blocker は危険 | **採用**: D10(期日+fallback) |
| R2(Critic) | v0.2.0 スコープが単調増加、削減提案ゼロ | **採用**: 二段分割で削減 |
| R3(Critic) | DoD の大半が検証不能、一部自己矛盾(数万 segment vs lazy 除外) | **採用**: §6 で受入成果物化。「数万 segment」の数値目標は削除(Advanced segment workflows テーマで「demonstrated usage requirement」を前提条件化) |
| R4(Critic) | 需要のエビデンスがない(star 0, 外部 contributor 0) | **棄却(記録のみ)**: 単一メンテナ+KAGRA内という前提を変えられないため、需要実証の追加は本文書の範囲外。ただし `ROADMAP.md` の既存の但し書き(優先順位は demand の測定ではない)を維持・強調する |
| R5(Critic) | v1.0 一括宣言は早すぎる。#400 モジュール単位 label が既存の代替 | **採用**: D11(stability label を主経路に) |
| R6(Critic) | 「v0.2.x はデータモデルを大きく壊さない」と #637 blocker の自己矛盾 | **採用**: 二段分割によりv0.2.0からモデル変更を分離することで解消 |
| R7(Critic) | 将来テーマに確定番号を振り、milestone 事前作成禁止ポリシーと矛盾 | **採用**: D13(名前ベースの並び) |
| M2(Critic) | #592 の7段直列 critical path が単一メンテナに厳しい | **棄却(記録のみ)**: 再分割案は実装計画の領域であり本文書の範囲外。Experiment data workflow テーマの着手時に見直す前提を注記するに留める |
| M4(Critic) | physics review の担い手が不在なのに v0.4-v0.6 が依存 | **採用**: §6 の該当テーマに制約として明記(担い手不在なら着手遅延) |
| M6(Critic) | PR #488 が長期 conflict のまま放置、Studio 前提が未検証 | **部分採用**: Phase 0 で #488/#645 の相互リンクのみ実施(merge判断は範囲外)。Studio 前提の検証は本文書の範囲外 |
| M10(Critic) | mesh field が bridge-not-reimplement 原則に反する可能性 | **採用**: §6 Mesh fields テーマに meshio/PyVista 委譲検討を明記 |
| M11(Critic) | リスク登録簿・撤退条件が皆無 | **部分採用**: #637 の fallback(D10)、Advanced segment workflows の usage requirement 前提化。GWpy 5.0 対応・bus factor 等その他のリスクは本文書の範囲外 |
| A-8(Auditor) | Histogram の v0.2.0 DoD が事実と逆(既に fail-closed 済み) | **採用**: §4.3 で全面書き換え |
| B-1〜B-3(Auditor) | ドメイン番号スキームが3種並存、v0.2.0スコープ記述に不一致 | **採用**: taxonomy / triage は本文書の11+4分類、release inclusion / DoD は `ROADMAP.md` を正本とする |
| B-4(Auditor) | v0.1.14 の結論が #653 で陳腐化 | **採用(2026-08-09 時点)、後に期限切れ**: 当初は「本文書・ROADMAP 改訂ともに v0.1.14 に言及しない」と採用したが、v0.1.14 は 2026-08-15 に実際に released され、`ROADMAP.md` は released セクションとして v0.1.13 と対称な形式で v0.1.14 を記載する方針に変更された(D16)。本裁定はその時点までの暫定判断であり、D16 が優先する |
| 項目4(Auditor 事実照合) | #555 critical path 前段が milestone 未設定 | **採用**: Experiment data workflow テーマ開始時に一括割当する運用へ(§8 の関連付け経由でPhase 0 issue に反映) |
| 項目12(Auditor) | GWADW 資料の page 引用が repo 内に実体なく検証不能 | **採用**: 本文書では GWADW page 引用を使わず、repo 内の一次資料(実装・既存 issue・既存設計文書)のみを根拠とする |

## 10. Verification checklist

- [ ] `ROADMAP.md` から本文書への相対リンクが実在する
- [ ] `ROADMAP.md` に "Five workstreams" のような数詞と実項目数の不一致がない
- [ ] `docs_redesign/explanation/roadmap.md` にバージョン番号・ドメイン番号が含まれない
- [ ] `conda run -n gwexpy python -m pytest tests/docs/ tests/test_issue_burn_down_roadmap.py tests/test_conda_forge_roadmap.py -q` が green
- [ ] `sphinx-build -b html` (en/ja) が warning なく通る
- [ ] `msgfmt --check` が roadmap.po に対して通る
- [ ] `gh issue list --milestone v0.2.0 --limit 200` が §8 の期待構成(contract-only、約10件)と一致する

上記が実行され結果が確認された時点で、ヘッダに
`Initial adoption verification: completed <date>` を追記する。living canonical design
としての `Status: active` は維持する。

## 11. Decision log

| ID | 決定 |
|---|---|
| D1 | ドメイン全文は `ROADMAP.md` でなく本文書に置く。ROADMAP は要約+リンクのみ |
| D2 | v0.2.0 milestone の名称・番号は変更しない(scope のみ contract-only に narrowing) |
| D3 | 公開 roadmap にドメイン番号・体系語彙を出さない |
| D4 | #413 は body 全面編集+履歴コメントで旧文言を保全する方式 |
| D9 | v0.2.0 を二段分割(Container Semantic Contract / Experiment data workflow)。GUI・docs工事は milestone 外 |
| D10 | #637 は decision-date method + fallback(`__array_ufunc__=None` + documented limitation)付き blocker。calendar date は TBD で、#675 が milestone mid-point で確定する **(2026-08-18: D21 により superseded。本行は歴史的記録として保持し、書き換えない)** |
| D11 | v1.0 は #400 モジュール単位 stability label を主経路とする criteria 節。一括宣言は補助 |
| D12 | Histogram の v0.2.0 スコープは raises 行の登録までとし、伝播規則設計は将来テーマ |
| D13 | Future themes は番号でなく名前で並べる。次の 1 minor のみ番号確定 |
| D14 | 2026-08-12 の追加議論を受け、Future themes に「I/O time and dispatch semantics」(#632/#634/#636 の Track A + #444→#616 の Track B)と「API compatibility and stabilization」(#400 継続、#639/#640)の2テーマを新設(§6)。前者は v0.1.14 が意図的に残した時刻解釈と reader dispatch の空白を埋め、後者は X3/#400 を起点とする継続テーマとして #639/#640 の宙に浮いた帰属を解消する。§5 matrix に両テーマの列を追加 |
| D15 | リリーステーマの状態語彙を `Committed`(v0.2.0 のみ、常に1つ)/ `Directional`(Future themes 全テーマ、version・date・scope の commitment なし)/ `Backlog`(Ecosystem & Interoperability 節のみ)の3語に固定。表示形式は見出し直下の `Status: <語>` 1行。released 節・Release policy・v1.0 criteria・Engineering hygiene は対象外(未分類の曖昧さではなく、本スキームの適用外と明記) |
| D16 | `ROADMAP.md` に v0.1.13 と対称な形式で `## v0.1.14 — I/O contract and maintenance hardening (released 2026-08-15)` 節を追加。CHANGELOG `[0.1.14]` の「#634 for v0.2.0」という当時の見込みを上書きし、#632/#634(partial mitigation)の現在の帰属は「I/O time and dispatch semantics」テーマ(Directional、milestone 未設定)であると明記。B-4(§9)の当初裁定はこれにより期限切れとなり、本決定が優先する |
| D17 | 本文書を active-plans 領域から安定した canonical の `docs/developers/design/` パスへ `git mv` により昇格。`Status: active` / `Authority: canonical` / `Audience: maintainer-facing`。日付付きパスへの archive コピーは作らない |
| D18 | **v0.2.0 milestone hygiene の planned application record**(merge 前に本 PR 内で確定。Step 11 は本表を再分類せず GitHub milestone にそのまま適用するのみ)。判定基準は `ROADMAP.md` `## v0.2.0 — Container Semantic Contract` 節の Definition of done(4項目、2026-08-15 rev 時点)。現在の milestone 実メンバー11件(issue 10件 + PR 1件)を下記 D18 詳細節の3分類に判定する |
| D19 | **Post-adoption planning-gate correction**(2026-08-16 Sol completion review)。#675 と #676 を v0.2.0 の `gate-supporting` member として追加する。前者は #637 の fallback を実行可能にする decision-date gate、後者は DoD 4 の baseline / regression-budget evidence を所有する。#581 は #676 が利用する共有 benchmark infrastructure であり milestone member にはしない。GitHub milestone への適用は本 follow-up の merge 後に行い、再分類しない |
| D20 | **#413 release-evidence correction**(2026-08-16 final Sol audit)。#413 は `gate-supporting` の release-notes / migration-guide owner として v0.2.0 milestone に残すが、issue body の fixed calendar-date 表現と `nproc`→`parallel` migration 項目は現行 scope を上書きできない。closeout merge 後、body を書き換えず superseding comment を投稿する。calendar date は TBD で method / tracking owner は #675、#403 migration は D18 により v0.2.0 外。分類・milestone membership は変更しない |
| D21 | **v0.2.0 SeriesMatrix fallback selection**(2026-08-18 maintainer ruling、#637)。v0.2.0 の outcome として **B0 / Phase-A fallback** を正式契約に採用し、composition/B1 は v0.2.0 では**採用しない**。B0 契約で明示的に支持されない direct NumPy ufunc は documented provisional limitation であり v0.2.0 の correctness defect ではない(supported な操作は class/units/axes/labels/metadata を保持し、unsupported は明示的に失敗する。bare ndarray / Quantity への silent downgrade は許容しない)。`__array_ufunc__ = None` は fail-closed 挙動の実装手段であって長期互換性保証ではなく、外部から観測可能な B0 契約が保証対象である。これにより **D10 の decision-date 機構(#675 が milestone mid-point で確定)は期限切れとなり、本決定が優先する**(D16 が §9 B-4 を上書きしたのと同じ扱い)。B1 再設計は future design work として open のまま残り、version / date の commitment を持たない。`adopted: false` の位置づけは下記 D21 詳細を参照。本 ruling は merge / tag / publication / v0.2.0 release を承認するものではない |
| Open-1 | CLI の consumer layer 分類と private API 依存の矛盾は未解決(§3)。トラッキング issue #674(milestone なし、architecture decision) |
| Open-3 | X4 の性能退行予算の具体的数値は未確定。#676 が v0.2.0 固有の baseline / budget を追跡する gate で、#581 は共有 infrastructure(D19) |

### D18 詳細 — v0.2.0 milestone 実メンバー11件の分類(2026-08-15、DoD 基準で確定)

判定基準: DoD 1(#612 contract matrix)/ DoD 2・4(#637 の正しさとperf regression budget)/ DoD 3(#402 HDF5 golden tests)。`DoD-required` = DoD 文面が当該 issue を名指しする。`gate-supporting` = DoD に直接名指しされないが、`ROADMAP.md` の v0.2.0 Workstreams が scope 内として明記する、または DoD の検証・release evidence に必要である。`unrelated` = 上記いずれにも該当しない。

Post-adoption note: #581 は D18 時点でも milestone member ではない。D19 で release 固有の
evidence owner #676 を `gate-supporting` として追加し、§7 の共有インフラ例外に基づいて
#581 を明示的 dependency / milestone 外の shared infrastructure として固定した。

| Issue/PR | タイトル要旨(2026-08-15 実測) | 分類 | 根拠 |
|---|---|---|---|
| #612 | Container arithmetic contract matrix umbrella | **DoD-required** | DoD 1 が直接名指し |
| #637 | SeriesMatrix composition redesign | **DoD-required** | DoD 2(正しさ)・DoD 4(性能予算)の両方が直接名指し |
| #402 | GWpy-native HDF5 readability golden tests | **DoD-required** | DoD 3 が直接名指し |
| #400 | Define v0.2.0 API stability labels and release policy | **gate-supporting** | DoD には名指しされないが、Workstreams が明記する通り「#612 の contract matrix がその状態を記録する語彙」を提供する。#400 なしでは DoD 1 の各エントリに付す stability label が未定義のまま残る |
| #413 | Prepare v0.2.0 release notes and migration guide | **gate-supporting** | `ROADMAP.md` Release policy「Documentation ... is part of each feature's definition of done」に基づく。#413 は DoD 1-3(#612/#637/#402)の成果をユーザー向けに記録する成果物であり、DoD 番号付き項目そのものではないが release の完了要件として扱う。body 内の stale scope は分類を変えず D20 の comment で補正する |
| #401 | Gate v0.2.0 release on contract audit wave #276/#278/#284/#286/#288 | **gate-supporting(要更新)** | audit dependency mapping のうち「HDF5 compatibility golden tests → #276/#278」は DoD 3(#402)の検証を裏付けるため gate-supporting と判定する。ただし同 issue の依存表は SegmentTable read/write・coupling segment schema・`method="median-mean"` など、Step 3 の二段分割(D9)で v0.2.0 の Non-goals/Bounded additions に移動済みの項目を今も P0 として記載しており陳腐化している。Step 11 適用時、当該 issue 本文の audit mapping 表を現行 DoD に合わせて更新するコメントを併せて投稿する(分類自体は変更しない) |
| #594 | investigate(io): Virgo `.ffl` を `TimeSeries.read` に直接渡せるか実測して確定する | **unrelated** | v0.2.0 DoD の対象は container 演算契約であり、Virgo `.ffl` I/O(domain 7)とは無関係。v0.1.13 で行ったのは「`.ffl` is unsupported」という文書訂正であり、#594 は open の follow-up investigation として残った。Ecosystem & Interoperability Backlog 項目3(Virgo data-path completion)への再帰属を推奨 |
| PR #625 | fix(io): let TimeSeries.read accept Virgo .ffl frame lists (#594) | **unrelated** | #594 と同一の理由。Ecosystem & Interoperability Backlog 項目3への再帰属を推奨 |
| #403 | Document deprecation policy: `nproc` → `parallel` alias for v0.2.0 | **unrelated** | DoD のいずれの項目の検証にも必要なく、v0.2.0 Workstreams も scope として明記しない。`nproc`/`parallel` の実挙動は v0.1.13 で #588 として fail-closed 化済みであり、本 issue は残る deprecation 通知文書化のみが scope。Step 11 では「v0.2.0 の技術的 DoD には不要」という理由のみ付記する |
| #508 | statistics: Monte Carlo provenance metadata doesn't survive Spectrogram copy/slice/serialization | **gate-supporting** | DoD の直接要件ではないが、`ROADMAP.md` の v0.2.0 Workstreams が「Carried-over reproducibility work」として明示的に scope 内へ置く。container contract の変更後も copy/slice/serialization で provenance を保持できることを release evidence として確認する |
| #513 | statistics/interop: `_t0_ns` precision(deferred from v0.1.11 review) | **gate-supporting** | issue 全体は warning stacklevel、`rng` Protocol typing、bool-input guard、`_t0_ns` precision の4項目を束ねているが、v0.2.0 の carried scope は `ROADMAP.md` Workstreams が明示する `_t0_ns` 精度部分のみ。container contract の変更が高精度 epoch の保持を後退させないことを release evidence として確認する |

**適用時の注意**(Step 11 向け): `unrelated` の除去では、当該 issue に上記根拠を要約した1行コメントを付け、追跡リンク喪失を防ぐ(#594/PR#625 → Ecosystem Backlog 項目3、#403 → 明示的な再帰属先なしのため milestone 除去のみ)。#401 は gate-supporting として残し、旧 P0 dependency mapping が歴史的であり Container Semantic Contract DoD に置き換えられたことをコメントする。#513 にも、4項目のうち v0.2.0 の carried scope は `_t0_ns` 精度部分のみであり、他の3項目を同 release scope へ昇格させるものではないことをコメントする。milestone description には「基準: `docs/developers/design/capability-domain-roadmap.md` D18 参照」の1行を追記する。適用前に、上記 DoD 引用が merge 後も本 D18 確定時点(2026-08-15)から変わっていないことを diff で確認する。

### D19 詳細 — post-adoption planning gates(2026-08-16、Sol review 対応)

| Issue | 分類 | merge 後の適用 | 根拠 |
|---|---|---|---|
| #675 | **gate-supporting(add)** | v0.2.0 milestone に追加し、#637 の decision-date gate を追跡する旨をコメント | #637 の calendar date 自体は TBD だが、milestone mid-point で日付を確定し fallback を発動可能にする method は確定済み。DoD 2 の fallback を運用可能にする planning gate であり、composition 実装そのものではない |
| #676 | **gate-supporting(add)** | v0.2.0 milestone に追加し、release 固有の baseline / numeric budget を追跡する旨をコメント | `ROADMAP.md` Workstreams が明示する v0.2.0 固有の pre-#637 baseline と regression budget を所有し、DoD 4 の release evidence を提供する。§7 の共有インフラ例外により、#581 は明示的に依存する共有 benchmark infrastructure として milestone 外に置く |

この追加は D18 の既存 8 members の分類を変更しない。merge 後は上表を GitHub milestone
へ機械的に適用し、milestone description に D18 と D19 の両方を参照させる。

### D20 詳細 — #413 release-evidence text correction

| Issue | 分類 | closeout merge 後の適用 | 根拠 |
|---|---|---|---|
| #413 | **gate-supporting(retain; text correction)** | issue body は歴史記録として保持し、superseding comment を1件投稿する | #413 は DoD 成果をユーザー向け release notes / migration guide に変換する evidence owner であり milestone に残す。一方、body の「fixed decision date」は D10/D19 後の現状と異なり、calendar date は TBD、method / tracker は #675 である。また `nproc`→`parallel` migration は D18 で unrelated とした #403 の scope であり、v0.2.0 release-evidence gate から除外する。この補正は #413 の分類や milestone membership を変更しない |

### D21 詳細 — v0.2.0 SeriesMatrix fallback selection(2026-08-18)

出典: #637 の maintainer ruling(2026-08-18T10:10:23Z)。

#### 決定内容

| 項目 | 決定 |
|---|---|
| v0.2.0 の outcome | **B0 / Phase-A fallback** を正式契約として採用する |
| composition / B1 | v0.2.0 では**採用しない** |
| #675 decision-date 機構 | 本決定により **superseded**。calendar decision date を遡って作る必要はない |
| B1 再設計 | future design work。**version / date の commitment を持たない** |

v0.2.0 の public contract は `tests/types/series_matrix_contract_manifest.py` と
`docs/developers/contracts/container_arithmetic_contract.md` が表す frozen B0 contract である。

#### `adopted: false` の位置づけ(二段構造)

`adopted: false` は D21 が作り出した判断ではない。次の二段で読む:

1. **B1 evidence / completion ledger** は、frozen B0 `slice` instability により candidate を
   non-adoptable と記録している(`docs/plans/evidence/v0.2.0-b1/completion-ledger.md`、
   `docs/plans/evidence/v0.2.0/completion-ledger.md`)。
2. **その evidence を踏まえたうえで**、D21 は v0.2.0 の outcome として B0 fallback を選択した。

したがって「D21 が意図的に fallback を選んだ」とだけ書くことも、
「slice instability が D21 の理由である」と書くことも、いずれも不正確である。

#### Open-2 の解決

**Open-2(「#637 の decision-date method は確定済みだが calendar date は TBD。#675 が
milestone mid-point で確定する v0.2.0 gate」)は D21 により解決したため、§11 の Open items
表から除外した。** #675 の decision-date mechanism は superseded であり、**calendar date を
retroactive に設定しない**。残る Open-N はリナンバーしない(Open-1 / Open-3 の識別子は不変)。
D10 および D20 に含まれる decision-date 関連の歴史的記述は、当時の記録として書き換えない。

#### Follow-up repository/GitHub synchronization(D21 の決定内容ではない)

以下は D21 が決めた事柄ではなく、**D21 を GitHub 状態へ反映するための運用手順**である。
正典に運用手順を混ぜないため、決定本文とは節を分けて記録する。適用は別途承認を要する。

| 対象 | 適用予定 |
|---|---|
| #637 | open のまま維持し、v0.2.0 milestone から外す |
| #637 body | release-blocker 記述を D21 outcome へ同期する(D4 方式 — body を更新し、旧文言を履歴コメントで保全する) |
| #675 | `not planned` で close する |
| #413 | **変更しない**(no-touch invariant。D20 の superseding comment は別作業) |
| v0.2.0 milestone description | 旧 decision-date 文言を D21 参照へ差し替える |

適用順序は「#637 の意味論を先に同期してから release milestone から外す」。
D21 ruling 自体は merge / tag / publication / v0.2.0 release を承認していない。

## 12. Appendix — 外部 AI 議論の要約(2026-08)

**出典注記**: 2026-08-12 の追加議論(I/O time/dispatch semantics の2トラック構造、
API compatibility テーマ、正本の階層設計)についても原本はセッション添付の
ChatGPT ログであり repo 外にあるため永続しない。この議論の原文は archive せず、
反映内容の要旨は D14〜D17(§11)に repo 内で自己完結的に記録済みである。

原議論(`v0.2.0_______.md`, 2026-08-08 23:19 〜 2026-08-09 08:11)の要旨:

1. 分類は 8ドメイン → 8ドメイン改訂版(Histogram追加)→ **11機能ドメイン + 4横断基盤 +
   consumer layer** の順に3回改訂された。最終案: Core data(Time/Frequency, Matrix,
   Histogram, Field, Segment)/ Data access(Experimental I/O, Interoperability)/
   Analysis(General, Commissioning, Modeling)/ Presentation(Visualization)/
   Cross-cutting(X1 Semantic, X2 Persistence, X3 API Stability, X4 Performance)/
   Consumer(GUI/CLI/Jupyter、ドメイン外)。
2. リリース系列: v0.2.0=Semantic Foundation(container contract + Field I/O +
   SegmentTable eager workflow + docs統合 + GUI分離)→ v0.3.0=Experiment Workflow
   (lazy/groupby)→ v0.4.0=Spatial Geometry & Visualization → v0.5.0=Mesh & Solver
   Interop → v0.6.0=Fisher → v0.x後半=Ecosystem/App readiness(番号未定)→
   v1.0=全ドメイン一括の public contract 安定化宣言。
3. 各ドメインに Minimum/v1.0/Long-term の3段階ゴールを定義する方式を提案。
4. 各 minor は「1 release statement + 2〜3 headline user stories」のみを持ち、
   issue の milestone 判定を user story 必要性で機械化する方式を提案。
5. Bridge-not-reimplement 原則("Bridge APIs to selected external libraries — not
   a closed analysis framework")を長期原則として維持することを提案。

**破棄した結論**(3視点レビューで問題判明のため本文書には反映しなかった):

- v0.1.14 の triage 提案(line 22-34、line 18 の「必ずリリースする必要はない」結論)
  — 議論終了の約 50 分後に issue #653 で実際の v0.1.14 scope(required 12件)が
  確定しており、この会話の結論は陳腐化している。出荷済み v0.1.14 の release policy と
  成果は `ROADMAP.md` / `CHANGELOG.md` の記録を正とする。
- Histogram の v0.2.0 完成条件を「silent downgrade の禁止」とする記述(line 1057,
  1145, 1153-1157)— 実装は既に v0.1.13 で fail-closed 化済みであり、事実誤認。
  §4.3 で修正済み。
- v0.3.0 DoD の「数万 segment を lazy なしで解析できる」(line 802, 804)—
  同一段落内で自己矛盾(lazy を明示的に除外しつつ数万規模を要求)。§6 で
  「demonstrated usage requirement」を前提条件とする形に置き換えた。
- GWADW2026 資料の page 番号引用(line 143, 163, 178 等)— 資料が repo 内に
  存在せず検証不能。本文書では repo 内の一次資料のみを根拠とする。
- v0.4.0/v0.5.0/v0.6.0 への確定バージョン番号の事前割当(line 651 表)—
  `ROADMAP.md` の「次々minor の milestone は前 minor 出荷まで作らない」ポリシーと
  矛盾する。D13 により名前ベースの並びに変更。
