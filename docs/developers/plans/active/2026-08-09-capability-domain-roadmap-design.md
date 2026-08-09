# Capability-domain 長期ロードマップ設計(v0.2.0 → v1.0)

> Last-updated: 2026-08-09 (rev 1 — 初版)
> Reviewer Status: **draft**(3視点レビュー反映済み、ユーザー最終レビュー待ち)

Status: planned

対象: `ROADMAP.md` / `docs_redesign/explanation/roadmap.md` / GitHub milestone・issue triage 規則。
**実装作業は含まない。** 実装(#637 の composition 移行、Field I/O、SegmentTable 等)は別タスクで扱う。

## 1. Goal

外部 AI との議論(2026-08-08〜09、以下「原議論」)で提案された gwexpy の長期構造 —
機能ドメイン分類、v0.2.0 → v1.0 のリリース系列、各ドメインの Minimum/v1.0/Long-term
ゴール、「release statement + headline user stories」方式 — を、3視点レビュー
(アーキテクチャ/批判的/事実監査。物理視点は今回未実施)で検証したうえで、
repo 内の一次文書として固定する。

原議論の原本は `/home/washimi/.paseo/uploads/upload_86861967-8f96-460e-92b2-6229ad8b9925/v0.2.0_______.md`
に存在するが、これは repo 外のアップロードファイルであり将来消失し得る。本文書の
§12 Appendix に要旨を転記し、repo 内で自己完結させる。

## 2. Scope / Non-scope

### Scope

1. 本文書(正本)— ドメイン定義・per-domain goals・release statements・triage 規則
2. `ROADMAP.md` の改訂設計への正本提供(実際の改訂は別コミット)
3. 公開 roadmap (`docs_redesign/explanation/roadmap.md`) 同期の方針
4. issue #413(v0.2.0 release notes)の記入内容の根拠
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
| Consumer(ドメイン外) | — | GWexpy Studio / pyaggui / CLI / Jupyter | `gwexpy/cli/` 含む | public API のみを消費する側。**未解決事項**: `gwexpy/cli/` は wheel 内かつ `._version` 等の private モジュールを参照しており、「public API のみで完結」という consumer layer の前提を現時点で満たしていない。CLI を core 側の一部として扱うか、private 依存を解消して真の consumer にするかは本文書では未決定とし、Decision log に open item として残す |

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
  取得し、変更後の退行を予算内(要具体数値化)に収める。
- **v1.0**: `#580`/`#581` のベンチマーク基盤に基づき、測定駆動で性能改善を優先付けする
  運用が定着する。lazy/distributed execution は「実証された利用要求」が前提。
- **Long-term**: 大規模 SegmentTable ワークフロー(demonstrated usage を前提とした
  lazy/out-of-core)への拡張。

## 5. Domain × release-theme matrix

列はリリーステーマ名(バージョン番号は次の 1 minor のみ確定、それ以降は名前で扱う。
§11 D13 参照)。◎=primary、○=secondary、—=対象外(意図的な選択であり欠落ではない)。

| Domain | v0.2.0 (Container Semantic Contract) | Experiment data workflow | Advanced segment workflows | Spatial geometry & layered viz | Mesh fields & solver interop | Fisher & advanced analysis | Later 0.x (ecosystem/app) | v1.0 |
|---|---|---|---|---|---|---|---|---|
| 1 Time/Frequency | ○ | — | — | — | — | — | — | criteria met |
| 2 Matrix | ◎ | — | — | — | — | ○ | — | criteria met |
| 3 Histogram | ○(raises 登録のみ) | — | ○(segment 集約) | ○(plotting) | ○(ROOT interop 強化) | ○(covariance fitting) | — | criteria met |
| 4 Field | — | ◎ | — | ◎ | ○ | — | — | criteria met |
| 5 Segment | — | ◎ | ◎ | — | — | — | — | criteria met |
| 6 Experimental I/O | ○(golden tests) | ○ | — | — | ○ | — | ○ | criteria met |
| 7 Interoperability | — | — | — | — | ◎ | — | ◎ | criteria met |
| 8 General Analysis | — | — | — | — | — | ○ | — | criteria met |
| 9 Commissioning | — | — | — | — | — | — | — | criteria met |
| 10 Modeling | — | — | — | — | — | ◎ | — | criteria met |
| 11 Visualization | — | — | ○ | ◎ | — | ○ | — | criteria met |
| X1 Semantic contract | ◎ | ○ | — | — | — | — | — | 全域適用 |
| X2 Provenance | ○ | ◎ | — | — | — | — | — | 全域適用 |
| X3 API stability | ◎ | — | — | — | — | — | — | 全域適用(stability label 完了) |
| X4 Performance | ◎(ベースライン取得) | — | ○(lazy 実証要求) | — | — | — | ○ | 全域適用 |

v1.0 列は「directional な予定」ではなく **criteria 列**である(全 domain が Minimum
goal を満たし、X1-X4 が全域適用済みであることの確認列)。他の列はすべて directional で
あり、re-scope され得る(§7 の disclaimer と同一の位置づけ)。

## 6. Release statements & headline user stories(正本)

`ROADMAP.md` と issue #413 はここからの転記とする。各 user story には
**実行可能な受入成果物(named end-to-end test か、グリーンで走る example notebook)**
を紐付ける — 「必要かどうか」の判定を実行可能にするため(§9 M5 への対応)。

### v0.2.0 — Container Semantic Contract

> Every supported operation on a GWexpy container preserves class, unit, axes,
> labels, and metadata, or raises explicitly — never a silent downgrade.

Headline user stories:
1. Container arithmetic follows a declarative, human-reviewed contract matrix
   (class × operand × operator × side × in-place); unsupported cells are explicit
   `raises` entries; the suite asserts its collected-case count.
   **受入成果物**: `tests/types/test_container_arithmetic_contract.py`(新設、収集数アサート付き)が green。
2. `np.sqrt(matrix)` and `(2 * u.s) * matrix` both return the correct class,
   values, units, and metadata via the SeriesMatrix composition redesign.
   **受入成果物**: `tests/types/test_series_matrix_ufunc.py` の当該ケースが green。
   決定日までに green にならない場合は `__array_ufunc__ = None` の documented
   limitation で出荷し、再設計は次テーマへ(D10)。
3. HDF5 written by GWexpy for GWpy-derived containers is readable by a
   GWpy-only process (no `import gwexpy`).
   **受入成果物**: `tests/io/test_hdf5_gwpy_compat.py`(新設)が gwpy-only subprocess で green。

Definition of done: 上記3成果物が green、かつリファクタ前に取得した性能ベースライン
に対して退行が予算内。

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

Design groundwork: [SegmentTable workflow plan](2026-08-01-segmenttable-workflow-design.md)。

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

Design groundwork: [layered visualization plan](2026-08-01-layered-visualization-design.md)。

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
| issue #413 | v0.2.0 (Container Semantic Contract) の release statement 転記先 |
| issue #400 | X3 Minimum のモジュール単位 stability label。v1.0 の主経路(D11) |

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
| B-1〜B-3(Auditor) | ドメイン番号スキームが3種並存、v0.2.0スコープ記述に不一致 | **採用**: 本文書を単一の正本とし、11+4分類のみを使用 |
| B-4(Auditor) | v0.1.14 の結論が #653 で陳腐化 | **採用**: 本文書・ROADMAP 改訂ともに v0.1.14 に言及しない(別会話スコープ) |
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

Status を `completed` にするのは上記が実行され結果が確認された時点とする
(`Status: completed (verified: <実行コマンドと結果>)` の形式、planning-docs ルール準拠)。

## 11. Decision log

| ID | 決定 |
|---|---|
| D1 | ドメイン全文は `ROADMAP.md` でなく本文書に置く。ROADMAP は要約+リンクのみ |
| D2 | v0.2.0 milestone の名称・番号は変更しない(scope のみ contract-only に narrowing) |
| D3 | 公開 roadmap にドメイン番号・体系語彙を出さない |
| D4 | #413 は body 全面編集+履歴コメントで旧文言を保全する方式 |
| D9 | v0.2.0 を二段分割(Container Semantic Contract / Experiment data workflow)。GUI・docs工事は milestone 外 |
| D10 | #637 は期日+fallback(`__array_ufunc__=None` + documented limitation)付き blocker |
| D11 | v1.0 は #400 モジュール単位 stability label を主経路とする criteria 節。一括宣言は補助 |
| D12 | Histogram の v0.2.0 スコープは raises 行の登録までとし、伝播規則設計は将来テーマ |
| D13 | Future themes は番号でなく名前で並べる。次の 1 minor のみ番号確定 |
| Open-1 | CLI の consumer layer 分類と private API 依存の矛盾は未解決(§3) |
| Open-2 | #637 の decision date は本文書執筆時点で未確定。Phase 0 追跡 issue に残タスク化 |
| Open-3 | X4 の性能退行予算の具体的数値は未確定 |

## 12. Appendix — 外部 AI 議論の要約(2026-08)

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
  確定しており、この会話の結論は陳腐化している。v0.1.x は別会話スコープのため
  本文書では扱わない。
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
