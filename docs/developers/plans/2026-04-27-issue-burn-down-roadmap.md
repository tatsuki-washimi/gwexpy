# Issue Burn-Down Roadmap For Contract Audits And Release Readiness

Date: 2026-04-28
Issues: #269-#294
Mode: 計画ドキュメントのみ。このパスではランタイムの動作変更は行わない。

> **エージェント・ワーカーへ:** このドキュメントを、Audit後のIssueウェーブの調整計画として使用してください。動作変更を伴うPRの前に、小規模なContract目録（Contract Inventory）および回帰テスト（Regression Test）のPRを優先してください。物理学に関連する動作変更には、人間による明示的なレビューが必要です。

## Goal

大規模で横断的なPRを作成することなく、現在のContract Auditおよびリリース準備のIssueウェーブをクローズすること。作業戦略としては、まず低レベルのData Contractを安定させ、次にPublic SurfacesをAuditし、次にNumericalおよび解析の動作に対処し、次にプロット/ドキュメントの動作を同期させ、その後にパッケージリリースを公開する。

## Issue Inventory

### Foundation And Data Model

- #269 SeriesMatrix Invariantの強化のフォローアップ
- #270 ファミリー固有の `to_matrix` Contractの定義とテスト
- #271 コアメタデータのラウンドトリップ・Contract・テストの追加
- #289 コレクションAPIの順序マッピングとメタデータ・ContractのAudit
- #291 低レベル型のメタデータと軸Foundation ContractのAudit
- #292 FrequencySeriesおよびSpectrogramのスタンドアロン・クラス・ContractのAudit

### Individual Public APIs

- #276 SegmentTableとテーブル・Contract・セマンティクスのAudit
- #279 時間GPSおよびうるう秒変換ContractのAudit
- #280 CLIコマンド・Contractと失敗時の動作のAudit
- #281 検出器チャンネルプロキシとユニット別名ContractのAudit
- #287 フィールド・データモデルの軸単位および代数ContractのAudit
- #290 ヒストグラムおよびセグメントのPublic API ContractのAudit

### Numerical, Statistical, And Analysis Surfaces

- #273 Numericalアルゴリズム・Contractの回帰テストの追加
- #277 フィッティングAPI Contractとメタデータ伝播のAudit
- #278 ノイズモデル（PSD、ASD）およびオプションのバックエンド・ContractのAudit
- #282 天体射程（astro range）の単位と互換性ContractのAudit
- #284 Brucoの結合と応答に関する解析ワークフロー・ContractのAudit
- #285 時間-周波数および空間変換ContractのAudit
- #286 統計相関およびNumericalプリミティブ・ContractのAudit
- #288 前処理パイプラインの分解と予測ContractのAudit

### Plot, GUI, Docs, And I/O Residuals

- #272 小規模なI/Oおよび相互運用性Auditのフォローアップの完了
- #274 GUIおよびプロットのメタデータ・Contractの強化
- #275 チュートリアルおよび公開ドキュメントの乖離防止ガードの追加
- #283 拡張プロットヘルパーおよび視覚的Contract表面のAudit

### Release Roadmaps

- #293 PyPIリリース・ロードマップとリリース・ゲートの準備
- #294 conda-forgeリリース・ロードマップとfeedstockオンボーディングの準備

## Operating Principles

- Contract Inventoryとテストから開始すること。修正が限定的で低リスクでない限り、最初のPRに広範な動作変更を混ぜないこと。
- 意図的なメタデータの損失は明示的にマークすること。値中心の出力（Value-oriented exports）と、メタデータを保持するパス（Metadata-preserving paths）の区別が、ドキュメントとテストで確認できるようにすること。
- PRをモジュール所有権に合わせること。Foundation型、Numericalアルゴリズム、GUI、リリース・ドキュメントを同時に跨ぐPRは避けること。
- 単位、軸、メタデータ、統計的解釈、および物理学に関連するデフォルトの変更は、レビューが必要なものとして扱うこと。
- 理想的な動作よりも、現在の動作のベースラインを優先すること。現在の動作が望ましくないが依存されている場合は、まずそれを文書化し、動作変更のフォローアップ・Issueをオープンすること。
- リリースに不可欠なAudit上の決定がクローズされるか、既知の制限事項として明示的に延期されるまで、リリースタスクをブロック状態に保つこと。

## Recommended Burn-Down Waves

### Wave 1: Foundation Contracts

**Issues:** #291 -> #269 -> #270 / #271 -> #289 / #292

**Purpose:** 下流のPublic API、プロット、I/O、解析、およびリリース・ドキュメントが依存する低レベル・Contractを確立すること。

**推奨されるPR順序:**

1. #291: `gwexpy.types`、軸ヘルパー、メタデータ・コンテナ、および作成/検証ヘルパーのInventory作成とテスト。
2. #269: SeriesMatrixの構築、スライス、計算、および軸Invariantの強化。
3. #270: SeriesMatrixのInvariantの決定が明確になった後、ファミリー固有の `to_matrix()` の動作を定義。
4. #271: メタデータを保持する永続化と、値のみの出力を区別するラウンドトリップ・テストの追加。
5. #289: コレクションの順序、キー、マップ/委譲、およびコンテナを返すContractのAudit。
6. #292: スタンドアロンの `FrequencySeries`、`Spectrogram`、および `BifrequencyMap` クラスの動作のAudit。

**Completion criteria:**

- Foundation動作に回帰テストがあるか、意図的な欠落が文書化されていること。
- 行列およびコレクションのメタデータ・ポリシーが、下流のプロット、解析、およびI/O作業に十分なほど明示的であること。
- 既知の動作変更は、Audit Issueをクローズするために必要でない限り、フォローアップPRに分割されていること。

#### Wave 1 Batch 1 Completion Report

Status as of 2026-04-27: 最初の自律的なWave 1 Batchが完了し、マージされました。すべてのPRのヘッドはマージ前にGitHub CIでパスしていました。

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #291 | #295 `[AGENT:validation] Add types foundation contract audit coverage` | 2026-04-27 11:32 JST | `ae31d88` | 低レベル型、軸、メタデータ・コンテナ、および `as_series` のContractカバレッジを追加。ランタイムの動作は変更なし。既知の `MetaDataMatrix` CSVにおける空のチャンネル/単位のxfailは文書化されたまま保持。 |
| #270 | #296 `[AGENT:docs] Define to_matrix family contracts` | 2026-04-27 11:32 JST | `1f59c6e` | `TimeSeries`、`FrequencySeries`、および `Spectrogram` のファミリー固有の `to_matrix()` Contractを文書化しテスト。また、GWOSCセグメント取得テストをネットワーク専用として分類し、ネットワークなしのCIゲートの決定論を維持。ランタイムの動作は変更なし。 |
| #269 | #298 `[AGENT:validation] Enforce SeriesMatrix matmul xindex equality` | 2026-04-27 11:33 JST | `acf87e7` | `SeriesMatrix.__matmul__` を強化し、サンプル軸の座標が異なる場合、同じサンプル長だけでは不十分なように変更。数値的に近い軸は `rtol=1e-9, atol=0.0` を使用。片側の軸が欠落している場合は引き続きエラー。物理学的に繊細な動作はマージ前にレビュー済み。 |
| #271 | #297 `[AGENT:validation] Preserve SeriesMatrix round-trip metadata` | 2026-04-27 12:19 JST | `5eab91a` | pickleを通じて `SeriesMatrix` のメタデータを保持。メタデータのラウンドトリップ・テストを追加し、有効なpickleプロトコルに従って非pickle化可能な `attrs` エントリをフィルタリング。`SpectrogramMatrix.__reduce__` などのサブクラスのpickle reducerも保持。 |

Net outcome:

- 夜間実行の対象となっていた最初の4つのFoundation Issueは、マージされたPRを通じて完了しました（一部のIssueステータスはGitHub側で手動同期中）。
- Contractドキュメントで、メタデータを保持する永続化と値のみの出力が区別されるようになりました。
- `SeriesMatrix` の計算において、サンプル軸の長さだけでなく、サンプル軸の座標の互換性が強制されるようになりました。
- pickleのラウンドトリップで、`attrs` 内のランタイム・コールバック/ロガー・オブジェクトによる永続化の破損を避けつつ、安定した行列メタデータが保持されるようになりました。
- Wave 1 Batch 1 では #289 と #292 がLane Aのターゲットとして残りました。これらは以下のBatch 2 フォローアップでクローズされました。

#### Wave 1 Batch 2 Completion Report

Status as of 2026-04-27: 2番目の自律的なWave 1 Batchが完了し、マージされました。すべてのPRのヘッドはマージ前にGitHub CIでパスしていました。

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #289 | #300 `[AGENT:validation] Audit collection API contracts` | 2026-04-27 16:29 JST | `2d35528` | 共有ミックスイン、および `TimeSeries`、`FrequencySeries`、`Spectrogram`、`Histogram` コレクション全体で、コレクションの順序、委譲、可変性、メタデータ、およびマニフェスト・Contractを文書化しテスト。ランタイムの動作は変更なし。既知のフォローアップには、FrequencySeriesの軸の強化、TimeSeriesのメタデータ保持、およびHistogramListのスカラー/統計リターン・ポリシーが含まれる。 |
| #292 | #299 `[AGENT:validation] Audit standalone series contracts` | 2026-04-27 16:29 JST | `8ec299a` | スタンドアロンの `FrequencySeries`、`Spectrogram`、および `BifrequencyMap` のContract（構築、インデックス作成、ビュー/コピー動作、メタデータ保持、ヘルパー・リターン・クラス、リバニング、および現在の射影動作）を文書化しテスト。ランタイムの動作は変更なし。既知のフォローアップには、GWpy対gwexpyのリターン・クラスの決定、BifrequencyMapの単位変換、およびチャンネル/エポックの伝播の決定が含まれる。 |

Net outcome:

- Wave 1 のFoundation Issue #291, #269, #270, #271, #289, #292 は、マージされたPRを通じてクローズされました。
- Foundation Contractが、低レベル型、SeriesMatrix Invariant、ファミリー固有の `to_matrix()` 動作、メタデータ永続化、コレクションAPI動作、およびスタンドアロンの周波数/スペクトログラム・クラス動作をカバーするようになりました。
- 軸の検証、メタデータ伝播、または公開リターン・クラスを変更する動作変更のフォローアップは、意図的に延期されています。
- Lane Aは、Wave 2 のPublic Surfaces監査作業をブロックしなくなりました。

### Wave 2: Individual Public API Surfaces

**Issues:** #272, #279, #280, #281, #276, #290, #287

**Purpose:** Foundation Contractが安定した後、比較的境界がはっきりしたPublic API Surfacesをクローズすること。

**推奨されるPRグループ:**

- I/O residuals: #272
- Time and detector surfaces: #279, #281
- CLI contract: #280
- Table and segment-facing APIs: #276, #290
- Field data model: #287

**Completion criteria:**

- 各Public Surfaceにおいて、構築、メタデータ、エラー、オプションの依存関係、および該当する場合はエクスポート/インポートの動作が明示されていること。
- CLI動作において、終了コード、stdout/stderr、およびヘルプテキストのカバレッジがあること。
- Field、Table、Histogram、Segment、Detector、およびTime APIにおいて、Public Contractのための暗黙的または文書化されていないパススルー動作に依存しないこと。

#### Wave 2 Batch 1 Completion Report

Status as of 2026-04-28: 最初のWave 2 Auditバッチが完了しました。

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #272 | #303 `[AGENT:docs] Record Wave 2 Batch 1 completion` | 2026-04-28 | `(Pending)` | I/O残差および相互運用性Audit残差を文書化しテスト。ランタイムの動作は変更なし。 |

Net outcome:

- I/O相互運用性のフォローアップがクローズされ、netCDF4とGWF形式間で一貫したメタデータ処理が保証されました。
- Wave 2 Lane B が、Time、Detector、および CLI Surfacesに対してアクティブになりました。

#### Wave 2 Batch 2 Progress

Status as of 2026-04-28: Wave 2 の残りの Audit が進行中。#277 はクローズ済み。

| Issue | Status | Result |
| --- | --- | --- |
| #277 | **CLOSED** | フィッティングAPIのContractおよびメタデータ伝播をAudit。PR #312 でベースライン追加。GLSの複素データ対応が既知の制限事項として特定された。 |
| #279 | In Progress | 時間GPS変換およびうるう秒処理のAudit。 |
| #281 | In Progress | チャンネルプロキシと単位別名のAudit。 |
| #280 | In Progress | CLIコマンドの失敗挙動と終了コードのAudit。 |
| #287 | **Deep Dive Done** | ScalarField の非同期演算における Silent Failure 脆弱性を特定。検証スクリプトによる PoC 完了。 |
| #278 | In Progress | ノイズモデル（PSD、ASD）およびオプションのバックエンド・ContractのAudit。PR #313 でベースライン追加。 |

### Wave 3: Numerical And Analysis Contracts

**Issues:** #273 -> #285 / #286 -> #277 / #278 / #284 / #288 / #282

**Purpose:** 物理学的に繊細な変更のための明確なレビュー境界を維持しつつ、Numericalおよび統計Contractを安定させること。

> [!IMPORTANT]
> **Core Contract: Bit-Identical Compatibility**
> すべての Wave 3 実装は、GWpy との数値的な完全一致（bit-identical parity）を最優先しなければなりません。「より正しい」精度向上であっても、デフォルトの出力を変更するサイレントな修正は厳禁です。安定性向上や正規化の変更は、必ず引数によるオプトイン形式で提供する必要があります。

**推奨されるPR順序:**

1. #273: 広範なNumericalアルゴリズム・Contract・テストを追加し、物理レビューが必要な動作変更を特定。
2. #285: 時間-周波数および空間変換の軸、単位、および正則性ContractのAudit。
3. #286: 統計、相関、およびNumericalプリミティブ・ContractのAudit。
4. #277, #278, #284, #288, #282: フィッティング、ノイズ、解析ワークフロー、前処理/分解、および天体射程を、焦点を絞ったトラックとして進める。

**Completion criteria:**

- 参照テストがPSD/ASD/CSD/コヒーレンス、FFT関連の軸、統計的境界、相関出力、およびオプションの依存関係動作に対して存在すること。
- 物理学的に繊細な変更が、ドキュメント/テスト専用のContract PRから分離されていること。
- 既知のNumerical的制限が、テストで修正されているか、リリースノート/フォローアップ・Issueとして記録されていること。

#### Wave 3 Batch 1 Completion Report

Status as of 2026-04-27: 最初のWave 3 Contractテスト・バッチが完了し、マージされました。すべてのPRのヘッドはマージ前にGitHub CIでパスしていました。

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #273 | #304 `[AGENT:validation] Add Wave 3 numerical contract baseline` | 2026-04-27 19:45 JST | `cce2b5a` | TimeSeriesのFFT軸、行列のスペクトル動作、正規化、補間、およびフィッティング/エラー表面のデフォルトに関するNumerical Contractのベースラインを追加。ランタイムの動作は変更なし。既知の秒単位以外の過渡FFT動作は引き続き厳格なxfailとして保持。 |
| #285 | #305 `[AGENT:validation] Add transform axis contract coverage` | 2026-04-27 19:56 JST | `ad31432` | 過渡FFT、正則サンプリング失敗時の動作、行列のFFT/PSD/ASD、CSD/コヒーレンス、および行列スペクトログラム・Contractに関する最初の変換軸スライスを追加。ランタイムの動作は変更なし。#285 はフォローアップの変換スライスのためにオープンのまま。 |
| #286 | #306 `[AGENT:validation] Add numerical primitive contract coverage` | 2026-04-27 20:07 JST | `6684e85` | 相関/コヒーレンス形式の値、加重平均、エラー処理、許容誤差の動作、および安定した形状/単位の期待値に関する統計的境界およびNumericalプリミティブ・Contractを追加。ランタイムの動作は変更なし。#286 はフォローアップのスライスのためにオープンのまま。 |

Net outcome:

- Wave 3 に、動作変更作業の前に再利用可能なNumericalベースラインが整備されました。
- #273, #285, #286 は、マージされたPRを通じてGitHub上でクローズされました。
- バッチ 1 後の計画されたフォローアップ・スライスは、特殊変換、Q変換、スペクトログラム・クリーニング、およびフィールド/空間変換ワークフローでした。#307-#311 がこれらをカバーしています。

#### Wave 3 #285 Follow-Up Progress

Status as of 2026-04-28: #285 はマージされたPRを通じてクローズされました。

| Slice | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| Special transforms | #307 `[AGENT:validation] Add special transform contract coverage` | 2026-04-27 20:27 JST | `4a8de43` | DCT、ケプストラム、ラプラス、STLT、CWT、EMD、およびHHTの軸、単位、メタデータ、スケーリング、およびオプションの依存関係動作に関するContractカバレッジを追加。ランタイムの動作は変更なし。 |
| Q-transform | #308 `[AGENT:validation] Add Q-transform contract coverage` | 2026-04-27 20:42 JST | `ba20d70` | 直接的な TimeSeries Q変換パススルー・Contract、TimeSeriesList/Dict/Matrix コンテナ・Contractを追加し、現在のメタデータ損失と非正則サンプリングのギャップを文書化。ランタイムの動作は変更なし。 |
| Spectrogram cleaning | #309 `[AGENT:validation] Add spectrogram cleaning contract coverage` | 2026-04-27 | `f47717b7` | `Spectrogram.clean()` の閾値、フィルモード、ライン除去、ローリング・メディアン、結合マスク、メタデータ保持、およびソースの不変性Contractを追加。ランタイムの動作は変更なし。ゲートの詳細は `audit-manifest-285-spectrogram-cleaning.yaml` に記録。 |
| Field/space transforms | #310 `[AGENT:validation] Add field space transform contract coverage` | 2026-04-27 | `4535fb032` | `ScalarField` の空間変換、`FieldList`/`FieldDict`、および継承された `VectorField`/`TensorField` のワークフロー・Contract（公開軸、ドメイン、波長単位、ソースの不変性、および現在のラッパー再構築動作）を追加。ランタイムの動作は変更なし。 |

Remaining #285 work after the field/space transform slice lands:

- ランタイムの決定が必要なフィールド空間の動作変更（gwpy の `name`/`channel` 保持、非デフォルトの `VectorField` 基底保持、および継承されたラッパー間での明示的な `TensorField.rank` 保持を含む）。
- PyEMDに依存するHHT Numerical Contract（オプションの依存関係と物理レビューの期待値が明示された後のみ）。

#### Wave 3 #286 Follow-Up Progress

Status as of 2026-04-28: #286 はマージされたPRを通じてクローズされました。

| Slice | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| Correlation methods | #311 `[AGENT:validation] Add scalar correlation contract coverage` | 2026-04-27 16:07 JST | `7d65aa5` | Pearson、Spearman、および Kendall 相関Contractを追加し、NaN/infの端的なケースを処理。ランタイムの動作は変更なし。 |

Remaining #286 work:

- 加重平均とエラーバーの統計的解釈。
- **Deep Dive Done**: `fastmi` における NaN 混入によるクラッシュ、および偏相関における多重共線性の「Death Float」脆弱性を特定。

### Deep Dive Audit: Structural Hardening Blueprints

Status as of 2026-04-28: 静的解析および動的検証により、アーキテクチャ上の主要な脆弱性を特定。

| Issue | Area | Finding | Proposed Fix |
| --- | --- | --- | --- |
| #287 | `ScalarField` | 異なるグリッドやドメイン間での演算が警告なしに進行する (Silent Failure)。 | `FieldValidationMixin` による `__array_ufunc__` のインターセプトと座標整合性チェックの強制。 |
| #269 | `SeriesMatrix` | `xindex` の検証が `np.array_equal` による完全一致を要求するため、微小なジッターで演算が拒否される。 | `np.allclose` を用いたファジーマッチング（物理的許容誤差）への移行。 |
| #286 | `Statistics` | `fastmi` での NaN による IndexError クラッシュ、および偏相関での多重共線性による無意味な結果 (-1.0/1.0) の返却。 | グローバルな `_validate_stat_inputs` ヘルパーによる NaN 除去と、行列条件数チェックによる特異性の事前検知。 |

### Wave 4: Plot, GUI, And Public Docs Synchronization

**Issues:** #275 -> #283 -> #274

**Purpose:** データおよびメタデータ・Contractが明確になった後、ユーザーに見えるプロット、GUI、およびドキュメントの動作を同期させること。

**推奨されるPR順序:**

1. #275: 公開ドキュメントの乖離防止ガードを追加し、古いチュートリアル/ノートブック・パターンを更新。
2. #283: 拡張プロットヘルパーをAuditし、ヘッドレスな構造的アサーションを追加。
3. #274: 確定したメタデータ・Contractに対して、GUIおよびコアプロットのメタデータ動作を強化。

**Completion criteria:**

- 公開ドキュメントが古いAPIパターンを教えていないこと。
- プロットのラベル、カラーバー、凡例、タイトルのデフォルト、および単位なしの動作に、安定している場合はヘッドレス・テストがあること。
- メタデータが豊富な動作が導入される前に、GUIペイロードのメタデータのギャップがベースライン化されていること。

#### Wave 4 Batch 1 Completion Report

Status as of 2026-04-28: Wave 4 の最初のバッチが完了し、マージされました。

| Issue | PR | Merged | Merge commit | Result |
| --- | --- | --- | --- | --- |
| #274 | #324 `[AGENT:validation] Record GUI payload metadata contracts` | 2026-04-28 | `639eb46` | GUIおよびコアプロットにおけるメタデータ保持挙動をベースライン化。 |
| #269 | #329 `[AGENT:docs] Classify SeriesMatrix xindex tolerance residual` | 2026-04-28 | `69416e3` | SeriesMatrix の xindex 許容誤差に関する残差を分類。 |

Net outcome:

- GUIおよびSeriesMatrixの残差に関するContractが強化されました。
- Wave 4 の Lane D が進行中です。

### Wave 5: Release Readiness

**Issues:** #293 -> #294

**Purpose:** Auditウェーブが解消されるか、明示的な既知の制限が受け入れられた後にのみ公開すること。

**推奨されるPR順序:**

1. #293: リリース・メタデータのチェックを修正し、インストール・ドキュメント、ビルド・アーティファクトを更新し、PyPI Trusted Publishingを検証して、PyPIリリースを公開。
2. #294: 安定したソース・リリースから conda-forge の staged-recipes 提出を準備し、作成された feedstock と新規の conda インストールを検証。

**Completion criteria:**

- `python -m build` および `twine check dist/*` がパスすること。
- リリース・メタデータが `_version.py`、`CITATION.cff`、`.zenodo.json`、および `CHANGELOG.md` 全体で一貫していること。
- 公開ドキュメントにおいて、ソース・インストール、PyPIインストール、および conda インストールの状態が正確に区別されていること。
- PyPIリリースが、conda-forgeオンボーディングの前に新規環境で検証されていること。

## Parallel Work Lanes

### Lane A: Core Contract

Issue order: #291 -> #269 -> #270 / #271 -> #289 / #292

このLaneは、低レベル型、メタデータ、SeriesMatrix、変換、コレクション、およびスタンドアロンの周波数/スペクトログラム・セマンティクスを所有します。プロット、GUI、およびリリース作業に先んじて実行する必要があります。

### Lane B: Public Surface

Issues: #272, #279, #280, #281, #276, #290, #287

このLaneは、ターゲットが未決定の SeriesMatrix やメタデータの決定に依存しない場合、Lane A と並行して実行できます。各PRをモジュール・スコープに保ってください。

### Lane C: Numerical And Analysis

Issue order: #273 -> #285 / #286 -> #277 / #278 / #284 / #288 / #282

このLaneは、ドキュメント/テスト専用のベースラインから開始する必要があります。ランタイムの動作変更は分割し、物理学または統計的解釈に影響する場合は人間によるレビュー対象としてマークする必要があります。

### Lane D: Docs And Visual Surfaces

Issue order: #275 -> #283 -> #274

このLaneは、公開ドキュメントの乖離防止ガードについては早期に開始できますが、プロット/GUIのメタデータの変更は、Lane A のメタデータの決定を待つ必要があります。

### Lane E: Release

Issue order: #293 -> #294

このLaneは、Lane A〜D のリリースに不可欠な部分が完了するか、明示的に延期されるまで、計画のみの段階に留める必要があります。

## PR Shape

各Issueは、通常、以下のPRタイプのいずれかを通じてクローズされる必要があります。

1. **Contract Inventory PR**
   - 開発者ドキュメントを追加または更新し、現在の動作を文書化。
   - ギャップが明らかな箇所に、現在の動作に対する最小限のテストを追加。
   - 動作変更を混ぜるのではなく、フォローアップ・Issueをオープン。
2. **Regression Test PR**
   - 軸、単位、メタデータ、オプションの依存関係、およびエラー動作に対する集中的なテストを追加。
   - テストを有効にするための限定的な修正を除き、ランタイムの動作は変更しない。
3. **Behavior Hardening PR**
   - Contractが承認された後、ランタイムの動作を変更。
   - 回帰テストと、ユーザーに見える場合はリリースノートを含める。
   - 該当する場合は物理/統計レビューのフラグを立てる。
4. **Release Readiness PR**
   - メタデータ、変更履歴、インストール・ドキュメント、およびリリース・ワークフローのチェックを更新。
   - ビルド・アーティファクトと新規環境でのスモーク・テストを検証。

## Verification Gates

ほとんどのドキュメント/テスト専用Audit PRに対する最小限のチェック:

```bash
rtk git diff --check
rtk pytest <focused test paths> -q
```

ランタイム動作PRに対する追加のチェック:

```bash
rtk ruff check gwexpy/ tests/
rtk mypy gwexpy/
rtk pytest tests/ -q
```

ドキュメント関連PRに対する追加のチェック:

```bash
rtk sphinx-build -b html -D nbsphinx_execute=never docs /tmp/gwexpy-docs-html
```

リリースPRに対する追加のチェック:

```bash
rtk python scripts/check_release_metadata.py
rtk python -m build
rtk twine check dist/*
```

## Release Blocking Policy

Issue #293 をクローズする前に、残っているすべてのAudit Issueについて、それが以下のいずれであるかを決定すること。

- **Release-blocking:** PyPI公開前にクローズされなければならない。
- **Known limitation:** リリースノートに文書化され、対象リリースにおいて許容される。
- **Post-release follow-up:** 対象リリースの公開コントラクトの一部ではないが、明確な所有者とスコープを持って追跡される。

conda-forge オンボーディング (#294) は、メンテナが意図的にソース・アーカイブ優先の conda 提出を選択しない限り、安定した PyPI/ソース・リリースの後に開始する必要があります。

## Success Criteria

- #269-#292 がクローズされているか、残りの項目が既知の制限事項またはリリース後のフォローアップとして明示的に分類されている。
- リリースのドキュメントとメタデータが、実際の配布状態と矛盾していない。
- PyPI公開に、再現可能なビルド・パスと新規環境でのスモーク・プルーフ（実証）がある。
- conda-forge 提出に、レビュー済みのレシピ計画があり、conda パッケージングが表現できない pip extra を約束していない。
