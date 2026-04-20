# AI Agent Bulletin Board (AIエージェント掲示板)

このファイルは、マルチエージェント環境における進捗管理と情報共有のハブです。各エージェントは自身の状況を更新してください。

## 1. エージェント一覧とステータス

| セッションID | 役割 | 担当内容 | ステータス |
|-------------|------|----------|------------|
| `gw-gm` | GM | Phase 7 docs/audit resync 総括 | ✅ Phase 7 review-ready |
| `gw-consultant` | Consultant | 戦略提案（文書・物理） | 🔵 待機中 (アドバイザー) |
| `gw-worker-1` | Worker 1 | ndscope alias / registry normalization | ✅ Phase 5 完了 / Phase 7 連携待ち |
| `gw-worker-2` | Worker 2 | Zarr timing metadata fail-fast | ✅ Phase 5 完了 / Phase 7 連携待ち |
| `gw-worker-3` | Worker 3 | docs / audit resync and truth ledger | ✅ Phase 7 review-ready |
| `gw-worker-4` | Worker 4 | regression validation / fixtures | ✅ Phase 5 完了 / Phase 7 連携待ち |

## 2. 現在のミッション

**目標:** Phase 7 として、実装済み変更と docs/audit 記録の不整合を解消する `docs/audit resync` を完了する。
**主要リソース:**
- 監査レポート: [統合監査レポート_計214件の指摘.md](../../../docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md)
- 実装計画書: [2026-04-18-docs-next-phase-implementation-plan.md](./2026-04-18-docs-next-phase-implementation-plan.md)
- 真実台帳: [truth_ledger.md](./truth_ledger.md)
- ガイドライン: [.agent/AGENTS.md](../../../.agent/AGENTS.md)

## 3. 進捗ログ (最新10件)

- **2026-04-18 13:28**: Coordinator がマルチエージェント環境をセットアップ。tmuxセッション起動完了。
- **2026-04-18 13:29**: 各エージェントに初期ブリーフィングを送信（ログイン待機状態）。
- **2026-04-18 13:45**: 各エージェントにミッションとリソースを伝達。GM/Consultant が Phase 0 の協議を開始。Workers は待機中。
- **2026-04-18 13:48**: エージェント設定を洗練。GM は監査レポートの全件読解に専念。コンサルタントおよび作業員は担当ペルソナと権限を付与し、待機状態へ移行。
- **2026-04-18 13:54**: GM による Phase 0 読解およびコンサルタントとの接続が完了。全意思決定ゲート（Gate A, B, C, D）の方針が確定。Phase 1（共通ページ契約）の実行フェーズへ移行可能。
- **2026-04-18 13:57**: ユーザー指示により Gate A (配布) を延期。8-19 対応等のソースコード修正を優先するため、ドキュメント上のステータス更新は後回しとする。GM は Gate A を除外して Phase 1 実行を進行。
- **2026-04-18 14:01**: Coordinator が 4名の作業員に Phase 1（共通ページ契約）のタスクを各3項目ずつ割り当て。全 Worker が実装フェーズに突入。Phase 1 では「対象読者・前提・用途」「コード例の入出力」等の標準化を日英ペアで実施。
- **2026-04-18 14:12**: Phase 1 の対象 24ファイル（12ペア）すべての標準化修正が完了。GM への報告を済ませ、Phase 2（高度なガイドとリファレンスのリンク改善）の計画フェーズへ移行。
- **2026-04-18 14:18**: GM が Phase 2 の worker 分担を確定し、実行フェーズ開始。
- **2026-04-18 14:20**: Phase 2 の全タスク（理論ページのバッジ付与・書誌一元化、API/チュートリアル間の 30ファイル以上の双方向リンク強化）が完了。全 57ファイルが修正済みとなり、Phase 3（ビジュアルナビゲーションの完成）へ移行。
- **2026-04-18 14:27**: GM が Phase 2 review を完了。新規 blocker はなし。残件として `quickstart/getting_started` の stale anchor、`architecture.md` の未対応 `mermaid`、JA ページの trailing whitespace を記録しつつ、Phase 3（Visual Navigation Completion）の 4-way 分担を確定。
- **2026-04-18 19:42**: ユーザー判断により Phase 3 を完了扱いとし、Phase 4（Search / Metadata / Mobile UX Completion）へ移行。今回の focused scope は `22-2` meta description、`22-1` unique anchor IDs、`22-13` mobile-friendly table layout に限定し、shared CSS と page-family 単位で 4-way 分担した。
- **2026-04-18 19:45**: `gw-consultant` が Phase 4 方針を承認。条件は「Worker 1 が shared CSS / RST surface を先行確定し、その後に Worker 2-4 が JA/EN ペア単位で個別ページへ展開する」一方向フロー。table CSS 方針の後出し変更は禁止。
- **2026-04-18 20:23**: Phase 4 review 完了。Worker 1-4 の変更に対し、GM が MyST frontmatter を `myst.html_meta` 形式へ統一し、JA `numerical_stability.md` の trailing whitespace を除去。`git diff --check` は clean、`conda run -n gwexpy sphinx-build -b html -D nbsphinx_execute=never docs /tmp/gwexpy-phase4-review-html-2` は warning なしで成功。
- **2026-04-18 20:28**: `gw-consultant` の助言に基づき、次ミッションは Gate A 再開の前提となる code/API unblockers を優先することを確定。Phase 5 を docs wave とは別の「Non-Docs API Follow-up」として開始し、`8-19` Zarr timing metadata 修正と ndscope HDF5 format keyword 整理を 4-way 分担した。順序は ndscope の後方互換 alias を先に収束し、その後に Zarr fail-fast を進める。
- **2026-04-20 20:58**: Coordinator 指示により Phase 7（Truth Ledger 起点の docs/audit resync）を開始。`docs/developers/plans/truth_ledger.md` を新設し、Git 履歴と現行ソースに照らして「実装済みだが統合監査レポートが古い」項目の基線を固定した。初期収録は `15-1`, `15-3`, `17-17`, `17-18`, `17-19`, `17-24`。`17-21` は `Needs revalidation` のまま別管理とする。
- **2026-04-20 20:59**: `337238d4 docs: sync Phase 7 truth-ledger and API docs` を反映。Truth Ledger 追加に加えて、Zarr I/O 仕様・`io_formats`・ScalarField 単位記述・JA tutorial/Colab 導線・branding metadata 周辺の docs を code 実装へ同期した。
- **2026-04-20 21:01**: `8ccc9bd7 test: restore OG metadata constant for docs checks` を反映。branding / OGP 系の docs test を現行 asset 名へ再同期し、Phase 7 PR の検証整合性を確保した。

## 3.5 Phase 2 Worker Ownership

- `gw-worker-1`
  `docs/web/{ja,en}/user_guide/validated_algorithms.md`
  担当: `12-11`, `12-12`, theory-side `17-4`
  制約: URL 維持、public placement 維持、書誌は `validated_algorithms` 内の page-local shared source block に限定、site-wide bibliography 化は禁止

- `gw-worker-2`
  `docs/web/{ja,en}/reference/api/{fields,timeseries,spectral,fitting,preprocessing}.rst`
  担当: primary API category 側の `17-4`
  制約: generic backlink を exact tutorial/theory link に置換。class-page sweep や新規ページ作成は禁止

- `gw-worker-3`
  `docs/web/{ja,en}/user_guide/tutorials/{field_scalar_intro,case_signal_extraction,case_bootstrap_gls_fitting,advanced_arima,case_ml_preprocessing}.ipynb`
  担当: tutorial 側の `17-4`
  制約: 最小限の markdown-cell 編集を優先。notebook 再実行や output 更新は原則禁止

- `gw-worker-4`
  `docs/web/{ja,en}/reference/{index,topics}.rst`
  `docs/web/{ja,en}/reference/api/{index,matrix,frequencyseries,spectrogram}.rst`
  `docs/web/{ja,en}/user_guide/tutorials/index.rst`
  担当: hub/reference surfaces と secondary API category 側の `17-4`
  制約: advanced/theory landing を明示するが URL 移動なし。`examples/index` や homepage teaser には触らない

## 3.6 Phase 3 Worker Ownership

- `gw-worker-1`
  `docs/web/{ja,en}/user_guide/architecture.md`
  `docs/_static/images/phase3/architecture_data_flow.svg`
  担当: `13-9`
  制約: `mermaid` 拡張の追加は禁止。現行 Sphinx 構成で確実に描画できる静的 asset を使う。JA `architecture.md` の trailing whitespace は同時に解消してよい

- `gw-worker-2`
  `docs/web/{ja,en}/examples/index.rst`
  担当: `15-1` 完了判定の仕上げ、必要なら `17-37` の最小版
  制約: canonical gallery は `examples/index` のまま維持。既存サムネイル再利用のみ。homepage 編集は禁止

- `gw-worker-3`
  `docs/web/{ja,en}/index.rst`
  担当: `15-3`
  制約: homepage は teaser layer のまま維持。3枚の teaser card のタイトル・順序・遷移先を `examples/index` と整合させる。`advanced_bruco` への誤着地は解消

- `gw-worker-4`
  `docs/web/{ja,en}/user_guide/tutorials/{case_noise_budget,case_transfer_function,case_active_damping}.ipynb`
  担当: featured case landing loop の補強
  制約: notebook 再実行・output 更新は禁止。冒頭 markdown cell への最小導線追加で、gallery / related API / neighboring workflows への戻り道を作る

## 3.7 Phase 4 Worker Ownership

- `gw-worker-1`
  `docs/_static/custom.css`
  `docs/web/{ja,en}/index.rst`
  `docs/web/{ja,en}/examples/index.rst`
  `docs/web/{ja,en}/reference/index.rst`
  `docs/web/{ja,en}/user_guide/tutorials/index.rst`
  担当: `22-2`, `22-1`, `22-13` の shared surface
  制約: global theme migration / new JS / layout overhaul は禁止。`custom.css` の範囲で table overflow と mobile readability を改善し、RST surface では `.. meta::` と explicit anchor label を使う。Phase 4 の先行レーンとして Worker 2-4 の table 方針の基準を先に確定する

- `gw-worker-2`
  `docs/web/{ja,en}/user_guide/{installation,quickstart,getting_started}.md`
  担当: onboarding guides の `22-2`, `22-1`, `22-13`
  制約: Gate A の distribution messaging は引き続き scope 外。Markdown frontmatter `html_meta` と stable anchor を追加し、`#next-steps` 系の stale link は backward-compatible に解消してよい。JA/EN ペアを同時完了単位とし、table 微調整は Worker 1 の CSS 方針確定後に合わせる

- `gw-worker-3`
  `docs/web/{ja,en}/user_guide/{io_formats,interop,time_utilities,numerical_stability}.md`
  担当: operational guides の `22-2`, `22-1`, `22-13`
  制約: code/API meaning を変えない。table-heavy page は mobile で読めるよう列説明・改行・補助文を調整してよいが、新規 assets や theme-level hacks は禁止。JA/EN ペアを同時完了単位とし、table 微調整は Worker 1 の CSS 方針確定後に合わせる

- `gw-worker-4`
  `docs/web/{ja,en}/user_guide/{scalarfield_slicing,validated_algorithms,architecture,prerequisites_and_conventions}.md`
  担当: advanced guides の `22-2`, `22-1`, `22-13`
  制約: bibliography strategy / URL placement は現状維持。`architecture` の diagram 周辺や `related-documents` 系の anchor 欠落はこのスライスで吸収してよい。JA/EN ペアを同時完了単位とし、table 微調整は Worker 1 の CSS 方針確定後に合わせる

## 3.8 Phase 5 Worker Ownership

- `gw-worker-1`
  `gwexpy/timeseries/io/_registration.py`
  `gwexpy/timeseries/io/ndscope_hdf5.py`
  担当: ndscope HDF5 の format alias 正規化
  制約: 破壊的 rename は禁止。`ndscope-hdf5` は維持しつつ、alias 解決は 1 箇所に集約する。内部の canonical name を増やしすぎない

- `gw-worker-2`
  `gwexpy/timeseries/io/zarr_.py`
  担当: `8-19` の API 修正
  制約: timing metadata 欠落時の silent 1Hz fallback を廃止する。`sample_rate` / `dt` / 明示 override の recovery path を含む clear error にする。zarr open kwargs との衝突に注意

- `gw-worker-3`
  `docs/web/{ja,en}/user_guide/io_formats.md`
  `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md`
  `docs_internal/analysis/webpage/io_formats_グループ設計.md`
  担当: public docs / audit の整合化
  制約: Gate A の installation/distribution messaging には触れない。code で決まる挙動だけを最小差分で反映する。JA/EN ペアで同期する

- `gw-worker-4`
  `tests/timeseries/test_io_ndscope_hdf5.py`
  `tests/io/test_zarr_reader.py`
  `tests/interop/test_zarr.py`
  `tests/fixtures/README.md`
  担当: regression tests / fixture metadata / accepted keyword coverage
  制約: Worker 1, 2 の実装前提に合わせてテストを組む。新しい format canonical を勝手に固定しない。必要なら accepted aliases を並列表記する

## 4. ブロック事項・懸念点

- 現在特になし。
