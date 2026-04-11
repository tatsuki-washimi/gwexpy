# GitHub Pages Documentation Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GitHub Pages 上の GWexpy ドキュメントを、「何から始めるか分からない」「ページ間の役割が曖昧」「信頼性や実用性の判断材料が足りない」という状態から、初学者・GWpy ユーザー・実務ユーザーが迷わず使える構成へ段階的に改善する。

**Architecture:** 既存の Sphinx + MyST + nbsphinx 構成は維持しつつ、まず情報設計と導線を `docs/web/{ja,en}/` の入口ページと索引ページで整理する。その後、摩擦の大きい個別ページを再設計し、最後にテーマ・検索・メタデータ・アクセシビリティを `docs/conf.py` と追加テンプレート/CSS で整える。

**Tech Stack:** Sphinx, MyST Markdown, nbsphinx, sphinx_design, GitHub Pages

---

## 0. 事前整理

- 対象分析資料:
  - `docs_internal/analysis/webpage/ChatGPTによる54の指摘.md`
  - `docs_internal/analysis/webpage/NotebookLMによる100の指摘.md`
  - `docs_internal/analysis/webpage/Geminiによる60の指摘.csv`
- 3資料の重複を統合すると、主要問題は次の5系統に集約できる。
  - 入口設計不足: トップページと「はじめに」で学習導線が重複し、用途別分岐も弱い。
  - 情報設計の曖昧さ: Tutorial / Case Study / Advanced Guide / Reference の境界が曖昧。
  - 導線不足: チュートリアル、ケーススタディ、API リファレンス、深掘り解説の相互リンクが弱い。
  - 信頼性表示不足: インストールの現状、安定性、検証根拠、更新状況、問い合わせ先が見えにくい。
  - ページ固有の摩擦: `io_formats.md`, `time_utilities.md`, `numerical_stability.md`, `scalarfield_slicing.md`, `validated_algorithms.md`, `architecture.md`, `gwexpy_for_gwpy_users_ja.md` に情報過多または説明不足が集中している。
- 既に解消済みのため今回の優先対象から外す項目:
  - 日本語の `intro_noise.ipynb`, `intro_fitting.ipynb`, `intro_table.ipynb` は既に存在する。
- 注意:
  - 「PyPI/Conda で正式公開する」はドキュメントだけでは完了しない。リリース作業と分離して扱う。
  - 英日で片側だけ直すと導線の非対称が再発するため、原則 `ja/en` ペアで同時修正する。

## 1. 優先度

- `P0`: 入口・ナビゲーション・インストールの再設計。最短で UX 改善効果が出る。
- `P1`: チュートリアル/ケーススタディ/リファレンスの整理と、深掘りページの冒頭要約追加。
- `P2`: テーマ、検索、メタデータ、アクセシビリティ、外部リンク、OGP などの仕上げ。
- `P3`: インタラクティブ図、PDF/ePub、ビジュアルインデックスなど追加価値施策。

## 2. 実行管理ルール

### 2.1 Acceptance Criteria の書き方

各 Task / Step は、実装時に必ず次の4点をセットで定義する。

- 成果物: 更新対象ファイルまたは生成物
- 受け入れ基準: レビュー時に合否判定できる条件
- 検証コマンド: `make html`, `linkcheck`, Notebook 検証など
- レビュー観点: IA、翻訳整合、技術内容、UI のどれを主に確認するか

例:

- 成果物: `docs/web/ja/user_guide/installation.md`
- 受け入れ基準:
  - `ja/en` が対になって更新されている
  - Python バージョン表記が他ページと矛盾しない
  - PyPI/Conda 未対応表記は「将来予定」として注記に退避されている
  - Windows/macOS/Linux または WSL の導入注意がある
- 検証コマンド: `cd docs && make html`
- レビュー観点: 初学者が迷わず導入できるか、現状と矛盾しないか

### 2.2 Owner / Reviewer

Task ごとに少なくとも次の担当を明示する。

- Owner: 実装責任者。本文修正、リンク接続、ビルド修正まで持つ。
- Reviewer: ドキュメント品質レビュー担当。`ja/en` 整合、導線、可読性を確認する。
- Domain Reviewer: 理論・数式・物理・検証ページで必須。`validated_algorithms.md`, `architecture.md`, `physics_models.md`, `numerical_stability.md` を対象とする。

推奨割り当て:

- Task 1-3: Docs Owner + UX/IA Reviewer
- Task 4: Docs Owner + Domain Reviewer
- Task 5: Docs Owner + Frontend/Build Reviewer
- Task 6: Release / CI Owner

### 2.3 CI 品質ゲート

Docs 改修 PR では、最低でも次を品質ゲートに含める。

- `cd docs && make html`
- `sphinx-build -b linkcheck docs docs/_build/linkcheck`
- 軽量な doctest または one-liner 検証
- 主要 Notebook の検証

推奨方針:

- 重依存 Notebook を全件 CI 実行しない
- P0/P1 でユーザー入口に使う Notebook だけを軽量データで検証対象にする
- 実行不能な Notebook は「事前実行済み成果物の表示検証」へ分離する

### 2.4 Notebook 運用ルール

Notebook は次の3種に分類して運用する。

- Lightweight: CI で自動実行する。Quickstart、基本チュートリアル、3行サンプル向け。
- Heavy: ローカルまたは専用ジョブでのみ実行する。外部依存や長時間実行を含む。
- Display-only: 出力はコミット済み、CI では構文・リンク・レンダリングのみを見る。

追加ルール:

- `NBS_EXECUTE` の運用方針を docs/CI 設定に明記する
- Notebook 出力管理は `nbstripout`, `jupyter-cache`, もしくは同等方式で統一する
- Colab/Binder に出す Notebook は依存最小版を別途維持する
- Quickstart の最短サンプルは Colab でも再現できる構成を優先する

### 2.5 ja/en 同期チェック

翻訳ズレを検知するため、少なくとも次のいずれかを導入する。

- `sphinx-intl` ベースの同期ワークフロー
- 見出し数、主要セクション名、toctree 項目の差分を出す簡易スクリプト

最低条件:

- P0/P1 範囲では `ja/en` の主要ページに構造差分が残らない
- 片言語のみ新設したページは TODO ではなくブロッカー扱いにする

### 2.6 品質メトリクス

リリース判定用に、次の定量指標を持つ。

- 重大リンク切れ: 0
- `make html` 失敗: 0
- P0 対象ページの `ja/en` 構造不整合: 0
- Quickstart 最短コードの実行確認: 1件以上
- 主要チュートリアル実行確認率: 80% 以上を目標
- 主要深掘りページの冒頭要約追加率: 100%

### 2.7 PR チェックリスト

各 Docs PR に次を含める。

- [ ] `cd docs && make html` が通る
- [ ] `linkcheck` を通した、または既知の外部要因を明記した
- [ ] 主要 one-liner または Quickstart サンプルを検証した
- [ ] `ja/en` の対応ページを同時更新した
- [ ] API シグネチャ、戻り値、注意事項を必要ページで補強した
- [ ] Pickle など重要なセキュリティ注意を目立つ位置に置いた
- [ ] Owner / Reviewer / Domain Reviewer を明記した

### 2.8 PR サイズとマージ戦略

レビューの停滞を防ぎ、品質を維持するために次の戦略をとる。

- **ページ単位の PR 分割**: 特に Task 4（深掘りページ）などは「1PR = 1ページ（または関連する2〜3ページ）」単位で細かく分割して提出する。
- **差分ベースの同期**: `ja` 側の修正 PR と `en` 側の修正 PR は原則として同時に出し、両方のレビューが完了してからマージする。
- **ベースブランチの活用**: 大規模な IA 変更が重なる場合は、作業用ブランチ（例: `feature/docs-overhaul`）を切り、そこへ小まめに PR をマージしていく運用も検討する。

### Task 1: 入口ページと全体ナビゲーションの再設計

**Files:**

- Modify: `docs/web/ja/index.rst`
- Modify: `docs/web/en/index.rst`
- Modify: `docs/web/ja/reference/index.rst`
- Modify: `docs/web/en/reference/index.rst`
- Modify: `docs/web/ja/examples/index.rst`
- Modify: `docs/web/en/examples/index.rst`
- Modify: `docs/conf.py`
- Create: `docs/_templates/layout.html`
- Create: `docs/_static/custom.css`

**Owner:** Docs Owner
**Reviewer:** UX/IA Reviewer
**Acceptance Criteria:**

- `ja/en` 両方のトップページから重複導線が削減されている
- サイドバー分類と本文の分類が一致している
- 言語切り替えが視認しやすい位置にある
- `cd docs && make html` が通る

- [x] **Step 1: トップページの役割を一つに絞る**

目的: `index.rst` から「Choose Your Guide / Learning Path / Next Steps」の重複を取り除き、Hero + ユーザー別カード + 主要導線の3ブロック構成に整理する。

- [x] **Step 2: グローバル情報設計を `Start / Learn / Reference / Advanced` 系に揃える**

目的: サイドバーの分類と本文中の案内文を一致させる。`examples` と `reference` を埋もれさせずトップレベルに維持する。

- [x] **Step 3: 言語切り替えを目立つ位置に移す**

実装候補: `docs/_templates/layout.html` でヘッダー上部またはサイドバー上部に言語切り替え導線を固定する。可能なら閲覧中ページの対応言語版へ遷移させる。

- [x] **Step 4: サイドバーの視認性を CSS で補強する**

実装候補: `docs/_static/custom.css` に現在地ハイライト、長い toctree の折りたたみ、横にはみ出す表のスクロール、モバイル時の余白調整を入れる。

- [x] **Step 5: Reference / Case Study の入口ページを入口として機能させる**

`reference/index.rst` は `api/index` と `classes` への単純リンクだけなので、用途別入口と索引方針を追記する。`examples/index.rst` は各ケースの価値、対象読者、使う API を短く示す。

- [x] **Step 6: ビルド確認**

Run: `cd docs && make html`
Expected: `ja/en` のトップページで重複セクションが減り、サイドバーと本文の分類が一致する。

### Task 2: オンボーディングとインストールの信頼性を修復する

**Files:**

- Modify: `docs/web/ja/user_guide/getting_started.md`
- Modify: `docs/web/en/user_guide/getting_started.md`
- Modify: `docs/web/ja/user_guide/installation.md`
- Modify: `docs/web/en/user_guide/installation.md`
- Modify: `docs/web/ja/user_guide/quickstart.md`
- Modify: `docs/web/en/user_guide/quickstart.md`
- Create: `docs/web/ja/user_guide/troubleshooting.md`
- Create: `docs/web/en/user_guide/troubleshooting.md`
- Create: `docs/web/ja/user_guide/citation.md`
- Create: `docs/web/en/user_guide/citation.md`
- Create: `docs/web/ja/user_guide/changelog.md`
- Create: `docs/web/en/user_guide/changelog.md`
- Modify: `docs/web/ja/index.rst`
- Modify: `docs/web/en/index.rst`

**Owner:** Docs Owner
**Reviewer:** Docs Reviewer
**Acceptance Criteria:**

- `getting_started.md` がファイル名ではなく人間向け導線になっている
- `installation.md` の Python 要件、配布状況、次アクションが現状と矛盾しない
- `troubleshooting.md`, `citation.md`, `changelog.md` が toctree から到達できる
- Quickstart の最短コードが少なくとも1環境で検証済み

- [x] **Step 1: `getting_started.md` を「学習ロードマップ」へ再設計する**

具体化:

- 生のファイル名リンクを人間向けタイトルへ置き換える。
- 「5分で最初のプロット」「30分で基本操作」「GWpy から移行」の3導線に整理する。
- 冒頭で対象読者、前提、所要時間、到達点を固定フォーマットで明示する。

- [x] **Step 2: `installation.md` を目的別セクションに分離する**

具体化:

- `最小導入 / 推奨導入 / 開発用 / GW extras / トラブルシューティング` の順に並べる。
- `Python 3.9+` と `Python 3.11+` の表記ズレを全体で統一する。
- 未リリースの `pip install gwexpy` / `conda install` は目立つ本文から外し、将来予定として扱う。

- [x] **Step 3: リリース状態と問い合わせ導線を明示する**

具体化:

- `changelog.md` で更新履歴の受け皿を作る。
- `citation.md` に引用方法と BibTeX を置く。
- `troubleshooting.md` に `nds2-client`, `minepy`, `MIC`, GUI, import 系の頻出エラーを集約する。

- [x] **Step 4: `quickstart.md` の最初の成功体験を短くする**

具体化:

- 冒頭に 3 行程度の最短コードを置く。
- 手元ファイル前提ではなく、取得しやすいサンプルデータまたは公開データへの導線を先頭に置く。

- [x] **Step 5: index / toctree に新ページを接続する**

目的: 作成した `troubleshooting.md`, `citation.md`, `changelog.md` を孤立させず、トップページとユーザーガイド両方から辿れるようにする。

- [x] **Step 6: ビルド確認**

Run: `cd docs && make html`
Expected: インストールの現状、学習開始点、困ったときの導線、引用方法がトップレベルから見つけられる。

### Task 3: Tutorials / Case Studies / GWpy 導線を再設計する

**Files:**

- Modify: `docs/web/ja/user_guide/tutorials/index.rst`
- Modify: `docs/web/en/user_guide/tutorials/index.rst`
- Modify: `docs/web/ja/examples/index.rst`
- Modify: `docs/web/en/examples/index.rst`
- Modify: `docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md`
- Modify: `docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md`
- Modify: `docs/web/ja/index.rst`
- Modify: `docs/web/en/index.rst`

**Owner:** Docs Owner
**Reviewer:** UX/IA Reviewer
**Acceptance Criteria:**

- `tutorials/index.rst` だけで学習順と難易度が把握できる
- `examples/index.rst` がケースの価値と利用 API を示している
- `gwexpy_for_gwpy_users_*.md` がインベントリ列挙ではなく差分ガイドになっている
- Tutorials / Examples / Reference の相互リンクが成立している

- [x] **Step 1: チュートリアル索引に難易度・所要時間・前提・得られる成果を追加する**

実装方針: まずは `tutorials/index.rst` 上の各カテゴリ説明とリンク名を整備し、全 Notebook 個別ページの改修は第2段で行う。

- [x] **Step 2: Tutorial と Case Study の境界を明確にする**

具体化:

- `case_*` ノートのうち学習用か実務例かを棚卸しし、`tutorials/index.rst` と `examples/index.rst` のどちらへ出すかを統一する。
- 「イベント同期ケーススタディ」のような残留項目を整理する。

- [x] **Step 3: `examples/index.rst` を単なる toctree からカード型一覧へ拡張する**

各ケースに対して `問題設定 / 利用 API / 難易度 / 期待出力` を短く付ける。既存サムネイル画像は活用し、足りないものは後続タスクで追加する。

- [x] **Step 4: GWpy 向けページを“差分ガイド”に変える**

`gwexpy_for_gwpy_users_ja.md` は現在ほぼ全機能インベントリになっているため、次に置き換える:

- `GWpy と何が違うか`
- `よく使う差分コード`
- `関連チュートリアル`
- `Stable / Experimental`
- `Pickle / I/O / Interop の注意点`

- [x] **Step 5: Tutorials ⇄ Reference ⇄ Examples の双方向リンクを追加する**

目的: 「学ぶ」「実例を見る」「API を調べる」の往復を各入口ページで保証する。

- [x] **Step 6: 目視確認**
296:
297: Expected: 初学者が `tutorials/index.rst` を開いたときに学習順が分かり、GWpy ユーザーが `gwexpy_for_gwpy_users_*.md` から具体例と API に辿れる。
298:
299: - [x] **Step 7: 用語集（Glossary）ページの作成**
300:
301: 目的: ドキュメント全体で統一すべき専門用語（例：Death Floats、分散膨張係数、Field 次元構造など）を `.. glossary::` ディレクティブで定義し、独立したページとして公開する。これにより、他ページからの `:term:` 参照を可能にする。
### Task 4: 摩擦の大きい深掘りページを再設計する

**Files:**

- Modify: `docs/web/ja/user_guide/io_formats.md`
- Modify: `docs/web/en/user_guide/io_formats.md`
- Modify: `docs/web/ja/user_guide/time_utilities.md`
- Modify: `docs/web/en/user_guide/time_utilities.md`
- Modify: `docs/web/ja/user_guide/numerical_stability.md`
- Modify: `docs/web/en/user_guide/numerical_stability.md`
- Modify: `docs/web/ja/user_guide/scalarfield_slicing.md`
- Modify: `docs/web/en/user_guide/scalarfield_slicing.md`
- Modify: `docs/web/ja/user_guide/validated_algorithms.md`
- Modify: `docs/web/en/user_guide/validated_algorithms.md`
- Modify: `docs/web/ja/user_guide/architecture.md`
- Modify: `docs/web/en/user_guide/architecture.md`
- Create: `docs/web/ja/user_guide/physics_models.md`
- Create: `docs/web/en/user_guide/physics_models.md`

**Owner:** Docs Owner
**Reviewer:** Docs Reviewer
**Domain Reviewer:** Physics/Algorithm Reviewer
**Acceptance Criteria:**

- 各ページ冒頭に対象読者、用途、次アクション、要約表または早見表がある
- 数式ページでは変数定義と対応 API リンクが追加されている
- `io_formats.md` で判断に必要な表が冒頭にある
- `time_utilities.md` に関数選択表とシグネチャがある
- `validated_algorithms.md` の「検証済み」の定義が明文化されている

- [x] **Step 1: `io_formats.md` を「最初に判断する表」中心へ組み替える**

具体化:

- 冒頭に `何を読みたいか / どのクラスで読むか / 自動判別の可否 / 必須引数 / 外部依存` の表を置く。
- stub や内部実装パスは一般ユーザー向け本文から外す。
- Pickle の警告、Audio/WAV の時刻扱い、Zarr の危険な既定値、DTTXML の注意点などは “重要事項” 列または警告ブロックへ格上げする。

- [x] **Step 2: `time_utilities.md` に選択早見表を追加する**

具体化:

- `to_gps / from_gps / tconvert` の使い分け表
- 関数シグネチャ
- 対応入力型と出力型の行列表
- timezone / leap second / 配列入力時の戻り値差異の説明

- [x] **Step 3: `numerical_stability.md` をユーザー行動中心に書き換える**

具体化:

- 「通常は何もしなくてよい」「困ったときだけ読む」を冒頭で明示する。
- `Death Floats` は標準用語へ言い換える。
- Safe Log と Adaptive Whitening の API 呼び出し例を明示し、適用範囲も表にする。

- [x] **Step 4: `scalarfield_slicing.md` の説明順を再構成する**

具体化:

- 最初に「なぜ 4D を維持するのか」を図または箇条書きで示す。
- `.squeeze()` の危険性と `axis=` 推奨を追記する。
- ブロードキャスト例では `reshape(3, 1, 1, 1)` の理由を次元構造と結び付けて説明する。

- [x] **Step 5: `validated_algorithms.md` を要約表ベースへ改める**

具体化:

- 冒頭に `アルゴリズム / 対応 API / 検証内容 / 想定読者 / 関連チュートリアル` の表を置く。
- “検証済み” の定義を明文化する。
- 各節から API リファレンスへ戻れるリンクを足す。

- [x] **Step 6: `architecture.md` を分割する**

具体化:

- `architecture.md` は設計・データフロー中心に縮約する。
- 物理モデルや数式の詳細は新規 `physics_models.md` へ逃がす。
- 数式直下に変数定義、節末に対応 API リンクを追加する。

- [x] **Step 7: ビルド確認**

Run: `cd docs && make html`
Expected: 各ページ冒頭で「誰向けか」「何を判断できるか」「次にどこへ行くか」が明確になる。

---

> 以下 Step 8–13 は Quality Audit (`GWexpy_Quality_Audit_Consolidated_Report.md` §4, §7) と
> `2026-04-08_validation_report.md` で確認された未対応項目を追加したものです。

- [ ] **Step 8: Python 要件表記の統一【Critical】**

対象ファイル:

- `docs/web/ja/user_guide/installation.md`
- `docs/web/en/user_guide/installation.md`
- 関連する全 `*.md` / `*.rst` の「Python 3.9+」表記

具体化:

- ドキュメント全体で「Python 3.9+」と書かれている箇所を「Python 3.11+」に統一する。
- `pip install gwexpy[stats]` など、ドキュメント記載の extras 名を `pyproject.toml` の実際の extras 名と照合し、食い違いを修正する。
- `cd docs && make html` でビルドが通ることを確認する。

受け入れ基準:

- `grep -r "3\.9" docs/web/` がインストール要件として残っていないこと
- `pyproject.toml` の `[project.optional-dependencies]` キー名とドキュメントの extras 名が完全一致すること

- [ ] **Step 9: CONTRIBUTING.md の "No Monkeypatching" 方針修正【High】**

対象ファイル: `CONTRIBUTING.md`

具体化:

- 現状の「Monkeypatching 禁止」の記述を確認し、gwexpy が I/O registry（`gwpy` の `io.registry` など）への注入を公式メカニズムとして利用していることを明記する。
- 「外部ライブラリの内部構造を直接書き換えることは禁止。ただし公開 API 経由の登録・注入は推奨パターン」のように整合させる。

受け入れ基準:

- CONTRIBUTING.md の方針文と実装（`gwexpy/interop/`, `gwexpy/io/` 等）が矛盾しないこと

- [ ] **Step 10: MyST admonition 記法の統一【High】**

対象: `docs/web/ja/**/*.md`, `docs/web/en/**/*.md` 内の `> [!NOTE]` / `> [!WARNING]` 形式

具体化:

- GitHub Callout 形式（`> [!NOTE]`, `> [!WARNING]`, `> [!TIP]` 等）を MyST 形式（`:::{note}`, `:::{warning}`, `:::{tip}` 等）へ一括置換する。
- `docs/conf.py` の `nitpicky` または `sphinx-build -W` 相当の設定を確認し、admonition 警告をエラー化する方針を検討する。
- 変更後に `cd docs && make html` でビルドが通ることを確認する。

受け入れ基準:

- `grep -r "\[!NOTE\]\|\[!WARNING\]\|\[!TIP\]" docs/web/` のヒット数が 0 であること
- `make html` が warning なしで通ること（または既知 warning のみ）

- [ ] **Step 11: CLI/GUI ガイドの状態明記【Medium】**

対象ファイル:

- `docs/web/ja/user_guide/` 内の CLI/GUI 関連ページ
- `docs/web/en/user_guide/` 内の CLI/GUI 関連ページ

具体化:

- `gwexpy.cli` がプレースホルダ実装であることを注記する（「将来実装予定。現時点では未公開。」等）。
- `gwexpy.gui`（pyaggui）について、インストール方法と最小起動コマンド（例: `python -m gwexpy.gui`）を `ja/en` で追記する。
- toctree から到達できることを確認する。

受け入れ基準:

- CLI ページで「プレースホルダ」である旨が明示されていること
- GUI ページに最小起動手順があること
- `cd docs && make html` が通ること

- [ ] **Step 12: `gwexpy.time` 活用例の追加【Medium】**

対象ファイル:

- `docs/web/ja/user_guide/time_utilities.md`
- `docs/web/en/user_guide/time_utilities.md`

具体化:

- 既存の選択早見表（Step 2 で追加済み）の直後に、`to_gps`, `from_gps`, `tconvert` それぞれの短い実行例（3〜5 行のコードブロック）を追加する。
- 配列入力時の戻り値差異と timezone/leap-second の注意事項をコード例で示す。
- `ja/en` を同時更新する。

受け入れ基準:

- 各関数に少なくとも 1 件の実行可能なコード例があること
- `cd docs && make html` が通ること

- [ ] **Step 13: ビルド確認**

Run: `cd docs && make html`
Expected: Step 8–12 の変更を含んだ状態でビルドが成功し、Python 要件・extras 名・admonition 記法の不整合が解消されている。

### Task 5: Reference・検索・テーマ・アクセシビリティを整える

**Files:**

- Modify: `docs/web/ja/reference/classes.rst`
- Modify: `docs/web/en/reference/classes.rst`
- Modify: `docs/web/ja/reference/index.rst`
- Modify: `docs/web/en/reference/index.rst`
- Modify: `docs/conf.py`
- Modify: `docs/_static/images/README.md`
- Create: `docs/_static/images/favicon.ico`
- Create: `docs/_static/images/ogp.png`
- Create: `docs/_templates/layout.html`
- Create: `docs/_static/custom.css`

**Owner:** Docs Owner
**Reviewer:** Frontend/Build Reviewer
**Acceptance Criteria:**

- Class Index の並びと訳語ルールが一貫している
- テーマ変更または CSS 補強後も `make html` が通る
- モバイル表示、長表、ダークモードの崩れが主要ページで解消している
- OGP、favicon、外部リンク導線が設定されている
- `linkcheck` で重大リンク切れがない

- [x] **Step 1: Class Index の並びと命名を統一する**

具体化:

- 索引は A-Z または明確な分類ルールのどちらかに寄せる。
- `TimePlaneTransform`, `SeriesMatrix`, Field 系の訳語揺れを修正する。
- `GWexpy / gwexpy / GWExPy` の表記ルールを文書化する。

- [x] **Step 2: `docs/conf.py` で外観と検索の下地を整える**

具体化:

- `html_theme` は IA 改修後に `furo` へ移行するか評価する。
- `html_theme_options`, `html_css_files`, `html_favicon`, `html_logo` 相当を設定する。
- OGP / description / external link icon / copy button / table overflow の仕組みを追加する。

- [x] **Step 3: Intersphinx と外部参照を整える**

具体化:

- GWpy, NumPy, Astropy などへの intersphinx を設定し、依存 API への導線を補う。
- 外部リンクは別タブ + アイコン表示へ寄せる。

- [x] **Step 4: ダークモード・モバイル・長表を検証する**

観点:

- ダークモードのコントラスト
- 横長表のスクロール
- 長い Tutorial / API nav の操作性
- 日本語検索とスニペットの品質

- [x] **Step 5: ビルド確認**

Run: `cd docs && make html`

Expected: 見た目の古さ、検索品質、モバイル崩れ、外部リンク離脱、ブランド未設定の問題が一段下がる。

- [x] **Step 6: 404 防止用のリダイレクト設定**

目的: ファイル名の変更や階層移動（例: `getting_started.md` の改名）によって既存のブックマークが切れるのを防ぐ。

実装: `docs/conf.py` または `sphinx-rediraffe` 等を用いて、旧URLから新URLへのリダイレクトマップを定義する。

- [x] **Step 7: ページ末尾へのフィードバック導線実装**

目的: ユーザーが改善案をすぐ送れるようにする。

実装: `docs/_templates/layout.html` 等を修正し、各ページ末尾に「このページを改善する（GitHubでIssueを作成）」リンクを自動挿入する。タイトルや現在のページURLを自動入力するクエリパラメータを活用する。

---

> 以下 Step 8–13 は Quality Audit (`GWexpy_Quality_Audit_Consolidated_Report.md` §4, §7) と
> `2026-04-08_validation_report.md` で確認された未対応項目を追加したものです。

- [ ] **Step 8: API Reference の大量欠落を解消【High】**

対象ファイル:

- `docs/web/ja/reference/api/index.rst`
- `docs/web/en/reference/api/index.rst`
- `docs/web/ja/reference/classes.rst`
- `docs/web/en/reference/classes.rst`

具体化:

- `api/index.rst` に以下の欠落パッケージを `automodule` または `autosummary` で追加する:
  `gwexpy.time`, `gwexpy.interop`, `gwexpy.cli`, `gwexpy.gui`, `gwexpy.histogram`, `gwexpy.segments` ほか欠落パッケージ計13件。
- `classes.rst` に `TimeSeriesMatrix`, `FrequencySeriesMatrix`, `VectorField`, `TensorField` を追記する。
- 各クラスの `.. rubric:: Methods` セクションに `TimeSeries.hilbert`, `TimeSeries.mix_down`, `TimeSeries.fit_arima` など漏れているメソッドを補う。
- `cd docs && make html` でビルドが通ることを確認する。

受け入れ基準:

- `gwexpy.time` および主要 Matrix クラスが Reference 索引から辿れること
- リンク切れがないこと

- [ ] **Step 9: Notebook 出力の公開品質基準を適用【Major】**

対象ノートブック（優先度高）:

- `advanced_correlation.ipynb`, `advanced_peak_tracking.ipynb`, `advanced_spectrogram_processing.ipynb`
- `case_bruco_ica_denoising.ipynb`, `case_glitch_analysis.ipynb`, `case_hdf5_provenance.ipynb`
- `case_dttxml_calibration.ipynb`, `case_gbd_format.ipynb`
- `intro_interop.ipynb`, `intro_frequencyseries.ipynb`

具体化:

- 各 notebook の出力セルから `UserWarning` / `DeprecationWarning` / `ConvergenceWarning` を除去する。
- `/home/washimi/...`, `/tmp/...` などのローカルパスを含む出力セルを削除またはマスクする。
- 警告を抑制する場合はコード側で `warnings.filterwarnings` を使い、出力セルを再生成する。
- `ja/en` 両ツリーを同時に対応する。

受け入れ基準:

- 対象 notebook の出力セルにローカルパス・個人ユーザー名が含まれないこと
- `UserWarning` / `DeprecationWarning` が出力セルに表示されないこと

- [ ] **Step 10: EN/JA ノートブック同期ポリシーの明文化と適用【Major】**

対象ファイル:

- `docs_internal/` 内に同期ポリシー文書を新規作成（例: `docs_internal/tech_notes/notebook_sync_policy.md`）
- `docs/web/en/user_guide/tutorials/advanced_coupling.ipynb`（JA: 同名ファイル）
- `docs/web/en/user_guide/tutorials/case_seismic_obspy.ipynb`（JA: 同名ファイル）
- `docs/web/en/user_guide/tutorials/advanced_hht.ipynb`（JA: 同名ファイル）

具体化:

- ポリシー文書に以下を明記する:
  - 同名 notebook は原則同一章構成とする
  - 英語版のみ提供する場合は `ja/` に wrapper ページ（`.md`）を置き英語版へ誘導する
  - 意図的な差分がある場合は notebook 冒頭セルに注記を入れる
- `advanced_coupling.ipynb`: JA に欠落している `## 5. Frequency Range Restriction` セクションを追加する。
- `case_seismic_obspy.ipynb`: JA に欠落している `## 5. Multi-channel Seismic Analysis` セクションを追加する。
- `advanced_hht.ipynb`: EN/JA の章構成差分が意図的かどうかを確認し、意図的でなければ同期する。

受け入れ基準:

- `advanced_coupling.ipynb`, `case_seismic_obspy.ipynb` の EN/JA 章構成が一致すること
- ポリシー文書が `docs_internal/` に存在すること

- [ ] **Step 11: en/ ツリー内の日本語 notebook を修正【Minor】**

対象ファイル: `docs/web/en/user_guide/tutorials/case_arima_burst_search.ipynb`

具体化:

- タイトルと導入本文が日本語になっているため、英語に翻訳する（または `ja/` に移動して `en/` には英語版を配置する）。
- `cd docs && make html` でリンク切れが発生しないことを確認する。

受け入れ基準:

- `docs/web/en/` 配下の notebook が英語で記述されていること

- [ ] **Step 12: linkcheck の既知2件を修正【Minor】**

対象:

- `docs/web/ja/user_guide/tutorials/case_dttxml_calibration.ipynb`（`https://dtt.ligo.org/` への内部リンク）
- `docs/web/ja/reference/` または `docs/web/en/reference/` 内の `time.rst`（LALSuite 404 URL）

具体化:

- `https://dtt.ligo.org/` は DNS 解決不能なため `docs/conf.py` の `linkcheck_ignore` に追加する。
- LALSuite 404 URL は有効な LALSuite ドキュメントトップページまたは GitHub リポジトリ URL へ差し替える。

受け入れ基準:

- `sphinx-build -b linkcheck docs docs/_build/linkcheck` で上記2件が出なくなること

- [ ] **Step 13: ビルド確認**

Run:

```bash
cd docs && make html
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

Expected: Step 8–12 の変更を含んだ状態でビルドおよび linkcheck が成功する。

### Task 6: ドキュメント外の依存・リリースブロッカーを切り出す

**Files:**

- Modify: `docs_internal/analysis/2026-04-08_github_pages_improvement_action_plan.md`
- Optional Create: `docs_internal/analysis/2026-04-08_github_pages_release_blockers.md`

**Owner:** Release / CI Owner
**Reviewer:** Project Maintainer
**Acceptance Criteria:**

- ドキュメントだけで完了しない課題が Docs PR から分離されている
- PyPI/Conda、CI、Notebook 実行、品質可視化の各論点に担当が付いている
- リリース判定に使うメトリクスとブロッカー一覧が別紙または本書に残っている

- [x] **Step 1: ドキュメントだけで閉じない課題を分離する**

対象:

- PyPI / Conda 公開
- CI の doctest / linkcheck / coverage 可視化
- 検証バッジや品質指標の公開
- サンプルコードの自動実行保証

- [x] **Step 2: これらを Docs PR と Release / CI PR に分離する**

理由: Docs 作業の完了条件を明確にし、公開サイト改善とリリース工程を混線させないため。

---

> 以下 Step 3–8 は Quality Audit (`GWexpy_Quality_Audit_Consolidated_Report.md` §4, §7) と
> `2026-04-08_validation_report.md` で確認された未対応項目を追加したものです。

- [ ] **Step 3: `examples/paper-figures/` のパス不備を修正【Critical】**

対象ファイル: `examples/paper-figures/` 配下の全スクリプト

具体化:

- `_repo_root` の検出ロジックを確認し、現状 `parents[N]` が誤っている場合は `parents[2]` に修正する。
- 出力先ディレクトリを `docs_internal/publications/` に統一する（スクリプト内のハードコードパスを変数化）。
- クリーンな仮想環境で `python examples/paper-figures/<script>.py` が正常完了することを確認する。

受け入れ基準:

- 出力ファイルが `docs_internal/publications/` に生成されること
- スクリプト内に `/home/washimi/` 等の絶対個人パスが残っていないこと

- [ ] **Step 4: `pyproject.toml` メタデータの整合【Medium】**

対象ファイル: `pyproject.toml`

具体化:

- `license` フィールドを文字列形式（`license = "MIT"`）から PEP 621 準拠の辞書形式（`license = {text = "MIT"}`）へ変更する。
- NumPy の下限を `>=1.23.2`、SciPy の下限を `>=1.10.0` に引き上げ（Python 3.11 を必須とする実態と整合させる）。
- `pip install -e .` が通ることを確認する。

受け入れ基準:

- `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); assert isinstance(d['project']['license'], dict)"` が通ること
- 依存解決エラーが出ないこと

- [ ] **Step 5: `all` extra の自己参照解消【Medium】**

対象ファイル: `pyproject.toml`

具体化:

- `[project.optional-dependencies]` の `all` キーが自身の他 extra を再帰参照している場合は、それをフラットな依存リスト（全パッケージ名の直接列挙）に展開する。
- `pip install -e ".[all]"` が警告なしで通ることを確認する。

受け入れ基準:

- `all` extra が自己参照サイクルを含まないこと
- `pip install -e ".[all]"` が正常終了すること

- [ ] **Step 6: 日本語チュートリアルの欠落補完【High】**

対象ファイル:

- `docs/web/ja/user_guide/tutorials/intro_noise.ipynb`（新規または翻訳）
- `docs/web/ja/user_guide/tutorials/intro_fitting.ipynb`（新規または翻訳）
- `docs/web/ja/user_guide/tutorials/intro_table.ipynb`（新規または翻訳）
- `docs/web/ja/user_guide/tutorials/index.rst`（toctree への追記）

具体化:

- 英語版 `docs/web/en/user_guide/tutorials/intro_noise.ipynb`, `intro_fitting.ipynb`, `intro_table.ipynb` の構成を確認する。
- 各ノートブックを日本語に翻訳するか、日本語 wrapper ページ（`.md` または `.rst`）から英語版へ誘導する構成を選択する。
- いずれの場合も `docs/web/ja/user_guide/tutorials/index.rst` の toctree に接続する。
- `cd docs && make html` でビルドが通ることを確認する。

受け入れ基準:

- 日本語チュートリアル索引から `intro_noise`, `intro_fitting`, `intro_table` に辿れること
- リンク切れがないこと

- [ ] **Step 7: CODE_OF_CONDUCT / SECURITY の連絡先確認【Low】**

対象ファイル: `CODE_OF_CONDUCT.md`, `SECURITY.md`

具体化:

- プレースホルダ（`[INSERT CONTACT METHOD]`, `[INSERT EMAIL ADDRESS]` 等）が残っていないかを確認する。
- 残っている場合は実連絡先（GitHub Issues URL または公開メールアドレス）を設定する。

受け入れ基準:

- 両ファイルにプレースホルダ文字列が残っていないこと

- [ ] **Step 8: リリースチェックリスト全項目の最終確認**

`GWexpy_Quality_Audit_Consolidated_Report.md` §6 の「リリース直前チェックリスト」を参照し、全項目が達成されていることを確認する。

達成確認項目:

- [ ] インストールガイドの Python 要件が `3.11+` で extras 名が正確であること
- [ ] `paper-figures` がクリーン環境で実行でき、指定パスに出力されること
- [ ] `pyproject.toml` の `license` と `requires-python` が仕様通りであること
- [ ] 日本語索引に「基礎」チュートリアルが含まれリンク切れがないこと
- [ ] `gwexpy.time` や主要 Matrix クラスが Reference 索引に含まれること
- [ ] `CODE_OF_CONDUCT.md` と `SECURITY.md` の連絡先がプレースホルダでないこと

## 3. 推奨実施順

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6

## 4. 最小マイルストーン

- Milestone A: トップページ、サイドバー、Getting Started、Installation が整理される。
- Milestone B: Tutorials / Examples / GWpy ガイドの導線が揃う。
- Milestone C: `io_formats`, `time_utilities`, `numerical_stability`, `scalarfield_slicing`, `validated_algorithms`, `architecture` が再設計される。
- Milestone D: テーマ・検索・OGP・アクセシビリティの仕上げが入る。

## 5. 完了条件

- `ja/en` の入口ページで重複導線が解消されている。
- `installation.md` と `getting_started.md` のバージョン・導入方法・次の行動が矛盾しない。
- Tutorials / Examples / Reference 間に双方向リンクがある。
- 深掘りページの冒頭に対象読者・用途・次アクションがある。
- `cd docs && make html` が通る。
- 主要ページのモバイル表示と長表崩れを目視確認している。
- `linkcheck` で重大リンク切れがない。
- Quickstart の最短サンプルが検証済みである。
- P0/P1 対象ページで `ja/en` の主要構造差分がない。

## 6. 実装メモ

- まずは IA と文言整理を優先し、ノートブック本文の大量修正は第2段に回す。
- `docs/conf.py` は最後に大きく触る方が差分を抑えやすい。
- `ja` を直したら対応する `en` を同時に更新する。片側だけ先行させない。
- `CITATION.cff` の整備が必要なら、`citation.md` だけで閉じずリポジトリルートの対応も Task 6 で扱う。

## 7. 付録: 用語集・表記ガイドライン

本計画に基づくドキュメント改修では、以下の用語集と表記ポリシーを基準とする。

### 7.1 表記ポリシー

- クラス／関数／モジュール名は常にコード表記（`` `Name` ``）。訳語は丸括弧で補う。
- 略語は初出で展開して括弧内に短縮形を置く。以後は短縮形を用いる。
- 固有名詞（プロジェクト名・外部ツール名）は英語表記を維持する（ドキュメント中で括弧付き訳語を可）。
- `BruCo` 等のキャピタライゼーションは**実装（ソース）を最終優先**とする。

### 7.2 用語一覧

| 英語（正式） | 日本語（確定） | 表示例（日本語ページ） | 説明 / 備考 |
| :--- | :--- | :--- | :--- |
| `ScalarField` | **スカラー場** | `` `ScalarField` ``（スカラー場） | GWexpy の 4 次元フィールドコンテナ（time, x, y, z）。ドメイン情報と軸メタデータを保持する主要データ構造。 |
| `TimeSeries` | **時系列信号** | `` `TimeSeries` ``（時系列信号） | 単一チャネルの時系列オブジェクト（GWpy 互換）。時刻軸と値を持つ基本データ型。 |
| `TimeSeriesMatrix` | **時系列行列** | `` `TimeSeriesMatrix` ``（時系列行列） | 多チャネルの時系列コンテナ。 |
| `FrequencySeriesMatrix` | **周波数系列行列** | `` `FrequencySeriesMatrix` ``（周波数系列行列） | 周波数ドメインの多チャネルコンテナ。 |
| `FieldList` / `FieldDict` | **FieldList / FieldDict（コレクション）** | `` `FieldList` ``（コレクション） | `ScalarField` 等のコレクション。 |
| `SeriesMatrix` | **シリーズ行列** | `` `SeriesMatrix` ``（シリーズ行列） | 多チャネル時系列コンテナ（別名付与あり）。 |
| ASD | **振幅スペクトル密度（ASD）** | 振幅スペクトル密度（ASD） | Amplitude Spectral Density。初出で英語展開。 |
| PSD | **パワースペクトル密度（PSD）** | パワースペクトル密度（PSD） | Power Spectral Density。 |
| CSD | **相互スペクトル密度（CSD）** | 相互スペクトル密度（CSD） | Cross Spectral Density。 |
| FFT / STFT / CWT / HHT | **FFT / STFT / CWT / HHT（略語）** | CWT（Continuous Wavelet Transform） | 各種時間–周波数変換（初出で展開）。 |
| Adaptive Whitening / AD-Whitening | **適応ホワイトニング（AD-Whitening）** | 適応ホワイトニング（AD-Whitening） | ノイズの局所的正則化／均質化手法。 |
| Whitening | **ホワイトニング** | ホワイトニング | ノイズスペクトルを均す処理の総称。 |
| NaN/Inf propagation (Death Floats) | **NaN/Inf の伝播（旧: Death Floats）** | NaN/Inf の伝播（旧: Death Floats） | 計算中に NaN/Inf が広がる問題。旧用語 `Death Floats` は注記して置換。 |
| VIF (Variance Inflation Factor) | **分散膨張係数（VIF）** | 分散膨張係数（Variance Inflation Factor, VIF） | 多重共線性の指標。 |
| Bruco / BruCo | **BruCo**（表記は実装に合わせる） | `` `BruCo` ``（表記はソースに合わせる） | BruCo 系の解析手法／モジュール名。キャピタライゼーションはコードに合わせる。 |
| Field API | **Field API（フィールド API）** | Field API（フィールド API） | `ScalarField` や関連操作を提供する公開インタフェース。 |
| `TimePlaneTransform` | **時間—周波数平面変換** | `` `TimePlaneTransform` ``（時間—周波数平面変換） | 時間–周波数平面に関する変換ユーティリティ。訳語は用途に合わせて調整。 |
| Safe Log | **Safe Log**（セーフログ） | Safe Log（例: 200 dB） | 対数化での下限を設ける処理。デフォルト値等を明記する。 |
| `SegmentTable` | **セグメントテーブル** | `` `SegmentTable` ``（セグメントテーブル） | セグメント（区間）情報を管理するテーブル。 |
| `CITATION.cff` | **CITATION.cff / 引用方法** | `CITATION.cff` | 論文やデータ引用のメタデータファイル。ルートに配置推奨。 |
| GWOSC | **GWOSC（LIGOオープンデータ）** | GWOSC（LIGO Open Science Center） | LIGO のオープンデータ配信サービス。 |
| `miniSEED` / `GWF` / `GBD` / `MTH5` / `TDMS` / `Zarr` | **各ファイル形式（英語）** | `miniSEED`（フォーマット） | 入出力フォーマットは原則英語表記。必要に応じ訳語を括弧で併記。 |
| Pickle | **Pickle（危険: シリアライズ形式）** | Pickle（警告: 任意コード実行の危険） | Python のシリアライズ形式。読み込みは信頼できる供給源のみに限定する旨を強調。 |
| `tconvert` / `to_gps` / `from_gps` / `LIGOTimeGPS` | **GPS時刻ユーティリティ関数** | `` `tconvert` ``（GPS時刻ユーティリティ関数） | GPS時刻と UTC 等を相互変換する関数群。配列入力や閏秒・タイムゾーンの扱いに注意。 |
| Leap second | **閏秒** | 閏秒（leap second） | 時刻変換で重要な概念。ドキュメントで扱いを明示。 |
| GPS time | **GPS時刻（GPS秒）** | GPS時刻（GPS秒） | GPS のエポックに基づく時刻。 |
| UTC | **協定世界時（UTC）** | UTC（協定世界時） | 標準時系。 |
| MCMC | **MCMC（Markov chain Monte Carlo）** | MCMC（Markov chain Monte Carlo） | ベイズ推定で使うサンプリング手法。初出で展開。 |
| ICA / PCA | **ICA（独立成分分析） / PCA（主成分分析）** | ICA（独立成分分析） | 次元削減・信号分離手法。 |
| Robust ICA | **ロバスト ICA** | ロバスト ICA | 外れ値やノイズに対して頑健な ICA 実装。 |
| ASPIRE / ICRR / LALSuite / PyCBC / Bilby | **外部ツール名（英語）** | `LALSuite`, `PyCBC` | 外部ライブラリ・機関等は英語表記で固定（必要なら括弧で説明）。 |
| 安定性ラベル（Stable / Experimental / Deprecated） | **安定性ラベル** | Stable / Experimental / Deprecated（日本語: 安定 / 実験的 / 非推奨） | APIの安定度を示すラベル。ドキュメント/リファレンスに明示。 |
| `FieldList`, `FieldDict` | **FieldList / FieldDict（コレクション）** | `` `FieldList` `` | 複数の Field を扱うコレクション型。 |
| Safe logging / Warning stacklevel | **警告出力（stacklevel付き）** | `warnings.warn(..., stacklevel=2)` | ユーザへの警告は stacklevel を付けて呼び出し元を示す運用。 |
| `TimeSeriesMatrix` / `FrequencySeriesMatrix` | **時系列行列 / 周波数系列行列** | `` `TimeSeriesMatrix` ``（時系列行列） | 多チャネルコンテナのカテゴリ名（繰返し）。 |

### 7.3 運用メモ

- **訳語変更**は Glossary を基準にして PR 単位で進める（まず Glossary をコミット → 一括修正 PR を分割）。
- **略語の扱い例**：振幅スペクトル密度（Amplitude Spectral Density, ASD） → 以後 `ASD`。
- **Pickle 警告**：I/O ページの冒頭に強調警告ボックスを置くこと。
- **時刻ユーティリティ**：`tconvert` 等は「GPS時刻ユーティリティ関数」として総称ページを設け、個別シグネチャを掲載。
- **BruCo**：実装上 `BruCo` なのか `Bruco` なのか確認し、Glossary で一括固定する。

