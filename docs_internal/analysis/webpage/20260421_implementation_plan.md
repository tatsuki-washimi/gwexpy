# [FINAL DETERMINED] 実装計画：GWexpy ドキュメントサイトの総合近代化と不具合修正

ブラウザ監査に基づき、レイアウト、構造、コンテンツ、プロット内言語の完全英数化、およびコピーエディティングを包括的に実施します。

## ユーザーレビューが必要な項目
- **プロット内言語の完全英数化**: 日本語ページであっても、グラフ内のタイトル、軸ラベル、凡例はすべて「半角英数字・記号（英語）」に統一します（文字化けの根本排除と国際化のため）。
- **ランディングページ**: 英語をデフォルトとし、「日本語」ボタンによる明示的な切り替え構造を採用します（注：lang-pill による切り替え UI は既実装済み。内容の見直しに集中）。
- **[CONFIRM] CI での Notebook 実行方針**: docs build (`docs-pr.yml`, `docs-pages.yml`) は `-D nbsphinx_execute=never` を明示指定。Notebook 実行は別ジョブ `scripts/notebook_gen/check_changed_notebooks.py` に分離されている。したがって、docs build を `always` に切り替える追加判断は現時点では不要。

## 実施内容：主要な改修・統制ポイント

### 1. プロットの品質管理と英数字統一 ✅ 完了
- **[DONE] [case_arima_burst_search.ipynb](docs/web/ja/user_guide/tutorials/case_arima_burst_search.ipynb)**
- **[DONE] ほか全 Notebook 資産 (*.ipynb)**
    - グラフ内の文字列（Title, Label, Legend）を日本語から英語（半角英数字）に置換。
    - 修正対象となった 21 ファイル（優先対象 4 件を含む全 58 ファイルを検索・修正済み）：
        - advanced_coupling, advanced_decomposition, advanced_field_analysis, advanced_fitting, advanced_modal_analysis ほか
    - CI 方針は上記の通り確認済み。docs build は非実行、Notebook 実行確認は changed-notebook CI に分離。

### 2. 文章品質とコピーエディティング（日英）
- **[DONE] [case_arima_burst_search.ipynb (ja)](docs/web/ja/user_guide/tutorials/case_arima_burst_search.ipynb)** ✅
    - 全 Markdown セルを完全和訳（プロフェッショナルトーン）。
- **[DONE] [user_guide/prerequisites_and_conventions.md](docs/web/ja/user_guide/prerequisites_and_conventions.md)** ✅
    - 「このページでわかること」→「このページの概要」、「このページの近道」→「目次」
    - 「学習導線」→「学習の流れ」等、不自然な直訳を自然な表現へ修正。
- **[DONE] [index.rst (ja)](docs/web/ja/index.rst)** ✅ — 「解析導線から選ぶ」→「解析目的から選ぶ」
- **[PENDING] ほか主要 RST/Notebook (ja/en)**
    - **英語**: より簡潔でアカデミックなトーンへのリライト（未実施）。

> **[削除済み]** ~~LOCALE FILES (gwexpy.po)~~ — `locales/` ディレクトリはプロジェクトに存在しない。コンテンツは `docs/web/ja/` / `docs/web/en/` の物理分離で i18n を実現しており、gettext による翻訳管理は使用していない。

### 3. ハブページの刷新（ランディングページ・インデックス）
- **[PENDING] [index.rst (Root)](docs/index.rst)**
    - 英語メインに刷新（feature grid, code examples 等）。未実施。
    - 「**日本語ドキュメント**」ボタンは **既実装済み**（lang-pill）。
- **[DONE] [index.rst (ja)](docs/web/ja/index.rst)** ✅
    - カードは `:link:` ディレクティブで既にリンク化済みであることを確認。
    - `FrequencySeriesMatrix` 画像の 404 修正：`/_static/images/phase3/gateway_hero_scientific.png` → `/_static/images/gateway_hero_scientific.png`

### 4. レイアウト修正（CSS / Build Config）
- **[DONE] [custom.css](docs/_static/custom.css)** ✅
    - ヘッダー重なり解消：`.wy-breadcrumbs` に `padding-right: 160px` を追加。
    - コンテンツ幅制限緩和：`.wy-nav-content` の `max-width` を 1100px に拡張。
    - Copy ボタン修正：`div[class^="highlight"]` を `overflow: visible` に設定し、ボタン座標を調整。
- **[DONE] [glossary.rst](docs/web/ja/user_guide/glossary.rst)** ✅（conf.py に代わり実施）
    - 全 35 用語の `(ja)` 接尾辞を削除（Sphinx term ラベルから非表示化）。
- **[CONFIRM] [conf.py](docs/conf.py)**（追加実装不要）
    - 日本語フォント（Noto Sans CJK JP, IPAexGothic）は **既実装済み**（行94-100）。

## 検証計画
- **物理解析検証**: すべてのプロットが英数字で正しく出力されているか、目視で全ページ確認。
- **文章目視検証**: 修正後の文章が日本人開発者・英語ネイティブから見て自然であるか確認。
- **ビルド検証**: `python -m sphinx -b html -W --keep-going -D nbsphinx_execute=never docs /tmp/gwexpy-docs-html` の成功、`linkcheck` によるデッドリンク不在確認。
