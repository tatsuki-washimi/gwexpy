# GWexpy エコシステム positioning と外部ツール interop — 計画書

> Last-updated: 2026-08-01 (rev 3 — Phase 1 実装完了・検証済み)
> Reviewer Status: **approved**（ユーザー承認済み 2026-08-01）

Status: in-progress

- **Phase 1（docs）: completed** (verified: `sphinx-build -b html docs_redesign` および
  `-D language=ja` の両方が EXIT=0、`pytest tests/docs/` が 105 passed / 4 skipped、
  `sphinx-build -b linkcheck docs_redesign` で新規ページ由来の broken・redirect ともに 0 件、
  `msgfmt --check` が対象 5 カタログすべて EXIT=0)
- **Phase 2〜4（実装）: planned**（未着手）

対象テーマ:
- **Phase 1（docs）= Ecosystem & Interoperability backlog**（#404 の過去の v0.2.0 割当とは切り分ける）
- **Phase 2〜4（実装）= milestone 未割当の backlog**（ユーザー決定:「docs のみ先行、実装は未定」）

## Phase 1 の実施結果（2026-08-01）

ブランチ `docs/ecosystem-positioning`。

新規:
- `docs_redesign/explanation/ecosystem.md`
- `docs_redesign/locales/ja/LC_MESSAGES/explanation/ecosystem.po`（135 エントリ全訳）
- `docs/developers/LICENSES_THIRD_PARTY.md`

変更:
- `docs_redesign/explanation/index.md`（グリッドカード + toctree）
- `docs_redesign/explanation/gwexpy_for_gwpy_users.md`（positioning 段落）
- `docs_redesign/explanation/roadmap.md`（Mid-Term 具体化 + 相互リンク）
- `docs_redesign/how-to/interop.md`（Related Pages）
- 上記 4 件に対応する ja カタログ
- `README.md`（`## Where gwexpy Fits`）、`ROADMAP.md`（Ecosystem & Interoperability backlog）、
  `CONTRIBUTING.md`（Third-Party Code and Licenses）、`docs/developers/index.rst`

計画からの差分:
1. **ライセンス表記の訂正** — 計画時点で未確認だった GWpy を BSD-3-Clause と誤記しかけたが、
   `gwpy/gwpy@main` の LICENSE 本体を読み **GPL-3.0** と確定。spicypy は GitLab の
   `LICENSE.txt` 本体、GWDama は PyPI sdist 同梱の `LICENSE` 本体で確認（Apache-2.0 / MIT）。
   pemcoupling は GitHub API が `license: NONE` を返すことを確認し #404 の判断を追認。
2. **`docs/developers/index.rst` への登録を追加** — 計画に含めていなかったが、
   `developers/**` は `docs/conf.py` の `exclude_patterns` にあり Sphinx ビルド対象外のため、
   GitHub 閲覧時の導線としてのみ追加（`-W` ビルドへの影響なし）。
3. **readthedocs URL を最終形に固定** — linkcheck が redirect を報告したため、
   `https://gwdetchar.readthedocs.io/` 等を `/en/stable/`・`/en/latest/` へ変更。
4. **`sphinx-intl update` の巻き添え更新を除外** — `about/changelog.po` ほか 3 件の更新と
   `locales/ja/LC_MESSAGES/reference/api/` の新規生成はスコープ外のため revert・削除した。

既知の残り（linkcheck EXIT=1 の理由、いずれも本変更と無関係の既存 broken）:
`about/changelog` の GitHub compare/tag URL 3 件、`about/citation` の
`gwpy.github.io/docs/stable/citing.html`、`reference/api/gwexpy.interop.openems_` の
`openems.de` — 別 issue 化の候補。

---

## Context

添付 transcript（`gwexpy_ecosystem_conversation_transcript.md`）は、GWexpy の
**ecosystem positioning**（GWpy / spicypy / GWDama / gwdetchar 系との棲み分け）と
**外部ツール interop 戦略**を詰めた議論の記録である。レイヤー分割、比較表、
core/optional/docs-only の切り分け、ライセンス方針、`gwexpy.io.gwdama` /
`gwexpy.interop.spicypy` という namespace 案、Differometor converter の API 案まで到達している。

しかし、その成果は **リポジトリに一切反映されていない**:

| 事実 | 確認方法 |
|---|---|
| 追跡ファイル内に `differometor` / `gwdama` / `spicypy` / `pemcoupling` の言及が **0 件** | `git grep -il -E 'differometor\|gwdama\|spicypy\|pemcoupling'` → 出力なし |
| `README.md` の `## Why gwexpy?` は自己主張 4 箇条のみで、他ライブラリとの比較・棲み分けは皆無 | README 全文確認 |
| Differometor issue #423–#427 は 5 本すべて open、コード・PR なし | `gh issue view 423..427` |
| #404（pemcoupling/GWDama ライセンス方針）は 2026-08-01 当時 v0.2.0 の release bucket に割り当てられていた。現在は open / milestone なしで、issue 本文も Ecosystem backlog を参照 | `gh issue view 404` |
| #404 が更新対象に挙げる `loadmap_v0.2.0.md` は **リポジトリに存在しない** | `find` で該当なし |

さらに計画立案中に、**既存 issue の前提が外部実態と食い違っている**ことが判明した（Phase 3 参照）。
#423 の「調査」を先に決着させないと、誤った API 設計をそのまま実装することになる。

意図する成果:

1. 議論の結論を、公開ドキュメント（`docs_redesign/`）と開発者向けポリシーとして定着させる
2. ライセンス方針を明文化し、All-Rights-Reserved コードの誤流用リスクを閉じる（#404 の deliverable）
3. GWDama / Differometor / spicypy の各 interop を、**実態に即し、既存の contract テスト網に適合する**
   実装可能な設計として backlog に置く

---

## rev 1 からの重大な訂正（調査で判明）

初版は成果物を `docs/web/{en,ja}/user_guide/` に置く計画だったが、**これは誤りだった**。

| 訂正点 | 実態 |
|---|---|
| **公開 docs は `docs_redesign/`** | `.github/workflows/docs-pages.yml` が `docs_redesign/**` の push で発火し、gh-pages の `/docs/`(EN) と `/docs/ja/`(JA) に publish する。**`docs/web/` は publish されていない**（`scripts/generate_docs_redirect_stubs.py` が旧 URL → 新 URL の meta-refresh stub を生成する凍結ツリー） |
| **翻訳方式が違う** | `docs_redesign/` は **gettext 単一ソース**（`conf.py` で `locale_dirs=["locales/"]`, `gettext_compact=False`）。ja は `docs_redesign/locales/ja/LC_MESSAGES/**/*.po` を訳す。`docs/web/` の「en/ja 並行 md」方式ではない |
| **Diátaxis 構成** | `tutorials/` `how-to/` `reference/` `explanation/` `about/`。棲み分け解説は **`explanation/`** が正しい格納先 |
| **「未測定の優先順位」は公開ページで禁止** | `tests/docs/test_docs_redesign_public_content.py:36-37` が `assert "What to Prioritize First" not in page` と `assert "not a ranking by usage statistics" in page` を強制。transcript の deliverable「連携優先順位」を**公開ページには書けない** |
| **旧ツリーも死んでいない** | `tests/interop/test_interop_docs_contract_sync.py:10-13` と `tests/io/test_io_docs_contract_sync.py:10-13` は**いまも `docs/web/{en,ja}/user_guide/{interop,io_formats,installation}.md` を assert する**。interop/IO の表を触る Phase 2/3 では新旧**両ツリー**の更新が必要 |

---

## 調査で確定した外部事実（一次情報で検証済み）

| 対象 | 事実 | 出典 | 計画への影響 |
|---|---|---|---|
| **Differometor** | MIT。PyPI に `differometor` として公開済み | GitHub リポジトリ / README | ライセンス上は安全（コードコピーはしない方針を維持） |
| **Differometor** | `df.run(S, params, values)` は **`carrier, signal, noise, detector_ports, *_` という生の JAX 配列タプル**を返す。result dataclass / namedtuple は**存在しない**。周波数軸は戻り値ではなく**入力側**に `[("f","frequency")]` として渡す | README のコード例 | **#424–#426 の前提（result オブジェクトの属性名 sniffing）は成立しない** → issue 本文の改訂が必要 |
| **Differometor** | 依存は JAX + Optax（GPU 版は CUDA） | README | core 依存にしない。テストはモック（既存 interop の主流戦略） |
| **spicypy** | **Apache-2.0**（transcript に記載なし、今回確認） | GitLab プロジェクトページ | 記載は安全。本計画ではコード連携しない |
| **GWDama** | **MIT**、最新 0.6.0（2025-10-22）、`requires-python >=3.9,<3.13` | PyPI JSON API | #404 の記載（MIT）と一致 |
| **GWDama** | 依存に **`gwpy>=3.0` と `lalsuite>=6.73`** を含む | PyPI `requires_dist` | **gwdama パッケージ自体には依存しない**設計が妥当 |
| **GWDama** | dataset attrs は `sample_rate`, `t0`, `channel`, `unit`。group は `dts_key` 由来の階層。`GwDataManager(dama_name, storage={'mem','tmp','disk'}, mode=...)` | 公式 docs (`gwdatamanager.html` / `dataset.html`) | **素の h5py だけで読み書き可能** → 依存ゼロで reader/writer が書ける |
| **pemcoupling** | LICENSE ファイルなし = All Rights Reserved（transcript の「GPLv3」は誤り） | #404 に記録済み | ドキュメントでも「GPLv3」と書かない。概念のみ参照 |

---

## 調査で確定したコードベース側の前提

### interop レイヤー

| 項目 | 実態 |
|---|---|
| 配置 | `gwexpy/interop/<tool>_.py`（末尾アンダースコア必須）。47 アダプタ + インフラ 7 |
| optional dep ガード | **`HAS_*` フラグは使わない。関数本体の先頭で `require_optional("<pkg>")` を呼ぶ**（`_optional.py:145-194`）。モジュールトップで外部を import しない（`__init__.py` が eager import するため） |
| extra 方針 | `_EXTRA_MAP[pkg] = None` で「bare install を案内」。**finesse / PySpice / skrf / torch / mne などニッチ・重量級はすべて extra を作らず bare install** |
| 型判定 | **完全に duck typing**。外部型への `isinstance` はゼロ。`getattr(obj, "x", default)` / `try-except AttributeError` |
| 返り値ディスパッチ | `cls` を第 1 引数で受け、`ConverterRegistry.get_constructor(...)` で Series / Dict / Matrix を分岐（`finesse_.py:84-133` が定型） |
| unit | 外部が単位を持たなければ **`unit=` を必須引数として要求**（finesse 流）。導出可能なら決定論的に導出（gwinc 流） |
| テスト | `tests/interop/test_interop_<tool>.py`。**重量級は MagicMock + `autouse` fixture で `require_optional` を monkeypatch**（`test_interop_finesse.py:123-130`）。モックは**テストファイル内に定義**（`tests/fixtures/` に interop 用の置き場は存在しない） |
| **契約テスト** | `docs/developers/contracts/public_interop_contract.json` と `tests/interop/test_interop_contract.py` / `test_interop_docs_contract_sync.py`。**`interop.__all__` に 1 行足すだけで CI が落ちる** |
| `VALID_EXTRAS` | `tests/interop/test_interop_contract.py:27-34` = `{audio, control, gw, netcdf4, seismic, zarr}`。新 extra を作るならここも更新必須 |

### I/O レイヤー

| 項目 | 実態 |
|---|---|
| 配置 | フォーマット実装は **`gwexpy/<family>/io/`**。`gwexpy/io/` は gwpy 互換シム + 共通ヘルパ置き場 |
| 登録 | `gwexpy/timeseries/io/_registration.py:56` の `register_timeseries_format(...)`。`auto_adapt=True` で TimeSeriesDict の reader/writer から TimeSeries / TimeSeriesMatrix 版を自動生成 |
| 登録トリガー | `gwexpy/timeseries/io/__init__.py:12-29` の import リスト。**追加を忘れると一切登録されない** |
| 命名慣習 | 階層 HDF5 は `hdf.<vendor>`（`hdf.ndscope`）、XML は `xml.<vendor>`、サブバックエンドは `<fmt>.<backend>` → **GWDama は `hdf.gwdama` が完全に慣習どおり** |
| 手本 | `gwexpy/timeseries/io/ndscope_hdf5.py`（357 行、`identify_*` / `read_timeseriesdict_*` / `write_timeseriesdict_*`）。docstring 冒頭にスキーマ図を書くのが定型（`:1-19`） |
| identifier の設計方針 | **「壊れたファイルを弾く条件」を identify に入れない**（`ndscope_hdf5.py:104-117` に理由がコメントされている）。identifier は canonical 名にのみ登録され、alias には登録されない（`_registration.py:313-324`） |
| タイミング欠落 | 黙って 0/1 にせず、**例外か `UserWarning`**。`gwexpy/interop/base.py:34` の `resolve_timing()` が最も筋の良い共通実装だが、**`timeseries/io/` 側ではまだ未使用**（zarr は独自実装、ndscope は例外送出） |
| gwpy 注入 kwargs | reader は `{"start","end","pad","gap","nproc","scaled"}` を除去する必要がある（`netcdf4_.py:179-180`） |
| **未知 attrs の保全** | **存在しない**。ndscope / zarr / nc の reader はいずれも未知 attrs を黙って読み捨てる。Matrix のみ `attrs_json` で一括退避（`series_matrix_io.py:187-192`、JSON 化不能な値は捨てる） |
| **契約テスト** | `docs/developers/contracts/public_io_contract.json`（schema v3、25 フォーマット）+ `tests/io_conformance/`。**conformance generator は `gwexpy` の import を AST で禁止されている**（`tests/io_conformance/conftest.py:83-104`） |
| 拡張子衝突 | `.h5`/`.hdf5` を `hdf5` / `hdf.ndscope` /（新）`hdf.gwdama` が取り合う。`hdf5` は `public_auto_identify: false` |

### docs / 計画書

| 項目 | 実態 |
|---|---|
| 公開ソース | `docs_redesign/`（Diátaxis: `tutorials/` `how-to/` `reference/` `explanation/` `about/`） |
| 翻訳 | gettext。`sphinx-build -b gettext` → `sphinx-intl update -p <tmp> -l ja` → `locales/ja/LC_MESSAGES/**/*.po` を訳す |
| explanation の toctree | `docs_redesign/explanation/index.md:74-82`（`architecture` / `prerequisites_and_conventions` / `gwexpy_for_gwpy_users` / `roadmap`） |
| 公開 roadmap | `docs_redesign/explanation/roadmap.md`。**バージョン番号を書かない方針**。Mid-Term に既に "Expanded interoperability guides for external scientific Python libraries" がある |
| 開発者 roadmap | ルート `ROADMAP.md`（43 行）。v0.2.0 節に interop 項目なし。**terrain/DEM の deferred future-theme 計画と既に切り分けられている** |
| 計画書テンプレ | `docs/developers/plans/active/2026-07-31-terrain-scalarfield-io-design.md`（`Last-updated` / `Reviewer Status` / `Status:` / 対象マイルストーン / 末尾に「イシュー本文ドラフト」） |
| Status 語彙 | `planned` / `in-progress` / `completed (verified: <command>)` の 3 種厳格。`completed` で `verified:` 省略は規則違反 |

---

## スコープと段階

| Phase | 内容 | 成果物 | milestone |
|---|---|---|---|
| **1** | エコシステム positioning 文書 + 第三者ライセンス方針 | docs のみ（コード変更ゼロ） | Ecosystem backlog（#404 の旧割当を再評価） |
| **2** | GWDama HDF5 interop 設計 | 設計 + issue ドラフト（実装は未着手） | backlog |
| **3** | Differometor interop 再設計 + #423–#427 改訂 | 改訂 issue 本文 + 設計 | backlog |
| **4** | spicypy 連携 | Phase 1 の docs に吸収（コード不要） | — |

**Phase 1 だけが今すぐ実行する作業**である。Phase 2–4 は設計として確定させ issue 化するが、
milestone は割り当てない。

---

## Phase 1: エコシステム positioning 文書（実行対象）

### 1-A. 新規 `docs_redesign/explanation/ecosystem.md`

`how-to/interop.md` が「**どう変換するか**」を答えるのに対し、`explanation/ecosystem.md` は
「**GWexpy はどのライブラリと何が違い、どこで棲み分けるのか**」を答える。Diátaxis 上、
理解指向は `explanation/` が正しい。隣に `gwexpy_for_gwpy_users.md`（GWpy との差分）と
`architecture.md` があり、「GWpy との関係」を語る場所として一貫する。

frontmatter は既存ページの慣行に合わせる:

```markdown
---
myst:
  html_meta:
    description: "How gwexpy relates to GWpy, spicypy, GWDama, and the wider gravitational-wave Python ecosystem."
---
```

節構成:

1. **導入** — GWexpy が GWpy 拡張であり、GWpy の公式構成要素ではないこと（README L17 の既存宣言と整合）
2. **レイヤー表**

   | Layer | Primary package |
   |---|---|
   | Standard GW data objects / base API | GWpy |
   | Matrix / multichannel / experiment-oriented analysis containers | **gwexpy** |
   | Signal processing / control systems | spicypy |
   | HDF5 data preparation / archive / ML-ready datasets | GWDama |
   | Detector characterization / summary pages / veto workflows | gwdetchar, gwsumm, gwvet, hveto |
   | Trigger generation / search / PE pipelines | pyomicron, PyCBC, bilby, pygwb, pycWB |
   | Data access | gwosc, gwdatafind |

3. **4 パッケージ詳細比較表** — GWpy / gwexpy / spicypy / GWDama を「一言でいうと / 主レイヤー /
   中心データ構造 / 設計思想 / 永続化 / gwexpy から見た関係」の 6 軸で（transcript の表を採用）
4. **関係の分類** — 各パッケージを `基盤` / `補完` / `参考` / `連携候補` / `非対象` に分類。
   分類語 5 種はページ冒頭で定義する
5. **gwexpy の差別化ポイント** — GWpy-compatible extension / Matrix-native / Typed analysis results /
   Broad I/O but not HDF5-only / Generic source→target coupling / Notebook-first
6. **Scope boundaries** — core に入れないものの明示（detector・site 固有の運用パイプライン、
   operator 向け HTML レポート、HTCondor/sitewide orchestration、veto/search パイプライン本体、
   trigger generator 本体、PEM→DARM 固定 API）
7. **Interoperability status** — 3 段階。`Implemented`（Finesse 3 / pygwinc / PySpice / ObsPy /
   LALSuite / PyCBC / SimPEG / MTH5 …、`how-to/interop.md` へリンク）、
   `Planned`（GWDama HDF5, Differometor）、`Docs-only`（spicypy, gwdetchar, gwsumm, hveto, pyomicron, pemcoupling）
8. **Third-party code policy** — 1-C への 2〜3 行の要約とリンク
9. **Related pages** — `how-to/interop.md`, `how-to/io_formats.md`, `explanation/roadmap.md`

**書かないこと（`test_docs_redesign_public_content.py` の制約と既存方針から）**:
- **連携の優先順位づけ・ランキング**。同テストが「未測定のランキングを公開ページに書くこと」を
  禁じている。transcript の「連携優先順位」は**開発者向けの ROADMAP.md 側**へ置く（1-D 参照）
- 他パッケージの品質評価・優劣の断定。事実ベースの役割記述に留める
- 未検証のライセンス表記。確認できないものは「要確認 (unverified)」と明記する
- transcript にあった対応度 A–E の 5 段階評価（粒度が細かく陳腐化しやすい）

なお `interop.md` に倣い、順序が優先度ではないことを示す定型句
（"not a ranking by usage statistics" と同趣旨の一文）を明示的に入れる。

### 1-B. 日本語翻訳（gettext）

```bash
sphinx-build -b gettext -D nb_execution_mode=off docs_redesign /tmp/gwexpy-gettext
sphinx-intl update -p /tmp/gwexpy-gettext -l ja
```

生成される `docs_redesign/locales/ja/LC_MESSAGES/explanation/ecosystem.po` を全訳。
toctree を触るので `explanation/index.po` も再生成・再訳が必要。

### 1-C. 新規 `docs/developers/LICENSES_THIRD_PARTY.md`（issue #404 の deliverable）

| Project | License | 確認方法 | Policy |
|---|---|---|---|
| pemcoupling | LICENSE ファイルなし = **All Rights Reserved** | リポジトリ直接確認 | コード・構造・実装の流用一切なし。ドメイン概念のみ独立に再設計 |
| GWDama | **MIT** | PyPI metadata + リポジトリ LICENSE | 法的には再利用可だが、方針として**薄い interop / reader のみ**。コードコピーなし |
| spicypy | **Apache-2.0** | GitLab プロジェクトページ | 概念・API 設計・docs 構成の参考のみ。コードコピーなし |
| Differometor | **MIT** | リポジトリ LICENSE | 薄い変換アダプタのみ。optimizer は再実装しない |

併記する事項:
- 「LICENSE ファイルが存在しない = All Rights Reserved であり、GPL より厳しい」という判断根拠
- ライセンス確認は PyPI classifier や README バッジではなく **LICENSE ファイル本体**で行う手順
- 確認できない場合は「要確認 (unverified)」と書き、推測で埋めない

### 1-D. 既存ファイルの改訂

| ファイル | 変更 |
|---|---|
| `docs_redesign/explanation/index.md` | `:74-82` の "Design and concepts" toctree に `ecosystem` を追加（`gwexpy_for_gwpy_users` の後）+ グリッドカード追加 |
| `docs_redesign/how-to/interop.md` | `## Related Pages` に `- [Where GWexpy sits in the GW Python ecosystem](../explanation/ecosystem)` を 1 行追加。**比較表本体は置かない** |
| `docs_redesign/explanation/gwexpy_for_gwpy_users.md` | `## Treat direct I/O and external-library conversion as separate guides` の直前に、3〜5 行の positioning 段落 + `ecosystem` へのリンク（表は置かない） |
| `docs_redesign/explanation/roadmap.md` | Mid-Term の既存行 "Expanded interoperability guides for external scientific Python libraries" を具体化（**バージョン番号は書かない**既存方針を維持）。`.po` 更新必須 |
| `README.md` | `## Why gwexpy?`(L53) の直後に `## Where gwexpy fits` を新設。5〜7 行 + docs リンク。**表は置かず** `ecosystem.md` に一本化（二重管理回避）。README は Sphinx 対象外なので EN のみ |
| `ROADMAP.md`（ルート） | 「Ecosystem & Interoperability（backlog）」節を新設。**ここに連携優先順位を置く**（開発者向け文書であり、公開ページのランキング禁止制約の対象外）。同時に現行 v0.2.0 節と deferred terrain/DEM 計画の切り分けを注記する |
| `CONTRIBUTING.md` | `## Design Principles: Modular Extensibility`(L47) の後に「Third-party code policy」小節を追加。`LICENSES_THIRD_PARTY.md` へのリンクと vendoring 禁止を明記 |

**`docs/web/` は触らない。** 凍結された非公開ツリーであり、`check_docs_sync.py` の en/ja 見出し数一致制約と
`check_terms.py`（ja のみ）を余計に背負うだけで公開されない。

### 1-E. 副次的に発見した既存バグ（別 issue、Phase 1 の対象外）

`gwexpy/interop/gwinc_.py:120-123` の docstring Examples は
`FrequencySeries.from_gwinc_budget("aLIGO")` / `FrequencySeriesDict.from_gwinc_budget("aLIGO")` と
書いているが、**これらのクラスメソッドは存在しない**（実際の呼び出しは
`gwexpy.interop.from_gwinc_budget(FrequencySeries, "aLIGO")` のみ）。
`pyproject.toml:338` の pytest 設定に `--doctest-modules` がないため検出されていない。
→ 独立した bug issue として起票する（本計画では修正しない）。

---

## Phase 2: GWDama HDF5 interop（設計確定・実装は backlog）

### 中核となる設計判断

> **gwdama パッケージには依存しない。素の h5py だけで GWDama 形式の階層 HDF5 を読み書きする。**

根拠: GWDama の runtime 依存は `gwpy>=3.0` に加えて **`lalsuite>=6.73`** を含み、optional extra
としても重すぎる。一方 GWDama が書く HDF5 は素の h5py で完全に読める（dataset attrs は
`sample_rate`, `t0`, `channel`, `unit`）。したがって **フォーマット互換**を実装し、
**パッケージ依存**は持たない。`gwdama` 本体は cross-check テスト用の optional 依存に留める。

この方針は副次的に、conformance generator が `gwexpy` を import できない制約
（`tests/io_conformance/conftest.py:83-104`）とも噛み合う — generator は h5py だけで
GWDama 形式のフィクスチャを作れる。

これは transcript の `read_gwdama` / `write_gwdama` 案（gwdama への依存を暗黙に想定）からの
意図的な変更である。

### モジュール構成

**新規** `gwexpy/timeseries/io/gwdama_hdf5.py`（`ndscope_hdf5.py` を手本に 3 関数構成、
docstring 冒頭にスキーマ図を書く）:

```python
def identify_gwdama_hdf5(origin, filepath, fileobj, *args, **kwargs) -> bool: ...
def read_timeseriesdict_gwdama(source, *, channels=None, group=None, **kwargs): ...
def write_timeseriesdict_gwdama(tsdict, target, *, group=None, overwrite=False, **kwargs): ...

register_timeseries_format(
    "hdf.gwdama",
    aliases=("gwdama", "gwdama-hdf5"),
    reader_dict=read_timeseriesdict_gwdama,
    writer_dict=write_timeseriesdict_gwdama,
    magic_identifier=identify_gwdama_hdf5,
    extension="hdf5",
)
```

**変更** `gwexpy/timeseries/io/__init__.py:12-29` の import リストに `gwdama_hdf5,` を追加
（**忘れると一切登録されない**）。

### 属性マッピング契約

| GWDama HDF5 attrs | GWexpy 側 | 欠落時 |
|---|---|---|
| `sample_rate` | `TimeSeries.sample_rate` | `dt` も無ければ `ValueError`（`ndscope_hdf5.py:39-89` の厳格方針に合わせる） |
| `t0` | `TimeSeries.t0` | `UserWarning` のうえ 0.0（`zarr_.py:164-176` の方針） |
| `channel` | `TimeSeries.channel` / dict key | dataset のパス名を使用 |
| `unit` | `TimeSeries.unit` | dimensionless |
| group / subgroup | `group=` selector で選択。省略時は全走査 | — |
| 上記以外の attrs | **保全方式は新規設計項目**（下記） | — |

### 未解決の設計判断（実装前に決める）

1. **未知 attrs の保全** — GWexpy には現状これを行う仕組みが**存在しない**（ndscope / zarr / nc は
   すべて黙って読み捨てる）。GWDama は任意 attrs をユーザーが自由に入れる形式なので、
   `gwexpy/io/utils.py:150` の `set_provenance()` に載せるか、Matrix の `attrs_json` 方式を
   1D シリーズにも広げるかを選ぶ必要がある。**Phase 2 の最初のタスク**
2. **`.h5` 拡張子衝突** — `hdf5` / `hdf.ndscope` / `hdf.gwdama` が同じ拡張子を取り合う。
   GWDama の root attrs（`time_stamp` / `dama_name` 等）をマジックとして identify するか、
   `hdf5` と同じく `public_auto_identify: false` にするかを最初に決める。
   identify に「壊れたファイルを弾く条件」は入れない（`ndscope_hdf5.py:104-117` の設計判断）
3. **任意深さのネスト** — `gwexpy/io/hdf5_collection.py` の manifest 方式（`gwexpy_keymap` /
   `gwexpy_order` / `gwexpy_layout`）が最も近い基盤だが**現状 1 階層固定**。拡張するか、
   GWDama reader 内で自前に平坦化するかを決める

### land すべきファイル一式

| 分類 | ファイル |
|---|---|
| 実装 | `gwexpy/timeseries/io/gwdama_hdf5.py`（新規）、`gwexpy/timeseries/io/__init__.py`（import 追加） |
| 契約 | `docs/developers/contracts/public_io_contract.json`（`formats` に 1 エントリ）、`public_io_contract.md`（"Boundary Decisions" に `### hdf.gwdama` 節。`### hdf.ndscope` が手本） |
| conformance | `tests/io_conformance/generators/gwdama.py`（新規、**gwexpy を import しない**）、`generators/__init__.py` の `GENERATOR_SPECS`、`contract.py` の `fixture_generator_by_format` |
| テスト | `tests/io_conformance/test_read_conformance.py` / `test_write_roundtrip.py`、`tests/io/test_gwdama_reader.py`（エッジケース）、`pytest.importorskip("gwdama")` による cross-check |
| docs（公開） | `docs_redesign/how-to/io_formats.md` + `locales/ja/.../io_formats.po` |
| docs（契約テスト対象・旧ツリー） | `docs/web/en/user_guide/io_formats.md`、`docs/web/ja/user_guide/io_formats.md`（`test_io_docs_contract_sync.py` が両方を assert） |
| その他 | `SUPPORTED_IO_MATRIX.md`、`CHANGELOG.md` |

extra は不要（h5py は `pyproject.toml:46` で base 依存）。

### 検証

```bash
conda run -n gwexpy python -c "import gwexpy; gwexpy.register_all()"
conda run -n gwexpy python scripts/ci/run_gate.py io-contract
conda run -n gwexpy python scripts/ci/run_gate.py io-conformance
```

---

## Phase 3: Differometor interop 再設計（#423–#427 の改訂）

### 既存 issue の前提が崩れている

#424–#426 は「Differometor の result オブジェクトを受け取り、`f` / `freq` / `frequency`、
`sensitivity` / `strain` / `asd` / `sqrt_psd` といった **属性名の揺れを吸収して**変換する」設計に
なっている。しかし実際の Differometor には result オブジェクトが存在せず、`df.run()` は
生の JAX 配列タプルを返すだけで、周波数軸は**入力側**に渡す。属性名 sniffing は実装しても走らない。

### 改訂後の API

```python
# gwexpy/interop/differometor_.py
from_differometor_sensitivity(cls, frequencies, sensitivity, *, unit=None, name=None)
from_differometor_noise_budget(cls, frequencies, components: Mapping[str, ArrayLike], *, unit=None)
from_differometor_response(cls, frequencies, response, *, outputs, inputs, unit=None)
from_differometor_population(candidates)          # -> pandas.DataFrame
from_differometor_run(cls, frequencies, run_result, *, unit=None)   # df.run() 戻りタプルの糖衣
```

既存の interop 定型に完全準拠する:
- `cls` を第 1 引数で受け、`ConverterRegistry` 経由で Series / Dict / Matrix を分岐（`finesse_.py:84-133`）
- Differometor は astropy 単位を持たないので、**`unit=` を必須で要求**（finesse と同じ扱い）
- **`differometor` を import しない**（呼び出し側が作った配列を受け取るだけ）ので、
  `from_pyoma_results()` / `from_metpy_dataarray()` と同じ **import-only adapter** 区分。
  contract JSON では `optional_dependencies: []` / `source_dependencies: ["differometor"]`
- JAX 配列 → NumPy 変換は `gwexpy/interop/jax_.py` の既存経路を再利用
- `name` は `f"{output} -> {input}"` のような意味のある文字列を必ず設定

### extra の扱い — #427 案からの変更

#427 は `simulation` または `design` extra の新設を求めているが、**既存方針からの逸脱**である:
`VALID_EXTRAS` は `{audio, control, gw, netcdf4, seismic, zarr}` の 6 種のみで、
finesse / PySpice / skrf / torch / mne などニッチ・重量級の interop backend は
すべて `_EXTRA_MAP[pkg] = None`（bare install 案内）で扱われている。

**推奨: `extras: []` + bare install（`pip install differometor`）**。
新設する場合は 5 箇所の同時更新が必要（`pyproject.toml` / `_optional.py` の 3 辞書 /
`test_interop_contract.py:27-34` の `VALID_EXTRAS` / en・ja `installation.md` / contract JSON）。

### land すべきファイル一式

| 分類 | ファイル |
|---|---|
| 実装 | `gwexpy/interop/differometor_.py`（新規）、`_optional.py`（`_OPTIONAL_DEPENDENCIES` + `_EXTRA_MAP`）、`interop/__init__.py`（import + `__all__`） |
| 任意 | `gwexpy/frequencyseries/frequencyseries.py` / `collections.py` にクラスメソッド束縛（**束縛したものだけを docstring Examples に書く** — 1-E の gwinc バグを繰り返さない） |
| 契約 | `docs/developers/contracts/public_interop_contract.json` に 1 エントリ |
| テスト | `tests/interop/test_interop_differometor.py`（MagicMock + `autouse` fixture。**モックはテストファイル内に定義**） |
| docs（公開） | `docs_redesign/how-to/interop.md` + `.po` |
| docs（契約テスト対象・旧ツリー） | `docs/web/{en,ja}/user_guide/interop.md`、`docs/web/{en,ja}/reference/api/interop.rst`、`docs/web/{en,ja}/reference/api/gwexpy.interop.differometor_.rst`（新規） |

### 改訂の進め方

1. **#423 を先に決着させる** — 本計画書の「調査で確定した外部事実」節が第一稿。
   `pip install differometor` して戻り値の shape / dtype / 物理単位を実測し、#423 にコメントで記録してクローズ
2. **#424–#426 の本文を上記 API に差し替え**（旧本文は issue コメントとして残し、経緯を消さない）
3. **#427 を改訂** — fixture は「result オブジェクトのモック」ではなく「代表的な配列 + 周波数軸」に変更し、
   置き場所も `tests/fixtures/differometor/` ではなく**テストファイル内**（既存慣習）。
   `simulation` extra は「既存方針では bare install。新設する場合は 5 箇所同時更新」と注記
4. milestone は割り当てない。ただし内部リリース計画が「v0.2.4: #423〜#427（#423 調査後に順序再評価）」
   としているため、**#423 完了が順序再評価のトリガー**である旨を issue に明記する

---

## Phase 4: spicypy（コード変更なし）

transcript の結論どおり「file reader/writer ではなく GWpy オブジェクトを共通語にした連携」。
**`gwexpy/interop/spicypy_.py` は作らない** — GWpy の `TimeSeries` / `FrequencySeries` が既に
共通語であり、変換アダプタを書く必要がないため（かつ `pyspice_.py` と名前空間が紛らわしい）。

実際にやること:
- `explanation/ecosystem.md` に spicypy 節（役割、Apache-2.0、gwexpy との補完関係、LPSD /
  Daniell's method / Huddle test / Wiener filter といった参考にする概念）
- Interoperability status に `Docs-only` として掲載
- LPSD / Daniell's method の GWexpy 実装可否は `ROADMAP.md` の backlog に 1 行置くだけ（本計画では設計しない）

---

## リスク

| リスク | 対処 |
|---|---|
| **ライセンス** — pemcoupling は All Rights Reserved | Phase 1-C で明文化。CONTRIBUTING に vendoring 禁止を追記。いずれもコードコピーしない |
| **依存肥大化** — GWDama が lalsuite を引き込む | h5py のみで実装し gwdama に依存しない（Phase 2 の中核判断）。Differometor も import しない |
| **契約テスト網による land 失敗** — `__all__` に 1 行足すだけで CI が落ちる | Phase 2/3 の「land すべきファイル一式」表を issue 本文にそのまま転記し、部分 land を防ぐ |
| **新旧 docs ツリーの二重管理** — 公開は `docs_redesign/`、契約テストは `docs/web/` | Phase 1 は `docs_redesign/` のみ。Phase 2/3 は両方を land 対象として明記 |
| **公開ページのランキング禁止** — `test_docs_redesign_public_content.py` | 優先順位は開発者向け `ROADMAP.md` に置き、公開ページには「順序は優先度ではない」旨を明記 |
| **scope creep** — detchar/summary/veto workflow まで抱え込む | `ecosystem.md` の "Scope boundaries" 節で core に入れないものを明示 |
| **site-specific 化** — Virgo farm / cvmfs / PEM→DARM 固有 API | 汎用 source→target として記述。GWDama reader も site 固有パスを扱わない |
| **外部 API の不安定性** — Differometor は commit 数が少なく API 未固定 | 生配列を受け取る設計にし、`differometor` を import しないことで影響面を最小化 |
| **未知 attrs 保全の仕組みが無い** | Phase 2 の最初の設計判断として明示（既存に無いものを「合流させる」と誤認しない） |

---

## 検証方法（Phase 1）

```bash
# 1. gettext 抽出と ja catalog 更新
conda run -n gwexpy sphinx-build -b gettext -D nb_execution_mode=off docs_redesign /tmp/gwexpy-gettext
conda run -n gwexpy sphinx-intl update -p /tmp/gwexpy-gettext -l ja

# 2. catalog の構文チェック
conda run -n gwexpy msgfmt --check --output-file /tmp/ecosystem.mo \
  docs_redesign/locales/ja/LC_MESSAGES/explanation/ecosystem.po

# 3. EN / JA ビルド
conda run -n gwexpy sphinx-build -b html docs_redesign docs_redesign/_build/html
conda run -n gwexpy sphinx-build -b html -D language=ja docs_redesign docs_redesign/_build/html/ja

# 4. 公開コンテンツ契約テスト
conda run -n gwexpy python -m pytest -q \
  tests/docs/test_docs_redesign_public_content.py \
  tests/docs/test_docs_redesign_release_facts.py

# 5. linkcheck（GWpy / spicypy / GWDama / Differometor の外部 URL を新規に張るため必須）
conda run -n gwexpy sphinx-build -b linkcheck docs_redesign /tmp/gwexpy-linkcheck

# 6. 旧ツリーを壊していないことの確認（docs-pr.yml と同じ）
conda run -n gwexpy sphinx-build -W --keep-going -b html docs docs/_build/html
```

完了条件:
- [ ] `docs_redesign/explanation/ecosystem.md` が存在し、`explanation/index.md` の toctree に登録されている
- [ ] `locales/ja/LC_MESSAGES/explanation/ecosystem.po` と `explanation/index.po` が全訳され `msgfmt --check` が通る
- [ ] EN / JA 両方の HTML ビルドが成功する
- [ ] `tests/docs/test_docs_redesign_public_content.py` が pass（**ランキング表現を含まない**ことの機械的確認）
- [ ] linkcheck が pass
- [ ] `docs/developers/LICENSES_THIRD_PARTY.md` の 4 パッケージすべてに「確認方法」列が埋まっている
- [ ] README に `## Where gwexpy fits` があり、docs へのリンクが有効
- [ ] `ROADMAP.md` に "Ecosystem & Interoperability" backlog 節がある
- [ ] #404 の acceptance criteria のうち、実在しない `loadmap_v0.2.0.md` に関する項目を
      `ROADMAP.md` + 本計画書への参照に読み替えたことを issue にコメント

Phase 2–4 は本計画では実行しないため、検証は各 issue 側に記載する。

---

## イシュー本文ドラフト

### 新規 A: `docs: add ecosystem positioning page (explanation/ecosystem)`

> ## Summary
> GWpy / spicypy / GWDama および周辺 GW エコシステムに対する gwexpy の位置付けを、
> `docs_redesign/explanation/ecosystem.md` として明文化する。
>
> ## Background
> 既存の `how-to/interop.md` は「どう変換するか」のカタログであり、「gwexpy が何であって何でないか」を
> 答えるページが存在しない。現状リポジトリの追跡ファイルには spicypy / GWDama / pemcoupling への
> 言及が一切ない（`git grep` で 0 件）。
>
> ## Deliverables
> - [ ] `docs_redesign/explanation/ecosystem.md` 新規作成（レイヤー表 / 4 パッケージ比較表 /
>       関係分類 / 差別化ポイント / scope boundaries / interoperability status / third-party code policy）
> - [ ] `docs_redesign/explanation/index.md` の "Design and concepts" toctree + グリッドカードに追加
> - [ ] `sphinx-intl update` で `locales/ja/LC_MESSAGES/explanation/ecosystem.po` を生成し全訳
>       （`explanation/index.po` も再生成・再訳）
> - [ ] `how-to/interop.md` の Related Pages から相互リンク（比較表本体は置かない）
> - [ ] `explanation/gwexpy_for_gwpy_users.md` に短い positioning 段落 + リンク
> - [ ] `README.md` に `## Where gwexpy fits` 節（短文 + docs リンク、表は置かない）
>
> ## Constraints
> - `tests/docs/test_docs_redesign_public_content.py` が「未測定のランキングを公開ページに書くこと」を
>   禁じている。**連携優先順位は書かない**（開発者向け `ROADMAP.md` に置く）
> - 未検証のライセンス表記を含めない。確認できないものは「要確認 (unverified)」と明記
> - `docs/web/` は凍結された非公開ツリーなので触らない
>
> ## Acceptance criteria
> - [ ] EN / JA 両方の `sphinx-build -b html` が成功
> - [ ] `msgfmt --check` が通る
> - [ ] `pytest tests/docs/test_docs_redesign_public_content.py` が pass
> - [ ] `sphinx-build -b linkcheck docs_redesign` が pass
>
> ## Stability
> stable (documentation only)

### 新規 B: `io: add GWDama HDF5 reader/writer (format="hdf.gwdama")`

> ## Summary
> GWDama が出力する階層 HDF5 を `TimeSeriesDict.read(..., format="hdf.gwdama")` で読み、
> gwexpy 側からも書き出せるようにする。**`gwdama` パッケージには依存せず、h5py のみで実装する。**
>
> ## Background
> GWDama (MIT, 0.6.0) の runtime 依存は `gwpy>=3.0` に加え `lalsuite>=6.73` を含み、optional
> extra としても重い。一方 GWDama の HDF5 は素の h5py で読める（dataset attrs: `sample_rate`,
> `t0`, `channel`, `unit`；group は `dts_key` 由来）。したがってフォーマット互換のみを実装する。
> この方針は「conformance generator は gwexpy を import してはならない」制約とも噛み合う。
>
> ## Design
> - 新規 `gwexpy/timeseries/io/gwdama_hdf5.py`（`ndscope_hdf5.py` と同じ
>   `identify_*` / `read_timeseriesdict_*` / `write_timeseriesdict_*` の 3 関数構成、docstring にスキーマ図）
> - `register_timeseries_format("hdf.gwdama", aliases=("gwdama","gwdama-hdf5"), ..., extension="hdf5")`
> - `gwexpy/timeseries/io/__init__.py` の import リストに追加（忘れると登録されない）
> - `group=` selector で subgroup を選択可能に
> - タイミングメタ欠落時は黙って 0/1 にしない（`sample_rate` 欠落は `ValueError`、`t0` 欠落は `UserWarning`）
>
> ## 実装前に決める設計判断
> - [ ] **未知 attrs の保全方式** — 現状 gwexpy にこの仕組みは存在しない（ndscope/zarr/nc は読み捨て）。
>       `set_provenance()` に載せるか、Matrix の `attrs_json` 方式を 1D にも広げるか
> - [ ] **`.h5` 拡張子衝突** — root attrs をマジックとして identify するか `public_auto_identify: false` にするか。
>       identify に「壊れたファイルを弾く条件」は入れない
> - [ ] **任意深さのネスト** — `gwexpy/io/hdf5_collection.py` の manifest 方式を拡張するか、reader 内で平坦化するか
>
> ## Deliverables
> - [ ] 実装: `gwexpy/timeseries/io/gwdama_hdf5.py`、`gwexpy/timeseries/io/__init__.py`
> - [ ] 契約: `public_io_contract.json` に 1 エントリ、`public_io_contract.md` に `### hdf.gwdama` 節
> - [ ] conformance: `tests/io_conformance/generators/gwdama.py`（**gwexpy を import しない**）、
>       `generators/__init__.py` の `GENERATOR_SPECS`、`contract.py` の `fixture_generator_by_format`
> - [ ] テスト: read conformance / write round-trip / エッジケース / `importorskip("gwdama")` cross-check
> - [ ] docs: `docs_redesign/how-to/io_formats.md` + `.po`、**および** `docs/web/{en,ja}/user_guide/io_formats.md`
>       （`test_io_docs_contract_sync.py` が旧ツリーを assert するため）
> - [ ] `SUPPORTED_IO_MATRIX.md`、`CHANGELOG.md`
>
> ## Verification
> ```bash
> python scripts/ci/run_gate.py io-contract
> python scripts/ci/run_gate.py io-conformance
> ```
>
> ## Stability
> provisional

### 新規 C: `bug(interop): gwinc_ docstring documents non-existent classmethods`

> `gwexpy/interop/gwinc_.py:120-123` の Examples は `FrequencySeries.from_gwinc_budget("aLIGO")` /
> `FrequencySeriesDict.from_gwinc_budget("aLIGO")` と書いているが、これらのクラスメソッドは存在しない
> （`hasattr(FrequencySeries, "from_gwinc_budget") == False`）。実際の呼び出しは
> `gwexpy.interop.from_gwinc_budget(FrequencySeries, "aLIGO")` のみ。
> finesse は逆にクラスメソッド束縛がある（`frequencyseries.py:1138`, `collections.py:523`）ため、
> docstring だけを直すか、束縛を追加して docstring どおりにするかの判断が要る。
>
> 検出されていない理由: `pyproject.toml:338` の pytest 設定に `--doctest-modules` がない。
> 副次的に「interop docstring の Examples を doctest 対象にするか」も検討事項。

### 既存 #404 の改訂（新規 issue は作らない）

#404 は 2026-08-01 当時 v0.2.0 bucket に割り当てられていたが、現在は open / milestone なしで、issue 本文も Ecosystem backlog を参照している。以下は当時実施した本文改訂の記録である:

- Deliverables の `loadmap_v0.2.0.md` 参照を削除し、「**`ROADMAP.md` および
  `docs/developers/plans/active/2026-08-01-ecosystem-interop-plan.md` を正とする**」に置換
  （理由: `loadmap_v0.2.0.md` はリポジトリに存在しない）
- 対象パッケージに **spicypy (Apache-2.0)** と **Differometor (MIT)** を追加し 4 件の表にする
- 「確認方法」列を追加（LICENSE ファイル本体を見る。PyPI classifier / README バッジを根拠にしない）
- 配置先を `docs/developers/LICENSES_THIRD_PARTY.md` に確定

### 既存 #423 の改訂: 調査結果を記録してクローズする

> 一次情報（GitHub README / PyPI）で確認した結果:
> - Differometor は MIT、PyPI に `differometor` として公開済み
> - `df.run(S, params, values)` は `carrier, signal, noise, detector_ports, *_` という
>   **生の JAX 配列タプル**を返す。result dataclass / namedtuple は存在しない
> - 周波数軸は戻り値ではなく**入力側** `[("f", "frequency")]` で与える
> - 依存は JAX + Optax（GPU 版は CUDA）
>
> **結論**: #424–#426 が前提としている「result オブジェクトの属性名 sniffing」は成立しない。
> 各 issue の本文を「明示的な `(frequencies, values)` を受け取る変換関数」設計へ改訂する。
>
> 残タスク: `pip install differometor` 後に戻り値の shape / dtype / 物理単位を実測し、
> 本コメントに追記してからクローズする。#423 の完了が v0.2.x 順序再評価のトリガーとなる。

### 既存 #424–#426 の改訂方針

- `from_differometor_sensitivity(cls, frequencies, sensitivity, *, unit=None, name=None) -> FrequencySeries`
- `from_differometor_noise_budget(cls, frequencies, components, *, unit=None) -> FrequencySeriesDict`
- `from_differometor_response(cls, frequencies, response, *, outputs, inputs, unit=None) -> FrequencySeries | FrequencySeriesMatrix`
- `from_differometor_population(candidates) -> pandas.DataFrame`
- 追加: `from_differometor_run(cls, frequencies, run_result, *, unit=None)` — `df.run()` 戻りタプルの糖衣
- 実装場所 `gwexpy/interop/differometor_.py`。**`differometor` を import しない**
  （contract では `optional_dependencies: []` / `source_dependencies: ["differometor"]`）
- `unit=` は必須引数として要求（finesse と同じ扱い。Differometor は astropy 単位を持たない）
- land 対象に契約 JSON / en・ja interop ガイド（旧ツリー）/ en・ja reference index / en・ja rst を含める
- 旧本文は issue コメントとして保存し、改訂の経緯を消さない

### 既存 #427 の改訂方針

- fixture を「result オブジェクトのモック」から「代表的な配列 + 周波数軸」へ変更
- 置き場所を `tests/fixtures/differometor/` から**テストファイル内のモックビルダ関数**へ変更
  （既存 interop テスト 60 ファイルすべてがこの方式で、`tests/fixtures/` に interop 用の置き場は存在しない）
- `simulation` / `design` extra は**既存方針からの逸脱**である旨を明記。推奨は
  `extras: []` + bare install。新設する場合は 5 箇所（`pyproject.toml` / `_optional.py` の 3 辞書 /
  `test_interop_contract.py:27-34` の `VALID_EXTRAS` / en・ja `installation.md` / contract JSON）の同時更新が必要

---

## 実行順序（Phase 1）

1. 本計画書を `docs/developers/plans/active/2026-08-01-ecosystem-interop-plan.md` としてリポジトリへ保存
2. `docs/developers/LICENSES_THIRD_PARTY.md` 作成（Phase 2–4 の記述が参照するため最初に確定）
3. `docs_redesign/explanation/ecosystem.md` 作成
4. `explanation/index.md` / `how-to/interop.md` / `explanation/gwexpy_for_gwpy_users.md` /
   `explanation/roadmap.md` の改訂
5. `sphinx-intl update` → `ecosystem.po` / `index.po` / `interop.po` / `roadmap.po` /
   `gwexpy_for_gwpy_users.po` の翻訳
6. `README.md` / `ROADMAP.md` / `CONTRIBUTING.md` の改訂
7. 検証（gettext → msgfmt → EN/JA ビルド → docs テスト → linkcheck → 旧ツリー `-W` ビルド）
8. issue 起票案（新規 A / B / C）と改訂案（#404 / #423–#427）をユーザーへ提示

---

## 明示的にスコープ外とするもの

- **Fisher Matrix API**（transcript 冒頭の議論）— #570–#574 として起票済み・設計完了。本計画では触れない
- LPSD / Daniell's method / Huddle test の GWexpy 実装 — `ROADMAP.md` backlog に 1 行置くのみ
- gwdetchar / gwsumm / gwvet / hveto / pyomicron との**コードレベル**連携 — docs 記載のみ
- `ChannelGroup` / `DataBundle` / `ExperimentDataset` といった新規抽象 — transcript で候補に挙がったが、
  既存の matrix container / `SegmentTable` との責務重複を検討していないため含めない
- pemcoupling 由来の product schema / status flags の実装 — `CouplingResult` 側の設計課題であり別件
- `docs/web/` 旧ツリーへの ecosystem ページ追加 — 凍結・非公開のため
- ルート `ROADMAP.md` の現行 v0.2.0 節と deferred terrain/DEM 計画の切り分け — 注記を入れるに留め、全面改訂はしない
