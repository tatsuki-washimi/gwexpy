# v0.2.0 以降の長期戦略 — roadmap 正本化と未反映分の取り込み（rev.7, 2026-08-15, mplan）

## Goal

v0.1.14 公開後の gwexpy において、(1) 2026-08-12 の未反映議論（time/I/O semantics の受け皿、API compatibility、正本階層）を repo 文書に反映し、(2) 長期ロードマップの正本を `ROADMAP.md` + canonical 設計文書に確立し、(3) PR #660 を post-v0.1.14 の採択 PR として merge し、(4) GitHub メタデータ（#657 / v0.2.0 milestone / 追跡 issue）を整合させる。

## Context

アップロードされた ChatGPT ログ（4232行）のうち 2026-08-12 分の3往復が未反映だった:
1. **時刻/epoch 意味論と I/O dispatch の2トラック構造** — v0.1.14 が意図的に残した #632/#634 と #636/#444/#616 に milestone もテーマ帰属も無い（実質的な穴）
2. **API Compatibility テーマ**（#639 の4分類、#640 の split）
3. **正本の階層設計**（repo 文書が正本、GitHub は実行管理）、`Committed / Directional / Backlog` の3状態、設計文書の日付なし安定パスへの昇格、#657 のクローズ

> **自己完結性の宣言**: 原 ChatGPT ログはセッション添付でリポジトリ外にあり永続しない。実行に必要な要旨は本計画の Context と Step 4 の D14〜D18 記述指示に全て含まれており、原文参照は不要。

本計画は3段階のレビューを経ている: ①外部レビュー（Approve with revisions、5修正点 → rev.2 で反映）、②mplan フェーズ2の3視点並列レビュー（Auditor / Critic / Audience → 本 rev.3 で反映、下記「レビュー反映記録」）、③plan-reviewer（Reviewer Status 参照）。

## Current baseline（2026-08-15 実測検証済み）

| 項目 | 状態 |
|---|---|
| **v0.1.14** | **RELEASED**。tag `v0.1.14` = `42eec70450b867f10b7a9331c3a0217ce589c564`、`origin/main` = `origin/maint/0.1` = 同 SHA。GitHub Release published、PyPI wheel/sdist 済、#653 closed、milestone v0.1.14 closed（12/12） |
| Zenodo | **実在確認済み**（Zenodo API 実測）: version DOI `10.5281/zenodo.21941441`、concept DOI `10.5281/zenodo.19059422`、version 0.1.14、2026-08-15。`CITATION.cff`/`.zenodo.json` は設計上 DOI フィールドを持たない（v0.1.13 と同じ。DOI は ROADMAP と release_notes に置く先例） |
| v0.2.0 milestone | open 11件 = issue 10件（#637 #612 #594 #513 #508 #413 #403 #402 #401 #400）+ PR #625。**#581 は milestone 外**（description が言及するのみ） |
| PR #660 | open / MERGEABLE / mergeStateStatus=**CLEAN**、head `ae85889ac` 1コミット、origin/main から 72 コミット遅れ。body に「#657 同期は paused」という陳腐化注記（実態は**実行済み**） |
| ~~manifest 削除リスク~~ | **撤回**（rev.2 の誤り）: `git merge-tree` 実測で 3-way merge は `audit-manifest-v0.1.14-release-readiness.yaml` を削除**しない**（merge-base に無く main 側で追加されたファイルは merge で保存される）。rev.2 の「削除リスク」は 2-dot diff の誤読。`docs/latest-release-announcement` branch への同種の申し送りも撤回（実 diff は +220/−0、manifest 非接触） |
| rebase 衝突見込み | なし（実測: merge-base..origin/main で `ROADMAP.md`・`docs_redesign/explanation/roadmap.md`・同 po の変更 0 コミット。**したがって計画中の L 番号は rebase 後も有効**。ただし編集時は見出しで再特定する） |
| #657 | open。body は「Done」全16項目 `[x]` の歴史的実行記録（書き換えない） |
| #637 decision date | TBD。ただし「milestone mid-point で確定し当該コメントを更新する」方式は 2026-08-11 コメントに**記録済み** |
| 設計文書 | active-plans 領域の日付付き capability-domain 設計文書。`Status: planned` / `Reviewer Status: draft`、§10 未チェック、§11 Open-1〜3 未解決。旧 dated path への repo 内参照は ROADMAP L9/L69/L295 の3件のみ |
| CHANGELOG との齟齬 | `[0.1.14]` Known limitations が「#634 for **v0.2.0**」と記載（当時の見込み）。新テーマ帰属と矛盾するため ROADMAP 側で上書き宣言が必要。#632 は Known limitations でなく変更一覧の `(#632 partial)` として記録されている |
| root ROADMAP テスト | 存在しない（tests/ の roadmap 系2テストは別ファイルを読む） |
| Sphinx 影響 | `docs/conf.py` の `exclude_patterns` に `developers/**` → 設計文書移動はビルド無影響。ただし canonical 文書はビルド済み docs から不可視のまま（ヘッダに maintainer-facing と明記して対処） |

## Assumptions

- conda env `gwexpy` が存在し pytest / sphinx が動く。gettext（`msgfmt`）と `sphinx-intl` が利用可能。`rtk proxy gh` が使える（不可なら素の `gh` + 結果が疑わしい場合のみ proxy 再確認）。L3 は issue 実態の確認に read-only の `rtk proxy gh issue view` を使ってよい。
- merge 方式は merge commit(履歴上「Merge pull request #NNN」形式が既存慣行)。#657 に記録する SHA は「main 上の merge commit SHA」に固定。
- force-push（rebase 後の PR 更新)・PR body 変更・issue コメント/作成・milestone 変更は全て**ユーザー承認ゲート**(github-mcp-workflow 準拠、本文全文プレビュー)。
- コミット・push はユーザー承認時のみ。

## Critical Path

Step 1 rebase → Step 3 ROADMAP 改訂 → Step 4/6/7（Step 3 完了後）→ Step 8 検証 → Step 9 merge → Step 10-12 GitHub metadata。Step 2（PR 再定義）と Step 5（兄弟文書）は Step 1 完了後いつでも並行可。**Step 4 は新テーマ文言を ROADMAP から転記するため Step 3 完了後**（rev.3 の「並行可」を訂正 — 2レーンが同一テーマ文言を独立執筆すると乖離するため）。

## 戦略（Phase 構成）

### Phase 0 — 再ベースライン【完了 = 本改訂】
v0.1.14 released を前提化。manifest 削除リスクの撤回、Zenodo DOI 確定、#581 誤認修正を反映済み。

### Phase 1 — PR #660 の rebase と再定義（Step 1-2）

- `git rebase origin/main`。**rebase を選ぶ理由（訂正版)**: manifest 保護ではなく、(a) ROADMAP 本文を v0.1.14 released 前提へ書き換えるため現在の main を土台にしないと内容が破綻する、(b) 72 コミット遅れの解消、(c) PR の意味を「8/9 の roadmap 案」から「**post-v0.1.14 の canonical roadmap 採択**」へ再定義するため。
- 検証: `git diff --find-renames --diff-filter=D --name-status origin/main...HEAD` → **空**（削除ゼロの一般不変条件。rename 検出を有効化し、Step 4 の canonical 文書への git mv を誤って D 扱いしない。旧 V1 の grep は点対策かつ rebase 後自明に通るため廃止）。
- force-push は承認ゲート。
- PR title → `docs(roadmap): adopt capability-domain long-term roadmap`。body 全面更新（「paused」注記は「#657 で実行済み」へ訂正)。更新後 `rtk proxy gh pr view 660 --json title,body` で反映確認。

### Phase 2 — Canonical roadmap adoption(Step 3-7、同一 PR)

**`ROADMAP.md`**（Step 3）

1. 冒頭リード文（L12-14）修正 + **authority wording の一方向化**: 「ROADMAP は包含基準（criteria）の正本、milestone は基準を適用した結果リストの正本。milestone のメンバ変更が ROADMAP の DoD を書き換えることはない（逆方向のみ許可）」。現行 L3-6 の「milestones are authoritative for exact issue-level scope」をこの語彙に改訂（循環の除去）。
2. **`## v0.1.14 — I/O contract and maintenance hardening (released 2026-08-15)` 節を新設**（v0.1.13 節より軽量な散文。表形式化はしない — v0.1.13 との体裁対称性優先）: 出荷 SHA / tag / **DOI `10.5281/zenodo.21941441`**（v0.1.13 節の SHA/tag/DOI 行 = 現 L76-78 と同形式）/ テーマ別要約 / **Known intentionally deferred contracts**（#632 = 警告のみ・変更一覧では `(#632 partial)`、#634 = legacy GPS-seconds のまま）/ 「per-change 正本は CHANGELOG [0.1.14]」/ **CHANGELOG 上書き宣言 1 文**:「CHANGELOG [0.1.14] の『#634 for v0.2.0』は当時の見込みであり、現在の帰属は下記テーマ（milestone 未設定）」。
3. **v0.1.15 は release policy に操作可能な線で記述**: 「patch に入れてよいのは silent-wrong-result を明示エラー化する narrowing のみ（既存 L21-26 の方針を維持）。新しい引数・型・関数の追加は minor 限定。#632/#634 は narrowing 部分は patch 可、API 追加部分は minor」。v0.1.15 自体は「corrective release が必要な場合のみ存在、事前 scope なし」（状態語は付けない）。
4. **新テーマ「I/O time and dispatch semantics」を Future themes の先頭に追加**（名称は既存テーマ「Experiment data workflow」との衝突を避けるため rev.2 の「Experiment I/O semantics」から戻す）。内部は **Gate でなく Track と呼ぶ**（何もブロックしない並行トラックのため）:
   - **Track A — Time interpretation contract**: #632 / #634 / #636。ROADMAP 側は観測可能な contract のみ記述（「単一 layer に集約」等の実装構成は設計文書側に置く）。
   - **Track B — Dispatch / reader semantics**: #444 → #616（この依存は明示）。
   - A→B の強制依存なし。両 Track 完了 = テーマ完了。
   - **scope-creep 防止（既存 L228-231 規約の適用)**: 各 Track に release statement 1行 + headline user story 1本 + 名前付き受入成果物 + Non-goals 1行を **ROADMAP 側にも**付与。「issue 追加は受入成果物の更新を伴う」を明記。domain 併記（`(domain: io)` 形式、既存 Unassigned ideas と同形式）。
5. **新テーマ「API compatibility and stabilization」を追加**。ただし**重複ホーム化を防ぐ位置づけ**: X3 / v0.2.0 Workstream「API stability labelling」（#400, Committed）を**起点とする継続テーマ**と定義し、v0.2.0 側と v1.0 節には相互参照のみ置く（同一作業の再記述をしない）。収容するのは現在ホームの無い #639（override audit 4分類）と #640（behavioural contract / docs への split）。Track A/B と同様に受入成果物 + Non-goals を付与。
6. Ecosystem backlog の Virgo 項目に #638 を追加、順序 `FFL contract → .ffl support → #638 hardening → dataDisplay/ROOT converters`。
7. **3状態語彙**: 定義表を ROADMAP に置く — `Committed` = active milestone に入っている / `Directional` = 将来テーマ候補（version・date・scope の commitment なし） / `Backlog` = capability 認識のみ。**表示形式は見出し直下の1行 `Status: <語>` に固定**。適用範囲は **v0.2.0 節と Future themes 節の全テーマブロック（例外なし）**。released 節・release policy・v1.0 criterion・Engineering hygiene は対象外と**明記**（「語が無い = 未分類」の曖昧さを除去）。`Committed` は常に1つ（v0.2.0）。
8. **「Roadmap maintenance」節を新設**（陳腐化対策）: (a) minor 出荷時に released 節を追加する手順、(b) `Directional → Committed` の昇格トリガ = milestone 作成と同時、(c) `Committed` 単一性、(d) テーマの retire 条件（受入成果物が main で green）、(e) `domain:*` ラベルは**非正本の検索用便宜**である旨の1行。
9. 設計文書へのリンク（L9/L69/L295）を新 path へ差し替え。

**設計文書の昇格**（Step 4）

- active-plans 領域の日付付き capability-domain 設計文書を `docs/developers/design/capability-domain-roadmap.md` へ `git mv`。
- ヘッダ: `Status: active` / `Authority: canonical` / `Audience: maintainer-facing（ビルド済み docs には含まれない — docs/conf.py が developers/** を exclude）`。`Reviewer Status: draft` 解消。§10 完了は将来 `Initial adoption verification: completed <date>` の別フィールド。
- 内容追補: §3 consumer layer 行と §5 マトリクス（11+4 ドメイン × release テーマ）に新テーマ2列 → §6 に release statement + user story + 受入成果物（**文言は Step 3 で確定した ROADMAP から転記して一致させる** — 独立執筆しない）→ §8 に v0.1.14 記載方針・§9 裁定 B-4 の期限切れ注記 → **§11 に D14〜D18 を追記**: D14 = 新テーマ2件（8/12 議論の要旨と採否を各1段落で含める — rev.2 の §13 新設は Critic 指摘により中止し、要旨は decision log に吸収）、D15 = 3状態表記、D16 = v0.1.14/15 記載方針、D17 = design/ 昇格、**D18 = milestone hygiene の判定記録**（外部レビュー2巡目指摘により前倒し — **Step 3 で確定した ROADMAP の v0.2.0 Definition of done案に基づき、Phase 4 記載の3分類基準〔DoD-required / gate-supporting / unrelated〕を実メンバ11件に適用した「planned application record」として、Step 8 の全検証通過までに本 PR 内で確定する**。merge 後の Step 11 は D18 を書き換えず GitHub milestone へ適用するのみ）→ §12 出典注記を「2026-08-12 discussion; original chat export not archived; substance captured in D14-D17」に更新。
- `docs/developers/design/README.md` 新設（**最小骨子を指定**: `design/` = 日付なしの living 設計文書 + 既存 `design_data/`（CSV 成果物）+ `gui/`（分析メモ）。隣接 `plans/` = 日付付き実行記録、`contracts/` = 規範的 I/O contract、`reports/` = 生成レポート、という4ディレクトリ境界表のみ。6行程度）。`docs/developers/plans/README.md` に `design/` との境界1行追記。
- dated 版の archive コピーは作らない（維持）。

**兄弟文書の事実訂正**（Step 5）— L8 だけでは不十分。**対象は V8 パターンの実測全ヒット23行 + 非ヒット1行（terrain L243）= 24行**（plan-reviewer 3巡目の実測で確定。サブイシュー単位の `Milestone: v0.2.0` 宣言も umbrella レベルと同種の stale 断定であり、同一ファイル内で半分だけ直す非一貫を避けるため全行を対象化）。**milestone 帰属の事実訂正（テーマ名/backlog への置換）のみ**を行い、設計内容は変えない（内容再設計に及ぶ場合は別 PR）:
- `2026-07-31-terrain-scalarfield-io-design.md`（16行）: L8, L243, L350, L408, L450, L482, L506, L531, L560, L586, L607, L627, L645, L659, L685, L726（L506 以降はサブイシュー単位の `Milestone: v0.2.0` 宣言。**L243 は「v0.2.0 では対象外を維持」という除外宣言であり narrowing 後も正確な可能性が高い — 検証の上、変更不要ならその根拠を対照表に記す**）
- `2026-08-01-segmenttable-workflow-design.md`（5行）: L8, L33, L67, L242, L481（L481 は見出し `## Sub-issues (v0.2.0)`）
- `2026-08-01-ecosystem-interop-plan.md`（3行）: L15, L71, L609（#404 は milestone 外に移動済み。L15 は Phase 完了判定の根拠のため訂正文言は文脈を保って書く）
- `2026-08-01-roadmap-reorganization-plan.md` は Status: completed の歴史記録 → 変更しない。
- V8 が上記以外の行に REVIEW を出した場合: 同種の stale 断定なら追加訂正し対照表に加える。stale でない（歴史記録・除外宣言等）なら残し、根拠を対照表に記す。

**contract test 新設**（Step 6）— `tests/docs/test_root_roadmap_contract.py`。見出し照合は **case-insensitive**（main 側 sentence case との揺れを固定化しない）。全文 snapshot は不可。assert 集合:
1. `## v0.2.0` 節が存在し、`Workstreams` と `Definition of done` のラベルを持つ（判定は行頭見出し**またはプレーンテキストのラベル行**の両方を許容、case-insensitive — 現行 ROADMAP は段落内ラベルであり、Step 3 に見出し化の指示は無い。見出し化を強制しない）
2. DoD 項目数 ≥ 4、かつ各項目が issue 参照（`#\d+`）を1つ以上含む（0件緑防止）
3. Future themes 節内の `###` テーマ見出し行に、具体的なバージョン番号の先取り割当（`v\d+\.\d+` を含む見出し、例: `### v0.3.0 ...`）が無い（本文中の issue 番号や他リリースへの相互参照としての `v\d+\.\d+` は許容 — 「将来テーマに版番号を先取りしない」という本来の不変条件を、見出しレベルの割当禁止として保証する。plan-reviewer レビュー後の訂正: 当初のセクション全体バージョントークン禁止では相互参照が構造的に不可能だったため見出しレベルへ限定）
4. `## v0.` 見出し集合 ⊆ {v0.1.13, v0.1.14, v0.2.0}（将来 minor の節が生えないことを守る。本文中の `v0.2.1+` 等は対象外）
5. v0.2.0 節と Future themes の各テーマブロックが `Status:` 行をちょうど1つ持つ
6. `Status: Committed` の出現 = 全文書で1箇所
7. ROADMAP 中の `docs/**` 相対リンク全件が実在（アンカー `#...` を剥がして判定。**旧 V6 shell チェックはこのテストに一本化して廃止**）
8. v0.1.14 節の deferred issue 番号集合 ⊆ `CHANGELOG.md` `[0.1.14]` 節の issue 番号集合
9. **必須 release 見出し `{v0.1.13, v0.1.14, v0.2.0}` ⊆ 実際の `## v0.` 見出し集合**（外部レビュー2巡目指摘: assert 4 の上限チェックだけでは v0.1.14 節が丸ごと欠落しても `∅ ⊆ X` で緑になる。本 assert が下限側を保証し、assert 4 と対で実質 `== {v0.1.13, v0.1.14, v0.2.0}` を成立させる）

**公開 roadmap**（Step 7）— 具体的な差分は次の2点のみ（構成・状態語なし・バージョン番号なし = D3 維持。committed/directional の区別は Near/Mid/Long-term の既存構成で表現され、状態語彙は maintainer 向けとして ROADMAP に留める — D15 に記録）:
1. `docs_redesign/explanation/roadmap.md` の `## Long-Term Directions` に1項目追加（案）: "Consistent time interpretation (time zones, numeric time scales and epochs) and uniform reader behaviour across supported experiment data formats"。**配置根拠**: 同ページ Reading Guide の Long-term 定義（exploratory directions ... not yet scheduled）が、milestone 無しの Directional テーマである本件に一致する（Mid-term は「docs/API cleanup 後に広く公開したい作業」でスケジュール含意が強い）。API compatibility テーマは既存 Long-term の "A stable, documented public API surface" bullet で被覆済みのため**追加しない**。
2. L6 の `*Last updated: 2026-08-09.*` を `2026-08-15` に更新。

~~po の fuzzy 解消~~ **中止**（実測: 当該 `#, fuzzy` は PO ヘッダの慣例マーカーで、消しても `sphinx-intl update` で復活する）。代わりに: 上記変更後に `sphinx-intl update` を回し、新規 msgid に翻訳を入れ、**content-level fuzzy / 空 msgstr を 0 に保つ**。

### Phase 3 — Review + verification + merge（Step 8-9）

- 検証 V1〜V5・V8（下記）を全て通す（V6 は contract test へ一本化済みの注記、V7 は Phase 4 で使用）。**V8 は補助 grep であり、L3 の完了判定の主体は列挙24行の目視確認**（L3 の成果物に訂正前後対照表を含める。変更不要と判断した行は根拠を記載）。
- **PR diff の独立レビュー**（計画レビューとは別物）: `japanese-style-reviewer`（ja 文書）+ `Reviewer`（diff 整合）を read-only で並列起動 → CRITICAL/HIGH 修正。
- merge は承認ゲート（merge commit 方式）。

### Phase 4 — GitHub metadata finalization（Step 10-12、merge 直後）

merge〜hygiene 完了までの短い乖離窓は、Phase 2-1 の一方向導出定義（milestone は ROADMAP 基準の適用結果であり、適用は遅延しうる）で吸収する。

- **#657**: body は書き換えず **final comment + close**。コメント: canonical path への採択、main 上の merge commit SHA、「original body 中の dated path は当時の作業場所」の注記。
- **v0.2.0 milestone hygiene**: 判定基準は **merge 済み ROADMAP の v0.2.0 Definition of done（Phase 2 で凍結、merge commit SHA で参照）**。判定中の DoD 書き足しは禁止（循環防止）。各 issue を3分類:
  - `DoD-required` = DoD 文面が名指し（例: #402 は DoD 3 が名指し。DoD 4 が名指すのは #637 の性能予算。#581 は DoD ではなく Workstreams 箇条書きにのみ登場し、かつ milestone 外なので判定対象外）
  - `gate-supporting` = DoD に直接名指しされないが、ROADMAP の v0.2.0 Workstreams が scope 内として明記する、または DoD の検証・release evidence に必要なもの
  - `unrelated` = 上記いずれでもない → 外す（外す際、当該 issue にテーマ帰属先を1行コメント。PR #625 も同様 — 追跡リンク喪失防止）
  - 判定対象は実メンバ 11件: #637 #612 #594 #513 #508 #413 #403 #402 #401 #400 + PR #625。**外部レビュー2巡目指摘により、この3分類・11件の判定は Step 4 で PR #660 内に D18（planned application record）として確定済み**（milestone description には「基準: capability-domain-roadmap.md D18 参照」の1行 + 既存 description への追記のみ。GitHub は意思決定記録の置き場にしない）。Step 11 では **D18 を再分類・書き換えず**、確定済みの分類を GitHub milestone にそのまま適用する（unrelated の除去 + 帰属先コメントの投稿のみ）。適用前に、Step 3 で確定した ROADMAP DoD の文言が merge 後も D18 確定時点から変わっていないことを diff で確認する。
- **Open-1〜3 の issue 化**: Open-2（#637 decision date — 方式は #637 コメントに記録済みなので追跡 issue を立てるだけ）/ Open-3（性能退行予算）= actionable、v0.2.0 planning dependency と明記。Open-1（`gwexpy/cli/` の consumer layer 前提違反）= architecture decision issue、milestone なし。
- レガシー `docs/web/{en,ja}/user_guide/roadmap.md` の陳腐化は **#606 に1コメントで申し送る**（本 PR では触らない）。

### Phase 5 — v0.2.0 execution readiness【本 mplan サイクルのスコープ外・後続作業】

完了条件のみ定義して引き継ぐ: (1) #637 decision date 確定（Open-2 issue close）、(2) #581 ベースライン取得と退行予算数値化（Open-3 issue close）、(3) #612 contract matrix 着手（収集数アサート付き）、(4) 設計文書 §10 の7項目実行 → `Initial adoption verification: completed <date>` 追記（`Status: active` は維持）。

## レーン表

| Lane | Role | Goal | Write scope | Dependencies | Verification | Deliverable |
|---|---|---|---|---|---|---|
| L0 | Orchestrator（親） | rebase / PR 操作 / 承認ゲート / 検証統括 / 正本保存 | git 操作、`docs/plans/2026-08-15-*.md` | — | V1、`gh pr view` | rebase 済み branch、更新済み PR、正本計画 |
| L1 | Crafter | ROADMAP.md 全面改訂（Step 3 の 1〜9） | `ROADMAP.md` | L0 | V3(test 9項目)、V6 | 改訂 ROADMAP |
| L2 | Crafter | 設計文書昇格 + D14〜D18 + README 2件（D18 = milestone hygiene 判定を PR 内で確定） | `docs/developers/plans/active/2026-08-09-*.md`（mv 元）、`docs/developers/design/`、`docs/developers/plans/README.md` | L0, **L1**（テーマ文言と DoD 案を ROADMAP から転記） | V2 | canonical 設計文書（D18 確定済み） |
| L3 | Builder | 兄弟文書の事実訂正（列挙24行） | 兄弟3文書 | L0 | **24行の目視確認（主体）+ V8（補助 grep）** | 訂正済み3文書 + 24行の訂正前後対照表（変更不要行は根拠を記載） |
| L4 | Builder | contract test 実装（assert 9項目） | `tests/docs/test_root_roadmap_contract.py` | L1 | V3 | テスト |
| L5 | Builder | 公開 roadmap 同期 + ja 翻訳 | `docs_redesign/explanation/roadmap.md`、`docs_redesign/locales/ja/LC_MESSAGES/explanation/roadmap.po` | L1 | V4、V5 | 同期済み公開面 |
| L6 | Orchestrator（親） | Phase 4 GitHub metadata（全て承認ゲート、D18 は L2 で確定済みの記録を適用するのみ） | GitHub のみ（設計文書 §11 D18 は L2 が PR 内で確定済みのため L6 は書き換えない） | Step 9 merge | V7 | closed #657、hygiene 済み milestone、issue 3件 |
| L7 | Reviewer 系（read-only） | Phase 3 の PR diff レビュー（japanese-style-reviewer + Reviewer） | なし（read-only） | Step 8 後 | 指摘リスト（CRITICAL/HIGH 修正の追跡） | レビュー所見 |

write scope は相互排他（L1=ROADMAP のみ、L2 と L3 は別ファイル、リンク差し替えは L1 が担当）。L1 と L3 は Step 1 後に並行可、**L2/L4/L5 は L1 完了後**。

## Model And Effort 表

| Lane | Model | Effort | 理由 |
|---|---|---|---|
| L0 / L6 | 親セッション | — | 承認ゲートと git 操作は委任しない（mplan ガードレール） |
| L1 | opus | high | 正本文書の語彙設計・authority 一方向化・9項目の整合を一度で成立させる必要 |
| L2 | opus | high | 523行文書への decision log 追補と canonical 整合。D18（milestone hygiene 判定11件）の確定も含み、誤ると正本と後続の GitHub 操作が汚染される |
| L3 | sonnet | medium | 対象行が列挙済みの事実訂正。判断は「設計内容を変えない」境界の維持のみ |
| L4 | sonnet | medium | 仕様（assert 9項目）確定済みの実装。軽量パーサが要るため low にはしない |
| L5 | sonnet | low | sphinx-intl 手順 + 少数 msgid の翻訳。機械的 |
| L7 | 既定（agent 定義） | medium | read-only、diff 対象が文書のみ |

## タスクリスト

- [x] **Step 0: 正本保存**（2026-08-15 実施）— 本計画を `docs/plans/2026-08-15-v020-roadmap-canonicalization.md` として保存（PR に同梱）。`~/.claude/plans/` 側はコピー
- [x] **Step 1: rebase** — `git rebase origin/main`、V1 確認、force-with-lease push（2026-08-15 実施、V1 PASS、push 成功）
- [x] **Step 2: PR #660 再定義**（2026-08-15 実施）— title/body を canonical adoption 版へ更新、`gh pr view` で反映確認済み
- [x] **Step 3: ROADMAP.md 改訂**（2026-08-15 実施、9項目完了、4巡のレビューで Blocker 4件・Major 6件・Minor 8件を解消し Approved — 最終巡で v0.1.14 節の #614/#615/#620 重複記載除去〔実 milestone 9 scope に置換〕と冒頭 release-state 文の陳腐化を修正）
- [x] **Step 4: 設計文書昇格 + 追補 + D18（milestone hygiene 判定11件）を PR 内で確定**（2026-08-15 実施、L2）— `git mv` で `docs/developers/design/capability-domain-roadmap.md` へ昇格（rename 検出で追跡）。ヘッダを `Status: active` / `Authority: canonical` / `Audience: maintainer-facing` に更新。§3 consumer layer 行に #674 を追記。§5 matrix に2新テーマ列（I/O time & dispatch semantics、API compatibility & stabilization）を追加。§6 に両テーマの release statement/user story/受入成果物/Non-goals を ROADMAP から逐語転記。§8 に v0.1.14 節の関係行を追加。§9 の B-4 裁定を D16 優先で訂正。§11 に D14–D18 を追記。D18 は `gh issue/pr view` で実測したタイトルに基づき実メンバー11件を分類（#612 #637 #402 = DoD-required、#400 #413 #401 #508 #513 = gate-supporting、#594 PR#625 #403 = unrelated）。`gate-supporting` は DoD の検証・release evidence に加え、ROADMAP Workstreams が scope 内として明記した member も保持する。#401 は Step 11 で旧 P0 mapping が歴史的である旨の superseding comment を付す。Open-1〜3 をトラッキング issue #674/#675/#676 に更新。§12 に出典注記を追加。`docs/developers/design/README.md` を新設、`docs/developers/plans/README.md` に境界1行を追加。commit/push は未実施。
- [x] **Step 5: 兄弟文書の事実訂正**（2026-08-15 実施、L3、24行）— terrain/ScalarField 16行、SegmentTable 5行、ecosystem 3行の stale milestone/theme assignment を訂正。設計/API/依存関係は変更せず、terrain L243 の exclusion semantics は維持。`git diff --check` と V8 補助 grep（0件）を通過し、24行の旧文→新文/根拠を目視確認済み。commit/push は未実施。
- [x] **Step 6: contract test 新設**（2026-08-15 実施、L4）— `tests/docs/test_root_roadmap_contract.py` を新設。9 invariants と parser 境界条件の回帰テストを含む20 testsを追加し、Terra の3巡レビューで false-green（version heading、Markdown reference link、path traversal、複数 deferred block等）を解消。単体20 passed、`tests/docs/` 156 passed / 4 skipped、ruff・`git diff --check` PASS。`ROADMAP.md` は `Workstreams:` を独立ラベル行にする意味不変の最小整形のみ。
- [x] **Step 7: 公開 roadmap 同期**（2026-08-15 実施、L5）— 公開 roadmap の更新日を2026-08-15へ更新し、Long-Term Directionsにtime interpretation / reader semanticsの項目を追加。対象gettext catalogのみ同期して自然な日本語訳を追加。content-level fuzzy・空msgstr 0件、`msgfmt` PASS、公開content tests 10 passed。Terraレビューは指摘ゼロでApproved。
- [ ] **Step 8: 検証 V1〜V5 全通し + V8（補助）と L3 対照表（24行）の目視確認** — 2026-08-15 の pre-commit 検証で V1〜V4/V8 は完了。V5 は EN/JA とも、変更されていない optional ROOT notebook `docs_redesign/how-to/interop/intro_interop.ipynb` の同一セルが 600 秒で `CellTimeoutError` となり exit 2。対象 notebook と build 設定は本 PR の差分外であり、PR 非因果の環境依存 known limitation として受容予定だが、warning 0 未達のため成功扱いにはしない。最終 commit 後に V1 を再実行してから Step 8 を close する。ログ: `/tmp/gwexpy-step8-docs-en2.log`、`/tmp/gwexpy-step8-docs-ja.log`、`/tmp/sphinx-err-fqz1yg6w.log`、`/tmp/sphinx-err-5qek3bw5.log`。
- [ ] **Step 9: PR diff レビュー → merge**（承認ゲート）
- [ ] **Step 10: #657 final comment + close**（承認ゲート）
- [ ] **Step 11: milestone hygiene — Step 4 で確定済みの D18 を GitHub milestone に適用**（承認ゲート）
- [ ] **Step 12: #606 申し送り**（承認ゲート）— Open-1〜3 の issue 化は先行タスクで完了済み（#674〜#676）。重複作成しない
- [ ] **Step 13: 完了レビュー** — 並列品質レビュー + completion-auditor（mplan フェーズ6）

## 検証

```bash
# V1. 削除ゼロの一般不変条件 + 変更ファイルが下記一覧と一致（外部レビュー2巡目指摘によりrename検出を
#     有効化 — 素の --diff-filter=D は canonical 文書への git mv を類似度次第で D+A と誤認しうる）
git diff --find-renames --diff-filter=D --name-status origin/main...HEAD   # → 空
git diff --name-only origin/main...HEAD | sort            # → 変更ファイル一覧と一致

# V2. 旧 dated path の残存参照ゼロ（許容箇所なし — 実測で歴史的参照は存在しない）
git grep -n '2026-08-09-capability-domain-roadmap''-design' && echo FAIL || echo PASS  # → PASS

# V3. 文書テスト（新設 contract test 含む）
conda run -n gwexpy python -m pytest tests/docs/ tests/test_issue_burn_down_roadmap.py \
  tests/test_conda_forge_roadmap.py -q

# V4. ja catalog（-o /dev/null で CWD への messages.mo 生成を防止）
msgfmt --check -o /dev/null docs_redesign/locales/ja/LC_MESSAGES/explanation/roadmap.po
# + content-level fuzzy / 空 msgstr が 0 であること（ヘッダ行 L6 の fuzzy は対象外）

# V5. Sphinx: .github/workflows/docs-pages.yml の build ステップと同一コマンドで en/ja とも
#     warning 0（en = L92 の sphinx-build、ja = L100 の sphinx-build -D language=ja。
#     docs_redesign/README.md にはビルド手順が無い — 参照しない）
# 2026-08-15 pre-commit 結果: EN/JA とも、変更対象外の optional ROOT notebook
#     `how-to/interop/intro_interop.ipynb` の同一セルが 600 秒で CellTimeoutError、exit 2。
#     notebook/build 設定に本 PR の差分はなく、PR 非因果の環境依存 known limitation として
#     受容予定。ただし warning 0 は未達であり成功扱いにしない。ログは Step 8 行に記録。

# V6. リンク実在チェックは contract test assert 7 に一本化（shell 版は廃止 — 0件マッチで
#     偽 OK を出す構造だったため）

# V7. milestone 構成（issue list は PR #625 を返さないため api を使う）
rtk proxy gh api 'repos/tatsuki-washimi/gwexpy/issues?milestone=3&state=all&per_page=100' \
  --jq '.[] | {number, title, state}'

# V8. 兄弟文書の stale 断定の残存チェック（**補助 grep** — L3 完了判定の主体は
#     計画列挙の24行の目視確認。grep は太字 `**v0.2.0**` や表現揺れを完全には
#     捕捉できないため、PASS 単独では完了と見なさない。列挙外の REVIEW ヒットは
#     Step 5 の triage ルールで個別判断）
grep -nE '対象マイルストーン: \*{0,2}v0\.2\.0|v0\.2\.0 milestone|v0\.2\.0 の起票済みスコープ|v0\.2\.0 の P0|Milestone:? v0\.2\.0|\(v0\.2\.0\)|v0\.2\.0 は単一' \
  docs/developers/plans/active/2026-07-31-terrain-scalarfield-io-design.md \
  docs/developers/plans/active/2026-08-01-segmenttable-workflow-design.md \
  docs/developers/plans/active/2026-08-01-ecosystem-interop-plan.md \
  && echo REVIEW || echo PASS   # → PASS（0件）。REVIEW の場合は残存行を個別判断
```

**変更ファイル一覧**（V1 の期待値、12 logical files）: `ROADMAP.md` / `docs/developers/design/capability-domain-roadmap.md`（rename）/ `docs/developers/design/README.md`（新規）/ `docs/developers/plans/README.md` / 兄弟3文書 / `tests/docs/test_root_roadmap_contract.py`（新規）/ `docs_redesign/explanation/roadmap.md` / `docs_redesign/locales/ja/LC_MESSAGES/explanation/roadmap.po` / `docs/plans/2026-08-15-v020-roadmap-canonicalization.md`（新規）/ `docs/developers/plans/manifests/audit-manifest-660-roadmap-canonicalization.yaml`（新規）

## レビュー反映記録（mplan フェーズ2 → rev.3）

| 指摘（出所） | 採否 | 反映 |
|---|---|---|
| manifest 削除リスクは 2-dot diff の誤読（Auditor F1 / Critic B1、双方が merge-tree で実証） | **採用** | baseline 撤回、rebase 理由差し替え、V1 を diff-filter=D に置換、申し送り撤回 |
| #581 は milestone 外（Auditor F2） | 採用 | hygiene 判定対象を実メンバ11件に訂正 |
| 兄弟文書の v0.2.0 断定は L8 以外にも残存（Auditor F3） | 採用 | 12行を列挙 → plan-reviewer 3巡目の実測で**24行に拡張**、事実訂正のみ同 PR |
| CHANGELOG「#634 for v0.2.0」との矛盾（Auditor F4） | 採用 | v0.1.14 節に上書き宣言1文 |
| Zenodo DOI 実在（Auditor F6） | 採用 | DOI 記載、条件節削除 |
| po の fuzzy はヘッダマーカー（Auditor F7） | 採用 | タスク差し替え（sphinx-intl update 方式） |
| 検証コマンド4本の実効性不足（Auditor F8-F11 / Critic M8 / Audience） | 採用 | V2/V4/V6/V7 修正 |
| hygiene 判定基準の循環（Critic B2 / Audience） | 採用 | DoD 凍結 + 3分類 + 書き足し禁止 |
| ROADMAP↔milestone の権限循環（Critic B3） | 採用 | 一方向導出の明文化 |
| contract test の assert 設計不備（Critic B4） | 採用 | 8項目に全面差し替え、case-insensitive |
| テーマ名衝突（Critic M1） | 採用 | 「I/O time and dispatch semantics」に戻す |
| Gate 語の誤用 + scope-creep 規約未適用（Critic M2） | 採用 | Track 改名 + 受入成果物を ROADMAP 側にも |
| 3状態の非網羅（Critic M3） | **修正採用** | 5語化はせず、3語 + 適用範囲と対象外を明記 + 表示形式固定（過剰語彙より既存散文との整合を優先） |
| API compatibility テーマは4つ目の重複ホーム（Critic M4 / 削除提案） | **修正採用** | 削除せず「X3/#400 起点の継続テーマ」と定義し相互参照のみに（外部レビュー承認済みテーマ新設の意図を保持） |
| 判断根拠を milestone description に書くのは階層違反（Critic M9） | 採用 | D18 方式 |
| merge 後 hygiene の乖離窓（Critic M10） | 注記採用 | 一方向導出定義で吸収（順序は維持） |
| v0.1.15 policy が既存 narrowing 方針と衝突（Critic M11） | 採用 | narrowing/patch 可・API 追加/minor の線に書き直し |
| §13 新設は archival ノイズ（Critic 削除提案） | 採用 | 中止し D14〜D17 に要旨吸収（rev.2 からの変更点） |
| v0.1.14 節を表1行に（Critic 削除提案） | **不採用** | v0.1.13 節との体裁対称性を優先 |
| design/README 削除（Critic 削除提案） | **不採用（縮小）** | 6行の境界表に縮小して維持（design/ 初の narrative 文書のため） |
| maintenance 手続き未定義（Critic M13） | 採用 | ROADMAP に maintenance 節新設 |
| canonical 文書がビルド docs から不可視（Critic M14） | 採用（軽量） | ヘッダに maintainer-facing 明記 |
| レガシー web roadmap 放置（Critic M15) | 採用（軽量） | #606 へコメント申し送り |
| ChatGPT ログのパス不明で §13 が実行不能（Audience Blocker） | 採用 | 自己完結性宣言 + §13 中止で解消 |
| DoD の定義・所在が未リンク（Audience Blocker） | 採用 | 参照先を明記（v0.2.0 節 DoD、merge SHA で凍結） |
| 行番号の rebase 後有効性（Audience） | 採用 | 実測（対象ファイル変更0コミット）を明記 + 見出し再特定を指示 |
| Phase 5 完了条件なし（Audience） | 採用 | スコープ外化 + 完了条件定義 |

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `ROADMAP.md` | release-level 正本 = **包含基準の正本**（policy / released / Committed=v0.2.0 / Directional themes / v1.0 criterion / maintenance 手続き） |
| `docs/developers/design/capability-domain-roadmap.md`（移動先） | architecture/planning 正本。`Status: active` / `Authority: canonical` / maintainer-facing |
| `tests/docs/test_root_roadmap_contract.py`（新設） | ROADMAP 構造の contract test（9 assert） |
| `docs_redesign/explanation/roadmap.md` + ja po | 公開向け要約（版番号・状態語を出さない） |
| GitHub Milestone / Issue | 基準適用結果の正本 / execution 正本 |
| `CHANGELOG.md` + release_notes | 出荷事実の per-change 正本 |
| `docs/plans/2026-08-15-v020-roadmap-canonicalization.md`（新設） | 本計画の repo 内正本 |

## スコープ外

- v0.1.14 公開の事後処理（conda-forge 等）
- #637 composition redesign / Field I/O / SegmentTable の実装
- PR #488（GUI 抽出）の merge 判断
- `docs/latest-release-announcement` branch（manifest 問題の申し送りは**撤回** — 実測で非接触。単純に stale なだけ）
- レガシー `docs/web/{en,ja}/user_guide/roadmap.md` の実修正(#606)

## Reviewer Status

- 外部レビュー（2026-08-15）: Approve with revisions → rev.2 で反映済み
- mplan フェーズ2（Auditor / Critic / Audience 並列、2026-08-15）: Blocker 5件・Major 多数 → rev.3 で全件処理（上表）
- plan-reviewer 1巡目（2026-08-15）: needs-revision（Blocker 0 / Major 5 / Minor 7）→ 本 rev.4 で全件反映（L5 パス修正、V5 参照先訂正、Step 7 差分具体化、assert 1 判定基準確定、L2 を L1 依存化、行番号引用訂正、Critical Path に Step 2、Assumptions にツール前提、L7 行追加、DoD 引用訂正、V8 新設、L3 ツール許可）
- plan-reviewer 2巡目（2026-08-15）: needs-revision（Blocker 0 / Major 1 / Minor 1）→ 本 rev.5 で反映（V8 パターンを実測12行に合わせて拡充 + 「補助 grep、主体は目視確認 + 対照表」に降格。Step 7 bullet を Long-Term Directions へ移し配置根拠を明記）。その他の rev.4 内容は再修正不要と判定済み
- plan-reviewer 3巡目（2026-08-15）: needs-revision（Blocker 0 / Major 1 / Minor 0）→ 本 rev.6 で反映（V8 実測23ヒットに基づき Step 5 の対象を24行に拡張 — 選択肢 (a) 全行対象化を採用。列挙外ヒットの triage ルールも明記）。V8 パターン自体と Step 7 配置は正確と確認済み
- plan-reviewer 4巡目（2026-08-15）: **approved**（Blocker 0 / Major 0 / Minor 0。24行列挙は実測と完全一致、「12行」残存なし、triage ルール無矛盾、rev.5→rev.6 で意図外の drift なしを確認済み）
- 外部レビュー2巡目（2026-08-15、rev.6 に対して）: **Approve with two small revisions**（D18 を merge 後に repo へ追記する手順が post-merge の commit/push/merge 経路を欠いていた点、contract test の assert 4/8 が v0.1.14 節の消失を検出しない点）+ 参考指摘1件（V1 の rename 誤検出リスク）→ 本 rev.7 で全件反映: D18 を Step 4（PR 内）で確定させ Step 11 は適用のみに変更、contract test に assert 9（必須 release 見出しの下限チェック）を追加、V1 に `--find-renames` を追加
---

# 先行タスク: 計画書 commit&push + v0.2.0 の GitHub 早期反映（rev.7 Step 1 実行前・独立作業）

## Context

rev.7 は外部レビュー2巡目で「execution-ready」と評価され、設計上の blocker はない。ユーザーは Step 1（rebase）に進む前に、(a) 未 commit の計画書（rev.7）を PR #660 に同梱し、(b) v0.2.0 の新テーマ構造（Track A/B、API compatibility）と Phase 4 で計画済みだった Open-1〜3 追跡 issue を、ROADMAP.md 本文の改訂（Step 3）を待たずに GitHub 上へ先出しで告知・記録したいという意図を明示した。これは rev.7 の Step 1〜13 の番号・内容を変更しない、**独立した先行タスク**として実行する（Step 順に割り込ませない）。

## 現状確認（実行前に再検証すること）

- 現在の branch `docs/longterm-roadmap` は `git log` 先頭が `ae85889ac`（rev.7 内 baseline に記載の PR #660 head と一致）。つまり **このブランチ = PR #660 のブランチ**であり、ここでの commit push はそのまま PR #660 を更新する。実行直前に `git branch --show-current` と `gh pr view 660 --json headRefName,headRefOid` で再確認する。
- `git status` で計画書以外の意図しない変更が無いことを確認してから commit 対象を計画書ファイルのみに絞る（`git add` は明示パス指定、`-A` は使わない）。

## タスク A: 計画書の commit & push

- 対象ファイル: `docs/plans/2026-08-15-v020-roadmap-canonicalization.md`（rev.7、`~/.claude/plans/` 側コピーは commit 対象外 — リポジトリ外のため）
- commit message 案: `docs(roadmap): add v0.2.0 canonicalization execution plan (rev.7)`
- push: `git push`（現在の upstream に対して。force 不要 — 新規ファイル追加のみで rebase 前のため）
- 検証: push 後に `gh pr view 660 --json commits --jq '.commits[-1].messageHeadline'` で反映確認、`git status` がクリーンであることを確認

## タスク B: PR #660 への概要コメント

- 内容: rev.7 の要旨（authority の一方向化＝ROADMAP=inclusion criteria の正本／milestone=適用結果の正本、新設2テーマ〔I/O time and dispatch semantics の Track A/B、API compatibility and stabilization〕、3状態語彙、D18 の仕組み）を1コメントにまとめ、「ROADMAP.md 本体の改訂は本 PR 内で Step 3 以降として追って反映する」旨を明記。計画書パスへのリンクを含める。
- **送信前ガードレール**（`rules/common/github-mcp-workflow.md` 準拠）: 本文全文をプレビュー提示し明示承認を得る。送信前に手動パターン + `gitleaks stdin` でスキャン。

## タスク C: 関連 issue への個別コメント（7件）

各 issue に、どのテーマ・Track に属するかを1〜2文で通知するコメントを投稿（milestone・version の確定は行わない — D18 適用は Step 11 まで実施しない）:

- **Track A（Time interpretation contract）**: #632, #634, #636
- **Track B（Dispatch / reader semantics、#444→#616 の依存を明記）**: #444, #616
- **API compatibility and stabilization（X3/#400 起点の継続テーマ）**: #639, #640

各コメントは PR #660 へのリンクを含め、詳細はそちらを参照するよう誘導する短文とする。7件それぞれ送信前に本文プレビュー＋承認＋機密情報スキャンを行う（バッチ承認する場合も、7件全文を一括プレビューしてから明示承認を得る）。

## タスク D: 追跡 issue の新規作成（3件、Phase 4 Step 12 から前倒し）

active-plans 領域にあった capability-domain 設計文書の §11 Open-1〜3 を基に、実行時に本文を起草して新規 issue を作成する（本計画書には概要のみ記載し、正確な文言は実行時に §11 を読み直して作成する）:

1. **Open-1**: `gwexpy/cli/` の consumer layer 前提違反 — architecture decision issue。**milestone なし**。
2. **Open-2**: #637 decision date の確定トラッキング — actionable、v0.2.0 planning dependency と明記。決定方式（milestone mid-point で確定）は #637 の 2026-08-11 コメントに既に記録済みである旨を issue 本文に引用する。
3. **Open-3**: 性能退行予算のベースライン計測・数値化（#581 起点）— actionable、v0.2.0 planning dependency と明記。

3件とも v0.2.0 milestone には追加しない（milestone 追加は Step 11 の D18 判定・適用の対象外 — 3分類は既存メンバー11件に対して確定済みであり、新規issueをここで追加すると D18 の凍結対象が変化するため）。作成後、rev.7 の「主要ファイル」または関連箇所に issue 番号を追記するかはユーザー確認後に判断する。

**送信前ガードレール**: 3件それぞれタイトル・本文全文をプレビューし明示承認を得てから作成。機密情報スキャンを実施。

## 実行順序

タスク A → B → C → D の順（A は他の前提、B/C/D はそれぞれ独立だが読みやすさのため順に提示・承認を得る）。全て完了後、rev.7 の Step 1（rebase）へ進む。

## 検証

```bash
# タスクA
git log -1 --oneline -- docs/plans/2026-08-15-v020-roadmap-canonicalization.md
git status --porcelain docs/plans/2026-08-15-v020-roadmap-canonicalization.md   # → 空
rtk proxy gh pr view 660 --json commits --jq '.commits[-1].messageHeadline'

# タスクB/C/D（送信後）
rtk proxy gh pr view 660 --json comments --jq '.comments[-1].body' | head -c 200
rtk proxy gh issue view 632 --json comments --jq '.comments[-1].body'   # 他6件も同様
rtk proxy gh issue list --search "in:title Open-1 OR Open-2 OR Open-3" --json number,title,milestone
```

## スコープ外（本先行タスクでは行わない）

- ROADMAP.md / 設計文書本体の改訂（rev.7 Step 3〜4 で実施）
- 既存 v0.2.0 milestone メンバー11件への milestone/label 変更（D18 適用は Step 11）
- #657 のクローズ（Step 10）
- PR #660 の title/body 全面改訂（Step 2）

## 実施記録（2026-08-15）

- [x] **タスク A**: commit `98af62d56`（`docs(roadmap): add v0.2.0 canonicalization execution plan (rev.7)`）を `docs/longterm-roadmap` へ push。PR #660 の `headRefOid` が `98af62d5617d22def761746b5683398309335fd4` に更新されたことを確認済み。
- [x] **タスク B**: PR #660 へ概要コメント投稿済み（https://github.com/tatsuki-washimi/gwexpy/pull/660#issuecomment-5301634224）。送信前に手動パターン+gitleaks（exit 0, no leaks）でスキャン済み。
- [x] **タスク C**: 7 issue へ Track/テーマ帰属コメントを投稿済み — Track A: #632 (issuecomment-5301658560), #634 (issuecomment-5301658711), #636 (issuecomment-5301658827)／Track B: #444 (issuecomment-5301658989), #616 (issuecomment-5301659126)／API compatibility: #639 (issuecomment-5301659298), #640 (issuecomment-5301659454)。milestone/label は変更していない。
- [x] **タスク D**: 追跡 issue 3件を新規作成、milestone 未割当（`milestone: null` を実測確認）— Open-1 = #674、Open-2 = #675（#637 の2026-08-11コメント https://github.com/tatsuki-washimi/gwexpy/issues/637#issuecomment-5248951171 を引用）、Open-3 = #676。

先行タスク完了。rev.7 の Step 1（rebase）へ進む。
