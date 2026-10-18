# Error-audit evidence → Issues/PRs → v0.2.6 work-state crosswalk

**Inventory date:** 2026-10-18 (repository snapshots pinned below). **Repository:** `tatsuki-washimi/gwexpy`. **Status:** proposed traceability document for review, **not** issue closure, patch approval, or release authorization.

- Product/audit observation baseline: [`c2898762ca68e20a1a241fffb3c3262b0a27290b`](https://github.com/tatsuki-washimi/gwexpy/commit/c2898762ca68e20a1a241fffb3c3262b0a27290b) (v0.2.5).
- Checked product main: [`035b7476849aa7842da6014069bcedc7f83fcf86`](https://github.com/tatsuki-washimi/gwexpy/commit/035b7476849aa7842da6014069bcedc7f83fcf86).
- Checked audit branch: [`5aec9eaf903db5d7fdc79c20c4faed11912a7171`](https://github.com/tatsuki-washimi/gwexpy/commit/5aec9eaf903db5d7fdc79c20c4faed11912a7171).
- Evidence source: [parent audit #765](https://github.com/tatsuki-washimi/gwexpy/issues/765), [WP4 static/warning report](https://github.com/tatsuki-washimi/gwexpy/blob/5aec9eaf903db5d7fdc79c20c4faed11912a7171/deliverables/audit-c-coverage-c2898762/2026-10-18-wp4-overall-static-and-warning-audit.md), [WP5 summary](https://github.com/tatsuki-washimi/gwexpy/blob/5aec9eaf903db5d7fdc79c20c4faed11912a7171/deliverables/audit-c-coverage-c2898762/wp5-final-summary.json), [WP5 Phase 4 handover](https://github.com/tatsuki-washimi/gwexpy/blob/5aec9eaf903db5d7fdc79c20c4faed11912a7171/deliverables/audit-c-coverage-c2898762/2026-10-18-wp5-phase4-handover-manifest.md), and individual Issue bodies. Original A/B/C/D reports sometimes reuse the same observations; multiple reports do **not** necessarily mean independent runtime reproductions.

## How to read this

**Four independent axes**, not a single done/open flag:

1. **Evidence:** `R` = observed runtime path with a real GWexpy execution (synthetic fixture allowed); `M` = mocked input / helper-only reproduction; `S` = source inspection only. `R報告` denotes a reported execution whose full raw trace was not independently reviewed here. `S + 隣接R` does not turn the unevaluated branch into an R.
2. **Contract:** observed behavior is not the same as approval of a new exception type, physics unit rule, default, or GWpy divergence. `契約→実装` means design/approval precedes a product change.
3. **Implementation:** no fixed product implementation, merged patch, or post-fix regression is established by these audit issues. A test of the pre-fix behavior does not count as a post-fix test.
4. **Release:** the last column is a **triage proposal, not an approved v0.2.6 milestone**. No issue is automatically a patch blocker or selected into v0.2.6. Major scientific/native safety findings deserve explicit risk assessment rather than automatic deferral.

## 1. Stage-3 audit: completed work and remaining acceptance

| Work package / claim | Evidence already produced | Remaining item (not a repeat of the completed investigation) |
| --- | --- | --- |
| WP0–WP5 integrated audit | 2,381-identifier population tracked; 521 detailed C-audit; 1,860 remainder with static/limited prior observations. 521 final WP5: **495 verified, 18 contract_pending, 5 environment_blocked, 3 differ**. | Formal independent G1–G6 acceptance/Issue disposition, not redoing WP0–WP5. Verified is test-contract evidence **not** 495 fixed product defects or blanket scientific parity. |
| WP2/WP3 | Initial WP2 pending 212 promoted to verified; 14/19 original environment blockers tested and promoted. | 4 live NDS2/DataFind methods and 1 ARIMA NumPy C-ABI blocker remain; do not relabel synthetic Arrakis/NDS2 conversions as end-to-end live-service verification. |
| WP4 warnings, [#781](https://github.com/tatsuki-washimi/gwexpy/issues/781) | WP4 report **already reconciles** textual 100 versus AST 99: one `gwexpy/interop/base.py:66` Sphinx docstring reference; 24/99 AST-identified warnings omit an explicit `stacklevel`; four-line regex produced 59 apparent omissions. | Preserve scan criteria and independently accept/publish the crosswalk; **separately** probe correct user-call-site attribution and fix only defective sites. 24 lacking a keyword ≠ 24 proven bad caller locations. Issue #781 text still predates WP4 reconciliation. |
| WP4 exception parity | 106 exception-related comparison entries revisited; zero-argument method calls are not valid proof of method behavior parity. | Valid-argument probes wherever needed; do not treat the 94 harness-generated TypeErrors as product defects or as full method equivalence. |
| WP5 evidence depth | GitHub WP5 JSON currently records L3=124/L2=188/L1=183 and 52 `ledger_label_mismatch`; final-integrated prose separately states 95/403/0, so the two documents are **not interchangeable**. | Review/normalize report semantics; no retroactive relabelling without evidence. Later local-only work reportedly refined 495 verified to L3=131/L2=230/L1=134 and analyzed all 52, but no such new commit was verified on this audit branch; independent acceptance/push are separate. |
| #765 completion language | The Phase-4 handover says population mapping/traceability satisfies the **author's proposed closure conditions**. | Parent [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765) is still open; verify formal acceptance and update tracking. Do **not** close solely on self-assessment; individual product fixes stay independent. |
| Product fixes & patch release | Audit branch generated tracking artifacts, **not product changes**. | Implement selected fixes, review behavior/physics/GWpy parity, qualify reviewed source, and satisfy release contract. No v0.2.6 release contract exists in current `RELEASING.md`. |

## 2. All 27 audit-derived Issues (one row per issue)

This maps **what the original audit already proved** separately from **what remains to make an implementation decision**. The release route is a recommendation for consideration, not a priority/severity/milestone assignment.

| Issue | Evidence | Completed observation / investigation | Still outstanding | Next gate | Release-route suggestion |
| --- | --- | --- | --- | --- | --- |
| [#766](https://github.com/tatsuki-washimi/gwexpy/issues/766) | R（A: subprocess / Python 3.11） | 欠損GWFでframel exit 139を再現。native経路まで追跡 | 別環境・backend比較、GWexpy/上流の責務、ガード実装とsubprocess回帰 | 契約判断→実装 | 別枠・安全性重点 |
| [#767](https://github.com/tatsuki-washimi/gwexpy/issues/767) | R/M（A/B/D） | 非互換PSD単位が警告後もpercentileに混入することを再現 | 実SegmentTable・下流影響、拒否/skip/opt-in契約、数値回帰 | 物理契約→実装 | 別枠・科学的リスク重点 |
| [#768](https://github.com/tatsuki-washimi/gwexpy/issues/768) | R（B/C/D） | zscore/robustの値と要素単位の不整合を再現 | `.units`/`meta.units`/要素・各shape/下流の保存契約を確定して修正 | 契約→実装 | 条件付き・科学レビュー |
| [#769](https://github.com/tatsuki-washimi/gwexpy/issues/769) | R（B/C/D） | 標準化後の物理単位残留を再現。未知method/定数scaleの挙動も報告 | 単位是正とmethod/定数契約を分割判定、容器別回帰 | 契約→実装 | 条件付き・科学レビュー |
| [#770](https://github.com/tatsuki-washimi/gwexpy/issues/770) | R（B/D） | radian/degree後のmatrix要素単位不整合を再現 | 角度・単位・要素/shapeの回帰、GWpy/独自契約の評価 | 契約→実装 | 条件付き・科学レビュー |
| [#771](https://github.com/tatsuki-washimi/gwexpy/issues/771) | R/M（B/D、一部A静的） | 非manifest HDF5の一部readerでskip/空collectionを観測 | 対象family列挙、valid-empty区別、manifest/上流契約、修正 | 範囲・契約→実装 | 別枠・データ欠落重点 |
| [#772](https://github.com/tatsuki-washimi/gwexpy/issues/772) | R（A） | 誤window長でSE correction factor=1になる挙動を再現 | 公開入口、SE因子の数学的契約、数値影響と回帰 | 物理契約→実装 | 条件付き・科学レビュー |
| [#773](https://github.com/tatsuki-washimi/gwexpy/issues/773) | M（B/C/D） | ROOT error Quantity不整合時に生数値へfallbackするhelperを再現 | 実ROOT `to_tgraph`、Series/Quantity枝、単位契約 | 実依存→契約→実装 | 別枠・物理単位 |
| [#774](https://github.com/tatsuki-washimi/gwexpy/issues/774) | R（B/C/D） | 不正frequency_unitがHzとして解釈されることを再現 | 外部HDF5実例、単位エラー契約、合法な別名/軸回帰 | 契約→実装 | 条件付き・科学レビュー |
| [#775](https://github.com/tatsuki-washimi/gwexpy/issues/775) | R（B/D） | read format推定が既知readerのI/O失敗を誤診断。暗黙`__context__`は残る | write/explicit-format枝の確認、例外伝播方針、回帰 | 契約→実装 | 条件付き・診断 |
| [#776](https://github.com/tatsuki-washimi/gwexpy/issues/776) | R（A） | MetaData不正単位が無警告で空文字化する挙動を再現。Channel別件は非再現 | 対応Astropy版・公開契約・伝播経路の確認、修正 | 契約→実装 | 条件付き・単位 |
| [#777](https://github.com/tatsuki-washimi/gwexpy/issues/777) | R（A） | HDF5でQuantity型attrsが無警告で落ちることを再現 | 永続化保証型の整理、save/reload、警告/拒否方針 | スキーマ契約→実装 | 別枠・データ損失 |
| [#778](https://github.com/tatsuki-washimi/gwexpy/issues/778) | R報告（B） | WhitenTransform inverseの元単位非復元を報告・コード照合 | B実測値の回帰化、コンテナ別数値/単位、inverse契約 | 契約→実装 | 条件付き・科学レビュー |
| [#779](https://github.com/tatsuki-washimi/gwexpy/issues/779) | R（B/C + GWpy比較） | `spectrogram2(method=...)`の内部periodogram TypeErrorを再現。GWpy同様 | docstringの既存仕様明確化なら局所変更。例外変更はGWpy互換承認・回帰 | docのみ先行可／例外は契約 | 小規模候補（文書部分） |
| [#780](https://github.com/tatsuki-washimi/gwexpy/issues/780) | M（B/C） | 模擬GWINC階層でdict modeが子traceを省略 | 実pygwinc階層・PSDのない正当なgroup区別、例外契約 | 実依存→契約 | 追加実証後 |
| [#781](https://github.com/tatsuki-washimi/gwexpy/issues/781) | S + 限定R（WP4報告、A/D） | WP4は100対99をdocstring1件で説明、AST99件・stacklevel欠落24件を記録 | 解析結果の独立受入・Issueへ反映、個別caller/wrapper位置の実行検証と修正 | **集計原因は調査済み**／位置是正未実施 | 小規模サブセットのみ条件付き |
| [#782](https://github.com/tatsuki-washimi/gwexpy/issues/782) | M（A） | 不正ASD unitにAstropy UnitsWarningが出てUnrecognizedUnitを保持する経路を観測 | 実FrequencySeries・対応Astropy・有効単位契約 | 実入力→契約 | 追加実証後 |
| [#783](https://github.com/tatsuki-washimi/gwexpy/issues/783) | R（A、Dは再利用） | Stage-3 probe `P08` は `load_dttxml_native` を実行済み。XML文字列をファイルパスとして渡し `OSError` 経路で `UserWarning`（`dttxml_common.py`、path 付き）の後 **空 `dict`** を返却。不正XML本文の parse 失敗→空 list ではない | 実ファイルパス上の不正/欠損 DTTXML、空 `dict` の意味、警告 vs 例外の公開契約 | 契約→実装 | 条件付き・診断 |
| [#784](https://github.com/tatsuki-washimi/gwexpy/issues/784) | R（B/C、GWpy 4.0.x） | `python -m gwexpy.time`のargparse import失敗を確認 | 公開CLIとしての支持範囲、対応GWpy版、native CLIとの比較後修正 | CLI契約→実装 | 契約次第で小規模 |
| [#785](https://github.com/tatsuki-washimi/gwexpy/issues/785) | S（A/D） | channel filter後の空Dict returnと単一Seriesの例外差をソース特定 | valid fixtureでall/partial/zero match、明示空リスト、GWpy/仕様 | 実行→契約 | 調査経路の追加必要 |
| [#786](https://github.com/tatsuki-washimi/gwexpy/issues/786) | S（A/D） | vectorized/parallel/futureのfallback枝とNaN挿入を構造確認 | 32ch以上を含む分岐実行・順位/警告、partial結果契約 | 実行→契約 | 別枠 |
| [#787](https://github.com/tatsuki-washimi/gwexpy/issues/787) | S（A/D） | significanceのzero-sigma未保護式を特定 | zero/near-zero sigmaを実行、統計上のNaN/inf契約 | 実行→物理契約 | 別枠 |
| [#788](https://github.com/tatsuki-washimi/gwexpy/issues/788) | S（B/D） | B-channel PSD・CSDの例外を`pass`する非対称枝を確認 | failure injectionとGUI表示・cache・log検証、部分更新契約 | 実行→契約 | 条件付き（単純log追加で確定不可） |
| [#789](https://github.com/tatsuki-washimi/gwexpy/issues/789) | S（A/D） | couplingに異なる3 fallback機構を特定。percentile枝はlogger.warningあり | 3枝それぞれ再現しCF/threshold/p10p90下流と契約を検証 | 実行→物理契約 | 別枠 |
| [#790](https://github.com/tatsuki-washimi/gwexpy/issues/790) | S（A/D） | 2つの異なるgrid不一致が同一UserWarningとなることを確認 | 両枝fixtureを実行、残存行/数値不変を確認し文言のみ修正 | 実行テスト→実装 | 小規模最優先候補 |
| [#791](https://github.com/tatsuki-washimi/gwexpy/issues/791) | S + 隣接R（D） | 不正xarray unitで警告後にunitを落とす挙動。警告は明示的 | 正当な単位形式・境界・caller位置を確認。既存仕様一致ならno-change | 契約確認／修正未決 | no-changeの可能性 |
| [#792](https://github.com/tatsuki-washimi/gwexpy/issues/792) | S（A/B、Dはxarray枝のみ） | MetPy/Pintの広いcatchとxarray fallbackを特定。**MetPy失敗枝の実行再現なし** | 実MetPy/Pintでdequantify/converter失敗を注入、属性/通知契約 | 実依存→契約 | 追加実証後 |

## 3. Additional audit findings *without* standalone Issues

| Source | Already done | Remaining / release treatment |
| --- | --- | --- |
| [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765)-C1 (`normalize_series`) | Phase-4 handover identifies `gwexpy/gui/plotting/normalize.py` dictionary fallback `val.get("data") or val.get("value")`, causing ambiguous NumPy truth evaluation. Also confirmed still in main source. | Small targeted fix + positive/negative array, zero-valued element, and fallback tests; decide whether to split/track as implementation Issue. **v0.2.6 shortlist candidate**, not fixed. |
| [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765)-C2 (`local_hurst`) | Handover identifies R/S backend short-window failure; main conversion of window to `w_samples` exists. | Decide valid R/S boundary and downstream behavior; add validation/diagnostics + backend-specific tests. Candidate only after contract check. |
| B-N01 histogram on-demand registration | Parent #765 contains initial negative-control and HDF5 positive-control probes; **no registration defect demonstrated**. | Only run supported non-HDF5 real format with fresh import if that question becomes release-relevant. No automatic fix. |
| B-N02 TimeSeriesMatrix CSV warning | Initial runtime captured warning before `ValueError` (expected warning exists), caller attribution library-internal. Cross-linked [#775](https://github.com/tatsuki-washimi/gwexpy/issues/775) / [#781](https://github.com/tatsuki-washimi/gwexpy/issues/781). | Test caller location / adjust stacklevel when selected; **not** a new unresolved discovery task. |
| C narrative `standardize_timeseries` | Ordinary TimeSeries with unit was checked and correctly made dimensionless; branch-specific fallback unverified. | No-change for tested case; only validate supported alternate class when needed. Do **not** merge into [#768](https://github.com/tatsuki-washimi/gwexpy/issues/768)/[#769](https://github.com/tatsuki-washimi/gwexpy/issues/769). |
| Other C narratives / upstream parity | `Plot.set` etc. identified as framework-injected; `spectrogram(method="mean")` GWpy-parity in B C15 tested scope; `Array2D.step`, `Plane2D.step`, short `q_gram` GWpy-parity limitations identified. | Keep as documented framework/upstream/coverage boundaries; no automatic code change or new v0.2.6 blocker. |

## 4. Relevant pre-existing Issues: split mixed work items

| Issue | Completed / known | Remaining / patch suitability |
| --- | --- | --- |
| [#513](https://github.com/tatsuki-washimi/gwexpy/issues/513) item 1 (warning `stacklevel`) | Direct-function versus TimeSeries-wrapper call-stack difference specified, source inspected. | Fix with **both** entry points attributed correctly; selective small patch candidate after deterministic warning probes. Do not simply bump all `stacklevel=2`. |
| [#513](https://github.com/tatsuki-washimi/gwexpy/issues/513) item 2 (`rng` typing) | Runtime accepts duck-typed `.random(size)` according to existing stub fixture. | Protocol/type annotation and static type test; public typing contract review. Separate from item 1. |
| [#513](https://github.com/tatsuki-washimi/gwexpy/issues/513) item 3 (`window=True`) | `bool` acceptance identified in existing `isinstance(int)` guard. | Explicit rejection changes supported input behavior; patch eligibility review + test. Separate from item 1. |
| [#513](https://github.com/tatsuki-washimi/gwexpy/issues/513) item 4 (`_t0_ns`) | Float64 epoch precision bottleneck identified. | Scientifically meaningful exact-epoch contract & representative type checks; not a documentation-only change. |
| [#640](https://github.com/tatsuki-washimi/gwexpy/issues/640) item 1 (`ignore_nan`) | Different defaults in TimeSeriesMatrix and FrequencySeriesMatrix described. | Decide defaults & numeric regressions; **not** an automatic small patch. |
| [#640](https://github.com/tatsuki-washimi/gwexpy/issues/640) item 2 (`TimeSeriesDict.append`) | Documentation omission identified; current implementation returns `self`. | Docstring clarification with no behavior change; small documentation candidate. |
| [#569](https://github.com/tatsuki-washimi/gwexpy/issues/569) | Existing plot label formatter and raw f-string inconsistency identified. | Replace limited sites + Matplotlib visual/axis-label regression; optional cosmetic candidate. |
| [#594](https://github.com/tatsuki-washimi/gwexpy/issues/594) | FFL read-vs-upstream test was **performed in PR #625's description** and design note marked verified; initial issue checklist still looks unfinished. | Track follow-up through PR #625; don't rerun original investigation as if never done. PR merge/rebase/review is separate. |
| [#638](https://github.com/tatsuki-washimi/gwexpy/issues/638) | PR #625 branch measured top-level FFL exception types and unbounded line length. | Depends on PR #625 code **not merged into main**; no standalone main bug, don't mix into v0.2.6. |
| [#681](https://github.com/tatsuki-washimi/gwexpy/issues/681) | `__array_ufunc__=None` diagnostic limitation and 96-regression failed prototype documented. | Contract/design research if prioritized; **not** a quick diagnostic tweak under current B0 semantics. |
| [#616](https://github.com/tatsuki-washimi/gwexpy/issues/616) | zarr `pad=` TypeError and registry path contrast actually reproduced. | Uniform backend contract/design choice. Error is loud; not a new reproduction task. |
| [#601](https://github.com/tatsuki-washimi/gwexpy/issues/601) | Need for a SegmentTable `errors=` policy documented. | New parameters/status API imply **minor feature**, not v0.2.6 patch. |
| [#403](https://github.com/tatsuki-washimi/gwexpy/issues/403) | `nproc` preserved as non-deprecated `parallel` compatibility alias under existing policy. | Future policy documentation/decision only; do not introduce a deprecation for patch. |

## 5. Open PR mapping (all four at this snapshot)

| PR | Existing completed artifact | Unfinished gate / disposition for lightweight v0.2.6 |
| --- | --- | --- |
| [#764](https://github.com/tatsuki-washimi/gwexpy/pull/764) | Dependabot proposal, **54 Python dependency updates**; branch exists but unmerged. | Broad environment changes require their own CI/compatibility review; **exclude** from small diagnosis patch. |
| [#763](https://github.com/tatsuki-washimi/gwexpy/pull/763) | Dependabot proposal, four GitHub Actions updates; unmerged. | Release/CI workflow behavior risk; **exclude** from patch source until separately reviewed. |
| [#717](https://github.com/tatsuki-washimi/gwexpy/pull/717) | Draft docs user-testing *plan*, no participant sessions completed per PR body. | User sessions/report, draft review; independent of package patch. |
| [#625](https://github.com/tatsuki-washimi/gwexpy/pull/625) | FFL real/synthetic contract investigation performed; patch with tests and feature path exists on PR (9 files, +758/−43 reported in GitHub). | Unmerged, originally based on old v0.1.13; update/review/qualify for separate release; #638 is dependent. Do not count as a released fix. |

## 6. Release decision queue — no further global rescreen required

**Candidate for shortest isolated changes (still subject to test and review):**
[#790](https://github.com/tatsuki-washimi/gwexpy/issues/790) (warning-only); [#779](https://github.com/tatsuki-washimi/gwexpy/issues/779) (docs-only subsection); [#640](https://github.com/tatsuki-washimi/gwexpy/issues/640) item 2 (docstring); [#513](https://github.com/tatsuki-washimi/gwexpy/issues/513) item 1 (small if wrapper/direct tests pass); [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765)-C1 (array truth-value bug). [#784](https://github.com/tatsuki-washimi/gwexpy/issues/784) only after confirming the intended CLI contract. [#569](https://github.com/tatsuki-washimi/gwexpy/issues/569) optional visual polish. These are not pre-approved v0.2.6 scope.

**Scientific/native hazard review, not simply postponed:** [#766](https://github.com/tatsuki-washimi/gwexpy/issues/766), [#767](https://github.com/tatsuki-washimi/gwexpy/issues/767), [#768](https://github.com/tatsuki-washimi/gwexpy/issues/768), [#769](https://github.com/tatsuki-washimi/gwexpy/issues/769), [#770](https://github.com/tatsuki-washimi/gwexpy/issues/770), [#771](https://github.com/tatsuki-washimi/gwexpy/issues/771), [#773](https://github.com/tatsuki-washimi/gwexpy/issues/773), [#774](https://github.com/tatsuki-washimi/gwexpy/issues/774), [#777](https://github.com/tatsuki-washimi/gwexpy/issues/777), [#778](https://github.com/tatsuki-washimi/gwexpy/issues/778). For each, decide the risk of releasing while the observed failure is outstanding and preserve the physics/GWpy contract. A small diff does not remove the need for scientific review.

**Reproduction and contract follow-ups remain genuinely pending:** source-only [#785](https://github.com/tatsuki-washimi/gwexpy/issues/785), [#786](https://github.com/tatsuki-washimi/gwexpy/issues/786), [#787](https://github.com/tatsuki-washimi/gwexpy/issues/787), [#788](https://github.com/tatsuki-washimi/gwexpy/issues/788), [#789](https://github.com/tatsuki-washimi/gwexpy/issues/789), [#790](https://github.com/tatsuki-washimi/gwexpy/issues/790), [#792](https://github.com/tatsuki-washimi/gwexpy/issues/792); mocked/partial real-input confirmation [#780](https://github.com/tatsuki-washimi/gwexpy/issues/780), [#782](https://github.com/tatsuki-washimi/gwexpy/issues/782); contract follow-up after executed abnormal-input probe for [#783](https://github.com/tatsuki-washimi/gwexpy/issues/783) (real on-disk malformed XML not yet in `P08` scope); intentional no-change outcome remains possible for [#791](https://github.com/tatsuki-washimi/gwexpy/issues/791). This does not invalidate the *completed static or mocked investigation*.

**Formal publication is separate:** the current [RELEASING.md](https://github.com/tatsuki-washimi/gwexpy/blob/main/RELEASING.md) says v0.2.5 is the published release and **no future version is selected/contracted**. Select v0.2.6 scope/owner/review lanes, add version-specific release contract, review exact code and evidence, run the prescribed candidate/promotion protocol only if selected by the contract, and verify GitHub + PyPI readback. An audit QA PASS never substitutes for product test and release authorization.

## 7. Review checklist for this crosswalk

- [ ] Check Issue bodies against this evidence-stage crosswalk; avoid overwriting original observations or assigning new P0–P3 priorities without agreement.
- [ ] Accept the WP4 warning reconciliation (99 executable calls, 24 missing explicit stacklevel) and update [#781](https://github.com/tatsuki-washimi/gwexpy/issues/781) **only for the completed scan-count subtask**. Keep actual caller-attribution remediation open.
- [ ] Reconcile contradictory WP5 depth reports and incorporate the reported local-only 52-case classification **after independent verification**, without silently changing baseline evidence.
- [ ] Decide whether to split #765-C1/C2 into implementation Issues and prioritize small candidates under a separate, version-specific release contract.
- [ ] Keep parent [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765) open pending its formally agreed acceptance state; no issue or PR is closed/merged by this document.

**Scope discipline:** This document is a work-state crosswalk of the 27 #765 child Issues, relevant legacy Issues, two handover defect candidates, and the four currently open PRs. It is **not** a review of every one of the repository's ~130 open Issues, a fresh execution of product regressions, or proof that local unpublished files are on GitHub.
