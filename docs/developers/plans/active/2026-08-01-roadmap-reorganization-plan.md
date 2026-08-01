# ロードマップ再編計画(v0.1.13 / v0.2.0 / Future themes)

> Last-updated: 2026-08-01 (rev 2 — 承認ゲート A 通過、GitHub 操作は一括実行で承認)
> Reviewer Status: **approved**(2026-08-01)

Status: completed (verified: `gh issue list --milestone v0.1.13 --state all --json number` → 18 issues matching the mapping table; `gh api repos/tatsuki-washimi/gwexpy/issues?milestone=3&state=all` → 30 items (29 issues + PR #488) for v0.2.0; `gh api repos/tatsuki-washimi/gwexpy/milestones?state=closed` → v0.1.3, v0.1.6, v0.1.12, v1.0.0 all closed; PR #488 and PR #453 carry roadmap-reorg comments; #507/#509/#510 unassigned with backlog comments; PR #613 merged as f47faa9ca via `gh pr merge 613 --squash --delete-branch`)

## 1. Goal

v0.1.12 リリース後のロードマップ検討(会話ログ `gwexpy_roadmap_conversation.md`、2026-08-01)で合意したバージョン体系を、(a) ルート `ROADMAP.md`、(b) 公開ドキュメント `docs_redesign/explanation/roadmap.md`、(c) GitHub milestone / issue 割当、の三点に整合的に反映する。

中核方針:

- **v0.1.13 は silent-corruption 安定化 patch**。例外を出さず誤った数値・単位・メタデータを返す既知バグの封鎖に限定し、新機能を含めない。
- **v0.2.0 は「Field I/O and Semantic Contracts」**。container 演算契約の公式化、Field 系 read/write の完成、eager SegmentTable workflow、docs 一本化、GUI 分離完了。
- **maintenance release(v0.1.14, v0.2.1+)には feature を事前割当しない**。
- **v0.3.0 以降はテーマの文書化のみ**。milestone は直前 minor のリリースまで作成しない。

## 2. Scope / Non-scope

### Scope

1. ルート `ROADMAP.md` の全面改訂(§8 のアウトライン)
2. 公開 `docs_redesign/explanation/roadmap.md` の整合(§9。バージョン番号は出さない)
3. GitHub 操作: milestone `v0.1.13` 新設、新規 issue 3 本起票(§6)、issue 割当・移動(§5)、不要 milestone の close、PR #453 / #488 の処理(§7)

### Non-scope

- v0.1.13 の実装作業そのもの(バグ修正コード)— 本計画完了後の次タスク
- 旧計画書の archive 移動(`tests/test_issue_burn_down_roadmap.py` / `tests/test_conda_forge_roadmap.py` が `active/` 内パスをピン留めしているため)
- GWexpy Studio リポジトリの新規作成
- `maint/0.1` 運用の変更(現在 main と同一。保険として温存)
- Ecosystem interop(GWDama / Differometor / Virgo 変換群)の実装・バージョン割当(Backlog 維持)

## 3. 最終バージョン体系

| Version | テーマ(一文) | milestone |
|---|---|---|
| **v0.1.13** | Silent-corruption stabilization patch — 誤った数値・単位・メタデータを黙って返す既知バグの封鎖。新機能なし | 新設する |
| **v0.1.14** | 事前割当なしの予備 maintenance release(v0.1.13 後に patch-safe な残件がある場合のみ発行) | 作らない |
| **v0.2.0** | Field I/O and Semantic Contracts — ①container 演算契約 ②Field 系 read/write(GSI DEM / GeoTIFF / canonical HDF5)③eager SegmentTable workflow ④docs 一本化(#606)⑤GUI 分離完了(PR #488) | 既存を拡張 |
| **v0.2.1+** | v0.2.0 公開後に判明した bug / regression の修正のみ。事前 feature 割当禁止 | 必要時のみ |
| **v0.3.0** | Advanced Segment Workflows(Reducer、groupby/aggregate、lazy SegmentFrame、reshape/explode) | 作らない(文書化のみ) |
| **v0.4.0** | Spatial Geometry + Layered Visualization(#556 系 GridGeometry/DetectorFrame、#558 系レイヤー描画) | 作らない(文書化のみ) |
| **v0.5.0** | Mesh-aware Fields + solver interop(#522、OpenFOAM #515 / FLOW-3D #516 / SPECFEM3D #517 / SimPEG #526) | 作らない(文書化のみ) |
| **v0.6.0** | Fisher forecasting(#570–#574、ORF は physics review 必須) | 作らない(文書化のみ) |
| Backlog | Ecosystem interop(GWDama #607、Differometor #423–#427、Virgo #598–#600)、I/O 性能(#580 系は #581 ベンチマーク基盤先行) | 割当なし |
| 未割当 | distributed execution、GUI 再設計(pyaggui 後)、meta-analysis、3D surface/volume model | 割当なし |

**GWexpy Studio**(初心者向け companion GUI): 別リポジトリ。v0.2.0 前は設計 prototype のみ、v0.2.0 リリース後に Quick Look alpha の開発開始。GWexpy 本体が約束するのは headless API(軽量 inspection、format 情報、serializable parameters、provenance)のみ。

## 4. 判断基準(v0.1.13 に入れる条件)

以下のいずれかに該当する open issue を v0.1.13 とする:

1. 正常終了しながら誤った数値・単位・dtype・時間軸を返す(P0: scientific/data integrity)
2. GWpy 互換 API の意味を破壊している(P1: contract integrity)
3. CI が実行していないテストを成功扱いする(P1)
4. 公開引数を受理しながら黙って無視する(P1–P2)
5. ドキュメントが未実装・存在しない API を実装済みとして示す(P1)
6. リスクが低く修正内容が明確な backward-compatible 小修正(P2)

新機能・API 追加・依存追加・大規模リファクタは含めない。

## 5. Issue → milestone マッピング表(承認対象の本体)

### 5.1 v0.1.13 へ割当(既存 16 件 + 新規 2 件 = 18 件)

| Issue | 現 milestone | 分類 | 内容 |
|---|---|---|---|
| #593 | なし | P0 | `from_root()` が TH1F/TH2F 等の非 double を float64 として誤読 |
| #575 | なし | P0 | Quantity/Unit 左演算で SeriesMatrix の型・単位・metadata 消失 |
| #576 | なし | P0 | SpectrogramMatrix 加減算が単位換算せず加算(1 cm を 1 m として) |
| #577 | なし | P0–P1 | `__array_ufunc__` 残欠陥(meta aliasing、name 上書き、`out=` 無視) |
| #578 | なし | P0 | FieldDict/FieldList が Quantity 演算で型・単位を失う |
| #579 | なし | P1 | Histogram が偶然の TypeError で保護されている脆い状態の固定 |
| #451 | なし | P1 | `TimeSeries.rms()` GWpy 互換破壊(→ PR #453 を rebase してマージ) |
| #508 | v0.1.12 | P1 | MC provenance が copy/slice/serialization で消失 |
| #511 | v0.1.12 | P1 | interop-mne CI gate が tests=0 でも green |
| #513 | v0.1.12 | P1–P2 | `_t0_ns` 精度、bool guard、stacklevel、rng typing |
| #559 | なし | P2 | `VectorField.plot(stride=)` の TypeError |
| #588 | なし | P2 | GWF reader の `parallel`/`nproc` silent no-op |
| #590 | なし | P2 | ndscope HDF5 writer の compression/chunks silent no-op |
| #594 | なし | 検証 | Virgo `.ffl` 対応宣言の実測(結果次第で fix 化 or 宣言修正) |
| #605 | **v0.2.0 から移動** | P1 | SegmentTable reference の未実装 API 記載を現行実装に一致させる |
| #608 | なし | P1 | GWinc docstring の存在しない classmethod |
| NEW-1 | — | P0 | WIN reader 12-bit sampling rate 誤読(§6.1、#519 から分離) |
| NEW-2 | — | P1 | reader の `start`/`end` silent 無視(§6.2、#584 から分離) |

補足: **#402(GWpy-only HDF5 golden tests)は v0.2.0 に残す**。v0.1.13 では手動検証のみ実施し、結果を issue にコメントする(release blocker にはしない)。

### 5.2 v0.1.12 の後始末

| Issue | 処理 |
|---|---|
| #507(有限標本 MC p 値補正) | milestone 解除 → backlog。physics review 完了時に patch release で再トリアージ |
| #509(MC cache semantics) | milestone 解除 → backlog。docstring/警告改善は v0.1.13 実装中に低コストなら同乗可 |
| #510(Student-t nu 不確かさ) | milestone 解除 → backlog |

残件ゼロになった時点で **v0.1.12 milestone を close**。

### 5.3 v0.2.0 へ追加(6 件 + 新規 1 件)

| Issue | 現 milestone | ワークストリーム |
|---|---|---|
| #520 | なし | Field I/O 共通 API(GWpy 形式 read/write 契約) |
| #521 | なし | heterogeneous FieldDict 一般化 |
| #547 | なし | ScalarField.read scaffold + GSI DEM reader |
| #551 | なし | GeoTIFF(2D exchange)+ round-trip regression |
| #553 | なし | Field direct I/O の public_io_contract 包含監査(v0.2.x から前倒し) |
| #606 | なし | docs/web(legacy)を docs_redesign へ統合 |
| NEW-3 | — | Container arithmetic semantic contract umbrella(§6.3) |

既存の v0.2.0 割当は維持: #355, #400–#404, #405, #406, #407, #408, **#409–#412(ユーザー裁定により維持)**, #413, #592, #596, #597, #601–#604(#605 のみ v0.1.13 へ移動)。

肥大リスク対応: #604(concat/join)・#406・#407 には「v0.2.0 の他項目完了時点で再査定し、必要なら v0.3.0 へ demote 可」のコメントを付す。

### 5.4 milestone の整理

| Milestone | 処理 | 理由 |
|---|---|---|
| v0.1.3 | close | 完全に空 |
| v0.1.6 | close | closed issue のみ |
| v1.0.0 | close | 完全に空。v1.0 条件は ROADMAP/設計文書で管理 |
| v0.1.12 | close(§5.2 完了後) | リリース済み |

close であり削除ではない(履歴保全・可逆)。

## 6. 新規 issue 起草文面(英語)

### 6.1 NEW-1: WIN reader 12-bit sampling rate 誤読

- Title: `WIN reader decodes only the low 8 bits of the 12-bit sampling rate (data corrupted for rates >= 256 Hz)`
- Labels: `bug`, `data-loss`, `io` / Milestone: v0.1.13

```
Split from #519 (native WIN writer): this is an existing *reader* correctness bug
and must not stay buried inside a writer feature request.

## Problem

`gwexpy/timeseries/io/win.py` decodes the per-channel sampling rate as

    datawide = float(buff[2] >> 4)   # L131
    srate = int(buff[3])             # L132

In the WIN format, the sampling rate is a 12-bit field: the low nibble of the
data-width byte holds the high 4 bits:

    sample_rate = ((buff[2] & 0x0F) << 8) | buff[3]

The current code therefore truncates any rate >= 256 Hz to its low byte
(e.g. 1000 Hz -> 232 Hz).

## Impact (P0, silent)

- The parser also derives the packet payload length from the rate
  (`xlen = (srate - 1) * datawide`), so for rates >= 256 Hz the byte stream is
  misaligned and the decoded samples themselves are garbage - not just the
  time axis.
- No exception is raised; the reader returns a plausible-looking
  TimeSeriesDict. Classic silent corruption.
- Rates <= 255 Hz are unaffected.

## Fix

1. Decode the 12-bit rate as above; keep <= 255 Hz behavior identical.
2. Regression tests with a spec-derived golden fixture (not produced by our
   own writer), covering at least one rate in 256-4095 Hz.

## Out of scope

- Native WIN writer (stays #519).
- Sample-size code 5 (32-bit absolute) support: currently raises
  NotImplementedError explicitly (win.py L185-190), so it is not silent;
  tracked in #519.
```

### 6.2 NEW-2: reader の `start`/`end` silent 無視

- Title: `read(start=..., end=...) silently ignored by some readers (NetCDF4 path strips the arguments without applying them)`
- Labels: `bug`, `io` / Milestone: v0.1.13

```
Split from #584 (backend push-down performance umbrella): the *correctness*
part - accepted public arguments that are silently no-ops - should be fixed
independently of any performance work.

## Problem

`gwexpy/timeseries/io/netcdf4_.py` (L179-181) strips gwpy-injected kwargs
before calling `xarray.open_dataset`:

    _gwpy_keys = {"start", "end", "pad", "gap", "nproc", "scaled"}
    xr_kwargs = {k: v for k, v in kwargs.items() if k not in _gwpy_keys}

`start` / `end` are then never applied as a selection, so
`TimeSeries.read(path, start=..., end=...)` returns the full time range while
appearing to honor the request. The same pattern exists at L313 for the
matrix path.

## Task

1. Audit readers that accept `start`/`end` (netcdf4_, zarr_, hdf5,
   ndscope_hdf5, ...) and classify: applied / ignored.
2. For each ignoring path, either apply the selection (post-load crop is an
   acceptable minimum) or raise/warn explicitly.
3. Regression test per affected reader: the returned span equals the
   requested range.

## Out of scope

Backend push-down (reading only the requested slice from disk) remains #584.
```

### 6.3 NEW-3: Container arithmetic semantic contract umbrella

- Title: `[Umbrella] Container arithmetic semantic contract: operand x operator x class coverage and unified policy`
- Labels: `contract`, `enhancement` / Milestone: v0.2.0

```
#575 #576 #577 #578 #579 are instances of a single bug family: external
operand types (Quantity / Unit / ndarray) capturing GWexpy containers and
silently dropping class, unit, or metadata. v0.1.13 fixes the known P0
instances; this umbrella formalizes the contract in v0.2.0 so the family
cannot regress.

## Known residual gaps (verified in code)

- Histogram classes define no `__array_ufunc__`
  (`gwexpy/histogram/_core.py`, `histogram.py`) - currently protected only by
  accident (#579).
- `FieldDict` arithmetic accepts only `np.isscalar` operands
  (`gwexpy/fields/collections.py` L179-205); Quantity/Unit are not scalars,
  so they fall through to undefined behavior (#578).
- `FieldList` defines no arithmetic operators at all.

## Deliverables

1. Contract document `docs/developers/contracts/container_arithmetic_contract.md`:
   - supported operands: python scalar, NumPy scalar, ndarray, Quantity,
     Unit, same-class container
   - supported operations: + - * / **, comparisons, reflected, in-place,
     explicitly selected ufuncs
   - preserved information: class, numerical value, physical unit, dtype,
     axes/xindex, row/col labels, channel, name, epoch, per-cell metadata
     independence (no aliasing)
   - unsupported-operation policy: TypeError / UnitConversionError /
     NotImplementedError - never a silent downgrade to bare ndarray/Quantity
2. Parametrized regression matrix: class x operand x operator x side
   (left/right) x in-place, asserting value, dtype, class, unit, axes,
   labels, metadata independence. Doubles as a compatibility gate for future
   NumPy/Astropy upgrades.
3. Close the three gaps listed above.
4. Migration notes for any behavior-visible change (three CHANGELOG
   categories: corrected wrong numerical results / prevented silent
   metadata-type loss / intentionally unsupported operations now raise).

## Related

#575 #576 #577 #578 #579 (P0/P1 fixes land in v0.1.13), #388 (Dict/List
collection arithmetic), #579 (Histogram).
```

## 7. GitHub 操作手順(実行順)

前提: 本計画書のユーザー承認(承認ゲート A)。各 issue 作成・milestone 変更は実行前にプレビュー提示。移動 issue には全件コメント「Moved as part of the roadmap reorganization (see docs/developers/plans/active/2026-08-01-roadmap-reorganization-plan.md).」を残す。

1. milestone `v0.1.13` 作成(description: `Silent-corruption stabilization patch — bug fixes only, no new features`)
2. 新規 issue 3 本を起票(§6。番号確定後、ROADMAP.md 最終化に使用)
3. §5.1 の 16 件を v0.1.13 へ割当(#605 は v0.2.0 から移動)
4. §5.2: #507 / #509 / #510 の milestone 解除 + コメント → v0.1.12 milestone close
5. §5.3 の 7 件を v0.2.0 へ割当。#604 / #406 / #407 に demote-possible コメント
6. §5.4: v0.1.3 / v0.1.6 / v1.0.0 milestone close
7. PR 処理:
   - PR #488(GUI 分離): milestone v0.2.0 を設定し、「マージは v0.1.13 リリース後」とコメント(ユーザー裁定 D-3)
   - PR #453(rms 修正): main へ rebase → CI → マージ。v0.1.13 実装の先頭タスク(本計画の完了条件には含めない)

ロールバック可否: milestone 変更・コメントは可逆。issue 作成は close でしか取り消せないため、プレビュー承認を必須とする。

## 8. ROADMAP.md 改訂アウトライン

- **前文**: テーマと方針を示す文書であり、直近リリースの正確なスコープは GitHub milestone が authoritative(現行注記を原則へ昇格)
- **Release policy**: patch はバグ修正のみ / maintenance release に feature を事前割当しない / 次々 minor の milestone は直前 minor リリースまで作らない
- **v0.1.13 — Silent-corruption stabilization patch (next)**: §5.1 をカテゴリ別に列挙(網羅列挙はこの節のみ)。#402 / #594 は「検証項目、release blocker ではない」と明記。Excluded: 新機能・API 追加・PR #488
- **v0.2.0 — Field I/O and Semantic Contracts**: 5 ワークストリーム各 1 段落 + umbrella/設計書リンク(個別番号は列挙しない)。Non-goals: 座標変換(#556)、basemap/layered viz(#558 系)、lazy/集約(v0.3.0)、mesh-aware model(#522)
- **Future themes (not scheduled)**: v0.3.0〜v0.6.0 を各 1 行 + 設計書リンク。「milestone は作成しない。テーマは変わり得る」と明記
- **Ecosystem & Interoperability (Backlog)**: 現行節を維持
- **Unassigned ideas**: distributed execution / GUI 再設計 / meta-analysis / 3D surface を各 1 行
- **GWexpy Studio (companion app)**: 別リポジトリ方針と本体側 headless API 契約を数行
- **Engineering hygiene (release-independent)**: 旧 v0.2.0/v0.3.0 節の mypy 厳格化・dependency locking・fixture 標準化・docs zero-warning を横断的継続項目として圧縮(内容を黙って捨てない)

## 9. 公開 roadmap(docs_redesign/explanation/roadmap.md)の方針

- **バージョン番号は出さない**(既存方針維持。release 契約と誤読されるのを防ぐ)
- Near-term に「silent data corruption の修正・信頼性強化」を追加
- Mid-term の抽象的記述を Field I/O(地理データ読込含む)・segment workflow 完結に具体化
- Long-term に空間ジオメトリ・メッシュ対応・Fisher forecasting を各 1 行
- `Last updated` 更新、ja `.po` 同期(`msgfmt --check` 必須)

## 10. 既存設計書との参照関係

| 設計書(docs/developers/plans/active/) | Reviewer Status | 対応バージョン |
|---|---|---|
| 2026-07-31-terrain-scalarfield-io-design.md | draft | v0.2.0 Field I/O(GSI DEM / GeoTIFF) |
| 2026-08-01-segmenttable-workflow-design.md | draft | v0.2.0(eager workflow)+ v0.3.0(集約・lazy) |
| 2026-08-01-layered-visualization-design.md | draft | v0.4.0 テーマの根拠文書 |
| 2026-08-01-ecosystem-interop-plan.md | approved | Backlog(Phase 2–4) |
| 2026-08-01-virgo-datadisplay-interop-design.md | draft | Backlog(#591 系) |
| 2026-07-30-v0.1.12-release-completion.md | approved | 完了済み(リリース 2026-07-31/08-01)。archive 移動は別途 |

## 11. Decision log

| # | 論点 | 裁定 | 日付 |
|---|---|---|---|
| D-1 | 実行範囲 | 全フェーズ実行(文書 + GitHub 再編) | 2026-08-01(ユーザー) |
| D-2 | #409–#412(median-mean / coupling schema)の扱い | v0.2.0 に残す | 2026-08-01(ユーザー) |
| D-3 | PR #488 のマージタイミング | v0.1.13 リリース後 | 2026-08-01(ユーザー) |
| D-4 | v0.3.0 以降の milestone | 作らない(文書化のみ) | 2026-08-01(ユーザー) |
| D-5 | #402 の所属 | v0.2.0 に残す。v0.1.13 では手動検証のみ | 推奨採用 |
| D-6 | #605 の所属 | v0.1.13(docs 正しさ修正) | 推奨採用 |
| D-7 | P0/P1 label の新設 | 追加しない(milestone + `data-loss`/`bug` で足りる) | 推奨採用 |
| D-8 | 公開 roadmap へのバージョン記載 | 記載しない(既存方針維持) | 推奨採用 |
| D-9 | v0.1.14 milestone | 必要時まで作らない | 推奨採用 |
| D-10 | 承認ゲート A | 承認・GitHub 操作は一括実行(操作ごとの再確認省略) | 2026-08-01(ユーザー) |

## 12. Verification checklist

### 文書 PR(PR-2)

- [ ] `sphinx-build -b html`(en / `-D language=ja`)EXIT 0
- [ ] `sphinx-build -b linkcheck` 新規 broken 0
- [ ] `conda run -n gwexpy python -m pytest tests/docs/` green
- [ ] `msgfmt --check` 対象 `.po` 全件 OK
- [ ] `conda run -n gwexpy python -m pytest tests/test_issue_burn_down_roadmap.py tests/test_conda_forge_roadmap.py` green(旧計画書の非破壊確認)
- [ ] ROADMAP.md 内の設計書相対リンクが実在する

### GitHub 再編完了時

- [ ] `gh issue list --milestone v0.1.13 --state open --json number` = §5.1 の 18 件と一致(差分ゼロ)
- [ ] v0.2.0 milestone 件数 = 既存 22(#605 移動後)+ 追加 7 = 29
- [ ] v0.1.12 / v0.1.3 / v0.1.6 / v1.0.0 milestone が closed
- [ ] 移動 issue 全件に説明コメントあり
- [ ] PR #488 に milestone v0.2.0 とタイミングコメントあり
- [ ] 本計画書の Status を `completed (verified: <上記コマンド>)` に更新
