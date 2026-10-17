# 付属個票 第2段階独立照合レポート

- 日付: 2026-10-17
- ベースライン: `c2898762ca68e20a1a241fffb3c3262b0a27290b`
- 親 Issue: [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765)（セクション *Appendix stage-2 raw ledger crosscheck* と整合）

## 目的

原報告 B/C が参照する chat2plan 機械可読台帳を、リポジトリ外原本から取得し、**起票漏れ照合**（追跡管理の完全性）を item 単位で再現可能に固定する。製品全体の異常系 QA 完了とは混同しない。

## 方法

1. B `pass2-population-and-dispositions.json`、C `pass1-population.json` / `pass2-api-status.jsonl` / `pass2-module-functions.txt` を配置。
2. `crosscheck_audit_appendix_stage2.py` を実行し、`crosscheck_audit_appendix_stage2_results.json` を生成。
3. 任意で D `pass2-surface-coverage.jsonl`（`agent-deliverables-20261017-175223.zip`）と B 内蔵 `public_surface_2650` を行単位突合。

## 結果サマリ

| 項目 | 結果 |
| --- | ---: |
| B ハンドラ件数 | 155 |
| `no_change` / `confirmed_issue` / `further_verification` | 141 / 10 / 4 |
| B ↔ C pass1 ハンドラ位置（順序・path・line） | 一致 |
| フラグ付き 14 ハンドラの未割当 candidate | 0 |
| C pass2 行数 | 3,046 |
| C 一意 API `member_id` | 2,219（pass2 欠落 0） |
| `pass2_executed` / `pass1_evidence_reused`（member 単位） | 6 / 2 |
| `pass2_not_evaluated` のみ（member 単位） | 361 |
| 慣例モジュール関数（txt） | 162 |
| B 2,650 ↔ D 2,650（id・candidate・disposition・順序） | 一致 |

**結論（起票漏れ）:** 機械可読台帳上、既知 B 候補 C1–C14 に対応しないフラグ付きハンドラはない。原報告で名前付きされた候補の Issue 写像は、155/2219 個票まで確認済み。

## 限界（検証の境界）

### B `no_change` 141 件

`note` 文言は **3 種類の定型文**に集約（件数内訳は JSON `b_no_change_note_buckets`）。Disposition ラベルと位置対応は検証したが、**141 箇所それぞれの実装正当性は実証していない**。

### C カバレッジ

- 1,850 member は `pass2_static_pattern_recheck` のみ（AST/パターン再チェック）。
- 162 慣例関数は jsonl 外；原報告付録どおり 160 `needs_more_checks` / 2 `pass1_evidence_reused`。
- D 2,650 行一致は **記録の整合性**であり、B と D の独立した実行検証ではない。

### 実行未了

- B-N01（Histogram 非 HDF5 契約・登録）
- B-N02（TimeSeriesMatrix CSV 警告位置、#775/#781 連鎖の実行再現）
- A 100 vs B/C/D 99 の warning 件数差（#781）

## 再現

```bash
python3 crosscheck_audit_appendix_stage2.py --artifact-dir . \
  --output crosscheck_audit_appendix_stage2_results.json
```

詳細は同梱 `README.md`。

## #765 クローズ判定との関係

親 Issue の **追跡管理完了**（finding → Issue / no_change / further_verification の写像）は、本照合で機械可読裏付けまで到達した。
**子 Issue の修正完了**や **API 異常系 QA 全体**は別軸。B-N01/B-N02 等の保留は #765 内 disposition と子 Issue リンクで引き継ぎ可能であり、親を Open に保つかは監査記録固定後に判断してよい。
