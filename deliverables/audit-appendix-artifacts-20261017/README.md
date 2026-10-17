# GWexpy 監査付属個票（B/C）— 第2段階機械可読成果物

固定ベースライン: `c2898762ca68e20a1a241fffb3c3262b0a27290b`
追跡用親 Issue: [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765)

## 含まれるファイル

| ファイル | 由来 | 用途 |
| --- | --- | --- |
| `pass2-population-and-dispositions.json` | chat2plan B (`20261017T075741Z-91daf9`) | 155 ハンドラ disposition、2,650 公開サーフェス（B 内蔵） |
| `pass1-population.json` | chat2plan C (`20261017T064433Z-c8bd51`) | 396 ファイル・API 行・155 静的ハンドラ位置 |
| `pass2-api-status.jsonl` | 同上 | Pass2 状態 3,046 行（api/file/io/dynamic） |
| `pass2-module-functions.txt` | 同上 | 2219 外の慣例公開関数 162 件 |
| `crosscheck_audit_appendix_stage2.py` | 本リポジトリ | 再実行可能な独立照合 |
| `crosscheck_audit_appendix_stage2_results.json` | スクリプト出力 | 機械検査の確定結果（`passed: true/false`） |
| `2026-10-17-audit-appendix-stage2-crosscheck.md` | 本リポジトリ | 人間可読の照合レポート |
| `SHA256SUMS` | 生成 | 原本4ファイル＋照合成果物のハッシュ |

## 再現手順

```bash
cd deliverables/audit-appendix-artifacts-20261017

python3 crosscheck_audit_appendix_stage2.py \
  --artifact-dir . \
  --output crosscheck_audit_appendix_stage2_results.json
```

D 台帳（B 2,650 行との突合）を含める場合、同ディレクトリの親に D ZIP があるときは自動で
`../agent-deliverables-20261017-175223.zip` を読みます。明示する例:

```bash
python3 crosscheck_audit_appendix_stage2.py \
  --artifact-dir . \
  --d-zip ../agent-deliverables-20261017-175223.zip \
  --output crosscheck_audit_appendix_stage2_results.json
```

終了コード `0` かつ JSON の `"passed": true` で、第2段階の機械検査が再現できています。

### D ZIP 依存（必読）

| 条件 | `b_d_surface_2650` | 全体 `passed` |
| --- | --- | --- |
| `--d-zip` / 同梱 D ZIP あり | 2,650 行突合実行 | B/C + B↔D すべて満たせば `true` |
| D 未指定かつ親 ZIP 不在 | `skipped: true` | B/C のみ満たせば **`true` のまま**（B↔D は未実施） |

完全な stage-2 再現には **`deliverables/agent-deliverables-20261017-175223.zip`**（監査 D、123KB 程度）を同じ `deliverables/` に置く。配布 ZIP `agent-deliverables-20261017-200607.zip` には D 本体は含めない（サイズ・出所の分離）。

## 解釈上の境界（重要）

- **起票漏れ照合**: B 155 / C 2,219 API の写像・件数・既知候補 ID の割当。
- **製品 QA**: 異常入力の網羅検証ではない。B `no_change` 141 は定型理由文の反復であり、個別正当性の実証ではない。
- **B ↔ D 2,650 一致**: 記録整合性であり、独立した二重の実行検証ではない。

## 残存検証（#765 / 子 Issue）

- #781 — `warnings.warn` 件数差と stacklevel クロスウォーク
- B-N01 / B-N02 — Histogram 非 HDF5、TimeSeriesMatrix CSV 警告経路（実行再現）
- C — 361 `pass2_not_evaluated` API、162 慣例関数のうち 160 未評価

## 原本パス（参考）

chat2plan 実行ディレクトリ（ローカルマシン上）:

- B: `~/.local/state/chat2plan/runs/20261017T075741Z-91daf9/execution/`
- C: `~/.local/state/chat2plan/runs/20261017T064433Z-c8bd51/execution/`

以前の ZIP [`agent-deliverables-20261017-195716.zip`](../agent-deliverables-20261017-195716.zip) は原本4ファイルの初回同梱。本ディレクトリは照合スクリプト・JSON 結果・README を追加した固定版です。
