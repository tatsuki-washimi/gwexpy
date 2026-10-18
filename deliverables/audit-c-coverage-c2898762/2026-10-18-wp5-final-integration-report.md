# WP5 最終統合台帳レポート

**作成日**: 2026-10-18  
**ベースライン台帳**: `c-coverage-c2898762.jsonl`（WP5では不変・読取専用）  
**最終台帳**: `wp5-final-ledger.jsonl`  

## 1. 自動集計（転記なし）

| Disposition | 件数 |
|---|---:|
| `verified` | 495 |
| `contract_pending` | 18 |
| `environment_blocked` | 5 |
| `differ` | 3 |

## 2. 証拠強度（verified の trial 連結再判定）

- L3（比較試行で実証）: 124
- L2（実質契約）: 188
- L1（スモーク/弱チェック）: 183
- 台帳ラベルと L3 実証の不一致: 52 件（`ledger_label_mismatch=true`）

## 3. WP4 是正項目の WP5 状態

- L3: 台帳 `gwpy_comparison` ラベルではなく **gwpy_comparison 試行のアサーション** で L3 を判定。
- 例外106: TypeError は **harness_zero_arg_typeerror / unclassified_typeerror** に再分類（94件一括確定を撤回）。
- 静的/B/Warning: `pass2-population-and-dispositions.json` 等から **件数を再計算**しハンドラIDを紐付け。

## 4. 遷移ログ件数: 252

