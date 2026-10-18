# WP5 → 第4段階 引継ぎパッケージ

**作成日**: 2026-10-18  
**親Issue**: #765  

## 未解決バケット（実装修正・優先順位は決定しない）

### 製品欠陥候補
- `normalize_series` 辞書入力の真偽値評価（`normalize.py` L27）— triage 候補、WP5では起票のみ。

### 上流依存の既知挙動（differ 3件）
- `Array2D/step`, `Plane2D/step`, `TimeSeries/q_gram` — GWpy 同等例外。

### ハーネス起因（contract_pending へ再分類）
- 残余 differ 21件のうち 18 件を harness/fixture 要因として contract_pending 化。

### 環境制約
- WP3 により LAL/PyCBC 変換系は verified へ昇格済み（trial: wp3-trials.jsonl）。

### 静的のみ調査（1,860）
- 2,381 母集団のうち C 監査外 1,860 は pass2 静的調査範囲。実行到達は WP5 範囲外。

## 既存 Issue 対応
- B 監査 flagged handlers → #767, #768, #769, #771, #773, #775, #787, #788 等（crosscheck JSON 参照）。

## ゲート自己評価（WP5 生成時点）
- G1/G2: 521 行 + wp5-final-ledger 生成、遷移ログ付き。
- G3: trial-linked depth / 例外再分類 JSON 出力。
- G4: wp5-cross-audit-evidence.json に handler ID 列挙。
- G5: 製品コード変更なし（wp5_qa_audit.py で確認）。
- G6: 本ドキュメント + issue_tracking フィールド。

