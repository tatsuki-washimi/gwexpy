# B-N01 / B-N02 — 初回診断プローブ（#765）

**位置づけ:** 初回実行観測。B-N01 の契約判定完了を意味しない。

- 監査ベースライン: `c2898762ca68e20a1a241fffb3c3262b0a27290b`
- 再実行:

```bash
cd /path/to/gwexpy
PYTHONPATH=. python3 deliverables/audit-b-n01-n02-probes/b_n01_n02_runtime_probe.py
```

## ファイル

| ファイル | 内容 |
| --- | --- |
| `b_n01_n02_runtime_probe.py` | プローブ本体 |
| `b_n01_n02_probe_results.json` | 環境・観測 JSON |
| `2026-10-17-b-n01-n02-runtime-notes.md` | 人間可読メモ |

## 区分ラベル

`observed` / `source_confirmed` / `contract_pending` / `not_tested` — 各ケース JSON 内。

## 関連 Issue

- B-N02 → #775（例外すり替え）、#781（警告表示位置）
- 新規 Issue は本プローブでは起票しない
