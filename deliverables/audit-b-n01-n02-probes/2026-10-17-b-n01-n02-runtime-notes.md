# B-N01 / B-N02 初回実行観測（2026-10-17）

詳細は `b_n01_n02_probe_results.json`（環境ブロック付き）。

## B-N01

| ケース | disposition | 結果 |
| --- | --- | --- |
| missing `.csv`（Dict/List） | `observed` | `FileNotFoundError`、警告 0（**負例**） |
| HDF5 roundtrip（Dict/List） | `observed` | 正常読込 |
| 非 HDF5 実ファイル | `contract_pending` | 契約上確認された形式未指定のため `not_tested` |

登録遅延による不具合: **`not_observed`**（missing 負例のみでは登録健全性は証明しない）。

## B-N02

| 項目 | disposition |
| --- | --- |
| 損失警告 → `ValueError` の順序 | `observed` |
| 警告 `filename`/`lineno`（`matrix.py:202` 付近） | `observed` |
| 例外 `__cause__`/`__context__` | JSON 参照（#775 経路） |
| 公開呼び出し行を指すべきか | `contract_pending` |

「警告が早い」ことと「表示行が公開 API か」は別論点（#781）。
