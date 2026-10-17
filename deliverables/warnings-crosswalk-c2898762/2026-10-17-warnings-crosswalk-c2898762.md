# #781 Warning 照合レポート（`c2898762ca68`）

- 日付: 2026-10-17
- 母集団: Git 追跡 `gwexpy/**/*.py` **396 ファイル**（`tests/`・docs・未追跡除外）
- 成果物: `warnings-crosswalk-c2898762.jsonl`、`warnings_crosswalk_scan.py`、`coupling_probe_result.json`

## 1. 100 対 99（呼び出し件数）

| 方式 | 件数 | 定義 |
| --- | ---: | --- |
| A `classify_warn.py` 相当（行部分文字列ヒット） | **100** | 行に `warnings.warn` を含む |
| テキスト（括弧バランス付き `warnings.warn(`） | **99** | 実呼び出し |
| B AST（`warnings.warn` の `Call`） | **99** | 実呼び出し |

**余分 1 件の原因:** `gwexpy/interop/base.py:66` — Sphinx ドキュメント文字列 `` Forwarded to :func:`warnings.warn`. ``。実行コードではない。

**結論:** 監査 B/C の **99** と AST 照合は一致。A の **100** は docstring 参照を 1 件多く数えた行ベース grep の結果。

## 2. 59 対 24（`stacklevel` 未指定）

| 方式 | 未指定件数 | 定義 |
| --- | ---: | --- |
| A WF-01 / `classify_warn.py` | **59** | ヒット行から **4 行**のウィンドウ内に `stacklevel` 文字列が無い |
| 括弧バランス付き呼び出し全体 | **24** | 呼び出し括弧内に `stacklevel` キーワードが無い |
| B AST | **24** | `Call` の keyword に `stacklevel` が無い |

A 報告の 4 区分（39 / 33 / 20 / 8）は、上記 **4 行ウィンドウ**を `classify_warn.py` と同条件で再現した値と一致。

**59 − 24 の内訳（99 実呼び出しに限定）:**

- **34 件:** 呼び出し全体には `stacklevel=` があるが、開始行から 3 行以内のウィンドウには無い（多行呼び出し）。AST では `stacklevel` **あり**と正しく判定。
- **24 件:** AST でも本当に `stacklevel` keyword が無い（B C12 リストと一致）。
- **+1 件:** docstring 行ヒット（100 件側の未指定に寄与）。

**結論:** **24** が `gwexpy/` 内の `stacklevel` 未指定 **実呼び出し**の固定リスト。**59** は同一集合に対する **粗い行ウィンドウ集計**であり、#781 の「24 箇所」と矛盾しない。

### `stacklevel` 未指定 24 箇所（AST 確定）

`analysis/coupling.py:618`、`interop/pandas_.py:152,168`、`io/dttxml_common.py:48,740,1235,1253,1480`、`signal/preprocessing/whitening.py:164`、`spectral/estimation.py:251,696`、`timeseries/_resampling.py:888`、`timeseries/_signal.py:1487`、`timeseries/_spectral_special.py:1258`、`timeseries/decomposition.py:417,697`、`timeseries/matrix.py:276`、`timeseries/matrix_analysis.py:733`、`timeseries/preprocess.py:277,1101,1109,1427`、`types/metadata.py:998,1003`。

## 3. 実行検証（代表: ERR-021 / `coupling.py:618`）

- **経路:** 公開メソッド `CouplingFunctionAnalysis.auto_calibrate_percentile_factor`（`compute()` 内からも利用されうる較正 API）。
- **環境:** ローカル checkout + `PYTHONPATH=.`、Python 3.12、scipy 利用。
- **手順:** `minimize_scalar` を mock し `success=False` を強制（`warnings_crosswalk_probe_coupling.py`）。
- **結果:**
  - category: `UserWarning`（明示 category なし）
  - message: `Auto-calibration failed to converge. Using best guess.`
  - 表示位置: `gwexpy/analysis/coupling.py:618`（警告呼び出し行＝デフォルト `stacklevel=1`）
- **解釈:** 警告は発火する。公開呼び出し元の行を指すかは **未判定**（#781 / #513 の stacklevel 論点）。数値不具合の断定はしない。

## 4. 既存 Issue 対応（新規起票なし）

| 項目 | 追跡 |
| --- | --- |
| 24 箇所 `stacklevel` 欠落 | 既知 **B C12** → **#781**（**#513** 項目1と部分重複） |
| `coupling.py:618` | **A ERR-021** → **#781**（実行で警告確認、診断 defect は未確定） |
| docstring 100 件目 | 集計方法の説明のみ。不具合候補ではない |
| #513 statistics ラッパー | 本 24 件リストに **含まれない**（`stacklevel=2` 済み） |

## 5. 再現

```bash
cd deliverables/warnings-crosswalk-c2898762
python3 warnings_crosswalk_scan.py \
  --jsonl warnings-crosswalk-c2898762.jsonl \
  --summary warnings-crosswalk-c2898762-summary.json
PYTHONPATH=/path/to/gwexpy/repo python3 warnings_crosswalk_probe_coupling.py
```

## 6. 残存（#781 スコープ外）

- 各 24 箇所について public ラッパー経由の **表示行**を個別に実行確認していない。
- D F-11（`whiten` 等）の attribution は本パス未再実行。
- Warning **カテゴリ**・filter 可視性の契約判断は未実施。
