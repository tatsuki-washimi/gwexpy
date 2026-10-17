# GWexpy C監査：残存 differ 44件の技術的原因究明と完全トリアージ台帳

更新日: 2026-10-18
監査対象コミット（ベースライン）: `c2898762ca68e20a1a241fffb3c3262b0a27290b`
作業ブランチ: `audit/c-coverage-508-parallel`
是正コミット: [`1fcc72d25f750f6eab69d62778034699cffbf2ff`](https://github.com/tatsuki-washimi/gwexpy/commit/1fcc72d25f750f6eab69d62778034699cffbf2ff)
親Issue: #765 (Open維持)
製品コード改変: **厳格0行（製品コード改変なし）**

---

## 1. エグゼクティブサマリー

GWexpy C監査母集団521件（並列バッチ508件＋事前13件）において残存していた `disposition=differ` 44件について、監査ハーネス・GWpyソースコード・GWexpy実装・NumPy/SciPy/Astropy挙動の4層照合による深層調査を実施しました。

### 主要な結論

1. **ステップ1（ハーネス偽陽性23件の解消と厳格契約検証）**:
   - **最優先11件（`shift` 5件、`value_at` 6件）**: `np.isclose(...)` の戻り値 `numpy.bool_` が検証側の厳格判定 `is True` で `False` と評価されていた不整合を特定。アサーション側の厳格判定（`is True`）を一切緩めず、生成側で明示的に Python `bool(np.isclose(...))` へ変換することで、11件すべてを安全に `verified` へ移行させました。
   - **残る12件（`is_contiguous` 6件、`crop` 3件、2D `diff` 3件）**: 単なるアサーション緩和を排し、API仕様に対応する厳格な期待値判定を実装しました。
     - `is_contiguous` (6件): 端点で接する区間（`shift(span[1] - span[0])`）で `contiguous_following == 1`、重複区間（`copy()`）で `overlapping_not_contiguous == 0` を厳格検証。
     - `crop` (3件): サンプル格子に整合する区間（`x0+dx, x0+2dx`）で切り出し、境界スパン一致・単位保持・出力点数1を厳格検証。GWpyとの完全一致を確認。
     - 2D `diff` (3件): 2D配列の仕様に従い最終軸 `axis=-1` の差分（`res.shape[-1] == obj.shape[-1] - 1`、他軸不変、`np.diff(..., axis=-1)` との値一致、単位保持）を厳格検証。GWpyとの完全一致を確認。
   - **ステップ1の結果**: `differ` は 44件から **21件** へ縮減し、`verified` は 246件から **269件** へ増加しました。

2. **GWpyパリティ記録の厳密な区別（14件の証拠上の位置付け）**:
   - 解消した23件のうち、台帳（`c-coverage-c2898762.jsonl`）に記録されているGWpy比較分類の実態は以下のとおりです：
     - `numeric_value_parity`: 5件（`Spectrogram/crop`, `Array2D/crop`, `Series/crop`, `Spectrogram/diff`, `Array2D/diff`）
     - `exception_parity`: 14件（`shift` 4件、`value_at` 5件、`is_contiguous` 5件）
     - `incomparable`: 4件（`Plane2D` の `shift`, `value_at`, `is_contiguous`, `diff`）
   - **14件が `exception_parity` と記録されている理由**: 並列バッチのGWpy比較試行において、引数必須メソッドに対し比較器が引数なしで呼出しを行ったため、GWexpy・GWpy双方が同一の `TypeError: missing required positional argument` を送出したことによります。
   - したがって、公的台帳上の記録としては厳密に `exception_parity` です。GWexpy内部での契約検証（`positive_control`）は有効引数で実呼出しに成功（`verified`）しており、また手動のオフライン検証では有効引数による完全な数値・戻り値一致を確認していますが、台帳の試行記録と混同しないよう本台帳では `exception_parity` として正確に記録します。
   - **残余21件中の `Array2D/read` および `Series/read` の位置付け**: 台帳（`c-coverage-c2898762.jsonl`）の試行記録では、両者ともに引数なし呼出し時の `TypeError: read() missing 1 required positional argument` 送出一致により `exception_parity` と記録されています。手動検証ではCSV往復時の単位非保存契約（無単位復帰）を特定していますが、公的台帳の証拠記録との完全な整合を期すため、トリアージ台帳でも `exception_parity (引数なし呼出し時のTypeError一致。CSVラウンドトリップについてはGWpyとの有効入力パリティ未実証)` として表記を統一しています。

3. **ステップ2（残余21件の契約・入力条件確定と完全トリアージ）**:
   - **上流由来の既知挙動 (`upstream_behavior` / 3件)**:
     - `Array2D.step`, `Plane2D.step`: Seriesから継承された `step()` が画像プロットに対し `drawstyle` を渡し `AttributeError: AxesImage.set() got an unexpected keyword argument 'drawstyle'` を送出。GWpy本体でも100%同一例外が発生することを確認。
     - `TimeSeries.q_gram`: データ長1秒（1024サンプル）ではQ変換の周波数ビン探索で `IndexError: index 513 is out of bounds` が発生。GWpy本体でも100%同一例外が発生することを確認（十分な長さ8秒のオフライン検証では正常動作）。
   - **時系列信号処理契約 (`contract_pending` / 6件)**:
     - `whiten`, `find_gates`, `gate`, `mask`, `q_transform`: 短データ（1024点）でのパラメータ境界エラーや必須引数（`deadtime: SegmentList`）の要求。
     - `zpk`: ハーネスが渡した極 `poles=[100]` が離散時間 z平面極（$|p| \ge 1$）として扱われ発散（NaN）。連続時間 `analog=True` または安定な極（$|p| < 1$）で正常動作を確認。
   - **モジュール関数・I/O契約 (`contract_pending` / `gwexpy_defect` 候補 / 12件)**:
     - `normalize_series`: ハーネスが素の `np.array` を渡したため `None` 復帰。Seriesオブジェクトでは正常に `(x, y)` を返却。なお、製品コード `normalize.py` L27 `data = val.get("data") or val.get("value")` に辞書入力時のNumPy配列真偽値評価例外およびゼロ配列フォールバックの潜在的製品バグが存在することを特定（Issue起票対象）。
     - `local_hurst`: バックエンド `hurst` パッケージ（R/S）が時系列長 $\ge 100$ を要求するため、ハーネスの `window=2` で NaN となっていた。窓長 $\ge 100$ で正常に有限値 $[0, 1]$ が計算されることを確認（ただし他バックエンドへの一律制約適用には留意が必要）。
     - `get_model`: ハーネス検証器がモデル関数（callable）返却に対し `finite` 配列検証を無条件要求していた不整合。
     - `unique_hdf5_key`, `normalize_layout`, `write_hdf5_manifest`, `detect_hdf5_layout`, `parse_timezone`, `read_timeseriesmatrix_audio`: ハーネス側の期待値誤認（`key__1` vs `key_1`、`dataset-per-entry` vs `dataset`、ZoneInfo vs timezone.utc 等）。

---

## 2. 状態遷移集計表

| 指標 | 修正前 (HEAD) | ステップ1修正後 | 差分 | 備考 |
| :--- | :---: | :---: | :---: | :--- |
| **全母集団** | 521 | 521 | 0 | 508並列＋事前13件 |
| **verified** | 246 | **269** | +23 | 23件のハーネス偽陽性解消・厳格検証 |
| **contract_pending** | 212 | 212 | 0 | 属性アクセス72件等の保留維持 |
| **differ** | 44 | **21** | -23 | 真の残余21件を確定 |
| **environment_blocked** | 19 | 19 | 0 | LAL/NDS2等の環境依存 |
| **全試行数** | 1,572 | 1,572 | 0 | 521母集団直結1,559＋参照プローブ13 |
| **QA監査 (qa_audit.py)** | 17/17 PASS | **17/17 PASS** | 0 | 全17項目完全合格（QA14: 269件） |

---

## 3. 原因群別の内訳と分類

残存44件は以下の4群に完全に一対一分類されます。

```
残存 44件
 ├── A群: 多次元・基底MRO (8件)
 │    ├── 解消 (5件): Spectrogram/crop, Array2D/crop, Spectrogram/diff, Array2D/diff, Plane2D/diff
 │    └── 残余 (3件): Array2D/step (upstream), Plane2D/step (upstream), Plane2D/append (contract)
 ├── B群: 時系列信号処理 (7件)
 │    ├── 解消 (0件)
 │    └── 残余 (7件): find_gates, gate, mask, q_gram (upstream), q_transform, whiten, zpk
 ├── C群: Series個別契約 (20件)
 │    ├── 解消 (18件):
 │    │    ├── shift (5件): FrequencySeries, Spectrogram, Array2D, Plane2D, Series
 │    │    ├── value_at (6件): FrequencySeries, Spectrogram, TimeSeries, Array2D, Plane2D, Series
 │    │    ├── is_contiguous (6件): FrequencySeries, Spectrogram, TimeSeries, Array2D, Plane2D, Series
 │    │    └── crop (1件): Series/crop
 │    └── 残余 (2件): Array2D/read, Series/read (CSV単位非保存の契約仕様)
 └── D群: モジュール関数・ヘルパー (9件)
      ├── 解消 (0件)
      └── 残余 (9件): get_model, normalize_series, unique_hdf5_key, detect_hdf5_layout,
                      write_hdf5_manifest, normalize_layout, parse_timezone, local_hurst, read_timeseriesmatrix_audio
```

---

## 4. 44件完全トリアージ台帳（10項目詳細）

各列の定義:
1. **識別子**: 対象APIの完全修飾名
2. **群**: 原因分類群（A: 多次元・MRO [8件] / B: 時系列信号処理 [7件] / C: Series個別契約 [20件] / D: モジュール関数 [9件]）
3. **API契約・期待値**: 本来の入出力仕様・前提条件・事後条件
4. **旧観測結果**: ベースライン（`c2898762`）における試行結果・エラー
5. **原因技術詳細**: 失敗に至った技術的根本原因
6. **ハーネス妥当性**: ハーネス側の入力フィクスチャ・アサーションの評価と是正内容
7. **GWpy比較**: 公的台帳記録上の判定（※オフライン有効引数検証結果を付記）
8. **最終判定**: `harness_false_positive`（解消済）/ `upstream_behavior` / `contract_pending` / `gwexpy_defect`（潜在）
9. **科学的リスク**: 物理量・信号処理計算への影響度
10. **追跡先**: 追跡コミット / 今後起票・改善候補

| # | 識別子 | 群 | API契約・期待値 | 旧観測結果 | 原因技術詳細 | ハーネス妥当性 | GWpy比較 | 最終判定 | 科学的リスク | 追跡先 |
| :-: | :--- | :-: | :--- | :--- | :--- | :--- | :--- | :-: | :--- | :--- |
| 1 | `FrequencySeries/shift` | C | `f0` を指定Δだけシフト | shift_applied=False判定 | `np.isclose` が `numpy.bool_` を返し `is True` 判定が失敗 | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 2 | `Spectrogram/shift` | C | 時間軸 `t0` をシフト | shift_applied=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 3 | `Array2D/shift` | C | `x0` をシフト | shift_applied=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 4 | `Plane2D/shift` | C | `x0` をシフト | shift_applied=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | incomparable (GWpyなし) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 5 | `Series/shift` | C | `x0` をシフト | shift_applied=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 6 | `FrequencySeries/value_at` | C | 座標 `f` の値を補間/取得 | value_matched=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 7 | `Spectrogram/value_at` | C | 座標 `(t, f)` の値を取得 | value_matched=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 8 | `TimeSeries/value_at` | C | 座標 `t` の値を取得 | value_matched=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 9 | `Array2D/value_at` | C | 座標 `(x, y)` の値を取得 | value_matched=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 10 | `Plane2D/value_at` | C | 座標 `(x, y)` の値を取得 | value_matched=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | incomparable (GWpyなし) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 11 | `Series/value_at` | C | 座標 `x` の値を取得 | value_matched=False判定 | 同上 (`numpy.bool_` 不整合) | 判定不整合（`bool()` 変換で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では数値一致) | **harness_false_positive (解消)** | 低（実動作は正常） | コミット 1fcc72d25 (是正済) |
| 12 | `FrequencySeries/is_contiguous` | C | 後続接合で1、重複で0を返却 | is_compatible_result=False | `obj.copy()` を渡したため完全重複（戻り値0）となり `is True` 失敗 | フィクスチャ誤り（接合区間生成と1/0厳格検証で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では1/0一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 13 | `Spectrogram/is_contiguous` | C | 後続接合で1、重複で0を返却 | is_compatible_result=False | 同上（`copy()` で戻り値0） | フィクスチャ誤り（接合区間生成と1/0厳格検証で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では1/0一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 14 | `TimeSeries/is_contiguous` | C | 後続接合で1、重複で0を返却 | is_compatible_result=False | 同上（`copy()` で戻り値0） | フィクスチャ誤り（接合区間生成と1/0厳格検証で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では1/0一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 15 | `Array2D/is_contiguous` | C | 後続接合で1、重複で0を返却 | is_compatible_result=False | 同上（`copy()` で戻り値0） | フィクスチャ誤り（接合区間生成と1/0厳格検証で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では1/0一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 16 | `Plane2D/is_contiguous` | C | 後続接合で1、重複で0を返却 | is_compatible_result=False | 同上（`copy()` で戻り値0） | フィクスチャ誤り（接合区間生成と1/0厳格検証で是正） | incomparable (GWpyなし) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 17 | `Series/is_contiguous` | C | 後続接合で1、重複で0を返却 | is_compatible_result=False | 同上（`copy()` で戻り値0） | フィクスチャ誤り（接合区間生成と1/0厳格検証で是正） | exception_parity (引数なし呼出し時のTypeError同等性 ※オフライン有効引数では1/0一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 18 | `Spectrogram/crop` | A | 指定時間区間を切り出し | bounds_valid=False | `dx=0.5` に対し `(0.2, 0.8)` 指定のためスパン `[0.0, 0.5)` となり `0.0 >= 0.2` 不成立 | 格子不整合（格子適合範囲 `(x0+dx, x0+2dx)` で是正） | numeric_value_parity (完全一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 19 | `Array2D/crop` | A | 指定X区間を切り出し | bounds_valid=False | 同上 (`dx=0.5` 格子外指定) | 格子不整合（格子適合範囲で是正） | numeric_value_parity (完全一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 20 | `Series/crop` | C | 指定X区間を切り出し | bounds_valid=False | 同上 (`dx=0.5` 格子外指定) | 格子不整合（格子適合範囲で是正） | numeric_value_parity (完全一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 21 | `Spectrogram/diff` | A | 最終軸 (`axis=-1`) で差分 | shape_decreased=False | 2D配列では列方向（axis=-1）差分のため `shape[0]` は不変。ハーネスが `shape[0]` 減少のみ検査 | アサーション誤り（`shape[-1]` 減少および値一致で是正） | numeric_value_parity (完全一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 22 | `Array2D/diff` | A | 最終軸 (`axis=-1`) で差分 | shape_decreased=False | 同上 (2D差分軸誤認) | アサーション誤り（`shape[-1]` 減少および値一致で是正） | numeric_value_parity (完全一致) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 23 | `Plane2D/diff` | A | 最終軸 (`axis=-1`) で差分 | shape_decreased=False | 同上 (2D差分軸誤認) | アサーション誤り（`shape[-1]` 減少および値一致で是正） | incomparable (GWpyなし) | **harness_false_positive (解消)** | 低（仕様通り動作） | コミット 1fcc72d25 (是正済) |
| 24 | `Array2D/step` | A | 1D階段プロット（非推奨） | AttributeError | Series.step()がAxesImageにdrawstyleを渡し例外 | 親クラスからの不整合継承。GWpyでも完全同一例外送出 | exception_parity (AttributeError一致) | **upstream_behavior** | 低（2D画像プロットでstepは非推奨） | 今後起票候補（親Issue #765 配下） |
| 25 | `Plane2D/step` | A | 1D階段プロット（非推奨） | AttributeError | 同上 | 同上 | incomparable (GWpyなし) | **upstream_behavior** | 低（同上） | 今後起票候補（親Issue #765 配下） |
| 26 | `Plane2D/append` | A | 同種Plane2Dの接合 | ValueError: discontiguous | ハーネスが `other` に `Array2D` を生成したため接合拒否 | ハーネスフィクスチャ誤り（`Plane2D` 同士であれば正常接合） | incomparable (GWpyなし) | **contract_pending** | 低（同種間では正常動作） | 次期ハーネス改善候補 |
| 27 | `Array2D/read` | C | CSVからの読み込み | unit_matches=False | CSVフォーマットは単位をシリアライズしないため無単位復帰 | CSV契約の誤認（CSV読込で単位一致を期待したハーネスの誤り） | exception_parity (引数なし呼出し時のTypeError一致。CSVラウンドトリップについてはGWpyとの有効入力パリティ未実証) | **contract_pending** | 低（CSV仕様上の既知挙動） | 仕様契約明文化候補 |
| 28 | `Series/read` | C | CSVからの読み込み | unit_matches=False | 同上 | 同上 | exception_parity (引数なし呼出し時のTypeError一致。CSVラウンドトリップについてはGWpyとの有効入力パリティ未実証) | **contract_pending** | 低（同上） | 仕様契約明文化候補 |
| 29 | `TimeSeries/find_gates` | B | 信号ゲーティング区間検出 | ValueError: Invalid Parameter | 1024点データに対し窓パラメータが超過 | 入力長・パラメータ不整合（長時間データまたは適切な閾値が必要） | exception_parity | **contract_pending** | 中（物理信号パラメータ適合要） | 契約テスト拡張候補 |
| 30 | `TimeSeries/gate` | B | 信号ゲーティング適用 | ValueError: Invalid Parameter | 同上 | 同上（`tzero, tpad` 等の明示指定が必要） | exception_parity | **contract_pending** | 中（同上） | 契約テスト拡張候補 |
| 31 | `TimeSeries/mask` | B | フラグ/デッドタイムマスク | ValueError: deadtime or flag | `deadtime` 引数が未指定 | ハーネス引数不足（`deadtime: SegmentList` 必須） | exception_parity | **contract_pending** | 低（必須引数未指定） | 契約テスト拡張候補 |
| 32 | `TimeSeries/q_gram` | B | Q変換イベント探索 | IndexError: index 513 out of bounds | 短データ（1024点）ではQプレーンのタイル探索がFFT長超過 | GWpy由来の短データ制限。8秒以上では両者とも正常出力 | exception_parity (IndexError一致) | **upstream_behavior** | 中（短データ時の境界ガード検討） | 今後起票候補（親Issue #765 配下） |
| 33 | `TimeSeries/q_transform` | B | 時間-周波数Q変換 | ValueError: Invalid Parameter | 1024点データに対しデフォルトQレンジ探索が超過 | 長時間データ（4秒以上）または `qrange` 明示指定が必要 | exception_parity | **contract_pending** | 中（解析パラメータ依存） | 契約テスト拡張候補 |
| 34 | `TimeSeries/whiten` | B | 白色化フィルタ適用 | ValueError: Invalid Parameter | 1024点データに対し `fftlength` 境界条件不成立 | GWpy共通。十分な長さ（4秒以上）または `fftlength` 指定が必要 | exception_parity | **contract_pending** | 中（PSD推定長依存） | 契約テスト拡張候補 |
| 35 | `TimeSeries/zpk` | B | 零点・極・ゲインフィルタ | finite_values=False (NaN) | ハーネスが渡した極 `[100]` が離散時間極として扱われ発散 | ハーネス引数誤り（連続時間 `analog=True` または安定極 $\|p\| < 1$ で正常） | exception_parity | **contract_pending** | 高（不安定フィルタ時の挙動保護） | 契約テスト拡張候補 |
| 36 | `get_model` | D | フィッティングモデル関数取得 | checks: finite_values=False | 関数（callable）を返すファクトリに対し配列有限性検証を要求 | ハーネス検証器のロジック不整合（関数評価ステップの欠落） | incomparable | **contract_pending** | 低（ファクトリは正常返却） | 次期ハーネス改善候補 |
| 37 | `normalize_series` | D | 系列入力の (x, y) 正規化 | is_tuple=False (None復帰) | ハーネスが素の `np.array` を渡し None 復帰。なお辞書入力時NumPy配列で `ValueError` 潜在バグ特定 | ハーネス入力誤り＋製品潜在バグ（L27 `or` 評価のNumPy曖昧性） | incomparable | **contract_pending / gwexpy_defect (潜在)** | 中（GUIプロット辞書入力時エラー） | 今後起票候補（親Issue #765 配下） |
| 38 | `unique_hdf5_key` | D | 一意なHDF5キー名生成 | success=False | ハーネスが `"key_1"` を期待したが実装は `"key__1"` を返却 | ハーネス期待値誤認（アンダースコア仕様の不一致） | incomparable | **contract_pending** | 低（一意性は担保） | 次期ハーネス改善候補 |
| 39 | `detect_hdf5_layout` | D | HDF5格納レイアウト自動判定 | success=False | ファイル内にデータセット/グループ未作成のため None 復帰 | ハーネスフィクスチャ誤り（空ファイルで判定要求） | incomparable | **contract_pending** | 低（データ存在時は正常動作） | 次期ハーネス改善候補 |
| 40 | `write_hdf5_manifest` | D | HDF5コレクションマニフェスト書込 | success=False | ハーネスが `"gwexpy_manifest"` 属性を検査したが実際は `"gwexpy_layout"` 等 | ハーネス属性名誤認 | incomparable | **contract_pending** | 低（マニフェストは正常書込） | 次期ハーネス改善候補 |
| 41 | `normalize_layout` | D | レイアウト文字列正規化 | success=False | `"dataset"` に対し `"dataset-per-entry"` が返るが `"dataset"` を期待 | ハーネス期待値誤認（標準レイアウト名の不一致） | incomparable | **contract_pending** | 低（正規化は正常） | 次期ハーネス改善候補 |
| 42 | `parse_timezone` | D | タイムゾーン文字列パース | success=False | `ZoneInfo('UTC') == timezone.utc` がPython標準ライブラリでFalse | ハーネスアサーション誤り（型相違によるFalse判定） | incomparable | **contract_pending** | 低（UTCオフセット0で正常） | 次期ハーネス改善候補 |
| 43 | `local_hurst` | D | 局所Hurst指数計算 | valid_range=False | 窓長2点指定のため下流 `compute_Hc` が `len >= 100` エラー送出 | ハーネス引数誤認（$W \ge 100$ 必須、出力長はスライド数） | incomparable | **contract_pending** | 中（短窓長時のガード・例外明示化） | 今後起票候補（親Issue #765 配下） |
| 44 | `read_timeseriesmatrix_audio` | D | 音声WAVから行列読込 | shape_match=False | 行列は `(1, 1, 3)` で時間軸は最終軸。ハーネスが `shape[0] == 3` を検査 | ハーネス形状検査誤認（時間軸は `shape[-1]`） | incomparable | **contract_pending** | 低（読込・形状は正常） | 次期ハーネス改善候補 |

---

## 5. 次期アクション・追跡Issueの提言（実在コミットと今後の起票候補の分離）

今回の調査により、残余21件の性質が明確に分離されました。製品コード変更0行の規律を維持しつつ、以下の追跡を推奨します。

1. **実在するコミットでの是正（23件）**:
   - コミット [`1fcc72d25f750f6eab69d62778034699cffbf2ff`](https://github.com/tatsuki-washimi/gwexpy/commit/1fcc72d25f750f6eab69d62778034699cffbf2ff) にて、`shift` (5件), `value_at` (6件), `is_contiguous` (6件), `crop` (3件), 2D `diff` (3件) のハーネス偽陽性を解消し、`verified` (269件) へ移行済み。
2. **製品不具合・改善の今後起票候補（3件、親Issue #765 配下）**:
   - **候補1（`normalize_series` 辞書真偽値曖昧性およびゼロ値フォールバック）**:
     `gwexpy/gui/plotting/normalize.py` L27 `data = val.get("data") or val.get("value")` において：
     - 複数要素配列 `np.array([1., 2., 3.])` で `ValueError: The truth value of an array with more than one element is ambiguous` が発生。
     - 単一要素ゼロ配列 `np.array([0.])` では falsy と判定されて誤って `value` 側へフォールバックする。
     - 対策案: `val.get("data") if val.get("data") is not None else val.get("value")` へ修正。
     - ※なお、本監査で観測された `differ`（素の `np.array` を渡したハーネス側の誤り）と、上記製品コードの潜在欠陥は別個の再現テストとして記録・追跡する。
   - **候補2（`local_hurst` バックエンド適合バリデーション）**:
     `gwexpy/timeseries/hurst.py` の `method="auto"` は R/S (`compute_Hc`)、`hurst-exponent`、`exp-hurst` を選択する。
     `window < 100` 制約は R/S バックエンド固有の制約であるため、一律拒否ではなくバックエンドに応じた適切な窓長バリデーション（または明確な `ValueError` / NaN フォールバック仕様）を策定する。
   - **候補3（`Array2D.step` / `Plane2D.step` 非推奨明示）**:
     画像系列における `step()` 呼出しに対し、`NotImplementedError` または適切な非推奨警告を送出する（GWpy互換ポリシーに基づく検討）。
   - **候補4（`TimeSeries.q_gram` 短データ時の境界保護）**:
     短データ（1024点）における上流由来の `IndexError` に対し、入力長事前検証と明確な診断メッセージの提供を検討する。
3. **監査ハーネスの次期改善候補（残余17件）**:
   - `Plane2D.append`、`get_model`、HDF5コレクションヘルパー、音声行列読込のフィクスチャ・期待値をAPI契約に即して補正し、次期PRで `verified` へ移行する。
