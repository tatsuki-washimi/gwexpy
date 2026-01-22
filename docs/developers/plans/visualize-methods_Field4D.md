# ScalarField 可視化（1D/2D）強化ロードマップ

## 前提（既存仕様の尊重）

* ScalarField は **4D維持**（整数indexも slice(i,i+1) に変換し axis length=1 を保つ）
* 軸 index は `Quantity` 配列が正
* 空間FFTは符号付き両側、波数は角波数 `k = 2π/λ`
* 時間FFTは GWpy `TimeSeries.fft()` 準拠（既実装）
* 可視化は **matplotlib** を使用（依存追加は避ける）

---

## フェーズ0（最優先）：共通インフラの追加

### 0.1 座標→インデックス変換ユーティリティ（必須）

**新規モジュール案**：`gwexpy/plot/_coord.py` または `gwexpy/types/_coord.py`

#### 実装する関数（新規）

1. `nearest_index(axis_index, value) -> int`

* `axis_index`：`Quantity` 1D 配列
* `value`：`Quantity` スカラー
* unit 不一致は `ValueError`
* 範囲外は `IndexError`（もしくは `ValueError`。どちらかに統一）
* tie-break は「小さい側を優先」（固定）

1. `slice_from_index(i: int) -> slice`

* `slice(i, i+1)` を返す

1. `slice_from_value(axis_index, value, method="nearest") -> slice`

* MVP は `method="nearest"` のみ対応でよい
* `nearest_index` を呼び `slice_from_index` へ

### Phase 0.1 DoD

* unit不一致、範囲外、tie-break がテストで固定される

---

### 0.2 値の取り出し規約（複素対策・重複防止）

**新規関数**：`select_value(data, mode="real|imag|abs|angle|power")`

* `power` は `abs(data)**2`（単位は unit^2 になることに注意）
* デフォルト：`real`（または `abs`。方針を固定して実装。推奨：time-domainは real、frequency/k-domain は abs 既定でもよいが、最初は一律 real でよい）

### Phase 0.2 DoD

* 複素入力で mode ごとに数値が一致するテスト
* `power` の unit の扱いが破綻しない（単位付きQuantityなら unit^2）

---

### 0.3 抽出API（解析メソッド；plotの基盤）

**新規**：`ScalarField.extract_points(...) -> TimeSeriesList`

* `points`: list[tuple[x,y,z]]（Quantity推奨）
* `interp="nearest"`（MVPはnearestのみ）
* 戻り値：`TimeSeriesList`（各要素が1点抽出の時系列）
* ラベル（metadata）に座標情報を保存（`(x=...,y=...,z=...)`）

**新規**：`ScalarField.extract_profile(...) -> (Quantity_axis, Quantity_values)` または `Series`

* `axis`: "x|y|z"
* `at`: dict で他軸を固定（例：`{"t":..., "y":..., "z":...}`）
* `reduce`: None または簡易平均（次フェーズでも可）
* MVPは `reduce=None`（一点断面）のみでよい

**新規**：`ScalarField.slice_map2d(plane="xy|xz|yz", at={...}) -> ScalarField`

* planeの2軸は残し、他2軸は length=1 のスライスにする
* 返り値は ScalarField（axis length=1規約を維持）

### Phase 0.3 DoD

* `extract_points` の返す各時系列の長さが t 軸と一致
* `slice_map2d` 後に plane 以外の2軸が length=1
* 単位・メタデータが落ちない

---

## フェーズ1：最小描画（研究即戦力）

### 1.1 2Dヒートマップ：`ScalarField.plot_map2d(...)`

**新規**：`gwexpy/plot/field4d.py` を作り、実処理を集約
`ScalarField.plot_map2d` は薄いラッパ（内部で plot モジュール関数を呼ぶ）

#### 推奨シグネチャ（MVP）

`plot_map2d(self, *, plane="xy", at=None, mode="real", method="pcolormesh", ax=None, add_colorbar=True, vmin=None, vmax=None, title=None) -> (fig, ax)`

* `at` が None の場合は、axis length=1 の軸を自動採用（曖昧なら例外）
* pcolormesh 既定（不等間隔に強い）
* 軸ラベルに unit を付与（例：`x [m]`）

### Phase 1.1 DoD

* `slice_map2d(...).plot_map2d(...)` が動作する
* 軸ラベルに unit が含まれる
* `add_colorbar=True` で colorbar が出る

---

### 1.2 1D重ね書き：`ScalarField.plot_timeseries_points(...)`

* 内部は `extract_points` を呼ぶのみ
* `labels` がなければ自動生成

### Phase 1.2 DoD

* 複数点で line が複数本描画される
* ラベルが自動で入る（legendはオプションでも可）

---

### 1.3 1D線プロファイル：`ScalarField.plot_profile(...)`

* 内部は `extract_profile` を呼ぶのみ

### Phase 1.3 DoD

* 指定 axis に沿った1Dプロファイルが描画される

---

## フェーズ2：比較・要約（高ROI）

### 2.1 フィールド差分／比：解析メソッド

**新規**：`ScalarField.diff(other, mode="diff|ratio|percent") -> ScalarField`

* modeの意味を固定
* unit の扱い：

  * diff：同unit
  * ratio/percent：dimensionless

**新規**：`ScalarField.zscore(baseline_t=(t1,t2), ...) -> ScalarField`

* baseline の抽出は axis0_domain="time" の時のみ
* baseline 期間の平均/標準偏差で zscore

**描画**：差分は `plot_map2d` に流す（`plot_map2d_diff` は不要）

### Phase 2.1 DoD

* diff/ratio の unit が正しい
* zscore の baseline 指定が範囲外で例外

---

### 2.2 時間要約マップ

**新規**：`ScalarField.time_stat_map(stat="mean|std|rms|max", t_range=(t1,t2), plane="xy", at={...}) -> ScalarField`

* 結果は t 軸を length=1 にして返す（時間要約済み）
* 表示は `plot_map2d` に流用

### Phase 2.2 DoD

* t_range 指定で正しい期間が使われる
* statごとに期待値と一致する小テスト

---

### 2.3 time–space map（ムービー代替）

**新規**：`ScalarField.time_space_map(axis="x|y|z", at={...}, mode="real", reduce=None) -> Array2D相当 or ScalarField`

* MVP：reduce=None（一点断面）
* 出力は (t, x) の2Dデータ（最小は numpy array + (t,x)のQuantity）
* 描画は `plot_map2d` 相当関数に流せる形にする

### Phase 2.3 DoD

* (t, x) のshapeが期待通り
* t/x の軸ラベルが描画で正しい

---

## フェーズ3：重い信号処理（最後）

### 3.1 xcorr / delay / coherence（解析→map→plot）

* `compute_xcorr(point_a, point_b, max_lag, ...) -> lag-series`
* `time_delay_map(ref_point, plane, at, ...) -> ScalarField`
* `coherence_map(ref_point, band=(f1,f2), ...) -> ScalarField`
* 表示は `plot_map2d` に統一

**注意**：このフェーズは計算コストが大きいので `stride` や `roi` を必須オプションにすること。

---

## テスト計画（pytest；最低限）

## T0：ユーティリティ

* `nearest_index` の unit不一致例外
* 範囲外例外
* tie-breakの固定

## T1：抽出

* `extract_points` が TimeSeriesList を返す（長さ一致）
* `slice_map2d` で2軸が残り2軸は length=1

## T2：plot（画像比較は不要）

* `plot_map2d` が `Axes` を返し、colorbarが追加される（`len(fig.axes)` 等で確認）
* `plot_timeseries_points` が line を複数生成する（`len(ax.lines)`）
* `plot_profile` が line を1本以上生成

## T3：比較

* diff/ratio の unit
* zscore の baseline範囲外例外

## T4：time_stat_map / time_space_map

* 小さな人工データで平均などが一致

---

## 実装メモ（重要）

* 描画関数は **必ず** `ax` を受け、与えられなければ `plt.subplots()` で作る
* `matplotlib` の import は plot モジュール内に閉じる（types 層に依存を持ち込まない）
* 2D描画は `pcolormesh` 既定（不等間隔軸でも正しい）
* 値の取り出し（real/abs等）は共通関数 `select_value` を必ず通す（重複排除）

---

## 受け入れ条件（DoDまとめ）

* フェーズ1完了時点で：

  * 断面2Dマップが1行で描ける（`slice_map2d(...).plot_map2d()`）
  * 任意点群の時系列重ね書きが1行で描ける
  * 任意断面の線プロファイルが描ける
* フェーズ2完了で：

  * diff/zscore/time要約/time–space が解析→描画に乗る
* テストが全て通る（画像比較なしで構造検証）

---

## 実装計画レビュー (2026-01-21T12:58 JST)

## モデル選定

* **採用モデル**: Claude Opus 4.5
* **理由**: 高精度なロジック実装と物理数学的な厳密さが求められるため、コード生成能力と推論能力のバランスに優れた Claude 系列を採用。特に複雑なスライス操作や単位計算のバグを最小限に抑える。

## 推奨スキル

1. **`extend_gwpy`**: `ScalarField` の GWpy/Astropy 継承におけるスライス操作での単位消失防止ガイド
2. **`check_physics`**: 空間FFTの波数（k）や Nyquist 周波数の扱い、物理単位の整合性検証
3. **`test_code`**: T0〜T4 テストの自動実行によるリグレッション防止
4. **`make_notebook`**: 可視化機能のチュートリアルノートブック作成
5. **`debug_axes`**: 軸スケール・目盛・単位ラベル表示のデバッグ

## 工数見積もり

| フェーズ | 難易度 | 推定時間 | クオータ消費 | 主な作業 |
| :--- | :--- | :--- | :--- | :--- |
| Phase 0: インフラ | 高 | 90分 | High | 座標変換ユーティリティ、抽出APIの実装、Unitテスト |
| Phase 1: 基本描画 | 中 | 60分 | Medium | Map2D, 1D Profile の Matplotlib 実装 |
| Phase 2: 比較・要約 | 中〜高 | 60分 | Medium | Diff/Ratio, Z-score, Time-Space Map の計算ロジック |
| **合計** | - | **約 3.5〜4 時間** | **High** | (Phase 0-2 を含む) |

### 懸念事項

* 4D データのスライス規約（length=1 を維持）が既存の GWpy メソッドと競合しないかの検証
* 複素数データの `power` モードにおける単位（unit^2）の扱い
* 大規模データに対する Matplotlib の描画パフォーマンス

---

## 実装進捗 (2026-01-21T13:15 JST)

### ✅ Phase 0: 共通インフラの追加（完了）

| 項目 | ファイル | 状態 |
| :--- | :--- | :--- |
| `nearest_index`, `slice_from_index`, `slice_from_value` | `gwexpy/plot/_coord.py` | ✅ 実装済み |
| `select_value` (real/imag/abs/angle/power) | `gwexpy/plot/_coord.py` | ✅ 実装済み |
| T0 テスト | `tests/plot/test_coord.py` | ✅ 21テスト通過 |

### ✅ Phase 1: 最小描画（完了）

| 項目 | メソッド | 状態 |
| :--- | :--- | :--- |
| 抽出API | `ScalarField.extract_points()` | ✅ 実装済み |
| 抽出API | `ScalarField.extract_profile()` | ✅ 実装済み |
| 抽出API | `ScalarField.slice_map2d()` | ✅ 実装済み |
| 2Dヒートマップ | `ScalarField.plot_map2d()` | ✅ 実装済み |
| 1D重ね書き | `ScalarField.plot_timeseries_points()` | ✅ 実装済み |
| 1D線プロファイル | `ScalarField.plot_profile()` | ✅ 実装済み |
| T1-T2 テスト | `tests/types/test_field4d_visualization.py` | ✅ 17テスト通過 |

### ✅ Phase 2: 比較・要約（完了）

| 項目 | メソッド | 状態 |
| :--- | :--- | :--- |
| 差分/比率 | `ScalarField.diff()` | ✅ 実装済み |
| Z-score | `ScalarField.zscore()` | ✅ 実装済み |
| 時間要約マップ | `ScalarField.time_stat_map()` | ✅ 実装済み |
| 時間-空間マップ | `ScalarField.time_space_map()` | ✅ 実装済み |
| 時間-空間描画 | `ScalarField.plot_time_space_map()` | ✅ 実装済み |
| T3-T4 テスト | `tests/types/test_field4d_visualization.py` | ✅ 11テスト通過 |

### ⏳ Phase 3: 重い信号処理（未着手）

* `compute_xcorr`, `time_delay_map`, `coherence_map` は今後の実装予定
