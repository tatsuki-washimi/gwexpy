# 実装計画：ScalarField 描画機能の実践とドキュメント拡充

**作成日**: 2026-01-21T13:22 JST  
**担当**: Antigravity (Claude Opus 4.5 連携)

## 1. 目的

`ScalarField` クラスの強力な描画・解析機能をユーザーが直感的に利用できるよう、チュートリアルを整理・拡充し、同時に数値的正当性を保証する不変量チェックを導入する。

## 2. 目標

1. **ノートブックの分割**: `intro_ScalarField.ipynb` (コア) と `plot_ScalarField.ipynb` (描画) に分離し、保守性を向上。
2. **再現性の確保**: 乱数 seed・単位・軸間隔を固定した標準データ生成セルの規格化。
3. **数値的検証の強化**: FFT不変量チェックや波長計算の spot check セルを導入。
4. **CLI用静止画例の作成**: `examples/` に最小限のプロットスクリプトを追加。

## 3. 実装詳細

### A. チュートリアル構成の刷新

#### 1. `intro_ScalarField.ipynb` (Core Specifications)

* 基本構成:
  * 初期化、メタデータ（Domain/AxisNames）
  * **4D維持スライス規約**の説明
  * 時間FFT/IFFT（GWpy互換性の実証）
  * 空間FFT/IFFT（両側、角波数の規約）
  * `ScalarFieldList` / `ScalarFieldDict` の基本操作
* **数値的不変量チェック（追加）**:
  * `ifft_time(fft_time(f)) ≈ f` の検証
  * `ifft_space(fft_space(f)) ≈ f` の検証
  * `k=0` における `wavelength = inf` の確認

#### 2. `plot_ScalarField.ipynb` (Visualization Practice)

* 可視化ユースケース:
  * 2D断面マップ（`plot_map2d`）
  * 多点時系列抽出と重ね書き（`plot_timeseries_points`）
  * 1D分布プロファイル（`plot_profile`）
  * 応用解析（`diff`, `zscore`, `time_stat_map`, `time_space_map`）

### B. 再現性用標準データ生成セルの導入

```python
import numpy as np
from astropy import units as u
from gwexpy.types import ScalarField

# 共通設定
rng = np.random.default_rng(seed=42)
t = np.linspace(0, 1, 100) * u.s
x = np.linspace(-1, 1, 50) * u.m
y = np.linspace(-1, 1, 50) * u.m
z = np.array([0]) * u.m  # 2D case
# ... ガウス波や平面波など、期待値が明瞭なデータの生成
```

### C. 静止画スクリプト (`examples/plot_field4d_demo.py`)

* ヘッドレス環境でも動作する `matplotlib.use('Agg')` を用いた最短プロット例。

## 4. 検証計画

* [x] **T5**: 分割された各ノートブックを `jupyter nbconvert --execute` で一括実行し、全セルがエラーなく終了すること。✅ PASSED

* [x] **T6**: FFT不変量チェックセルの Assertion が全てパスすること。✅ PASSED
* [x] **T7**: 静止画スクリプトがエラーなく `.png` を出力すること。✅ PASSED

---

## 7. 実装完了 (2026-01-21T13:30 JST)

### 成果物

| ファイル | 説明 | 状態 |
| :--- | :--- | :--- |
| `examples/tutorials/intro_ScalarField.ipynb` | コア機能 + 数値的不変量チェック | ✅ 更新済み |
| `examples/tutorials/plot_ScalarField.ipynb` | 描画機能の実践チュートリアル | ✅ 新規作成 |
| `examples/plot_field4d_demo.py` | ヘッドレス静止画スクリプト | ✅ 新規作成 |
| `field4d_demo.png` | 上記スクリプトの出力サンプル | ✅ 生成確認 |

### 所要時間

* **計画時間**: 約40分
* **実績時間**: 約8分
* **効率化要因**: 詳細な実装計画の存在とAIネイティブなコード生成

## 5. 実行戦略 (Using Skills)

* **Model**: **Claude 4 Opus**（複雑なプロットと物理検証の両立）

* **Skills**: `make_notebook`, `check_physics`, `estimate_effort`, `test_notebooks`
* **Total Time**: **~40 mins** (Estimated by `estimate_effort`)
* **Quota**: **High** (Large notebook generation & cell outputs)

## 6. 注意事項

* **再現性重視**: 全てのランダムプロセスに seed を明示。

* **依存性**: 描画は `matplotlib.pyplot` を基本とし、外部依存（seaborn等）は避ける。
* **アウトライン**: 各ノートブックには「何を示すセルか」を明記した Markdown ヘッダーを付与する。
