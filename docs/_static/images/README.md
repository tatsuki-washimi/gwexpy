# GWexpy ドキュメント用プロット画像

このディレクトリには、ドキュメントトップページで使用するプロット画像を配置します。

## 必要な画像

### 1. hero_plot.png (イントロ用メイン画像)

**仕様:**
- サイズ: 幅800px、高さ400-600px
- フォーマット: PNG
- DPI: 150
- 内容: GWexpyの可視化能力を示す代表的なプロット

**生成方法:**

`intro_timeseries.ipynb` または `intro_plotting.ipynb` を実行し、最も魅力的なプロットを選択:

```python
# Jupyter Notebook内で実行
import matplotlib.pyplot as plt
from gwexpy.timeseries import TimeSeries

# サンプルデータを読み込み（または生成）
# ts = TimeSeries.read('example.gwf', 'H1:STRAIN')
# または
ts = TimeSeries(...) # ノートブックから適切なサンプルを選択

# プロット生成
fig = ts.plot()
fig.set_size_inches(10, 6)  # アスペクト比調整

# 保存
fig.savefig(
    'docs/_static/images/hero_plot.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)
plt.close(fig)
```

**推奨ソース:**
- `docs/web/ja/user_guide/tutorials/intro_timeseries.ipynb` のセル出力
- `docs/web/ja/user_guide/tutorials/intro_plotting.ipynb` のセル出力

## Stage 2 用画像（将来追加予定）

### 2. ケーススタディのサムネイル

- `case_noise_budget_thumb.png` (400x300px)
- `case_transfer_function_thumb.png` (400x300px)
- `case_active_damping_thumb.png` (400x300px)

**ソース:**
- `docs/web/ja/user_guide/tutorials/case_noise_budget.ipynb`
- `docs/web/ja/user_guide/tutorials/case_transfer_function.ipynb`
- `docs/web/ja/user_guide/tutorials/case_active_damping.ipynb`

## 注意事項

- 画像ファイルは Git にコミットしてください
- PNG 形式を使用し、適切に最適化してください
- ファイルサイズは 500KB 以下を推奨
- 背景は白 (`facecolor='white'`) を使用
