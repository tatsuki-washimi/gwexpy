# Debug Plot Axes

プロットの軸スケール（対数軸等）、目盛、表示範囲の不具合を診断・修正。

## Diagnostic Procedures

### 1. Verify Internal State

再現スクリプトを作成し、軸の内部状態を確認：

```python
print(ax.get_yscale())  # 'log' or 'linear'
print(ax.get_ylim())
print(ax.get_yticks())
```

### 2. Identify Visual Linearity Issues

内部状態が 'log' なのに視覚的にリニアに見える場合：
- データ範囲が狭すぎる（例：1.0 vs 2.23）
- 目盛が適切に配置されていない

### 3. Confirm Judgment Logic

`gwexpy/plot/defaults.py` 内の `determine_yscale` / `determine_xscale` が
データを正しく認識しているか、debug print で確認。

## Correction Guidelines

### 1. Enforce Scale Application

`Plot.__init__` 内の `super().__init__` の後に明示的に `ax.set_yscale()` を呼び出し。
必ず `ax.autoscale_view()` を呼んで表示を更新。

### 2. Automatic Range Expansion (Log Scale)

データ範囲が 100x（2桁）未満の場合、対数目盛がほとんど表示されずリニアに見える。
`determine_ylim` ロジックで中央値を基準に約2桁の範囲を確保。

### 3. Robust Type Checking

環境差による `isinstance` チェックの失敗を防ぐため、
`type(obj).__name__` や `hasattr(obj, 'frequencies')` を併用。

### 4. Prevent Duplicate Display (IPython/Jupyter)

`Plot` クラスに `_repr_html_ = None` を設定し、
repr と `plt.show` の二重表示を防止。
