# Plot

**安定性:** 安定

**継承元:** [`gwpy.plot.Plot`](https://gwpy.readthedocs.io/en/latest/reference/gwpy.plot.Plot/)

## 主な用途

`Plot` は GWexpy における可視化の入口です。単一 series だけでなく、
`SeriesMatrix` や `SpectrogramMatrix` のような多チャンネル入力も扱えます。
GWpy 互換の Figure 操作を保ちながら、行列入力の自動展開、サブプロット配置、
高密度オーバーレイの間引き、spectrogram 系の colorbar 配置を補助します。

## 代表的なシグネチャ

```python
Plot(*args, separate=None, geometry=None, monitor=None, decimate_threshold=50000, decimate_points=10000, **kwargs)
plot_mmm(median, min_s, max_s, ax=None, **kwargs)
```

## 最小例

```python
from gwexpy.plot import Plot

fig = Plot(ts_matrix, separate=True, figsize=(10, 6))
_ = fig.plot_mmm(median_series, min_series, max_series, alpha_fill=0.15)
```

## GWexpy 固有の挙動

- `SeriesMatrix` と `SpectrogramMatrix` を自動的にサブプロットへ展開します。
- list / dict 入力のラベルを legend と軸へ引き継ぎます。
- spectrogram 系入力では colorbar の配置を自動化できます。
- `decimate_threshold` を超える高密度オーバーレイは自動で間引きされます。

## 関連理論

- [FFT_Conventions](FFT_Conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API リファレンス

継承メソッドを含む完全な API は以下の生成済みリファレンスを参照してください。

```{eval-rst}
.. currentmodule:: gwexpy.plot

.. autoclass:: Plot
   :members: __init__, plot_mmm, show
   :show-inheritance:
```
