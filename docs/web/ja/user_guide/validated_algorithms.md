# 検証済みアルゴリズム

以下のアルゴリズムは、12種AIによるクロス検証（2026-02-01）で妥当性が確認されました。
このページでは、検証済み実装とその参考文献を記載しています。

## k-space計算

**関数**: {meth}`gwexpy.fields.ScalarField.fft_space`

**合意度**: 10/12のAIモデルが正しさを確認

角波数の計算は物理学の標準定義に従っています：

$$
k = 2\pi \cdot \text{fftfreq}(n, d)
$$

これは $k = 2\pi / \lambda$ を満たし、以下と一致しています：

- Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2
- NumPy `fftfreq` ドキュメント
- GWpy FrequencySeries (Duncan Macleod et al., SoftwareX 13, 2021)

`2π` 係数は正しく適用され、単位は `1/dx_unit` (rad/length) として
適切に設定されています。


## 振幅スペクトル（トランジェントFFT）

**関数**: `TimeSeries._fft_transient`

**合意度**: 誤った批判への明確な反駁とともに検証済み

トランジェントFFTは密度スペクトルではなく、**振幅スペクトル** を返します：

$$
\text{amplitude} = \text{rfft}(x) / N
$$

DCとナイキスト周波数を除く片側成分は2倍されます。

この規約により、正弦波のピーク振幅を直接読み取ることができます。
`dt` を掛けるべきという提案は密度スペクトル (V/√Hz) に適用されるもので、
用途が異なります。

**参考文献**:

- Oppenheim & Schafer, *Discrete-Time Signal Processing* (3rd ed., 2010), §8.6.2
- SciPy `rfft` ドキュメント


## VIF（分散膨張率）

**関数**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

**合意度**: 8/12のAIモデルが正しさを確認

VIF計算式はPercival & Walden (1993) に準拠しています：

$$
\text{VIF} = \sqrt{1 + 2 \sum_{k=1}^{M-1} \left(1 - \frac{k}{M}\right) |\rho(kS)|^2}
$$

**重要**: これは多重共線性診断に使用される回帰VIF (1/(1-R²)) とは異なります。
名称の衝突が混乱を招きましたが、スペクトル解析においては実装は正しいです。

**参考文献**:

- Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications*
  (1993), Ch. 7.3.2, Eq.(56)
- Bendat, J.S. & Piersol, A.G., *Random Data* (4th ed., 2010)


## 予測タイムスタンプ（ARIMA）

**関数**: {meth}`gwexpy.timeseries.arima.ArimaResult.forecast`

**合意度**: 誤った懸念への反駁とともに検証済み

予測開始時刻は以下のように計算されます：

$$
t_{\text{forecast}} = t_0 + n_{\text{obs}} \times \Delta t
$$

これは等間隔・ギャップなしのデータを前提としています。
GPS時刻はTAI連続秒数を使用するLIGO/GWpy規約に従っています。

一部モデルが提起したうるう秒の懸念は、重力波データ解析で使用される
GPS/TAI時刻系には該当しません。

**参考文献**:

- GWpy TimeSeries.epoch ドキュメント
- LIGO GPS時刻規約 (LIGO-T980044)
- Box, G.E.P. & Jenkins, G.M., *Time Series Analysis* (1976), Ch. 4


# 前提条件と規約

以下のセクションでは、これらのアルゴリズムを使用する際に
ユーザーが認識すべき重要な前提条件と規約を説明します。


## MCMC固定共分散前提

**関数**: {meth}`gwexpy.fitting.FitResult.run_mcmc`

**前提**: 共分散行列 Σ は **パラメータ非依存（固定）** であること。

この前提の下では、対数行列式項 `log|Σ|` は定数であり、
対数尤度から正しく省略できます：

$$
\log p(y|\theta) = -\frac{1}{2} r^T \Sigma^{-1} r + \text{const}
$$

Σ がモデルパラメータ θ に依存する場合は、`log|Σ|` を含む
完全な対数尤度を使用する必要があり、実装の修正が必要です。

**複素データ**: 複素数値データの場合、エルミート形式
`r.conj() @ cov_inv @ r` は **円形複素ガウス** 分布
（実部と虚部が等分散で無相関）を仮定しています。

**参考文献**:

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006), Ch. 2.2
- Gelman et al., *Bayesian Data Analysis* (3rd ed., 2013), §14.2


## 角波数 vs サイクル波数

**関数**: {meth}`gwexpy.fields.ScalarField.fft_space`

**規約**: `k` 軸は **角波数** [rad/length] を返します。
サイクル波数 [1/length] ではありません。

- 角波数: $k = 2\pi / \lambda$
- サイクル波数: $\nu = 1 / \lambda$
- 変換: $\nu = k / (2\pi)$


## 降順軸の符号規約

**関数**: {meth}`gwexpy.fields.ScalarField.fft_space`

**規約**: 空間軸が降順 (dx < 0) の場合、k軸は位相因子規約
$e^{+ikx}$ との物理的整合性を保つために **符号反転** されます。

これにより、データ格納順序に関係なく、正の `k` が
正のx方向に伝播する波に対応することが保証されます。

**参考文献**: Jackson, *Classical Electrodynamics* (3rd ed., 1998), §4.2


## ブートストラップにおけるVIF適用

**関数**: {func}`gwexpy.spectral.estimation.bootstrap_spectrogram`

**条件付き動作**:

- **標準ブートストラップ** (`block_size=None`): Welchオーバーラップ相関を
  補正するためにVIF補正が適用されます。
- **ブロックブートストラップ** (`block_size` 指定): 二重補正を防ぐため
  VIF補正は **無効化** (factor=1.0) されます。

これにより、ブロック構造と解析的VIFの両方が同時に適用された場合の
分散推定の過大評価を防止します。


## ブートストラップの定常性前提

**関数**: {func}`gwexpy.spectral.estimation.bootstrap_spectrogram`

**前提**: 入力データは **定常過程** であること。

Moving Block Bootstrap は、データの統計的性質が時間とともに
変化しないことを仮定しています。非定常データ（グリッチ、トランジェント、
ドリフトするノイズフロア）では、信頼区間が偏る可能性があります。

**推奨** `block_size`: 時間相関のあるスペクトログラムでは、
少なくともストライド長（セグメント間隔）以上を設定してください。
floatまたはQuantityで秒単位で指定します。

**参考文献**:

- Künsch, H.R., "The jackknife and the bootstrap for general stationary
  observations", Ann. Statist. 17(3), 1989
- Politis, D.N. & Romano, J.P., "The stationary bootstrap",
  J. Amer. Statist. Assoc. 89(428), 1994


## 検証について

このページに記載されているすべてのアルゴリズムは、複数の独立した
レビュー手法によるクロス検証を通じて厳密な検証を受けています。
実装は確立された参考文献と照らし合わせて検証され、正確性が
確認されています。

詳細な検証レポートと技術レビューのドキュメントについては、
開発者ドキュメントを参照してください。
