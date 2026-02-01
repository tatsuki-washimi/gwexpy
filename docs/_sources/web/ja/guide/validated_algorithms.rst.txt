検証済みアルゴリズム
====================

以下のアルゴリズムは、12種AIによるクロス検証（2026-02-01）で妥当性が確認されました。
このページでは、検証済み実装とその参考文献を記載しています。

k-space計算
-----------

**関数**: :meth:`gwexpy.fields.ScalarField.fft_space`

**合意度**: 10/12のAIモデルが正しさを確認

角波数の計算は物理学の標準定義に従っています：

.. math::

    k = 2\pi \cdot \text{fftfreq}(n, d)

これは :math:`k = 2\pi / \lambda` を満たし、以下と一致しています：

- Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2
- NumPy ``fftfreq`` ドキュメント
- GWpy FrequencySeries (Duncan Macleod et al., SoftwareX 13, 2021)

``2π`` 係数は正しく適用され、単位は ``1/dx_unit`` (rad/length) として
適切に設定されています。


振幅スペクトル（トランジェントFFT）
-----------------------------------

**関数**: ``TimeSeries._fft_transient``

**合意度**: 誤った批判への明確な反駁とともに検証済み

トランジェントFFTは密度スペクトルではなく、**振幅スペクトル** を返します：

.. math::

    \text{amplitude} = \text{rfft}(x) / N

DCとナイキスト周波数を除く片側成分は2倍されます。

この規約により、正弦波のピーク振幅を直接読み取ることができます。
``dt`` を掛けるべきという提案は密度スペクトル (V/√Hz) に適用されるもので、
用途が異なります。

**参考文献**:

- Oppenheim & Schafer, *Discrete-Time Signal Processing* (3rd ed., 2010), §8.6.2
- SciPy ``rfft`` ドキュメント


VIF（分散膨張率）
-----------------

**関数**: :func:`gwexpy.spectral.estimation.calculate_correlation_factor`

**合意度**: 8/12のAIモデルが正しさを確認

VIF計算式はPercival & Walden (1993) に準拠しています：

.. math::

    \text{VIF} = \sqrt{1 + 2 \sum_{k=1}^{M-1} \left(1 - \frac{k}{M}\right) |\rho(kS)|^2}

**重要**: これは多重共線性診断に使用される回帰VIF (1/(1-R²)) とは異なります。
名称の衝突が混乱を招きましたが、スペクトル解析においては実装は正しいです。

**参考文献**:

- Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications*
  (1993), Ch. 7.3.2, Eq.(56)
- Bendat, J.S. & Piersol, A.G., *Random Data* (4th ed., 2010)


予測タイムスタンプ（ARIMA）
---------------------------

**関数**: :meth:`gwexpy.timeseries.arima.ArimaResult.forecast`

**合意度**: 誤った懸念への反駁とともに検証済み

予測開始時刻は以下のように計算されます：

.. math::

    t_{\text{forecast}} = t_0 + n_{\text{obs}} \times \Delta t

これは等間隔・ギャップなしのデータを前提としています。
GPS時刻はTAI連続秒数を使用するLIGO/GWpy規約に従っています。

一部モデルが提起したうるう秒の懸念は、重力波データ解析で使用される
GPS/TAI時刻系には該当しません。

**参考文献**:

- GWpy TimeSeries.epoch ドキュメント
- LIGO GPS時刻規約 (LIGO-T980044)


検証について
------------

これらの検証は、12種のAIモデルを使用した包括的なアルゴリズム監査の
一環として実施されました：

- ChatGPT 5.2 Pro (Deep Research)
- Claude Opus 4.5 (Antigravity, IDE)
- Copilot (IDE)
- Cursor
- Felo
- Gemini 3 Pro (Antigravity, CLI, Web)
- Grok
- NotebookLM
- Perplexity

完全な検証レポートは開発者ドキュメントで参照できます。
