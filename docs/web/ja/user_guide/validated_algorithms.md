# 検証済みアルゴリズム

`gwexpy` で使用されている主要な数値数値アルゴリズムは、科学的な正確性を保証するために厳密な検証プロセスを経て実装されています。本ページでは、検証済みのアルゴリズム一覧とその根拠（参考文献・検証手法）をまとめています。

## 検証の基準 (Validation Criteria)

「検証済み」と表記されるアルゴリズムは、以下のいずれか一つ以上のプロセスを満たしています。

1.  **文献照合 (Literature Check)**: 確立された教科書や査読済み論文の数式と 1 対 1 で整合していること。
2.  **相互比較 (Cross-Verification)**: GWpy, SciPy, LALSuite 等の標準的なライブラリと同一条件下で計算結果が一致すること。
3.  **数値実験 (Numerical Experiment)**: 既知の解析解を持つテストデータに対し、理論的な誤差範囲内で収束すること。
4.  **独立レビュー (Peer Review)**: コア開発者以外の専門家によるコードおよび物理ロジックの監査をパスしていること。

---

## 検証済みアルゴリズム一覧

| アルゴリズム | 対応 API | 検証方法 | 主な用途 | 関連チュートリアル |
| :--- | :--- | :--- | :--- | :--- |
| **k-space 計算** | `.fft_space()` | 文献・SciPy 比較 | 空間相関、波数解析 | [Field 入門](tutorials/field_scalar_intro.ipynb) |
| **Transient FFT** | `._fft_transient()` | 文献・NumPy 比較 | 短時間バースト解析 | — |
| **VIF 補正** | `calculate_correlation_factor()` | 文献 (Percival) | スペクトル誤差推定 | [Bootstrap チュートリアル](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **予測時刻計算** | `ArimaResult.forecast()` | LIGO 時刻規約 | 故障予兆・トレンド予測 | — |
| **適応ホワイトニング** | `.whiten(eps="auto")` | 数値実験 | 極小信号の安定抽出 | [数値安定性](numerical_stability.md) |

---

## アルゴリズム詳細と物理的根拠

### 1. k-space 計算
**対象**: {meth}`gwexpy.fields.ScalarField.fft_space`

角波数の計算は、物理学の標準定義 $k = 2\pi / \lambda$ に従っています。

$$
k = 2\pi \cdot \text{fftfreq}(n, d)
$$

**根拠**:
- Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2
- NumPy `fftfreq` ドキュメントとの整合性確認済み。

---

### 2. 振幅スペクトル（トランジェントFFT）
**対象**: `TimeSeries._fft_transient`

密度スペクトルではなく、**ピーク振幅**を直接読み取れる規約を採用しています。

$$
\text{amplitude} = \text{rfft}(x) / N
$$

**根拠**:
- Oppenheim & Schafer, *Discrete-Time Signal Processing* (3rd ed., 2010), §8.6.2
- 正弦波の既知振幅入力に対する数値実験により検証済み。

---

### 3. VIF (分散膨張係数)
**対象**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

Welch 法等におけるセグメント間のオーバーラップによる有効サンプル数の減少を補正します。

$$
\text{VIF} = \sqrt{1 + 2 \sum_{k=1}^{M-1} \left(1 - \frac{k}{M}\right) |\rho(kS)|^2}
$$

**根拠**:
- Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications* (1993), Eq.(56)
- **注意**: 多重共線性診断の VIF ($1/(1-R^2)$) とは定義が異なります。

---

### 4. 予測タイムスタンプ (ARIMA)
**対象**: {meth}`gwexpy.timeseries.arima.ArimaResult.forecast`

GPS 時刻系における連続性を保証するため、LIGO 規約に従った歩進計算を行います。

**根拠**:
- LIGO GPS 時刻規約 (LIGO-T980044)
- 閏秒を含まない TAI 秒としての整合性確認済み。

---

## 前提条件と制約事項

利用にあたって以下の物理的・統計的前提条件に注意してください。

- **MCMC 固定共分散**: `run_mcmc` は共分散行列 $\Sigma$ がパラメータ非依存であることを前提としています。
- **ブートストラップの定常性**: `bootstrap_spectrogram` は入力データが定常過程であることを仮定しており、非定常なグリッチを含むデータでは誤差推計が偏る可能性があります。
- **角波数規約**: `fft_space` は角波数 [rad/length] を返します。サイクル波数 [1/length] への変換には $2\pi$ による除算が必要です。

## 監査証跡

詳細なユニットテスト結果、文献との比較スクリプト、および過去のレビュー記録は `docs_internal/verification/` ディレクトリに保管されています。
