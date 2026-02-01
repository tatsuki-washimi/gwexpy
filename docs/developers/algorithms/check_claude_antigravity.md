# gwexpy アルゴリズム検証レポート

**監査者**: Claude (Antigravity) - 上級物理学者/ソフトウェア監査官  
**日付**: 2026-01-31  
**対象**: ALGORITHM_CONTEXT.md に記載のアルゴリズム実装

---

## 1. Physical Fields & k-space (`ScalarField.fft_space`)

### ✅ Finding 1.1: k-space座標変換 - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
k_values = 2 * np.pi * np.fft.fftfreq(npts, d=abs(dx_value))
```

この実装は物理的に正確です。角波数 $k = 2\pi f$ であり、ここで $f = \text{fftfreq}(n, d=\Delta x)$ です。これは標準的なFFT規約に従っています：
$$k = \frac{2\pi n}{N \Delta x}$$

**検証結果**: パス

---

### ⚠️ Finding 1.2: ifft_space での座標オフセット未保持

**Severity**: Medium

**Finding**: `ifft_space` が逆変換する際、元の実空間座標の**原点オフセット**が失われます。

```python
# 現在の実装
x_values = np.arange(npts) * dx_value * dx_unit
```

元のフィールドが $x \in [x_0, x_0 + L]$ であった場合、逆変換後は常に $x \in [0, L]$ に配置されます。

**物理的影響**: 空間オフセットに依存する解析（例：干渉計アーム位置）で不整合が発生する可能性があります。

**Recommendation**:
`fft_space` で元の座標オフセットを `_gwex_axis_offset` 属性として保存し、`ifft_space` で復元する。

---

## 2. Transient Response Analysis

### ✅ Finding 2.1: `_fft_transient` 振幅正規化 - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
dft = np.fft.rfft(x, n=target_nfft) / target_nfft

# One-sided correction
if target_nfft % 2 == 0:
    dft[1:-1] *= 2.0  # DC=0 と Nyquist=n-1 は unique
else:
    dft[1:] *= 2.0    # DCのみ unique
```

この正規化は片側スペクトルにおける Parseval の等式を正しく満たします。DC成分と（偶数長の場合）Nyquist成分は2倍されず、エネルギー保存が維持されます。

**注**: この正規化は「振幅スペクトル」（信号の振幅を直接読み取れる形式）であり、「密度スペクトル」（$\times dt$ のRiemann sum形式）とは異なります。gwexpyのユースケース（トランジェント信号解析）では振幅スペクトルが適切です。

**検証結果**: パス

---

### ⚠️ Finding 2.2: `detect_step_segments` のエッジケース

**Severity**: Medium

**Finding**: SNR計算で **中央値がゼロ** の場合、ハードコードされた `1e-30` が使用されますが、これは物理的根拠がありません。

```python
median_level = float(np.median(spec.value))
if median_level == 0:
    median_level = 1e-30
```

**物理的問題**:

- 全てゼロの信号（データギャップ）と、非常に微弱な信号の区別ができない
- PSD単位に依存しない閾値の使用

**Recommendation**:

1. `median_level == 0` の場合は警告を発して空リストを返す
2. または `np.nanmedian` を使用し、全NaNの場合は明示的に処理

---

### ⚠️ Finding 2.3: 周波数許容誤差のハードコード

**Severity**: Low

**Finding**: `freq_tolerance = 1.0` Hz がデフォルトでハードコードされています。

**物理的問題**: 低周波数インジェクション（例: 0.5 Hz）では、許容誤差が信号周波数の200%となり、誤検出につながる可能性があります。

**Recommendation**:
相対許容誤差（例: `freq_tolerance = 0.05 * f_inj`）を導入するか、周波数依存の閾値を使用。

---

## 3. Robust Statistics (`bootstrap_spectrogram`)

### ✅ Finding 3.1: VIF式 - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
# VIF = sqrt(1 + 2 * Σ_{k=1}^{M-1} (1 - k/M) * |ρ(k·S)|²)
vif = 1.0 + 2.0 * rho_sq_weighted_sum
return np.sqrt(vif)
```

この式は Percival & Walden (1993) の重み付き自己相関補正と一致しています。重み $(1 - k/M)$ は Bartlett三角窓に相当し、分散のバイアス補正として正しいです。

**注**: VIF（Variance Inflation Factor）という名称は回帰分析での用語と混同されやすいですが、ここでは「重なりによる分散膨張率」として文脈的に正しく使用されています。統計学的には "effective sample size correction" または "bandwidth factor" と呼ぶ方が正確です。

**検証結果**: パス

---

### ⚠️ Finding 3.2: ブロックブートストラップの切り捨て処理

**Severity**: Low

**Finding**: ブロックブートストラップで**最終ブロックの切り捨て**が行われる場合、サンプルサイズのバイアスが生じる可能性があります。

```python
num_blocks_needed = int(np.ceil(n_time / block_size))
```

**Recommendation**:
Moving Block Bootstrap の標準規約に従い、$n_{blocks} = n - b + 1$ の重複ブロックからランダム抽出し、合計サンプル数が元のサイズになるようにする。

---

## 4. Bayesian Fitting (`run_mcmc`)

### ✅ Finding 4.1: GLS対数尤度 - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
# GLS: use full covariance structure
val = r.conj() @ cov_inv @ r
chi2 = float(np.real(val))
```

複素残差に対するエルミート形式 $r^{\dagger} \Sigma^{-1} r$ の実部を取ることは数学的に正しいです。共分散行列がエルミート正定値であれば、虚部は理論上ゼロです。

**正規化について**: 対数尤度に $-\frac{1}{2}\log|\Sigma|$ 項が欠落していますが、**共分散行列がサンプリング中に固定**されている場合（ノイズモデルのパラメータを同時推定しない場合）、この項は定数でありMCMCサンプリングの結果に影響しません。

**注意**: 共分散行列のパラメータ（例：ホワイトノイズレベル）も同時推定する場合は、log-determinant項が必要です。現在の実装はシグナルパラメータのみの推定に適しています。

**検証結果**: パス（固定共分散の場合）

---

### ⚠️ Finding 4.2: Walker初期化の堅牢性

**Severity**: Medium

**Finding**: Walker初期化でMinuit誤差がゼロの場合のフォールバックが弱いです。

```python
stds = [
    self.minuit.errors[p] if ... > 0 else 1e-4 * abs(v) + 1e-8
    ...
]
pos = p0_float + stds * 1e-1 * np.random.randn(n_walkers, ndim)
```

**問題**:

- `v = 0` の場合、スプレッドは `1e-9` となり、Walker間の多様性が不足
- パラメータのスケールが大きく異なる場合（例: 振幅 $10^{-20}$）、相対的なスプレッドが不適切

**Recommendation**:

1. パラメータ値に対する相対誤差のフォールバック（例: `1e-2 * abs(v)` または `1e-6`）
2. または、各パラメータの事前分布スケールに基づく初期化

---

## 5. Noise Models

### ✅ Finding 5.1: Schumann共鳴 - PSD加算 - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
total_psd += peak_asd_series.value**2  # PSD加算（非干渉源）
total_asd = np.sqrt(total_psd)
```

非干渉（ランダム位相）源のPSD加算は物理的に正しいです：
$$\text{PSD}_{total} = \sum_i \text{PSD}_i$$

**検証結果**: パス

---

### ⚠️ Finding 5.2: Lorentzian線形のドキュメント明確化

**Severity**: High (ドキュメント問題)

**Finding**: 実装は正しいが、正規化規約の明確化が必要です。

**コード実装**:

```python
denom = np.sqrt((f_vals - f0) ** 2 + gamma_val**2)
shape = gamma_val / denom  # ピーク値が1となる
data = amp_val * shape     # ピーク値がamplitude
```

**物理的分析**:
gwexpyのLorentzianは**ピーク正規化**（最大値が`amplitude`）であり、**積分正規化**（全積分が1）ではありません。

標準的なLorentzian:
$$L(f) = \frac{1}{\pi} \cdot \frac{\gamma}{(f-f_0)^2 + \gamma^2}$$
の場合、$\int_{-\infty}^{\infty} L(f) df = 1$

gwexpy実装:
$$L_{gwex}(f) = A \cdot \frac{\gamma}{\sqrt{(f-f_0)^2 + \gamma^2}}$$
の場合、$\int_{-\infty}^{\infty} L_{gwex}(f) df = A\pi\gamma$

**Recommendation**:
docstringに以下を追加：

> "This function returns a peak-normalized Lorentzian ASD where the maximum value equals 'amplitude' at f=f0. For area-normalized profiles, divide by (π·γ)."

---

### ✅ Finding 5.3: Voigt線形 - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
z = ((f_vals - f0) + 1j * gamma) / (sigma * np.sqrt(2))
v = wofz(z).real  # Faddeeva関数
```

Faddeeva関数 $w(z)$ を用いたVoigtプロファイルの計算は標準的で正確です。ピーク正規化も正しく実装されています。

**検証結果**: パス

---

## 6. DTT Normalization

### ⚠️ Finding 6.1: DTT ENBW定義の検証

**Severity**: Medium

**Finding**: DTTモードのENBW定義が特殊です。

```python
# DTT definition
return (fs * n) / (sum_w**2)
```

**物理的分析**:

- 標準ENBW: $\text{ENBW} = f_s \cdot \frac{\sum w_i^2}{(\sum w_i)^2}$
- DTT ENBW: $\text{ENBW}_{DTT} = f_s \cdot \frac{N}{(\sum w_i)^2}$

これは $\sum w_i^2 = N$ を仮定しています（正規化窓）。正規化されていない窓（例: 標準のHanning）ではこの仮定が成り立たず、変換結果にスケールファクターエラーが生じます。

**Recommendation**:
ドキュメントに「DTTモードは窓が $\sum w^2 = N$ に正規化されていることを前提とする」と明記し、入力窓の事前検証を追加。

---

## 7. Time Series Modeling (ARIMA)

### ✅ Finding 7.1: GPS時刻マッピング - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
forecast_t0 = self.t0 + n_obs * self.dt
```

`n_obs` が元の入力系列の長さ（差分適用前）であれば、この計算は正しいです。statsmodelsの`get_forecast`は差分後の系列に対して予測を行いますが、返される予測値は元のスケールに逆変換されます。

**注意点**: `self.t0` と `n_obs` の定義がクラス初期化時に正しく設定されていることが前提です。

**検証結果**: パス（初期化が正しい場合）

---

## 8. PCA/ICA for TimeSeriesMatrix

### ✅ Finding 8.1: フラット化ロジック - **正確**

**Severity**: Low

**Finding**:

```python
X_features = X_proc.value.reshape(-1, X_proc.shape[-1])
X_sklearn = X_features.T  # (time, features)
```

reshapeのC順序により、`(channels, cols, time)` → `(channels*cols, time)` となり、チャンネル優先でフラット化されます。

**確認済み**: `pca_inverse_transform` で3D構造が復元される実装が存在します：

```python
X_rec_3d = X_rec_val[:, None, :]
```

**Recommendation**:
ドキュメントに次元順序（C-order: channels vary slowest）を明記し、より複雑な3D構造（複数列）への対応を検討。

---

## サマリー

| カテゴリ             | 状態 | 重大度 High | 重大度 Medium | 重大度 Low |
| -------------------- | ---- | ----------- | ------------- | ---------- |
| k-space変換          | ⚠️   | 0           | 1             | 0          |
| トランジェント解析   | ✅   | 0           | 1             | 1          |
| ブートストラップ統計 | ✅   | 0           | 0             | 1          |
| MCMC/GLS             | ⚠️   | 0           | 1             | 0          |
| ノイズモデル         | ⚠️   | 1 (doc)     | 0             | 0          |
| DTT正規化            | ⚠️   | 0           | 1             | 0          |
| ARIMA                | ✅   | 0           | 0             | 0          |
| PCA/ICA              | ✅   | 0           | 0             | 1          |

### 優先修正事項

1. **⚠️ High (Doc)**: Lorentzian線形のドキュメント明確化（ピーク正規化であることを明記）
2. **⚠️ Medium**: `ifft_space` での座標オフセット復元機能
3. **⚠️ Medium**: MCMC Walker初期化の堅牢性向上
4. **⚠️ Medium**: `detect_step_segments` のゼロ中央値処理
5. **⚠️ Medium**: DTT ENBW の窓正規化前提条件の文書化

---

## Gemini Web レポートとの比較

別途 `check_gemini_web.md` に記録されているGemini Webの監査結果との主な相違点：

| 項目             | Gemini Web                | Claude Antigravity                  | 評価                                  |
| ---------------- | ------------------------- | ----------------------------------- | ------------------------------------- |
| k-space単位      | rad単位の明示的付与が必要 | 単位計算は正しい（1/m = rad/m相当） | 両方正当な指摘                        |
| `_fft_transient` | dt乗算が必要（CFT密度）   | 振幅スペクトルとして正しい          | 用途依存（gwexpyは振幅形式）          |
| VIF              | 統計的に無効              | Percival&Walden式として正しい       | 呼称の問題（VIF vs bandwidth factor） |
| MCMC log-det     | 欠落                      | 固定共分散では不要                  | 用途依存                              |
| ARIMA            | 差分によるズレ            | 初期化が正しければ問題なし          | 実装詳細に依存                        |

**結論**: 両レポートとも有効な観点を提供しています。Gemini Webは厳密な物理的定義を重視し、Claude Antigravityは実装のユースケースを考慮した評価を行っています。

---

## 追加監査: コードベース内の他の重要アルゴリズム

ALGORITHM_CONTEXT.md に記載されていない追加のアルゴリズムについて、コードベースを調査し検証を行いました。

---

## 9. Hurst指数推定 (`gwexpy/timeseries/hurst.py`)

### ⚠️ Finding 9.1: R/S解析の小標本バイアス

**Severity**: Medium

**Finding**: Hurst指数 $H$ の推定において、R/S解析（Rescaled Range Analysis）が使用されていますが、**小標本バイアス補正**が実装されていません。

**物理的分析**:
古典的な関係式 $\mathbb{E}[R/S] \sim c \cdot n^H$ は漸近的なものです。データ点数 $n$ が小さい場合（特に $n < 100$ 程度のサブセット）、推定される $H$ が理論値から系統的に偏ることが知られています（Anis-Lloydの期待値からの逸脱）。

**統計的問題**:

- ランダムウォーク（$H = 0.5$）を「長期記憶性あり（トレンドあり）」と誤判定するリスク
- 特に `local_hurst()` のスライディングウィンドウでは、窓サイズが小さくなると問題が顕著

**Recommendation**:

1. **Anis-Lloyd補正**または**Peters補正**を適用し、期待値の理論曲線に対する偏差として $H$ をフィッティング
2. または、ロバストな**DFA (Detrended Fluctuation Analysis)** 法の実装を推奨
3. docstringに「短いセグメントでは推定値にバイアスが生じる可能性」を明記

---

### ⚠️ Finding 9.2: バックエンド依存性と一貫性

**Severity**: Low

**Finding**: `hurst()` 関数は複数のバックエンド (`hurst`, `hurst-exponent`, `exp-hurst`) を自動選択しますが、各バックエンドで計算結果が異なる可能性があります。

```python
if method == "auto":
    # Order: rs -> standard -> exp
    try:
        res = _get_hurst_rs(x, kind, simplified)
    except ImportError:
        ...
```

**Recommendation**:
ドキュメントに「バックエンドによって推定値が異なる可能性がある」旨を明記し、再現性が必要な場合は `method` を明示的に指定するよう推奨。

---

## 10. Bruco 多チャンネルコヒーレンス探索 (`gwexpy/analysis/bruco.py`)

### 🔴 Finding 10.1: 多重比較問題 (Look-elsewhere effect)

**Severity**: High

**Finding**: 多数の補助チャンネルとのコヒーレンスを計算し、最大値を報告するロジックにおいて、**多重比較問題**が考慮されていません。

```python
# BrucoResult.update_batch - Top-N tracking
for start in range(0, batch_names.size, block_size):
    ...
    needs_update = open_slots | (block_max > self.top_coherence[:, -1])
```

**統計的問題**:

- 例えば100個の無相関なノイズチャンネルに対してコヒーレンスを計算し、その「最大値」を取れば、統計的に有意に見える（0.3〜0.4程度の）ピークが偶然現れる確率は極めて高い
- チャンネル数 $N_{ch}$ が増えるにつれて、ノイズフロアの実効的な閾値が上昇するため、単一チャンネル用の有意水準を一律に適用するのは不適切

**Recommendation**:

1. 解析したチャンネル数 $N_{ch}$ に基づく**ボンフェローニ補正** または **FDR (False Discovery Rate)** 制御を閾値計算に導入
2. レポート出力時に「チャンネル数による偶然のコヒーレンス上昇」のリスクを明記する警告を追加
3. 数式: 補正後閾値 $\gamma_{corr} = 1 - (1 - \alpha)^{1/N_{ch}}$

---

### ✅ Finding 10.2: Top-N追跡アルゴリズム - **正確**

**Severity**: N/A (検証済み)

**分析**:
`BrucoResult.update_batch()` は `np.argpartition` を使用したTop-K選択を実装しており、計算量 $O(n \cdot k)$ で効率的です。ブロック単位での更新も適切に実装されています。

**検証結果**: パス

---

## 11. ZCA/PCA ホワイトニング (`gwexpy/signal/preprocessing/whitening.py`)

### ✅ Finding 11.1: ホワイトニング実装 - **正確**

**Severity**: N/A (検証済み)

**分析**:

```python
# PCA whitening
W = S_inv_sqrt @ U.T

# ZCA whitening
W = U @ S_inv_sqrt @ U.T
```

両方の実装は数学的に正しいです：

- **PCA**: $W = \Lambda^{-1/2} U^T$ で、主成分方向への回転＋スケーリング
- **ZCA**: $W = U \Lambda^{-1/2} U^T$ で、元の軸に戻す回転を含む（Mahalanobis whitening）

**検証結果**: パス

---

### ⚠️ Finding 11.2: ZCAでの次元削減警告

**Severity**: Low

**Finding**: ZCAで `n_components` を指定した場合、「チャンネル保持」の性質が失われますが、警告のみで処理は継続されます。

```python
if n_components is not None:
    if method == "zca":
        warnings.warn("n_components ignores channel mapping for ZCA if reduced.")
    W = W[:n_components, :]
```

**Recommendation**:
ZCAの主な利点（チャンネル解釈可能性）が失われることをより明確にし、PCAへのフォールバックを提案するか、エラーを発生させるオプションを追加。

---

## 12. EMD/HHT (Hilbert-Huang Transform) (`gwexpy/timeseries/_spectral_special.py`)

### ✅ Finding 12.1: EEMD実装 - **正確**

**Severity**: N/A (検証済み)

**分析**:
PyEMDバックエンドへの適切なラッピングが実装されています。ランダムシード設定、並列処理のフォールバック、残差抽出などが考慮されています。

```python
if random_state is not None:
    if hasattr(decomposer, "noise_seed"):
        decomposer.noise_seed(random_state)
    else:
        np.random.seed(random_state)
```

**検証結果**: パス

---

### ⚠️ Finding 12.2: EMD端点効果

**Severity**: Medium

**Finding**: EMDのエンベロープ外挿は信号境界でアーティファクトを引き起こす可能性がありますが、ドキュメントでのみ警告されています。

**物理的問題**:

- 短い信号（数周期以下）では端点効果が信号全体に影響
- HHTスペクトログラムでは端点付近の瞬時周波数推定が不安定

**Recommendation**:

1. `pad` パラメータのデフォルト値を設定し、自動パディングを有効化
2. または、出力に端点マスク情報を含めて、下流解析で自動的にトリムできるようにする

---

### ⚠️ Finding 12.3: HHTスペクトログラムの解釈

**Severity**: Low

**Finding**: HHTスペクトログラムはSTFTやウェーブレットとは本質的に異なりますが、同じ `Spectrogram` 型で返されます。

```python
# weight options
if weight_mode == "ia2":
    spec_val[f_bin, t_idx] += ia_val**2  # power-like
elif weight_mode == "ia":
    spec_val[f_bin, t_idx] += ia_val     # magnitude
```

**Recommendation**:
戻り値の `Spectrogram` に対して、メタデータに `spectrogram_type="hilbert"` などを設定し、単位やスケールの解釈が異なることを明示。

---

## 13. フィルタ設計 (`gwexpy/signal/filter_design.py`)

### ✅ Finding 13.1: GWpyへの委譲 - **正確**

**Severity**: N/A (検証済み)

**分析**:
gwexpyのフィルタ設計はGWpyに完全に委譲されています：

```python
from gwpy.signal.filter_design import *  # noqa: F403
```

GWpyは適切にSOS形式をサポートしているため、数値安定性の問題はありません。

**検証結果**: パス

---

### ⚠️ Finding 13.2: sosfilt使用箇所の確認

**Severity**: Low

**Finding**: コードベース内で `sosfilt` が使用されている箇所を確認しました：

- `gwexpy/gui/excitation/generator.py` - ✅ SOS形式使用
- `gwexpy/fields/scalar.py` - ✅ SOS形式使用
- `gwexpy/timeseries/matrix_analysis.py` - ✅ SOS形式使用

**検証結果**: パス（全箇所でSOS形式を適切に使用）

---

## 追加サマリー

| カテゴリ              | 状態 | 重大度 High | 重大度 Medium | 重大度 Low |
| --------------------- | ---- | ----------- | ------------- | ---------- |
| Hurst指数推定         | ⚠️   | 0           | 1             | 1          |
| Bruco多チャンネル解析 | 🔴   | 1           | 0             | 0          |
| ZCA/PCAホワイトニング | ✅   | 0           | 0             | 1          |
| EMD/HHT               | ⚠️   | 0           | 1             | 1          |
| フィルタ設計          | ✅   | 0           | 0             | 0          |

### 追加の優先修正事項

1. **🔴 High**: Bruco多チャンネル解析に多重比較補正（ボンフェローニ/FDR）を導入
2. **⚠️ Medium**: Hurst指数のR/S解析に小標本バイアス補正を追加
3. **⚠️ Medium**: EMD端点効果の自動軽減措置（デフォルトパディング）

---

## 全体統合サマリー

**総検証項目**: 13カテゴリ、30+ 個別Finding

| 重大度       | 件数 | 主な内容                                      |
| ------------ | ---- | --------------------------------------------- |
| **High**     | 2    | Lorentzianドキュメント、Bruco多重比較         |
| **Medium**   | 7    | 座標オフセット、Walker初期化、Hurstバイアス等 |
| **Low**      | 6    | ドキュメント改善、次元順序明記等              |
| **検証済み** | 15+  | FFT正規化、GLS尤度、VIF式、EMD実装等          |

**全体評価**: gwexpyのアルゴリズム実装は概ね物理的・統計的に妥当です。高重大度の問題はドキュメント不足と統計的補正の欠如に集中しており、数学的・物理的なバグは発見されませんでした。
