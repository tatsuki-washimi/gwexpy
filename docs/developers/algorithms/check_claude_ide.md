# gwexpy アルゴリズム検証レポート（Claude Code IDE 独立監査）

対象ドキュメント (`ALGORITHM_CONTEXT.md`) およびソースコードの直接検証に基づき、指定された6つの領域について物理的・統計的妥当性および数値安定性の観点から監査を行いました。

## 監査結果サマリ

| ID   | 領域                     | 項目                       | 重大度          | 概要                                                                      |
| ---- | ------------------------ | -------------------------- | --------------- | ------------------------------------------------------------------------- |
| 4.1  | Bayesian Fitting         | `run_mcmc` (GLS)           | **High**        | GLS対数尤度における $\log\det\Sigma$ 正規化項の欠落                       |
| 6.1  | Dimensionality Reduction | `PCA`/`ICA` 逆変換         | **Medium-High** | 逆変換時の3D構造復元で cols 次元を常に1に固定                             |
| 2.1  | Transient Analysis       | `detect_step_segments`     | **Medium**      | freq_tolerance と FFTビン分解能の不整合リスク                             |
| 2.2  | Transient Analysis       | `_fft_transient`           | **Medium**      | 振幅スケーリング規約の未定義（振幅 vs PSD vs ASD）                       |
| 3.1  | Robust Statistics        | VIF + bootstrap            | **Medium**      | VIF補正とブートストラップCIの二重補正リスク                               |
| 3.2  | Robust Statistics        | `bootstrap_spectrogram`    | **Medium**      | ブロックブートストラップの端点効果・定常性仮定                            |
| 4.2  | Bayesian Fitting         | 複素残差                   | **Medium**      | 複素ガウスモデルの統計仮定が未明示                                        |
| 1.2  | Physical Fields          | `ScalarField.fft_space`    | **Low-Medium**  | $dx < 0$ での $k$ 軸符号反転の位相規約リスク                             |
| 5.1  | Time Series Modeling     | ARIMA Forecasting          | **Low-Medium**  | 欠損データにおける `forecast_t0` のズレ                                   |
| 1.1  | Physical Fields          | `ScalarField.fft_space`    | **Low**         | 角波数 vs サイクル波数の混同リスク（ドキュメント不足）                    |
| 7.1  | Coupling Analysis        | `CouplingFunctionAnalysis` | **Medium**      | SigmaThreshold のガウス仮定および `n_avg` 計算の端数処理                 |
| 8.1  | Noise Models             | `lorentzian`               | **Medium**      | ピーク正規化 Lorentzian と標準的な面積正規化形式の不一致                  |
| 8.2  | Noise Models             | `geomagnetic_background`   | **Medium**      | 単位自動判定の脆弱なヒューリスティック                                    |
| 9.1  | Signal Normalization     | `enbw` (DTT mode)          | **Low-Medium**  | DTT モード ENBW の窓正規化前提                                           |
| 10.1 | Transfer Function        | `transfer_function`        | **Medium**      | Transient モードの除算正則化欠如                                         |
| 11.1 | Whitening                | ZCA whitening              | **Low-Medium**  | `n_components` 使用時のチャネル保存性の喪失                              |
| 12.1 | Hurst Exponent           | `local_hurst`              | **Low**         | ループ内 MockTS 生成の非効率性                                           |
| 13.1 | Laplace Transform        | `TimeSeries.laplace`       | **Medium**      | 指数関数オーバーフローのガードレール欠如                                  |

---

## 1. Physical Fields & k-space（`ScalarField.fft_space`）

### Finding 1.1: $k$ 空間変換スケーリングの妥当性

- **Severity**: Low
- **Source**: `gwexpy/fields/scalar.py:540`
- **Finding**:
  `k_values = 2 * np.pi * np.fft.fftfreq(npts, d=abs(dx_value))` は、角波数 $k$ \[rad/unit\] の標準的定義と一致しており物理的に正しい。`ifft_space` での `dx = 2π/(n·|dk|)` による逆変換復元も $dk = 2\pi/(n\,dx)$ の関係を再構成しており整合している。
- **Recommendation**:
  ドキュメントで $k$ が角波数 (rad/length) であることを明示し、サイクル波数 (1/length) との混同を防止する注記を追加。

### Finding 1.2: $dx < 0$ での $k$ 軸符号反転

- **Severity**: Low-Medium
- **Source**: `gwexpy/fields/scalar.py:542-543`
- **Finding**:
  ```python
  if dx_value < 0:
      k_values = -k_values
  ```
  座標軸が降順（負のステップ）の場合に波数の符号を反転させているが、FFTの離散化は配列インデックス順序に対して定義されるため、負の $dx$ を物理軸の向きとして吸収する設計は慣習から外れやすい。他ライブラリ（xarray等）との相互運用で混乱要因となり得る。
- **Recommendation**:
  $dx < 0$ を許容する場合、以下を仕様として文書化すべき：
  - value の格納順と座標軸（metadata）の対応関係
  - $k$ の符号反転が位相因子 $e^{+ikx}$ / $e^{-ikx}$ のどちらの規約に対応するか
  - 互換性の観点では「FFTは $d = |dx|$ で定義し、座標の向きはメタデータとして管理」が誤用防止に有効

---

## 2. Transient Response Analysis

### Finding 2.1: 自動ステップ検出の数値的不安定性

- **Severity**: Medium
- **Source**: `gwexpy/analysis/response.py:192-264`
- **Finding**:
  `detect_step_segments` のデフォルト値は `fftlength=1.0`（→ $\Delta f = 1$ Hz）、`freq_tolerance=1.0` Hz。$\text{freq\_tolerance} = \Delta f$ という設定は最小限の余裕しかなく、SNRが十分でもビン飛び（ノイズによるピーク位置の ±1 ビン変動）で偽のステップ境界を検出するリスクがある。

  また、SNR判定 (`peak_vals > median_level * snr_threshold`) はパワースペクトルの中央値に対する比であり、PSD推定では各周波数ビンの分布はχ²的であるため、メディアンレベルとの単純比較はスペクトルの形状（線スペクトル vs 広帯域）に依存する。
- **Recommendation**:
  - `freq_tolerance ≥ 2·Δf` を推奨デフォルトとする
  - 境界判定に「連続 $k$ 点で変化が持続」等のヒステリシス（デバウンス）ロジックを導入する設計指針を明記

### Finding 2.2: `_fft_transient` における振幅スケーリング

- **Severity**: Medium
- **Finding**:
  `_fft_transient` は `rfft(x)/N` + 非DC/非Nyquist成分の2倍という処理を行っている。これは正弦波入力に対して振幅値を返す「片側振幅スペクトル」の定義には合致するが、以下が未記載だと誤用を招く：
  - これは振幅スペクトル（ピーク値）なのか RMS なのか
  - PSD/ASD推定の前処理として使う場合のスケーリング整合
  - `ifft` 側での pad・片側補正の逆操作（Parseval整合含む）
- **Recommendation**:
  - DFT規約 $\tilde{x}_k = \frac{1}{N}\sum_n x_n e^{-i2\pi kn/N}$ の採用を docstring に明記
  - 単一正弦波 $x[n] = A\sin(2\pi f_0 n / f_s)$ を入力し、該当ビンの片側振幅が $A$ になることを確認するユニットテストを追加

---

## 3. Robust Statistics（VIF・block bootstrap）

### Finding 3.1: VIF とブートストラップの併用における統計的解釈

- **Severity**: Medium
- **Source**: `gwexpy/spectral/estimation.py:159-232`
- **Finding**:
  VIF計算式自体は Bendat & Piersol に基づく標準形で数学的に正しい：
  ```python
  rho_sq_weighted_sum += weight * (rho**2)
  vif = 1.0 + 2.0 * rho_sq_weighted_sum
  return np.sqrt(vif)
  ```
  - `max_lag = ceil(nperseg / step)` の打ち切りは妥当（shift ≥ nperseg で窓の自己相関がゼロ）
  - `weight = 1 - k/n_blocks` の Bartlett 型重みも正しい

  問題は、VIF補正とブートストラップCIを同時に適用する場合の**二重補正リスク**。VIFは「パラメトリックな標準誤差の補正」、ブートストラップは「分布のノンパラメトリック推定」であり、併用すると信頼区間の過小評価につながる懸念がある。
- **Recommendation**:
  - ブートストラップ法を用いる場合は分布から直接CIを求める（Percentile法等）のが自然であり、VIF補正はパラメトリック推定に限定する
  - 適用条件を API ドキュメントで整理

### Finding 3.2: ブロックブートストラップの端点処理と定常性仮定

- **Severity**: Medium
- **Source**: `gwexpy/spectral/estimation.py:252+`
- **Finding**:
  Moving Block Bootstrap の形式 (`num_possible_blocks = n_time - block_size + 1`) では、端点付近のデータが含まれる頻度が中央部より少なくなる。また非定常信号（ドリフト・トレンド・注入オン/オフ混在等）にブロックシャッフルを行うと時間構造が破壊される。`block_size` のデフォルトが `None`（= 標準ブートストラップにフォールバック）の場合、重なり窓 Welch の相関を無視してしまう。
- **Recommendation**:
  - Circular Block Bootstrap の検討、またはサンプル数の十分性を前提要件とする
  - `block_size` のデフォルトを `nperseg - noverlap`（ストライド）の整数倍に設定するか、自動推定ロジックを追加する設計指針を明記
  - 非定常データへの適用には警告を設ける

---

## 4. Bayesian Fitting（`run_mcmc` の GLS 対数尤度）

### Finding 4.1: $\log\det\Sigma$ 正規化項の欠落【最重要】

- **Severity**: **High**
- **Source**: `gwexpy/fitting/core.py:633-644`
- **Finding**:
  ```python
  if cov_inv is not None:
      val = r.conj() @ cov_inv @ r
      chi2 = float(np.real(val))
  else:
      chi2 = float(np.sum(np.abs(r / dy) ** 2))
  return -0.5 * chi2
  ```
  正しい多変量ガウス尤度は：
  $$\log \mathcal{L} = -\frac{1}{2}\left(\mathbf{r}^\dagger \Sigma^{-1} \mathbf{r} + \log\det\Sigma + N\log 2\pi\right)$$

  - **$\Sigma$ がパラメタ非依存（固定）の場合**: $\log\det\Sigma$ は定数であり、事後分布の形状（最尤点、相対確率）に影響しない。パラメタ推定は正しい。ただしモデル比較（ベイズファクター等）には使えない。
  - **$\Sigma$ がパラメタ依存の場合**: 推定が系統的に歪む。特にノイズ振幅やハイパーパラメータを同時推定するモデルでは、**分散を小さく見積もる解に落ち込む**原因となる。
- **Recommendation**:
  - `cov_inv` が固定であるという前提条件を docstring および assertion で明示
  - パラメタ依存共分散を扱う場合は `np.linalg.slogdet` を用いて $\log\det\Sigma$ を含めるよう仕様要件として追加

### Finding 4.2: 複素残差のエルミート形式

- **Severity**: Medium
- **Source**: `gwexpy/fitting/core.py:636-638`
- **Finding**:
  ```python
  val = r.conj() @ cov_inv @ r
  chi2 = float(np.real(val))
  ```
  エルミート形式 $\mathbf{r}^\dagger \Sigma^{-1} \mathbf{r}$ の実部を取るのは、数値誤差で微小虚部が生じるケースへの対処として合理的。ただし、複素データの確率モデル（circular complex Gaussian vs 実虚独立等）により尤度の係数が変わる場合がある。
- **Recommendation**:
  「circular complex Gaussian noise を仮定」等、採用する統計モデルを docstring に明記。

---

## 5. Time Series Modeling（ARIMA → GPS-aware TimeSeries）

### Finding 5.1: ARIMA 予測結果のタイムスタンプ整合性

- **Severity**: Low-Medium
- **Source**: `gwexpy/timeseries/arima.py:140-145`
- **Finding**:
  ```python
  n_obs = int(self.res.nobs) if hasattr(self.res, "nobs") else len(self.res.fittedvalues)
  forecast_t0 = self.t0 + n_obs * self.dt
  ```
  等間隔・欠損なしの前提では正しい。ただし `nan_policy='impute'` で欠損を補間した場合、`res.nobs`（statsmodels 内部の観測点数）が元のサンプル数と一致しない可能性がある。
- **Recommendation**:
  - `n_obs` ではなく、モデルに入力された最終データのタイムスタンプ `t_end` を基準に `forecast_t0 = t_end + dt` とする方式を推奨
  - 欠損データ入力時に forecast の GPS 時刻が正しいことを確認するテストケースを追加

---

## 6. Dimensionality Reduction（PCA/ICA flatten・復元）

### Finding 6.1: PCA/ICA 逆変換時の3次元構造復元の不整合

- **Severity**: Medium-High
- **Source**: `gwexpy/timeseries/decomposition.py:196, 313`
- **Finding**:
  fit 側で `(channels, cols, time)` → `(channels*cols, time)` と平坦化：
  ```python
  X_features = X_proc.value.reshape(-1, X_proc.shape[-1])  # line 196
  ```
  逆変換側で cols 次元を常に 1 に固定：
  ```python
  X_rec_3d = X_rec_val[:, None, :]  # line 313
  ```
  元データが `cols > 1` の場合、flatten で畳み込まれた `Channels × Cols` 個の特徴量は復元されるものの、元の `(Channels, Cols, Time)` 構造は失われる。ICA 側 (`decomposition.py:532`) も同様に `rec_3d = rec.T[:, None, :]` としている。

  コメント (line 303-311) にも「元の形状が不明」「(features, 1, samples) をデフォルトとする」と記載されており、設計上の制限として認識されている。
- **Recommendation**:
  - `pca_fit` / `ica_fit` 実行時に `input_meta` へ元の形状 `(channels, cols)` を保存し、逆変換で `reshape(channels, cols, time)` とする
  - 現状の実装では `cols > 1` のデータに対して構造破壊が起こるため、`cols > 1` 入力時に警告を発するか、制限事項として API ドキュメントに明記

---

## 7. Coupling Function Analysis

### Finding 7.1: SigmaThreshold のガウス仮定と `n_avg` 計算

- **Severity**: Medium
- **Source**: `gwexpy/analysis/coupling.py:76-140`
- **Finding**:
  `SigmaThreshold` は、パワースペクトルの統計変動を $\sigma / \sqrt{n_\text{avg}}$ で評価し、閾値を `factor = 1 + sigma / sqrt(n_avg)` として設定している。これは各周波数ビンの PSD 推定が正規分布に従うことを暗黙に仮定している。

  しかし、Welch 法による PSD 推定の各ビンは自由度 $2K$（$K$: 平均セグメント数）のχ²分布に従い、$K$ が小（< 10）の場合はガウス近似の精度が低い。特に PSD のテール（高い値側）では閾値が緩すぎる傾向がある。

  また `n_avg` の計算：
  ```python
  n_avg = max(1, (duration - overlap) / (fftlength - overlap))
  ```
  `overlap` と `fftlength` の単位整合（秒 vs サンプル）はコード上では保証されているが、`fftlength == overlap` の場合にゼロ除算が発生する。実用上そのような設定はないが、防御的チェックがない。

- **Recommendation**:
  - $K$ が小さい場合にはχ²分位点ベースの閾値、あるいは少なくとも対数正規近似を検討
  - `fftlength == overlap` のガードを追加
  - `SigmaThreshold` の docstring にガウス近似の適用条件（$K \gtrsim 10$）を明記

---

## 8. Noise Models（スペクトル線形状・磁場背景）

### Finding 8.1: Lorentzian 関数の正規化規約

- **Severity**: Medium
- **Source**: `gwexpy/noise/peaks.py`
- **Finding**:
  実装されている Lorentzian の形式は：
  $$L(f) = A \cdot \frac{\gamma}{\sqrt{(f - f_0)^2 + \gamma^2}}$$
  これは **ピーク正規化**（$L(f_0) = A$）であり、物理学で標準的に用いられる **面積正規化** Lorentzian：
  $$L_\text{standard}(f) = \frac{A}{\pi} \cdot \frac{\gamma}{(f - f_0)^2 + \gamma^2}$$
  とは異なる。また、分母が $\sqrt{\cdot}$ であるため、これは厳密には Lorentzian のべき乗スペクトル密度（ASD 表現: $\sqrt{\text{PSD}}$）に近い形式であり、PSD として使う場合は2乗が必要になる。

  Voigt プロファイルも Faddeeva 関数 `wofz` で正しく実装されているが、出力を Lorentzian と同一次元で比較する場合、正規化の整合に注意が必要。

- **Recommendation**:
  - docstring に「ピーク正規化形式」であることを明記し、$A$ がピーク値（ASD 相当）であることを示す
  - PSD フィットに使う場合の変換式 $\text{PSD} = L(f)^2$ を明記

### Finding 8.2: `geomagnetic_background` の単位自動判定ヒューリスティック

- **Severity**: Medium
- **Source**: `gwexpy/noise/magnetic.py`
- **Finding**:
  `geomagnetic_background` は `amplitude_1hz` の数値の大きさで単位を推定するヒューリスティックを含んでいる：
  ```python
  if amplitude_1hz < 1e-9:  # T 系
  ```
  この手法は脆弱であり、SI 単位だが振幅が非常に小さい磁場計（SQUID など）や、逆にガウス系単位で大きな値を取る場合に誤判定する。

  また、Schumann 共鳴のモデリングで `total_psd += peak_asd.value**2` とインコヒーレント和を取っているのは、各モードが独立であるという仮定のもとで正しい。
- **Recommendation**:
  - 単位は暗黙の推定ではなく、`unit` パラメータとして明示的に受け取る設計を推奨
  - 現行のヒューリスティックを残す場合は、推定結果をログ出力し、ユーザーが確認できるようにする

---

## 9. Signal Normalization（ENBW・DTT 変換）

### Finding 9.1: DTT モード ENBW の窓正規化前提

- **Severity**: Low-Medium
- **Source**: `gwexpy/signal/normalization.py`
- **Finding**:
  ENBW の計算において、DTT モードでは：
  ```python
  enbw_dtt = (fs * n) / (sum_w ** 2)
  ```
  ここで `sum_w = sum(w)` は窓関数の総和。この式は「窓の2乗和 $\sum w_i^2 = N$」（すなわち RMS 正規化された窓）を仮定しているが、scipy の窓関数はピーク正規化（最大値=1）で返されるため、直接適用すると ENBW 値が窓の種類によって系統的にずれる。

  `convert_scipy_to_dtt` 変換関数では、この違いを吸収する補正係数：
  ```python
  ratio = (sum_w2 * n) / (sum_w ** 2)
  ```
  が適用されている。コード内にはコメントアウトされた初期の誤った導出（line 119付近）が残っているが、最終的に使われている式（line 129付近）は正しい。

- **Recommendation**:
  - コメントアウトされた誤導出を削除するか、「参考：これは誤り」と注記して学習リソースとする
  - DTT/scipy 間のスケーリング差は混乱しやすいため、docstring に変換関係の数式を明示

---

## 10. Transfer Function（Transient モード）

### Finding 10.1: Transient モードの除算正則化欠如

- **Severity**: Medium
- **Source**: `gwexpy/timeseries/_signal.py:1211-1410`
- **Finding**:
  `transfer_function(mode='transient')` は `FFT(B) / FFT(A)` の直接比計算を行っている。内部の `_divide_with_special_rules` は `0/0 → NaN`、`x/0 → ±inf` を明示的に処理しているが、分母がゼロに近い（ノイズフロア以下の）周波数ビンでは数値的に巨大な値が発生し、解析不能なスパイクを生じる。

  `steady` モード（Welch 法の平均化）では平均化により分母のゼロ近傍が緩和されるのに対し、transient モードは単発の割り算であり不安定性が顕著。

- **Recommendation**:
  - 正則化パラメータ `epsilon` を導入し、`FFT(B) / (FFT(A) + epsilon)` 形式を提供
  - あるいはコヒーレンス閾値 $\gamma^2 > \gamma_\text{min}^2$ でマスクするオプションを追加
  - 最低限、docstring に transient モードの出力に inf/NaN が含まれ得ることを明記

---

## 11. Whitening（PCA/ZCA）

### Finding 11.1: ZCA + `n_components` 使用時のチャネル保存性の喪失

- **Severity**: Low-Medium
- **Source**: `gwexpy/signal/preprocessing/whitening.py`
- **Finding**:
  ZCA whitening は `W = U @ S^{-1/2} @ U^T` により、出力が入力と同じ空間に留まる（チャネルの物理的解釈が保持される）ことが利点である。しかし `n_components < n_features` を指定すると $U$ が切り詰められ、`W` の次元が `(n_features, n_components)` となり、出力の次元が変わる。この場合、ZCA の「入力空間保存」という性質は失われる。

  コード内で `eps=1e-12` による安定化と `pinv(W)` による逆変換が実装されており、数値的にはロバストだが、`n_components` を指定した ZCA の出力が物理チャネルと1対1対応しなくなる点は、ユーザーにとって直感に反する。

- **Recommendation**:
  - ZCA モードで `n_components` を指定した場合に「チャネル保存性が失われる」旨の warning を発する（現状コード内にもコメントとして認識されている）
  - PCA whitening では次元削減が自然であるため、次元削減が必要な場合は PCA whitening を推奨する旨をドキュメントに記載

---

## 12. Hurst Exponent

### Finding 12.1: `local_hurst` の実装効率

- **Severity**: Low
- **Source**: `gwexpy/timeseries/hurst.py`
- **Finding**:
  `local_hurst` は時系列をスライディングウィンドウで分割し、各窓に対して Hurst 指数を計算する。各窓データに対して `MockTS` クラスのインスタンスをループ内で生成しており、外部ライブラリ（`hurst`/`hurst-exponent`/`exp-hurst`）のインターフェース要件を満たすためのアダプタとして機能している。

  機能的には正しいが、窓数が数千〜数万になる場合、オブジェクト生成のオーバーヘッドが無視できなくなる。また、マルチバックエンドのフォールバックチェーン（`hurst` → `hurst-exponent` → `exp-hurst`）は起動時に一度だけ評価すれば十分だが、ループの外側で行われているか確認が必要。

- **Recommendation**:
  - `MockTS` のインスタンス生成を1回に抑え、データのみを差し替える設計を検討
  - バックエンドの選択はモジュールレベルで1回だけ実施し、ループ内では関数ポインタのみを使用

---

## 13. Laplace Transform

### Finding 13.1: 指数関数オーバーフローのガードレール欠如

- **Severity**: Medium
- **Source**: `gwexpy/timeseries/` 内の Laplace 変換実装
- **Finding**:
  `TimeSeries.laplace` メソッドでは、$\sigma$ が負の値（不安定極方向）の場合に $\exp(-\sigma t)$ が指数的に増大する。$-\sigma \cdot t_\text{max} > 709$（`float64` の限界）の場合、計算途中で `inf` が発生する。

  一方、`stlt`（Short-Time Laplace Transform）メソッドでは短い窓に分割することで事実上のガードレールが機能しているが、フルレンジの `laplace` メソッドには明示的なオーバーフローチェックがない。

- **Recommendation**:
  - 計算前に `max(-sigma * t_array)` をチェックし、閾値（例: 700）を超える場合に `OverflowWarning` を発するか、例外を投げるロジックを追加
  - `stlt` と同様のガードレールを `laplace` にも実装

---

## 補足観察

### `_inverse_scaler` のブロードキャスト整合

- **Severity**: Low
- **Source**: `gwexpy/timeseries/decomposition.py:127-132`
- **Finding**: `_inverse_scaler` は `val * scale + mean` を返すが、scaler は元の `(time, features)` 軸で fit されており、逆変換後の `(features, 1, time)` 形状とのブロードキャストが正しく動作するかは shape に依存する。cols=1 の制約下では問題ないが、一般化した場合にバグの原因となる。

### emcee ウォーカー初期化の境界違反リスク

- **Severity**: Low
- **Source**: `gwexpy/fitting/core.py:666`
- **Finding**: `pos = p0_float + stds * 1e-1 * np.random.randn(n_walkers, ndim)` にて、パラメタに境界制約がある場合にランダム摂動で初期位置が境界外に出る可能性がある。`log_prob` で `-np.inf` を返すため発散はしないが、多数のウォーカーが無効な初期位置になると MCMC の収束が遅延する。

### FastCoherenceEngine（Welch coherence）のスケーリング

- **Severity**: Low
- **Source**: `gwexpy/analysis/bruco.py:75-147`
- **Finding**: スケール `2/(fs*sum(w²))` は片側PSD密度の典型形だが、DC/Nyquist の片側係数（×1 か ×2 か）の扱いが scipy と完全一致するかは要確認。Coherence は比であるため多くのスケール誤差は相殺されるが、CSD/PSD の推定が同一スケールであることが重要。

---

## 結論

1. **即座の対応が推奨される項目（High）**: MCMC の GLS 尤度における $\log\det\Sigma$ の欠落。現在の `cov_inv` が固定であるという前提を明示的にアサートし、将来の拡張時にバグを防止すべき。
2. **設計仕様の明確化が必要な項目（Medium-High）**: PCA/ICA の3D形状復元。`cols=1` の制約を明示するか、元の shape を保存して復元するかを決定すべき。
3. **追加検出項目で対応が推奨される項目（Medium）**:
   - Coupling Analysis: SigmaThreshold のガウス仮定の適用条件明記と `n_avg` のエッジケース防御
   - Noise Models: Lorentzian の正規化規約の明示と磁場背景の単位自動判定の改善
   - Transfer Function (Transient): 除算正則化の導入
   - Laplace Transform: オーバーフローガードレールの追加
4. **基本ロジックの妥当性**: k空間変換、VIF計算、Welch系処理、ARIMA GPS時刻管理、Voigt プロファイル、Schumann 共鳴モデル、ENBW 変換係数、PCA/ZCA whitening の基本ロジックは概ね標準に沿っており、重大な数学的誤りは検出されなかった。

---

*監査実施: Claude Code IDE (claude-opus-4-5-20251101), 2026-01-31*
*追加アルゴリズム監査追記: 2026-02-01*
