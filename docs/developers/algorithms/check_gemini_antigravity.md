# gwexpy アルゴリズム実装監査レポート

対象ドキュメント (`ALGORITHM_CONTEXT.md`) に基づき、指定された6つの領域について物理的・統計的妥当性および数値安定性の観点から監査を行いました。

## 監査結果サマリ

| ID  | 領域                     | 項目                           | 重大度      | 概要                                                            |
| --- | ------------------------ | ------------------------------ | ----------- | --------------------------------------------------------------- | ------ | --------- |
| 1   | Physical Fields          | `ScalarField.fft_space`        | Low         | $dx < 0$ による $k$ 生成ロジックの非標準的な挙動                |
| 2   | Transient Analysis       | `ResponseFunctionAnalysis`     | Medium      | ステップ検出における周波数分解能と許容誤差の競合                |
| 3   | Robust Statistics        | `bootstrap_spectrogram`        | Medium      | 重複ブロックブートストラップにおける定常性仮定と端点効果        |
| 4   | Bayesian Fitting         | `run_mcmc` (GLS)               | **High**    | GLS対数尤度計算における共分散行列の正規化項 ($\log              | \Sigma | $) の欠落 |
| 5   | Time Series Modeling     | `ARIMA` Forecasting            | Low         | 欠損データや不定間隔データにおける予測時刻 `forecast_t0` のズレ |
| 6   | Dimensionality Reduction | `PCA`/`ICA`                    | Medium/High | 逆変換時の3次元構造復元における次元情報の欠落 (cols次元の消失)  |
| 7   | Special Transforms       | `Laplace`/`CWT`                | Medium      | Laplace変換のオーバーフローリスクおよびCWTの出力形式整合性      |
| 8   | Signal Utils             | `TransferFunction` (Transient) | Medium      | Transientモードにおける除算の不安定性 (Infスパイク)             |

---

## 1. Physical Fields & k-space

### Finding 1.1: $k$ 空間変換スケーリングの妥当性

- **Severity**: Low
- **Finding**:
  コード内の `k_values = 2 * np.pi * np.fft.fftfreq(npts, d=abs(dx_value))` という計算は、物理的な波数 $k$ (rad/unit) の定義と一致しており正しい。`d=abs(dx)` とすることで、FFTの基本的な定義域と整合させている。
  一方で、`if dx_value < 0: k_values = -k_values` という処理は、座標軸が反転している場合に波数の符号を反転させているが、これは物理的意味（波動の進行方向等）としては解釈可能だが、FFTの標準的な出力（正の周波数、負の周波数）の並びとは直感的に逆行する可能性がある。
- **Recommendation**:
  `dx < 0` の場合の $k$ の定義（符号の意味）をドキュメントに明記すること。特に位相因子 $e^{ikx}$ と $e^{-ikx}$ のどちらを採用しているかと整合性が取れているか確認を推奨する。

## 2. Transient Response Analysis

### Finding 2.1: 自動ステップ検出の数値的不安定性

- **Severity**: Medium
- **Finding**:
  `detect_step_segments` において `abs(f - current_freq) > freq_tolerance` で周波数変化を検出しているが、FFTの周波数分解能 $\Delta f = 1/T$ に対して `freq_tolerance` が小さい場合、離散化誤差（binning error）やノイズによるピーク位置の揺らぎ（jitter）を誤ってステップ変化として検出するリスクがある。
- **Recommendation**:
  `freq_tolerance` のデフォルト値を周波数分解能 $\Delta f$ 以上に設定するか、あるいは `freq_tolerance` の判定を連続する複数時点での変化確認（デバウンス処理）と組み合わせることを検討されたい。

### Finding 2.2: `_fft_transient` における振幅のスケーリング

- **Severity**: Medium
- **Finding**:
  `_fft_transient` は `rfft / N` および係数 `2.0` の乗算を行っている。これは正弦波入力に対してその振幅をスペクトル値として返す「振幅スペクトル」の定義には合致するが、過渡応答（Transient）解析で重要となる信号の総エネルギー（Parsevalの定理）や、ノイズ解析で用いられるPSD（パワー密度）とはスケーリングが異なる。用途が限定的であり、誤用を招きやすい。
- **Recommendation**:
  このメソッドが返す値が「正弦波振幅相当値」であることを明記し、エネルギー保存が必要な計算やPSD比較を行う場合には適切な補正係数（帯域幅補正等）を適用するよう注意書きを追加する。

## 3. Robust Statistics

### Finding 3.1: VIFとブートストラップの併用における統計的解釈

- **Severity**: Low/Medium
- **Finding**:
  VIF ($\text{Variance Inflation Factor}$) の計算式自体はWelch法におけるオーバーラップの影響評価として正しい。しかし、`bootstrap_spectrogram` 内でブロックブートストラップを行う場合、それ自体がデータの相関構造（依存性）を模倣しようとする手法であるため、VIFによる分散補正をさらに適用すると、信頼区間の過小評価または二重補正につながる懸念がある。
- **Recommendation**:
  ブートストラップ法を用いる場合は、ブートストラップ分布から直接信頼区間を求める（Percentile法など）のが自然であり、VIF補正はパラメトリックな標準誤差推定の場合に限定するなど、適用条件を整理する。

### Finding 3.2: ブロックブートストラップの端点処理

- **Severity**: Medium
- **Finding**:
  `num_possible_blocks = n_time - block_size + 1` としており、Moving Block Bootstrap の形式をとっているが、時系列の端点付近のデータが含まれる頻度が中央部より少なくなる。データ長が短い場合、このバイアスが無視できない。また、非定常な信号（過渡応答が含まれる場合など）に対してブロックシャッフルを行うと、時間構造が破壊される。
- **Recommendation**:
  Circular Block Bootstrap を検討するか、あるいはサンプル数が十分大きいことを前提要件とする。非定常データへの適用には警告を設ける。

## 4. Bayesian Fitting

### Finding 4.1: GLS対数尤度における正規化項の欠落

- **Severity**: High
- **Finding**:
  `run_mcmc` で使用される `log_prob` 関数では、`chi2` ($\chi^2 = r^{\dagger} \Sigma^{-1} r$) のみを用いて対数尤度を $\log P \propto -0.5 \chi^2$ と計算している。
  しかし、多変量正規分布の対数尤度は $\log \mathcal{L} = -\frac{1}{2} (\chi^2 + \log \det \Sigma + N \log 2\pi)$ である。
  共分散行列 $\Sigma$ (`cov_inv` の逆行列) がモデルパラメータに依存せず固定であれば $\log \det \Sigma$ は定数となり無視できるが、フィッティングパラメータにノイズモデル（ジッター項など）が含まれ $\Sigma$ が変動する場合、この項が欠落していると最適解が大きく歪む（分散を小さく見積もる解に落ち込む）原因となる。
- **Recommendation**:
  共分散行列がパラメータ依存で変化する場合は、必ず $\log \det \Sigma$ （または `slogdet`）を尤度計算に含めるよう数式を修正する。固定共分散の場合のみ現状の実装が許容される旨を注記する。

## 5. Time Series Modeling

### Finding 5.1: ARIMA予測結果のタイムスタンプ整合性

- **Severity**: Low
- **Finding**:
  `forecast_t0 = self.t0 + n_obs * self.dt` は、入力データが欠損なく完全に連続している場合のみ正しい。欠損値処理（drop）やリサンプリングが行われた場合、`n_obs`（観測点数）と物理的な時間経過（Duration/dt）が一致しなくなり、予測開始時刻が過去にずれる可能性がある。
- **Recommendation**:
  `n_obs` ではなく、モデルに入力された最終データのタイムスタンプ (`t_end`) を基準に `forecast_t0 = t_end + dt` と計算することを推奨する。

## 6. Dimensionality Reduction

### Finding 6.1: PCA逆変換時の3次元構造復元の不整合

- **Severity**: Medium/High
- **Finding**:
  PCAの逆変換 (`pca_inverse_transform`) において、`X_rec_3d = X_rec_val[:, None, :]` という操作が行われている。これにより、復元後の形状は `(Features, 1, Time)` に固定される。
  もし入力データの形状が `(Channels, Cols, Time)` で `Cols > 1` であった場合、`flatten` 処理 (`reshape(-1, time)`) で `Channels * Cols` 個の特徴量に展開された情報は復元されるものの、元の `Cols` 次元構造は失われ、すべて第1次元（Features）または第2次元に丸め込まれてしまう。
- **Recommendation**:
  `pca_fit` 実行時に元の形状（Original Shape）をメタデータとして保存し、逆変換時にその形状に合わせて `reshape` するロジックを追加する。現状の実装では `Cols > 1` のデータに対して構造破壊が起こる。

## 7. Special Spectral Transforms (HHT/Laplace/CWT)

### Finding 7.1: Laplace変換における数値安定性 (Overflow)

- **Severity**: Medium
- **Finding**:
  `TimeSeries.laplace` メソッドには、`stlt` (Short-Time Laplace) と異なり、指数関数 $\exp(-\sigma t)$ の増幅に対するガードレール（オーバーフローチェック）が実装されていません。
  $\sigma$ が負の値（発散方向）で、かつデータの時間長 $T$ が長い場合、$-\sigma T$ が浮動小数点の限界（約709）を超え、計算途中で `inf` が発生する可能性があります。
- **Recommendation**:
  `stlt` メソッドと同様に、計算前に `max(-sigma * t_max)` をチェックし、オーバーフローが予測される場合は例外を投げるか、ユーザーに警告するロジックを追加することを推奨します。

### Finding 7.2: CWT出力形式とSpectrogramの整合性

- **Severity**: Low
- **Finding**:
  `cwt` メソッドで `output='spectrogram'` を選択した際、`Spectrogram` オブジェクトが生成されますが、CWT係数（通常は複素数）がそのまま渡されています。
  `gwpy` の `Spectrogram` や多くのプロットルーチンは、実数値（パワーまたは振幅）を期待することが多く、複素数のまま保持されていると可視化時に絶対値を取るなどの追加操作が必要になります。また、`unit` の扱いも振幅次元のままです。
- **Recommendation**:
  係数をそのまま返す仕様であれば、docstringに「複素係数を含む」旨を明記するか、`magnitude=True` オプションを設けて絶対値を返す便宜を図ると親切です。

## 8. Signal Processing Utilities

### Finding 8.1: Transient Transfer Function の正則化

- **Severity**: Medium
- **Finding**:
  `transfer_function(mode='transient')` は `FFT(B) / FFT(A)` の単純比計算を行っています。分母 `FFT(A)` がゼロに近い周波数や、ノイズフロア以下の領域では、計算結果が発散（`inf` または巨大な値）し、解析不能なスパイクを生じます。
  `steady` モード（Welch法の平均）と異なり、単発の割り算は極めて不安定です。
- **Recommendation**:
  除算の安定化を図るため、正則化パラメータ（`reg` または `epsilon`）を導入し、`FFT(B) / (FFT(A) + epsilon)` の形にするか、コヒーレンス閾値でマスクする等のオプションを提供することを推奨します。

---

以上
