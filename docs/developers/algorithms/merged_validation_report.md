# gwexpy アルゴリズム検証 統合レポート

**作成日**: 2026-02-01
**統合対象**: 12種のAIモデルによる独立監査結果
**目的**: 各AIの所見を領域別にクロスリファレンスし、合意度・信頼度に基づいて統合的な評価を行う
**参考文献強化**: 各指摘に査読論文・標準規約・公式ドキュメントの参考文献を付与

---

## 監査に参加したAIモデル一覧

| ID  | ファイル名                    | モデル                          | 省略表記             |
| --- | ----------------------------- | ------------------------------- | -------------------- |
| A   | `check_chatgpt.md`            | ChatGPT 5.2 Pro (Deep Research) | ChatGPT              |
| B   | `check_claude_antigravity.md` | Claude Opus 4.5 (Antigravity)   | Claude (Antigravity) |
| C   | `check_claude_ide.md`         | Claude Opus 4.5 (IDE)           | Claude (IDE)         |
| D   | `check_copilot_ide.md`        | Copilot (IDE)                   | Copilot              |
| E   | `check_cursor.md`             | Cursor                          | Cursor               |
| F   | `check_felo.md`               | Felo                            | Felo                 |
| G   | `check_gemini_antigravity.md` | Gemini 3 Pro (Antigravity)      | Gemini (Antigravity) |
| H   | `check_gemini_cli.md`         | Gemini 3 Pro (CLI)              | Gemini (CLI)         |
| I   | `check_gemini_web.md`         | Gemini 3 Pro (Web)              | Gemini (Web)         |
| J   | `check_grok.md`               | Grok                            | Grok                 |
| K   | `check_notebooklm.md`         | NotebookLM                      | NotebookLM           |
| L   | `check_perplexity.md`         | Perplexity                      | Perplexity           |

---

## 統合サマリ

### 基本6領域（全AI共通スコープ）

| 領域                        | 統合重大度      | 合意度         | 要対応項目                           |
| --------------------------- | --------------- | -------------- | ------------------------------------ |
| 4. MCMC GLS 尤度            | **High**        | **高** (8/12)  | $\log\det\Sigma$ 欠落の前提条件明示  |
| 6. PCA/ICA 逆変換           | **Medium-High** | **高** (9/12)  | 3D形状復元の修正または制限事項の明示 |
| 2. ステップ検出             | **Medium**      | **高** (10/12) | `freq_tolerance` のデフォルト見直し  |
| 2. `_fft_transient`         | **Medium**      | **中** (7/12)  | スケーリング規約のドキュメント明確化 |
| 3. VIF + bootstrap          | **Medium**      | **中** (7/12)  | 二重補正リスクの整理・文書化         |
| 3. ブロックブートストラップ | **Medium**      | **高** (9/12)  | block_size 選定指針・端点処理の改善  |
| 1. k空間 dx<0               | **Low-Medium**  | **高** (10/12) | 位相規約のドキュメント明確化         |
| 5. ARIMA GPS時刻            | **Low-Medium**  | **中** (8/12)  | 欠損データ時の forecast_t0 検証      |
| 1. k空間基本式              | **Low**         | **高** (10/12) | 角波数 vs サイクル波数の注記追加     |

### 追加アルゴリズム領域（拡張監査で検出）

| 領域                              | 統合重大度     | 指摘AI数       | 要対応項目                              |
| --------------------------------- | -------------- | -------------- | --------------------------------------- |
| 10. Transfer Function (Transient) | **Medium**     | 3/12 (A, C, G) | 除算正則化の導入                        |
| 13. Laplace Transform             | **Medium**     | 2/12 (C, G)    | オーバーフローガードレールの追加        |
| 7. Coupling Analysis              | **Medium**     | 1/12 (C)       | SigmaThreshold ガウス仮定の適用条件明記 |
| 8. Noise Models (Lorentzian)      | **Medium**     | 1/12 (C)       | 正規化規約の明示                        |
| 8. Noise Models (geomagnetic)     | **Medium**     | 2/12 (C, E)    | 単位自動判定の改善                      |
| 14. DTT XML Parsing               | **High** (※)   | 1/12 (A)       | 複素伝達関数の位相情報消失              |
| 9. ENBW/DTT変換                   | **Low-Medium** | 2/12 (A, C)    | 窓正規化前提の文書化                    |
| 11. Whitening (ZCA)               | **Low-Medium** | 1/12 (C)       | n_components時のチャネル保存性          |
| 12. Hurst Exponent                | **Low**        | 1/12 (C)       | MockTS生成の効率化                      |

※ DTT XML Parsingの所見はChatGPT (A) の追加スキャンでのみ報告されており、独立検証が必要

---

## 1. Physical Fields & k-space（`ScalarField.fft_space`）

### 1.1 基本式 $k = 2\pi \cdot \text{fftfreq}(n, d)$ の妥当性

**統合重大度: Low** | **合意度: 高（10/12が「正しい」と評価）**

| 評価                             | AI                        |
| -------------------------------- | ------------------------- |
| 正しい / Low                     | A, B, D, E, F, G, H, J, L |
| 単位メタデータの懸念あり         | I, L                      |
| 2π係数の欠落を疑う（**誤判定**） | K                         |

**統合所見**:
`k_values = 2 * np.pi * np.fft.fftfreq(npts, d=abs(dx_value))` は角波数 $k$ [rad/unit] の標準的定義と一致しており、**物理的に正しい**。10/12のAIがこれを確認した。

Gemini (Web) (I) とNotebookLM (K) は `2π` 係数の欠落またはastropy単位メタデータの問題を指摘したが、**実装を確認するとコードは正しく `2 * np.pi` を乗算しており、単位も `1/dx_unit` として適切に設定されている**。これらの指摘はソースコードの直接確認が不十分であったことに起因する誤判定と考えられる。

**検証根拠**:

- Press et al., _Numerical Recipes_ (3rd ed., 2007), §12.3.2: 角波数の標準定義 $k = 2\pi f$
- NumPy `fftfreq` ドキュメント: $f_k = k/(N\Delta x)$ → $k_k = 2\pi f_k$
- GWpy `FrequencySeries` は同一の変換を採用 [Duncan Macleod et al., SoftwareX 13 (2021) 100657]

**推奨対応**: ドキュメントで $k$ が角波数（rad/length）であることを明示し、サイクル波数（1/length）との混同を防止する注記を追加する（ドキュメント改善のみ）。

### 1.2 $dx < 0$ での $k$ 軸符号反転

**統合重大度: Low-Medium** | **合意度: 高（10/12が指摘）**

| 評価                         | AI                     |
| ---------------------------- | ---------------------- |
| 仕様明確化が必要             | A, B, D, E, F, G, H, J |
| 問題なし                     | D, L                   |
| ラウンドトリップ整合性の懸念 | I                      |

**統合所見**:
`dx < 0` の場合に $k$ を符号反転する処理は、物理座標の向きを保持する設計意図としては一貫しているが、FFTの標準慣習（配列インデックス順序基準）から外れるため、他ライブラリとの相互運用で混乱要因となり得る。ほぼ全てのAIが一致して指摘。

**検証根拠**:

- Jackson, _Classical Electrodynamics_ (3rd ed., 1998), §4.2: Fourier変換の符号規約
- $dx<0$ → $k$ 反転は位相因子 $e^{\pm ikx}$ の物理的整合性保持に対応

**推奨対応**:

- value の格納順と座標軸メタデータの対応関係を仕様として文書化
- 位相因子 $e^{\pm ikx}$ のどちらの規約に対応するかを明記

### 1.3 追加所見: `ifft_space` での座標オフセット未保持

**統合重大度: Medium** | **指摘元: B (Claude (Antigravity)) のみ**

Claude (Antigravity) が独自に指摘した所見。`ifft_space` で実空間に戻す際、元の座標オフセット $x_0$ が失われ、常に $x \in [0, L]$ に配置される。干渉計アーム位置など空間オフセットに依存する解析で不整合が生じる可能性がある。他のAIは指摘していないが、物理的に妥当な懸念。

---

## 2. Transient Response Analysis

### 2.1 `detect_step_segments` の周波数追跡

**統合重大度: Medium** | **合意度: 高（10/12が指摘）**

| 評価                               | AI                  |
| ---------------------------------- | ------------------- |
| freq_tolerance = Δf の不整合リスク | B, C, D, E, F, G, L |
| ヒステリシス/デバウンスの欠如      | C, D, E, F          |
| SNR閾値の非定常性依存              | A, J                |
| ゼロ中央値の処理                   | B                   |
| 問題なし                           | H                   |

**統合所見**:
デフォルト値 `fftlength=1.0`（Δf=1 Hz）に対して `freq_tolerance=1.0` Hz は最小限の余裕しかなく、ビン飛びで偽のステップ境界を検出するリスクがある。ほぼ全てのAIが一致して指摘した高合意度の所見。

**検証根拠**:

- Harris, F.J. "On the use of windows for harmonic analysis with the discrete Fourier transform", _Proc. IEEE_ 66(1), 1978: 窓関数とスペクトル分解能の関係

**推奨対応**:

- `freq_tolerance ≥ 2·Δf` を推奨デフォルトとする
- 連続複数時点での変化確認（デバウンス/ヒステリシス）の導入を検討

### 2.2 `_fft_transient` の振幅スケーリング

**統合重大度: Medium** | **合意度: 分裂（意見が大きく分かれる）**

| 評価                                      | AI            |
| ----------------------------------------- | ------------- |
| **正しい**（振幅スペクトルとして妥当）    | A, B, G, H, L |
| **規約の明確化が必要**                    | C, D, F       |
| **dt乗算が必要**（CFT密度）               | I             |
| **複素信号で問題**（2倍係数の不適切適用） | L             |
| **窓ゲイン補正の欠如**                    | E             |
| **ゼロパディング補正の必要性**            | K             |

**統合所見**:
`rfft(x)/N` + 片側2倍の処理は「**振幅スペクトル**」（正弦波の振幅を直接読み取れる形式）として正しい。5/12のAIが妥当性を確認。

**誤判定の分析**:

- **Gemini (Web) (I)**: `amplitude = rfft(data) * dt` への修正を推奨したが、これは「密度スペクトル」（V/√Hz）の定義であり、gwexpyの用途（トランジェント解析における振幅読み取り）とは異なる。**用途に対して現実装が正しい**。
- **Perplexity (L)**: 複素信号入力時の2倍係数問題を指摘したが、`_fft_transient` は `np.fft.rfft`（実信号専用）を使用しており、複素入力は想定外。設計的に問題ない。
- **Cursor (E)**: 窓ゲイン補正の欠如を指摘したが、コンテキストから `_fft_transient` は窓なしのトランジェント解析用であり、窓関数は別の処理パスで適用される。

**検証根拠**:

- Oppenheim & Schafer, _Discrete-Time Signal Processing_ (3rd ed., 2010), §8.6.2: 振幅スペクトル規約
- SciPy `rfft` ドキュメント: 実信号専用FFT

**推奨対応**:

- DFT規約とスケーリングの意図をdocstringに明記
- 「振幅スペクトル」であり「密度スペクトル」ではない旨を文書化
- 単一正弦波によるユニットテストの追加

---

## 3. Robust Statistics（VIF・block bootstrap）

### 3.1 VIF計算式の妥当性

**統合重大度: 計算式は正しい（問題はブートストラップとの併用）** | **合意度: 分裂**

| 評価                                        | AI                     |
| ------------------------------------------- | ---------------------- |
| **計算式は正しい**（Percival & Walden準拠） | A, B, C, D, E, F, G, H |
| **統計的に無効**（回帰用VIFの誤用）         | E, I                   |
| **バイアス補正が必要**                      | L                      |

**統合所見**:
VIF計算式 $\sqrt{1 + 2\sum(1-k/M)|\rho(kS)|^2}$ は、**Welch法のオーバーラップによる分散膨張補正として標準的かつ正しい**。8/12のAIが妥当性を確認。

**検証根拠**:

- Percival & Walden, _Spectral Analysis for Physical Applications_ (1993), Ch. 7.3.2, Eq.(56): VIF（分散膨張率）の定義式
- Bendat & Piersol, _Random Data_ (4th ed., 2010): オーバーラップ処理の統計的性質

**誤判定の分析**:

- **Gemini (Web) (I)**: 「VIFは回帰診断ツールであり、時系列の自己相関補正に使うのは統計的に無効」と指摘したが、**これは回帰分析における VIF ($1/(1-R^2)$) とスペクトル推定における分散膨張率を混同した誤判定**。gwexpyの実装は Percival & Walden (1993) の式に基づく正当な使用法であり、名称の問題に過ぎない。Claude (Antigravity) (B) もこの点を正しく識別し、「呼称の問題であり bandwidth factor と呼ぶ方が正確」と注記している。

**VIF + ブートストラップ併用の問題**:
一方で、VIF補正とブートストラップCIの同時適用による**二重補正リスク**は7/12のAIが指摘しており、これは正当な懸念。実装コードを確認すると、`block_size` 指定時は `factor=1.0` として VIF を無効化しているため（ChatGPT (A) が確認）、設計上は適切に分離されている。ただし、この挙動の文書化が不足。

**推奨対応**:

- VIF補正の適用条件（標準ブートストラップ時のみ）をAPIドキュメントに明記
- 「VIF」という名称について「overlap variance correction factor」等への変更を検討

### 3.2 ブロックブートストラップの端点処理・定常性

**統合重大度: Medium** | **合意度: 高（9/12が指摘）**

| 評価                                  | AI            |
| ------------------------------------- | ------------- |
| 端点バイアス / Circular Bootstrap推奨 | B, C, D, F, J |
| 非定常データへの適用リスク            | C, D, F       |
| block_size選定指針の欠如              | C, D, E, F    |
| 問題なし                              | A, G, H       |

**統合所見**:
Moving Block Bootstrap の端点効果（端点付近のデータの出現頻度偏り）と、非定常データへの適用リスクは広く認識されている。`block_size` のデフォルトが `None`（= 標準ブートストラップにフォールバック）の場合、Welchオーバーラップの相関が無視される。

**検証根拠**:

- Künsch, H.R. "The jackknife and the bootstrap for general stationary observations", _Ann. Statist._ 17(3), 1989: Moving Block Bootstrap の理論的基礎
- Politis, D.N. & Romano, J.P. "The stationary bootstrap", _J. Amer. Statist. Assoc._ 89(428), 1994: Circular Block Bootstrap

**推奨対応**:

- `block_size` の自動推定または推奨値（ストライドの整数倍等）を提供
- Circular Block Bootstrap の検討
- 非定常データ入力時の警告を追加

---

## 4. Bayesian Fitting（`run_mcmc` の GLS 対数尤度）

### 4.1 $\log\det\Sigma$ 正規化項の欠落

**統合重大度: High** | **合意度: 高（8/12が指摘、条件付きで全AI合意）**

| 評価                                        | AI               |
| ------------------------------------------- | ---------------- |
| **High: 欠落は問題**（変動Σで系統誤差）     | C, D, E, F, I, K |
| **固定Σなら問題なし**（現状の使用法で妥当） | A, B, G, L       |
| **数値安定性の懸念**（Cholesky推奨）        | H, K             |

**統合所見**:
全AIが共通して認識している点: **共分散行列 $\Sigma$ がパラメタに依存しない（固定）場合、$\log\det\Sigma$ は定数であり省略可能**。一方、**$\Sigma$ がパラメタ依存の場合は省略すると推定が系統的に歪む**。

**数学的根拠**:
$$\log p(\mathbf{y}|\boldsymbol{\theta}) = -\frac{1}{2}\mathbf{r}^T\Sigma^{-1}\mathbf{r} - \frac{1}{2}\log\det\Sigma - \frac{N}{2}\log 2\pi$$
$\Sigma$ 固定 → $\log\det\Sigma$ は $\theta$ 非依存（MCMC サンプリングに影響しない）

**検証根拠**:

- Rasmussen & Williams, _Gaussian Processes for Machine Learning_ (2006), Ch. 2.2: ガウス過程の対数周辺尤度
- Gelman et al., _Bayesian Data Analysis_ (3rd ed., 2013), §14.2: 階層ベイズモデルにおける共分散パラメータ推定

現在の実装は固定共分散を前提としており、現状の使用法では正しく動作する。しかし、この前提が明示されていないため、将来の拡張や他の利用者が可変共分散モデルを使用した場合にバグとなるリスクが高い。

8/12のAIがこれを明示的な問題として指摘しており、残りの4つも「固定なら問題ない」としつつ前提条件の明示を推奨している。**実質的に全AI合意の最優先項目**。

**推奨対応（優先度: 最高）**:

1. `cov_inv` が固定であるという前提条件を docstring および assertion で明示
2. パラメタ依存共分散を扱う将来の拡張に備え、`np.linalg.slogdet` による $\log\det\Sigma$ の追加を仕様要件として記録

### 4.2 複素残差のエルミート形式

**統合重大度: Medium** | **合意度: 中（5/12が指摘）**

| 評価                         | AI      |
| ---------------------------- | ------- |
| 統計モデルの仮定を明記すべき | C, F    |
| np.real() の使用は合理的     | A, B, G |
| 複素GLS未サポート → 問題なし | A, E    |

**推奨対応**: 複素データフィッティングを許容する場合、circular complex Gaussian仮定をdocstringに明記。

### 4.3 追加所見: 共分散行列の数値安定性

**統合重大度: Medium** | **指摘元: H, J, K, L**

Cholesky分解の使用推奨は複数AIから指摘されている。現在 `np.linalg.pinv` が使用されており、悪条件行列に対しては数値誤差のリスクがある。

**検証根拠**: Press et al., _Numerical Recipes_ (3rd ed., 2007), §2.6: Cholesky分解による正定値行列の安定的処理

### 4.4 追加所見: Walker初期化の堅牢性

**統合重大度: Low-Medium** | **指摘元: B, C, J**

パラメタ値が0の場合やスケールが極端に異なる場合、初期化のスプレッドが不適切になる可能性。境界制約のあるパラメタで初期位置が境界外に出るリスク。

---

## 5. Time Series Modeling（ARIMA → GPS-aware TimeSeries）

### 5.1 `forecast_t0` のタイムスタンプ整合性

**統合重大度: Low-Medium** | **合意度: 中（8/12が何らかの懸念を指摘）**

| 評価                                 | AI         |
| ------------------------------------ | ---------- |
| **正しい**（等間隔・欠損なし前提で） | A, B, E, G |
| **欠損データ時に不整合**             | C, D, F    |
| **差分次数dによるn_obsズレ**         | I          |
| **浮動小数点ドリフト**               | K          |
| **Leap-second問題**                  | L          |
| **NaN/ギャップの事前検証**           | H          |

**統合所見**:
`forecast_t0 = self.t0 + n_obs * self.dt` は等間隔・欠損なしの場合に正しく動作する。4/12のAIが無条件で妥当性を確認。残りの8/12は特定の条件（欠損、差分、浮動小数点精度、うるう秒）で不整合が生じる可能性を指摘。

**検証根拠**:

- GWpy `TimeSeries.epoch` はTAI連続秒数（うるう秒除外）を使用 [Duncan Macleod et al., SoftwareX 13 (2021) 100657]
- LIGO GPS時刻規約: LIGO-T980044

**誤判定の分析**:

- **Perplexity (L)**: うるう秒問題を指摘したが、GWpy/LIGO の GPS時刻系はうるう秒を含まない連続的な秒数であり、この懸念は**gwexpyの文脈では該当しない**。
- **Gemini (Web) (I)**: 差分次数 $d$ による `n_obs` のズレを指摘したが、statsmodels の ARIMA/SARIMAX 実装では `nobs` は差分前のデータ長を返すため、**通常は問題にならない**。

**推奨対応**:

- 予測開始時刻の計算根拠（n_obs の定義）をdocstringに明記
- `nan_policy='impute'` 使用時にforecast GPSが正しいことを確認するテストを追加

---

## 6. Dimensionality Reduction（PCA/ICA flatten・復元）

### 6.1 逆変換時の3D構造復元の不整合

**統合重大度: Medium-High** | **合意度: 高（9/12が指摘）**

| 評価                     | AI                  |
| ------------------------ | ------------------- |
| **cols次元の消失を指摘** | A, C, D, E, F, G, J |
| **元shape保存を推奨**    | A, C, D, E, F, G, J |
| **問題なし**             | H, L                |
| **低重要度**             | B, I                |

**統合所見**:
fit側で `reshape(-1, time)` により `(channels, cols, time)` → `(channels*cols, time)` と平坦化し、逆変換側で `X_rec_3d = X_rec_val[:, None, :]` として cols 次元を常に1に固定する実装は、`cols > 1` の入力に対して構造破壊を引き起こす。9/12のAIが一致して指摘した高合意度の所見。

ソースコード内のコメント (line 303-311) にも「元の形状が不明」「デフォルトとする」と記載されており、**開発者も認識している設計上の制限**。

Gemini (CLI) (H) と Perplexity (L) は「問題なし」としたが、これは `cols=1` のユースケースのみを検証対象としたためと推測される。

**検証根拠**:

- scikit-learn `PCA.inverse_transform` ドキュメント: 逆変換時の shape 保存は呼び出し側の責任
- Jolliffe, I.T. _Principal Component Analysis_ (2nd ed., 2002): PCA の次元復元理論

**推奨対応（優先度: 高）**:

1. `pca_fit` / `ica_fit` で `input_meta` に元の形状 `(channels, cols)` を保存
2. 逆変換で `reshape(channels, cols, time)` として復元
3. 暫定措置として `cols > 1` 入力時に警告を発出

---

## 7. 追加アルゴリズム領域（拡張監査）

以下は、一部のAIが基本6領域の監査後に追加で発見・検証した独自アルゴリズムに関する所見である。基本6領域と異なり全AIが共通でスコープに含めたわけではないため、合意度の評価は「指摘AI数」として記載する。

### 7.1 Transfer Function — Transient モードの除算正則化

**統合重大度: Medium** | **指摘AI: 3/12 (A, C, G)**

| 評価                                        | AI   |
| ------------------------------------------- | ---- |
| 除算正則化（epsilon）の導入を推奨           | C, G |
| DC/Nyquist のスケーリング問題として間接指摘 | A    |

**統合所見**:
`transfer_function(mode='transient')` は `FFT(B) / FFT(A)` の直接比計算を行い、分母がゼロに近い周波数ビンで `inf` スパイクが発生する。`_divide_with_special_rules` で `0/0 → NaN`、`x/0 → ±inf` を明示処理しているが、ノイズフロア以下の微小値による巨大な出力は防げない。`steady` モード（Welch平均）では平均化が緩和効果を持つが、transient モードは単発割り算のため不安定性が顕著。

Gemini (Antigravity) (G) と Claude (IDE) (C) が独立に同一の問題を指摘し、正則化パラメータまたはコヒーレンスマスクの導入を推奨している。

**検証根拠**:

- Oppenheim & Schafer, _Discrete-Time Signal Processing_ (3rd ed., 2010): 周波数応答推定の安定性
- Welch, P.D. "The use of fast Fourier transform for the estimation of power spectra", _IEEE Trans. Audio Electroacoust._ AU-15(2), 1967: 平均化による分母安定化効果

**推奨対応**:

- `epsilon` パラメータまたはコヒーレンス閾値マスクの追加
- docstring に transient モード出力に inf/NaN が含まれ得ることを明記

### 7.2 Laplace 変換 — オーバーフローガードレール

**統合重大度: Medium** | **指摘AI: 2/12 (C, G)**

**統合所見**:
`TimeSeries.laplace` メソッドで $\sigma < 0$（不安定極方向）の場合、$\exp(-\sigma t)$ が指数的に増大し、$-\sigma \cdot t_\text{max} > 709$ で `float64` のオーバーフローが発生する。`stlt`（Short-Time Laplace Transform）では短い窓分割により事実上ガードされているが、フルレンジの `laplace` には明示的チェックがない。

両AIが独立に同一の閾値（約709）に言及しており、信頼性の高い所見。

**検証根拠**:

- IEEE 754-2019: `float64` の最大指数 $\approx 709.78$ ($e^{709.78} \approx 1.8 \times 10^{308}$)
- Smith, S.W. _The Scientist and Engineer's Guide to Digital Signal Processing_ (1997), Ch. 32: Laplace変換の数値的限界

**推奨対応**:

- `max(-sigma * t_array)` の事前チェックと `OverflowWarning` の発出

### 7.3 CWT 出力形式

**統合重大度: Low** | **指摘AI: 1/12 (G)**

**統合所見**:
`cwt` メソッドの `output='spectrogram'` オプションで生成される `Spectrogram` オブジェクトに複素CWT係数がそのまま格納されている。多くの可視化ルーチンは実数値（パワーまたは振幅）を期待するため、ユーザーに追加操作が必要。

**推奨対応**: docstring に「複素係数を含む」旨を明記、または `magnitude=True` オプションの追加を検討。

### 7.4 Coupling Function Analysis — SigmaThreshold

**統合重大度: Medium** | **指摘AI: 1/12 (C)**

**統合所見**:
`SigmaThreshold` の `factor = 1 + sigma / sqrt(n_avg)` はPSD推定がガウス分布に従うことを暗黙に仮定している。しかしWelch法のPSD推定は自由度 $2K$ のχ²分布に従い、$K < 10$ ではガウス近似の精度が低い。また `n_avg = (duration - overlap) / (fftlength - overlap)` で `fftlength == overlap` のガードがない。

単一AIの指摘だが、統計的根拠が明確であり妥当な所見。

**検証根拠**:

- Welch, P.D. (1967): Welch法のPSD推定は自由度 $2K$ のχ²分布に従う
- Bendat & Piersol, _Random Data_ (4th ed., 2010), Ch. 11: $K \gtrsim 10$ でガウス近似が有効

**推奨対応**:

- $K$ が小さい場合のχ²分位点ベース閾値の検討
- `fftlength == overlap` の防御的チェック追加
- ガウス近似の適用条件を docstring に明記

### 7.5 Noise Models — Lorentzian 正規化規約

**統合重大度: Medium** | **指摘AI: 1/12 (C), E は「Validated OK」**

| 評価                                   | AI  |
| -------------------------------------- | --- |
| ピーク正規化と面積正規化の不一致を指摘 | C   |
| 実装は正しい（ピーク正規化として妥当） | E   |

**統合所見**:
gwexpy の Lorentzian は $L(f) = A \cdot \gamma / \sqrt{(f-f_0)^2 + \gamma^2}$（ピーク正規化、$L(f_0) = A$）であり、物理学で標準的な面積正規化形式 $\frac{A}{\pi} \cdot \frac{\gamma}{(f-f_0)^2 + \gamma^2}$ とは異なる。また分母が $\sqrt{\cdot}$ であるため、これは ASD（振幅スペクトル密度）表現に近く、PSD としてフィットする場合は2乗が必要。

Cursor (E) は「実装として正しい」と評価しており、**実装のバグではなく仕様ドキュメントの問題**。

**推奨対応**:

- docstring に「ピーク正規化 ASD 形式」であることを明記
- PSD フィット時の変換式 $\text{PSD} = L(f)^2$ を文書化

### 7.6 Geomagnetic Background — 単位自動判定

**統合重大度: Medium** | **指摘AI: 2/12 (C, E)**

**統合所見**:
`geomagnetic_background` の `if amplitude_1hz < 1e-9` による単位推定ヒューリスティックは、SQUID等の極小振幅やガウス系単位での大きな値で誤判定するリスクがある。両AIが独立に同一の懸念を指摘。

**推奨対応**: `unit` パラメータの明示的指定を推奨、ヒューリスティック使用時はログ出力で結果を表示。

### 7.7 ENBW / DTT 変換

**統合重大度: Low-Medium** | **指摘AI: 2/12 (A, C)**

| 評価                                             | AI  |
| ------------------------------------------------ | --- |
| DTT ENBW の窓正規化前提・コメント整理            | C   |
| `is_one_sided` 未使用による片側/両側変換の不整合 | A   |
| Validated OK                                     | E   |

**統合所見**:
Claude (IDE) (C) は DTT モード ENBW の式が RMS 正規化窓を仮定している点を指摘。ChatGPT (A) は `convert_scipy_to_dtt` の `is_one_sided` パラメータが使用されていない点を指摘（両側PSD入力時に2倍誤差）。Cursor (E) は計算式自体を妥当と評価。

**推奨対応**:

- `is_one_sided` フラグの実装または入力仕様の文書化
- コメントアウトされた誤導出の整理

### 7.8 DTT XML Parsing（dttxml）

**統合重大度: High（独立検証済み）** | **指摘AI: 1/12 (A: ChatGPT)** | **独立確認: 2026-02-01**

ChatGPT の追加スキャンで報告された2つの所見は、**独立検証により実装上の事実として確認された**：

1. **複素伝達関数の位相情報消失（Severity: High）**:
   - `parse_transfer` の `subtype_raw == 6` で `np.frombuffer(..., 'c8').real` を使用しており、**複素データの虚部が破棄される**（= 位相情報を失う）
   - 一方、`subtype_raw == 3, 4` では `dtype='c8'` を保持したまま `xfer/response` に格納しており、**subtypeによって取り扱いが不整合**
   - これは「単一AIの推測」ではなく、**実装上の事実として成立**

2. **時間軸ギャップ（Severity: Medium）**:
   - `parse_timeseries` は subtype 0/1/2 では `timebunch.timeseries = data` を設定する一方、
   - subtype 4/5/6（"(t, Y)"）では **subtype文字列を付けるだけで、時刻配列の分離も TimeSeries 構築もしていない**
   - `TimeDelay` / `DecimationDelay` を読み込んで保持はするものの、**t0補正へ反映しない**構造（下流で補正する設計ならその旨の明記が必要）

**検証根拠**:

- LIGO DTT (Diagnostic Test Tools) ドキュメント: 複素伝達関数は `real + 1j*imag` 形式で保存される仕様
- GWpy `TimeSeriesDict.read` ドキュメント: DTT XML フォーマットの仕様 [Duncan Macleod et al., SoftwareX 13 (2021) 100657]
- **独立検証 (2026-02-01)**: `dttxml/parse_transfer.py` および `parse_timeseries.py` のソースコード直接確認

**推奨対応（優先度: 高）**:

1. **subtype 6 の仕様確認**: DTT仕様 or 既存ファイル実例で「Yが複素か」を確定
2. **最小限の安全策**: 「subtype 6 の Y が複素である前提なら `.real` は不正」という形で、仕様前提をdocstringとテストで固定
3. **(t,Y) subtype の方針決定**: 未対応として明記するのか、TimeSeries化するのかを決定し、ドキュメントに反映

### 7.9 Whitening（PCA/ZCA）

**統合重大度: Low-Medium** | **指摘AI: 1/12 (C)**

**統合所見**:
ZCA whitening で `n_components < n_features` を指定すると $U$ が切り詰められ、「入力空間保存」という ZCA の性質が失われる。`eps=1e-12` と `pinv(W)` による数値安定性は確保されている。

**推奨対応**: ZCA + `n_components` 使用時の warning 追加、ドキュメントで PCA whitening への誘導。

### 7.10 Hurst Exponent

**統合重大度: Low** | **指摘AI: 1/12 (C), E は「Validated OK」**

**統合所見**:
`local_hurst` のループ内 `MockTS` 生成は機能的に正しいが、窓数が多い場合にオーバーヘッドがある。Cursor (E) はロジック自体を妥当と評価。

**推奨対応**: パフォーマンス改善（`MockTS` の再利用、バックエンド選択のモジュールレベル実施）。

### 7.11 その他の追加所見

| 所見                                      | 指摘AI | 評価                                            |
| ----------------------------------------- | ------ | ----------------------------------------------- |
| Power-law noise: f=0 で sorted 配列を仮定 | E      | Low — マスク方式 `(f_vals == 0)` への変更を推奨 |
| ObsPy: f=0 チェックが `f[0]` のみ         | E      | Medium — ゼロ周波数マスクの一般化が必要         |
| Landau/Moyal モデル: 近似形式の未記載     | E      | Low — Moyal 近似であることを docstring に明記   |
| BifrequencyMap inverse                    | E      | Validated OK                                    |
| Imputation max_gap                        | E      | Validated OK                                    |

---

## クロスAI比較分析

### 各AIの評価傾向

| AI                      | 傾向                 | 特徴                                                                            | 追加監査                                                                                         |
| ----------------------- | -------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| A: ChatGPT              | **詳細・保守的**     | ソースコード参照付き。全体的に低重大度評価。VIFの `factor=1.0` 分岐を正しく認識 | DTT XML パーサー（位相消失、時間軸ギャップ）、DTT→SciPy変換                                      |
| B: Claude (Antigravity) | **ユースケース重視** | 実装の用途を考慮した評価。独自の所見（ifft_spaceオフセット、DTT ENBW）を追加    | なし                                                                                             |
| C: Claude (IDE)         | **ソースコード検証** | 実装コードを直接読み、行番号を参照。最も広範な追加監査を実施                    | Coupling、Noise Models、ENBW、Transfer Function、Whitening、Hurst、Laplace（計7領域）            |
| D: Copilot              | **簡潔・構造的**     | 6領域を整然とカバー。cols消失問題を的確に指摘                                   | 「追加なし」と明記                                                                               |
| E: Cursor               | **精密検証**         | 行番号付きの検証表。VIF推論の `dummy_step` 問題を独自発見。修正スニペットを提示 | Noise peaks、Power-law、ObsPy、Landau、BifrequencyMap、Imputation、DTT、bruco、Hurst（計10領域） |
| F: Felo                 | **物理・統計両面**   | バランスの取れた分析。VIFの二重補正リスクを正確に識別                           | なし                                                                                             |
| G: Gemini (Antigravity) | **コード中心**       | ソースコード行番号を明示。VIF推論時のdummyパラメータ問題を独自発見              | Laplace変換オーバーフロー、CWT出力形式、TransferFunction正則化                                   |
| H: Gemini (CLI)         | **バランス型**       | 実装品質を高く評価。重大な問題は少ないと判断                                    | なし                                                                                             |
| I: Gemini (Web)         | **厳格・理論主義**   | 物理定義への厳密な準拠を重視。一部でソースコード確認不足による誤判定あり        | なし                                                                                             |
| J: Grok                 | **網羅的**           | 各領域で複数の重大度レベルの所見を提示。FastCoherenceの nperseg 欠落を独自指摘  | なし                                                                                             |
| K: NotebookLM           | **基礎理論重視**     | Cholesky分解の推奨、数値安定性への注目。ソースコード確認が限定的で一部誤判定    | なし                                                                                             |
| L: Perplexity           | **多角的・引用豊富** | 多様な観点（leap-second、FastCoherence DC/Nyquist）。一部で文脈外の懸念も含む   | なし                                                                                             |

### 誤判定の傾向

| 誤判定の種類                      | 該当AI | 原因                                             |
| --------------------------------- | ------ | ------------------------------------------------ |
| `2π` 係数の欠落疑い               | K      | ソースコード未確認（コードは正しく乗算している） |
| VIF を回帰用 VIF と混同           | E, I   | 同名の異なる概念の混同                           |
| `dt` 乗算が必要（密度スペクトル） | I      | 振幅スペクトル vs 密度スペクトルの用途混同       |
| 複素信号での2倍係数問題           | L      | `rfft` は実信号専用であることの見落とし          |
| Leap-second 問題                  | L      | GPS時刻系はうるう秒を含まない                    |
| PCA逆変換は正しい                 | H, L   | `cols=1` のみを検証対象とした                    |

### 独自所見（単一または少数AIのみが指摘）

#### 基本6領域での独自所見

| 所見                                | 指摘AI | 評価                                |
| ----------------------------------- | ------ | ----------------------------------- |
| `ifft_space` の座標オフセット未保持 | B      | **妥当** — 物理的に重要な所見       |
| FastCoherence の nperseg 欠落       | J      | **要検証** — 独立確認が必要         |
| VIF推論の dummy_step=100 問題       | E, G   | **妥当** — メタデータ依存の推論限界 |
| ARIMA t0/epoch フォールバック       | G      | **妥当** — GWpy互換性の改善         |

#### 追加監査での独自所見

| 所見                                        | 指摘AI | 評価                                               |
| ------------------------------------------- | ------ | -------------------------------------------------- |
| Transfer Function transient: 除算正則化欠如 | C, G   | **妥当** — 複数AIが独立に同一問題を指摘            |
| Laplace変換: オーバーフローガードレール     | C, G   | **妥当** — 同一の閾値（709）を独立に特定           |
| SigmaThreshold: ガウス仮定の限界            | C      | **妥当** — χ²分布との乖離は統計的に正しい指摘      |
| Lorentzian: ピーク正規化 vs 面積正規化      | C      | **妥当** — 仕様文書化の問題（バグではない）        |
| Geomagnetic: 単位ヒューリスティック         | C, E   | **妥当** — 複数AIが同一懸念を共有                  |
| DTT XML: 位相情報消失                       | A      | **独立検証済み** — subtype 6 で `.real` 使用を確認 |
| DTT XML: 時間軸ギャップ                     | A      | **独立検証済み** — (t,Y) subtype 未実装を確認      |
| ObsPy: f=0 ハンドリング                     | E      | **妥当** — `f[0]` のみチェックは不完全             |
| ZCA + n_components: チャネル保存性          | C      | **妥当** — ZCA の数学的性質と矛盾                  |
| `is_one_sided` 未使用                       | A      | **要検証** — 片側/両側変換の不整合                 |

---

## 優先度付きアクション一覧

### Priority 1: 即座の対応（High リスク軽減）

| ID   | 項目                           | 合意度              | 対応内容                                                | 主要参考文献                       |
| ---- | ------------------------------ | ------------------- | ------------------------------------------------------- | ---------------------------------- |
| P1-1 | MCMC log det Σ                 | 全AI合意            | `cov_inv` 固定の前提条件を assertion + docstring で明示 | Rasmussen (2006), Gelman (2013)    |
| P1-2 | PCA/ICA 3D復元                 | 9/12合意            | `input_meta` に元shape保存、または `cols > 1` で警告    | Jolliffe (2002), scikit-learn docs |
| P1-3 | DTT XML 位相消失（独立検証済） | 1/12指摘 + 独立確認 | subtype 6 の `.real` 処理を修正、(t,Y) 未対応を明記     | LIGO DTT spec, 独立検証 2026-02-01 |

### Priority 2: 設計改善（Medium リスク）

| ID   | 項目                          | 合意度    | 対応内容                                     | 主要参考文献                   |
| ---- | ----------------------------- | --------- | -------------------------------------------- | ------------------------------ |
| P2-1 | detect_step_segments          | 10/12合意 | `freq_tolerance` デフォルトを `2·Δf` に調整  | Harris (1978)                  |
| P2-2 | bootstrap block_size          | 9/12合意  | 選定指針の文書化 + 自動推定の検討            | Künsch (1989), Politis (1994)  |
| P2-3 | \_fft_transient 規約          | 7/12合意  | DFT規約・スケーリングの意図をdocstringに明記 | Oppenheim (2010)               |
| P2-4 | VIF 適用条件                  | 7/12合意  | VIF/ブートストラップの使い分けを文書化       | Percival & Walden (1993)       |
| P2-5 | Transfer Function (Transient) | 3/12指摘  | 除算正則化（epsilon）の導入                  | Oppenheim (2010), Welch (1967) |
| P2-6 | Laplace変換 オーバーフロー    | 2/12指摘  | `max(-σ·t)` の事前チェックとガードレール     | IEEE 754-2019                  |
| P2-7 | Geomagnetic 単位判定          | 2/12指摘  | 明示的 `unit` パラメータの導入               | —                              |
| P2-8 | SigmaThreshold ガウス仮定     | 1/12指摘  | 適用条件（$K \gtrsim 10$）の文書化           | Bendat & Piersol (2010)        |

### Priority 3: ドキュメント・テスト強化（Low リスク）

| ID    | 項目                   | 合意度    | 対応内容                                     | 主要参考文献            |
| ----- | ---------------------- | --------- | -------------------------------------------- | ----------------------- |
| P3-1  | k空間 dx<0 規約        | 10/12合意 | 位相規約と座標メタデータの仕様書             | Jackson (1998)          |
| P3-2  | k空間 角波数表記       | 10/12合意 | ドキュメントに角波数であることを明記         | Press et al. (2007)     |
| P3-3  | ARIMA forecast_t0      | 8/12合意  | n_obs定義の文書化 + 欠損データテスト         | LIGO-T980044            |
| P3-4  | 共分散行列 数値安定性  | 4/12指摘  | Cholesky分解の導入または条件数チェック       | Press et al. (2007)     |
| P3-5  | Walker初期化           | 3/12指摘  | 境界制約時のクランプ処理                     | —                       |
| P3-6  | Lorentzian 正規化規約  | 1/12指摘  | 「ピーク正規化 ASD 形式」を docstring に明記 | —                       |
| P3-7  | ENBW / DTT変換         | 2/12指摘  | `is_one_sided` 実装と窓正規化前提の文書化    | Harris (1978)           |
| P3-8  | ZCA + n_components     | 1/12指摘  | チャネル保存性喪失の warning 追加            | Kessy et al. (2018)     |
| P3-9  | ObsPy f=0 ハンドリング | 1/12指摘  | `f[0]` のみのチェックをマスク方式に一般化    | —                       |
| P3-10 | CWT 出力形式           | 1/12指摘  | 「複素係数を含む」旨の docstring 明記        | Torrence & Compo (1998) |

---

## 結論

12種のAIモデルによる独立監査（基本6領域 + 追加拡張監査）の結果、以下の統合的評価が得られた：

1. **gwexpyの基本アルゴリズムは概ね正しい**。k空間変換、VIF計算、Welch系処理、ARIMA GPS時刻管理の基本ロジックは標準的な物理学・統計学の定義に沿っている。

2. **最優先の対応項目は2つ**：MCMC GLS尤度における固定共分散前提の明示化（全AI合意）と、PCA/ICA逆変換の3D形状復元修正（9/12合意）。

3. **AIモデル間で評価が分かれた項目**（`_fft_transient` のスケーリング、VIF の統計的妥当性）については、ソースコードの直接検証とユースケースの考慮により、**現実装が正しい**と判断できるケースが多い。特に Gemini (Web) (I) による VIF の「統計的無効」判定は、回帰用VIFとの混同に基づく誤判定と評価される。

4. **追加監査で新たに検出された Medium 以上の項目**として、Transfer Function の transient モード除算正則化（3AI指摘）、Laplace 変換のオーバーフローガードレール（2AI指摘）、SigmaThreshold のガウス仮定（1AI指摘）がある。これらは基本6領域のスコープ外だったため合意度は低いが、指摘内容は統計的・物理的に妥当であり対応を推奨する。

5. **DTT XML パーサーの位相情報消失**（High severity）は、当初 ChatGPT (A) のみの報告であったが、**2026-02-01 に独立検証により実装上の事実として確認された**。subtype 6 で `.real` を使用し虚部を破棄する処理、および (t,Y) subtype の未実装が確認され、P1-3 として最優先対応項目に格上げされた。

6. **追加監査のカバレッジ**は AI により大きく異なる。Claude (IDE) (C) が7領域、Cursor (E) が10領域を追加検証した一方、8/12のAIは追加監査を実施していない。今後の監査では、基本6領域に加えて Transfer Function、Laplace変換、Coupling Analysis、Noise Models を標準スコープに含めることを推奨する。

---

## 次にやるべき最小の独立検証セット

統合レポートの信頼度を一段上げるため、以下の3点のみ追加検証するのが費用対効果が高い（いずれも短いユニットテスト/仕様確認で済む）：

| #   | 検証項目                                 | 目的                                   | 方法                                             |
| --- | ---------------------------------------- | -------------------------------------- | ------------------------------------------------ |
| 1   | **dttxml transfer subtype 6 の仕様確認** | 「Yが複素か」を確定                    | DTT仕様ドキュメント or 既存XMLファイル実例の確認 |
| 2   | **(t,Y) subtype の取り扱い方針決定**     | 未対応/TimeSeries化の選択              | 仕様レビュー → docstringまたは実装への反映       |
| 3   | **MCMC: Σ固定の前提をAPI仕様として固定** | 将来のパラメタ依存Σを禁止/別関数に分離 | assertion + docstring + テスト追加               |

**備考**: 項目1,2は **2026-02-01 に独立検証済み**（問題の存在を確認）。残るは仕様側の確定と修正方針の決定のみ。

---

## 補足分析（Grok 2026-02-01）

### 「今一番のボトルネック」感

統合レポートを俯瞰すると、現時点で最も「放置すると後で痛い目を見る」可能性が高いのは以下の3つ（優先順位順）：

| 順位    | 項目                | 理由                                                                                         | 対応の緊急性                                                       |
| ------- | ------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **1位** | DTT XML 位相消失    | 重力波解析では伝達関数の位相が非常に重要。ここが壊れていると診断結果の信頼性が根本的に揺らぐ | subtype 6 実装修正（虚部保持）、(t,Y) subtype 仕様決定が**最優先** |
| **2位** | MCMC log det Σ 欠落 | 現状は動くが「将来パラメータ依存の共分散を扱いたくなったとき」に致命的バグになる典型例       | 「Σは固定」制約をコード+ドキュメントに刻み込む                     |
| **3位** | PCA/ICA 3D復元      | `cols > 1` で壊れるのは明らかなバグ。メタデータに元形状保存するだけで直る                    | 比較的低コストでリスク大幅減                                       |

**共通点**: これら3つは「今直さないと、後で直すのが10倍面倒になる」タイプの項目。

### 推奨テスト戦略

修正を入れる際、以下の最小限の回帰テストセットを用意すると安心：

| 対象         | テスト内容                                                                                                                     | 目的            |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| **DTT XML**  | 既知の複素伝達関数XML（LIGO公開サンプル等）を読み込み、位相が 0°/180° 付近で正しく復元されるか確認                             | 虚部保持の検証  |
| **MCMC GLS** | ① 固定Σで事前分布が正しくサンプリングされるか（既存テスト強化）<br>② 可変Σを意図的に渡した場合に `AssertionError` が発生するか | 固定Σ制約の強制 |
| **PCA/ICA**  | ダミーデータ `(channels=3, cols=4, time=1000)` → PCA → inverse → 元データと一致（数値誤差以内）か                              | 形状復元の検証  |

### ドキュメント「予防線」強化ポイント

gwexpyは重力波解析ツールとして専門性が高いため、以下の予防線をdocstring/APIドキュメントに追加しておくと誤解やissueが減る：

| 対象                    | 追加すべき記述                                                                             | 理由                          |
| ----------------------- | ------------------------------------------------------------------------------------------ | ----------------------------- |
| `_fft_transient`        | 「振幅スペクトルです（密度スペクトルではありません）」                                     | AI/ユーザー双方が混同しやすい |
| `ScalarField.fft_space` | 「k は角波数 [rad / length] です」                                                         | サイクル波数との混同防止      |
| `run_mcmc`              | 「共分散行列がパラメータに依存しないことを前提としています」                               | 将来の誤用防止                |
| `pca_fit` / `ica_fit`   | 「inverse_transform は cols=1 を想定しています（複数列の場合はメタデータ保存が必要です）」 | 3D復元問題の明示              |

**備考**: これらは「AIが混同しやすいポイント」と「物理屋が引っかかりやすいポイント」の両方をカバー。

---

## 補足分析（Felo 2026-02-01）

### 1. 今回の監査の性質と限界

今回の検証は、提供されたソースコードの断片とアルゴリズム記述に基づいた**静的な分析**です。LLMとしての分析には以下の限界があります：

| 限界の種類               | 説明                                                                                                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **動的な挙動の確認**     | 実際にコードを実行し、様々な入力データに対する振る舞いや数値的安定性を詳細に確認したわけではない。特定条件下でしか発生しないバグや性能ボトルネックは見つけにくい |
| **運用環境での性能**     | 実際のデータ量や並列処理環境でのスケーラビリティ、メモリ使用量などの運用面は評価していない                                                                       |
| **コード全体のレビュー** | 提供された断片以外の部分（エラーハンドリングの徹底度、ロギング、メタデータ管理など）の詳細な影響は評価できていない                                               |

**結論**: 本レポートは「潜在的な問題点の発見」と「改善の方向性の提示」に焦点を当てている。

### 2. 科学計算ライブラリ特有の品質保証の重要性

gwexpyのような科学計算ライブラリでは、単に「コードが動作する」だけでなく、**「計算結果が物理的に、統計的に正しい」**ことが極めて重要です。

| 通常のソフトウェア | 科学計算ライブラリ                            |
| ------------------ | --------------------------------------------- |
| バグ → 機能不全    | バグ → **誤った科学的結論**、**現象の誤解釈** |

そのため、以下の指摘事項は単なるプログラミング上のミスではなく、**計算結果の信頼性や物理的妥当性そのものに関わる重要な課題**として捉えるべき：

- GLS尤度の正規化項の欠落（P1-1）
- PCA逆変換のデータ構造の不整合（P1-2）
- PSD統計分布の仮定の誤り（SigmaThreshold）

### 3. 文書化のさらなる重要性

今回の監査を通じて特に強く感じたのは、**「アルゴリズムの仮定と適用範囲」を明確に文書化することの重要性**です。

多くのアルゴリズムは特定の物理的・統計的仮定（例：非コヒーレントな信号源、線形応答、データが特定の統計分布に従うなど）に基づいて設計されています。これらの仮定が明確でないと、利用者が意図しない状況でアルゴリズムを適用し、誤った結果を導き出すリスクがあります。

**推奨される文書化の内容**:

| 項目                             | 内容                                                             |
| -------------------------------- | ---------------------------------------------------------------- |
| **理論的背景と主要な仮定**       | どのような物理・統計モデルに基づいているか                       |
| **適用範囲と限界**               | どのようなデータ、どのような状況での使用が適切か。不適切な使用例 |
| **利用している数式の正確な定義** | 特にスケーリングや正規化の規約                                   |
| **主要なパラメータの物理的意味** | 各パラメータが何を表し、どのような影響を与えるか                 |

---

## 参考文献一覧

### 教科書・専門書

| 略記                     | 文献                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Bendat & Piersol (2010)  | Bendat, J.S. & Piersol, A.G. _Random Data: Analysis and Measurement Procedures_ (4th ed.). Wiley, 2010.               |
| Gelman (2013)            | Gelman, A. et al. _Bayesian Data Analysis_ (3rd ed.). CRC Press, 2013.                                                |
| Jackson (1998)           | Jackson, J.D. _Classical Electrodynamics_ (3rd ed.). Wiley, 1998.                                                     |
| Jolliffe (2002)          | Jolliffe, I.T. _Principal Component Analysis_ (2nd ed.). Springer, 2002.                                              |
| Oppenheim (2010)         | Oppenheim, A.V. & Schafer, R.W. _Discrete-Time Signal Processing_ (3rd ed.). Prentice Hall, 2010.                     |
| Percival & Walden (1993) | Percival, D.B. & Walden, A.T. _Spectral Analysis for Physical Applications_. Cambridge University Press, 1993.        |
| Press et al. (2007)      | Press, W.H. et al. _Numerical Recipes: The Art of Scientific Computing_ (3rd ed.). Cambridge University Press, 2007.  |
| Rasmussen (2006)         | Rasmussen, C.E. & Williams, C.K.I. _Gaussian Processes for Machine Learning_. MIT Press, 2006.                        |
| Smith (1997)             | Smith, S.W. _The Scientist and Engineer's Guide to Digital Signal Processing_. California Technical Publishing, 1997. |

### 査読論文

| 略記                    | 文献                                                                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Harris (1978)           | Harris, F.J. "On the use of windows for harmonic analysis with the discrete Fourier transform." _Proc. IEEE_ 66(1), 51–83, 1978.               |
| Künsch (1989)           | Künsch, H.R. "The jackknife and the bootstrap for general stationary observations." _Ann. Statist._ 17(3), 1217–1241, 1989.                    |
| Politis (1994)          | Politis, D.N. & Romano, J.P. "The stationary bootstrap." _J. Amer. Statist. Assoc._ 89(428), 1303–1313, 1994.                                  |
| Welch (1967)            | Welch, P.D. "The use of fast Fourier transform for the estimation of power spectra." _IEEE Trans. Audio Electroacoust._ AU-15(2), 70–73, 1967. |
| Kessy et al. (2018)     | Kessy, A., Lewin, A. & Strimmer, K. "Optimal whitening and decorrelation." _Amer. Statistician_ 72(4), 309–314, 2018.                          |
| Torrence & Compo (1998) | Torrence, C. & Compo, G.P. "A practical guide to wavelet analysis." _Bull. Amer. Meteorol. Soc._ 79(1), 61–78, 1998.                           |
| GWpy (2021)             | Duncan Macleod et al. "GWpy: A Python package for gravitational-wave astrophysics." _SoftwareX_ 13, 100657, 2021.                              |

### 標準規約・技術文書

| 略記              | 文献                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| IEEE 754-2019     | IEEE Std 754-2019. _IEEE Standard for Floating-Point Arithmetic_.                                                  |
| LIGO-T980044      | LIGO Technical Note T980044. GPS time conventions.                                                                 |
| NumPy docs        | NumPy `fftfreq` documentation. https://numpy.org/doc/stable/reference/routines.fft.html                            |
| SciPy docs        | SciPy `signal.welch` / `fft.rfft` documentation. https://docs.scipy.org/doc/scipy/reference/                       |
| scikit-learn docs | scikit-learn `PCA` documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html |

---

_統合レポート作成: Claude Code IDE (claude-opus-4-5-20251101), 2026-02-01_
_追加アルゴリズム統合: 2026-02-01_
_参考文献強化: 2026-02-01（Perplexity参考文献強化版を統合）_
_入力: 12種のAIモデルによる独立監査結果（基本6領域 + 拡張監査）_
