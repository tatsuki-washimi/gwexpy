## 以下は、提示された gwexpy の該当アルゴリズム断片（および記述）に対する物理・統計妥当性／数値安定性の監査レポートです。指示に従い、コード全面改変やファイル置換は行わず、必要なら短い式・スニペット程度の提案に留めます。

検証結果サマリ（重要度順）
領域 主な結論 重大度 4) MCMC（GLS尤度） 正規分布GLSの対数尤度として正規化項が欠落。パラメタ依存共分散なら推論が誤る High 2) \_fft_transient 振幅保存 “振幅スペクトル”としては整合し得るが、PSD/ASDやParseval整合の前提が曖昧。Ifft復元のスケーリング整合要確認 Medium 3) VIF & ブートストラップ VIF式の形は妥当だが、Welch平均の自由度（χ²分布）とブートストラップ設計が混在し、CI解釈がズレ得る Medium 1) k空間変換（ScalarField.fft_space） k = 2π fftfreq(n,d)は正しい。dx<0で符号反転は座標向きの扱いとして一貫し得るが、慣習的には注意点あり Low〜Medium 5) ARIMA→GPS TimeSeries forecast_t0 = t0 + n_obs\*dtは概ね正しい。欠損・不規則サンプリングや内部インデックス差でズレる危険 Low〜Medium 6) PCA/ICAのflatten/復元 flattenは合理的だが、逆変換で X_rec_3d = X_rec_val[:, None, :] は cols次元を常に1に潰しているように見え、一般形で不整合 Medium〜High

---

1. Physical Fields & k-space（ScalarField.fft_space）
   Finding 1.1 — k = 2π \* fftfreq(n, d=|dx|) 自体は物理的に正しい
   Severity: Low

- 離散フーリエ変換に対する波数グリッドとして
  [ k_m = 2\pi ,\mathrm{fftfreq}(n, d)\quad (d=\Delta x) ] は標準的で、単位も dx が [m] なら k は [rad/m] になり整合します。
- ifft_space で dx = 2π/(n\*|dk|) としているのも、dk = 2π/(n dx) の関係を再構成しており筋が通っています。
  Recommendation
- ドキュメント上で「k は 角波数（rad/length）」であることを明示し、サイクル波数（1/length）と混同しない注意書きを入れると事故が減ります。
  Finding 1.2 — dx_value < 0 で k を反転する設計は一貫性はあるが注意点
  Severity: Medium
- 座標軸が降順（負のステップ）というメタデータを許す設計なら、k符号を反転して「物理座標の向き」を保持しようとする意図は理解できます。
- ただし、FFTの離散化は配列インデックス順序に対して定義されるため、負のdxを“物理軸の向き”として吸収するのは慣習から外れやすく、他ライブラリ（xarray等）との相互運用で混乱要因になります。
- 典型的には「データ配列を物理座標の昇順に並べ替える」か、「座標軸情報のみ負のdxを許すが FFTは|dx|で定義」など、方針を固定します。
  Recommendation
- dx<0 を許すなら、少なくとも以下を仕様化：
- value の格納順と座標軸（metadata）の対応
- k の符号反転により、実空間での位相因子 exp(+ikx)/exp(-ikx) のどちらの規約に合うか
- 互換性の観点では、「FFTは d=abs(dx)」で定義し、別途“座標の向き”はメタデータとして扱う方が誤用が減ります（コードを書き換えろという意味ではなく、仕様明確化の推奨）。

---

2. Transient Response Analysis
   2.1 自動ステップ検出（detect_step_segments）
   FINDING 2.1 — 周波数追跡の量子化・窓長依存で分割が不安定になり得る
   Severity: Medium

- スペクトログラムのピーク周波数は FFT ビンに量子化され、分解能は (\Delta f = 1/T\_{\mathrm{fft}})。
  freq_tolerance の設定が (\Delta f) 未満だと、ノイズ・ビン飛びで偽の「周波数変更」を検出しやすいです。
- SNR閾値でゲートしていても、ステップ境界では過渡応答・リーケージが増え、ピーク追跡が破綻しやすい（短い窓ほど顕著）。
  Recommendation
- 許容幅を (\Delta f) と整合させる（例：freq_tolerance >= 1/Tfft を目安）。
- 境界判定に「連続k点で変化が持続」などのヒステリシスを導入する設計指針を明記（実装提案ではなく、アルゴリズム妥当化の要件として）。
  2.2 \_fft_transient の振幅保存
  FINDING 2.2 — “1/Nで割って片側2倍”は振幅スペクトルとしては妥当だが、用途が限定される
  Severity: Medium
- 実信号の rFFT を片側化する際の「非DC/非Nyquistを2倍」は、片側の複素振幅を両側に対応付ける操作として一般的です。
- ただし本実装は dft = rfft(x)/N としており、これは多くの数値ライブラリの「前方FFTはスケールしない」規約とは異なるスケーリングです。これ自体は悪ではありませんが、以下が未記載だと混乱します：
- これは 振幅スペクトル（時間領域の正弦波振幅に一致させたい等）なのか
- あるいは PSD/ASD推定の前処理なのか（その場合スケールが異なる）
- ifft 側で pad と片側補正をどう戻すか（Parseval整合含む）
  Recommendation
- 仕様として、少なくとも次を明文化：
- 本DFTの定義：(\tilde x_k = \frac{1}{N}\sum_n x_n e^{-i2\pi kn/N}) を採用している等
- 片側補正を入れた後の量が「片側振幅（peak）」なのか「RMS」なのか
- チェック項目（テスト観点）として、単一正弦波 (x[n]=A\sin(2\pi f_0 n/fs)) を入れたとき、該当ビンの片側振幅が期待値（例：A）になるかを確認するのが有効です。

---

3. Robust Statistics（VIF・block bootstrap）
   3.1 VIF（重なり窓による分散増加）
   FINDING 3.1 — 提示式は概ね標準的だが、前提（WELCH平均・窓・ブロック長）を明確化すべき
   Severity: Medium

- 記載の [ \mathrm{VIF}=\sqrt{1+2\sum_{k=1}^{M-1}\left(1-\frac{k}{M}\right)|\rho(kS)|^2} ] は、相関を持つ平均の分散膨張として妥当な形です（(\rho) が窓で決まる自己相関、Sがシフト）。
- ただし Welch PSD は本来、各セグメントの periodogram が（近似的に）スケールドχ²で、平均化数（有効自由度）がCIに直接効きます。VIFとブートストラップCIを同時にやる場合、「どの不確かさを補正しているのか」が曖昧になる恐れがあります。
  Recommendation
- VIF補正を「標準誤差の補正」目的に限定するのか、ブートストラップCIにも組み込むのか、運用を整理。
- 文献ベースでの位置付け：Welchの有効自由度（equivalent degrees of freedom; EDF）とオーバーラップの扱いを参照し、VIFが近似である旨を注記。
  3.2 bootstrap_spectrogram のブロック設計
  FINDING 3.2 — MOVING BLOCK BOOTSTRAPの「ブロック→再標本化」が時間相関には有効だが、端点・定常性仮定に注意
  Severity: Medium
- 時間方向に相関があるスペクトログラム（特に重なり窓Welch）に対して、ブロックブートストラップは合理的です。
- 一方で、非定常（ドリフト・トレンド・注入オン/オフが混じる等）のとき、ブロック再標本化は分布を歪めます。
- block_size が短すぎると相関を壊し、長すぎると有効標本が不足してCIが不安定。
  Recommendation
- block_size 選定指針を提示（例：相関長の数倍、あるいは nperseg-noverlap と整合）。
- 端点効果（循環ブロックかどうか）や、ブロック抽出可能数 n_time - block_size + 1 が小さい場合の警告を仕様化。

---

4. Bayesian Fitting（run_mcmc のGLS対数尤度）
   Finding 4.1 — GLSの対数尤度として「正規化項（log det Σ）」が欠落
   Severity: High

- 現在の log_prob は（共分散逆行列がある場合） [ \chi^2 = r^\dagger \Sigma^{-1} r,\quad \log p \propto -\frac12\chi^2 ] の形です。
- これは Σがパラメタに依存しない（固定）なら、事後分布の形（最尤点や相対確率）に対して定数差なので問題になりにくいです。
- しかし Σ（またはスケール）がパラメタに依存するモデル（例：未知ノイズ振幅、ハイパーパラメタ、周波数依存ノイズモデルを推定）では、正しいガウス尤度 [ \log \mathcal{L} = -\frac12\left(r^\dagger \Sigma^{-1} r + \log\det \Sigma + N\log 2\pi\right) ] の (\log\det\Sigma) を落とすと推定が系統的に誤ります。
  Recommendation
- 「cov_invは固定である」ことを明示し、もし可変なら (\log\det\Sigma) を含めるべき、という仕様上の要件を追加。
- 数値安定性の観点では、(\log\det\Sigma) は slogdet や Cholesky 分解で扱うのが標準（ここでは実装提案ではなく、計算方針の推奨）。
  Finding 4.2 — 複素残差の扱いは要注意（統計モデルの定義が必要）
  Severity: Medium
- np.real(r.conj() @ cov_inv @ r) としてエルミート形式の実部を取るのは、数値誤差で微小虚部が出るケースには合理的。
- ただし、そもそも複素データの確率モデル（複素正規、実虚独立など）により係数が変わる場合があります。
  Recommendation
- 複素フィットを許すなら「複素ガウス雑音（circular complex Gaussian）を仮定」等を明記。

---

5. Time Series Modeling（ARIMA→GPS-aware TimeSeries）
   Finding 5.1 — forecast_t0 = t0 + n_obs\*dt は等間隔前提で妥当だが、欠損があるとズレる
   Severity: Low〜Medium

- 観測が完全に等間隔・欠損なしなら、予測開始GPSはその式で整合します。
- しかし、ARIMA実装が内部で欠損処理（drop/補間）や、インデックス周波数推定を行うと、n_obs が「元のサンプル数」と一致しない可能性があります。
  Recommendation
- 予測の開始時刻は「モデルが認識している最終時刻 + dt」に基づく、という仕様を明確化（n_obs の定義を固定）。
- 欠損がある入力に対しては、予測TimeSeriesの時刻軸が元データのGPSと一致することを確認するテスト項目を推奨。

---

6. Dimensionality Reduction（PCA/ICA flatten・復元）
   Finding 6.1 — 逆変換で cols 次元を固定1にしており、一般の3D構造が復元できない可能性
   Severity: Medium〜High

- fit側で X_proc.value.reshape(-1, time) として (channels, cols) を feature 次元に畳み込むのは妥当です。
- しかし inverse 側で X_rec_3d = X_rec_val[:, None, :] としており、これは feature 次元を (features, 1, time) に戻しているように見えます。元が (channels, cols, time) で cols>1 の場合、形状復元が不整合になります。
- もし設計として cols を常に1に限定するなら問題ありませんが、その場合はAPI上の TimeSeriesMatrix の一般性と齟齬が出ます。
  Recommendation
- 「flatten時に保存した元shape（channels, cols）」をメタデータとして保持し、逆変換でそのshapeに戻す、という要件を仕様に追加（ここでは概念提案）。
- 少なくとも、cols>1入力時に例外/警告を出すか、仕様で未対応と明記。

---

追加観察（参考：断片から見える軽微論点）
FastCoherenceEngine（Welch coherence）
Severity: Low

- スケール 2/(fs\*sum(w^2)) は片側PSD密度の典型形ですが、厳密にはDC/Nyquistの扱いが異なる場合があり、scipyと完全一致を要求するなら差が出得ます。
- coherence は比なので多くのスケール誤差は相殺されますが、CSD/PSDの推定が同一スケールであることが重要で、その点は満たしています。
  Recommendation
- DC/Nyquistの片側係数（×1か×2か）をscipyの定義と突き合わせ、必要なら「近似一致」か「完全一致」かを仕様化。

---

結論（監査所見）

- 物理座標系FFT（k空間）やWelch系の基本ロジックは概ね標準に沿っています。
- 一方で、MCMCのGLS尤度が“正しいガウス尤度”として一般に不完全（logdet項欠落）であり、将来の拡張や共分散推定を含む解析で高リスクです（High）。
- \_fft_transient と PCA復元は「用途・前提」を仕様として固定しないと、利用者が期待する物理量（振幅/PSD/ASD、3D形状復元）とズレる可能性が高いです（Medium〜High）。
  必要なら、上記High/Medium項目について「どの前提なら現状でも正当化できるか（固定共分散、cols=1限定、等）」を整理した妥当化条件リスト（チェックリスト形式）も作成できます。






追加検証：コードベース内の他の独自アルゴリズムに対する監査所見
提示された文書を再精査し、初回レポートで詳細に触れなかった独自アルゴリズムについて追加検証を行いました。
----------------------------------------
7. Coupling Function Estimation（結合関数推定）
7.1 結合関数の定義と物理的妥当性
FINDING 7.1 — √((ΔP_TGT)/(ΔP_WIT)) の物理的解釈が曖昧
Severity: Medium
問題点：
# gwexpy/analysis/coupling.py
delta_wit = psd_wit_inj.value - psd_wit_bkg.value
delta_tgt = psd_tgt_inj.value - psd_tgt_bkg.value
with np.errstate(divide="ignore", invalid="ignore"):
 sq_cf = delta_tgt[valid_mask] / delta_wit[valid_mask]
 cf_values[valid_mask] = np.sqrt(sq_cf)
 * この定義は「witness チャネルのパワー変化に対する target チャネルのパワー変化の比の平方根」です。
 * 物理的には、線形結合関数 H(f) を仮定すると、パワーは振幅の二乗なので： [ P_{\mathrm{tgt}} = |H(f)|^2 P_{\mathrm{wit}} ] したがって結合関数の振幅は： [ |H(f)| = \sqrt{\frac{P_{\mathrm{tgt}}}{P_{\mathrm{wit}}}} ]
 * しかし、差分 ΔP を使う場合、これは「注入による増分」を前提としており、以下の仮定が暗黙的に必要です：
 1. 背景ノイズが注入前後で変化しない（定常性）
 2. 注入信号が witness と target で線形結合関係にある
 3. 背景ノイズと注入信号が無相関
問題となるケース：
 * 背景ノイズ自体が非定常（ドリフト、グリッチ）
 * 注入が非線形応答を引き起こす
 * ΔP_wit が負またはゼロに近い（分母が不安定）
Recommendation:
 * 仕様書に以下を明記：
 * 「線形時不変系（LTI）を仮定」
 * 「注入振幅が背景ノイズに対して十分小さい（線形応答領域）」
 * 「定常背景ノイズを仮定」
 * 数値安定性：delta_wit の最小閾値（例：背景の1%以上）を設けることを推奨
 * 不確かさ伝播：ΔP の統計誤差を考慮した CF の信頼区間推定が望ましい
7.2 閾値判定戦略の統計的妥当性
FINDING 7.2 — SIGMATHRESHOLD の中心極限定理仮定が不適切な場合がある
Severity: Medium
class SigmaThreshold(ThresholdStrategy):
 """
 Threshold = mean + sigma * (mean / sqrt(n_avg))
 Assumes Gaussian distribution of PSD values (Central Limit Theorem).
 """
 def check(self, psd_inj, psd_bkg, raw_bkg=None, **kwargs):
 n_avg = kwargs.get("n_avg", 1.0)
 factor = 1.0 + (self.sigma / np.sqrt(n_avg))
 return psd_inj.value > (psd_bkg.value * factor)
問題点：
 1. Welch PSD の統計分布は χ² であり、正規分布ではない
 * 各周波数ビンの periodogram は指数分布（自由度2のχ²）
 * n_avg 個の平均は自由度 2×n_avg の χ² 分布
 * 中心極限定理が効くには n_avg が十分大きい必要（通常 n≥30 が目安）
 2. 標準誤差の式が不正確
 * 正しくは、χ² 分布（自由度 ν=2n_avg）の標準偏差は： [ \sigma_{\mathrm{PSD}} = \frac{\mu}{\sqrt{n_{\mathrm{avg}}}} ]
 * コード中の mean / sqrt(n_avg) は形式的には一致しているが、係数が χ² 分布の性質から導出されるべき
 3. 片側検定か両側検定か不明確
 * 「注入により増加」を検出するなら片側検定が適切
 * sigma の解釈（標準偏差の何倍か）と有意水準 α の対応が不明
Recommendation:
 * χ² 分布に基づく閾値を使用： [ \mathrm{Threshold} = \mu_{\mathrm{bkg}} \times \frac{\chi^2_{\alpha}(2n_{\mathrm{avg}})}{2n_{\mathrm{avg}}} ] ここで χ²_α は有意水準 α の χ² 分位点
 * または、n_avg が大きい場合（≥30）に限定して正規近似を使用し、その条件を明記
 * 文献参照：Bendat & Piersol "Random Data" の Welch 法の統計的性質
FINDING 7.3 — PERCENTILETHRESHOLD の生スペクトログラム使用は妥当だが計算コストに注意
Severity: Low
class PercentileThreshold(ThresholdStrategy):
 def threshold(self, psd_inj, psd_bkg, raw_bkg=None, **kwargs):
 spec = raw_bkg.spectrogram(...)
 p_bkg_values = spec.percentile(self.percentile)
 return p_bkg_values.value * self.factor
観察：
 * 経験的分布からパーセンタイルを取る方法は、分布の仮定に依存しないため頑健
 * ただし、各周波数ビンで独立にパーセンタイルを計算しており、多重比較の問題（False Discovery Rate）が生じ得る
Recommendation:
 * 多重比較補正（Bonferroni, FDR など）の必要性を検討
 * 計算コスト：スペクトログラム全体を毎回計算するのは非効率なので、キャッシュ機構の検討
----------------------------------------
8. Noise Models（ノイズモデル）
8.1 Schumann Resonance モデル
FINDING 8.1 — PSD加算（非コヒーレント和）の物理的前提
Severity: Low
# gwexpy/noise/magnetic.py
for f0, Q, A in modes:
 peak_asd_series = lorentzian_line(f0, A * amplitude_scale, Q=Q, ...)
 total_psd += peak_asd_series.value**2
total_asd = np.sqrt(total_psd)
観察：
 * 各モード（共鳴ピーク）の ASD を二乗してから加算し、最後に平方根を取る
 * これは 非コヒーレント（位相ランダム）な複数源の合成 を仮定
物理的妥当性：
 * Schumann 共鳴の各モードが独立な励起源（世界中の雷活動）から来る場合、位相はランダムなので PSD 加算は妥当
 * ただし、同一励起源から複数モードが出る場合（モード間の位相相関）は、厳密には複素振幅の加算が必要
Recommendation:
 * 現在の実装は「標準的な Schumann モデル」として妥当
 * ドキュメントに「各モードは非コヒーレントと仮定」と明記
 * より精密なモデル（モード間相関を含む）が必要な場合は、複素振幅での合成を別途実装
8.2 Voigt Profile の正規化
FINDING 8.2 — ピーク振幅正規化の数値安定性
Severity: Low
# gwexpy/noise/peaks.py
z = ((f_vals - f0) + 1j * gamma) / (sigma * np.sqrt(2))
v = wofz(z).real
z0 = (1j * gamma) / (sigma * np.sqrt(2))
peak_factor = wofz(z0).real
data = amp_val * (v / peak_factor)
観察：
 * Faddeeva 関数 wofz を使った Voigt プロファイルの実装は標準的
 * ピーク位置（f=f0）での値で正規化してピーク振幅を amplitude に合わせる
潜在的問題：
 * peak_factor がゼロに近い場合（極端なパラメタ）に数値不安定
 * sigma → 0（純粋 Lorentzian）または gamma → 0（純粋 Gaussian）の極限での挙動
Recommendation:
 * peak_factor の最小値チェック（例：peak_factor < 1e-10 で警告）
 * 極限ケースのテスト：
 * sigma ≫ gamma：Gaussian に収束
 * gamma ≫ sigma：Lorentzian に収束
----------------------------------------
9. Signal Processing Utilities
9.1 DTT Normalization の物理的整合性
FINDING 9.1 — ENBW 定義の違いによる混乱リスク
Severity: Medium
# gwexpy/signal/normalization.py
def get_enbw(window, fs, mode="standard"):
 if mode == "dtt":
 # DTT definition: (fs / N) * (1 / mean(w)^2)
 return (fs * n) / (sum_w**2)
 # Standard: fs * sum(w^2) / sum(w)^2
 return fs * sum_w2 / (sum_w**2)
観察：
 * 2つの ENBW（Equivalent Noise Bandwidth）定義が混在
 * DTT（LIGO診断ツール）と標準的な信号処理の定義が異なる
物理的意味：
 * 標準 ENBW：理想的な矩形フィルタで同じノイズパワーを通す帯域幅 [ \mathrm{ENBW}_{\mathrm{std}} = f_s \frac{\sum w_n^2}{(\sum w_n)^2} ]
 * DTT ENBW：異なる正規化規約（詳細は LIGO 内部文書に依存）
問題点：
 * 変換係数 convert_scipy_to_dtt が正しいかは、DTT の内部実装に依存
 * ユーザーが誤った mode を選ぶと、PSD の絶対値が大きくずれる
Recommendation:
 * DTT モードの使用条件を明確化（「LIGO DTT との比較専用」など）
 * 変換の検証：既知の信号（白色ノイズ、正弦波）で DTT ツールと数値比較
 * 可能なら LIGO の公式文書を引用
9.2 Imputation の max_gap 制約
FINDING 9.2 — 大ギャップ内の補間値を NAN に戻すロジックの境界条件
Severity: Low
# gwexpy/signal/preprocessing/imputation.py
if has_gap_constraint and len(valid_indices) > 1:
 valid_times = x[valid_indices]
 big_gaps = np.where(np.diff(valid_times) > gap_threshold)[0]
 for idx in big_gaps:
 t_start, t_end = valid_times[idx], valid_times[idx+1]
 val[(x > t_start) & (x < t_end)] = np.nan
観察：
 * 補間後に、大きなギャップ内の値を NaN に戻す処理
 * 境界条件：x > t_start と x < t_end は開区間
潜在的問題：
 * ギャップの境界点（t_start, t_end）自体は有効データなので、その直近の補間点が残る可能性
 * 数値誤差により x == t_start が厳密に一致しない場合の挙動
Recommendation:
 * 境界の扱いを明確化（開区間か閉区間か）
 * 浮動小数点比較の許容誤差を考慮（np.isclose など）
 * エッジケースのテスト：ギャップが1サンプルだけの場合など
----------------------------------------
10. Hilbert-Huang Transform (HHT)
10.1 EMD（経験的モード分解）の収束性
FINDING 10.1 — EMD の停止条件と数値安定性（概念的指摘）
Severity: Medium
背景：
 * 文書には HHT の概念的記述のみで詳細実装は未提示
 * EMD は反復的アルゴリズムで、以下の問題が知られている：
 1. 停止条件の曖昧さ：IMF（固有モード関数）の定義を満たすまで反復するが、厳密な収束判定が難しい
 2. エンドポイント効果：信号の端点でスプライン補間が発散しやすい
 3. モード混合：異なる周波数成分が1つの IMF に混入
Recommendation:
 * 停止条件の明確化：
 * Cauchy 型収束（連続する反復での変化が閾値以下）
 * 最大反復回数の設定
 * エンドポイント処理：
 * ミラーリング、外挿、窓関数などの手法を選択可能に
 * モード混合の検出：
 * 各 IMF の瞬時周波数の統計的検証
 * 参考文献：Huang et al. (1998) の原論文、および EEMD（Ensemble EMD）などの改良手法
10.2 Hilbert Spectral Analysis の物理的解釈
FINDING 10.2 — 瞬時周波数の定義と負周波数の扱い
Severity: Low
観察：
 * Hilbert 変換により解析信号を構成し、瞬時振幅・位相・周波数を抽出
 * 瞬時周波数は位相の時間微分：( f_{\mathrm{inst}}(t) = \frac{1}{2\pi}\frac{d\phi}{dt} )
潜在的問題：
 * 負の瞬時周波数：位相が減少する場合に発生し、物理的解釈が困難
 * 急激な位相変化：ノイズやグリッチで瞬時周波数が発散
Recommendation:
 * 負周波数の処理方針を明記（絶対値を取る、除外する、など）
 * 瞬時周波数の平滑化（移動平均など）を検討
 * 物理的に意味のある範囲（例：0 < f < Nyquist）でのフィルタリング
----------------------------------------
11. Hurst Exponent 推定
11.1 局所 Hurst 指数の窓長依存性
FINDING 11.1 — 窓長とスケーリング範囲の整合性
Severity: Medium
# gwexpy/timeseries/hurst.py
def local_hurst(timeseries, window, step=None, ...):
 for i, s in enumerate(starts):
 segment_val = x_full[s:e]
 h = hurst(MockTS(segment_val), ...)
 H_vals[i] = h
背景：
 * Hurst 指数は自己相似性（フラクタル性）の指標で、スケーリング解析（R/S 解析、DFA など）で推定
 * 窓長が短すぎると、スケーリング範囲が不足して推定が不安定
 * 窓長が長すぎると、非定常性の局所変化を捉えられない
問題点：
 * window パラメタと内部の Hurst 推定アルゴリズムのスケール範囲が整合しているか不明
 * 例：DFA では最小スケールが窓長の 1/10 程度必要
Recommendation:
 * 窓長の最小値を推定手法に応じて設定（例：DFA なら最低 100 サンプル）
 * スケーリング範囲の自動選択または明示的指定
 * 推定誤差（信頼区間）の評価：ブートストラップや理論的分散
11.2 非定常性の影響
FINDING 11.2 — トレンド除去の必要性
Severity: Medium
観察：
 * Hurst 指数は定常過程を前提とする統計量
 * 非定常（トレンド、ドリフト）があると、見かけ上の長期相関が増加し H > 0.5 となる
Recommendation:
 * 各窓でのトレンド除去（線形、多項式）をオプション化
 * DFA（Detrended Fluctuation Analysis）の使用を推奨（トレンド除去が組み込まれている）
 * 非定常性の事前検定（ADF テストなど）
----------------------------------------
12. 全体的な数値安定性とエッジケース
12.1 ゼロ除算・オーバーフロー対策
FINDING 12.1 — NP.ERRSTATE の使用は対症療法的
Severity: Low〜Medium
観察：
 * 複数箇所で np.errstate(divide="ignore", invalid="ignore") を使用
 * これは警告を抑制するだけで、根本的な数値問題を解決しない
問題となるケース：
# coupling.py
with np.errstate(divide="ignore", invalid="ignore"):
 sq_cf = delta_tgt[valid_mask] / delta_wit[valid_mask]
 cf_values[valid_mask] = np.sqrt(sq_cf)
 * delta_wit がゼロに近い場合、sq_cf が inf になる
 * sq_cf が負の場合、sqrt が NaN になる
Recommendation:
 * 事前検証：分母の最小閾値チェック
 valid = (delta_wit > threshold) & (delta_tgt > 0)
 * クリッピング：極端な値を制限
 sq_cf = np.clip(delta_tgt / delta_wit, 0, max_cf**2)
 * ログ記録：無効値の発生頻度を監視
12.2 浮動小数点精度の限界
FINDING 12.2 — GPS 時刻の高精度演算
Severity: Low
観察：
 * GPS 時刻は大きな整数（10^9 秒オーダー）+ 小数部
 * t0 + n*dt のような演算で、浮動小数点誤差が蓄積
Recommendation:
 * 高精度時刻演算には astropy.time.Time や datetime64[ns] を使用
 * または、基準時刻からの相対時刻（小さな数）で計算し、最後に GPS に変換
----------------------------------------
13. 統計的検定の多重比較問題
13.1 周波数ビン全体での閾値判定
FINDING 13.1 — FALSE DISCOVERY RATE の制御が必要
Severity: Medium
背景：
 * 結合関数推定や応答関数解析で、多数の周波数ビン（数百〜数千）で独立に閾値判定
 * 有意水準 α=0.05 で各ビンを検定すると、偽陽性が多数発生
例：
 * 1000 ビンで α=0.05 なら、期待される偽陽性は 50 個
Recommendation:
 * Bonferroni 補正：α_corrected = α / N_bins（保守的）
 * FDR 制御：Benjamini-Hochberg 法（より検出力が高い）
 * または、「連続する k 個のビンで閾値超過」などのクラスタリング条件
----------------------------------------
14. ドキュメント・テストの推奨事項
14.1 単体テストの充実
推奨テストケース：
 1. 既知の解析解との比較
 * 単一正弦波 → FFT → 振幅・位相の検証
 * 白色ノイズ → PSD → 理論値との一致
 * AR(1) 過程 → Hurst 指数 → 理論値
 2. エッジケース
 * 空配列、単一サンプル
 * 全 NaN、全ゼロ
 * 極端なパラメタ（σ→0, Q→∞ など）
 3. 数値安定性
 * 条件数の悪い行列（GLS）
 * 非常に大きい/小さい値（オーバーフロー/アンダーフロー）
 4. 往復変換
 * FFT → IFFT → 元信号の復元
 * PCA → inverse_transform → 元データの復元
14.2 物理単位の一貫性チェック
推奨：
 * astropy.units を活用した自動単位検証
 * 各関数の入出力で期待される単位を docstring に明記
 * 単位不整合時のエラーメッセージ
----------------------------------------
総合的な優先度付き推奨事項
優先度 項目 理由 Critical MCMC の GLS 尤度に logdet 項を追加 パラメタ推定の系統誤差 High PCA 逆変換の 3D 形状復元を修正 データ構造の破壊 High 結合関数推定の物理的前提を文書化 誤用による誤った結論 Medium Welch PSD の χ² 統計に基づく閾値 偽陽性/偽陰性の制御 Medium 多重比較補正の実装 統計的妥当性 Medium Hurst 推定の窓長検証 推定精度 Low DTT 正規化の検証テスト LIGO ツールとの互換性 Low 数値安定性の事前チェック強化 ロバスト性向上
----------------------------------------
この追加監査により、初回レポートでカバーしきれなかった統計的検定の多重比較問題、ノイズモデルの物理的前提、EMD/HHT の数値安定性、Hurst 指数の窓長依存性などが明らかになりました。特に結合関数推定の物理的妥当性と統計的閾値判定の χ² 分布への対応は、科学的結論に直接影響するため、優先的な対応が推奨されます。