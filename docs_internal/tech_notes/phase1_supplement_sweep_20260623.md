# Phase 1 数値ロバスト性 sweep — 追補（roc.py + 網羅性ギャップ26件）

- 実施日: 2026-06-23
- sweep_base_sha: `e59f62e5bcfbe96c82fac909eb07c3ef303c6d44`（探索時 origin/main `81119dd34` から進行。対象ディレクトリは差分ゼロのため探索結果は有効）
- 親レポート: `docs_internal/tech_notes/phase1_numerical_robustness_sweep_20260622.md`
- 既存 issue: #455–#460（テーマ別）, #461（umbrella tracking）
- 手法: マルチエージェント Workflow（finder×3 → 各 finding を敵対的3票検証 [refute-by-default] → 網羅性クリティック）。88 エージェント / 約21分。

## サマリー

- **確定 26 件**（P2=19 / P3=7、うち new_suspect 3 件）
- subsystem 別: statistics=14 / spectral=6 / fitting=6
- **roc.py（最大の取りこぼし）: 確定 9 件**（calculate_roc 6 + evaluate_detection_performance 系 3）
- 棄却（strict bar 未達・borderline）: 2 件（SP3, SP9 — 後述。実在の下位問題だが silent-failure lens で却下、loud crash / 無効化診断）
- **scope freeze 達成**: 対象内（gwexpy/{statistics,spectral,fitting}）完全未着手ファイル **0 件**、deferred 0 件。`__init__.py`×3 のみ out-of-scope。
- severity profile: **典型有効データで silently-wrong な P1 はゼロ**。全件 P2/P3 の縮退入力ロバスト性（親レポートの結論と整合）。verifier は finder 提案の P1 を「縮退入力のため P2」に修正。

## 確定検出（テーマ別）

### G1. statistics/roc.py — calculate_roc / evaluate_detection_performance の入力契約・数値ロバスト性（モジュール全体が縮退入力未対応）
_起票案: 新規 issue（取りこぼしモジュール）_  / 件数: 9

#### [P2] `gwexpy/statistics/roc.py:28-32` (cid=S1a, 2/3 not-refuted)
- class: silent_wrong_result
- repro: `from gwexpy.statistics.roc import calculate_roc; import numpy as np; print(calculate_roc(np.array([0,0,0]), np.array([0.1,0.5,0.9])))  # -> (array([0,1]), array([0,1]), 0.5). Empty positive class silently returns AUC=0.5.`
- fix: Replace the `return np.array([0,1]), np.array([0,1]), 0.5` at L31-32 with an explicit contract: either `raise ValueError('ROC undefined: positive class is empty (n_pos=0)')` or return np.nan for AUC. Do NOT return the meaningless default 0.5, which is indistinguishable from a chance-level real result.

#### [P2] `gwexpy/statistics/roc.py:28-32` (cid=S1b, 3/3 not-refuted)
- class: silent_wrong_result
- repro: `from gwexpy.statistics.roc import calculate_roc; import numpy as np; print(calculate_roc(np.array([1,1,1]), np.array([0.1,0.5,0.9])))  # n_neg=0 -> (array([0,1]),array([0,1]),0.5)`
- fix: Same L31-32 guard as S1a but split the message: raise ValueError('ROC undefined: negative class is empty (n_neg=0)') so the two degenerate cases are distinguishable. Never return 0.5.

#### [P2] `gwexpy/statistics/roc.py:28-32` (cid=(new), 3/3 not-refuted) **[NEW]**
- class: silent_wrong_result
- repro: `from gwexpy.statistics.roc import calculate_roc; import numpy as np; print(calculate_roc(np.array([-1,1,-1,1]), np.array([0.1,0.9,0.2,0.8])))  # -> AUC 0.5. Also labels {1,2}: calculate_roc(np.array([2,1,2,1]),...) -> 0.5`
- fix: At L28-29 the code hard-codes the positive/negative encoding as exactly {1,0}. Either (a) validate that y_true contains only {0,1} and raise ValueError otherwise, or (b) treat any non-positive label as negative (n_neg = np.sum(y_true != 1)) and document the convention. Currently any label not literally 0 is dropped from n_neg, so the common {-1,+1} sklearn-style encoding produces n_neg=0 -> silent

#### [P2] `gwexpy/statistics/roc.py:24` (cid=S1c, 3/3 not-refuted)
- class: silent_wrong_result
- repro: `from gwexpy.statistics.roc import calculate_roc; import numpy as np; print(calculate_roc(np.array([0,1,0,1]), np.array([0.1,0.9,0.2,0.8]), n_points=1))  # fpr=[1.],tpr=[1.],auc=0.0 ; n_points=0 -> empty arrays, auc=0.0`
- fix: Validate n_points at function entry: `if n_points < 2: raise ValueError(f'n_points must be >= 2 for a meaningful ROC curve, got {n_points}')`. With <2 thresholds the trapezoidal integral degenerates (single point -> AUC=0.0; zero points -> empty arrays -> trapz of empty -> 0.0), silently producing a wrong AUC.

#### [P2] `gwexpy/statistics/roc.py:24` (cid=S1d, 3/3 not-refuted)
- class: silent_wrong_result
- repro: `from gwexpy.statistics.roc import calculate_roc; import numpy as np; print(calculate_roc(np.array([0,1,0,1]), np.array([5.0,5.0,5.0,5.0])))  # all-equal scores -> tpr=fpr=all 1.0, auc=0.0`
- fix: Detect the degenerate case where np.max(y_score)==np.min(y_score) (L24): the linspace collapses to a constant threshold and y_score>=thresh is all-True for every point, giving tpr=fpr=1 everywhere and AUC=0.0. Raise ValueError('ROC undefined: all scores are equal') or return np.nan, instead of silently returning 0.0.

#### [P2] `gwexpy/statistics/roc.py:24,70-82` (cid=S1e, 3/3 not-refuted)
- class: silent_nan_propagation
- repro: `from gwexpy.statistics.roc import calculate_roc; import numpy as np; import warnings; warnings.simplefilter('ignore'); print(calculate_roc(np.array([0,1,0,1]), np.array([0.1,np.nan,0.2,0.8])))  # auc=0.0 silently. Mechanism: np.min/np.max return nan -> linspace(nan,nan) all nan -> y_score>=nan all False -> tpr=fpr=0 -> auc=0.0`
- fix: At L24, np.min(y_score)/np.max(y_score) propagate NaN if any score is NaN, producing all-NaN thresholds; every comparison y_score>=nan is False, so tpr=fpr=0 and AUC=0.0 (silently wrong, NOT NaN). Add a NaN check at entry: `if not np.all(np.isfinite(y_score)): raise ValueError('y_score contains NaN/inf')`, or adopt a documented nan_policy. In evaluate_detection_performance (L70,80) np.nanmax alrea

#### [P2] `gwexpy/statistics/roc.py:70,80` (cid=S1f, 2/3 not-refuted)
- class: crash
- repro: `import numpy as np; np.nanmax(np.array([]))  # ValueError: zero-size array to reduction operation fmax which has no identity. Reached via evaluate_detection_performance when method_func returns a map whose .value is empty (e.g. a fully-cropped Spectrogram).`
- fix: At L70 and L80, guard before np.nanmax: if the score map .value has size 0, raise a clear ValueError('method_func returned an empty score map'); and if it is all-NaN, np.nanmax emits a RuntimeWarning and returns NaN which then silently corrupts the AUC (links to S1e). Use np.nanmax inside a check for size>0 and finite content.

#### [P2] `gwexpy/statistics/roc.py:31-32,49-51` (cid=S1g, 2/3 not-refuted)
- class: silent_wrong_result
- repro: `See S1a/S1b. calculate_roc returns the literal 0.5 at L32 for empty class; and L49-51 (trapz over degenerate fpr/tpr) returns 0.0 for the all-equal/NaN/n_points cases (S1c/S1d/S1e). In no case is np.nan returned, so a caller cannot distinguish 'undefined' from a real chance-level/anti-correlated AUC.`
- fix: Unify the contract: undefined ROC (empty class, all-equal scores, NaN scores, n_points<2) must raise ValueError OR return np.nan, never a numeric default (0.5 at L32, 0.0 from L50 trapz). Document the chosen policy in the docstring (L19-23). The current docstring promises an AUC float with no mention of degenerate behavior.

#### [P2] `gwexpy/statistics/roc.py:45-47` (cid=(new), 3/3 not-refuted) **[NEW]**
- class: silent_wrong_result
- repro: `Static: L45 idx=np.argsort(fpr); L46-47 sort fpr and tpr by fpr only. When multiple thresholds share an fpr but differ in tpr, the tie order is arbitrary (argsort is not stable by tpr), so np.trapz at L50 can integrate a non-monotone tpr-vs-fpr curve and slightly mis-estimate AUC.`
- fix: Sort by (fpr, tpr) lexicographically, e.g. idx = np.lexsort((tpr, fpr)), so that for equal FPR the points are ordered by increasing TPR — the standard ROC convention — yielding a correct trapezoidal AUC. Minor magnitude; hardening only.

### G2. statistics — rayleigh_test p=0 床（NaN→p=0）
_起票案: #459 に追加 or 新規_  / 件数: 1

#### [P2] `gwexpy/statistics/rayleigh_test.py:55-59` (cid=S6, 3/3 not-refuted)
- class: silent_wrong_result
- repro: `import numpy as np; from astropy import units as u; from gwexpy.spectrogram import Spectrogram; from gwexpy.statistics.rayleigh_test import rayleigh_pvalue; spec=Spectrogram(np.array([[1.0,np.nan],[0.9,1.1]]), dt=1.0*u.s, f0=10*u.Hz, df=1*u.Hz); print(rayleigh_pvalue(spec, n_samples=10, n_monte_carlo=500).value)  # NaN r_val -> p=0.0 (maximally significant)`
- fix: NaN in r_vals (L51) flows into np.searchsorted at L55/L57: NaN sorts to the end, so upper_counts=len(dist)-len(dist)=0 and lower_counts=len(dist)... actually upper=0 and min(upper,lower)=0 -> p=0.0. A NaN Rayleigh statistic is thus silently reported as p=0.0 (maximally significant), which then triggers a FALSE veto in to_segments (np.any(p<alpha)). Guard: detect ~np.isfinite(r_vals) and set those

### G3. statistics — 再現性（RNG seed / metadata 欠落）
_起票案: 新規 issue（reproducibility lane）_  / 件数: 2

#### [P3] `gwexpy/statistics/rayleigh_test.py:73,88` (cid=S8, 2/3 not-refuted)
- class: reproducibility
- repro: `Static: L73 _RAYLEIGH_STAT_CACHE is module-global; L88 uses np.random.rand(n) with the global NumPy RNG and no seed/Generator parameter. Two runs of rayleigh_pvalue with identical inputs give different p-values (within MC error). The cache also makes the result depend on call order / process-wide prior np.random consumption, and is not thread-safe (concurrent populate of the same key races).`
- fix: Accept an optional rng/seed parameter (np.random.Generator) threaded through rayleigh_pvalue -> _get_rayleigh_stat_null_distribution, default to a documented seed for reproducibility, and use rng.random(n) instead of np.random.rand. Record the seed and n_monte_carlo in the returned Spectrogram metadata. For thread-safety, guard the cache write or document single-threaded use.

#### [P3] `gwexpy/statistics/gauch.py:123-129` (cid=S9, 2/3 not-refuted)
- class: reproducibility
- repro: `Static: GauChResult is constructed at L123-129 recording only fftlength and stride into metadata (**metadata). n_monte_carlo (the Monte-Carlo trial count that determines p-value resolution and is consumed at L106) is NOT passed and therefore not recorded in the result. Combined with the global-RNG draws at L140 (no seed) the GauCh result is non-reproducible and its provenance is incomplete.`
- fix: Pass n_monte_carlo (and a seed/rng) into GauChResult metadata at L123-129, e.g. add `n_monte_carlo=n_monte_carlo` and `seed=seed`. Thread an optional rng/seed through compute_gauch -> _get_rayleigh_lilliefors_pvalue and use rng instead of np.random.rand at L140 for reproducible null distributions.

### G4. statistics — student_t_indicator STFT gwpy 互換・統計妥当性
_起票案: 新規 issue（#451 と検討）_  / 件数: 2

#### [P3] `gwexpy/statistics/student_t_indicator.py:104` (cid=S5, 3/3 not-refuted)
- class: silent_wrong_result
- repro: `Static: L104 samples=np.concatenate([np.real(segments), np.imag(segments)]) treats the real and imaginary parts of the STFT coefficients as 2*window independent iid Student-t samples. For the DC bin (f=0) and the Nyquist bin the STFT coefficient is purely real (imag ~ 0), so concatenating ~window zeros from the imaginary part biases the t-fit (L110) toward a spurious low-variance / wrong-nu estima`
- fix: Exclude the DC (index 0) and Nyquist frequency bins from the real+imag-concatenation treatment, or fit only the real part for those bins. Document that re/im are assumed independent (valid only for 0<f<Nyquist). Also note window default 40 -> only 80 samples per t-fit, which is statistically thin; document the minimum window for a reliable nu estimate.

#### [P3] `gwexpy/statistics/student_t_indicator.py:66-76,116` (cid=S4, 3/3 not-refuted)
- class: silent_wrong_result
- repro: `Static: L66 nfft=int(fftlength*fs); L67 nstep=int(stride*fs); L76 noverlap=nfft-nstep. If stride>fftlength then nstep>nfft and noverlap becomes negative, which scipy.signal.stft rejects (ValueError: noverlap must be less than nperseg). Also L116 out_times = t[window//2 : window//2 + n_out] uses scipy stft segment times t which are seconds-from-zero, NOT GPS — the returned Spectrogram times lose th`
- fix: (1) Validate stride<=fftlength (or document) before L76 to avoid the opaque scipy noverlap error; clamp or raise a clear message. (2) At L116/L118-121, offset out_times by the input ts epoch: out_times = ts.t0.value + t[...] (or carry ts.times) so the resulting Spectrogram is on the GPS axis. (3) scipy.signal.stft uses a Hann window and no detrend by default, differing from gwpy's spectrogram defa

### G5. spectral — bootstrap/correlation 縮退ロバスト性（dtype/rebin/n_boot/covariance/correlation）
_起票案: #460 を拡張 or 新規_  / 件数: 6

#### [P2] `gwexpy/spectral/estimation.py:58, 84` (cid=SP1, 3/3 not-refuted)
- class: data_integrity / dtype truncation
- repro: `import numpy as np; from gwexpy.spectral.estimation import bootstrap_spectrogram; from gwexpy.spectrogram import Spectrogram; from astropy import units as u
np.random.seed(1); arr = np.random.randint(1,4,(40,1))
spec_int = Spectrogram(arr, dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
spec_float = Spectrogram(arr.astype(float), dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
import numpy as np; np.random.seed(2); ri = boot`
- fix: In _bootstrap_resample_jit (L58) and _bootstrap_resample_py (L84) allocate resampled_stats with a float dtype: `dtype=np.result_type(data.dtype, np.float64)` (or simply np.float64). Each per-resample np.mean/np.median/np.nanmean is currently stored into an integer buffer and truncated toward zero before the final center/percentile aggregation, biasing the result.

#### [P3] `gwexpy/spectral/estimation.py:242-260 (NaN-window path), corrected from candidate ~L222-260` (cid=SP2, 3/3 not-refuted)
- class: silent NaN in CI / unguarded non-finite
- repro: `import numpy as np; from gwexpy.spectral.estimation import calculate_correlation_factor, bootstrap_spectrogram; from gwexpy.spectrogram import Spectrogram; from astropy import units as u
print(calculate_correlation_factor(np.full(100, np.nan), 100, 50, 5))  # -> nan
np.random.seed(0); spec = Spectrogram(np.random.rand(20,5)+1, dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
r = bootstrap_spectrogram(spec, n_boo`
- fix: After computing `energy` (L242) and `rho` (L255), guard against non-finite values: if not np.isfinite(energy) or energy==0 return 1.0; and skip/zero non-finite rho terms. At minimum, validate `np.isfinite(vif)` before sqrt (L259-260) and fall back to 1.0 with a warning, so a malformed/NaN window does not silently NaN-out err_low/err_high in bootstrap_spectrogram (L703-704).

#### [P2] `gwexpy/spectral/estimation.py:736-749` (cid=SP4, 3/3 not-refuted)
- class: imputation bias / silent NaN in covariance / no transparency
- repro: `import numpy as np; from gwexpy.spectral.estimation import bootstrap_spectrogram; from gwexpy.spectrogram import Spectrogram; from astropy import units as u
np.random.seed(0); data = np.random.rand(20,4)+1; data[:,2]=np.nan
spec = Spectrogram(data, dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
fs, bfm = bootstrap_spectrogram(spec, n_boot=50, return_map=True, ignore_nan=True)
print(np.isnan(bfm.value).any())`
- fix: Two issues: (a) mean-imputation (L744-748) biases covariance toward zero with no transparency marker/warning - document or switch to masked/pairwise covariance (np.ma or pandas .cov). (b) An all-NaN column makes col_mean[j]=NaN so the fill leaves NaNs and np.cov returns a NaN row/col silently - detect all-NaN columns and warn/raise before np.cov.

#### [P2] `gwexpy/spectral/estimation.py:493-517 (corrected: guard at 493, int trunc at 497, no-op when bin_size<=1 at 499)` (cid=SP5, 3/3 not-refuted)
- class: silent no-op / unguarded non-finite -> crash or silent skip
- repro: `import numpy as np; from gwexpy.spectral.estimation import bootstrap_spectrogram; from gwexpy.spectrogram import Spectrogram; from astropy import units as u
np.random.seed(0); spec = Spectrogram(np.random.rand(20,10)+1, dt=1.0*u.s, f0=0*u.Hz, df=1.0*u.Hz)
bootstrap_spectrogram(spec, n_boot=50, rebin_width=0.5)        # bin_size=int(0.5/1)=0 -> silent no-op, nfreq stays 10, no warning
bootstrap_spe`
- fix: L493: validate np.isfinite(rebin_width) and reject NaN/Inf with ValueError. L497-499: when bin_size<=1 (rebin_width<df), warn that the requested rebin had no effect instead of silently doing nothing; when int() truncation rounds the width down, note the effective width. L503-505: truncation that drops trailing bins should also warn.

#### [P2] `gwexpy/spectral/estimation.py:548-553, 559, 565` (cid=SP6, 3/3 not-refuted)
- class: degenerate zero-variance CI / dead guard (bare pass)
- repro: `import numpy as np, warnings; from gwexpy.spectral.estimation import bootstrap_spectrogram; from gwexpy.spectrogram import Spectrogram; from astropy import units as u
np.random.seed(0); spec = Spectrogram(np.random.rand(5,4)+1, dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
r = bootstrap_spectrogram(spec, n_boot=200, block_size=5.0)  # block_size==n_time(5)
print(r.error_low.value)  # [0 0 0 0] zero-width CI,`
- fix: Replace the bare `pass` at L553 (and/or guard L559-565) with: when num_possible_blocks <= 1 (i.e. block_size >= n_time), warn that only one block placement is possible so every resample is identical and the CI collapses to zero width; clamp block_size to n_time-1 or raise. The author's own comment '# warn?' flags this as unhandled.

#### [P2] `gwexpy/spectral/estimation.py:467-468 (validation), 635-644 (CI)` (cid=SP7, 3/3 not-refuted)
- class: degenerate zero-width CI
- repro: `import numpy as np, warnings; from gwexpy.spectral.estimation import bootstrap_spectrogram; from gwexpy.spectrogram import Spectrogram; from astropy import units as u
np.random.seed(0); spec = Spectrogram(np.random.rand(5,4)+1, dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
r = bootstrap_spectrogram(spec, n_boot=1, method='mean')
print(r.error_low.value)  # [0 0 0 0]: single replicate -> p_low==p_high==center`
- fix: L467-468 currently only rejects n_boot<1. Add a warning (or higher minimum) when n_boot is too small to estimate the requested ci: with n_boot=1, np.(nan)percentile over axis=0 of a length-1 array returns the single value for every quantile so err_low==err_high==0. Recommend warning when n_boot < a few hundred, and at minimum warn at n_boot==1 that the CI is undefined.

### G6. fitting — GLS 直接ソルバ未検証・非PSD chi2
_起票案: #457 を拡張_  / 件数: 2

#### [P3] `gwexpy/fitting/gls.py:68 (solve); 54 (cov inv)` (cid=F1, 2/3 not-refuted)
- class: uncaught-exception/crash on rank-deficient or underdetermined design
- repro: `import numpy as np; from gwexpy.fitting.gls import GLS
# n_params > n_samples
GLS(np.array([[1.,2.,3.]]), np.array([1.])).solve()  # -> LinAlgError: Singular matrix
# rank-deficient X (collinear columns), enough samples
GLS(np.array([[1.,1.],[2.,2.],[3.,3.]]), np.array([1.,2.,3.])).solve()  # -> LinAlgError: Singular matrix`
- fix: In GLS.solve(), replace np.linalg.solve(XTW @ self.X, XTW @ self.y) with np.linalg.lstsq(..., rcond=None) for rank-deficient robustness, or validate self.X.shape[0] >= self.X.shape[1] and check np.linalg.matrix_rank / condition number, raising a clear ValueError ('design matrix is rank-deficient / underdetermined: n_samples<n_params') before solving. Same applies to the cov->cov_inv path at line 5

#### [P2] `gwexpy/fitting/gls.py:176 (also core.py:729 for the MCMC mirror)` (cid=F2, 3/3 not-refuted)
- class: silently-wrong scientific result: negative chi2 from non-PSD cov_inv
- repro: `import numpy as np; from gwexpy.fitting.gls import GeneralizedLeastSquares
def lin(x,a,b): return a*x+b
x=np.array([1.,2.,3.]); y=np.array([1.,2.,3.])
ci=np.array([[1.,0,0],[0,-5.,0],[0,0,1.]])  # symmetric but indefinite (a user-supplied bad cov_inv)
g=GeneralizedLeastSquares(x,y,ci,lin)
print(g(0.0,0.0))  # -> -10.0  (chi2 < 0; Minuit then minimizes a NON-CONVEX objective -> garbage fit)`
- fix: When cov_inv is supplied directly (no cov / no Cholesky path), validate it is symmetric positive-definite once in __init__ (e.g. attempt np.linalg.cholesky(cov_inv) or check np.all(np.linalg.eigvalsh(...)>0)) and raise ValueError on failure; or document/enforce that the chi2 must be non-negative. Mirror the same guard in core.py run_mcmc's cov_inv branch (line 726-731).

### G7. fitting — models 定義域エラー（power_law/exponential）
_起票案: #455 を拡張_  / 件数: 1

#### [P2] `gwexpy/fitting/models.py:73 (power_law); 65 (exponential)` (cid=F3, 3/3 not-refuted)
- class: silent NaN/Inf from model domain violation
- repro: `import numpy as np; from gwexpy.fitting.models import power_law
power_law(np.array([-1.,2.,3.]), 1.0, 0.5)   # -> [nan, 1.414, 1.732]  (x<0 with non-integer alpha)
power_law(np.array([0.,1.]), 1.0, -1.0)       # -> [inf, 1.0]            (x=0 with negative alpha)`
- fix: power_law: guard the negative/zero base. Either document that x must be > 0, or compute A * np.sign(x)*np.abs(x)**alpha / use np.where to emit a controlled result, and raise/warn when np.any(x<=0) for non-integer alpha. At minimum surface a warning instead of silently propagating NaN/Inf into the chi2 (which becomes NaN and stalls Minuit silently).

### G8. fitting — core fit_series sigma crop / run_mcmc 縮退
_起票案: 新規 issue_  / 件数: 2

#### [P2] `gwexpy/fitting/core.py:1095-1101 (searchsorted crop) and 1106-1107 (length check)` (cid=F4, 3/3 not-refuted)
- class: sigma/y misalignment and spurious ValueError on bin-aligned x_range
- repro: `import numpy as np; from gwexpy.frequencyseries import FrequencySeries
fs = FrequencySeries(np.arange(10.0), f0=0, df=1.0)
sig = np.arange(10.0)+0.1
# x_range endpoints exactly on bin centers -> half-open crop drops the right endpoint,
# but searchsorted(side='right') keeps it -> length mismatch:
fs.fit('pol1', x_range=(2.0, 6.0), sigma=sig)  # crop yields 4 bins, sigma slice yields 5 -> ValueErro`
- fix: Do not recompute the sigma crop independently with searchsorted against half-open gwpy crop semantics. Instead crop sigma using the SAME index set the data crop produced — e.g. compute the boolean mask / integer indices once from target vs series (np.isin on the actual cropped frequencies, or align by target.frequencies.value), then index sigma_arr with it. This makes endpoint inclusivity identica

#### [P2] `gwexpy/fitting/core.py:581 (signature, n_walkers=32) and 759-763 (sampler creation)` (cid=F7, 2/3 not-refuted)
- class: uncaught-exception/crash when n_walkers < 2*ndim
- repro: `Static + runtime: emcee.EnsembleSampler.run_mcmc raises RuntimeError ('unadvisable to use a red-blue move with fewer walkers than twice the number of dimensions') when n_walkers < 2*ndim. run_mcmc (core.py:581) takes n_walkers (default 32) with NO validation against ndim=len(float_params) (654). A model with >16 free parameters at the default n_walkers=32 (2*17=34>32), or any user passing a small`
- fix: After computing ndim (core.py:654), validate/auto-correct: if n_walkers < 2*ndim, either raise a clear ValueError naming both numbers, or bump n_walkers = max(n_walkers, 2*ndim + (2*ndim)%2) with a warning. Also enforce n_walkers even.

### G9. fitting — highlevel single-bin ndof / FitResult plot None
_起票案: #458 を拡張_  / 件数: 1

#### [P3] `gwexpy/fitting/highlevel.py:227-235 (single-bin mask) -> core.py:251 (ndof=0)` (cid=F6, 3/3 not-refuted) **[NEW]**
- class: degenerate fit: single-frequency-bin crop yields ndof<=0 / silent nan reduced_chi2
- repro: `Static proof: in fit_bootstrap_spectrum, if freq_range selects exactly one PSD bin, mask has a single True -> psd has 1 point and cov_cropped is 1x1 (highlevel.py:231-234). The 1x1 GLS solve succeeds, but FitResult.ndof = max(0, 1 - n_params) = 0 for any model with >=1 free param (core.py:251), so reduced_chi2 returns np.nan (core.py:256) and the 'fit' is unconstrained/degenerate with no warning.`
- fix: After applying the freq mask (highlevel.py:233), validate mask.sum() >= n_free_params + 1 (or at least >= 2) and raise/warn ('freq_range selects too few bins for a meaningful GLS fit: N bins, M parameters'). Pair with the existing F13 empty-mask guard.

## 棄却 / borderline（strict confirmed bar 未達）

### `gwexpy/spectral/estimation.py:136 (np.isnan TypeError); 155-158 (duration guard skip)` (cid=SP3, 1/3 not-refuted)
3 verifier の判断が割れた。実在する下位欠陥だが、silent-failure lens で「loud crash / 無効化された診断であり silent ではない」と却下され strict bar（2/3 not-refuted + evidence）を満たさず。**P3 hardening（メッセージ品質）として親 issue に折り込む候補**。
**Refuted as a Phase 1 confirmed silent-failure finding, but retained as a P3 hardening note**（「欠陥なし」ではなく「本 sweep の confirmed 条件には届かない」）。

- vote0: Repro is runnable and triggers exactly as claimed: at gwexpy/spectral/estimation.py:136, np.isnan(data) on an object-dtype array raises a cryptic TypeError ('ufunc isnan not supported for the input types'). Line numbers are accurate: L135 np.asarray, L136 np.i
- vote1: Confirmed against working tree (matches sweep_base e59f62e5b; git diff for this file is empty). Sub-claim 1 (np.isnan TypeError on object dtype at L136) is CONFIRMED by a runnable repro: TimeSeries(np.array([1,2,None,4], dtype=object), dt=1.0*u.s) constructs w
- vote2: Under the silent-failure lens (does it silently produce a wrong/NaN result with NO warning/exception?), the finding does NOT qualify. Both sub-claims raise loud exceptions, not silent wrong results. Sub-claim (1): the np.isnan TypeError at L136 is real and rep

### `gwexpy/spectral/estimation.py:481-490` (cid=SP9, 1/3 not-refuted)
3 verifier の判断が割れた。実在する下位欠陥だが、silent-failure lens で「loud crash / 無効化された診断であり silent ではない」と却下され strict bar（2/3 not-refuted + evidence）を満たさず。**P3 hardening（メッセージ品質）として親 issue に折り込む候補**。
**Refuted as a Phase 1 confirmed silent-failure finding, but retained as a P3 hardening note**（「欠陥なし」ではなく「本 sweep の confirmed 条件には届かない」）。

- vote0: Confirmed via airtight static proof. estimation.py:486 reads `if mean_avg > 0 and (std_avg / mean_avg) > 0.5:`. The `mean_avg > 0` short-circuit silently disables the non-stationarity heuristic for any spectrogram whose cross-frequency mean is <= 0 (e.g. dB-sc
- vote1: The candidate's repro does NOT validly trigger the claimed bug on its stated input. Two independent defects: (1) On the exact repro input (seed=0, randn*5-50, +=30), the relative variation is std_avg/|mean_avg| = 15.18/34.65 = 0.438, which is BELOW the 0.5 thr
- vote2: The L486 guard `if mean_avg > 0 and (std_avg/mean_avg) > 0.5` does skip the non-stationarity warning when the cross-frequency mean is <= 0 (dB-scaled input), and the lines/file match the working tree exactly (estimation.py:481-490, base SHA verified, diff empt

## 網羅性クリティック

- file classification: checked=11 / out-of-scope=3 / deferred=0
- **対象内完全未着手ファイル: 0 件** → scope freeze 成立

| file | class | note |
|------|-------|------|
| `gwexpy/statistics/__init__.py` | out-of-scope | Package re-export only (15 lines); exclusion candidate per task (__init__.py). N |
| `gwexpy/statistics/roc.py` | checked | Headline miss of prior sweep. Verified empirically: calculate_roc has 4 confirme |
| `gwexpy/statistics/gauch.py` | checked | Re-verified F21 (sigma2==0 NaN, lines confirmed at 87/98), F27 (p=0 floor, L114/ |
| `gwexpy/statistics/rayleigh_test.py` | checked | Re-verified F22 (n_monte_carlo=0 -> len(dist)=0 div, L59), F23 (n_samples=0 all- |
| `gwexpy/statistics/dq_flag.py` | checked | Re-verified F24 (empty times IndexError L79) and F25 (single-step dt fallback 1. |
| `gwexpy/statistics/student_t_indicator.py` | checked | Re-verified F26 (bare except L112-113). STFT-vs-gwpy time-axis gap: out_times us |
| `gwexpy/spectral/__init__.py` | out-of-scope | Package re-export only (13 lines); exclusion candidate (__init__.py). |
| `gwexpy/spectral/estimation.py` | checked | Re-verified F30/F31 (block_size>n_time crash, no-op guard L548-553, randint L565 |
| `gwexpy/fitting/__init__.py` | out-of-scope | Package wiring / enable_series_fit monkeypatch (83 lines); __init__.py exclusion |
| `gwexpy/fitting/core.py` | checked | Re-verified F01-F06 (cost div-by-zero L96-97/L131, complex dy, empty-x plot L383 |
| `gwexpy/fitting/gls.py` | checked | Re-verified F07 (inv no cond check L54), F08 (solve no cond check L68), F14 (sil |
| `gwexpy/fitting/highlevel.py` | checked | Re-verified F13 (empty-mask degenerate fit L227-266), F19 (2-panel blank, L299-3 |
| `gwexpy/fitting/models.py` | checked | Re-verified F09-F12, F15-F18 (sigma/tau/gamma/Q zero-division, get_model None re |
| `gwexpy/fitting/mixin.py` | checked | FittingMixin.fit (L11-59) inspected for the gwpy-compat gap. cov/cost_function a |

### finder メタ分析（なぜ前回 roc.py を取りこぼしたか）

TASK2 finder meta-analysis. The 3 prior finders (statistics/spectral/fitting) DID exhibit the same early-stop / structural blind spot that caused the roc.py whole-file miss, in three distinct ways. (1) WHOLE-FILE SKIP (statistics): roc.py was entirely skipped because the finder stopped at the 4 files with obvious Monte-Carlo/Spectrogram structure (gauch, rayleigh_test, student_t, dq_flag) and never applied any bug-class lens to roc.py. This supplement confirms 4 real roc.py defects in calculate_roc alone (empty-array ValueError at L24 BEFORE the n_pos guard at L31; all-equal y_score -> silent AUC=0.0; NaN y_score -> every >= comparison False -> silent AUC=0.0, no warning; tied-FPR AUC bias measured 0.541 vs sklearn 0.567) plus 3 in evaluate_detection_performance (hardcoded A1=0 'clean' contract L66, nanmax-on-all-NaN, n_trials=0 empty arrays). (2) STRUCTURAL BLIND SPOT BY CLASS-OF-OBJECT (fitting): the gls.py finder examined the GeneralizedLeastSquares COST class and the inv/solve lines but never analyzed the separate GLS DIRECT-SOLVER class as a unit because of the shared name prefix -> missed GLS.__init__ having NO X/y length validation (confirmed: builds eye(5) for mismatched 3-row X, opaque matmul crash) and the non-PSD cov_inv NEGATIVE-chi2 path (confirmed g(0.0)=-2.0, Minuit drives to -inf). (3) SINGLE-LENS TUNNEL VISION (spectral + fitting models + statistics): finders applied ONE lens and stopped. estimation.py finder used only NaN+block-index lenses and never applied the dtype/data_integrity lens -> missed the confirmed integer-dtype truncation in _bootstrap_resample_py/_jit (dtype=data.dtype at L58/L84; verified per-resample means 0.5->0 silently). models.py finder applied only the division-by-zero lens and stopped, missing the domain lens (power_law inf at f=0 / nan at x<0; polynomial overflow). core.py run_mcmc finder caught only ndim==0 and missed the broader log_prob silent -inf swallow (L739-741). The COMMON ROOT CAUSE is identical to the roc.py miss: finders anchored on a salient structural feature (Monte-Carlo loop / cost-class / division operator) and treated structurally-different code (a plain solver class, a pure-numpy ROC integrator, dtype 

**結論**: 前回 finder は salient な構造特徴（Monte-Carlo ループ / cost クラス / 除算演算子）にアンカーし、構造的に異なるコード（plain solver クラス、pure-numpy ROC 積分器、dtype 宣言、定義域端）を out-of-frame として扱った。roc.py の whole-file miss はこの共通根本原因の典型例。追補 sweep は同種の盲点（GLS 直接ソルバ、dtype、domain lens、log_prob swallow）を実際に検出・確定した。

## 親レポート 棄却 2 件（今回の再評価）

- `gwexpy/fitting/mixin.py`: Gap-list claim that FittingMixin.fit silently loses cov is REFUTED: cov passes through **kwargs (L19) into fit_series (L50-58) which accepts it -> GLS fit is performed. No defect.
- `gwexpy/fitting/core.py:1095`: Gap-list reversed-x_range sigma misalignment is now MITIGATED: L1098 sorts lo/hi and L1106 length-checks dy vs y. Residual ascending-x_full assumption is minor; downgraded, not a confirmed defect.

## 推奨アクション（v0.1.8 hardening lane）

1. **roc.py を最優先で hardening**（G1, 9件）。calculate_roc / evaluate_detection_performance に統一契約（縮退入力は ValueError or 明示 np.nan、0.5/0.0 の黙示返却禁止）+ 回帰テスト（sklearn 照合含む）。**現状ほぼ無テストのモジュール**。
2. **共通方針の踏襲**: 縮退入力は ValueError か明示 UserWarning+NaN（#450/#452 C1 方針・親レポート推奨と整合）。silent accidental NaN を排除。
3. 既存 issue 拡張: GLS→#457 / models→#455 / highlevel→#458 / spectral→#460。新規: roc.py(G1) / 再現性(G3) / student_t(G4) / fitting core(G8)。
4. **issue 案はテーマ群（G1–G9）単位**。確定後・ユーザー承認後に起票。

## 関連
- リリース: v0.1.8 hardening lane
- 親 sweep: `phase1_numerical_robustness_sweep_20260622.md`