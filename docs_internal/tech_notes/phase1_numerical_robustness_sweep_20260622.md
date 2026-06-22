# Phase 1 数値ロバスト性 sweep — 調査レポート

- **日付**: 2026-06-22
- **対象**: `gwexpy/statistics` / `gwexpy/spectral` / `gwexpy/fitting`（3,514 LOC・13 ファイル）
- **手法**: マルチエージェント並列 sweep（10 finder）→ 敵対的検証（refute 前提）→ モジュール別網羅性 critic（3体）
- **規模**: 48 エージェント・271 ツール実行・約 194 万トークン・約 9.3 分
- **ベース**: origin/main `81119dd34`（= v0.1.7 ベース。対象 3 パッケージはローカル作業ツリーと一致）
- **Workflow**: `phase1-numerical-robustness-sweep`（run `wf_9bb29977-efc`）

## 背景・目的

これまでのバグ波（v0.1.2〜v0.1.7）は I/O・interop・types に集中しており、`statistics`/`spectral`/`fitting` は
規模に対しテスト密度・修正頻度が低い「手薄」領域だった（statistics はテスト 2 ファイル、spectral 3、fitting 7）。
`#451`（`TimeSeries.rms` の gwpy 互換破壊）もこの領域で顕在化している。本 sweep は #450/#452 で実証した
「並列 sweep → 敵対的検証 → 網羅性 critic」構成を、4 つのバグクラスレンズ
（数値ロバスト性 / silent failure / API契約 / データ整合）で当該 3 領域へ横断適用したもの。

## サマリー

- 生検出 **35** → 敵対的検証で **確定 32 / 棄却 3**
- 重大度: **P2 = 22 / P3 = 10 / P1 = 0**
- クラス: numerical = 22 / silent_failure = 7 / api_contract = 2 / data_integrity = 1
- 確定数（モジュール別）: statistics **9** / spectral **3** / fitting **20**

> **重要な結論**: P1（通常入力での silent データ破壊・crash）は **0 件**。確定 32 件は全て
> 縮退・エッジ入力（sigma=0 / 空配列 / all-NaN / 全パラメータ固定など）でのみ顕在化する **堅牢性バグ**。
> I/O sweep（#450/#452, P1 silent data loss 多数）とは性質が異なり、緊急度は中。ただし NaN が
> p値・χ²・PSD に静かに混入する経路が多く、科学計算の信頼性として要対処。

> **dedup 注記**: `fitting/core.py:96-97` は 2 件（ゼロ除算＋複素 dy・別個の実バグ）、
> `spectral/estimation.py` の `block_size > n_time` は 2 件が同一バグ → 実質 31 distinct。

## 確定検出（32 件）

### fitting（20 件）

#### F01 [P2] Division by zero when dy contains a zero element silently produces inf chi2

- **箇所**: `gwexpy/fitting/core.py:96-97` — `ComplexLeastSquares.__call__`
- **クラス**: `numerical` / **検証信頼度**: medium / **breaking**: False
- **内容**: In ComplexLeastSquares.__call__, lines 96-97 compute `res_real = (self.y.real - ym.real) / self.dy` and `res_imag = (self.y.imag - ym.imag) / self.dy` with no guard against zero elements in `self.dy`. If the caller passes a sigma array that contains any zero value, NumPy division-by-zero produces `inf` in the residuals, which propagates through `np.sum(res_real**2 + res_imag**2)` as `inf`. Minuit receives `inf` as fval on the very first evaluation and typically terminates with a failed fit status and NaN parameters — without raising an exception or emitting any warning. The same defect exists in RealLeastSquares.__call__ at line 131.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.fitting.core import ComplexLeastSquares
  
  def model(x, a): return a * x.astype(complex)
  
  x = np.array([1.0, 2.0, 3.0])
  y = np.array([1.0+0j, 2.0+0j, 3.0+0j])
  dy = np.array([1.0, 0.0, 1.0])  # zero in position 1
  
  cost = ComplexLeastSquares(x, y, dy, model)
  print(cost(1.0))  # prints inf — Minuit will fail silently
  ```
- **修正案**: Add a zero-guard before the division in both cost classes. In `__init__` (or at the start of `__call__`), replace zero (and near-zero) dy values with a safe floor, or raise ValueError if any dy <= 0:

    # In ComplexLeastSquares.__init__ and RealLeastSquares.__init__:
    dy_arr = np.asarray(dy, dtype=float)
    if np.any(dy_arr <= 0):
        raise ValueError(f'dy must be strictly positive; got zeros/negatives at indices {np.where(dy_arr <= 0)[0]}')
    self.dy = dy_arr

Alternatively, add a floor in fit_series before constructing the cost object:
    dy = np.maximum(dy, SAFE_FLOOR_STRAIN)
- **検証コメント**: Confirmed real. core.py:96-97 (ComplexLeastSquares.__call__) and the parallel line 131 (RealLeastSquares.__call__) divide residuals by self.dy with no zero guard. The caller fit_series (lines 1082-1108) builds dy from the user-supplied sigma with only a length check (1106-1107) and NO value validation, so a sigma array containing 0 reaches the division. I ran the exact repro: ComplexLeastSquares is directly instantiable and the cost returns a non-finite value, which Minuit consumes as fval, yielding a silently failed fit.

Two inaccuracies in the report, neither of which negates the bug: (1) The repro produces nan, not inf, at a=1.0 because the model passes exactly through the zero-sigma point (0/0 = nan); it becomes inf only when the residual there is nonzero. The end consequence (non-finite fval, silent failure) is the same. (2) The claim "without emitting any warning" is wrong — NumPy emits a RuntimeWarning ("divide by zero" / "invalid value encountered in divide"). However such warnings do not halt the fit and are routinely ignored.

Severity corrected to P2 rather than the rubric's P1: a zero uncertainty/sigma is a physically meaningless, degenerate input (infinite weight, zero error bar), not normal valid data, so this is an edge/degenerate-input defect. Additionally, iminuit's own cost.LeastSquares — which RealLeastSquares is explicitly written to replace (docstring lines 112-114) — exhibits the same unguarded behavior with zero sigma, so this matches upstream library norms rather than being a deviation that corrupts results on valid input.

#### F02 [P2] Division by zero when dy contains a zero element silently produces inf chi2

- **箇所**: `gwexpy/fitting/core.py:131` — `RealLeastSquares.__call__`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: RealLeastSquares.__call__ at line 131 computes `res = (self.y - ym) / self.dy` with no guard against zero elements in `self.dy`. Identical root cause to the ComplexLeastSquares defect: a zero in sigma silently produces inf in the cost function, causing Minuit to fail without a diagnostic.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.fitting.core import RealLeastSquares
  
  def model(x, a): return a * x
  
  x = np.array([1.0, 2.0, 3.0])
  y = np.array([1.0, 2.0, 3.0])
  dy = np.array([1.0, 0.0, 1.0])  # zero in position 1
  
  cost = RealLeastSquares(x, y, dy, model)
  print(cost(1.0))  # prints inf
  ```
- **修正案**: Same as ComplexLeastSquares: validate dy > 0 in __init__ or apply a safe floor. Since both classes share the same pattern, a common _validate_dy helper or a base class is appropriate.
- **検証コメント**: Confirmed real defect with a minor inaccuracy in the repro. RealLeastSquares.__init__ (core.py:119-126) stores dy verbatim and __call__ (core.py:131) computes `res = (self.y - ym) / self.dy` with no zero guard; fit_series only checks sigma length (core.py:1106-1107), never positivity, so a zero in user-supplied `sigma` flows unguarded to line 131 (RealLeastSquares is instantiated at core.py:1178). The pattern is genuinely identical to ComplexLeastSquares (core.py:96-97). Reachability via the public fit_series API is confirmed.

However, the repro's claimed output is wrong. With the given data (y=[1,2,3], model=a*x, a=1), ym=[1,2,3], so the residual at the zero-dy index 1 is exactly 0, giving 0.0/0.0 = nan, not inf. I verified empirically: the literal repro prints nan, while a variant with a nonzero residual at the zero-dy position prints inf. So the title's "silently produces inf" is only true for nonzero residuals; for the given repro it is nan. Either way the cost becomes non-finite and Minuit fails without a diagnostic identifying the zero sigma.

Severity P2 is correct: it requires a degenerate/physically-invalid input (a sigma value of exactly zero, i.e. zero measurement uncertainty), not ordinary valid data, and the consequence is a silent fit failure rather than data loss or a plausibly-wrong result. Not P1 (no crash on valid input, no silent wrong-but-plausible result). The numerical-class designation is accurate.

#### F03 [P2] Complex-valued dy produces complex chi2 which Minuit cannot minimize

- **箇所**: `gwexpy/fitting/core.py:96-97` — `ComplexLeastSquares.__call__`
- **クラス**: `api_contract` / **検証信頼度**: high / **breaking**: False
- **内容**: ComplexLeastSquares stores `self.dy = dy` as-is (line 81). If the caller supplies a complex array for dy (e.g. from a complex-valued sigma), then `res_real = (...) / self.dy` and `res_imag = (...) / self.dy` are complex-valued. `res_real**2 + res_imag**2` is then complex, and `np.sum(...)` returns a complex scalar. Minuit requires a real-valued cost; it will either raise a TypeError internally (which surfaces as a cryptic iminuit error) or — depending on the iminuit version — silently cast to float by discarding the imaginary part, producing a wrong (incomplete) chi2 with no warning. The docstring acknowledges the isotropic assumption but does not enforce it.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.fitting.core import ComplexLeastSquares
  
  def model(x, a): return a * x.astype(complex)
  
  x = np.array([1.0, 2.0])
  y = np.array([1.0+0j, 2.0+0j])
  dy = np.array([1.0+0.1j, 1.0+0.1j])  # complex dy
  
  cost = ComplexLeastSquares(x, y, dy, model)
  print(cost(1.0))  # returns complex value — Minuit call will fail
  ```
- **修正案**: In ComplexLeastSquares.__init__, cast dy to a real dtype and raise if the imaginary part is non-negligible:

    dy_arr = np.asarray(dy)
    if np.iscomplexobj(dy_arr):
        if np.any(np.abs(dy_arr.imag) > 1e-12 * np.abs(dy_arr.real)):
            raise ValueError('dy must be real-valued for ComplexLeastSquares')
        dy_arr = dy_arr.real
    self.dy = dy_arr.astype(float)
- **検証コメント**: Confirmed real and reproducible. ComplexLeastSquares.__init__ stores dy as-is (core.py:80-82) with no real-cast or validation. Lines 96-97 compute res_real/res_imag by dividing real residuals by self.dy; if dy is complex these become complex, and np.sum(res_real**2+res_imag**2) at line 100 returns a complex scalar. Verified empirically with the reporter's repro: cost(1.0) returns np.complex128, and iminuit does NOT raise a hard TypeError — it emits a ComplexWarning and silently casts to real, discarding the imaginary part of the weighting (Minuit.migrad ran to a wrong/incomplete chi2). So the description's "silently cast" branch is the actual observed behavior; the chi2 weighting becomes wrong with no error.

Reachability via the public API: in fit_series the scalar-sigma branch (core.py:1089) does float(sigma), which would itself reject a complex scalar via TypeError, but the array-sigma branch (core.py:1093 sigma_arr = np.asarray(sigma)) performs NO real-cast and NO np.iscomplexobj check, so a complex sigma array flows straight to dy (line 1103) into ComplexLeastSquares(x, y, dy, model) (line 1176). The reporter's repro also instantiates the public class directly, so the code is definitely reachable.

Severity: corrected_severity P2 (matches claimed). The consequence is a silent wrong result, but it is only triggered by an INVALID/degenerate input: sigma/dy represents a measurement uncertainty, which is physically a real positive quantity (docstring at core.py:997 calls it "Per-point error estimates" and the class docstring states error is "assumed isotropic for real/imag"). No internal path ever produces a complex dy (default np.ones is real; the cov diagonal at line 1169 uses np.sqrt(np.maximum(...)), always real). Valid usage is never affected. This is the textbook P2 profile (edge/degenerate input only), not P1, since P1 requires wrong results on VALID input. The fix would be a cheap guard (e.g., dy = np.abs(np.asarray(dy)) or a np.iscomplexobj raise) but its absence does not corrupt any legitimate workflow.

#### F04 [P2] min()/max() on empty self.x crashes when x_range crops all data

- **箇所**: `gwexpy/fitting/core.py:383` — `FitResult.plot`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: When neither x_range nor self.x_fit_range is set, the fallback branch on line 383 calls `np.linspace(min(self.x), max(self.x), num_points)`. If `self.x` is an empty numpy array (e.g., because x_range cropped the series to zero length, or the series itself was empty), Python's built-in `min()`/`max()` raises `ValueError: min() iterable argument is empty` for a numpy array (or `numpy` raises 'zero-size array to reduction operation minimum'). There is no length guard before this call.
- **再現**:
  ```python
  import numpy as np
  from unittest.mock import MagicMock
  from gwexpy.fitting.core import FitResult
  m = MagicMock(); m.fval=0.0; m.nfit=1; m.parameters=()
  fr = FitResult(m, lambda x: x, np.array([]), np.array([]))
  fr.plot()  # raises ValueError: min() iterable argument is empty
  ```
- **修正案**: Add a guard before line 383:
```python
if len(self.x) == 0:
    raise ValueError("FitResult.x is empty; cannot produce plot x range.")
```
Or use `np.linspace(np.min(self.x), np.max(self.x), num_points)` inside a try/except with a meaningful message.
- **検証コメント**: Verified at /home/washimi/work/gwexpy/gwexpy/fitting/core.py:383. In FitResult.plot, when is_complex is False and both x_range (line 281 kwarg) and self.x_fit_range (line 378) are None, the else branch executes `x_plot = np.linspace(min(self.x), max(self.x), num_points)` using Python's builtin min()/max(). There is no length guard between the start of plot() (line 266) and line 383. self.x is assigned directly from the constructor argument (line 165) with no validation, and FitResult is constructed at line 1227 from `x = target.value` where target = series.crop(*x_range) (line 1030) — an x_range that crops away all data, or an empty input series, yields an empty self.x. Empirically confirmed: (1) `min(np.array([]))` raises `ValueError: min() iterable argument is empty`; (2) the exact claimed repro (MagicMock minuit + empty arrays) raises that ValueError at line 383 via the traced call path. The technical claim and repro are accurate. Severity is P2, not P1: the crash only occurs in a degenerate state (zero data points in the fit), which is not valid input for a meaningful fit; a normal fit always has data and never reaches an empty self.x. The consequence is a confusing exception rather than silent wrong results, and it is reachable only via edge/degenerate cropping. The same unguarded pattern also appears at line 901 (np.min/np.max, which raises a different but analogous ValueError on empty arrays), reinforcing that this is an edge-case omission rather than a normal-path defect.

#### F05 [P2] np.min/np.max on empty self.x crashes in plot_fit_band

- **箇所**: `gwexpy/fitting/core.py:901` — `FitResult.plot_fit_band`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: Line 901 calls `np.linspace(np.min(self.x), np.max(self.x), num_points)` without checking whether `self.x` is empty. `numpy.min` on a zero-size array raises `ValueError: zero-size array to reduction operation minimum which has no identity`. This path is reachable when `plot_fit_band()` is called on a FitResult built from an empty-after-crop series.
- **再現**:
  ```python
  import numpy as np
  from unittest.mock import MagicMock
  from gwexpy.fitting.core import FitResult
  m = MagicMock(); m.fval=0.0; m.nfit=1; m.parameters=()
  fr = FitResult(m, lambda x: x, np.array([]), np.array([]))
  fr.samples = np.empty((100, 0))
  fr.mcmc_labels = []
  fr.plot_fit_band()  # raises ValueError on np.min(self.x)
  ```
- **修正案**: Add before line 901:
```python
if len(self.x) == 0:
    raise ValueError("FitResult.x is empty; cannot produce plot x range.")
```
- **検証コメント**: Confirmed. /home/washimi/work/gwexpy/gwexpy/fitting/core.py line 901 calls `np.linspace(np.min(self.x), np.max(self.x), num_points)` with no guard for empty `self.x`. The only guard in plot_fit_band (lines 894-895) checks `self.samples is None`, not data emptiness. The constructor (lines 143-185) assigns `self.x = x` with zero validation, and the public `fit()` entrypoint crops data via `series.crop(*x_range)` (line 1030) and constructs FitResult (line 1227) without any non-empty check, so an empty-after-crop series is admitted. I verified numpy behavior directly: `np.min(np.array([]))` raises `ValueError: zero-size array to reduction operation minimum which has no identity`. The bug is genuine. Severity P2 (not higher) is correct because it only triggers on empty/degenerate data, not valid input: to reach line 901 the path also requires `self.samples` to be populated, which in the normal pipeline means a successful `run_mcmc` on zero data points (itself a degenerate, ill-posed fit). The claimed repro is artificial (it sets `fr.samples`/`fr.mcmc_labels` manually to bypass run_mcmc) but it does faithfully reach and trigger the exact ValueError at line 901. Note line 383 (`min(self.x), max(self.x)`) has the same latent issue. Classified as numerical/edge robustness gap = P2 as claimed.

#### F06 [P2] emcee.EnsembleSampler crashes with LinAlgError when ndim==0 (all params fixed)

- **箇所**: `gwexpy/fitting/core.py:654-763` — `FitResult.run_mcmc`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: When all Minuit parameters are fixed, `float_params` is empty and `ndim = len(float_params) == 0`. The code constructs `pos` with shape `(n_walkers, 0)` and passes `ndim=0` to `emcee.EnsembleSampler`. Calling `sampler.run_mcmc(pos, n_steps)` then raises `numpy.linalg.LinAlgError: cond is not defined on empty arrays` inside emcee's stretch-move proposal. There is no guard for `ndim == 0` before reaching the sampler construction.
- **再現**:
  ```python
  # Verified with emcee 3.1.6:
  import numpy as np
  import emcee
  sampler = emcee.EnsembleSampler(32, 0, lambda theta: 0.0)
  sampler.run_mcmc(np.random.randn(32, 0), 10)
  # => numpy.linalg.LinAlgError: cond is not defined on empty arrays
  
  # In FitResult context:
  # result = fit_series(series, model, fixed=['a','b'])
  # result.run_mcmc()  # all params fixed => ndim=0 => same crash
  ```
- **修正案**: Add an explicit guard immediately after line 654 (ndim assignment):
```python
if ndim == 0:
    raise ValueError(
        "run_mcmc() requires at least one free parameter. "
        "All parameters are currently fixed."
    )
```
This converts an opaque LinAlgError into a clear, actionable message.
- **検証コメント**: Confirmed real. In gwexpy/fitting/core.py, run_mcmc computes float_params at line 653 and ndim=len(float_params) at line 654 with no guard for ndim==0. When all Minuit parameters are fixed, float_params is empty, ndim=0, pos gets shape (n_walkers, 0) at line 759, and emcee.EnsembleSampler(n_walkers, 0, log_prob) is constructed at line 762 then run at line 763. I reproduced the exact failure with the installed emcee 3.1.6: emcee.EnsembleSampler(32, 0, ...).run_mcmc(np.random.randn(32,0), 10) raises numpy.linalg.LinAlgError: cond is not defined on empty arrays. The path is reachable via the public entry point: fixed=['a','b'] sets m.fixed[name]=True (lines 1217-1219) for all params, so the claimed repro is accurate. Notably the codebase already anticipates the all-fixed case elsewhere (line 1224: "if m.nfit > 0" guards Hesse), but run_mcmc has no equivalent guard. Severity P2 is correct: this is a degenerate edge case (zero free parameters makes MCMC meaningless), not a crash on a meaningful workflow on valid input, so it does not rise to P1. The fix is a friendly guard (e.g., raise ValueError with a clear message when ndim==0) rather than the opaque LinAlgError.

#### F07 [P2] np.linalg.inv used on cov with no singularity or condition-number check

- **箇所**: `gwexpy/fitting/gls.py:54` — `GLS.__init__`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: When a user supplies `cov` (but not `cov_inv`) to `GLS`, the constructor computes `self.cov_inv = np.linalg.inv(np.asarray(cov))`. `np.linalg.inv` silently returns a numerically garbage result for ill-conditioned matrices (large condition number) without raising or warning. Only an exactly singular matrix raises `LinAlgError`. In GW spectral covariance matrices, near-singularity is common, and the resulting bogus `cov_inv` is then used in `solve()` to produce a plausible-looking but incorrect beta vector.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.fitting.gls import GLS
  
  # Nearly singular covariance
  n = 4
  cov = np.eye(n) + 1e-15 * np.ones((n, n))  # condition number ~1e15
  X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
  y = np.array([1.0, 2.0, 3.0, 4.0])
  gls = GLS(X, y, cov=cov)
  # No warning; cov_inv is garbage at double precision
  beta = gls.solve()  # returns wrong result silently
  ```
- **修正案**: After computing the inverse, check the condition number and warn:
```python
cov_arr = np.asarray(cov)
cond = np.linalg.cond(cov_arr)
if cond > 1.0 / np.finfo(float).eps:
    import warnings
    warnings.warn(
        f"Covariance matrix is ill-conditioned (cond={cond:.3e}); "
        "the computed inverse may be inaccurate. Consider using cov_inv directly.",
        RuntimeWarning,
        stacklevel=2,
    )
self.cov_inv = np.linalg.inv(cov_arr)
```
Alternatively use `np.linalg.pinv` or a Cholesky-based solve instead of explicit inversion.
- **検証コメント**: Core claim is real. At /home/washimi/work/gwexpy/gwexpy/fitting/gls.py:54, GLS.__init__ does `self.cov_inv = np.linalg.inv(np.asarray(cov))` with NO condition-number or symmetry/positive-definiteness guard. The GLS class (lines 26-69) has no validation at all; the Cholesky+try/except fallback at lines 136-141 belongs to the separate GeneralizedLeastSquares class and does not protect GLS. np.linalg.inv only raises LinAlgError for exactly-singular matrices and silently returns garbage for ill-conditioned ones (verified: inv of diag([1,1,1,1e-16]) returns a 1e16 entry with no warning). The bad cov_inv then flows into solve() (line 68).

Reachability and consequence confirmed empirically: with a genuinely ill-conditioned SPD covariance (cond ~1.2e16), GLS.solve() silently returned beta=[2.87, 0.42] while the regularized/correct answer is [1, 1] — a ~138% relative error, no warning or error raised.

Important correction to the repro: the literal repro example `np.eye(n) + 1e-15*np.ones((n,n))` is NOT ill-conditioned — its condition number is ~1.0 (verified), not the ~1e15 the repro comment claims (the rank-1 perturbation only shifts one eigenvalue to 1+4e-15). Running the literal repro returns the CORRECT beta=[1,1], demonstrating nothing. The bug is genuine but the provided repro must be replaced with a real ill-conditioned matrix to trigger it.

Severity P2 (not P1): the defect only manifests for near-singular/ill-conditioned covariance — a degenerate/edge input. On ordinary well-conditioned valid input GLS produces correct results. The GW-domain near-singularity argument gives it plausibility but it remains an edge-condition robustness gap, not silent corruption on arbitrary valid input. Recommended fix: add an np.linalg.cond check (or use cond-aware pinv / Cholesky-based solve) and warn/raise when conditioning exceeds a threshold.

#### F08 [P2] Normal equations solved with np.linalg.solve with no ill-conditioning guard

- **箇所**: `gwexpy/fitting/gls.py:68` — `GLS.solve`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: `np.linalg.solve(XTW @ self.X, XTW @ self.y)` forms the normal-equation matrix `XTW @ X` (shape n_params × n_params) and solves it directly. For an ill-conditioned or rank-deficient design matrix `X`, `np.linalg.solve` does not raise for near-singular inputs — it returns a numerically invalid `beta` silently. The condition number of the normal equations is the square of the condition number of `X`, so moderate collinearity in the design matrix is amplified.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.fitting.gls import GLS
  
  # Nearly collinear design matrix
  n = 10
  X = np.column_stack([np.ones(n), np.ones(n) + 1e-10 * np.arange(n)])
  y = np.random.default_rng(0).normal(size=n)
  gls = GLS(X, y)  # identity cov
  beta = gls.solve()  # no warning; cond(XTW @ X) >> 1/eps, result is garbage
  ```
- **修正案**: Add a condition-number check before solving, or use `np.linalg.lstsq` which handles rank-deficiency robustly:
```python
A = XTW @ self.X
b = XTW @ self.y
cond = np.linalg.cond(A)
if cond > 1.0 / np.finfo(float).eps:
    import warnings
    warnings.warn(
        f"Normal equations matrix is ill-conditioned (cond={cond:.3e}); "
        "solution may be inaccurate.",
        RuntimeWarning,
        stacklevel=2,
    )
beta = np.linalg.solve(A, b)
return beta
```
- **検証コメント**: Confirmed real. In gwexpy/fitting/gls.py, GLS.solve (lines 59-69) forms the normal-equation matrix and solves it directly at line 68: `beta = np.linalg.solve(XTW @ self.X, XTW @ self.y)`. (1) No guard exists: __init__ (lines 42-57) only does np.asarray / cov_inv computation; solve() has no condition-number, rank, or LinAlgError handling. (2) The repro is reachable: GLS(X, y) with no cov sets cov_inv = np.eye (line 57), so solve() computes np.linalg.solve(XᵀX, Xᵀy), exactly the cited line. (3) The numerical claim is accurate standard linear algebra: forming XᵀWX squares the condition number of X, and np.linalg.solve does not raise for merely ill-conditioned (non-exactly-singular) inputs, returning an unwarned, numerically degraded beta. A whitened lstsq (SVD/QR) would avoid squaring κ. The inline comment on line 67 ("better stability than explicit inverse") is true vs inv() but understates that normal-equations solve is still inferior to whitened lstsq. Severity stays P2, not P1: the failure manifests only for ill-conditioned/rank-deficient design matrices (the repro deliberately injects a 1e-10 near-collinear column), which are degenerate inputs where the GLS estimate is intrinsically ill-defined. For ordinary well-conditioned fitting problems the result is correct, so this is an edge/degenerate robustness improvement, matching the P2 definition.

#### F09 [P2] Division by zero when sigma=0

- **箇所**: `gwexpy/fitting/models.py:57` — `gaussian`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: The expression `(x - mu) / sigma` performs an unguarded scalar division. When `sigma=0` and `x` is a Python scalar, Python raises `ZeroDivisionError`. When `x` is a NumPy array, NumPy silently produces `inf`/`nan` (with a RuntimeWarning that may be suppressed), which then propagates through `np.exp` to return an array of `nan`s. No guard exists.
- **再現**:
  ```python
  from gwexpy.fitting.models import gaussian; gaussian(0.0, 1.0, 0.0, 0.0)  # ZeroDivisionError for scalar sigma
  ```
- **修正案**: Add a guard: `if sigma == 0: raise ValueError('sigma must be non-zero')` at the top of the function body, or use `np.where(sigma != 0, ...)` for vectorised sigma.
- **検証コメント**: Confirmed at /home/washimi/work/gwexpy/gwexpy/fitting/models.py:57: `return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)` performs an unguarded division by `sigma`. No validation guard exists in the function, in the MODELS registry (lines 219-231), or in the fitting `get_model` path. The `sigma` in fitting/core.py (line 977+) is data uncertainty, a separate parameter, and does not guard this model parameter.

Reproduced both claimed behaviors exactly:
- Scalar: `gaussian(0.0, 1.0, 0.0, 0.0)` raises `ZeroDivisionError: float division by zero`.
- NumPy array: emits `RuntimeWarning: divide by zero` / `invalid value encountered in divide` and returns `nan` (e.g. `[nan 0.]`), which propagates silently if warnings are suppressed.

The mechanism and consequence described in the finding are fully accurate. However, severity is P2, not P1: `sigma=0` is a mathematically degenerate parameter for a Gaussian (zero-width delta-function limit is undefined), not a valid operating input. The same unguarded division pattern exists across sibling models (exponential `/tau`, landau `/sigma`, etc.), so this is a degenerate-edge-case class of issue rather than a crash on legitimate input. Crashing/NaN on a degenerate parameter matches P2 (edge/degenerate only).

#### F10 [P2] Silent wrong result when gamma=0: nan at center, zero everywhere else

- **箇所**: `gwexpy/fitting/models.py:117` — `lorentzian`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: When `gamma=0`, the numerator `gamma**2 = 0` and the denominator is `(x-x0)**2 + 0`. For `x != x0` the function returns `0.0` (mathematically the true limit is a Dirac delta, so fitting will silently accept a degenerate parameter). At `x == x0` exactly (scalar or array element), the result is `0/0 = nan`. No guard, no warning, no error.
- **再現**:
  ```python
  import numpy as np; from gwexpy.fitting.models import lorentzian; lorentzian(np.array([0.0, 1.0]), 1.0, 0.0, 0.0)  # array([nan, 0.])
  ```
- **修正案**: Add `if gamma == 0: raise ValueError('gamma must be non-zero')` at the top of the function.
- **検証コメント**: Confirmed at models.py:117: `return A * gamma**2 / ((x - x0) ** 2 + gamma**2)`. With gamma=0, numerator and the gamma**2 denominator term are both 0, so for x!=x0 the result is 0.0 and at x==x0 it is 0/0 = nan. No guard or validation exists in lorentzian or upstream; the repro `lorentzian(np.array([0.0,1.0]),1.0,0.0,0.0)` -> array([nan, 0.]) is accurate. So the numerical claim is real.

However, severity is P2 (edge/degenerate only), not P1, because gamma=0 is a degenerate parameter, not valid input on a real call path. The docstring defines gamma as HWHM (the width); a zero-width Lorentzian is physically meaningless. The function is a fitting model driven by iminuit, which optimizes parameters and would not normally be seeded with gamma=0. No actual caller passes gamma=0 (the only other use, noise/peaks.py:lorentzian_line, is a separate implementation). Furthermore, the same no-guard posture is consistent across every model in this file (gaussian/sigma, exponential/tau, power_law/x**alpha, lorentzian_q/Q all break on their own degenerate inputs), so this is a deliberate module-wide convention rather than a unique defect. The 'silent wrong result' framing is also slightly overstated: x==x0 yields nan (a visible signal a fitter would reject), and x!=x0 yields 0.0 which is the correct pointwise limit. Real but edge-only -> P2.

#### F11 [P2] Division by zero on sigma=0 at line 172 before the guard at line 178

- **箇所**: `gwexpy/fitting/models.py:172-182` — `voigt`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: Line 172 computes `z = ((x - x0) + 1j*gamma) / (sigma * np.sqrt(2))`. When `sigma=0` this divides by zero. The existing guard at lines 177-181 only protects the normalization constant `peak` after `_wofz` is already called on an infinite `z`. For scalar `sigma=0` Python raises `ZeroDivisionError`; for NumPy array input the division silently produces `inf+nanj` before the guard fires.
- **再現**:
  ```python
  import numpy as np; from gwexpy.fitting.models import voigt; voigt(np.array([0.0, 1.0]), 1.0, 0.0, 0.0, 1.0)  # z is inf, wofz(inf) is 0, peak guard fires on fallback but z result is already garbage
  ```
- **修正案**: Add `if sigma == 0: raise ValueError('sigma must be non-zero for voigt profile')` before line 172, so the guard at line 178 becomes reachable only via the normal code path.
- **検証コメント**: Confirmed by execution. With sigma=0, voigt crashes before the guard at line 178 can help, and the guard is effectively dead code for this case. Two distinct failure modes:

- Scalar x, sigma=0: line 172 `z = ((x - x0) + 1j*gamma) / (sigma * np.sqrt(2))` raises ZeroDivisionError (complex / 0), exactly as claimed.
- Array x, sigma=0: line 172 does NOT crash — it produces inf+nanj/nan+nanj with only a RuntimeWarning (warnings-as-errors aside). The actual crash for array input occurs one line later at line 175, `z0 = 1j * gamma / (sigma * np.sqrt(2))`, which is always a scalar complex/0 and raises ZeroDivisionError. So the finding's substance (sigma=0 fails before the guard) is correct, but its attribution of the array-input crash to line 172 is imprecise — the crash is at line 175.

The author clearly intended to handle sigma->0 (note `max(sigma, 1e-30)` at line 180), but the protection only guards the fallback peak constant inside the `if abs(peak) < 1e-12` block (line 178), never the primary z/z0 divisions at lines 172/175. Hence the guard never executes for sigma=0; lines 172/175 raise first. No upstream caller validation prevents sigma=0 from reaching this code (the function is a public model fed directly to curve fitters).

Severity P2 is appropriate: sigma is a Gaussian standard deviation; sigma=0 is a degenerate/boundary value (zero-width limit), not a typical valid fit point. It is a real crash but only on a degenerate edge case, with evident author intent to handle it. Not P1 (no silent wrong result on ordinary valid input; the inf/nan path for array x would still crash at line 175 rather than silently returning garbage).

#### F12 [P2] Division by zero when tau=0

- **箇所**: `gwexpy/fitting/models.py:81` — `damped_oscillation`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: `np.exp(-x / tau)` is unguarded. Scalar `tau=0` raises `ZeroDivisionError`; NumPy float `tau=0.0` silently produces `nan`/`inf` in the array without any warning by default.
- **再現**:
  ```python
  from gwexpy.fitting.models import damped_oscillation; damped_oscillation(1.0, 1.0, 0.0, 1.0)  # ZeroDivisionError
  ```
- **修正案**: Add `if tau == 0: raise ValueError('tau must be non-zero')` at the start of the function body.
- **検証コメント**: Confirmed at /home/washimi/work/gwexpy/gwexpy/fitting/models.py:81: `return A * np.exp(-x / tau) * np.sin(...)` divides by `tau` with no guard or validation. Reproduced exactly: `damped_oscillation(1.0, 1.0, 0.0, 1.0)` (mapping x=1.0, A=1.0, tau=0.0, f=1.0) raises ZeroDivisionError because `-x/tau` is a pure Python float division (1.0/0.0). With an array `x` (the actual fitting path), it instead emits a default-on RuntimeWarning ("divide by zero encountered in divide") and silently yields inf/nan. No caller guards this: the only consumer is fitting/core.py:1027 (`get_model`), which feeds the function straight to iminuit; iminuit can probe tau=0 during minimization, where the silent nan/inf could pollute the cost function. The claim's numerical description is accurate. Severity remains P2 rather than P1: tau=0 is a degenerate, non-physical parameter (zero decay time has no meaning), so this is an edge/boundary case, not a crash/wrong-result on valid input. Note the same unguarded division pattern exists in sibling models (exponential tau, gaussian sigma, lorentzian_q Q), while voigt does guard (lines 177-181), so the codebase is internally inconsistent here.

#### F13 [P2] No guard for empty mask after freq_range crop — degenerate fit silently proceeds

- **箇所**: `gwexpy/fitting/highlevel.py:227-266` — `fit_bootstrap_spectrum`
- **クラス**: `silent_failure` / **検証信頼度**: high / **breaking**: False
- **内容**: When freq_range=(fmin, fmax) covers no PSD frequency bins, mask on line 231 is all-False. psd becomes an empty FrequencySeries (len 0), cov_cropped is a 0×0 matrix, and fit_series is called with zero data points. In fit_series the covariance shape check is (0,0)==(0,0) which passes, GeneralizedLeastSquares receives an empty x/y, and Minuit runs on a cost function with no data — producing numerically undefined parameter estimates and errors (all NaN or unchanged init values) with no warning or exception raised to the caller.
- **再現**:
  ```python
  import numpy as np; from astropy import units as u; from gwexpy.spectrogram import Spectrogram; from gwexpy.fitting.highlevel import fit_bootstrap_spectrum; data = np.random.rand(20, 50); spec = Spectrogram(data, dt=1.0*u.s, f0=10*u.Hz, df=1*u.Hz); fit_bootstrap_spectrum(spec, lambda f, A, alpha: A*f**alpha, freq_range=(1000, 2000), initial_params={'A':1,'alpha':-1}, plot=False)  # freq axis is 10-59 Hz, range 1000-2000 Hz matches nothing
  ```
- **修正案**: After line 231, add: if not np.any(mask): raise ValueError(f'freq_range=({fmin}, {fmax}) selects no frequency bins from the PSD (range {freqs[0]:.3g}–{freqs[-1]:.3g} Hz). Check your freq_range.')
- **検証コメント**: Verified real. (1) No guard: fit_bootstrap_spectrum (highlevel.py:231-234) builds mask=(freqs>=fmin)&(freqs<=fmax) and never checks mask.any()/len; psd[mask] and cov_map.value[np.ix_(mask,mask)] become length-0 / 0x0, then flow into fit_series with no validation. (2) Shape-check bypass confirmed: in fit_series (core.py:1160 `if cov_inv.shape != (n,n)`) with n=len(y)=0 and cov_inv.shape=(0,0), the check (0,0)!=(0,0) is False so it passes silently; the identical guard in GeneralizedLeastSquares.__init__ (gls.py:128) also passes. (3) Numerical consequence confirmed empirically: np.linalg.cholesky on a 0x0 matrix succeeds; the GLS cost returns chi2=0.0 for any params (empty residual sum); Minuit.migrad() does NOT raise — it returns valid=False, init param values unchanged, and zero errors. fit_series (core.py:1222-1245) calls migrad() then returns FitResult(m,...) without ever inspecting m.valid/m.fmin, so a degenerate fit (init values, 0 errors) is returned to the caller with no warning or exception. Severity P2 not P1: this fires only on a user-supplied freq_range that selects zero PSD bins (a misconfiguration / degenerate input), not on valid in-range data; it does not corrupt otherwise-correct fits. It is a silent-failure-on-bad-input rather than wrong-result-on-valid-input or a crash. Reachability caveat: the path requires bootstrap_spectrogram to first succeed and yield a non-empty PSD whose bins all fall outside freq_range; the literal toy repro's first stage was not fully exercised, but the defect is plainly reachable through the normal documented pipeline. Fix is a one-line guard raising on mask.sum()==0.

#### F14 [P3] Silent discard of cov when Cholesky decomposition fails — no warning emitted

- **箇所**: `gwexpy/fitting/gls.py:137-141` — `GeneralizedLeastSquares.__init__`
- **クラス**: `silent_failure` / **検証信頼度**: high / **breaking**: False
- **内容**: When the caller supplies the optional `cov` argument to use Cholesky for numerical stability, the constructor attempts `np.linalg.cholesky(self.cov)`. If the matrix is not strictly positive definite (e.g. it is PSD, numerically perturbed, or slightly indefinite), the `except np.linalg.LinAlgError` branch silently sets `self.cov_cho = None` and falls back to the direct `cov_inv` path — with no warning, log message, or exception. The caller receives no signal that the explicit stability request was ignored. In GW spectral estimation covariance matrices are commonly near-singular, making this a realistic scenario, not merely a theoretical edge case.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.fitting.gls import GeneralizedLeastSquares
  
  def model(x, a): return a * x
  
  n = 4
  x = np.arange(n, dtype=float)
  y = np.ones(n)
  # Rank-1 covariance: not positive definite
  cov = np.ones((n, n))  # singular, Cholesky will fail
  cov_inv = np.eye(n)
  gls = GeneralizedLeastSquares(x, y, cov_inv, model, cov=cov)
  # No exception, no warning — cov_cho is None, Cholesky path silently abandoned
  assert gls.cov_cho is None  # passes, caller has no idea
  ```
- **修正案**: Add a warnings.warn call in the except branch:
```python
except np.linalg.LinAlgError:
    import warnings
    warnings.warn(
        "cov is not positive definite; Cholesky decomposition failed. "
        "Falling back to direct cov_inv. Numerical stability may be reduced.",
        RuntimeWarning,
        stacklevel=2,
    )
    self.cov_cho = None
```
- **検証コメント**: The behavior described is real: at gls.py:137-141 a Cholesky failure is caught and `self.cov_cho` is silently reset to None with only a code comment — no warning/log/exception. The repro is valid: the constructor signature `(x, y, cov_inv, model, cov=None)` accepts cov_inv=eye(4) (passes the shape check at line 128) and cov=ones((4,4)) (rank-1, singular), so `np.linalg.cholesky` raises LinAlgError and the except branch executes, leaving cov_cho=None as asserted.

However the claimed P1 consequence (silent data loss / wrong result) is NOT accurate. `cov_inv` is a separate, mandatory, caller-supplied argument, not derived from `cov`. The fallback at __call__ lines 174-176 computes chi2 = r @ self.cov_inv @ r — i.e., it uses exactly the authoritative inverse-covariance the caller already provided. Per the docstring (lines 92-94), `cov` is purely an optional numerical-stability optimization; the two paths are mathematically equivalent (||L^-1 r||^2 == r^T cov_inv r). So there is no wrong result, no data loss, and no crash on valid input — the χ² stays correct to the precision of the supplied cov_inv. The only genuine defect is the missing transparency: the caller's explicit request for the more-stable Cholesky path is ignored without notification. That is a real but minor observability/UX gap, not silent data corruption. P1 requires silent wrong result/crash, which does not occur; this is correctly a P3 (add a warnings.warn in the except branch).

#### F15 [P3] Division by zero when tau=0

- **箇所**: `gwexpy/fitting/models.py:65` — `exponential`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: The expression `-x / tau` is unguarded. Scalar `tau=0` raises `ZeroDivisionError`; array `x` with `tau=0.0` silently returns `nan`/`inf` without raising.
- **再現**:
  ```python
  from gwexpy.fitting.models import exponential; exponential(1.0, 1.0, 0.0)  # ZeroDivisionError
  ```
- **修正案**: Add `if tau == 0: raise ValueError('tau must be non-zero')` at the start of the function body.
- **検証コメント**: Confirmed the function at models.py:65 is `return A * np.exp(-x / tau)` with no guard. Repro verified exactly: scalar `exponential(1.0, 1.0, 0.0)` raises ZeroDivisionError (Python scalar float division), and no caller validates tau (the only references are the MODELS dict at lines 222-224; the `exponential` in noise/wave.py is an unrelated function).

However the severity claim (P2) and consequence description are overstated:
1) The array claim "silently returns nan/inf" is inaccurate. For typical positive x with tau=0.0, the array path returns 0.0 (since -x/0 = -inf and exp(-inf)=0), NOT nan/inf, and it emits a RuntimeWarning (not silent). Only mixed-sign x arrays produce nan (at x=0) / inf (at x<0).
2) tau=0 is a degenerate/unphysical parameter value for an exponential decay model. This is a fit-model function consumed by an optimizer (iminuit), not a function fed user data on a valid domain. The same unguarded-denominator pattern is consistent across every model in this file (gaussian sigma, lorentzian gamma, landau sigma, lorentzian_q Q), which is conventional for fit models where the optimizer explores parameter space and the backend handles nan/inf.

This is not silent data loss, not a wrong result on valid input, and not a crash on valid input — it is a degenerate/edge-only condition. Per the rubric (P3=minor; P2=edge/degenerate only), it sits at the boundary; given the overstated description and the conventional pattern, P3 is the accurate severity rather than P2.

#### F16 [P3] Division by zero when sigma=0

- **箇所**: `gwexpy/fitting/models.py:94` — `landau`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: `lam = (x - mu) / sigma` is unguarded. Scalar `sigma=0` raises `ZeroDivisionError`; array input silently returns `nan`/`inf`. Additionally, large negative `lam` values (e.g. `x << mu` and `sigma` small) cause `np.exp(-lam)` to overflow to `inf`, making the outer `np.exp` return 0.0 silently.
- **再現**:
  ```python
  from gwexpy.fitting.models import landau; landau(0.0, 1.0, 0.0, 0.0)  # ZeroDivisionError
  ```
- **修正案**: Add `if sigma == 0: raise ValueError('sigma must be non-zero')` before computing `lam`.
- **検証コメント**: The code at line 94 (`lam = (x - mu) / sigma`) is genuinely unguarded, so the finding describes a real characteristic. However the claimed consequences are overstated and the trigger is purely degenerate input.

Repro reachability: `landau(0.0, 1.0, 0.0, 0.0)` maps to x=0.0, A=1.0, mu=0.0, sigma=0.0, giving `0.0/0.0`, which with Python scalars does raise ZeroDivisionError. So claim (1) is technically reproducible — but `sigma=0` is a mathematically invalid scale parameter for any distribution, never a legitimate input.

Claim (2) array nan/inf: accurate but again only for the invalid sigma=0 case.

Claim (3) is essentially wrong as stated: for large negative lam (x << mu), np.exp(-lam)->inf and the outer np.exp(-0.5*inf)->0.0. That 0.0 is the mathematically CORRECT tail limit of the Landau/Moyal density, not a silent wrong result. It only emits a numpy RuntimeWarning; the returned value is correct. So the alleged overflow bug is not a real correctness defect.

Context: landau is a fit-model function (lines 84-95) used with iminuit/curve_fit where the optimizer supplies positive scale parameters. Every sibling model in the file is identically unguarded by design — gaussian/sigma (L57), exponential/tau (L65), damped_oscillation/tau (L81), lorentzian_q/Q (L138). No valid (non-degenerate) input produces a silently wrong result or crash on valid input.

This is an edge/degenerate-only robustness nit, not P1 (no silent data loss/wrong result on valid input) and not really P2 (not an operationally meaningful edge case in a fitting context). Downgrade to P3.

#### F17 [P3] Division by zero when Q=0

- **箇所**: `gwexpy/fitting/models.py:138` — `lorentzian_q`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: `gamma = x0 / (2 * Q)` is unguarded. Scalar `Q=0` raises `ZeroDivisionError`. NumPy float `Q=0.0` produces `inf` for `gamma`, making the subsequent Lorentzian return `A` everywhere, which is a wrong but plausible-looking result.
- **再現**:
  ```python
  from gwexpy.fitting.models import lorentzian_q; lorentzian_q(1.0, 1.0, 100.0, 0.0)  # ZeroDivisionError for int Q
  ```
- **修正案**: Add `if Q == 0: raise ValueError('Q must be non-zero')` before computing `gamma`.
- **検証コメント**: Line 138 `gamma = x0 / (2 * Q)` is indeed unguarded, and the repro `lorentzian_q(1.0, 1.0, 100.0, 0.0)` does raise ZeroDivisionError (confirmed for both int 0 and float 0.0). So the literal divide-by-zero exists. However, the severity claim (P2) and the most alarming part of the description are overstated/incorrect:

1. The description claims "NumPy float Q=0.0 produces inf for gamma, making the subsequent Lorentzian return A everywhere, which is a wrong but plausible-looking result." This is FALSE. I verified: with numpy inputs, gamma=inf gives `inf**2 / ((x-x0)**2 + inf**2)` = inf/inf = NaN, not A. The result is an array of `nan` plus RuntimeWarnings (divide by zero / invalid value) — an obvious failure that surfaces the problem, not a silent plausible-looking wrong result. The claimed silent-corruption consequence does not occur.

2. Q=0 is not a valid input on any realistic path. Q is the quality factor of a resonance, which is strictly > 0; Q=0 is physically meaningless. Callers (scripts/make_*_notebook.py, fit_series) supply positive Q values / p0. There is no valid-input crash here, only a degenerate/invalid-input crash.

3. This unguarded-division pattern is the module-wide convention: gaussian uses `(x-mu)/sigma` (sigma=0 crashes), exponential uses `-x/tau` (tau=0 crashes), landau uses `(x-mu)/sigma`, lorentzian divides by gamma. None validate parameters by design — these are bare model functions for a fitter. lorentzian_q is consistent with its siblings.

Because the crash only occurs on an invalid/degenerate parameter (not valid input), and the headline "silent wrong result returning A everywhere" claim is factually incorrect (it returns NaN with warnings), this is a minor edge-case hardening item, not a P2. Corrected to P3.

#### F18 [P3] Returns None silently for non-string, non-callable input

- **箇所**: `gwexpy/fitting/models.py:241-244` — `get_model`
- **クラス**: `silent_failure` / **検証信頼度**: high / **breaking**: False
- **内容**: When `name` is neither a string nor callable (e.g. an integer, a list, or `None`), `get_model` returns `Python None` with no warning or error. The caller receives `None` and will encounter an unhelpful `TypeError: 'NoneType' object is not callable` far from the actual source of the mistake.
- **再現**:
  ```python
  from gwexpy.fitting.models import get_model; m = get_model(42); m(1.0)  # TypeError: 'NoneType' object is not callable
  ```
- **修正案**: Replace `return None` on line 244 with `raise TypeError(f'name must be a string or callable, got {type(name).__name__!r}')`.
- **検証コメント**: The code at models.py:241-244 does return None silently when `name` is neither str nor callable: `if not isinstance(name, str): if callable(name): return name; return None`. The claimed repro is accurate — `get_model(42)` returns None and calling it raises a deferred `TypeError: 'NoneType' object is not callable`.

However, three facts cap the severity at minor (P3) rather than anything higher:
1. The sole internal caller (gwexpy/fitting/core.py:1027) is guarded by `if isinstance(model, str): model = get_model(model_name)`, so this None path is never reachable through the package's own fitting flow. There is no silent-data-loss or wrong-result risk on any valid internal code path.
2. This is deliberate, documented behavior, not an accidental failure: tests/fitting/test_models.py:305-310 explicitly assert `get_model(None) is None` and `get_model(42) is None` (with a `# type: ignore[arg-type]` annotation acknowledging the misuse). The function signature is `str | Any`, and callable inputs are passed through intentionally.
3. The only way to hit it is an external caller directly passing a nonsensical, type-checker-flagged argument to the public function — an edge/misuse case, not a crash on valid input.

So the finding is technically real (a clearer TypeError/ValueError at the source would be friendlier API ergonomics), but the consequence is limited to a deferred, somewhat-unhelpful error for deliberate misuse that is already covered by tests as intended behavior. P3 is appropriate; not P1/P2 because no valid input path crashes or produces wrong results.

#### F19 [P3] MCMC branch creates a 2-panel figure layout but never populates the second panel

- **箇所**: `gwexpy/fitting/highlevel.py:299-303` — `_plot_bootstrap_fit`
- **クラス**: `data_integrity` / **検証信頼度**: high / **breaking**: False
- **内容**: When show_mcmc=True and result.samples is not None, n_plots is set to 2 and the code creates fig = plt.figure(figsize=(14,5)) with a 1×2 subplot grid, adding only ax1 = fig.add_subplot(1,2,1). The second subplot (1,2,2) is never created or filled. result.plot_corner() at line 357 opens a separate corner figure; plt.figure(fig.number) then switches back to the main figure. The main figure is displayed with a blank right panel: the (1,2) grid occupies the figure canvas but slot 2 is empty, producing a misleading and incomplete visualization every time run_mcmc=True.
- **再現**:
  ```python
  Call fit_bootstrap_spectrum(..., run_mcmc=True, plot=True) with a model that converges — result.samples will be populated, n_plots==2, and the saved/displayed figure will show a blank right half.
  ```
- **修正案**: Either (a) remove the 2-subplot layout entirely (the corner plot is already in its own figure via plot_corner()) so n_plots is always 1, or (b) add ax2 = fig.add_subplot(1, 2, 2) in the else-branch and draw the corner plot content into ax2. The simplest fix is (a): remove the n_plots==2 branch and always create a single-panel figure, relying on result.plot_corner() for the posterior visualization.
- **検証コメント**: Confirmed real and deterministic. In _plot_bootstrap_fit (highlevel.py), when show_mcmc and result.samples is not None, line 294 sets n_plots=2 and the else branch (lines 300-302) builds fig=plt.figure(figsize=(14,5)) with ax1=fig.add_subplot(1,2,1), storing only axes=[ax1]. No code ever creates subplot (1,2,2). At lines 355-358 the corner plot is drawn into its OWN separate figure via result.plot_corner(), then plt.figure(fig.number) switches back to the main 14x5 figure whose right half remains an empty canvas slot. The repro is reachable: fit_bootstrap_spectrum(run_mcmc=True, plot=True) runs MCMC at line 274 (populating result.samples) and calls _plot_bootstrap_fit with show_mcmc=True at line 284, so the wide-but-half-empty figure is produced every time. No guard prevents it.\n\nHowever the claimed severity (P2) and class (data_integrity) are wrong. The numerical fit, parameters, errors, and the corner plot itself are all correct and present; the only defect is a cosmetic layout flaw — the main figure reserves a 1x2 wide canvas but fills only the left panel, leaving blank space on the right. This is purely presentational with zero impact on data correctness, no crash, and no data loss, so it is P3 (minor), not P2, and not a data_integrity issue.

#### F20 [P3] np.linspace crashes with ValueError when frequencies array is empty

- **箇所**: `gwexpy/fitting/highlevel.py:328` — `_plot_bootstrap_fit`
- **クラス**: `numerical` / **検証信頼度**: medium / **breaking**: False
- **内容**: Line 328: x_plot = np.linspace(frequencies.min(), frequencies.max(), 200) where frequencies = psd.frequencies.value. If psd has been cropped to zero length (reachable when freq_range selects no bins and the empty psd is somehow passed to _plot_bootstrap_fit, or if the PSD itself is degenerate), numpy raises 'ValueError: zero-size array to reduction operation fmin (with no identity)' for .min() on a size-0 array. This is a crash on degenerate input that produces no informative error message about the root cause.
- **再現**:
  ```python
  Call _plot_bootstrap_fit directly with an empty FrequencySeries: from gwexpy.fitting.highlevel import _plot_bootstrap_fit; import numpy as np; from gwexpy.frequencyseries import FrequencySeries; from astropy import units as u; psd = FrequencySeries(np.array([]), frequencies=u.Quantity([], unit=u.Hz)); _plot_bootstrap_fit(mock_result, psd, False)  # crashes at frequencies.min()
  ```
- **修正案**: Add a guard before line 328: if len(frequencies) == 0: raise ValueError('Cannot plot: PSD has no frequency bins (empty after masking).') Alternatively, consolidate with the fix for the empty-mask in fit_bootstrap_spectrum (Finding 1), which prevents an empty psd from ever reaching this point.
- **検証コメント**: The mechanical claim is correct: line 328 `np.linspace(frequencies.min(), frequencies.max(), 200)` with `frequencies = psd.frequencies.value` (line 307) raises numpy's ValueError 'zero-size array to reduction operation fmin' when psd has length 0. No guard exists for empty psd in `_plot_bootstrap_fit` or its caller. So the bug is technically real and the numerical consequence is accurate.

However, severity P2 is overstated, for two reasons:
1. `_plot_bootstrap_fit` is a private helper (leading underscore) with exactly one internal caller (line 284, gated by `plot=True`). The only way to hit line 328 with an empty array specifically is the artificial direct-call repro shown, which hand-builds an empty FrequencySeries plus a mock result and bypasses the real pipeline.
2. In any realistic end-to-end path, an empty psd produced by `freq_range` selecting zero bins (lines 227-233, all-False mask) flows first into `fit_series` at line 259 (GLS fit: covariance-matrix inversion + multi-parameter optimization). Fitting 0 data points is degenerate and crashes/errors upstream before plotting is ever reached at line 328. So the np.linspace crash is not the operative failure point for valid-but-misconfigured pipeline usage.

This is a genuine but minor robustness gap on degenerate/misconfigured input only: no wrong results, no silent data loss, and no crash on valid input. Per the rubric (P2 = degenerate edge reachable via the real API; P3 = minor), it belongs at P3 because the line is reachable in practice only via direct private-helper invocation, while the natural pipeline fails earlier at fitting.

### statistics（9 件）

#### F21 [P2] Division by zero / NaN propagation when sigma2==0 for a frequency bin

- **箇所**: `gwexpy/statistics/gauch.py:87-98` — `compute_gauch`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: sigma2 = np.mean(window_asds**2, axis=0) / 2.0 is exactly 0 for any frequency bin whose ASD values are all zero in the current window (e.g., DC bin, Nyquist bin, a notched-out spectral line, or a synthetic test signal). The subsequent tcdf = 1.0 - np.exp(-(sorted_window**2) / (2.0 * sigma2)) then evaluates 0/0 (NaN) for zero-valued samples and x/0 (±Inf -> exp(-Inf)=0) for non-zero samples. The resulting NaN or wrong tcdf silently propagates into statistic_map and then into pvalue_map without any warning or exception.
- **再現**:
  ```python
  import numpy as np; from gwexpy.statistics.gauch import compute_gauch; from gwexpy.timeseries import TimeSeries; ts = TimeSeries(np.random.randn(4096), sample_rate=1024); # DC bin (index 0) of spectrogram.value is typically 0 -> sigma2[0]==0 -> NaN in pvalue_map[:,0]
  ```
- **修正案**: Guard sigma2 before the division: sigma2 = np.where(sigma2 > 0, sigma2, np.nan). This propagates NaN explicitly for dead bins rather than silently returning 0 or NaN through a 0/0 path. Alternatively raise/warn if any sigma2==0.
- **検証コメント**: The numerical defect at gauch.py:87-98 is genuine but mis-triaged. CONFIRMED: line 87 `sigma2 = np.mean(window_asds**2, axis=0) / 2.0` yields exactly 0 for any frequency bin whose ASD values are all zero across the full window, and line 98 `tcdf = 1.0 - np.exp(-(sorted_window**2) / (2.0 * sigma2))` then computes 0/0 -> NaN for the zero samples. I verified this directly: an all-zeros TimeSeries through the public `ts.gauch()` -> compute_gauch produces 100% NaN in statistic_map and pvalue_map, with only a transient `RuntimeWarning: invalid value encountered in divide` (easily suppressed, not an exception). No guard, epsilon, or validation exists anywhere in compute_gauch (gauch.py:33-129) or the caller `TimeSeries.gauch` (timeseries/_statistics.py:248). So silent NaN propagation is real.

HOWEVER the claimed repro and P1 rating are WRONG. The repro asserts "DC bin (index 0) of spectrogram.value is typically 0". This is false: I ran the exact repro (random TimeSeries, fftlength=1.0) and the windowed real spectrogram has nonzero power in EVERY bin including DC (min DC power ~8.6e-6, zero exact-zeros across all 513x40 bins), so sigma2 is never 0 and the full output contains NO NaN and emits NO warning when run in isolation (verified with `-W error::RuntimeWarning`). The earlier single warning I saw was a per-line dedup artifact from a zero-signal run in the same process. The defect therefore does NOT trigger on ordinary/valid GW data; it requires a genuinely degenerate input (literal all-zeros series, or a perfectly notched/synthetic bin with zero spectral leakage). That is an edge/degenerate condition, not silent corruption of valid results.

Per the rubric (P1 = silent wrong result on valid input; P2 = edge/degenerate only), this is P2: a legitimate robustness gap (the project's own scientific-workflow rules mandate epsilon/np.errstate guards on divisions) worth a defensive fix, but not a P1 because the stated trigger is unreachable with normal input.

#### F22 [P2] Division by zero (len(dist)==0) when n_monte_carlo=0 produces silent all-NaN output

- **箇所**: `gwexpy/statistics/rayleigh_test.py:59` — `rayleigh_pvalue`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: Line 59 computes `2.0 * np.minimum(upper_counts, lower_counts) / len(dist)`. When `n_monte_carlo=0` is passed to `rayleigh_pvalue`, `_get_rayleigh_stat_null_distribution` returns an empty array (len 0). NumPy then evaluates the integer division `0 / 0` element-wise, emitting only a RuntimeWarning and returning an all-NaN array. The function proceeds to `np.clip` and returns a Spectrogram filled with NaN — a wrong-but-plausible result with no exception or user-visible error.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.statistics.rayleigh_test import rayleigh_pvalue
  from gwexpy.spectrogram import Spectrogram
  spec = Spectrogram(np.ones((4, 8)), times=np.arange(4)*1.0, frequencies=np.arange(8)*1.0, unit='', name='test')
  result = rayleigh_pvalue(spec, n_samples=10, n_monte_carlo=0)  # returns all-NaN silently
  ```
- **修正案**: Add a guard at the top of `rayleigh_pvalue` (or inside `_get_rayleigh_stat_null_distribution`) before using `len(dist)`: `if len(dist) == 0: raise ValueError('n_monte_carlo must be >= 1; got 0')`. Alternatively guard in `_get_rayleigh_stat_null_distribution`: `if n_trials <= 0: raise ValueError(f'n_trials must be positive, got {n_trials}')`.
- **検証コメント**: Verified the code path. With n_monte_carlo=0, _get_rayleigh_stat_null_distribution (lines 75-95) sets null_stats=np.zeros(0), the for-loop over range(0) is skipped, and it caches/returns an empty array (the cache key is n_samples=10, so a fresh key produces a genuine cache miss and empty result — not masked by caching). Back in rayleigh_pvalue, line 55 upper_counts = 0 - searchsorted([], r_vals) = 0 and line 57 lower_counts = 0, so line 59 evaluates 2.0 * np.minimum(0,0) / len(dist) = 0.0 / 0 → element-wise NaN with only a RuntimeWarning. np.clip(NaN) stays NaN, so the function returns an all-NaN Spectrogram with no exception. No guard or validation exists anywhere in the function or its caller to prevent this. The claimed numerical consequence (silent all-NaN output) is accurate, and the repro reaches the line as described. However, the trigger is a degenerate input: n_monte_carlo=0 means zero background trials, which no valid scientific call would pass (default is 1000); negative values are similarly unguarded. This is an edge/degenerate robustness gap, not wrong results on valid input — so P2, not P1. A simple guard raising ValueError for n_monte_carlo < 1 (and ideally n_samples < 1) is the appropriate fix.

#### F23 [P2] n_samples=0 silently fills null distribution with NaN, causing wrong p-values

- **箇所**: `gwexpy/statistics/rayleigh_test.py:88-91` — `_get_rayleigh_stat_null_distribution`
- **クラス**: `silent_failure` / **検証信頼度**: high / **breaking**: False
- **内容**: When `n_samples=0` (i.e. `n=0`), `np.random.rand(0)` returns an empty array, so `s` is empty. `np.std([])` and `np.mean([])` both return NaN with RuntimeWarnings only. The entire `null_stats` array is filled with NaN, which `np.sort` preserves. Back in `rayleigh_pvalue`, `np.searchsorted` on an all-NaN sorted array returns 0 for every query, so `upper_counts = len(dist) - 0 = n_monte_carlo` and `lower_counts = 0`, making `p_vals = 0.0` for all bins — a completely wrong result silently passed to the caller as a valid Spectrogram.
- **再現**:
  ```python
  from gwexpy.statistics.rayleigh_test import _get_rayleigh_stat_null_distribution
  import numpy as np
  dist = _get_rayleigh_stat_null_distribution(0, 100)  # all NaN, no exception
  print(np.isnan(dist).all())  # True
  ```
- **修正案**: Add `if n <= 0: raise ValueError(f'n_samples must be >= 1, got {n}')` at the top of `_get_rayleigh_stat_null_distribution`. This is the natural entry-point guard; it also protects `rayleigh_pvalue` when called with `n_samples=0`.
- **検証コメント**: The mechanism is fully confirmed by reading the code and runtime verification. In `_get_rayleigh_stat_null_distribution` (lines 88-91), `n=0` makes `np.random.rand(0)` return an empty array, so `s` is empty. `np.std([])`/`np.mean([])` return NaN with only RuntimeWarnings (no exception), so every entry of `null_stats` becomes NaN (line 91), and `np.sort` (line 93) preserves NaN. Back in `rayleigh_pvalue`, the all-NaN sorted `dist` makes `np.searchsorted` return 0 for every finite query (NaN sorts to the end, so finite values insert at index 0): `upper_counts = len(dist) - 0 = n_monte_carlo`, `lower_counts = 0`, giving `p_vals = 2.0 * min(n,0)/n = 0.0` for all bins. I reproduced exactly this: empty s, NaN r, 5 RuntimeWarnings, and p_vals = [0,0,0]. The numerical/contract consequence in the report is accurate, and no guard or validation prevents it anywhere in the call path (lines 21-95). Downgrading from P1 to P2: `n_samples` is the number of segments used to compute the Rayleigh statistic, and `n_samples=0` is a degenerate/invalid input — a valid Rayleigh spectrogram cannot be derived from zero segments, and there are no internal callers passing 0. Per the rubric (P1 = wrong result on valid input; P2 = edge/degenerate only), this silent-failure-on-degenerate-input is P2. The missing input validation is a real gap worth a cheap fix (raise ValueError for n < 2, or at minimum n <= 0), but it does not corrupt results for legitimate usage.

#### F24 [P2] IndexError on empty p_value_map: times[0] and times[-1] crash when times is empty

- **箇所**: `gwexpy/statistics/dq_flag.py:79` — `to_segments`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: When p_value_map contains zero time steps (e.g. after frequency crop leaves an empty array, or an empty Spectrogram is passed), `times = p_value_map.times.value` is an empty array. The loop on line 56 does nothing, but line 79 unconditionally accesses `times[0]` and `times[-1]`, raising IndexError. The fallback `dt = 1.0` on line 51 does not prevent the crash because that branch only guards `len(times) > 1` vs `== 1`, not `== 0`.
- **再現**:
  ```python
  import numpy as np
  from gwexpy.statistics.dq_flag import to_segments
  from gwexpy.spectrogram import Spectrogram
  # empty spectrogram: 0 time steps, 4 freq bins
  sp = Spectrogram(np.empty((0, 4)), times=np.array([]), frequencies=np.array([1,2,3,4]), unit='')
  to_segments(sp)  # -> IndexError: index 0 is out of bounds for axis 0 with size 0
  ```
- **修正案**: Add an early-return guard after computing `times`:
```python
if len(times) == 0:
    name = f"{p_value_map.name}_veto" if p_value_map.name else "non_gaussian_veto"
    return DataQualityFlag(name=name, active=SegmentList(), known=SegmentList())
```
Place this immediately after `times = p_value_map.times.value` (after line 50).
- **検証コメント**: Confirmed real and reproduced. In /home/washimi/work/gwexpy/gwexpy/statistics/dq_flag.py, line 50 sets `times = p_value_map.times.value`. The dt guard on line 51 only distinguishes `len(times) > 1` from else (covering both 1 and 0), so it does NOT protect against the empty case. The loop on line 56 over `is_bad` is a no-op for empty input, but line 79 unconditionally evaluates `Segment(times[0]-dt/2.0, times[-1]+dt/2.0)`, indexing `times[0]` and `times[-1]`. When `times` is empty this raises `IndexError: index 0 is out of bounds for axis 0 with size 0`. I reproduced this exactly with the provided empty-Spectrogram repro. I also verified the len==1 case does NOT crash (times[0] and times[-1] both valid), so only the 0-length case is affected. No upstream guard prevents this. One minor inaccuracy in the report: cropping the frequency axis (line 43) does not empty the time axis, so that specific sub-scenario cannot trigger it; the real trigger is passing/constructing a Spectrogram with 0 time steps (the second scenario described), which is a valid code path. Severity P2 is correct: it requires a degenerate empty input (no valid non-empty data is affected), and it fails loudly with IndexError rather than producing silent wrong results, so it does not meet the P1 bar of crash-on-valid-input or silent data corruption.

#### F25 [P2] dt fallback to 1.0 for single-time-step input produces wrong segment boundaries

- **箇所**: `gwexpy/statistics/dq_flag.py:51` — `to_segments`
- **クラス**: `silent_failure` / **検証信頼度**: high / **breaking**: False
- **内容**: When the spectrogram has exactly one time step (`len(times) == 1`), the code falls back to `dt = 1.0` (line 51). This hardcoded value is used to compute `active_seg_start = times[0] - 0.5` and the `known` segment as `[times[0]-0.5, times[0]+0.5]`. If the actual time resolution is, say, 0.25 s, the reported segment spans 1.0 s instead of 0.25 s — a factor-of-4 error in segment duration. No warning is emitted.
- **再現**:
  ```python
  # Single time step at t=100.0 with true dt=0.25
  import numpy as np
  from gwexpy.spectrogram import Spectrogram
  from gwexpy.statistics.dq_flag import to_segments
  sp = Spectrogram(np.array([[0.01]]), times=np.array([100.0]),
                   frequencies=np.array([10.0]), unit='')
  sp.dt  # suppose it carries dt=0.25 from construction
  flag = to_segments(sp, alpha=0.05)
  # flag.known covers [99.5, 100.5] — 1.0 s wide instead of the correct 0.25 s
  ```
- **修正案**: Read `dt` from the spectrogram metadata instead of recomputing it from array differences:
```python
try:
    dt = p_value_map.dt.value
except Exception:
    dt = times[1] - times[0] if len(times) > 1 else float(np.nan)
```
This uses the object's own dt attribute and avoids the arbitrary 1.0 fallback.
- **検証コメント**: Confirmed real. In /home/washimi/work/gwexpy/gwexpy/statistics/dq_flag.py line 51, `dt = times[1] - times[0] if len(times) > 1 else 1.0` computes the time resolution from the times array and silently falls back to the hardcoded 1.0 when only one time sample exists. This dt then drives the half-bin offsets at lines 59, 62, 69 and the `known` interval at line 79, so a single-bin spectrogram with true resolution 0.25 s yields a 1.0 s-wide segment (4x error), with no warning. The bug is genuine because: (1) no guard prevents it — `to_segments` is a public entry point and the function never references `p_value_map.dt`, which the Spectrogram does expose (matrix_core.py line 95 maps `dt`->`dx`, and spectrogram.py shows `dt`/`dx` is real construction metadata, e.g. `dt=0.25*u.s`); (2) the repro input (one time step) reaches line 51's else-branch directly; (3) the numerical consequence (segment width = 1.0 s instead of true dt) is accurate. Correct fix would be to read `p_value_map.dt` instead of fabricating 1.0. However this is strictly a degenerate single-time-bin case — typical spectrograms have many time bins and hit the correct `times[1]-times[0]` branch — and even in the single-bin case the result is only wrong when dt != 1.0 (the common default 1.0 s is coincidentally correct). It produces a wrong boundary, not a crash or data loss on typical valid input, so it stays P2 (edge/degenerate only), matching the claimed severity.

#### F26 [P2] Bare except swallows all stats.t.fit failures silently, filling nu_map with NaN without warning

- **箇所**: `gwexpy/statistics/student_t_indicator.py:112-113` — `compute_student_t_nu`
- **クラス**: `silent_failure` / **検証信頼度**: high / **breaking**: False
- **内容**: The `except Exception` block on line 112 catches every exception from `stats.t.fit(samples)` — including convergence failures, singular-fit conditions, and unexpected scipy errors — and silently stores `np.nan` in `nu_map[i, j]`. No warning or log message is emitted. A caller processing a long time series where the fit fails for many bins will receive a Spectrogram silently riddled with NaN values, with no indication that anything went wrong. This is especially dangerous because downstream code such as `to_segments` calling `np.any(p_value_map.value < alpha, axis=1)` will treat NaN comparisons as False, silently dropping those bins from the veto.
- **再現**:
  ```python
  # Force a constant (zero-variance) segment so t.fit raises
  import numpy as np
  from unittest.mock import patch
  from scipy import stats
  with patch.object(stats.t, 'fit', side_effect=RuntimeError('fit failed')):
      # all calls to stats.t.fit will raise, entire nu_map will be NaN
      result = compute_student_t_nu(ts, fftlength=1.0, window=40)
  # result.value is all NaN with no warning
  ```
- **修正案**: Replace the bare except with a logged warning:
```python
import warnings
try:
    nu, _, _ = stats.t.fit(samples)
    nu_map[i, j] = nu
except Exception as exc:  # noqa: BLE001
    warnings.warn(
        f"stats.t.fit failed at time_idx={i}, freq_idx={j}: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )
    nu_map[i, j] = np.nan
```
- **検証コメント**: The code pattern is real: student_t_indicator.py lines 109-113 wrap stats.t.fit in a bare `except Exception` that stores np.nan with no warning or log. The downstream consequence is also factually correct — I verified np.nan < 0.05 evaluates to False, so to_segments (dq_flag.py line 47, np.any(p_value_map.value < alpha, axis=1)) silently drops NaN bins from the veto. However, adversarial checks weaken the severity: (1) The claimed repro is wrong — a constant/zero-variance segment does NOT make t.fit raise; stats.t.fit(np.ones(80)) returns a finite (39.1, 1.0, ~2e-18). The except path is only reachable via unittest.mock forcing side_effect=RuntimeError, i.e., a synthetic exception that does not arise from the documented valid input. So no crash/failure on valid input is demonstrated. (2) scipy.stats.t.fit is a robust MLE that rarely raises on finite real-valued data; it typically returns a poor fit or warns rather than raising, making this a degenerate/rare-path issue, not a common one. (3) The NaN-as-False behavior is conservative for a veto (drops uncertain bins rather than falsely vetoing). This is a genuine silent-failure/robustness defect worth fixing (emit a warning instead of silent swallow), but it is an edge/degenerate-path issue, not silent data loss or wrong results on valid/typical input. Hence P2, not P1.

#### F27 [P3] p-value of exactly 0.0 returned when dn exceeds all Monte Carlo samples

- **箇所**: `gwexpy/statistics/gauch.py:114` — `compute_gauch`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: pvalue_map = (len(dist) - indices) / len(dist) yields 0.0 when the observed statistic dn is larger than every value in the cached Lilliefors distribution. A Monte Carlo p-value of 0.0 is numerically invalid; the statistically correct lower bound is 1/n_trials. The value 0.0 causes -log10(pvalue) = Inf in any downstream significance map, silently producing infinity in derived data products.
- **再現**:
  ```python
  import numpy as np; from gwexpy.statistics.gauch import _get_rayleigh_lilliefors_pvalue; p = _get_rayleigh_lilliefors_pvalue(1e6, n=40, n_trials=10); assert p == 0.0  # triggers the defect
  ```
- **修正案**: Replace line 114 with: pvalue_map = np.maximum((len(dist) - indices) / len(dist), 1.0 / len(dist)). Apply the same fix in the return value of _get_rayleigh_lilliefors_pvalue (line 149): return float(max(np.sum(dist >= dn) / len(dist), 1.0 / len(dist)))
- **検証コメント**: The underlying mechanism is real: line 114 `pvalue_map = (len(dist) - indices) / len(dist)` returns exactly 0.0 when np.searchsorted returns len(dist), i.e. when the observed dn exceeds every cached Lilliefors sample. There is no 1/n_trials floor anywhere in the file (line 149 in _get_rayleigh_lilliefors_pvalue has the same omission). A Monte Carlo p-value of 0.0 is indeed statistically improper; the correct floor is 1/n_trials. So a defect exists.

However, the report overstates it and the repro is invalid: (1) The repro calls _get_rayleigh_lilliefors_pvalue, which uses line 149 (np.sum(dist >= dn)/len(dist)), NOT the cited line 114 in compute_gauch — different code path. (2) The repro input dn=1e6 is non-physical: dn = max|ecdf - tcdf| with both CDFs in [0,1], so dn is always in [0,1]; it can never be 1e6. The genuine trigger is dn within [0,1] exceeding all MC samples (plausible only with strongly non-Gaussian data and small n_monte_carlo). (3) The claimed concrete consequence (-log10(p)=Inf in derived products) does NOT occur in any in-repo consumer: the sole downstream user, plot_gauch_dashboard line 66, already guards with np.clip(p_map.value, 1e-10, 1.0) before -log10. The Inf risk applies only to hypothetical user-side code.

Net: a genuine but minor statistical-correctness edge case (missing 1/n_trials floor), triggered only in a degenerate scenario, with the only in-repo downstream path already protected. This is P3, not P2: no silent data corruption or wrong result on valid typical input, and the claimed consequence is already mitigated where it would matter.

#### F28 [P3] np.random.rand(n) can return 0, causing log(0)=-Inf -> Inf in null distribution

- **箇所**: `gwexpy/statistics/gauch.py:140` — `_get_rayleigh_lilliefors_pvalue`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: np.random.rand samples from [0, 1). The value 0.0 is representable and has probability ~2^-53 per draw but is non-negligible over many trials. np.log(0) = -Inf, np.sqrt(Inf) = Inf. That Inf enters null_dns[i], and after np.sort the cached dist ends with Inf. Any finite observed dn will then have np.searchsorted return an index less than len(dist), making the count of dist>=dn include Inf, which is harmless numerically, but the corrupted Inf entry permanently occupies the tail of the null distribution, shifting all quantile estimates and p-values for this window size for the lifetime of the process.
- **再現**:
  ```python
  import numpy as np; np.random.seed(0); vals = np.random.rand(40); # repeatedly draw until a 0 appears; null_sample = np.sqrt(-2.0 * np.log(vals)); # if vals contains 0: null_sample contains Inf
  ```
- **修正案**: Replace np.random.rand(n) with np.random.uniform(low=np.finfo(float).tiny, high=1.0, size=n), or clip: u = np.clip(np.random.rand(n), np.finfo(float).tiny, 1.0); null_sample = np.sqrt(-2.0 * np.log(u))
- **検証コメント**: Line 140 `np.sqrt(-2.0*np.log(np.random.rand(n)))` has no guard against rand() returning exactly 0.0, so log(0)=-inf can occur. There IS a real latent defect: a zero draw corrupts the cached Lilliefors null distribution for the lifetime of the process (cache populated at lines 136-146, keyed by window size). Empirically confirmed the corrupted slot shifts p-values vs a clean distribution. HOWEVER the finding's described mechanism is wrong: it is NOT inf that enters null_dns[i] and the cached dist does NOT end with inf. Because the inf propagates into s2_est at line 141 (np.mean(inf)/2 = inf), the division at line 144 `-(sorted_null**2)/(2.0*inf)` yields nan at the inf-entry (-inf/inf), and np.max returns nan (line 145). So null_dns[i] = nan, and np.sort places nan at the END of the cached array (verified). Net effect on p-values: in the searchsorted path (line 113) and the scalar np.sum(dist>=dn) path (line 149) the nan slot inflates the denominator len(dist) while contributing nothing useful, biasing p-values by ~1/n_trials (~0.1% with default 1000 trials). Severity is overstated as P2: probability of rand()==0.0 is ~2^-53 per draw; over n*n_trials (e.g. 40*1000=40000) draws the expected number of zeros is ~4e-12 — effectively unreachable in any realistic process lifetime, not 'non-negligible' as claimed. The repro is also non-deterministic ('repeatedly draw until a 0 appears') and would require ~2^53 iterations. This is a genuine but degenerate numerical-robustness edge case with negligible impact magnitude: P3, not P2. A trivial guard (np.clip rand to a tiny epsilon, or 1.0 - rand) would close it.

#### F29 [P3] n_monte_carlo argument silently ignored on cache hit for the same window size

- **箇所**: `gwexpy/statistics/gauch.py:106-107` — `compute_gauch / _get_rayleigh_lilliefors_pvalue`
- **クラス**: `api_contract` / **検証信頼度**: high / **breaking**: False
- **内容**: _LILLIEFORS_CACHE is keyed only by n (window size). If a first call uses n_monte_carlo=100 (or the default 1000), and a later call specifies n_monte_carlo=10000 for higher precision, the cached low-resolution distribution is returned without warning. The caller's explicit n_monte_carlo value is silently discarded, violating the documented API contract.
- **再現**:
  ```python
  from gwexpy.statistics.gauch import _LILLIEFORS_CACHE, _get_rayleigh_lilliefors_pvalue; _get_rayleigh_lilliefors_pvalue(0.3, n=40, n_trials=10); _get_rayleigh_lilliefors_pvalue(0.3, n=40, n_trials=100000); len(_LILLIEFORS_CACHE[40])  # returns 10, not 100000
  ```
- **修正案**: Key the cache as (n, n_trials) instead of just n: replace `if n not in _LILLIEFORS_CACHE` with `if (n, n_trials) not in _LILLIEFORS_CACHE` and update all accesses to _LILLIEFORS_CACHE[(n, n_trials)]. Alternatively document that the cache is not keyed by n_trials and warn when a caller supplies a different n_trials than the cached one.
- **検証コメント**: Confirmed real. `_LILLIEFORS_CACHE` is keyed only by window size `n` (line 136: `if n not in _LILLIEFORS_CACHE:`). The null distribution is generated using `n_trials` exclusively on a cache miss (lines 138-146); on a cache hit it returns the previously stored distribution (lines 148-149) and the new `n_trials` value is never consulted. `compute_gauch` passes its `n_monte_carlo` argument straight through to this function (line 106), so the documented `n_monte_carlo` parameter ("Number of Monte Carlo trials for background distribution") is silently discarded on any subsequent call sharing the same `window`. No guard or validation prevents this. The repro is accurate: after a first call with n_trials=10 stores a size-10 distribution under key 40, a second call with n_trials=100000 returns the cached size-10 distribution, so `len(_LILLIEFORS_CACHE[40])` is 10. Severity is correctly P3, not higher: the consequence is a coarser/lower-resolution background distribution than requested, which yields more quantized but still statistically valid p-values — not a wrong mathematical result, silent data loss, or crash. It also only manifests under an uncommon cross-call usage pattern (same window, changing n_monte_carlo within one process), and the default/first-use path always honors the requested value. This is a genuine but minor API-contract issue.

### spectral（3 件）

#### F30 [P2] block_size > n_time silently passes through the guard and then crashes with ValueError in np.random.randint

- **箇所**: `gwexpy/spectral/estimation.py:548-566` — `bootstrap_spectrogram`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: Lines 549-553 detect `block_size >= n_time` and execute `pass`, providing no error or fallback. Execution continues to line 559 where `num_possible_blocks = n_time - block_size + 1` evaluates to 0 (when block_size == n_time + 1 or larger), and line 565 calls `np.random.randint(0, num_possible_blocks, ...)` with `high=0`, which raises `ValueError: high <= 0`. The crash message gives no hint that the block size is too large for the input.
- **再現**:
  ```python
  import numpy as np; from gwexpy.spectral.estimation import bootstrap_spectrogram; from gwexpy.spectrogram import Spectrogram; from astropy import units as u; spec = Spectrogram(np.random.rand(5, 10), dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz); bootstrap_spectrogram(spec, n_boot=10, block_size=6)
  ```
- **修正案**: Replace the `pass` at line 553 with an explicit raise: `raise ValueError(f'block_size ({block_size} samples) must be smaller than the number of time bins ({n_time}).')` or clamp to `min(block_size, n_time - 1)` with a warning.
- **検証コメント**: Confirmed by running the exact repro: it raises `ValueError: high <= 0`. The guard at estimation.py:549-553 only executes `pass` for `block_size >= n_time`, providing no validation or fallback. Note block_size is in SECONDS, converted to samples at lines 542-545 (parse_fftlength_or_overlap(6, sample_rate=1/dt) -> 6 samples for the repro's dt=1s). Execution then continues to line 559 where `num_possible_blocks = n_time - block_size + 1`; for n_time=5 and block_size>5 this is <=0, and line 565 `np.random.randint(0, num_possible_blocks, ...)` raises ValueError with `high <= 0`, which gives no hint that block_size is too large. One refinement to the report: the boundary is block_size > n_time (strictly greater) that crashes; block_size == n_time yields num_possible_blocks=1 and works fine (verified: bs=5 OK, bs=6/7 crash). Severity stays P2: this is a degenerate misconfiguration (block window longer than the entire spectrogram duration), not a crash on normal valid input or silent wrong result. The genuine defect is the unhelpful error message instead of a clear validation error/fallback. Lines cited: 548-553 (no-op guard), 559 (zero/negative count), 562-566 (randint crash); parse conversion at 534-545.

#### F31 [P2] block_size > n_time causes np.random.randint(low, high<=0) crash

- **箇所**: `gwexpy/spectral/estimation.py:548-565` — `bootstrap_spectrogram`
- **クラス**: `numerical` / **検証信頼度**: high / **breaking**: False
- **内容**: When block_size (in samples, after conversion) is greater than n_time, num_possible_blocks = n_time - block_size + 1 evaluates to a value <= 0. The guard at line 549–553 detects this condition but does nothing (bare `pass`), so execution falls through to line 565 where np.random.randint(0, num_possible_blocks, ...) is called with high <= 0. NumPy raises ValueError: 'low >= high' at that point. The block_size='auto' path can also reach this: if _infer_overlap_ratio returns a ratio > n_time, block_size becomes >= n_time.
- **再現**:
  ```python
  import numpy as np
  from astropy import units as u
  from gwexpy.spectrogram import Spectrogram
  from gwexpy.spectral.estimation import bootstrap_spectrogram
  
  data = np.random.rand(5, 10)  # n_time=5
  spec = Spectrogram(data, dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
  # block_size=6 > n_time=5 → num_possible_blocks=0 → crash
  bootstrap_spectrogram(spec, n_boot=10, block_size=6.0)
  ```
- **修正案**: Replace the `pass` at line 553 with an early-exit or a clamp. For example:
```python
if block_size >= n_time:
    warnings.warn(
        f"block_size ({block_size}) >= n_time ({n_time}); "
        "falling back to standard (iid) bootstrap.",
        RuntimeWarning,
    )
    block_size = None
```
Then the `if block_size is not None and block_size > 1:` branch is skipped and the standard iid path at line 597 is used instead.
- **検証コメント**: Confirmed real. In /home/washimi/work/gwexpy/gwexpy/spectral/estimation.py, bootstrap_spectrogram sets n_time = data.shape[0] (line 463) and the only validation on block_size vs n_time is the guard at lines 549-553 (`if block_size >= n_time:` whose body is a bare `pass`). When block_size > n_time, line 559 computes num_possible_blocks = n_time - block_size + 1 <= 0, and line 565 calls np.random.randint(0, num_possible_blocks, (n_boot, num_blocks_needed)) with high <= 0, which raises ValueError: 'low >= high'.

Repro reachability verified: with dt=1.0*u.s, sample_rate=1.0; parse_fftlength_or_overlap(6.0, sample_rate=1.0) returns samples=round(6.0)=6 (fft_args.py lines 91-93, ~115). So block_size becomes 6, n_time=5, num_possible_blocks=0 → crash at line 565. No earlier validation rejects this (lines 465-477 only check n_time<2, n_boot, ci, method). The repro reaches the code as claimed.

Two precision notes on the description: (1) The crash requires block_size strictly > n_time. The boundary case block_size == n_time gives num_possible_blocks = 1, and randint(0, 1, ...) succeeds (returns zeros). So the guard's stated condition (`>=`) is broader than the actual crash condition (`>`), but this does not affect the verdict since the guard does nothing in either case. (2) The 'auto' path (lines 525-533: block_size = int(np.ceil(ratio)) from _infer_overlap_ratio = duration/stride) can indeed produce block_size > n_time for spectrograms whose duration/dt ratio exceeds the number of time bins, so that secondary route is also plausible.

Severity P2 (not P1): this is a degenerate/misconfiguration input (block window longer than the available number of time segments), not a crash on ordinary valid input. The author's existing-but-empty guard shows the boundary was anticipated. It is a hard crash (ValueError), so it is not P3. P2 (edge/degenerate only) is the correct rating.

#### F32 [P2] All-NaN column with ignore_nan=True silently produces NaN center and CI without any function-level warning

- **箇所**: `gwexpy/spectral/estimation.py:81-96` — `_bootstrap_resample_py`
- **クラス**: `silent_failure` / **検証信頼度**: high / **breaking**: False
- **内容**: When ignore_nan=True and a bootstrap-resampled column col is entirely NaN (e.g. an entire frequency bin of the spectrogram is NaN), np.nanmedian(col) / np.nanmean(col) return NaN and emit only a low-level NumPy RuntimeWarning. The NaN propagates into resampled_stats and then into center, err_low, err_high of the returned FrequencySeries without any diagnostic from bootstrap_spectrogram itself. The caller receives a plausible-looking FrequencySeries with NaN values at the affected frequencies and no indication of which bins were problematic or why.
- **再現**:
  ```python
  import numpy as np
  from astropy import units as u
  from gwexpy.spectrogram import Spectrogram
  from gwexpy.spectral.estimation import bootstrap_spectrogram
  
  data = np.random.rand(20, 5)
  data[:, 2] = np.nan  # entire frequency bin 2 is NaN
  spec = Spectrogram(data, dt=1.0*u.s, f0=0*u.Hz, df=1*u.Hz)
  result = bootstrap_spectrogram(spec, n_boot=50, ignore_nan=True)
  # result.value[2] is NaN with no logger warning from the function
  ```
- **修正案**: After the _bootstrap_resample_* call, check for all-NaN columns and issue a logger.warning:
```python
nan_freq_mask = np.all(np.isnan(resampled_stats), axis=0)
if np.any(nan_freq_mask):
    logger.warning(
        "%d frequency bin(s) are entirely NaN in all bootstrap resamples "
        "(e.g. entire input column was NaN). Output will contain NaN at "
        "those frequencies.",
        int(np.sum(nan_freq_mask)),
    )
```
- **検証コメント**: Confirmed real. In /home/washimi/work/gwexpy/gwexpy/spectral/estimation.py, bootstrap_spectrogram has no guard preventing an all-NaN frequency column from silently producing NaN center/CI when ignore_nan=True.

Path trace for the repro (20 time bins, frequency bin 2 all-NaN, ignore_nan=True): (1) Input validation at lines 465-477 only checks n_time>=2, n_boot>=1, 0<ci<1, and method; no NaN check (NaN is intentionally allowed via ignore_nan). The repro passes all checks. (2) The non-stationarity heuristic at lines 481-490 computes np.nanmean(data, axis=1) — averaging across frequency per time bin — so with 4 of 5 bins valid, segment_avgs is finite and no warning fires; there is no per-frequency-column all-NaN check anywhere. (3) In _bootstrap_resample_py (lines 88-95), every resample of column 2 contains only NaN, so np.nanmedian(col)/np.nanmean(col) returns NaN, emitting only a low-level NumPy RuntimeWarning. (4) center via np.nanmedian/np.nanmean (lines 626/631) and CI via np.nanpercentile (lines 637-638) over the all-NaN column return NaN. (5) These flow into the returned FrequencySeries center/error_low/error_high (lines 712-729) with no logger warning or error from the function. The claimed numerical and contract consequences are all accurate.

Severity: P2 is correct, not P1. The output is NaN (a visible, inspectable signal), not a plausible-but-wrong number, so it is not silent data corruption on valid input. The function explicitly supports NaN and handles partial-NaN columns (the typical case) correctly; only a fully-NaN frequency bin degenerates. This is an edge/degenerate case matching the P2 definition, not P1 (which requires silent wrong result/data loss on typical valid input or a crash).

## 棄却（3 件）

敵対的検証で否定。特に #2 は finder の主張が数学的に誤っていた点を訂正している。

1. **`gwexpy/fitting/models.py:73`** — Silent nan when x < 0 and alpha is non-integer
   - 棄却理由: The finding's central premise — "No warning is issued under the default np.errstate ... silently wrong" — is factually false. I verified at /home/washimi/work/gwexpy/gwexpy/fitting/models.py:73 (`return A * x**alpha`). NumPy's default errstate is {'invalid': 'warn'} (confirmed via np.geterr()), so evaluating `power_law(np.array([-1.0, 1.0]), 1.0, 0.5)` emits `RuntimeWarning: invalid value encountered in sqrt` pointing at line 73 — both via the warnings machinery and to stderr under default Python warning filters. The result is array([nan, 1.]), but it is NOT silent; the "silent_failure" classification is incorrect.

The remaining substance is: out-of-domain input (negative float x with non-integer alpha) yields nan. This is the IEEE/NumPy-correct result for a real-valued power on a mathematically ill-posed input (x^0.5 for x<0 is not real). Every other model in the file (gaussian, exponential, landau, etc.) is likewise an unguarded minimal fitting primitive, and the same nan-with-warning behavior is universal across NumPy (`np.sqrt(-1.0)`). Power-law fitting models take positive independent variables (frequency, time, amplitude); negative x is a degenerate misuse, and the code already surfaces it via a warning. No guard is needed to satisfy any documented contract, and adding one would diverge from ecosystem-standard NumPy semantics. Not a P1 (no silent wrong result — it warns), not a P2 (warned, IEEE-correct, degenerate-only). Reject.

2. **`gwexpy/spectral/estimation.py:249-257`** — VIF weight denominator uses n_blocks instead of the actual loop upper limit, underestimating error-bar inflation when max_lag < n_blocks
   - 棄却理由: The finding inverts the statistics. The weight `(1 - k/n_blocks)` at line 256 is correct; the denominator MUST be the number of averaged blocks (n_blocks), not the loop's upper bound.

The variance of the mean of N correlated, identically-distributed estimates is the textbook identity Var(mean) = (sigma^2/N)[1 + 2 * sum_{k=1}^{N-1} (1 - k/N) r(k)]. The factor (1 - k/N) arises purely from counting block pairs: there are (N - k) pairs separated by lag k out of N^2 total, giving 2*(N-k)/N^2 = (2/N)(1 - k/N). N here is unambiguously the number of blocks = n_blocks. So weight = 1 - k/n_blocks (line 256) is exactly right.

The loop bound `min(n_blocks, max_lag)` (line 249) with max_lag = ceil(nperseg/step) (line 247) is a pure truncation/optimization: for k >= max_lag the window-shift `shift = k*step >= nperseg`, so the two window copies don't overlap and the autocorrelation rho(k*step) is exactly zero (line 251-252 also explicitly breaks on this). Skipping those zero-valued terms changes nothing. It is NOT the summation upper limit M of the variance formula.

For the repro (nperseg=1000, noverlap=800, n_blocks=100): step=200, max_lag=5, loop runs k=1..4 with weights (1 - k/100) ≈ 1, which is correct — with 100 blocks the pair-count correction is negligible and the correlated lags should contribute nearly fully. The finding's proposed "fix" would use (1 - k/5), which both (a) is the statistically wrong quantity (lag truncation point has nothing to do with the pair-count normalization) and (b) would spuriously suppress the correlation correction. The code's ~1.546 is the correct value; the claimed "correct" 1.4412 is the buggy value.

The docstring's symbol `M` (line 181) is loosely written and conflates the truncation with the block count, but the implementation is correct. No guard issue, repro reaches the code, but the claimed consequence is backwards. Reject.

3. **`gwexpy/spectral/estimation.py:139-151`** — _to_seconds silently returns None on Quantity unit-conversion failure, disabling the fftlength > duration guard
   - 棄却理由: The mechanism is partially accurate but the claimed consequence is false. VERIFIED: `_to_seconds(4.0 * u.Hz)` does return None, because astropy's UnitConversionError is a subclass of ValueError (MRO: UnitConversionError -> UnitsError -> ValueError), so the `except (AttributeError, TypeError, ValueError)` clause at lines 145-147 catches it. Consequently `fftlength_sec is None`, the guard at line 155 short-circuits, and the `duration < fftlength_sec` check at line 157 is skipped. So far the report's mechanism holds.

HOWEVER, the central claimed impact — that the invalid fftlength "may produce a wrong result rather than a clear error" with "behavior depend[ing] entirely on GWpy internals" — does NOT occur. I ran the exact repro `estimate_psd(ts, fftlength=4.0*u.Hz)` and every non-time Quantity variant (Hz, m, dimensionless): all raise a loud `ValueError` from GWpy ("Quantity truthiness is ambiguous...") rather than silently returning a wrong PSD. There is no silent data loss, no wrong result, and no crash on valid input. Note `4.0 * u.Hz` is invalid input (wrong dimension), not valid input.

The only real effect is cosmetic: for a wrong-dimensioned Quantity the user gets GWpy's generic ValueError instead of a tailored message — and estimate_psd's own message ("fftlength must not exceed data duration") would itself be misleading for a dimension error anyway. The valid case `8.0 * u.s` correctly triggers the duration guard with the proper message.

The `silent_failure` classification is inaccurate: the swallowed exception inside `_to_seconds` is immediately followed by a loud downstream failure. There is no silent failure observable to the user. Rejecting: the bug as described (silent skip leading to wrong result) does not reproduce; impact is below P3 (cosmetic error-message quality, not a defect). A reasonable code-quality improvement would be to validate fftlength dimensionality explicitly, but that is an enhancement, not a bug.

## 網羅性 critic — 取りこぼし / 追補候補

本 sweep が当てきれなかった領域。**最重要は `statistics/roc.py` の完全な取りこぼし**
（finder が Monte-Carlo 構造の 4 ファイルで止まり roc.py に未着手。初期スポット確認で見つけた
`tp/n_pos`・`fp/n_neg` のゼロ除算が残存）。次回 sweep / 修正計画で優先する。

### gwexpy.statistics（9 ギャップ）
- **roc.py / calculate_roc (whole function, lines 14-51)**
  - 取りこぼし理由: The confirmed findings list contains ZERO entries for roc.py. The sweep appears to have stopped at the four files with obvious Monte-Carlo / Spectrogram structure and never applied any bug-class lens to roc.py. Yet calculate_roc has multiple degenerate-input numerical bugs: (a) line 24 np.linspace(np.min(y_score), np.max(y_score), n_points) raises/produces a degenerate single-point grid when all y_score are equal (constant scores -> min==max -> thresholds all identical -> AUC meaningless); (b) np.min/np.max raise ValueError on empty y_score; (c) y_score containing NaN (which is exactly what nanmax can still leave, or what upstream maps produce) makes min/max NaN and every comparison y_score >= thresh False, silently yielding AUC that is not flagged.
  - 追補: Apply the numerical + silent_failure lens to calculate_roc. Add tests: all-equal y_score, empty arrays, y_score with NaN/Inf, single-element arrays. Check whether n_pos/n_neg==0 early-return value 0.5 is the right AUC convention vs raising. Verify AUC sign/orientation when fpr ties exist (argsort on duplicate fpr can interleave tpr non-monotonically, biasing the trapezoid integral).
- **roc.py / calculate_roc AUC integration with tied FPR (lines 44-50)**
  - 取りこぼし理由: data_integrity lens not applied: the sweep focused on division-by-zero and NaN classes, not on correctness of the integration itself. Sorting only by fpr (np.argsort(fpr)) without a secondary sort on tpr means that at a given FPR the TPR values may be ordered arbitrarily, and np.trapezoid over a non-monotone-within-tie sequence gives a biased/incorrect AUC. Also AUC can legitimately come out > 1 or < 0 with pathological inputs and is never clipped or validated.
  - 追補: Add a regression test comparing calculate_roc AUC against sklearn.metrics.roc_auc_score on random data with many tied scores; assert agreement within tolerance. Check that secondary sort key (tpr) is needed and that AUC is bounded in [0,1].
- **roc.py / evaluate_detection_performance (lines 54-86)**
  - 取りこぼし理由: api_contract + silent_failure lens not applied to this function at all. It assumes glitch_generator accepts an A1=0 kwarg to mean 'clean' (line 66 comment 'Assuming 0 amplitude is clean') -- a hardcoded, undocumented contract that silently breaks for any generator without an A1 parameter. It also assumes score maps expose .value and that np.nanmax over an all-NaN map is meaningful (np.nanmax on all-NaN raises RuntimeWarning and returns nan, which then poisons calculate_roc). n_trials=0 yields empty y_true/y_score passed to calculate_roc -> empty min/max crash.
  - 追補: Document/validate the glitch_generator contract (does A1 always exist?). Add tests for: generator without A1 kwarg, method_func returning an all-NaN map, n_trials=0. Guard np.nanmax against all-NaN slices.
- **student_t_indicator.py / compute_student_t_nu STFT windowing & gwpy-compat (lines 65-92, 116)**
  - 取りこぼし理由: The sweep caught the bare-except (line 112-113) but did not apply the gwpy-compat / data_integrity lens to the STFT path. compute_student_t_nu uses scipy.signal.stft directly instead of ts.spectrogram, so (a) it does NOT apply the analysis window/detrending gwpy uses, meaning its FFT coefficients and segment count differ from the other methods in this same package (gauch uses ts.spectrogram); (b) noverlap=nfft-nstep can go negative when stride>fftlength, which scipy rejects; (c) int(fftlength*fs) truncation silently changes the effective fftlength for non-integer products; (d) out_times = t[window//2:...] uses scipy's t which is offset-relative, not the absolute GPS epoch of ts -- a likely wrong-time-axis bug parallel to the dq_flag dt finding but never checked.
  - 追補: Compare segment count/times from compute_student_t_nu against gauch's ts.spectrogram for identical inputs. Test stride>fftlength (negative noverlap), non-integer fftlength*fs, and verify out_times carry the correct GPS epoch (t0) rather than scipy's zero-based t.
- **student_t_indicator.py / stats.t.fit on tiny/degenerate samples (lines 104-110)**
  - 取りこぼし理由: numerical lens applied only at the except clause, not at the statistical validity of the fit input. samples = concat(real, imag) has length 2*window; for small window (and the DC bin f=0 or Nyquist bin where imag is ~0) the t-fit is on near-constant data -> scale ~0, nu unbounded/garbage, often returned WITHOUT raising (so the bare-except never triggers and a nonsense nu is stored, not NaN). Real/imag of a real-signal STFT at DC/Nyquist are not independent identically-distributed t samples, violating the method's own assumption.
  - 追補: Test compute_student_t_nu on a pure-DC or constant input and on the f=0 / Nyquist bins; assert the returned nu is finite-and-sane or NaN, not a silent garbage value. Consider excluding DC/Nyquist bins where imag==0.
- **rayleigh_test.py / rayleigh_pvalue two-sided p-value flooring & gwpy-compat (lines 55-62, 78-91)**
  - 取りこぼし理由: The sweep flagged n_monte_carlo=0 / n_samples=0 degenerate cases but did not apply the data_integrity lens to the two-sided p-value math itself, nor the gwpy-compat lens to the null-distribution formula. (a) p_vals can be exactly 0.0 when r_vals is beyond every Monte-Carlo sample (same class as the confirmed gauch P3 line 114, but NOT flagged for rayleigh) -> log/odds downstream blow up; (b) the comment at lines 78-91 claims to match GWpy's std(ASD)/(mean(ASD)*sqrt((4-pi)/pi)) but this is asserted, never verified -- if gwpy's actual rayleigh_spectrogram normalization differs, every p-value is systematically biased; (c) NaN in r_vals flows through searchsorted to give p=2.0*min(...)/len, an unvalidated number.
  - 追補: Add the same min-p clamp recommendation as gauch line 114 to rayleigh (p floor of 1/len(dist)). Empirically compare _get_rayleigh_stat_null_distribution against gwpy's actual rayleigh_spectrogram output on Gaussian noise to confirm the normalization constant matches. Test r_vals containing NaN.
- **dq_flag.py / to_segments crop result and alpha/NaN contract (lines 42-47)**
  - 取りこぼし理由: The sweep caught the empty-times IndexError and the single-step dt fallback, but not the data_integrity lens on the comparison p_value_map.value < alpha. NaN p-values (produced by every other module's degenerate paths above) compare False, so NaN bins are silently treated as 'good' rather than 'unknown' -- masking exactly the failures the upstream audit found. Also crop(fmin,fmax) (line 43) may return an empty-frequency Spectrogram, making np.any(..., axis=1) collapse to all-False (every time good) with no warning; and alpha outside (0,1) or non-finite is never validated.
  - 追補: Test to_segments with a p_value_map containing NaN bins (assert they are not silently classed as good). Test crop producing zero frequency bins. Validate alpha in (0,1). Confirm behavior is consistent with how upstream NaN p-values should map to 'known but undetermined' segments.
- **Cross-module: _LILLIEFORS_CACHE / _RAYLEIGH_STAT_CACHE global mutable state & reproducibility (gauch.py:132, rayleigh_test.py:73)**
  - 取りこぼし理由: The sweep flagged the per-call symptom (n_monte_carlo ignored on cache hit, gauch P3 line 106-107) but did not apply the api_contract/reproducibility lens to the module-global caches as a class. Issues not covered: (a) no seeding of np.random anywhere, so results are non-reproducible run-to-run -- a research-reproducibility defect flagged explicitly in the scientific-workflow rules; (b) the caches are process-global and unbounded, so parallel/pytest runs leak state across tests and across different n_trials requests for the same n in BOTH files (not just gauch); (c) thread-safety: concurrent first-touch of the same n races on dict population.
  - 追補: Add a seed/rng parameter (np.random.default_rng) to compute_gauch, rayleigh_pvalue and the null-distribution generators for reproducibility. Add a test asserting identical output for identical seed. Decide whether caches should key on (n, n_trials, seed) and document/guard the global-state contract; add a cache-clear hook for test isolation.
- **gauch.py / Spectrogram time-axis alignment & metadata loss (lines 118-121)**
  - 取りこぼし理由: The sweep examined the numerical inner loop and Monte-Carlo edge cases but not the api_contract/data_integrity lens on output construction. out_times = spec.times[window//2 : window//2 + n_out] assumes spec.times length >= window//2+n_out and that center-of-window indexing is the intended convention (off-by-one vs the student_t version which also uses window//2 -- worth cross-checking they agree). The output Spectrogram is built without unit/name/epoch metadata (unlike rayleigh_pvalue which sets name/unit), so dq_flag's name-based flag naming (dq_flag line 74) silently falls back to the generic name. n_monte_carlo on the GauChResult metadata is not recorded either.
  - 追補: Cross-check that gauch and student_t use the same window-center time convention on identical inputs. Verify out_times slicing cannot under-run for edge window sizes. Confirm the result Spectrogram carries name/unit/epoch so downstream to_segments produces a meaningful flag name; add a regression test on the propagated metadata.

### gwexpy.spectral.estimation（9 ギャップ）
- **bootstrap_spectrogram / _bootstrap_resample_py + _jit (line 58, 84): resampled_stats = np.zeros((...), dtype=data.dtype)**
  - 取りこぼし理由: The sweep focused on NaN handling and the block-index crash, treating the data array as implicitly float. It never applied the data_integrity lens to dtype. Spectrogram.value can be integer (synthetic data, counts, or a Spectrogram built from int arrays), and the output buffer inherits that dtype, so every np.mean/np.median/np.nanmean result is silently truncated toward zero. Verified: storing mean([2,3])=2.5 into an int buffer yields 2.
  - 追補: Construct resampled_stats with a float dtype (e.g. np.result_type(data.dtype, np.float64) or just float) instead of data.dtype. Add a test feeding an integer-valued Spectrogram and assert the bootstrap center is not integer-truncated (compare against a float reference).
- **calculate_correlation_factor (lines 222-260): numerical lens not applied**
  - 取りこぼし理由: This function is the core variance-inflation math but the sweep only checked the bootstrap-index path. The numerical/degenerate-input lens was never turned on it: (a) a window array containing NaN/Inf propagates silently into energy and rho with no guard; (b) energy==0 is guarded but a near-zero energy (tiny window) is not, so rho can blow up; (c) rho is theoretically in [-1,1] but with a malformed window rho**2 can exceed 1 and inflate the factor without warning; (d) noverlap >= nperseg makes step<=0 (guarded) but noverlap negative is not validated.
  - 追補: Add input validation: reject/repair NaN-Inf in win_array, validate 0 <= noverlap < nperseg, and clamp or warn when |rho|>1. Add tests with a NaN-containing window, a degenerate (all-zero-except-one) window, and noverlap negative.
- **estimate_psd duration guard (lines 135-158): silent skip when dt or fftlength are Quantity-with-incompatible-units or dt is None**
  - 取りこぼし理由: The sweep confirmed the NaN rejection but did not apply the silent_failure lens to the duration check. _to_seconds returns None on any unit-conversion failure, and the guard `if dt_sec is not None and fftlength_sec is not None` then silently skips the duration validation. So a fftlength with wrong units, or a TimeSeries missing dt, bypasses the 'fftlength must not exceed duration' check entirely and defers to gwpy with a less clear error. Also np.isnan(data).any() runs on getattr(timeseries,'value',timeseries) which can raise TypeError for non-numeric/object dtype rather than a clean message.
  - 追補: Decide whether a failed unit conversion should warn or raise rather than silently skip the guard. Add tests: fftlength given as a frequency Quantity (wrong unit), a TimeSeries without dt, and an object/complex-dtype input to estimate_psd.
- **bootstrap_spectrogram return_map covariance branch (lines 731-759): mean-imputation distorts covariance and branch is untested**
  - 取りこぼし理由: return_map=True is an opt-in path and no test in test_estimation.py or test_estimation_semantics.py exercises it (the test list shows only rebin/units/methods). The sweep stopped at the center/CI computation. The ignore_nan=True branch fills NaNs with column means before np.cov, which biases the covariance toward zero (imputed points add no deviation) with no warning to the user, and the data_integrity 'imputation transparency' rule is violated (no *_source marker, no note in output). If a whole column is NaN, col_mean is NaN and the fill leaves NaNs, so np.cov returns all-NaN silently.
  - 追補: Add tests for return_map=True with and without NaNs; assert covariance shape/units and that an all-NaN column does not silently produce an all-NaN matrix. Document or warn that NaN imputation biases the returned covariance, or switch to pairwise/masked covariance.
- **bootstrap_spectrogram rebin_width block (lines 493-517): int truncation and silent no-op**
  - 取りこぼし理由: The sweep concentrated on block_size and NaN, not rebin. bin_size = int(rebin_width / df) silently floors, so a requested rebin width is quietly rounded down; rebin_width smaller than df gives bin_size<=1 and is silently ignored (no warning that no rebinning happened); rebin_width<=0 is filtered but NaN/Inf rebin_width is not. After rebinning, an all-NaN frequency slice with ignore_nan=True produces a NaN-center bin with no warning (same class as the confirmed all-NaN finding but on the rebinned axis).
  - 追補: Warn when bin_size<=1 (rebin requested but ineffective) and when truncation drops bins. Validate rebin_width is finite. Add a test with rebin_width<df and rebin_width=NaN.
- **bootstrap_spectrogram block_size == n_time dead branch (lines 548-553)**
  - 取りこぼし理由: The sweep flagged block_size > n_time (crash) but the boundary block_size == n_time hits `if block_size >= n_time: ... pass` — a no-op that was inspected enough to confirm the >n_time crash but not the ==n_time degenerate behavior. With block_size==n_time, num_possible_blocks = n_time - block_size + 1 = 1, so np.random.randint(0,1,...) always returns start index 0: every bootstrap sample is identical to the original ordering, producing zero variance and zero-width CI with no warning. The TODO-style `pass` comment ('warn?') signals the author themselves left this unhandled.
  - 追補: Replace the empty pass with an explicit warning (or clamp) when num_possible_blocks <= 1, since the resulting CI is degenerate (zero width). Add a test with block_size equal to n_time asserting a warning and non-degenerate handling.
- **bootstrap_spectrogram center/CI computation with degenerate n_boot or all-equal resamples (lines 624-641)**
  - 取りこぼし理由: n_boot>=1 is validated but n_boot==1 is allowed; with a single bootstrap replicate, np.percentile over axis=0 of length-1 gives p_low==p_high==center, so err_low/err_high are exactly 0 with no warning that the CI is meaningless. The sweep validated the n_boot<1 boundary error but not the n_boot==1 degenerate case. Also the silent_failure lens was not applied to the percentile step when all resamples are NaN (nanpercentile of all-NaN axis warns at numpy level and yields NaN, which is the surfacing point of the confirmed all-NaN finding but at lines 637-638, not 81-96 as cited).
  - 追補: Warn (or require n_boot>=some minimum, e.g. >= a few hundred) when n_boot is too small for the requested ci. Note: the confirmed all-NaN finding's true surfacing line is the nanmedian/nanpercentile at 626/637, not _bootstrap_resample_py at 81-96 — re-anchor that finding.
- **gwpy-compat surface: estimate_psd return type and method/kwargs passthrough (lines 160-169)**
  - 取りこぼし理由: The sweep was numerical-robustness scoped and did not compare the gwpy compatibility surface. estimate_psd forwards method= straight to timeseries.psd; an unknown method name raises a raw gwpy/scipy error rather than the module's own validated message (contrast with bootstrap_spectrogram which validates method in {median,mean}). The return path does res.view(FrequencySeries) when res is not already a gwexpy FrequencySeries, which assumes res is ndarray-viewable and shares the gwpy memory/metadata contract; a gwpy FrequencySeries subclass mismatch (units, f0/df propagation) is not verified.
  - 追補: Add a gwpy-vs-gwexpy comparison test: run TimeSeries.psd directly and estimate_psd, assert identical values, frequency axis, and unit. Test estimate_psd with an invalid method= and decide whether to validate up-front. Confirm .view(FrequencySeries) preserves f0/df/unit metadata.
- **Non-stationarity heuristic (lines 481-490): only warns, never applied to block path; numerical guard on ratio**
  - 取りこぼし理由: The heuristic computes std_avg/mean_avg with a mean_avg>0 guard, so for spectra with mean_avg==0 or negative (e.g. a dB-scaled or mean-subtracted spectrogram) the non-stationarity check is silently skipped. The sweep did not consider non-PSD-scaled inputs. Also the heuristic only runs for n_time>4 with no note for 2<n_time<=5 that the check was skipped.
  - 追補: Decide behavior when mean_avg<=0 (dB or signed input) — either skip with a note or use a different dispersion metric. Add a test with a dB-scaled spectrogram confirming no crash and sensible (or explicitly disabled) stationarity warning.

### gwexpy.fitting（8 ギャップ）
- **gls.py :: GLS class (lines 26-69), GLS.__init__ and GLS.solve**
  - 取りこぼし理由: The sweep examined only the GeneralizedLeastSquares COST class (used by Minuit) and never touched the separate GLS DIRECT-SOLVER class. The two classes share a name prefix, so the auditor likely treated gls.py as 'covered' after flagging cov_inv=np.linalg.inv at line 54 and np.linalg.solve at line 68 -- both of which are actually inside GLS, but the solve() normal-equations path (X.T@W@X possibly singular when X is rank-deficient or n_params>n_samples) and the OLS fallback np.eye(len(y)) path were not analyzed as a numerical/api-contract unit.
  - 追補: Apply the numerical + api_contract lens to GLS.solve(): test (a) rank-deficient or collinear design matrix X (singular X.T@W@X -> LinAlgError with no guard), (b) n_params > n_samples (under-determined), (c) cov passed as non-invertible at line 54, (d) X/y length mismatch (no validation in __init__, unlike GeneralizedLeastSquares which validates shape). Add a condition-number / lstsq-fallback recommendation.
- **gls.py:176 :: GeneralizedLeastSquares.__call__ non-Cholesky branch (chi2 = r @ cov_inv @ r)**
  - 取りこぼし理由: The confirmed gls.py findings target the Cholesky-failure silent discard (137-141) and the inv/solve calls in the OTHER class. The fallback quadratic-form path when cov_cho is None was not assessed for the case where cov_inv is supplied directly (via cost_function=GLS instance or 2D ndarray pinv) and is not positive semidefinite -- this silently yields a negative chi2 that Minuit will 'minimize' toward -inf.
  - 追補: Test GeneralizedLeastSquares with a non-PSD cov_inv (e.g. an indefinite symmetric matrix, or pinv of a near-singular cov): assert __call__ can return negative values and that the fit diverges. Recommend a one-time PSD/eigenvalue check or clamping. Also note: this path is reachable from fit_series via np.linalg.pinv(cov_arr) at core.py:1153 when cov is rank-deficient, with no warning.
- **models.py :: power_law (line 73), exponential negative tau, Polynomial/make_pol_func**
  - 取りこぼし理由: The models sweep applied the division-by-zero lens (sigma=0, tau=0, gamma=0, Q=0) but stopped there. It did not apply the data_integrity/domain lens: power_law A*x**alpha produces NaN/complex for x<0 with fractional alpha and inf for x=0 with alpha<0; x**i in Polynomial/make_pol_func overflows for large x and high degree; exponential with tau<0 grows unboundedly. None of these are zero-division so the lens that was used missed them entirely.
  - 追補: Test power_law with x containing 0 and negative values plus fractional alpha (expect nan/inf propagating silently into chi2); test pol9 with large |x| for overflow warnings; document/guard domain assumptions. Confirm whether fit_series ever feeds x<=0 (frequency arrays can include DC bin f=0) into power_law during real fits -- a realistic GW PSD fit starting at f=0 would hit this.
- **core.py :: fit_series sigma cropping via np.searchsorted (lines 1095-1107)**
  - 取りこぼし理由: The sweep's empty-array lens caught crop-produces-empty cases but did not apply the data_integrity lens to the searchsorted-based sigma cropping. searchsorted assumes x_full is sorted ascending; for a reversed/unsorted index, or when x_range is given as (hi, lo), idx0/idx1 can be wrong or produce a dy slice whose length silently mismatches the cropped y, hitting the ValueError at 1107 or worse, a silently misaligned sigma-to-data mapping (wrong weights, no error).
  - 追補: Test fit_series with (a) per-point sigma + descending xindex, (b) x_range given reversed (x0>x1) combined with array sigma, (c) duplicate x values at the crop boundary. Verify sigma stays aligned with the cropped y; compare the searchsorted crop against series.crop's actual boundary semantics.
- **core.py :: FitResult.ndof / reduced_chi2 / params (api_contract + numerical, lines 244-256, 224-235)**
  - 取りこぼし理由: The sweep focused on crashes and inf/nan from division, not on contract correctness of computed statistics. ndof doubles n_data for complex y (line 247-249) but GLS rejects complex data, and for GLS the effective DOF with a full covariance is debatable -- reduced_chi2 may be reported with an inconsistent normalization. Also params/errors read minuit.errors which can be nan when hesse fails or is skipped (nfit==0 path at 1224); ParameterValue stores error=None handling but downstream f-string formatting in highlevel (line 342 result.errors[k]:.2g) will raise on None/nan.
  - 追補: Test: (a) all-params-fixed fit (nfit==0, hesse skipped) then access .errors and _plot_bootstrap_fit param annotation -> likely formatting crash or 'nan'; (b) a fit where migrad fails to converge so minuit.errors are nan -> reduced_chi2 and params silently propagate nan; (c) verify ndof for GLS equals n - nfit (not 2n) and that reduced_chi2 normalization matches the GLS chi2 definition.
- **highlevel.py :: fit_bootstrap_spectrum freq_range mask + downstream (lines 227-256) beyond empty-mask**
  - 取りこぼし理由: The confirmed finding covers the empty-mask-degenerate-fit case. Not examined: a mask selecting exactly 1 frequency bin (n=1) -> 1x1 covariance, GLS/Cholesky on a single point, and reduced_chi2 with ndof<=0; nor the data_integrity risk that psd[mask] and cov_map.value[np.ix_(mask,mask)] assume psd.frequencies and cov_map axes are identically ordered and identical length (no assertion). A mismatch silently fits with a misaligned covariance.
  - 追補: Test freq_range that selects exactly one bin and two bins; assert behavior is a clear error not a silent degenerate fit. Add a check/test that cov_map frequency axis equals psd.frequencies before np.ix_ cropping (compare the two arrays). Also test freq_range with fmin>fmax.
- **core.py :: run_mcmc initialization & sampling robustness (lines 747-770) beyond ndim==0**
  - 取りこぼし理由: The confirmed finding is only the ndim==0 (all-fixed) LinAlgError. The broader silent_failure lens was not applied: log_prob swallows ValueError/TypeError/ZeroDivisionError -> -inf (lines 739-741), so a model that is broken for ALL parameter values makes every walker -inf and emcee silently produces a useless chain with no warning; initial pos can start outside minuit limits (jitter pushes p0 past a hard bound) yielding -inf walkers; n_walkers < 2*ndim violates emcee's requirement and raises an opaque error.
  - 追補: Test run_mcmc with (a) a model that raises for all theta -> assert a warning/RuntimeError rather than a silently degenerate sampler; (b) n_walkers=2 with ndim=3 (emcee minimum-walker violation); (c) a tight limit where the jitter ball at line 759 straddles the bound -> count -inf walkers. Recommend validating walker count and checking that not all initial log_probs are -inf.
- **gwpy-compat surfaces: __init__.py monkeypatch/enable_series_fit, mixin.py FittingMixin.fit, FitResult unit propagation bound_model (core.py 196-222)**
  - 取りこぼし理由: The audit lens was numerical robustness, so gwpy-interop/api_contract surfaces were out of frame. enable_series_fit only patches if not hasattr(Series,'fit') -- if gwpy ever adds its own .fit, gwexpy's silently never installs (silent no-op). FittingMixin.fit does not forward cov or cost_function (only sigma), so series.fit(model, cov=...) silently routes cov into **kwargs and reaches fit_series as an unexpected path. bound_model swallows unit-conversion failures (205-207, 217-219) returning unconverted values silently.
  - 追補: Compare gwexpy fit entry points against gwpy: (a) test series.fit(..., cov=cov_matrix) actually performs GLS (currently cov is not an explicit param of FittingMixin.fit -> verify it passes through kwargs correctly to fit_series); (b) test enable_series_fit behavior if Series already has fit; (c) test bound_model with an incompatible x unit -> confirm it warns rather than silently returning raw values; (d) confirm 'rtHz'-style non-astropy units round-trip without silent unit loss.

## 推奨アクション

| 優先 | 内容 |
|---|---|
| 1 | **`roc.py` 専用 re-sweep**（最大の取りこぼし。ゼロ除算が実在） |
| 2 | 「無言 NaN 汚染」系（rayleigh n=0 / student_t bare except / gauch sigma2=0 / spectral all-NaN）を共通方針で修正：縮退入力は `ValueError` か明示 `UserWarning`+NaN（#450/#452 C1 方針と整合） |
| 3 | fitting 形状関数のゼロ除算（gaussian/lorentzian/voigt/...）を共通 epsilon ガード or 入力検証で一括（scientific-workflow の Numerical Robustness 準拠） |
| 4 | RNG seed 引数追加（再現性）、グローバル可変キャッシュの test 分離 |
| 割当 | これら P2/P3 は **v0.1.8** の hardening lane に合流。gwpy 互換系（estimate_psd return / student_t STFT / FittingMixin.fit）は #451 と共に検討 |

## 関連

- 上流調査: #450 / #452（matrix root fix / interop dispatch）
- 計画: `~/.claude/plans/issues-pr-woolly-gem.md`（v0.1.7 スコープ）
- 既存の関連レポート: `phase0_exception_analysis.md`, `phase1_dangerous_defaults.md`, `test_coverage_gap_report.md`
