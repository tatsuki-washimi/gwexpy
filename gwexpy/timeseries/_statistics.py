"""
Statistical analysis and correlation mixin for TimeSeries.
Provides skewed, kurtosis, Granger causality, distance correlation,
and classic correlations (Pearson, Kendall, MIC).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from gwexpy.types._stats import StatisticalMethodsMixin

from ._typing import TimeSeriesAttrs


class GrangerResult(float):
    """
    Subclass of float that holds Granger causality test results.
    Inheriting from float allows p-value formatting like `f"{result:.4f}"`.
    Dictionary-like access is also supported for backward compatibility.
    """

    def __new__(cls, min_p_value, best_lag, p_values):
        return super().__new__(cls, min_p_value)

    def __init__(self, min_p_value, best_lag, p_values):
        self.min_p_value = min_p_value
        self.best_lag = best_lag
        self.p_values = p_values

    def __getitem__(self, key):
        if key == "min_p_value":
            return self.min_p_value
        if key == "best_lag":
            return self.best_lag
        if key == "p_values":
            return self.p_values
        raise KeyError(key)

    def __contains__(self, key):
        return key in ["min_p_value", "best_lag", "p_values"]

    def keys(self):
        return ["min_p_value", "best_lag", "p_values"]

    def items(self):
        return [
            ("min_p_value", self.min_p_value),
            ("best_lag", self.best_lag),
            ("p_values", self.p_values),
        ]


class StatisticsMixin(TimeSeriesAttrs, StatisticalMethodsMixin):
    """
    Mixin class to add statistical analysis and correlation methods to TimeSeries.

    Inherits basic statistics (mean, std, skewness, kurtosis, etc.) from
    StatisticalMethodsMixin and adds correlation and causality methods.
    """

    # skewness and kurtosis are now inherited from StatisticalMethodsMixin

    # ===============================
    # Correlation & Causality
    # ===============================

    def correlation(self, other, method="pearson", **kwargs):
        """
        Calculate correlation coefficient with another TimeSeries.

        Args:
            other (TimeSeries): The series to compare with.
            method (str): 'pearson', 'kendall', 'mic', or 'distance'.
            **kwargs: Additional arguments passed to the underlying function.

        Returns:
            float: The correlation coefficient.
        """
        # Data preparation: align length and strip units
        x, y = self._prep_stat_data(other)

        if method == "pearson":
            return self._calculate_pearson(x, y)
        elif method == "kendall":
            return self._calculate_kendall(x, y)
        elif method == "mic":
            return self._calculate_mic(x, y, **kwargs)
        elif method == "distance":
            return self._calculate_distance_correlation(x, y, **kwargs)
        elif method == "fastmi":
            return self._calculate_fastmi(x, y, **kwargs)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    def mic(self, other, alpha=0.6, c=15, est="mic_approx"):
        """
        Calculate Maximal Information Coefficient (MIC) using minepy.

        Note: On Python 3.11+, minepy must be built from source.
        Use `python scripts/install_minepy.py` provided in the gwexpy repository.
        """
        return self.correlation(other, method="mic", alpha=alpha, c=c, est=est)

    def pcc(self, other):
        """
        Calculate Pearson Correlation Coefficient.
        """
        return self.correlation(other, method="pearson")

    def ktau(self, other):
        """
        Calculate Kendall's Rank Correlation Coefficient.
        """
        return self.correlation(other, method="kendall")

    def fastmi(self, other, **kwargs):
        """
        Estimate mutual information using a fast copula/probit + FFT-based estimator.
        """
        return self.correlation(other, method="fastmi", **kwargs)

    def distance_correlation(self, other):
        """
        Calculate Distance Correlation (dCor).

        Distance correlation is a measure of dependence between two random vectors
        that is 0 if and only if the random vectors are independent.
        It can detect non-linear relationships.

        Args:
            other (TimeSeries): The series to compare with.

        Returns:
            float: The distance correlation (0 to 1).
        """
        return self.correlation(other, method="distance")

    def partial_correlation(
        self,
        other,
        *,
        controls=None,
        method: str = "residual",
        **kwargs,
    ):
        """
        Calculate partial correlation between self and other, controlling for controls.

        Parameters
        ----------
        other : TimeSeries
            The series to compare with.
        controls : TimeSeries or list[TimeSeries], optional
            Control variables to regress out. If None, falls back to Pearson correlation.
        method : {"residual", "precision"}
            - "residual": regress out controls and correlate residuals.
            - "precision": compute partial correlation via precision matrix.
        **kwargs
            Extra arguments for the underlying solver (e.g., rcond for pinv).
        """
        x, y, controls_mat = self._prep_partial_data(other, controls)

        if controls_mat.size == 0:
            return self._calculate_pearson(x, y)

        if method == "residual":
            rx = self._residualize(x, controls_mat)
            ry = self._residualize(y, controls_mat)
            return self._calculate_pearson(rx, ry)
        elif method == "precision":
            data = np.column_stack([x, y, controls_mat])
            cov = np.cov(data, rowvar=False)
            rcond = kwargs.pop("rcond", None)
            try:
                precision = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                precision = np.linalg.pinv(cov, rcond=rcond)
            denom = np.sqrt(precision[0, 0] * precision[1, 1])
            if denom == 0:
                return np.nan
            return -precision[0, 1] / denom
        else:
            raise ValueError(f"Unknown partial correlation method: {method}")

    def granger_causality(self, other, maxlag=10, test="ssr_ftest", verbose=False):
        """
        Check if 'other' Granger-causes 'self'.

        Null Hypothesis: The past values of 'other' do NOT help in predicting 'self'.

        Args:
            other (TimeSeries): The potential cause series.
            maxlag (int): Maximum lag to check.
            test (str): Statistical test to use ('ssr_ftest', 'ssr_chi2test', etc.).
            verbose (bool): Whether to print verbose output.

        Returns:
            float: The minimum p-value across all lags up to maxlag.
                   A small p-value (e.g., < 0.05) indicates Granger causality.
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            raise ImportError(
                "The 'statsmodels' package is required for Granger Causality. "
                "Please install it via `pip install statsmodels` or `pip install gwexpy[stat]`."
            )

        # Align data
        x, y = self._prep_stat_data(other)

        # Format for statsmodels: [response, predictor] = [self, other]
        # We want to check if 'other' causes 'self'.
        data = np.stack([x, y], axis=1)

        # grangercausalitytests returns a dict: {lag: (test_result, params, ...)}
        # test_result[0] contains statistics like {'ssr_ftest': (F-stat, p-value, df_denom, df_num), ...}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="verbose is deprecated",
                category=FutureWarning,
                module="statsmodels",
            )
            res = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)

        # Extract p-values for the specified test across all lags
        p_values = []
        lags = []

        for lag, result in res.items():
            test_result = result[0][test]
            p_val = test_result[1]
            p_values.append(p_val)
            lags.append(lag)

        # Find best lag (minimum p-value)
        min_idx = np.argmin(p_values)
        min_p = p_values[min_idx]
        best_lag = lags[min_idx]

        return GrangerResult(
            min_p_value=min_p, best_lag=best_lag, p_values=dict(zip(lags, p_values))
        )

    # --- Internal Methods ---

    def _prep_stat_data(self, other):
        """Helper to align and strip units from data."""
        # Check dimensions
        if self.ndim > 1:
            raise ValueError(
                f"Statistical methods are only supported for 1D TimeSeries. "
                f"Current shape: {self.shape}. Please select a specific channel."
            )

        # Check sample rate match (approximate check)
        if self.sample_rate != other.sample_rate:
            warnings.warn(
                "Sample rates do not match. Resampling 'other' to match 'self'."
            )
            # Resample 'other' to match 'self'
            other = other.resample(self.sample_rate)

        # Align length (crop to shorter)
        min_len = min(len(self), len(other))
        x = self.value[:min_len]
        y = other.value[:min_len]

        # Ensure numpy array and flatten
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        return x, y

    def _calculate_pearson(self, x, y):
        # scipy.stats.pearsonr returns (statistic, p-value)
        return stats.pearsonr(x, y)[0]

    def _calculate_kendall(self, x, y):
        # scipy.stats.kendalltau returns (statistic, p-value)
        return stats.kendalltau(x, y)[0]

    def _calculate_mic(self, x, y, alpha=0.6, c=15, est="mic_approx"):
        try:
            from minepy import MINE
        except ImportError:
            try:
                # Some mictools installations might expose minepy
                import mictools  # noqa: F401
                from minepy import MINE
            except ImportError:
                raise ImportError(
                    "The 'minepy' package is required for MIC calculation. "
                    "On Python 3.11+, please install it using 'python scripts/install_minepy.py' "
                    "from the gwexpy source directory. "
                    "Alternatively, try `pip install mictools` or `pip install gwexpy[stat]`."
                )

        # MINE calculation
        m = MINE(alpha=alpha, c=c, est=est)
        m.compute_score(x, y)
        return m.mic()

    def _calculate_distance_correlation(self, x, y, **_kwargs):
        try:
            import dcor
        except ImportError as exc:
            raise ImportError(
                "The 'dcor' package is required for distance correlation. "
                "Please install it via `pip install dcor` or `pip install gwexpy[stat]`."
            ) from exc
        try:
            return dcor.distance_correlation(x, y)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("dcor failed to compute distance correlation.") from exc

    def _calculate_fastmi(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        grid_size: int | None = None,
        quantile: float = 0.001,
        eps: float = 1e-12,
    ) -> float:
        """
        Copula-based mutual information estimator inspired by "fastMI" (Purkayastha & Song, 2023).

        Notes
        -----
        - Intended for 1D continuous arrays.
        - Uses empirical CDF -> probit transform -> self-consistent FFT density estimates.
        - Returns MI in nats (natural log), non-negative.
        """
        x = np.asarray(x, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        n = min(len(x), len(y))
        if n < 4:
            return 0.0
        x = x[:n]
        y = y[:n]

        if grid_size is None:
            # Heuristic: keep 2D FFT moderate while scaling with data size.
            target = int(
                max(
                    64,
                    min(512, 2 ** int(np.ceil(np.log2(max(64, int(np.sqrt(n) * 8)))))),
                )
            )
            grid_size = target
        if grid_size < 16:
            raise ValueError("grid_size must be >= 16")
        if not (0.0 <= quantile < 0.5):
            raise ValueError("quantile must be within [0, 0.5)")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        ux = self._fastmi_pseudo_obs(x, eps=eps)
        uy = self._fastmi_pseudo_obs(y, eps=eps)
        vx = stats.norm.ppf(ux)
        vy = stats.norm.ppf(uy)

        vxy = np.column_stack([vx, vy])
        g_xy = self._fastmi_sc_pdf_at_points(
            vxy, grid_size=grid_size, quantile=quantile, eps=eps
        )
        g_x = self._fastmi_sc_pdf_at_points(
            vx[:, None], grid_size=grid_size, quantile=quantile, eps=eps
        )
        g_y = self._fastmi_sc_pdf_at_points(
            vy[:, None], grid_size=grid_size, quantile=quantile, eps=eps
        )

        phi_x = stats.norm.pdf(vx)
        phi_y = stats.norm.pdf(vy)
        phi_x = np.clip(phi_x, eps, None)
        phi_y = np.clip(phi_y, eps, None)

        c_xy = g_xy / (phi_x * phi_y)
        c_x = g_x / phi_x
        c_y = g_y / phi_y

        c_xy = np.clip(c_xy, eps, None)
        c_x = np.clip(c_x, eps, None)
        c_y = np.clip(c_y, eps, None)

        mi = float(np.mean(np.log(c_xy) - np.log(c_x) - np.log(c_y)))
        if not np.isfinite(mi):
            return 0.0
        return max(0.0, mi)

    @staticmethod
    def _fastmi_pseudo_obs(x: np.ndarray, *, eps: float) -> np.ndarray:
        n = len(x)
        # Mid-rank empirical CDF values in (0, 1).
        r = stats.rankdata(x, method="average")
        u = (r - 0.5) / n
        return np.clip(u, eps, 1.0 - eps)

    @staticmethod
    def _fastmi_connected_component(mask: np.ndarray) -> np.ndarray:
        """
        Keep only the connected component of mask that contains the zero-frequency bin.
        """
        if mask.ndim == 1:
            ms = np.fft.fftshift(mask)
            m = ms.shape[0]
            c = m // 2
            if not ms[c]:
                out = np.zeros_like(ms, dtype=bool)
                out[c] = True
                return np.fft.ifftshift(out)
            left = c
            right = c
            while left - 1 >= 0 and ms[left - 1]:
                left -= 1
            while right + 1 < m and ms[right + 1]:
                right += 1
            out = np.zeros_like(ms, dtype=bool)
            out[left : right + 1] = True
            return np.fft.ifftshift(out)

        if mask.ndim == 2:
            ms = np.fft.fftshift(mask)
            m0, m1 = ms.shape
            c0, c1 = m0 // 2, m1 // 2
            out = np.zeros_like(ms, dtype=bool)
            if not ms[c0, c1]:
                out[c0, c1] = True
                return np.fft.ifftshift(out)

            stack = [(c0, c1)]
            out[c0, c1] = True
            while stack:
                i, j = stack.pop()
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m0 and 0 <= nj < m1 and ms[ni, nj] and not out[ni, nj]:
                        out[ni, nj] = True
                        stack.append((ni, nj))
            return np.fft.ifftshift(out)

        raise ValueError("mask must be 1D or 2D")

    @classmethod
    def _fastmi_sc_pdf_at_points(
        cls,
        v: np.ndarray,
        *,
        grid_size: int,
        quantile: float,
        eps: float,
    ) -> np.ndarray:
        """
        Estimate PDF of v (n,d) at sample points using SC FFT estimator on a regular grid.
        """
        v = np.asarray(v, dtype=float)
        if v.ndim != 2:
            raise ValueError("v must be 2D (n, d)")
        n, d = v.shape
        if d not in (1, 2):
            raise ValueError("fastmi currently supports d=1 or d=2")

        if quantile > 0:
            lo = np.quantile(v, quantile, axis=0)
            hi = np.quantile(v, 1.0 - quantile, axis=0)
        else:
            lo = np.min(v, axis=0)
            hi = np.max(v, axis=0)

        span = hi - lo
        span = np.where(span <= 0, 1.0, span)
        lo = lo - 0.05 * span
        hi = hi + 0.05 * span
        L = hi - lo
        L = np.where(L <= 0, 1.0, L)

        s = v - lo
        # Bin to grid
        m = int(grid_size)
        dx = L / m
        idx = np.floor(s / dx).astype(int)
        idx = np.clip(idx, 0, m - 1)

        if d == 1:
            hist = np.zeros((m,), dtype=float)
            np.add.at(hist, idx[:, 0], 1.0)
        else:
            hist = np.zeros((m, m), dtype=float)
            np.add.at(hist, (idx[:, 0], idx[:, 1]), 1.0)

        # ECF on FFT grid (shifted variable on [0, L]).
        C = np.fft.ifftn(hist) * (m**d) / n
        absC2 = (C * np.conj(C)).real

        # Primary threshold and connected hypervolume around origin.
        cmin2 = 4.0 * (n - 1) / (n**2)
        mask = absC2 >= cmin2
        mask = cls._fastmi_connected_component(mask)

        # SC characteristic function estimate.
        denom = np.maximum(absC2, cmin2)
        ratio = 4.0 * (n - 1) / (n**2 * denom)
        inner = np.clip(1.0 - ratio, 0.0, 1.0)
        factor = 1.0 - np.sqrt(inner)

        phi_hat = (n * C) / (2.0 * (n - 1)) * factor
        phi_hat = np.where(mask, phi_hat, 0.0)
        if d == 1:
            phi_hat[0] = 1.0
        else:
            phi_hat[0, 0] = 1.0

        # Density on spatial grid via Fourier series reconstruction.
        f_grid = np.real(np.fft.fftn(phi_hat)) / float(np.prod(L))
        f_grid = np.maximum(f_grid, 0.0)

        # Evaluate at points with simple linear (1D) / bilinear (2D) interpolation.
        if d == 1:
            return cls._fastmi_interp1d(f_grid, s[:, 0], dx=float(dx[0]), eps=eps)
        return cls._fastmi_interp2d(
            f_grid,
            s[:, 0],
            s[:, 1],
            dx0=float(dx[0]),
            dx1=float(dx[1]),
            eps=eps,
        )

    @staticmethod
    def _fastmi_interp1d(
        f: np.ndarray, x: np.ndarray, *, dx: float, eps: float
    ) -> np.ndarray:
        m = f.shape[0]
        x = np.clip(x, 0.0, (m - 1) * dx)
        u = x / dx
        i0 = np.floor(u).astype(int)
        i1 = np.clip(i0 + 1, 0, m - 1)
        t = u - i0
        out = (1.0 - t) * f[i0] + t * f[i1]
        return np.clip(out, eps, None)

    @staticmethod
    def _fastmi_interp2d(
        f: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
        *,
        dx0: float,
        dx1: float,
        eps: float,
    ) -> np.ndarray:
        m0, m1 = f.shape
        x0 = np.clip(x0, 0.0, (m0 - 1) * dx0)
        x1 = np.clip(x1, 0.0, (m1 - 1) * dx1)
        u0 = x0 / dx0
        u1 = x1 / dx1
        i0 = np.floor(u0).astype(int)
        j0 = np.floor(u1).astype(int)
        i1 = np.clip(i0 + 1, 0, m0 - 1)
        j1 = np.clip(j0 + 1, 0, m1 - 1)
        t0 = u0 - i0
        t1 = u1 - j0

        f00 = f[i0, j0]
        f10 = f[i1, j0]
        f01 = f[i0, j1]
        f11 = f[i1, j1]

        out = (
            (1.0 - t0) * (1.0 - t1) * f00
            + t0 * (1.0 - t1) * f10
            + (1.0 - t0) * t1 * f01
            + t0 * t1 * f11
        )
        return np.clip(out, eps, None)

    def _prep_partial_data(self, other, controls):
        x, y = self._prep_stat_data(other)

        if controls is None:
            return x, y, np.empty((len(x), 0))

        if not isinstance(controls, (list, tuple)):
            controls_list = [controls]
        else:
            controls_list = list(controls)

        controls_arrays = []
        for ctrl in controls_list:
            if hasattr(ctrl, "sample_rate"):
                if self.sample_rate != ctrl.sample_rate:
                    warnings.warn(
                        "Sample rates do not match. Resampling 'controls' to match 'self'."
                    )
                    ctrl = ctrl.resample(self.sample_rate)
                ctrl_values = ctrl.value
            else:
                ctrl_values = np.asarray(ctrl)
            controls_arrays.append(np.asarray(ctrl_values).flatten())

        min_len = min(len(x), len(y), *(len(c) for c in controls_arrays))
        x = x[:min_len]
        y = y[:min_len]
        controls_arrays = [c[:min_len] for c in controls_arrays]

        controls_mat = (
            np.column_stack(controls_arrays)
            if controls_arrays
            else np.empty((min_len, 0))
        )
        return x, y, controls_mat

    @staticmethod
    def _residualize(x, controls_mat):
        if controls_mat.size == 0:
            return x - np.mean(x)
        n_samples = controls_mat.shape[0]
        design = np.column_stack([controls_mat, np.ones(n_samples)])
        coef, *_ = np.linalg.lstsq(design, x, rcond=None)
        return x - design @ coef
