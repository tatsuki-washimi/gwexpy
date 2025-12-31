"""
Statistical analysis and correlation mixin for TimeSeries.
Provides skewed, kurtosis, Granger causality, distance correlation, 
and classic correlations (Pearson, Kendall, MIC).
"""
import warnings
import numpy as np
from scipy import stats

class StatisticsMixin:
    """
    Mixin class to add statistical analysis and correlation methods to TimeSeries.
    """

    # ===============================
    # Basic Statistics (Gaussianity)
    # ===============================

    def skewness(self, nan_policy='propagate'):
        """
        Compute the skewness of the time series data.
        
        Skewness is a measure of the asymmetry of the probability distribution 
        of a real-valued random variable about its mean.
        
        Args:
            nan_policy (str): 'propagate', 'raise', 'omit'.
            
        Returns:
            float: The skewness.
        """
        return stats.skew(self.value, nan_policy=nan_policy)

    def kurtosis(self, fisher=True, nan_policy='propagate'):
        """
        Compute the kurtosis (Fisher or Pearson) of the time series data.
        
        Kurtosis is a measure of the "tailedness" of the probability distribution 
        of a real-valued random variable.
        
        Args:
            fisher (bool): If True, Fisher's definition is used (normal ==> 0.0). 
                           If False, Pearson's definition is used (normal ==> 3.0).
            nan_policy (str): 'propagate', 'raise', 'omit'.
            
        Returns:
            float: The kurtosis.
        """
        return stats.kurtosis(self.value, fisher=fisher, nan_policy=nan_policy)

    # ===============================
    # Correlation & Causality
    # ===============================

    def correlation(self, other, method='pearson', **kwargs):
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

        if method == 'pearson':
            return self._calculate_pearson(x, y)
        elif method == 'kendall':
            return self._calculate_kendall(x, y)
        elif method == 'mic':
            return self._calculate_mic(x, y, **kwargs)
        elif method == 'distance':
            return self._calculate_distance_correlation(x, y, **kwargs)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    def mic(self, other, alpha=0.6, c=15, est="mic_approx"):
        """
        Calculate Maximal Information Coefficient (MIC) using minepy.
        """
        return self.correlation(other, method='mic', alpha=alpha, c=c, est=est)

    def pcc(self, other):
        """
        Calculate Pearson Correlation Coefficient.
        """
        return self.correlation(other, method='pearson')

    def ktau(self, other):
        """
        Calculate Kendall's Rank Correlation Coefficient.
        """
        return self.correlation(other, method='kendall')
    
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
        return self.correlation(other, method='distance')

    def granger_causality(self, other, maxlag=10, test='ssr_ftest', verbose=False):
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
        
        return {
            'min_p_value': min_p,
            'best_lag': best_lag,
            'p_values': dict(zip(lags, p_values))
        }

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
            warnings.warn("Sample rates do not match. Resampling 'other' to match 'self'.")
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
            raise ImportError(
                "The 'minepy' package is required for MIC calculation. "
                "Please install it via `pip install minepy` or `pip install gwexpy[stat]`."
            )
        
        # MINE calculation
        m = MINE(alpha=alpha, c=c, est=est)
        m.compute_score(x, y)
        return m.mic()

    def _calculate_distance_correlation(self, x, y):
        try:
            import dcor
        except ImportError:
            raise ImportError(
                "The 'dcor' package is required for distance correlation. "
                "Please install it via `pip install dcor` or `pip install gwexpy[stat]`."
            )
        return dcor.distance_correlation(x, y)
