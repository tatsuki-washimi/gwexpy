"""
gwexpy.signal.preprocessing.ml
-------------------------------

Preprocessing pipeline for machine learning.

This module provides preprocessing utilities (data splitting, band-pass 
filtering, standardization) used in noise removal tasks like DeepClean,
implemented as a generic scikit-learn-style Transformer API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix

    from .standardize import StandardizationModel


class MLPreprocessor:
    """
    Preprocessing pipeline for machine learning.

    A scikit-learn-style Transformer that performs data splitting,
    band-pass filtering, and per-channel standardization. Can be used for
    noise removal tasks like DeepClean, as well as Random Forest, XGBoost,
    and other machine learning models.

    Parameters
    ----------
    sample_rate : Quantity or float
        Sampling rate (in Hz).
    freq_low : list[float] or None, optional
        Low-frequency cutoff for the band-pass filter (supports multiple bands).
        If None, filtering is skipped.
    freq_high : list[float] or None, optional
        High-frequency cutoff for the band-pass filter (supports multiple bands).
        If None, filtering is skipped.
    filt_order : int, optional
        Order of the Butterworth filter (default: 8).
    valid_frac : float, optional
        Proportion of validation data (0.0 to 1.0, default: 0.0).
        If 0.0, no splitting is performed.
    standardization_method : str, optional
        Standardization method ('zscore' or 'robust', default: 'zscore').

    Attributes
    ----------
    X_scaler_ : StandardizationModel or None
        Standardization model for reference channels (set after fit).
    y_scaler_ : StandardizationModel or None
        Standardization model for target channel (set after fit).
    filter_coeffs_ : list[np.ndarray] or None
        Band-pass filter coefficients (SOS format, set after fit).
    is_fitted_ : bool
        Flag indicating if fit is complete.

    Examples
    --------
    Basic usage:

    >>> from gwexpy.timeseries import TimeSeriesMatrix, TimeSeries
    >>> from gwexpy.signal.preprocessing import MLPreprocessor
    >>>
    >>> # Load data
    >>> witnesses = TimeSeriesMatrix(...)  # (n_channels, n_samples)
    >>> strain = TimeSeries(...)            # (n_samples,)
    >>>
    >>> # Preprocessing pipeline
    >>> preprocessor = MLPreprocessor(
    ...     sample_rate=4096,
    ...     freq_low=[55.0],
    ...     freq_high=[65.0],
    ...     valid_frac=0.2
    ... )
    >>>
    >>> # Split -> fit -> transform
    >>> X_train, y_train, X_valid, y_valid = preprocessor.split(witnesses, strain)
    >>> preprocessor.fit(X_train, y_train)
    >>> X_train_proc, y_train_proc = preprocessor.transform(X_train, y_train)
    >>> X_valid_proc, y_valid_proc = preprocessor.transform(X_valid, y_valid)

    Notes
    -----
    Processing order follows the DeepClean v2 implementation:
    1. Data splitting (chronological)
    2. Learn X standardization parameters (no filtering)
    3. Design filter coefficients
    4. Filter y -> Learn y standardization parameters

    Important notes:
    - **X is not filtered** (reference channels are standardized as raw data).
    - **y is filtered before standardization** (target channel is band-limited).
    """

    def __init__(
        self,
        sample_rate: u.Quantity | float,
        freq_low: list[float] | None = None,
        freq_high: list[float] | None = None,
        filt_order: int = 8,
        valid_frac: float = 0.0,
        standardization_method: str = "zscore",
    ):
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.filt_order = filt_order
        self.valid_frac = valid_frac
        self.standardization_method = standardization_method

        # Internal state (set after fit)
        self.X_scaler_: Optional[StandardizationModel] = None
        self.y_scaler_: Optional[StandardizationModel] = None
        self.filter_coeffs_: Optional[list[np.ndarray]] = None
        self.is_fitted_ = False

    def split(
        self,
        X: TimeSeriesMatrix,
        y: TimeSeries,
    ) -> tuple[TimeSeriesMatrix, TimeSeries, TimeSeriesMatrix, TimeSeries]:
        """
        Split data into training and validation sets.

        Parameters
        ----------
        X : TimeSeriesMatrix
            Reference channels (shape: (n_channels, n_samples))
        y : TimeSeries
            Target channel (shape: (n_samples,))

        Returns
        -------
        X_train : TimeSeriesMatrix
            Training reference channels
        y_train : TimeSeries
            Training target channel
        X_valid : TimeSeriesMatrix
            Validation reference channels
        y_valid : TimeSeries
            Validation target channel
        """
        if self.valid_frac == 0.0:
            # No split: use crop() to create zero-length validation data
            empty_X = X.crop(start=X.t0, end=X.t0)
            empty_y = y.crop(start=y.t0, end=y.t0)
            return X, y, empty_X, empty_y

        # Adjust to integer seconds (DeepClean compatibility)
        sample_rate_hz = self._get_sample_rate_hz()
        total_length = len(y)
        valid_size_float = self.valid_frac * total_length
        valid_length_sec = int(valid_size_float / sample_rate_hz)
        valid_size = int(valid_length_sec * sample_rate_hz)
        train_size = total_length - valid_size

        # Split using crop() method (preserves time information)
        train_end_time = X.t0 + train_size * X.dt
        valid_start_time = train_end_time

        X_train = X.crop(end=train_end_time)
        X_valid = X.crop(start=valid_start_time)
        y_train = y.crop(end=train_end_time)
        y_valid = y.crop(start=valid_start_time)

        return X_train, y_train, X_valid, y_valid

    def fit(
        self,
        X: TimeSeriesMatrix,
        y: TimeSeries | None = None,
    ) -> MLPreprocessor:
        """
        Learn statistics and filter coefficients.

        Parameters
        ----------
        X : TimeSeriesMatrix
            Reference channels (training data)
        y : TimeSeries or None, optional
            Target channel (training data)
            If None, skip y standardization.

        Returns
        -------
        self : MLPreprocessor
            Fitted preprocessor
        """
        from scipy.signal import butter

        from gwexpy.signal.preprocessing import standardize

        # 1. Learn X standardization parameters
        X_val = self._extract_value(X)  # Get ndarray
        X_std, X_model = standardize(
            X_val,
            method=self.standardization_method,
            axis=-1,  # Standardize along time axis (independent per channel)
            return_model=True,
        )
        self.X_scaler_ = X_model

        # 2. Design filter coefficients
        if self.freq_low is not None and self.freq_high is not None:
            sample_rate_hz = self._to_float_rate(self.sample_rate)
            self.filter_coeffs_ = []
            for f_low, f_high in zip(self.freq_low, self.freq_high):
                sos = butter(
                    self.filt_order,
                    [f_low, f_high],
                    btype="bandpass",
                    fs=sample_rate_hz,
                    output="sos",
                )
                self.filter_coeffs_.append(sos)
        else:
            self.filter_coeffs_ = None

        # 3. Learn y standardization parameters
        if y is not None:
            # Filter y
            y_filt = self._apply_bandpass(y)

            # Learn y standardization parameters
            y_val = self._extract_value(y_filt)
            y_std, y_model = standardize(
                y_val,
                method=self.standardization_method,
                axis=-1,
                return_model=True,
            )
            self.y_scaler_ = y_model
        else:
            self.y_scaler_ = None

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: TimeSeriesMatrix,
        y: TimeSeries | None = None,
    ) -> tuple[TimeSeriesMatrix, TimeSeries] | TimeSeriesMatrix:
        """
        Apply filtering and standardization.

        Parameters
        ----------
        X : TimeSeriesMatrix
            Reference channels
        y : TimeSeries or None, optional
            Target channel
            If None, return only X.

        Returns
        -------
        X_proc : TimeSeriesMatrix
            Processed X (dimensionless_unscaled unit)
        y_proc : TimeSeries (if y is specified)
            Processed y (dimensionless_unscaled unit)
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first")

        # 1. Standardize X
        X_val = self._extract_value(X)
        # Reshape mean and scale for broadcasting
        assert self.X_scaler_ is not None
        mean = self.X_scaler_.mean
        scale = self.X_scaler_.scale

        # Convert to array if scalar (handles single-channel cases)
        mean = np.atleast_1d(mean)
        scale = np.atleast_1d(scale)

        # Reshape according to the number of dimensions in X_val
        if X_val.ndim == 3:
            mean = mean[:, None, None]
            scale = scale[:, None, None]
        elif X_val.ndim == 2:
            mean = mean[:, None]
            scale = scale[:, None]
        X_std_val = (X_val - mean) / scale
        X_proc = self._reconstruct_timeseries_matrix(
            X_std_val, X, unit=u.dimensionless_unscaled
        )

        # 2. Handle y
        if y is not None:
            if self.y_scaler_ is None:
                raise RuntimeError("y was specified, but fit() did not use y.")

            # Filter y
            y_filt = self._apply_bandpass(y)

            # Standardize y
            y_val = self._extract_value(y_filt)
            y_mean = self.y_scaler_.mean
            y_scale = self.y_scaler_.scale

            # Convert to array if scalar
            y_mean = np.atleast_1d(y_mean)
            y_scale = np.atleast_1d(y_scale)

            # Reshape according to the number of dimensions in y_val
            if y_val.ndim > 1 and y_mean.ndim > 0:
                y_mean = y_mean[:, None]
                y_scale = y_scale[:, None]
            y_std_val = (y_val - y_mean) / y_scale
            y_proc = self._reconstruct_timeseries(
                y_std_val, y_filt, unit=u.dimensionless_unscaled
            )

            return X_proc, y_proc

        return X_proc

    # Helper methods (private)

    def _to_float_rate(self, sample_rate: Any) -> float:
        """Convert sample_rate to a float in Hz."""
        if hasattr(sample_rate, "to"):
            # Convert to Hz if it is a Quantity
            return float(sample_rate.to(u.Hz).value)
        # Return as is if it is a float
        return float(sample_rate)

    def _extract_value(self, ts) -> np.ndarray:
        """Retrieve ndarray from TimeSeries or TimeSeriesMatrix."""
        if hasattr(ts, "value"):
            return ts.value
        return np.asarray(ts)

    def _reconstruct_timeseries_matrix(self, val, original, unit=None):
        """Reconstruct TimeSeriesMatrix from ndarray."""
        if unit is not None:
            new_mat = original.__class__(val, t0=original.t0, dt=original.dt, unit=unit)
        else:
            new_mat = original.__class__(val, t0=original.t0, dt=original.dt)

        # Preserve metadata
        if hasattr(original, "channel_names"):
            new_mat.channel_names = original.channel_names

        return new_mat

    def _reconstruct_timeseries(self, val, original, unit=None):
        """Reconstruct TimeSeries from ndarray."""
        if unit is not None:
            new_ts = original.__class__(val, t0=original.t0, dt=original.dt, unit=unit)
        else:
            new_ts = original.__class__(val, t0=original.t0, dt=original.dt)

        # Preserve metadata
        if hasattr(original, "name"):
            new_ts.name = original.name

        return new_ts

    def _apply_bandpass(self, y):
        """Apply band-pass filter."""
        if self.filter_coeffs_ is None:
            # No filtering
            return y

        from scipy.signal import sosfiltfilt

        y_val = self._extract_value(y)
        y_filt_val = np.zeros_like(y_val)

        # Apply filters for each band and sum them
        for sos in self.filter_coeffs_:
            y_filt_val += sosfiltfilt(sos, y_val, axis=-1)

        # Reconstruct as TimeSeries
        return self._reconstruct_timeseries(y_filt_val, y)


__all__ = ["MLPreprocessor"]
