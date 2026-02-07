"""Collections for ScalarField objects in the `gwexpy.fields` namespace."""

from __future__ import annotations

import numpy as np
from astropy import units as u

from .scalar import ScalarField

__all__ = ["FieldList", "FieldDict"]

# Tolerance for axis coordinate comparison
_AXIS_RTOL = 1e-9
_AXIS_ATOL = 1e-12


class FieldList(list):
    """List-like collection for `ScalarField` objects with batch operations."""

    def __init__(self, items=None, validate=False):
        if items is None:
            items = []
        super().__init__(items)
        if validate:
            self._validate()

    def _validate(self):
        """Validate that all items are ScalarField with compatible metadata."""
        if not self:
            return

        first = self[0]
        if not isinstance(first, ScalarField):
            raise TypeError(f"Expected ScalarField, got {type(first)}")

        ref_unit = first.unit
        ref_axis_names = first.axis_names
        ref_axis0_domain = first.axis0_domain
        ref_space_domains = first.space_domains

        ref_axes = [
            first._axis0_index,
            first._axis1_index,
            first._axis2_index,
            first._axis3_index,
        ]

        for i, item in enumerate(self[1:], 1):
            if not isinstance(item, ScalarField):
                raise TypeError(f"Item {i}: Expected ScalarField, got {type(item)}")
            u_item = item.unit if item.unit is not None else u.dimensionless_unscaled
            u_ref = ref_unit if ref_unit is not None else u.dimensionless_unscaled
            if not u_item.is_equivalent(u_ref):
                raise ValueError(
                    f"Item {i}: Inconsistent unit. Expected equivalent to {ref_unit}, got {item.unit}"
                )
            if item.axis_names != ref_axis_names:
                raise ValueError(
                    f"Item {i}: Inconsistent axis_names. "
                    f"Expected {ref_axis_names}, got {item.axis_names}"
                )
            if item.axis0_domain != ref_axis0_domain:
                raise ValueError(
                    f"Item {i}: Inconsistent axis0_domain. "
                    f"Expected {ref_axis0_domain}, got {item.axis0_domain}"
                )
            if item.space_domains != ref_space_domains:
                raise ValueError(
                    f"Item {i}: Inconsistent space_domains. "
                    f"Expected {ref_space_domains}, got {item.space_domains}"
                )

            item_axes = [
                item._axis0_index,
                item._axis1_index,
                item._axis2_index,
                item._axis3_index,
            ]
            for ax_idx, (ref_ax, item_ax) in enumerate(zip(ref_axes, item_axes)):
                if ref_ax.shape != item_ax.shape:
                    raise ValueError(
                        f"Item {i}: Axis {ax_idx} shape mismatch. "
                        f"Expected {ref_ax.shape}, got {item_ax.shape}"
                    )
                ref_axis_unit = getattr(ref_ax, "unit", u.dimensionless_unscaled)
                item_axis_unit = getattr(item_ax, "unit", u.dimensionless_unscaled)
                if ref_axis_unit != item_axis_unit:
                    raise ValueError(
                        f"Item {i}: Axis {ax_idx} unit mismatch. "
                        f"Expected {ref_axis_unit}, got {item_axis_unit}"
                    )
                ref_val = getattr(ref_ax, "value", ref_ax)
                item_val = getattr(item_ax, "value", item_ax)
                if not np.allclose(
                    np.asarray(ref_val),
                    np.asarray(item_val),
                    rtol=_AXIS_RTOL,
                    atol=_AXIS_ATOL,
                ):
                    raise ValueError(
                        f"Item {i}: Axis {ax_idx} coordinate mismatch. "
                        f"Axis values differ beyond tolerance."
                    )

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields, returning FieldList."""
        return self.__class__([f.fft_time(**kwargs) for f in self])

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields, returning FieldList."""
        return self.__class__([f.ifft_time(**kwargs) for f in self])

    def fft_space_all(self, axes=None, **kwargs):
        """Apply fft_space to all fields, returning FieldList."""
        return self.__class__([f.fft_space(axes=axes, **kwargs) for f in self])

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space to all fields, returning FieldList."""
        return self.__class__([f.ifft_space(axes=axes, **kwargs) for f in self])

    def resample_all(self, rate, **kwargs):
        """Apply resample to all fields, returning FieldList."""
        return self.__class__([f.resample(rate, **kwargs) for f in self])

    def filter_all(self, *args, **kwargs):
        """Apply filter to all fields, returning FieldList."""
        return self.__class__([f.filter(*args, **kwargs) for f in self])

    def sel_all(self, **kwargs):
        """Apply sel to all fields, returning FieldList."""
        return self.__class__([f.sel(**kwargs) for f in self])

    def isel_all(self, **kwargs):
        """Apply isel to all fields, returning FieldList."""
        return self.__class__([f.isel(**kwargs) for f in self])


class FieldDict(dict):
    """Dict-like collection for `ScalarField` objects with batch operations."""

    def __init__(self, items=None, validate=False):
        if items is None:
            items = {}
        super().__init__(items)
        if validate:
            self._validate()

    def copy(self) -> FieldDict:
        """Return a copy of this FieldDict."""
        return self.__class__({k: v.copy() for k, v in self.items()})

    def __mul__(self, other):
        if np.isscalar(other):
            return self.__class__({k: v * other for k, v in self.items()})
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if np.isscalar(other):
            return self.__class__({k: v + other for k, v in self.items()})
        # Note: If other is FieldDict, we might want to Zip add?
        # But for now, stick to scalar as per plan Phase 2.
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            return self.__class__({k: v - other for k, v in self.items()})
        return NotImplemented

    def __rsub__(self, other):
        if np.isscalar(other):
            return self.__class__({k: other - v for k, v in self.items()})
        return NotImplemented

    def _validate(self):
        """Validate that all values are ScalarField with compatible metadata."""
        if not self:
            return

        values = list(self.values())
        first = values[0]
        if not isinstance(first, ScalarField):
            raise TypeError(f"Expected ScalarField, got {type(first)}")

        ref_unit = first.unit
        ref_axis_names = first.axis_names
        ref_axis0_domain = first.axis0_domain
        ref_space_domains = first.space_domains

        ref_axes = [
            first._axis0_index,
            first._axis1_index,
            first._axis2_index,
            first._axis3_index,
        ]

        for key, item in list(self.items())[1:]:
            if not isinstance(item, ScalarField):
                raise TypeError(f"Key '{key}': Expected ScalarField, got {type(item)}")
            u_item = item.unit if item.unit is not None else u.dimensionless_unscaled
            u_ref = ref_unit if ref_unit is not None else u.dimensionless_unscaled
            if not u_item.is_equivalent(u_ref):
                raise ValueError(
                    f"Key '{key}': Inconsistent unit. "
                    f"Expected equivalent to {ref_unit}, got {item.unit}"
                )
            if item.axis_names != ref_axis_names:
                raise ValueError(
                    f"Key '{key}': Inconsistent axis_names. "
                    f"Expected {ref_axis_names}, got {item.axis_names}"
                )
            if item.axis0_domain != ref_axis0_domain:
                raise ValueError(
                    f"Key '{key}': Inconsistent axis0_domain. "
                    f"Expected {ref_axis0_domain}, got {item.axis0_domain}"
                )
            if item.space_domains != ref_space_domains:
                raise ValueError(
                    f"Key '{key}': Inconsistent space_domains. "
                    f"Expected {ref_space_domains}, got {item.space_domains}"
                )

            item_axes = [
                item._axis0_index,
                item._axis1_index,
                item._axis2_index,
                item._axis3_index,
            ]
            for ax_idx, (ref_ax, item_ax) in enumerate(zip(ref_axes, item_axes)):
                if ref_ax.shape != item_ax.shape:
                    raise ValueError(
                        f"Key '{key}': Axis {ax_idx} shape mismatch. "
                        f"Expected {ref_ax.shape}, got {item_ax.shape}"
                    )
                ref_axis_unit = getattr(ref_ax, "unit", u.dimensionless_unscaled)
                item_axis_unit = getattr(item_ax, "unit", u.dimensionless_unscaled)
                if ref_axis_unit != item_axis_unit:
                    raise ValueError(
                        f"Key '{key}': Axis {ax_idx} unit mismatch. "
                        f"Expected {ref_axis_unit}, got {item_axis_unit}"
                    )
                ref_val = getattr(ref_ax, "value", ref_ax)
                item_val = getattr(item_ax, "value", item_ax)
                if not np.allclose(
                    np.asarray(ref_val),
                    np.asarray(item_val),
                    rtol=_AXIS_RTOL,
                    atol=_AXIS_ATOL,
                ):
                    raise ValueError(
                        f"Key '{key}': Axis {ax_idx} coordinate mismatch. "
                        f"Axis values differ beyond tolerance."
                    )

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields, returning FieldDict."""
        return self.__class__({k: v.fft_time(**kwargs) for k, v in self.items()})

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields, returning FieldDict."""
        return self.__class__({k: v.ifft_time(**kwargs) for k, v in self.items()})

    def fft_space_all(self, axes=None, **kwargs):
        """Apply fft_space to all fields, returning FieldDict."""
        return self.__class__(
            {k: v.fft_space(axes=axes, **kwargs) for k, v in self.items()}
        )

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space to all fields, returning FieldDict."""
        return self.__class__(
            {k: v.ifft_space(axes=axes, **kwargs) for k, v in self.items()}
        )

    def resample_all(self, rate, **kwargs):
        """Apply resample to all fields, returning FieldDict."""
        return self.__class__({k: v.resample(rate, **kwargs) for k, v in self.items()})

    def filter_all(self, *args, **kwargs):
        """Apply filter to all fields, returning FieldDict."""
        return self.__class__({k: v.filter(*args, **kwargs) for k, v in self.items()})

    def sel_all(self, **kwargs):
        """Apply sel to all fields, returning FieldDict."""
        return self.__class__({k: v.sel(**kwargs) for k, v in self.items()})

    def isel_all(self, **kwargs):
        """Apply isel to all fields, returning FieldDict."""
        return self.__class__({k: v.isel(**kwargs) for k, v in self.items()})

    # Spectral analysis methods - apply to each component
    def psd(self, axis=0, **kwargs):
        """Compute power spectral density for each component.

        Applies psd() to each ScalarField component and returns a new FieldDict.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute PSD. Default is 0 (time axis).
        **kwargs
            Additional arguments passed to ScalarField.psd().

        Returns
        -------
        FieldDict
            New FieldDict with PSD of each component.

        Examples
        --------
        >>> vfield_psd = vector_field.psd(axis=0, fftlength=1.0)
        >>> # Each component's PSD: {'x': Ex_psd, 'y': Ey_psd, 'z': Ez_psd}
        """
        return self.__class__({k: v.psd(axis=axis, **kwargs) for k, v in self.items()})

    def asd(self, axis=0, **kwargs):
        """Compute amplitude spectral density for each component.

        Applies asd() to each ScalarField component and returns a new FieldDict.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute ASD. Default is 0 (time axis).
        **kwargs
            Additional arguments passed to ScalarField.asd().

        Returns
        -------
        FieldDict
            New FieldDict with ASD of each component.
        """
        return self.__class__({k: v.asd(axis=axis, **kwargs) for k, v in self.items()})

    # Signal processing methods
    def filter(self, *args, **kwargs):
        """Apply filter to each component.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to ScalarField.filter().

        Returns
        -------
        FieldDict
            Filtered FieldDict.
        """
        return self.__class__({k: v.filter(*args, **kwargs) for k, v in self.items()})

    def resample(self, rate, **kwargs):
        """Resample each component.

        Parameters
        ----------
        rate : float or Quantity
            New sample rate.
        **kwargs
            Additional arguments passed to ScalarField.resample().

        Returns
        -------
        FieldDict
            Resampled FieldDict.
        """
        return self.__class__({k: v.resample(rate, **kwargs) for k, v in self.items()})

    def astype(self, dtype, **kwargs):
        """Convert each component to specified data type.

        Parameters
        ----------
        dtype : dtype
            Target data type.
        **kwargs
            Additional arguments.

        Returns
        -------
        FieldDict
            FieldDict with converted types.
        """
        return self.__class__({k: v.astype(dtype, **kwargs) for k, v in self.items()})

    # Filtering methods
    def highpass(self, frequency, **kwargs):
        """Apply highpass filter to each component.

        Parameters
        ----------
        frequency : float
            Cutoff frequency.
        **kwargs
            Additional arguments passed to filter method.

        Returns
        -------
        FieldDict
            Highpass-filtered FieldDict.
        """
        return self.__class__({k: v.highpass(frequency, **kwargs) for k, v in self.items()})

    def lowpass(self, frequency, **kwargs):
        """Apply lowpass filter to each component.

        Parameters
        ----------
        frequency : float
            Cutoff frequency.
        **kwargs
            Additional arguments passed to filter method.

        Returns
        -------
        FieldDict
            Lowpass-filtered FieldDict.
        """
        return self.__class__({k: v.lowpass(frequency, **kwargs) for k, v in self.items()})

    def bandpass(self, flow, fhigh, **kwargs):
        """Apply bandpass filter to each component.

        Parameters
        ----------
        flow : float
            Low frequency cutoff.
        fhigh : float
            High frequency cutoff.
        **kwargs
            Additional arguments passed to filter method.

        Returns
        -------
        FieldDict
            Bandpass-filtered FieldDict.
        """
        return self.__class__({k: v.bandpass(flow, fhigh, **kwargs) for k, v in self.items()})

    def notch(self, frequency, **kwargs):
        """Apply notch filter to each component.

        Parameters
        ----------
        frequency : float or list of float
            Frequency(ies) to notch.
        **kwargs
            Additional arguments passed to filter method.

        Returns
        -------
        FieldDict
            Notch-filtered FieldDict.
        """
        return self.__class__({k: v.notch(frequency, **kwargs) for k, v in self.items()})

    def zpk(self, zeros, poles, gain, **kwargs):
        """Apply zero-pole-gain filter to each component.

        Parameters
        ----------
        zeros : array_like
            Filter zeros.
        poles : array_like
            Filter poles.
        gain : float
            Filter gain.
        **kwargs
            Additional arguments.

        Returns
        -------
        FieldDict
            Filtered FieldDict.
        """
        return self.__class__({k: v.zpk(zeros, poles, gain, **kwargs) for k, v in self.items()})

    # FFT methods
    def fft(self, axis=0, **kwargs):
        """Apply FFT to each component.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute FFT. Default is 0.
        **kwargs
            Additional arguments.

        Returns
        -------
        FieldDict
            FFT of each component.
        """
        return self.__class__({k: v.fft(axis=axis, **kwargs) for k, v in self.items()})

    def ifft(self, axis=0, **kwargs):
        """Apply inverse FFT to each component.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute IFFT. Default is 0.
        **kwargs
            Additional arguments.

        Returns
        -------
        FieldDict
            IFFT of each component.
        """
        return self.__class__({k: v.ifft(axis=axis, **kwargs) for k, v in self.items()})

    # Preprocessing methods
    def detrend(self, type="linear"):
        """Apply detrend to each component.

        Parameters
        ----------
        type : str, optional
            Type of detrending ('linear' or 'constant'). Default is 'linear'.

        Returns
        -------
        FieldDict
            Detrended FieldDict.
        """
        return self.__class__({k: v.detrend(type=type) for k, v in self.items()})

    def taper(self, **kwargs):
        """Apply taper to each component.

        Parameters
        ----------
        **kwargs
            Arguments passed to ScalarField.taper().

        Returns
        -------
        FieldDict
            Tapered FieldDict.
        """
        return self.__class__({k: v.taper(**kwargs) for k, v in self.items()})

    def crop(self, start=None, end=None, copy=True):
        """Crop each component.

        Parameters
        ----------
        start : float or Quantity, optional
            Start time.
        end : float or Quantity, optional
            End time.
        copy : bool, optional
            Whether to copy data. Default is True.

        Returns
        -------
        FieldDict
            Cropped FieldDict.
        """
        return self.__class__({k: v.crop(start=start, end=end, copy=copy) for k, v in self.items()})

    def pad(self, pad_width, **kwargs):
        """Pad each component.

        Parameters
        ----------
        pad_width : int or tuple
            Number of values to pad.
        **kwargs
            Additional arguments passed to ScalarField.pad().

        Returns
        -------
        FieldDict
            Padded FieldDict.
        """
        return self.__class__({k: v.pad(pad_width, **kwargs) for k, v in self.items()})

    # Mathematical operations
    def abs(self):
        """Compute absolute value of each component.

        Returns
        -------
        FieldDict
            Absolute values.
        """
        return self.__class__({k: v.abs() for k, v in self.items()})

    def sqrt(self):
        """Compute square root of each component.

        Returns
        -------
        FieldDict
            Square root values.
        """
        return self.__class__({k: v.sqrt() for k, v in self.items()})

    def mean(self, axis=None, **kwargs):
        """Compute mean of each component.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute mean.
        **kwargs
            Additional arguments.

        Returns
        -------
        FieldDict
            Mean values.
        """
        return self.__class__({k: v.mean(axis=axis, **kwargs) for k, v in self.items()})

    def median(self, axis=None, **kwargs):
        """Compute median of each component.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute median.
        **kwargs
            Additional arguments.

        Returns
        -------
        FieldDict
            Median values.
        """
        return self.__class__({k: v.median(axis=axis, **kwargs) for k, v in self.items()})

    def std(self, axis=None, **kwargs):
        """Compute standard deviation of each component.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute std.
        **kwargs
            Additional arguments.

        Returns
        -------
        FieldDict
            Standard deviation values.
        """
        return self.__class__({k: v.std(axis=axis, **kwargs) for k, v in self.items()})

    def rms(self, axis=None):
        """Compute RMS of each component.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute RMS.

        Returns
        -------
        FieldDict
            RMS values.
        """
        return self.__class__({k: v.rms(axis=axis) for k, v in self.items()})

    # Advanced signal processing
    def whiten(self, **kwargs):
        """Whiten each component.

        Parameters
        ----------
        **kwargs
            Arguments passed to ScalarField.whiten().

        Returns
        -------
        FieldDict
            Whitened FieldDict.
        """
        return self.__class__({k: v.whiten(**kwargs) for k, v in self.items()})

    def convolve(self, fir, **kwargs):
        """Convolve each component with FIR filter.

        Parameters
        ----------
        fir : array_like
            Filter coefficients.
        **kwargs
            Additional arguments passed to ScalarField.convolve().

        Returns
        -------
        FieldDict
            Convolved FieldDict.
        """
        return self.__class__({k: v.convolve(fir, **kwargs) for k, v in self.items()})

    def inject(self, other, alpha=1.0):
        """Inject signal into each component.

        Parameters
        ----------
        other : FieldDict or ScalarField
            Signal to inject. If ScalarField, injects into all components.
            If FieldDict, injects matching components.
        alpha : float, optional
            Scaling factor. Default is 1.0.

        Returns
        -------
        FieldDict
            FieldDict with injected signal.
        """
        if isinstance(other, FieldDict):
            return self.__class__({k: v.inject(other[k], alpha=alpha) for k, v in self.items()})
        else:
            # Inject same signal into all components
            return self.__class__({k: v.inject(other, alpha=alpha) for k, v in self.items()})

    # Cross-spectral analysis
    def csd(self, other, **kwargs):
        """Compute CSD between corresponding components.

        Parameters
        ----------
        other : FieldDict
            Other FieldDict with matching keys.
        **kwargs
            Arguments passed to ScalarField.csd().

        Returns
        -------
        FieldDict
            CSD for each component pair.
        """
        if not isinstance(other, FieldDict):
            raise TypeError("other must be a FieldDict")
        return self.__class__({k: v.csd(other[k], **kwargs) for k, v in self.items() if k in other})

    def coherence(self, other, **kwargs):
        """Compute coherence between corresponding components.

        Parameters
        ----------
        other : FieldDict
            Other FieldDict with matching keys.
        **kwargs
            Arguments passed to ScalarField.coherence().

        Returns
        -------
        FieldDict
            Coherence for each component pair.
        """
        if not isinstance(other, FieldDict):
            raise TypeError("other must be a FieldDict")
        return self.__class__({k: v.coherence(other[k], **kwargs) for k, v in self.items() if k in other})

    def spectrogram(self, stride, **kwargs):
        """Compute spectrogram of each component.

        Parameters
        ----------
        stride : float
            Time step between consecutive spectrograms.
        **kwargs
            Arguments passed to ScalarField.spectrogram().

        Returns
        -------
        FieldDict
            Spectrogram for each component.
        """
        return self.__class__({k: v.spectrogram(stride, **kwargs) for k, v in self.items()})

    # Time series utilities
    def append(self, other, **kwargs):
        """Append another FieldDict.

        Parameters
        ----------
        other : FieldDict
            FieldDict to append.
        **kwargs
            Arguments passed to ScalarField.append().

        Returns
        -------
        FieldDict
            Concatenated FieldDict.
        """
        if not isinstance(other, FieldDict):
            raise TypeError("other must be a FieldDict")
        return self.__class__({k: v.append(other[k], **kwargs) for k, v in self.items() if k in other})

    def prepend(self, other, **kwargs):
        """Prepend another FieldDict.

        Parameters
        ----------
        other : FieldDict
            FieldDict to prepend.
        **kwargs
            Arguments passed to ScalarField.prepend().

        Returns
        -------
        FieldDict
            Concatenated FieldDict.
        """
        if not isinstance(other, FieldDict):
            raise TypeError("other must be a FieldDict")
        return self.__class__({k: v.prepend(other[k], **kwargs) for k, v in self.items() if k in other})

    # Correlation analysis
    def autocorrelation(self, **kwargs):
        """Compute autocorrelation of each component.

        Parameters
        ----------
        **kwargs
            Arguments passed to ScalarField.autocorrelation().

        Returns
        -------
        FieldDict
            Autocorrelation for each component.
        """
        return self.__class__({k: v.autocorrelation(**kwargs) for k, v in self.items()})

    def correlate(self, other, **kwargs):
        """Cross-correlate with another FieldDict.

        Parameters
        ----------
        other : FieldDict
            Other FieldDict with matching keys.
        **kwargs
            Arguments passed to ScalarField.correlate().

        Returns
        -------
        FieldDict
            Cross-correlation for each component pair.
        """
        if not isinstance(other, FieldDict):
            raise TypeError("other must be a FieldDict")
        return self.__class__({k: v.correlate(other[k], **kwargs) for k, v in self.items() if k in other})

    # Resampling
    def interpolate(self, sample_rate, **kwargs):
        """Interpolate each component to new sample rate.

        Parameters
        ----------
        sample_rate : float or Quantity
            New sample rate.
        **kwargs
            Arguments passed to ScalarField.interpolate().

        Returns
        -------
        FieldDict
            Interpolated FieldDict.
        """
        return self.__class__({k: v.interpolate(sample_rate, **kwargs) for k, v in self.items()})

    # Rayleigh statistics
    def rayleigh_spectrum(self, **kwargs):
        """Compute Rayleigh spectrum of each component.

        Parameters
        ----------
        **kwargs
            Arguments passed to ScalarField.rayleigh_spectrum().

        Returns
        -------
        FieldDict
            Rayleigh spectrum for each component.
        """
        return self.__class__({k: v.rayleigh_spectrum(**kwargs) for k, v in self.items()})

    def rayleigh_spectrogram(self, stride, **kwargs):
        """Compute Rayleigh spectrogram of each component.

        Parameters
        ----------
        stride : float
            Time step between consecutive spectrograms.
        **kwargs
            Arguments passed to ScalarField.rayleigh_spectrogram().

        Returns
        -------
        FieldDict
            Rayleigh spectrogram for each component.
        """
        return self.__class__({k: v.rayleigh_spectrogram(stride, **kwargs) for k, v in self.items()})
