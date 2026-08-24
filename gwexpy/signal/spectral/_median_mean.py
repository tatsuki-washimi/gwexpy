"""GWexpy-owned registration for the LAL median-mean PSD method."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gwpy.signal.spectral import register_method
from gwpy.signal.spectral._lal import median_mean as _lal_median_mean

if TYPE_CHECKING:
    from gwpy.frequencyseries import FrequencySeries
    from gwpy.timeseries import TimeSeries


def median_mean(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,
    window: Any | None = None,
    plan: Any | None = None,
) -> FrequencySeries:
    """Calculate a median-mean PSD using LAL's FINDCHIRP implementation.

    This is a GWexpy extension exposed through the GWpy-compatible
    ``TimeSeries.psd(method="median-mean")`` method surface.  The numerical
    implementation is delegated to LAL; the Allen et al. FINDCHIRP reference
    is the same source used for the median-bias correction in
    :func:`gwexpy.signal.spectral.median_bias`.

    LAL is an optional gravitational-wave backend.  Importing this module and
    looking up the method does not import LAL, but calculating a spectrum
    requires that backend to be installed.
    """
    return _lal_median_mean(
        timeseries,
        segmentlength,
        noverlap=noverlap,
        window=window,
        plan=plan,
    )


register_method(median_mean, name="median-mean")

__all__ = ["median_mean"]
