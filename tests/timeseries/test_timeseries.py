"""
Tests inherited from gwpy.timeseries.tests.test_timeseries.

Some tests may be marked as xfail due to upstream API changes.
"""
import pytest

from gwpy.timeseries.tests import test_timeseries as gwpy_test_module
from gwpy.timeseries.tests.test_timeseries import *  # noqa: F401,F403


# Override failing tests that expect warnings no longer emitted by newer gwpy versions
class TestTimeSeries(gwpy_test_module.TestTimeSeries):  # noqa: F405
    """Extended TestTimeSeries with xfail markers for known upstream issues."""

    @pytest.mark.xfail(
        reason="gwpy no longer emits UserWarning for median_mean with lal",
        strict=False,
    )
    def test_psd_lal_median_mean(self, *args, **kwargs):
        return super().test_psd_lal_median_mean(*args, **kwargs)

    @pytest.mark.xfail(
        reason="gwpy no longer emits DeprecationWarning for median_mean with lal",
        strict=False,
    )
    def test_spectrogram_median_mean(self, *args, **kwargs):
        return super().test_spectrogram_median_mean(*args, **kwargs)
