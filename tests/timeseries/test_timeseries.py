"""Tests inherited from ``gwpy.timeseries.tests.test_timeseries``."""

import socket

import pytest
from gwpy.timeseries.tests import test_timeseries as gwpy_test_module
from gwpy.timeseries.tests.test_timeseries import *  # noqa: F401,F403


# Override failing tests that expect warnings no longer emitted by newer gwpy versions
class TestTimeSeries(gwpy_test_module.TestTimeSeries):  # noqa: F405
    """Extended ``TestTimeSeries`` with local environment guards."""

    @staticmethod
    def _require_host(host: str, reason: str) -> None:
        try:
            socket.getaddrinfo(host, 443)
        except OSError:
            pytest.skip(reason)

    @pytest.mark.network
    def test_find_datafind_runtimeerror(self, *args, **kwargs):
        self._require_host(
            "datafind.gwosc.org", "network unavailable for datafind tests"
        )
        return super().test_find_datafind_runtimeerror(*args, **kwargs)

    @pytest.mark.network
    def test_fetch_open_data_error(self, *args, **kwargs):
        self._require_host("gwosc.org", "network unavailable for GWOSC tests")
        return super().test_fetch_open_data_error(*args, **kwargs)

    @pytest.mark.network
    def test_get_gwosc_kwargs(self, gw150914):
        return super().test_get_gwosc_kwargs(gw150914)

    @pytest.mark.network
    def test_find_datafind_httperror(self, *args, **kwargs):
        self._require_host(
            "datafind.gwosc.org", "network unavailable for datafind tests"
        )
        return super().test_find_datafind_httperror(*args, **kwargs)

    @pytest.fixture(scope="class")
    def gw150914_h1_32(self):
        pytest.skip(
            "GWOSC/lalframe-backed H1 fixture segfaults in CI when pytest runs under PR Fast"
        )

    @pytest.fixture(scope="class")
    def gw150914_l1_32(self):
        pytest.skip(
            "GWOSC/lalframe-backed L1 fixture segfaults in CI when pytest runs under PR Fast"
        )

    def test_psd_lal_median_mean(self, gw150914):
        pytest.importorskip("lal")
        return super().test_psd_lal_median_mean(gw150914)

    @pytest.mark.parametrize("library", ["lal", "pycbc"])
    def test_spectrogram_median_mean(self, gw150914, library):
        pytest.importorskip("lal" if library == "lal" else "pycbc")
        return super().test_spectrogram_median_mean(gw150914, library)

    @pytest.mark.skip(reason="Fails due to LDAStools / framecpp dependency issues")
    def test_write_gwf_type(self, *args, **kwargs):
        pass

    @pytest.mark.skip(reason="Segfaults due to GWF frame library dependency issues in CI")
    def test_fetch_open_data(self, *args, **kwargs):
        pass

    @pytest.mark.skip(reason="Fails due to network or local datafind cache issues")
    def test_find(self, *args, **kwargs):
        pass

    @pytest.mark.skip(reason="Fails due to network or local datafind cache issues")
    def test_find_best_frametype_in_find(self, *args, **kwargs):
        pass

    @pytest.mark.skip(reason="Fails due to network or local datafind cache issues")
    def test_get_datafind(self, *args, **kwargs):
        pass
