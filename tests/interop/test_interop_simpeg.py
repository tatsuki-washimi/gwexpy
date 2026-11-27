import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries

try:
    import simpeg
    from simpeg.data import Data
    from simpeg.electromagnetics import frequency_domain as fdem
    from simpeg.electromagnetics import time_domain as tdem
except ImportError:
    simpeg = None


@pytest.mark.skipif(simpeg is None, reason="simpeg not installed")
class TestSimPEGInterop:
    def test_timeseries_to_simpeg(self):
        """Test TimeSeries -> SimPEG Data -> TimeSeries cycle."""

        # 1. Create TimeSeries
        data = np.random.randn(100)
        t0 = 0.0
        dt = 0.01
        ts = TimeSeries(data, t0=t0, dt=dt, unit="V", name="V_meas")

        # 2. To SimPEG
        simpeg_data = ts.to_simpeg(
            location=[10, 0, 0], rx_type="PointElectricField", orientation="x"
        )

        assert isinstance(simpeg_data, Data)
        assert len(simpeg_data.dobs) == 100

        # Check survey type
        assert isinstance(simpeg_data.survey, tdem.Survey)

        # Check Rx properties
        src = simpeg_data.survey.source_list[0]
        rx = src.receiver_list[0]

        # Check locations
        np.testing.assert_array_equal(rx.locations, [[10.0, 0.0, 0.0]])

        # Check orientation
        # orientation is vector [1, 0, 0] for 'x'
        np.testing.assert_allclose(rx.orientation, [1.0, 0.0, 0.0])

        # Check times
        np.testing.assert_allclose(rx.times, ts.times.value)

        # 3. From SimPEG
        ts_rec = TimeSeries.from_simpeg(simpeg_data)

        assert isinstance(ts_rec, TimeSeries)
        np.testing.assert_allclose(ts_rec.value, ts.value)
        assert ts_rec.dt.value == ts.dt.value
        assert ts_rec.t0.value == ts.t0.value

    def test_frequencyseries_to_simpeg(self):
        """Test FrequencySeries -> SimPEG Data -> FrequencySeries cycle."""

        # 1. Create FrequencySeries
        data = np.random.randn(10)
        freqs = np.linspace(1, 10, 10)
        fs = FrequencySeries(data, frequencies=freqs, unit="V", name="VF_meas")

        # 2. To SimPEG
        simpeg_data = fs.to_simpeg(
            location=[0, 10, 0], rx_type="PointMagneticFluxDensity", orientation="z"
        )

        assert isinstance(simpeg_data, Data)
        assert len(simpeg_data.dobs) == 10

        # Check survey type
        assert isinstance(simpeg_data.survey, fdem.Survey)

        # FDEM has 1 source per frequency
        assert len(simpeg_data.survey.source_list) == 10

        # Check Rx properties of first source
        src0 = simpeg_data.survey.source_list[0]
        rx0 = src0.receiver_list[0]

        # Check locations
        np.testing.assert_array_equal(rx0.locations, [[0.0, 10.0, 0.0]])

        # Check orientation
        # orientation is vector [0, 0, 1] for 'z'
        # SimPEG returns vector even if initialized with string
        np.testing.assert_allclose(rx0.orientation, [0.0, 0.0, 1.0])

        # 3. From SimPEG
        fs_rec = FrequencySeries.from_simpeg(simpeg_data)

        assert isinstance(fs_rec, FrequencySeries)
        np.testing.assert_allclose(fs_rec.value, fs.value)
        np.testing.assert_allclose(fs_rec.frequencies.value, fs.frequencies.value)


class TestSimpegScaleInvarianceMocked:
    """Verify SimPEG TDEM uniformity check without requiring simpeg installed."""

    @staticmethod
    def _mock_tdem_data(times):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        class FakeData:
            def __init__(self, dobs, survey):
                self.dobs = dobs
                self.survey = survey

        rx = SimpleNamespace(times=np.asarray(times, dtype=float))
        src = SimpleNamespace(receiver_list=[rx])
        survey = SimpleNamespace(source_list=[src])
        fake_simpeg = MagicMock()
        fake_simpeg.data.Data = FakeData
        return fake_simpeg, FakeData(np.ones(len(times)), survey)

    def test_non_uniform_tdem_raises(self, monkeypatch):
        from gwexpy.interop import simpeg_ as simpeg_mod

        fake_simpeg, data_obj = self._mock_tdem_data([0.0, 1.0, 3.0, 6.0])
        monkeypatch.setattr(simpeg_mod, "require_optional", lambda name: fake_simpeg)
        with pytest.raises(ValueError, match="non-uniform"):
            simpeg_mod.from_simpeg(TimeSeries, data_obj)

    def test_small_scale_non_uniform_tdem_is_not_linearized(self, monkeypatch):
        # numpy's default absolute tolerance accepts this axis as before, but
        # its first interval would move the later samples: keep them explicit.
        from gwexpy.interop import simpeg_ as simpeg_mod

        times = np.array([0.0, 1e-20, 3e-20, 6e-20])
        fake_simpeg, data_obj = self._mock_tdem_data(times)
        monkeypatch.setattr(simpeg_mod, "require_optional", lambda name: fake_simpeg)
        ts = simpeg_mod.from_simpeg(TimeSeries, data_obj)
        np.testing.assert_array_equal(ts.times.value, times)

    def test_small_scale_uniform_tdem_succeeds(self, monkeypatch):
        from gwexpy.interop import simpeg_ as simpeg_mod

        fake_simpeg, data_obj = self._mock_tdem_data([1e-20, 2e-20, 3e-20, 4e-20])
        monkeypatch.setattr(simpeg_mod, "require_optional", lambda name: fake_simpeg)
        ts = simpeg_mod.from_simpeg(TimeSeries, data_obj)
        assert ts.dt.value == pytest.approx(1e-20, rel=1e-12, abs=0.0)
        assert ts.t0.value == pytest.approx(1e-20, rel=1e-12, abs=0.0)

    def test_near_uniform_tdem_keeps_explicit_times(self, monkeypatch):
        from gwexpy.interop import simpeg_ as simpeg_mod

        times = np.array([0.0, 1.0, 2.0 + 1e-7, 3.0 + 2e-7])
        fake_simpeg, data_obj = self._mock_tdem_data(times)
        monkeypatch.setattr(simpeg_mod, "require_optional", lambda name: fake_simpeg)
        ts = simpeg_mod.from_simpeg(TimeSeries, data_obj)
        np.testing.assert_array_equal(ts.times.value, times)
