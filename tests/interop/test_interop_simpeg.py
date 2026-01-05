
import pytest
import numpy as np
from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries

try:
    import simpeg
    from simpeg.electromagnetics import time_domain as tdem
    from simpeg.electromagnetics import frequency_domain as fdem
    from simpeg.data import Data
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
        simpeg_data = ts.to_simpeg(location=[10, 0, 0], rx_type="PointElectricField", orientation='x')

        assert isinstance(simpeg_data, Data)
        assert len(simpeg_data.dobs) == 100

        # Check survey type
        assert isinstance(simpeg_data.survey, tdem.Survey)

        # Check Rx properties
        src = simpeg_data.survey.source_list[0]
        rx = src.receiver_list[0]

        # Check locations
        np.testing.assert_array_equal(rx.locations, [[10., 0., 0.]])

        # Check orientation
        # orientation is vector [1, 0, 0] for 'x'
        np.testing.assert_allclose(rx.orientation, [1., 0., 0.])

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
        simpeg_data = fs.to_simpeg(location=[0, 10, 0], rx_type="PointMagneticFluxDensity", orientation='z')

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
        np.testing.assert_array_equal(rx0.locations, [[0., 10., 0.]])

        # Check orientation
        # orientation is vector [0, 0, 1] for 'z'
        # SimPEG returns vector even if initialized with string
        np.testing.assert_allclose(rx0.orientation, [0., 0., 1.])

        # 3. From SimPEG
        fs_rec = FrequencySeries.from_simpeg(simpeg_data)

        assert isinstance(fs_rec, FrequencySeries)
        np.testing.assert_allclose(fs_rec.value, fs.value)
        np.testing.assert_allclose(fs_rec.frequencies.value, fs.frequencies.value)

