"""
Unit tests for gwexpy/interop/mt_.py using hierarchical fake classes.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries


# =============================================================================
# Fake MTH5 Hierarchy
# =============================================================================

class FakeChannel:
    def __init__(self, name, data=None, sample_rate=1.0, start=0.0, units="m"):
        self.name = name
        self.hdf5_dataset = np.asarray(data) if data is not None else np.array([0.0, 1.0])
        self.sample_rate = sample_rate
        self.start = start
        self.units = units


class FakeRunGroup:
    def __init__(self, name):
        self.name = name
        self.channels = {}
        self.groups_list = []

    def get_channel(self, name):
        if name not in self.channels:
            raise KeyError(name)
        return self.channels[name]

    def add_channel(self, name, channel_type, data=None, sample_rate=1.0, start=0.0, **kwargs):
        units = kwargs.get("units", "m")
        ch = FakeChannel(name, data, sample_rate, start, units=units)
        ch.channel_type = channel_type
        self.channels[name] = ch
        self.groups_list.append(name)
        return ch


class FakeStationGroup:
    def __init__(self, name):
        self.name = name
        self.runs = {}
        self.groups_list = []

    def get_run(self, name):
        if name not in self.runs:
            raise KeyError(name)
        return self.runs[name]

    def add_run(self, name):
        run = FakeRunGroup(name)
        self.runs[name] = run
        self.groups_list.append(name)
        return run


class FakeStationsGroup:
    def __init__(self):
        self.stations = {}
        self.groups_list = []

    def add_station(self, name):
        st = FakeStationGroup(name)
        self.stations[name] = st
        self.groups_list.append(name)
        return st
    
    def get_station(self, name):
        if name not in self.stations:
            raise KeyError(name)
        return self.stations[name]


class FakeSurveyGroup:
    def __init__(self, name):
        self.name = name
        self.stations_group = FakeStationsGroup()
        self.groups_list = []

    def get_station(self, name):
        return self.stations_group.get_station(name)


class FakeMTH5:
    def __init__(self, file_version="0.2.0"):
        self.file_version = file_version
        self.surveys = {}
        self.stations = {}
        self.station_list = []
        self.surveys_group = MagicMock()
        self.surveys_group.groups_list = []
        self._is_open = False

    def open_mth5(self, filename, mode="r"):
        self._is_open = True

    def close_mth5(self):
        self._is_open = False

    def add_survey(self, name):
        sur = FakeSurveyGroup(name)
        self.surveys[name] = sur
        self.surveys_group.groups_list.append(name)
        return sur

    def get_survey(self, name):
        if name not in self.surveys:
            raise KeyError(name)
        return self.surveys[name]

    def get_station(self, name, survey=None):
        if self.file_version == "0.2.0":
            if survey:
                return self.surveys[survey].get_station(name)
            raise AttributeError("survey required for v0.2.0")
        else:
            if name not in self.stations:
                raise KeyError(name)
            return self.stations[name]

    def add_station(self, name):
        st = FakeStationGroup(name)
        self.stations[name] = st
        if hasattr(self, "station_list"):
            self.station_list.append(name)
        return st


# =============================================================================
# Tests
# =============================================================================

@pytest.fixture
def mock_all():
    """Comprehensive mock for all dependencies."""
    mock_mth5_mod = MagicMock()
    
    # Simple MagicMock for TimeSeries constructor
    mock_ts_class = MagicMock(name="MOCK_TS_CONSTRUCTOR")
    mock_ts_instance = MagicMock(name="MOCK_TS_INSTANCE")
    mock_ts_class.return_value = mock_ts_instance
    
    # Patch the reference inside mt_ directly to be 100% sure
    from gwexpy.interop._registry import ConverterRegistry
    with patch.object(ConverterRegistry, "get_constructor", return_value=mock_ts_class):
        with patch("gwexpy.interop.mt_.require_optional", return_value=mock_mth5_mod):
            with patch.dict(sys.modules, {"mth5": mock_mth5_mod, "mth5.mth5": mock_mth5_mod}):
                # Mock astropy.time.Time 
                class FakeTime:
                    def __init__(self, val, format=None):
                        # Force fallback for specific strings in tests
                        if isinstance(val, str) and val == "TRIGGER_FALLBACK":
                            raise ValueError("MOCK_FAIL")
                        elif val == "2011-09-14T01:46:25":
                            self.gps = 1000000000.0
                        else:
                            try:
                                self.gps = float(val)
                            except:
                                self.gps = 0.0

                with patch("astropy.time.Time", FakeTime):
                    from gwexpy.interop import mt_
                    yield SimpleNamespace(
                        mt=mt_,
                        ts_class=mock_ts_class,
                        mth5=mock_mth5_mod
                    )


class TestFromMTH5Mock:
    def test_v020_survey_search(self, mock_all):
        m = FakeMTH5(file_version="0.2.0")
        sur1 = m.add_survey("S1")
        sur2 = m.add_survey("S2")
        st = sur2.stations_group.add_station("TargetStation")
        run = st.add_run("R1")
        run.add_channel("Ex", "electric", data=[1], sample_rate=100.0)
        
        # station in S2
        mock_all.mt.from_mth5(m, "TargetStation", "R1", "Ex")
        assert mock_all.ts_class.called

    def test_v020_explicit_survey(self, mock_all):
        m = FakeMTH5(file_version="0.2.0")
        sur = m.add_survey("TargetSurvey")
        st = sur.stations_group.add_station("S1")
        run = st.add_run("R1")
        run.add_channel("Ex", "electric", data=[10])
        
        mock_all.mt.from_mth5(m, "S1", "R1", "Ex", survey="TargetSurvey")
        np.testing.assert_array_equal(mock_all.ts_class.call_args[0][0], [10])

    def test_time_parsing_fallbacks(self, mock_all):
        m = FakeMTH5(file_version="0.1.0")
        st = m.add_station("S1")
        run = st.add_run("R1")
        
        # Astropy fail -> float success
        with patch("gwexpy.interop.mt_.float", return_value=1234.56):
            run.add_channel("FAIL_STR", "electric", data=[1], start="TRIGGER_FALLBACK")
            mock_all.mt.from_mth5(m, "S1", "R1", "FAIL_STR")
            assert float(mock_all.ts_class.call_args[1]["t0"].to_value("s")) == 1234.56
        
        # Total fail (float fails too) -> default 0
        mock_all.ts_class.reset_mock()
        run.add_channel("FAIL_ALL", "electric", data=[1], start="REALLY_BAD")
        with patch("gwexpy.interop.mt_.float", side_effect=ValueError):
             mock_all.mt.from_mth5(m, "S1", "R1", "FAIL_ALL")
             assert float(mock_all.ts_class.call_args[1]["t0"].to_value("s")) == 0.0

    def test_from_filename(self, mock_all):
        m = FakeMTH5(file_version="0.1.0")
        mock_all.mth5.MTH5.return_value = m
        st = m.add_station("S1")
        run = st.add_run("R1")
        run.add_channel("Ex", "electric", data=[1])
        
        mock_all.mt.from_mth5("data.h5", "S1", "R1", "Ex")
        assert m._is_open is False


class TestToMTH5Mock:
    def test_v020_existing_groups(self, mock_all):
        m = FakeMTH5(file_version="0.2.0")
        sur = m.add_survey("Survey01")
        st = sur.stations_group.add_station("S1")
        run = st.add_run("R1")
        
        ts = TimeSeries([1], dt=1, name="Ex")
        # Reuse existing survey, station, run
        mock_all.mt.to_mth5(ts, m, station="S1", run="R1")
        assert "Ex" in run.channels

    def test_sr_derivation_none(self, mock_all):
        m = FakeMTH5(file_version="0.1.0")
        ts = MagicMock(spec=TimeSeries)
        ts.value = [1]
        ts.name = "Ex"
        del ts.sample_rate
        ts.dt = None
        ts.t0 = None
        
        # Hits lines 128 & 136
        mock_all.mt.to_mth5(ts, m, station="S1", run="R1")
        ch = m.stations["S1"].runs["R1"].channels["Ex"]
        assert ch.sample_rate == 1.0
        assert ch.start == 0.0

    def test_start_fail_to_float(self, mock_all):
        m = FakeMTH5(file_version="0.1.0")
        ts = MagicMock(spec=TimeSeries)
        ts.value = [1]
        ts.name = "Ex"
        ts.sample_rate = 1 * u.Hz
        ts.t0 = MagicMock()
        ts.t0.to.return_value.value.__float__.side_effect = TypeError
        
        mock_all.mt.to_mth5(ts, m, station="S1", run="R1")
        assert m.stations["S1"].runs["R1"].channels["Ex"].start == 0.0


    def test_os_remove_oserror_suppression(self, mock_all):
        m = FakeMTH5(file_version="0.1.0")
        mock_all.mth5.MTH5.return_value = m
        ts = TimeSeries([1], dt=1)
        
        with patch("os.path.exists", return_value=True):
            with patch("os.path.getsize", return_value=0):
                with patch("os.remove", side_effect=OSError("MOCK_OS_ERROR")):
                    mock_all.mt.to_mth5(ts, "empty.h5")
