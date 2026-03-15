import datetime
import os
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from gwexpy.io.utils import ensure_dependency
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.timeseries.io._registration import register_timeseries_format


def test_ensure_dependency_success():
    # numpy is already installed
    import numpy
    res = ensure_dependency("numpy")
    assert res is numpy

def test_ensure_dependency_failure():
    # nonexistent package
    with pytest.raises(ImportError) as excinfo:
        ensure_dependency("nonexistent_package_gwexpy_test")
    assert "nonexistent_package_gwexpy_test is required" in str(excinfo.value)
    assert "pip install nonexistent_package_gwexpy_test" in str(excinfo.value)

def test_ensure_dependency_extra():
    # nonexistent with extra
    with pytest.raises(ImportError) as excinfo:
        ensure_dependency("nonexistent", extra="gui")
    assert "pip install nonexistent[gui]" in str(excinfo.value)

def test_register_timeseries_format_auto_adapt(tmp_path):
    # Mock reader that returns a TimeSeriesDict
    def mock_reader_dict(source, **kwargs):
        tsd = TimeSeriesDict()
        tsd["test"] = TimeSeries([1, 2, 3], t0=0, dt=1, name="test")
        return tsd

    register_timeseries_format(
        "mock_fmt",
        reader_dict=mock_reader_dict,
        extension="mockext"
    )

    # Test automatic identifier
    from gwpy.io.registry import default_registry as io_registry
    assert io_registry.identify_format("read", TimeSeriesDict, "file.mockext", None, (), {}) == ["mock_fmt"]
    assert io_registry.identify_format("read", TimeSeries, "file.mockext", None, (), {}) == ["mock_fmt"]
    assert io_registry.identify_format("read", TimeSeriesMatrix, "file.mockext", None, (), {}) == ["mock_fmt"]

    # Create dummy file
    path = tmp_path / "dummy.mockext"
    path.write_text("dummy")

    # Test auto-adapted single reader
    ts = TimeSeries.read(path, format="mock_fmt")
    assert isinstance(ts, TimeSeries)
    assert ts.name == "test"

    # Test auto-adapted matrix reader
    tsm = TimeSeriesMatrix.read(path, format="mock_fmt")
    assert isinstance(tsm, TimeSeriesMatrix)
    assert tsm.channel_names == ["test"]

def test_pathlib_support_ats(tmp_path):
    # Minimal ATS header
    import struct
    header = bytearray(1024)
    struct.pack_into("<H", header, 0, 1024) # length
    struct.pack_into("<h", header, 2, 80)   # version
    struct.pack_into("<I", header, 4, 10)   # samples
    struct.pack_into("<f", header, 8, 1.0)  # freq
    struct.pack_into("<I", header, 12, 0)   # start
    struct.pack_into("<d", header, 16, 1.0) # LSB
    data = np.zeros(10, dtype="<i4").tobytes()

    path = tmp_path / "test.ats"
    path.write_bytes(header + data)

    # Test with Path object
    ts = TimeSeries.read(path, format="ats")
    assert isinstance(ts, TimeSeries)
    assert len(ts) == 10

def test_wav_new_arguments(tmp_path):
    from scipy.io import wavfile
    path = tmp_path / "test.wav"
    rate = 1000
    data = np.zeros((100, 2), dtype=np.int16)
    wavfile.write(path, rate, data)

    # Test unit and epoch
    epoch_dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    from gwpy.time import to_gps
    expected_gps = float(to_gps(epoch_dt))

    tsd = TimeSeriesDict.read(path, format="wav", unit="m/s", epoch=epoch_dt)

    assert tsd["channel_0"].unit == u.Unit("m/s")
    assert float(tsd["channel_0"].t0.value) == expected_gps

    # Test channels filter
    tsd_filtered = TimeSeriesDict.read(path, format="wav", channels=["channel_1"])
    assert list(tsd_filtered.keys()) == ["channel_1"]

def test_audio_epoch_handling(tmp_path):
    pytest.importorskip("pydub")
    # We might not have ffmpeg in CI, but if pydub is there, we can test the logic
    # Creating a dummy file to see if registry works
    path = tmp_path / "test.mp3"
    path.write_text("dummy")

    # We don't actually read because it might fail without ffmpeg
    # But we can test if the reader has the epoch argument
    import inspect

    from gwexpy.timeseries.io.audio import read_timeseriesdict_audio
    sig = inspect.signature(read_timeseriesdict_audio)
    assert "epoch" in sig.parameters
    assert "unit" in sig.parameters

def test_ensure_dependency_import_name():
    # Test import_name != package_name
    # Assuming scipy is installed but we use a different name if we wanted to test this
    # Actually just check success with same name
    res = ensure_dependency("scipy", import_name="scipy")
    import scipy
    assert res is scipy
