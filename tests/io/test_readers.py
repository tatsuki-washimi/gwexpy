from datetime import datetime
from datetime import timezone as dt_timezone
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import to_gps

from gwexpy.timeseries import TimeSeriesDict
from gwexpy.timeseries.io.gbd import GBD_FULL_SCALE


def _write_minimal_gbd(tmp_path, counts=4):
    channels = ["CH1", "CH2"]
    dt = 0.5
    start_str = "2024/01/01 00:00:00"
    header_lines = [
        "HeaderSiz=0",
        f"$$Time Start={start_str}",
        "$$Time Stop=2024/01/01 00:00:02",
        f"$$Data Sample={dt}",
        "$$Data Type=Little,Int16",
        "$$Data Order=" + ",".join(channels),
        f"$$Data Counts={counts}",
    ]
    while True:
        header = "\r\n".join(header_lines) + "\r\n"
        size = len(header.encode("ascii"))
        new_line = f"HeaderSiz={size}"
        if header_lines[0] == new_line:
            break
        header_lines[0] = new_line
    data = np.arange(counts * len(channels), dtype=np.int16)
    body = data.tobytes()
    path = tmp_path / "sample.gbd"
    path.write_bytes(header.encode("ascii") + body)
    return path, start_str, dt, channels, data.reshape(counts, len(channels))


def _write_gbd_with_amp_ranges(tmp_path, counts=4):
    channels = ["CH1", "CH2"]
    dt = 0.5
    start_str = "2024/01/01 00:00:00"
    header_lines = [
        "HeaderSiz=0",
        f"$$Time Start={start_str}",
        "$$Time Stop=2024/01/01 00:00:02",
        f"$$Data Sample={dt}",
        "$$Data Type=Little,Int16",
        "$$Data Order=" + ",".join(channels),
        f"$$Data Counts={counts}",
        "$$Amp1",
        "CH1 = 0, 0, 10 V",
        "CH2 = 1, 0, -5 mV",
        "$$AmpEnd",
    ]
    while True:
        header = "\r\n".join(header_lines) + "\r\n"
        size = len(header.encode("ascii"))
        new_line = f"HeaderSiz={size}"
        if header_lines[0] == new_line:
            break
        header_lines[0] = new_line
    data = np.arange(counts * len(channels), dtype=np.int16)
    body = data.tobytes()
    path = tmp_path / "amp_sample.gbd"
    path.write_bytes(header.encode("ascii") + body)
    reshape_data = data.reshape(counts, len(channels))
    scales = {
        "CH1": 10.0 / GBD_FULL_SCALE,
        "CH2": 0.005 / GBD_FULL_SCALE,
    }
    return path, reshape_data, channels, scales


def test_gbd_reader_requires_timezone(tmp_path):
    path, _, _, _, _ = _write_minimal_gbd(tmp_path)
    with pytest.raises(ValueError):
        TimeSeriesDict.read(path, format="gbd")


def test_gbd_reader_parses_minimal_file(tmp_path):
    path, start_str, dt, channels, data = _write_minimal_gbd(tmp_path)
    tsd = TimeSeriesDict.read(path, format="gbd", timezone="UTC")
    assert set(tsd.keys()) == set(channels)
    ts = tsd[channels[0]]
    assert len(ts) == data.shape[0]
    assert np.isclose(ts.dt.value, dt)
    expected_t0 = float(
        to_gps(
            datetime.strptime(start_str, "%Y/%m/%d %H:%M:%S").replace(
                tzinfo=dt_timezone.utc
            )
        )
    )
    assert np.isclose(ts.t0.value, expected_t0)
    assert ts.unit.to_string() == "V"
    np.testing.assert_array_equal(tsd[channels[1]].value, data[:, 1])


def test_gbd_reader_applies_amp_range_scaling(tmp_path):
    path, data, channels, scales = _write_gbd_with_amp_ranges(tmp_path)
    tsd = TimeSeriesDict.read(path, format="gbd", timezone="UTC")
    for idx, channel in enumerate(channels):
        expected = data[:, idx] * scales[channel]
        np.testing.assert_allclose(tsd[channel].value, expected, rtol=0, atol=0)


def test_gbd_alarm_channels_are_dimensionless():
    path = Path("gwexpy/gui/test-data/graphtec.GBD")
    tsd = TimeSeriesDict.read(path, format="gbd", timezone="UTC")
    for channel in ("Alarm", "AlarmOut"):
        ts = tsd[channel]
        assert ts.unit.is_equivalent(u.dimensionless_unscaled)
        assert np.all(np.isin(ts.value, [0.0, 1.0]))


def test_dttxml_products_requires_argument(tmp_path):
    dummy = tmp_path / "dummy.xml"
    dummy.write_text("<dttxml></dttxml>")
    with pytest.raises(ValueError):
        TimeSeriesDict.read(dummy, format="dttxml")


def test_miniseed_pad_behavior(tmp_path):
    obspy = pytest.importorskip("obspy")

    start = obspy.UTCDateTime(0)
    tr1 = obspy.Trace(data=np.ones(5))
    tr1.stats.delta = 1.0
    tr1.stats.starttime = start
    tr1.stats.station = "XX"
    tr1.stats.channel = "BHZ"

    tr2 = obspy.Trace(data=np.ones(5) * 2.0)
    tr2.stats.delta = 1.0
    tr2.stats.starttime = start + 7
    tr2.stats.station = "XX"
    tr2.stats.channel = "BHZ"

    stream = obspy.Stream([tr1, tr2])
    path = tmp_path / "test.mseed"
    stream.write(path, format="MSEED")

    tsd = TimeSeriesDict.read(path, format="miniseed", pad=np.nan)
    ts = tsd[next(iter(tsd))]
    assert len(ts) == 12
    np.testing.assert_array_equal(ts.value[:5], np.ones(5))
    assert np.isnan(ts.value[5])
    assert np.isnan(ts.value[6])
    np.testing.assert_array_equal(ts.value[7:], np.ones(5) * 2.0)
