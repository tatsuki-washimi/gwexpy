import datetime
import struct
from pathlib import Path

import numpy as np
import pytest
from gwpy.time import to_gps

from gwexpy.timeseries import TimeSeries

_SAMPLE_ATS = Path("gwexpy/gui/test-data/134_V01_C02_R000_THx_BL_128H.ats")


def _read_ats_header_fields(path: Path) -> tuple[int, int, int, float, int, float, int]:
    """
    Read minimal ATS header fields using Metronix offsets (little endian).

    Returns
    -------
    header_length, header_vers, ui_samples, sample_freq, start_unix, lsb_mV, bit_indicator
    """
    raw = path.read_bytes()[:1024]
    header_length = struct.unpack_from("<H", raw, 0x00)[0]
    header_vers = struct.unpack_from("<h", raw, 0x02)[0]
    ui_samples = struct.unpack_from("<I", raw, 0x04)[0]
    sample_freq = struct.unpack_from("<f", raw, 0x08)[0]
    start_unix = struct.unpack_from("<I", raw, 0x0C)[0]
    lsb_mV = struct.unpack_from("<d", raw, 0x10)[0]
    bit_indicator = struct.unpack_from("<h", raw, 0xAA)[0]
    return header_length, header_vers, ui_samples, sample_freq, start_unix, lsb_mV, bit_indicator


def test_ats_sample_metadata_and_scaling():
    if not _SAMPLE_ATS.exists():
        pytest.skip("sample ATS file is missing")

    header_length, header_vers, ui_samples, sample_freq, start_unix, lsb_mV, bit_indicator = (
        _read_ats_header_fields(_SAMPLE_ATS)
    )
    assert header_vers == 80
    assert header_length == 1024
    assert sample_freq == pytest.approx(128.0)
    assert bit_indicator in (0, 1)

    ts = TimeSeries.read(_SAMPLE_ATS, format="ats")
    assert len(ts) == ui_samples
    assert ts.sample_rate.value == pytest.approx(sample_freq)
    expected_t0 = float(
        to_gps(datetime.datetime.fromtimestamp(start_unix, tz=datetime.timezone.utc))
    )
    assert ts.t0.value == pytest.approx(expected_t0)

    # Check first sample scaling against on-disk int32 (version 80 is 32-bit little endian).
    raw0 = struct.unpack_from("<i", _SAMPLE_ATS.read_bytes(), header_length)[0]
    expected0 = float(raw0) * float(lsb_mV) / 1000.0
    assert ts.value[0] == pytest.approx(expected0)


def test_ats_rejects_cea_sliced_header(tmp_path):
    # Minimal header that declares version 1080; reader should refuse to silently mis-parse.
    header = bytearray(1024)
    struct.pack_into("<H", header, 0x00, 1024)  # uiHeaderLength
    struct.pack_into("<h", header, 0x02, 1080)  # siHeaderVers
    struct.pack_into("<I", header, 0x04, 1)  # uiSamples
    struct.pack_into("<f", header, 0x08, 1.0)  # rSampleFreq
    struct.pack_into("<I", header, 0x0C, 0)  # uiStartDateTime
    struct.pack_into("<d", header, 0x10, 1.0)  # dblLSBMV
    # 1 dummy sample to make file non-empty
    payload = struct.pack("<i", 1)
    path = tmp_path / "sliced.ats"
    path.write_bytes(header + payload)

    with pytest.raises(NotImplementedError):
        TimeSeries.read(path, format="ats")


def test_ats_reads_int64_when_bit_indicator_is_set(tmp_path):
    header = bytearray(1024)
    struct.pack_into("<H", header, 0x00, 1024)  # uiHeaderLength
    struct.pack_into("<h", header, 0x02, 81)  # siHeaderVers
    struct.pack_into("<I", header, 0x04, 2)  # uiSamples
    struct.pack_into("<f", header, 0x08, 2.0)  # rSampleFreq
    struct.pack_into("<I", header, 0x0C, 1)  # uiStartDateTime
    struct.pack_into("<d", header, 0x10, 1000.0)  # dblLSBMV (mV/count)
    struct.pack_into("<h", header, 0xAA, 1)  # bit_indicator -> int64 data
    # Two int64 samples, little endian
    data = np.array([1, -2], dtype="<i8").tobytes()
    path = tmp_path / "int64.ats"
    path.write_bytes(header + data)

    ts = TimeSeries.read(path, format="ats")
    # dblLSBMV=1000 mV/count => 1 count => 1 V after /1000
    np.testing.assert_allclose(ts.value[:2], np.array([1.0, -2.0], dtype=float))

