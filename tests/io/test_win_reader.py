from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeriesDict
from gwexpy.timeseries.io.win import _apply_4bit_deltas

_SAMPLE_WIN = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "dummy.win"


def _skip_if_obspy_sqlalchemy_incompatible() -> None:
    """Skip ObsPy interop tests when ObsPy's pinned deps conflict with the env."""
    import importlib.metadata as im

    try:
        im.version("obspy")
    except im.PackageNotFoundError:
        return

    reqs = im.requires("obspy") or []
    needs_sqlalchemy_lt2 = any(
        r.lower().startswith("sqlalchemy") and "<2" in r for r in reqs
    )
    if not needs_sqlalchemy_lt2:
        return

    try:
        sqlalchemy_ver = im.version("SQLAlchemy")
    except im.PackageNotFoundError:
        return

    major = int(sqlalchemy_ver.split(".", 1)[0])
    if major >= 2:
        pytest.skip(
            "ObsPy requires SQLAlchemy<2 but SQLAlchemy>=2 is installed; skipping "
            "ObsPy-dependent WIN reader tests.",
            allow_module_level=True,
        )


def test_win_reader_matches_obspy_for_sample(monkeypatch):
    _skip_if_obspy_sqlalchemy_incompatible()
    obspy = pytest.importorskip("obspy")
    from unittest.mock import MagicMock, patch

    # Create a mock ObsPy Stream
    tr = obspy.Trace(data=np.array([100, 101, 102], dtype=np.int32))
    tr.stats.channel = "A1_01"
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = obspy.UTCDateTime(2023, 1, 1, 0, 0, 0)
    mock_stream = obspy.Stream(traces=[tr])

    # Provide a dummy path (doesn't need to exist on disk for the mock test)
    dummy_win = Path("tests/fixtures/data/dummy.win")

    # Mock both the internal _read_win_fixed and obspy.read
    with (
        patch("gwexpy.timeseries.io.win._read_win_fixed", return_value=mock_stream),
        patch("obspy.read", return_value=mock_stream),
    ):
        tsd = TimeSeriesDict.read(dummy_win, format="win", timezone="UTC")
        st = obspy.read(str(dummy_win))

        assert len(tsd) == len(st)
        obspy_by_id = {tr.id: tr for tr in st}

        for key, ts in tsd.items():
            assert key in obspy_by_id
            tr = obspy_by_id[key]
            assert len(ts) == tr.stats.npts
            assert ts.sample_rate.value == pytest.approx(tr.stats.sampling_rate)
            np.testing.assert_array_equal(ts.value, tr.data)


def test_win_apply_4bit_deltas_handles_odd_and_even_counts():
    # Deltas are signed 4-bit in [-8, 7]. Use small positive nibbles for clarity.
    # Byte 0x12 -> +1, +2 ; 0x34 -> +3, +4
    sdata = bytes([0x12, 0x34])

    out = [100]
    _apply_4bit_deltas(out, sdata, n_deltas=4)
    assert out == [100, 101, 103, 106, 110]

    out = [100]
    _apply_4bit_deltas(out, sdata, n_deltas=3)
    assert out == [100, 101, 103, 106]


def _make_win_packet(
    year_2d: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    channel_id: int,
    rate: int,
    width_code: int,
    absolute: int,
    deltas: tuple[int, ...],
) -> bytes:
    import struct

    def _bcd(v: int) -> int:
        return int(f"{v:02d}", 16)

    header = bytes(
        [
            (channel_id >> 8) & 0xFF,
            channel_id & 0xFF,
            (width_code << 4) | ((rate >> 8) & 0x0F),
            rate & 0xFF,
        ]
    )
    if width_code == 0:
        nibbles = [d & 0x0F for d in deltas]
        if len(nibbles) % 2:
            nibbles.append(0)
        payload = bytes(
            (nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles), 2)
        )
    else:
        payload = b"".join(d.to_bytes(width_code, "big", signed=True) for d in deltas)
    body = (
        bytes(
            [
                _bcd(year_2d),
                _bcd(month),
                _bcd(day),
                _bcd(hour),
                _bcd(minute),
                _bcd(second),
            ]
        )
        + header
        + struct.pack(">i", absolute)
        + payload
    )
    return struct.pack(">i", len(body) + 4) + body


@pytest.mark.parametrize("bad_tz", [None, "", "   "])
def test_win_missing_or_empty_timezone_fails_closed(tmp_path, bad_tz):
    _skip_if_obspy_sqlalchemy_incompatible()
    pytest.importorskip("obspy")
    from gwexpy.timeseries.io.win import _read_win_fixed, read_win_file

    path = tmp_path / "tz_missing.win"
    path.write_bytes(_make_win_packet(24, 1, 15, 9, 0, 0, 0x1201, 2, 1, 10, (1,)))

    with pytest.raises(ValueError, match="timezone is required for WIN files"):
        read_win_file(path, timezone=bad_tz)
    with pytest.raises(ValueError, match="timezone is required for WIN files"):
        _read_win_fixed(path, timezone=bad_tz)
    with pytest.raises(ValueError, match="timezone is required for WIN files"):
        TimeSeriesDict.read(path, format="win", timezone=bad_tz)


def test_win_timezone_utc_vs_asia_tokyo_differs_by_nine_hours(tmp_path):
    _skip_if_obspy_sqlalchemy_incompatible()
    pytest.importorskip("obspy")

    path = tmp_path / "tz_shift.win"
    path.write_bytes(_make_win_packet(24, 1, 15, 9, 0, 0, 0x1201, 2, 1, 10, (1,)))

    tsd_utc = TimeSeriesDict.read(path, format="win", timezone="UTC")
    tsd_jst = TimeSeriesDict.read(path, format="win", timezone="Asia/Tokyo")
    tsd_off = TimeSeriesDict.read(path, format="win", timezone="+09:00")

    t0_utc = float(tsd_utc["...1201"].t0.value)
    t0_jst = float(tsd_jst["...1201"].t0.value)
    t0_off = float(tsd_off["...1201"].t0.value)

    assert t0_utc - t0_jst == pytest.approx(9 * 3600.0)
    assert t0_jst == pytest.approx(t0_off)
    assert tsd_jst._gwexpy_io["timezone"] == "Asia/Tokyo"
    assert tsd_jst._gwexpy_io["epoch_source"] == "win_header_timezone"


def test_win_multi_file_read_forwards_timezone_and_provenance(tmp_path):
    _skip_if_obspy_sqlalchemy_incompatible()
    pytest.importorskip("obspy")

    p1 = tmp_path / "part1.win"
    p2 = tmp_path / "part2.win"
    p1.write_bytes(_make_win_packet(24, 1, 15, 9, 0, 0, 0x1201, 2, 1, 10, (1,)))
    p2.write_bytes(_make_win_packet(24, 1, 15, 9, 0, 2, 0x1201, 2, 1, 20, (2,)))

    tsd = TimeSeriesDict.read([p1, p2], format="win", timezone="Asia/Tokyo", pad=-1.0)
    ts = tsd["...1201"]
    assert len(ts) == 6
    np.testing.assert_array_equal(ts.value, [10.0, 11.0, -1.0, -1.0, 20.0, 22.0])
    assert tsd._gwexpy_io["timezone"] == "Asia/Tokyo"
    assert tsd._gwexpy_io["epoch_source"] == "win_header_timezone"
    assert tsd._gwexpy_io["n_sources"] == 2


def test_win_4bit_and_24bit_decoding_regression_with_timezone(tmp_path):
    _skip_if_obspy_sqlalchemy_incompatible()
    pytest.importorskip("obspy")

    p4 = tmp_path / "four_bit.win"
    p4.write_bytes(_make_win_packet(24, 1, 15, 9, 0, 0, 0x1201, 4, 0, 100, (7, -8, 1)))
    tsd4 = TimeSeriesDict.read(p4, format="win", timezone="+09:00")
    np.testing.assert_array_equal(tsd4["...1201"].value, [100, 107, 99, 100])

    p24 = tmp_path / "twentyfour_bit.win"
    p24.write_bytes(
        _make_win_packet(24, 1, 15, 9, 0, 0, 0x1202, 3, 3, 50, (-8_388_608, 8_388_607))
    )
    tsd24 = TimeSeriesDict.read(p24, format="win", timezone="+09:00")
    np.testing.assert_array_equal(tsd24["...1202"].value, [50, 50 - 8_388_608, 50 - 1])
