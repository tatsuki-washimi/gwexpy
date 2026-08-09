from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import gwexpy.timeseries.io.win as win_io


def _bcd(value: int) -> int:
    """Encode a two-digit decimal value in the WIN header representation."""
    return int(f"{value:02d}", 16)


def _channel_block(
    channel: int,
    *,
    rate: int = 2,
    absolute: int = 100,
    width_code: int = 1,
) -> bytes:
    """Build a channel block directly from the documented WIN wire fields."""
    header = bytes(
        [
            0x12,
            channel,
            (width_code << 4) | ((rate >> 8) & 0x0F),
            rate & 0xFF,
        ]
    )
    if width_code == 0:
        n_deltas = rate - 1
        payload = bytes([0x11] * (n_deltas // 2))
        if n_deltas % 2:
            payload += b"\x10"
    elif width_code == 1:
        payload = bytes([1] * (rate - 1))
    else:
        raise AssertionError(f"unsupported test width code: {width_code}")
    return header + struct.pack(">i", absolute) + payload


def _packet(when: datetime, *blocks: bytes, trailing: bytes = b"") -> bytes:
    body = (
        bytes(
            [
                _bcd(when.year % 100),
                _bcd(when.month),
                _bcd(when.day),
                _bcd(when.hour),
                _bcd(when.minute),
                _bcd(when.second),
            ]
        )
        + b"".join(blocks)
        + trailing
    )
    return struct.pack(">i", len(body) + 4) + body


def _write_packets(tmp_path, name: str, *packets: bytes):
    path = tmp_path / name
    path.write_bytes(b"".join(packets))
    return path


@pytest.mark.parametrize(
    ("seconds", "reason"),
    [
        ((0, 0), "duplicate"),
        ((1, 0), "backward"),
        ((0, 2), "gap"),
    ],
)
def test_win_rejects_nonconsecutive_global_packet_timestamps(tmp_path, seconds, reason):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        f"global-{reason}.win",
        *(
            _packet(origin + timedelta(seconds=second), _channel_block(1))
            for second in seconds
        ),
    )

    with pytest.raises(ValueError, match=reason):
        win_io._read_win_fixed(path)


@pytest.mark.parametrize(
    "start",
    [
        datetime(2026, 1, 1, 0, 0, 59, tzinfo=UTC),
        datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
    ],
)
def test_win_accepts_global_sequence_across_minute_and_day_boundaries(tmp_path, start):
    pytest.importorskip("obspy")
    path = _write_packets(
        tmp_path,
        "boundary.win",
        _packet(start, _channel_block(1, absolute=10)),
        _packet(start + timedelta(seconds=1), _channel_block(1, absolute=20)),
    )

    stream = win_io._read_win_fixed(path)

    assert len(stream) == 1
    np.testing.assert_array_equal(stream[0].data, [10, 11, 20, 21])


def test_win_allows_channel_late_start_and_early_end_and_sets_each_t0(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "partial-channels.win",
        _packet(origin, _channel_block(1, absolute=10)),
        _packet(
            origin + timedelta(seconds=1),
            _channel_block(1, absolute=20),
            _channel_block(2, absolute=30),
        ),
        _packet(origin + timedelta(seconds=2), _channel_block(2, absolute=40)),
    )

    traces = {trace.stats.channel: trace for trace in win_io._read_win_fixed(path)}

    np.testing.assert_array_equal(traces["1201"].data, [10, 11, 20, 21])
    np.testing.assert_array_equal(traces["1202"].data, [30, 31, 40, 41])
    assert traces["1201"].stats.starttime == win_io.UTCDateTime(origin)
    assert traces["1202"].stats.starttime == win_io.UTCDateTime(
        origin + timedelta(seconds=1)
    )


def test_win_rejects_channel_reappearance_after_an_internal_gap(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "channel-gap.win",
        _packet(origin, _channel_block(1)),
        _packet(origin + timedelta(seconds=1), _channel_block(2)),
        _packet(origin + timedelta(seconds=2), _channel_block(1)),
    )

    with pytest.raises(ValueError, match="channel 1201.*gap"):
        win_io._read_win_fixed(path)


def test_win_rejects_duplicate_channel_block_within_packet(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "duplicate-channel.win",
        _packet(origin, _channel_block(1), _channel_block(1)),
    )

    with pytest.raises(ValueError, match="channel 1201.*more than once"):
        win_io._read_win_fixed(path)


def test_win_rejects_channel_sample_rate_change(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "rate-change.win",
        _packet(origin, _channel_block(1, rate=2)),
        _packet(origin + timedelta(seconds=1), _channel_block(1, rate=3)),
    )

    with pytest.raises(ValueError, match="channel 1201.*sample rate"):
        win_io._read_win_fixed(path)


def test_win_rejects_decoded_sample_count_mismatch(tmp_path, monkeypatch):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "decoded-count.win",
        _packet(origin, _channel_block(1, rate=4, width_code=0)),
    )
    monkeypatch.setattr(win_io, "_apply_4bit_deltas", lambda *args: None)

    with pytest.raises(ValueError, match="decoded sample count"):
        win_io._read_win_fixed(path)


def test_win_rejects_truncated_packet_payload(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    packet = _packet(origin, _channel_block(1, rate=3))
    declared_size = struct.unpack(">i", packet[:4])[0]
    truncated = struct.pack(">i", declared_size + 1) + packet[4:]
    path = _write_packets(tmp_path, "truncated.win", truncated)

    with pytest.raises(ValueError, match="truncated.*payload"):
        win_io._read_win_fixed(path)


def test_win_rejects_overlong_packet_payload(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "overlong.win",
        _packet(origin, _channel_block(1, rate=3), trailing=b"\xff"),
    )

    with pytest.raises(ValueError, match="overlong.*payload"):
        win_io._read_win_fixed(path)
