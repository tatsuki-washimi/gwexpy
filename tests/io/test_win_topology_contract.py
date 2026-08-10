from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import gwexpy.timeseries.io.win as win_io
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _bcd(value: int) -> int:
    """Encode a two-digit decimal value in the WIN header representation."""
    return int(f"{value:02d}", 16)


def _channel_block(
    channel: int,
    *,
    rate: int = 2,
    absolute: int = 100,
    width_code: int = 1,
    deltas: tuple[int, ...] | None = None,
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
    deltas = (1,) * (rate - 1) if deltas is None else deltas
    if len(deltas) != rate - 1:
        raise AssertionError("test deltas must match the declared sample rate")
    if width_code == 0:
        nibbles = [delta & 0x0F for delta in deltas]
        if len(nibbles) % 2:
            nibbles.append(0)
        payload = bytes(
            (nibbles[index] << 4) | nibbles[index + 1]
            for index in range(0, len(nibbles), 2)
        )
    elif width_code in (1, 2, 3, 4):
        payload = b"".join(
            delta.to_bytes(width_code, "big", signed=True) for delta in deltas
        )
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
        datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC),
    ],
)
def test_win_accepts_global_sequence_across_calendar_boundaries(tmp_path, start):
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


def test_win_advances_century_only_at_bcd_99_to_00_rollover(tmp_path):
    pytest.importorskip("obspy")
    start = datetime(1999, 12, 31, 23, 59, 59, tzinfo=UTC)
    rollover = datetime(2000, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "century-rollover.win",
        _packet(start, _channel_block(1, absolute=10)),
        _packet(rollover, _channel_block(2, absolute=20)),
    )

    traces = {
        trace.stats.channel: trace
        for trace in win_io._read_win_fixed(path, century="19")
    }

    assert traces["1201"].stats.starttime == win_io.UTCDateTime(start)
    assert traces["1202"].stats.starttime == win_io.UTCDateTime(rollover)


def test_win_keeps_supplied_century_before_bcd_year_99(tmp_path):
    pytest.importorskip("obspy")
    start = datetime(1998, 12, 31, 23, 59, 59, tzinfo=UTC)
    next_year = datetime(1999, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "same-century.win",
        _packet(start, _channel_block(1, absolute=10)),
        _packet(next_year, _channel_block(2, absolute=20)),
    )

    traces = {
        trace.stats.channel: trace
        for trace in win_io._read_win_fixed(path, century="19")
    }

    assert traces["1201"].stats.starttime == win_io.UTCDateTime(start)
    assert traces["1202"].stats.starttime == win_io.UTCDateTime(next_year)


def test_win_validates_packet_cadence_with_exact_integer_timestamps(
    tmp_path, monkeypatch
):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "exact-cadence.win",
        _packet(origin, _channel_block(1)),
        _packet(origin + timedelta(seconds=2), _channel_block(1)),
    )

    class ExactTimestamp:
        def __init__(self, year, month, day, hour, minute, second):
            value = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
            epoch_delta = value - datetime(1970, 1, 1, tzinfo=UTC)
            self.ns = (epoch_delta.days * 86_400 + epoch_delta.seconds) * 1_000_000_000

        def __sub__(self, other):
            raise AssertionError("packet cadence must not use floating subtraction")

        def __str__(self):
            return f"ExactTimestamp(ns={self.ns})"

    monkeypatch.setattr(win_io, "UTCDateTime", ExactTimestamp)

    with pytest.raises(ValueError, match="gap"):
        win_io._read_win_fixed(path)


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


@pytest.mark.parametrize(
    ("width_code", "deltas"),
    [
        (0, (-8, 7)),
        (1, (-128, 127)),
        (2, (-32_768, 32_767)),
        (3, (-8_388_608, 8_388_607)),
        (4, (-2_147_483_648, 2_147_483_647)),
    ],
)
def test_win_decodes_each_codec_at_signed_delta_bounds(tmp_path, width_code, deltas):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        f"width-{width_code}.win",
        _packet(
            origin,
            _channel_block(
                1,
                rate=3,
                absolute=0,
                width_code=width_code,
                deltas=deltas,
            ),
        ),
    )

    trace = win_io._read_win_fixed(path)[0]

    np.testing.assert_array_equal(trace.data, [0, deltas[0], sum(deltas)])
    assert trace.stats.sampling_rate == 3.0


@pytest.mark.parametrize("width_code", range(5))
def test_win_decodes_rate_one_for_each_codec_width(tmp_path, width_code):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        f"rate-one-width-{width_code}.win",
        _packet(
            origin,
            _channel_block(1, rate=1, absolute=-2_147_483_648, width_code=width_code),
        ),
    )

    trace = win_io._read_win_fixed(path)[0]

    np.testing.assert_array_equal(trace.data, [-2_147_483_648])
    assert trace.stats.sampling_rate == 1.0


def test_win_preserves_inferred_integer_dtype_for_normal_samples(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "normal-integer-dtype.win",
        _packet(origin, _channel_block(1, absolute=100, deltas=(1,))),
    )

    trace = win_io._read_win_fixed(path)[0]

    assert trace.data.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(trace.data, [100, 101])


def test_win_does_not_wrap_cumulative_samples_above_int32_max(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    int32_max = np.iinfo(np.int32).max
    path = _write_packets(
        tmp_path,
        "cumulative-int32-overflow.win",
        _packet(origin, _channel_block(1, absolute=int32_max, deltas=(1,))),
    )

    trace = win_io._read_win_fixed(path)[0]

    assert trace.data.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(trace.data, [int32_max, int32_max + 1])


def test_win_preserves_first_seen_channel_order_when_packet_order_changes(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "channel-order.win",
        _packet(
            origin,
            _channel_block(2, absolute=20),
            _channel_block(1, absolute=10),
        ),
        _packet(
            origin + timedelta(seconds=1),
            _channel_block(1, absolute=30),
            _channel_block(2, absolute=40),
        ),
    )

    stream = win_io._read_win_fixed(path)

    assert [trace.stats.channel for trace in stream] == ["1202", "1201"]
    np.testing.assert_array_equal(stream[0].data, [20, 21, 40, 41])
    np.testing.assert_array_equal(stream[1].data, [10, 11, 30, 31])
    assert [trace.stats.sampling_rate for trace in stream] == [2.0, 2.0]
    assert all(trace.stats.starttime == win_io.UTCDateTime(origin) for trace in stream)


@pytest.mark.parametrize("reader", [TimeSeriesDict, TimeSeries])
def test_win_public_readers_preserve_data_and_emit_one_utc_warning(tmp_path, reader):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "public-reader.win",
        _packet(origin, _channel_block(2, absolute=20)),
        _packet(
            origin + timedelta(seconds=1),
            _channel_block(1, absolute=30),
            _channel_block(2, absolute=40),
        ),
        _packet(origin + timedelta(seconds=2), _channel_block(1, absolute=10)),
    )

    with pytest.warns(UserWarning, match="#632") as caught:
        result = reader.read(path, format="win")

    assert len(caught) == 1
    if reader is TimeSeriesDict:
        assert list(result) == ["...1201", "...1202"]
        np.testing.assert_array_equal(result["...1202"].value, [20, 21, 40, 41])
        np.testing.assert_array_equal(result["...1201"].value, [30, 31, 10, 11])
        assert result["...1202"].sample_rate.value == 2.0
        assert result["...1201"].sample_rate.value == 2.0
        assert (result["...1201"].t0 - result["...1202"].t0).to_value("s") == 1.0
    else:
        np.testing.assert_array_equal(result.value, [30, 31, 10, 11])
        assert result.sample_rate.value == 2.0


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


def test_win_rejects_huge_declared_length_before_oversized_read(tmp_path, monkeypatch):
    pytest.importorskip("obspy")
    path = _write_packets(
        tmp_path,
        "huge-declared-length.win",
        struct.pack(">i", np.iinfo(np.int32).max) + b"\x00" * 6,
    )
    raw_file = path.open("rb")

    class GuardedFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raw_file.close()

        def read(self, size=-1):
            if size > path.stat().st_size:
                raise AssertionError("WIN reader requested an oversized allocation")
            return raw_file.read(size)

        def seek(self, *args):
            return raw_file.seek(*args)

        def tell(self):
            return raw_file.tell()

    monkeypatch.setattr(
        win_io, "open", lambda *args, **kwargs: GuardedFile(), raising=False
    )

    with pytest.raises(ValueError, match="truncated.*payload"):
        win_io._read_win_fixed(path)


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00\x00",
        struct.pack(">i", 9) + b"\x00" * 5,
    ],
)
def test_win_rejects_malformed_packet_length(tmp_path, payload):
    pytest.importorskip("obspy")
    path = _write_packets(tmp_path, "malformed-length.win", payload)

    with pytest.raises(ValueError, match="packet length"):
        win_io._read_win_fixed(path)


def test_win_rejects_zero_packet_length_even_with_trailing_record(tmp_path):
    pytest.importorskip("obspy")
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_packets(
        tmp_path,
        "zero-length-with-trailing-record.win",
        struct.pack(">i", 0),
        _packet(origin, _channel_block(1)),
    )

    with pytest.raises(ValueError, match="invalid WIN packet length: 0"):
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
