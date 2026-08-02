import struct

import numpy as np
import pytest

from gwexpy.timeseries.io.win import _read_win_fixed


def _golden_payload(rate: int, width_code: int, step: int) -> bytes:
    """Build payload bytes directly from the WIN delta encoding rules."""
    n_deltas = rate - 1
    if width_code == 0:
        packed = bytes([(step << 4) | step] * (n_deltas // 2))
        if n_deltas % 2:
            packed += bytes([step << 4])
        return packed
    if width_code == 1:
        deltas = (step if index % 2 == 0 else -step for index in range(n_deltas))
        return bytes(delta & 0xFF for delta in deltas)
    raise AssertionError(f"test fixture width code {width_code} is not defined")


def _golden_channel(
    channel: int,
    rate: int,
    width_code: int,
    absolute: int,
    step: int,
) -> bytes:
    """Build one WIN channel record from its documented wire fields."""
    header = bytes(
        [
            0x12,
            channel,
            (width_code << 4) | ((rate >> 8) & 0x0F),
            rate & 0xFF,
        ]
    )
    return (
        header + struct.pack(">i", absolute) + _golden_payload(rate, width_code, step)
    )


def _golden_packet(second: int, step: int) -> bytes:
    records = [
        _golden_channel(1, 255, 0, 1000 if second == 5 else 2000, step),
        _golden_channel(2, 256, 0, 2000 if second == 5 else 4000, step),
        _golden_channel(3, 1000, 1, 30 if second == 5 else 60, step),
        _golden_channel(4, 4095, 1, 40 if second == 5 else 80, step),
    ]
    body = bytes([0x23, 0x01, 0x02, 0x03, 0x04, second]) + b"".join(records)
    return struct.pack(">i", len(body) + 4) + body


def _expected_samples(rate: int, width_code: int, absolute: int, step: int):
    if width_code == 0:
        return absolute + step * np.arange(rate, dtype=np.int32)
    deltas = np.array(
        [step if index % 2 == 0 else -step for index in range(rate - 1)],
        dtype=np.int32,
    )
    return np.concatenate([[absolute], absolute + np.cumsum(deltas, dtype=np.int32)])


def test_win_decodes_full_12bit_rates_and_preserves_packet_alignment(tmp_path):
    pytest.importorskip("obspy")
    path = tmp_path / "rates-12bit.win"
    path.write_bytes(_golden_packet(5, 1) + _golden_packet(6, 2))

    stream = _read_win_fixed(path)
    traces = {trace.stats.channel: trace for trace in stream}

    assert set(traces) == {"1201", "1202", "1203", "1204"}
    for channel, rate, width_code, first, second in (
        ("1201", 255, 0, 1000, 2000),
        ("1202", 256, 0, 2000, 4000),
        ("1203", 1000, 1, 30, 60),
        ("1204", 4095, 1, 40, 80),
    ):
        trace = traces[channel]
        expected = np.concatenate(
            [
                _expected_samples(rate, width_code, first, 1),
                _expected_samples(rate, width_code, second, 2),
            ]
        )

        assert width_code in (0, 1)
        assert trace.stats.channel == channel
        assert trace.stats.sampling_rate == float(rate)
        assert trace.stats.npts == 2 * rate
        np.testing.assert_array_equal(trace.data, expected)


def test_win_zero_encoded_rate_fails_closed(tmp_path):
    pytest.importorskip("obspy")
    body = bytes([0x23, 0x01, 0x02, 0x03, 0x04, 0x05])
    body += bytes([0x12, 0x09, 0x00, 0x00])
    path = tmp_path / "zero-rate.win"
    path.write_bytes(struct.pack(">i", len(body) + 4) + body)

    with pytest.raises(ValueError, match="sample rate must be positive"):
        _read_win_fixed(path)
