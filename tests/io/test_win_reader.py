from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeriesDict
from gwexpy.timeseries.io.win import _apply_4bit_deltas

_SAMPLE_WIN = Path("gwexpy/gui/test-data/pmon_win_03.110909.010535.win")


def test_win_reader_matches_obspy_for_sample():
    obspy = pytest.importorskip("obspy")
    if not _SAMPLE_WIN.exists():
        pytest.skip("sample WIN file is missing")

    tsd = TimeSeriesDict.read(_SAMPLE_WIN, format="win")
    st = obspy.read(str(_SAMPLE_WIN))

    assert len(tsd) == len(st)
    obspy_by_id = {tr.id: tr for tr in st}

    # gwexpy keys are ObsPy Trace.id strings
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
