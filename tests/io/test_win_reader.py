from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeriesDict


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

