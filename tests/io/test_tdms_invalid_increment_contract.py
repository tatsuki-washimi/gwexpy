"""Public TDMS readers reject waveform timing without a usable increment."""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

nptdms = pytest.importorskip("nptdms")


_ABSENT = object()
_INVALID_INCREMENTS = (
    pytest.param(_ABSENT, id="absent"),
    pytest.param(True, id="boolean-true"),
    pytest.param(0.0, id="zero"),
    pytest.param(-0.1, id="negative-finite"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
)
_PUBLIC_READERS = (
    pytest.param(TimeSeries, id="timeseries"),
    pytest.param(TimeSeriesDict, id="timeseriesdict"),
    pytest.param(TimeSeriesMatrix, id="timeseriesmatrix"),
)


def _write_tdms(path, increment):
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    properties = {"unit_string": "V"}
    if increment is not _ABSENT:
        properties["wf_increment"] = increment
    channel = ChannelObject(
        "Group",
        "Signal",
        np.array([11, 13, 17], dtype=np.int16),
        properties=properties,
    )
    with TdmsWriter(str(path)) as writer:
        writer.write_segment([RootObject(), GroupObject("Group"), channel])


@pytest.mark.parametrize("increment", _INVALID_INCREMENTS)
@pytest.mark.parametrize("reader", _PUBLIC_READERS)
@pytest.mark.parametrize(
    "epoch", [None, 1234567890.0], ids=["source", "epoch-override"]
)
def test_public_tdms_readers_reject_missing_or_invalid_waveform_increment(
    tmp_path, increment, reader, epoch
):
    path = tmp_path / "invalid-increment.tdms"
    _write_tdms(path, increment)
    kwargs = {"format": "tdms"}
    if epoch is not None:
        kwargs["epoch"] = epoch

    with pytest.raises(ValueError, match="wf_increment"):
        reader.read(path, **kwargs)
