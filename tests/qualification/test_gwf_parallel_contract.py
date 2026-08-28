"""Strict real-file qualification for the public GWF ``parallel=`` contract."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from gwpy.timeseries import StateVector, StateVectorDict

from gwexpy.timeseries import TimeSeries, TimeSeriesDict

pytestmark = pytest.mark.skipif(
    os.environ.get("GWEXPY_POST_RELEASE_QUALIFICATION") != "1",
    reason="post-release qualification is opt-in",
)

TIME_CHANNEL = "X1:QUALIFICATION-TIMESERIES"
STATE_CHANNEL = "X1:QUALIFICATION-STATE_VECTOR"
STATE_BITS = ["ready", "active", "valid"]


def _write_real_lal_frames(tmp_path: Path) -> tuple[list[Path], dict[str, np.ndarray]]:
    """Write two contiguous frames through the provisioned LAL backend."""
    import lal
    import lalframe

    assert lal.__name__ == "lal"
    assert lalframe.__name__ == "lalframe"
    expected = {
        TIME_CHANNEL: np.arange(16, dtype=np.float64),
        STATE_CHANNEL: np.arange(16, dtype=np.uint32) % 8,
    }
    sources: list[Path] = []
    for index, start in enumerate((1000, 1001)):
        item = slice(index * 8, (index + 1) * 8)
        payload = TimeSeriesDict(
            {
                TIME_CHANNEL: TimeSeries(
                    expected[TIME_CHANNEL][item],
                    sample_rate=8 * u.Hz,
                    t0=start,
                    unit="V",
                    name=TIME_CHANNEL,
                    channel=TIME_CHANNEL,
                ),
                STATE_CHANNEL: TimeSeries(
                    expected[STATE_CHANNEL][item],
                    sample_rate=8 * u.Hz,
                    t0=start,
                    unit="count",
                    name=STATE_CHANNEL,
                    channel=STATE_CHANNEL,
                ),
            }
        )
        source = tmp_path / f"X1-QUALIFICATION-{start}-1.gwf"
        payload.write(str(source), format="gwf", backend="lalframe")
        assert source.is_file() and source.stat().st_size > 0
        sources.append(source)
    return sources, expected


def _parent_backend_must_not_read(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("parallel read executed the parent-process backend")


def test_real_lal_frames_use_spawn_children_for_all_four_public_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gwpy.timeseries.io.gwf import core as timeseries_gwf_core

    sources, expected = _write_real_lal_frames(tmp_path)
    monkeypatch.setattr(
        timeseries_gwf_core, "read_timeseriesdict", _parent_backend_must_not_read
    )
    monkeypatch.setattr(
        timeseries_gwf_core, "read_statevectordict", _parent_backend_must_not_read
    )

    timeseries = TimeSeries.read(
        sources, TIME_CHANNEL, format="gwf", backend="lalframe", parallel=2
    )
    timeseries_dict = TimeSeriesDict.read(
        sources, [TIME_CHANNEL], format="gwf", backend="lalframe", parallel=2
    )
    statevector = StateVector.read(
        sources,
        STATE_CHANNEL,
        format="gwf",
        backend="lalframe",
        parallel=2,
        bits=STATE_BITS,
    )
    statevector_dict = StateVectorDict.read(
        sources,
        [STATE_CHANNEL],
        format="gwf",
        backend="lalframe",
        parallel=2,
        bits=STATE_BITS,
    )

    assert type(timeseries) is TimeSeries
    assert type(timeseries_dict) is TimeSeriesDict
    assert type(timeseries_dict[TIME_CHANNEL]) is TimeSeries
    assert type(statevector) is StateVector
    assert type(statevector_dict) is StateVectorDict
    assert type(statevector_dict[STATE_CHANNEL]) is StateVector
    for series in (timeseries, timeseries_dict[TIME_CHANNEL]):
        np.testing.assert_array_equal(series.value, expected[TIME_CHANNEL])
        assert series.unit == u.V
        assert series.t0 == 1000 * u.s
        assert series.dt == 0.125 * u.s
        assert series.name == TIME_CHANNEL
        assert series.channel is None or series.channel.name == TIME_CHANNEL
    for series in (statevector, statevector_dict[STATE_CHANNEL]):
        np.testing.assert_array_equal(series.value, expected[STATE_CHANNEL])
        assert series.unit == u.count
        assert series.t0 == 1000 * u.s
        assert series.dt == 0.125 * u.s
        assert series.name == STATE_CHANNEL
        assert series.channel is None or series.channel.name == STATE_CHANNEL
        assert list(series.bits) == STATE_BITS


@pytest.mark.parametrize(
    ("reader", "selector"),
    [
        (TimeSeries.read, TIME_CHANNEL),
        (TimeSeriesDict.read, [TIME_CHANNEL]),
        (StateVector.read, STATE_CHANNEL),
        (StateVectorDict.read, [STATE_CHANNEL]),
    ],
)
def test_url_and_local_source_mix_fails_closed_before_any_backend(
    reader: Callable[..., Any],
    selector: str | list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gwpy.io.gwf import core as gwf_core
    from gwpy.timeseries.io.gwf import core as timeseries_gwf_core

    monkeypatch.setattr(gwf_core, "get_channel_names", _parent_backend_must_not_read)
    monkeypatch.setattr(
        timeseries_gwf_core, "read_timeseriesdict", _parent_backend_must_not_read
    )
    monkeypatch.setattr(
        timeseries_gwf_core, "read_statevectordict", _parent_backend_must_not_read
    )
    monkeypatch.setattr(
        StateVector.read.registry, "read", _parent_backend_must_not_read
    )

    with pytest.raises(TypeError, match="local GWF frame paths"):
        reader(
            [
                "https://frames.example.test/X1-QUALIFICATION-1000-1.gwf",
                Path("X1-QUALIFICATION-1001-1.gwf"),
            ],
            selector,
            format="gwf",
            backend="lalframe",
            parallel=2,
        )
