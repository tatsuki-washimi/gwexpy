from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.timeseries._gwf_io import (
    _extract_gwf_read_args,
    _resolve_gwf_format,
    read_gwf_timeseriesdict,
)

FIXTURE_DATA = Path(__file__).parent.parent / "fixtures" / "data" / "test.gwf"
CHANNEL = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"
AUX_CHANNEL = "K1:CAL-CS_PROC_DARM_CONTROL_DQ"


def has_gwf_backend(backend: str | None = None) -> bool:
    try:
        from gwpy.io.gwf.core import get_channel_names

        kwargs = {"backend": backend} if backend is not None else {}
        return bool(get_channel_names(FIXTURE_DATA, **kwargs))
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return False


def write_gwf_parts(
    tmp_path: Path,
    *,
    starts: tuple[float, float] = (1000.0, 1001.0),
) -> tuple[list[Path], np.ndarray]:
    """Create two contiguous real GWF files for multi-file read tests."""
    rate_hz = 16.0
    samples_per_file = 16
    files = []
    expected = []

    for index, start in enumerate(starts):
        values = np.arange(
            index * samples_per_file,
            (index + 1) * samples_per_file,
            dtype=float,
        )
        ts = TimeSeries(
            values,
            sample_rate=rate_hz,
            t0=start,
            channel=CHANNEL,
            name=CHANNEL,
        )
        path = tmp_path / f"part{index}.gwf"
        TimeSeriesDict({CHANNEL: ts}).write(path, format="gwf")
        files.append(path)
        expected.append(values)

    return files, np.concatenate(expected)


def write_gwf_dict_parts(
    tmp_path: Path,
    *,
    starts: tuple[float, float] = (1000.0, 1001.0),
) -> tuple[list[Path], dict[str, np.ndarray]]:
    """Create two-channel real GWF files for TimeSeriesDict read tests."""
    rate_hz = 16.0
    samples_per_file = 16
    files = []
    expected = {CHANNEL: [], AUX_CHANNEL: []}

    for index, start in enumerate(starts):
        values = np.arange(
            index * samples_per_file,
            (index + 1) * samples_per_file,
            dtype=float,
        )
        tsd = TimeSeriesDict(
            {
                CHANNEL: TimeSeries(
                    values,
                    sample_rate=rate_hz,
                    t0=start,
                    channel=CHANNEL,
                    name=CHANNEL,
                ),
                AUX_CHANNEL: TimeSeries(
                    values + 100.0,
                    sample_rate=rate_hz,
                    t0=start,
                    channel=AUX_CHANNEL,
                    name=AUX_CHANNEL,
                ),
            }
        )
        path = tmp_path / f"dict_part{index}.gwf"
        tsd.write(path, format="gwf")
        files.append(path)
        expected[CHANNEL].append(values)
        expected[AUX_CHANNEL].append(values + 100.0)

    return files, {
        channel: np.concatenate(parts) for channel, parts in expected.items()
    }


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_single_channel_string():
    tsd = TimeSeriesDict.read(FIXTURE_DATA, CHANNEL, format="gwf")

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert tsd[CHANNEL].name == CHANNEL
    assert len(tsd[CHANNEL]) > 0


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_channels_kwarg():
    tsd = TimeSeriesDict.read(FIXTURE_DATA, channels=[CHANNEL], format="gwf")

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_autodetects_extension_and_channels():
    tsd = TimeSeriesDict.read(FIXTURE_DATA)

    assert isinstance(tsd, TimeSeriesDict)
    assert CHANNEL in tsd


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_list_source_autodetects_extension_and_channels():
    tsd = TimeSeriesDict.read([FIXTURE_DATA], CHANNEL)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert tsd[CHANNEL].name == CHANNEL
    assert len(tsd[CHANNEL]) > 0


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize(
    "read_kwargs",
    [
        {},
        {"format": "gwf"},
        {"format": "gwf", "parallel": 1},
        {"format": "gwf", "nproc": 1},
    ],
)
def test_read_gwf_timeseriesdict_list_source_with_format_variants(read_kwargs):
    tsd = TimeSeriesDict.read([FIXTURE_DATA], CHANNEL, **read_kwargs)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert tsd[CHANNEL].name == CHANNEL
    assert len(tsd[CHANNEL]) > 0


def test_read_gwf_timeseriesdict_empty_source_list_rejected():
    with pytest.raises(ValueError, match="must be non-empty"):
        TimeSeriesDict.read([], CHANNEL, format="gwf")


def test_read_gwf_timeseriesdict_filters_empty_parts(monkeypatch):
    from gwpy.timeseries.io.gwf import core as gwf_core

    series = TimeSeries(
        np.arange(4.0),
        sample_rate=4.0,
        t0=1000.0,
        channel=CHANNEL,
        name=CHANNEL,
    )

    def fake_read_timeseriesdict(source, *args, **kwargs):
        if source == "empty.gwf":
            return {}
        return {CHANNEL: series}

    monkeypatch.setattr(gwf_core, "read_timeseriesdict", fake_read_timeseriesdict)

    tsd = read_gwf_timeseriesdict(
        ["empty.gwf", "full.gwf"],
        [CHANNEL],
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )

    assert list(tsd) == [CHANNEL]
    np.testing.assert_allclose(tsd[CHANNEL].value, series.value)


def test_read_gwf_timeseriesdict_all_empty_parts_raise(monkeypatch):
    from gwpy.timeseries.io.gwf import core as gwf_core

    monkeypatch.setattr(gwf_core, "read_timeseriesdict", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="No data found in any provided GWF source"):
        read_gwf_timeseriesdict(
            ["empty0.gwf", "empty1.gwf"],
            [CHANNEL],
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize(
    "read_kwargs",
    [
        {},
        {"format": "gwf"},
        {"format": "gwf", "parallel": 1},
        {"format": "gwf", "parallel": 2},
        {"format": "gwf", "nproc": 1},
    ],
)
def test_read_gwf_timeseriesdict_multiple_files_with_format_variants(
    tmp_path, read_kwargs
):
    files, expected = write_gwf_parts(tmp_path)

    tsd = TimeSeriesDict.read(files, CHANNEL, **read_kwargs)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert tsd[CHANNEL].name == CHANNEL
    assert len(tsd[CHANNEL]) == len(expected)
    assert float(tsd[CHANNEL].t0.value) == pytest.approx(1000.0)
    assert float(tsd[CHANNEL].dt.value) == pytest.approx(1.0 / 16.0)
    np.testing.assert_allclose(tsd[CHANNEL].value, expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_multiple_files_preserves_two_channels(tmp_path):
    files, expected = write_gwf_dict_parts(tmp_path)

    tsd = TimeSeriesDict.read(files, [CHANNEL, AUX_CHANNEL])

    assert isinstance(tsd, TimeSeriesDict)
    assert set(tsd) == {CHANNEL, AUX_CHANNEL}
    for channel in (CHANNEL, AUX_CHANNEL):
        assert tsd[channel].name == channel
        assert len(tsd[channel]) == len(expected[channel])
        assert float(tsd[channel].t0.value) == pytest.approx(1000.0)
        assert float(tsd[channel].dt.value) == pytest.approx(1.0 / 16.0)
        np.testing.assert_allclose(tsd[channel].value, expected[channel])


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_gap_ignore_allows_overlapping_files(tmp_path):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1000.5))

    tsd = TimeSeriesDict.read(files, CHANNEL, gap="ignore")

    assert len(tsd[CHANNEL]) == 32
    assert float(tsd[CHANNEL].t0.value) == pytest.approx(1000.0)
    np.testing.assert_allclose(tsd[CHANNEL].value, np.arange(32.0))


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize("read_kwargs", [{}, {"pad": -1.0}])
def test_read_gwf_timeseriesdict_multiple_files_sorts_by_span(tmp_path, read_kwargs):
    files, expected = write_gwf_parts(tmp_path)

    tsd = TimeSeriesDict.read(files[::-1], CHANNEL, **read_kwargs)

    assert len(tsd[CHANNEL]) == len(expected)
    assert float(tsd[CHANNEL].t0.value) == pytest.approx(1000.0)
    np.testing.assert_allclose(tsd[CHANNEL].value, expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize("source_factory", [list, tuple])
def test_read_gwf_timeseriesdict_multiple_files_autodetects_channels(
    tmp_path, source_factory
):
    files, expected = write_gwf_parts(tmp_path)

    tsd = TimeSeriesDict.read(source_factory(files))

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert len(tsd[CHANNEL]) == len(expected)
    np.testing.assert_allclose(tsd[CHANNEL].value, expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseries_multiple_files_autodetects_channel(tmp_path):
    files, expected = write_gwf_parts(tmp_path)

    ts = TimeSeries.read(files)

    assert isinstance(ts, TimeSeries)
    assert ts.name == CHANNEL
    assert len(ts) == len(expected)
    np.testing.assert_allclose(ts.value, expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize("read_kwargs", [{}, {"pad": -1.0}])
def test_read_gwf_timeseries_multiple_files_sorts_by_span(tmp_path, read_kwargs):
    files, expected = write_gwf_parts(tmp_path)

    ts = TimeSeries.read(files[::-1], CHANNEL, **read_kwargs)

    assert len(ts) == len(expected)
    assert float(ts.t0.value) == pytest.approx(1000.0)
    np.testing.assert_allclose(ts.value, expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesmatrix_multiple_files_with_channel_selector(tmp_path):
    files, expected = write_gwf_parts(tmp_path)

    matrix = TimeSeriesMatrix.read(files, CHANNEL)

    assert isinstance(matrix, TimeSeriesMatrix)
    assert matrix.shape == (1, 1, len(expected))
    assert list(matrix.channel_names) == [CHANNEL]
    np.testing.assert_allclose(matrix.view(np.ndarray)[0, 0], expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize("read_kwargs", [{}, {"pad": -1.0}])
def test_read_gwf_timeseriesmatrix_multiple_files_sorts_by_span(tmp_path, read_kwargs):
    files, expected = write_gwf_parts(tmp_path)

    matrix = TimeSeriesMatrix.read(files[::-1], CHANNEL, **read_kwargs)

    assert matrix.shape == (1, 1, len(expected))
    np.testing.assert_allclose(matrix.view(np.ndarray)[0, 0], expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize(
    ("starts", "message"),
    [
        ((1000.0, 1001.5), "Cannot append discontiguous TimeSeries"),
        ((1000.0, 1000.5), "Cannot append overlapping TimeSeriess"),
    ],
)
def test_read_gwf_timeseriesdict_multiple_files_rejects_gaps_and_overlaps(
    tmp_path, starts, message
):
    files, _ = write_gwf_parts(tmp_path, starts=starts)

    with pytest.raises(ValueError, match=message):
        TimeSeriesDict.read(files, CHANNEL)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize(
    ("reader", "message"),
    [
        (TimeSeries.read, "Cannot append discontiguous TimeSeries"),
        (TimeSeriesMatrix.read, "Cannot append discontiguous TimeSeries"),
    ],
)
def test_read_gwf_multiple_files_rejects_gap_without_pad(tmp_path, reader, message):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    with pytest.raises(ValueError, match=message):
        reader(files, CHANNEL)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize(
    ("reader", "message"),
    [
        (TimeSeries.read, "Cannot append overlapping TimeSeriess"),
        (TimeSeriesMatrix.read, "Cannot append overlapping TimeSeriess"),
    ],
)
def test_read_gwf_multiple_files_rejects_overlap_without_pad(tmp_path, reader, message):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1000.5))

    with pytest.raises(ValueError, match=message):
        reader(files, CHANNEL)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize(
    ("pad", "expected_pad"),
    [
        (np.nan, np.nan),
        (-1.0, -1.0),
    ],
)
def test_read_gwf_timeseriesdict_multiple_files_pads_gaps(tmp_path, pad, expected_pad):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    tsd = TimeSeriesDict.read(files, CHANNEL, pad=pad)

    assert len(tsd[CHANNEL]) == 40
    gap_values = tsd[CHANNEL].value[16:24]
    if np.isnan(expected_pad):
        assert np.isnan(gap_values).all()
    else:
        np.testing.assert_allclose(gap_values, expected_pad)
    np.testing.assert_allclose(tsd[CHANNEL].value[:16], np.arange(16, dtype=float))
    np.testing.assert_allclose(
        tsd[CHANNEL].value[24:],
        np.arange(16, 32, dtype=float),
    )


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_multiple_files_gap_pad_defaults_to_zero(tmp_path):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    tsd = TimeSeriesDict.read(files, CHANNEL, gap="pad")

    assert len(tsd[CHANNEL]) == 40
    np.testing.assert_allclose(tsd[CHANNEL].value[16:24], 0.0)
    np.testing.assert_allclose(tsd[CHANNEL].value[:16], np.arange(16, dtype=float))
    np.testing.assert_allclose(tsd[CHANNEL].value[24:], np.arange(16, 32, dtype=float))


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_multiple_files_gap_ignore_concatenates(tmp_path):
    files, expected = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    tsd = TimeSeriesDict.read(files, CHANNEL, gap="ignore")

    assert len(tsd[CHANNEL]) == len(expected)
    assert float(tsd[CHANNEL].t0.value) == pytest.approx(1000.0)
    np.testing.assert_allclose(tsd[CHANNEL].value, expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_multiple_files_gap_raise_overrides_pad(tmp_path):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    with pytest.raises(ValueError, match="Cannot append discontiguous TimeSeries"):
        TimeSeriesDict.read(files, CHANNEL, gap="raise", pad=-1.0)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
@pytest.mark.parametrize("read_kwargs", [{"parallel": 1}, {"parallel": 2}, {"nproc": 1}])
def test_read_gwf_timeseriesdict_multiple_files_pad_accepts_parallel_kwargs(
    tmp_path, read_kwargs
):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    tsd = TimeSeriesDict.read(files, CHANNEL, pad=-7.0, **read_kwargs)

    assert len(tsd[CHANNEL]) == 40
    np.testing.assert_allclose(tsd[CHANNEL].value[16:24], -7.0)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_multiple_files_respects_inner_start_end(tmp_path):
    files, _ = write_gwf_parts(tmp_path)

    tsd = TimeSeriesDict.read(files, CHANNEL, start=1000.25, end=1001.75)

    assert len(tsd[CHANNEL]) == 24
    assert float(tsd[CHANNEL].t0.value) == pytest.approx(1000.25)
    np.testing.assert_allclose(tsd[CHANNEL].value, np.arange(4, 28, dtype=float))


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_multiple_files_pads_requested_outer_interval(
    tmp_path,
):
    files, _ = write_gwf_parts(tmp_path)

    tsd = TimeSeriesDict.read(
        files,
        CHANNEL,
        start=999.5,
        end=1002.5,
        pad=-5.0,
    )

    assert len(tsd[CHANNEL]) == 48
    assert float(tsd[CHANNEL].t0.value) == pytest.approx(999.5)
    np.testing.assert_allclose(tsd[CHANNEL].value[:8], -5.0)
    np.testing.assert_allclose(tsd[CHANNEL].value[8:40], np.arange(32, dtype=float))
    np.testing.assert_allclose(tsd[CHANNEL].value[40:], -5.0)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesdict_multiple_files_gap_raise_rejects_outer_interval(
    tmp_path,
):
    files, _ = write_gwf_parts(tmp_path)

    with pytest.raises(ValueError, match="does not cover requested interval"):
        TimeSeriesDict.read(
            files,
            CHANNEL,
            start=999.5,
            end=1002.5,
            gap="raise",
        )


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseries_multiple_files_pads_gaps(tmp_path):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    ts = TimeSeries.read(files, pad=-2.0)

    assert len(ts) == 40
    np.testing.assert_allclose(ts.value[16:24], -2.0)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseries_multiple_files_gap_ignore_concatenates(tmp_path):
    files, expected = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    ts = TimeSeries.read(files, CHANNEL, gap="ignore")

    assert len(ts) == len(expected)
    np.testing.assert_allclose(ts.value, expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseries_multiple_files_pads_requested_outer_interval(tmp_path):
    files, _ = write_gwf_parts(tmp_path)

    ts = TimeSeries.read(files, CHANNEL, start=999.5, end=1002.5, pad=-5.0)

    assert len(ts) == 48
    assert float(ts.t0.value) == pytest.approx(999.5)
    np.testing.assert_allclose(ts.value[:8], -5.0)
    np.testing.assert_allclose(ts.value[8:40], np.arange(32, dtype=float))
    np.testing.assert_allclose(ts.value[40:], -5.0)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesmatrix_multiple_files_pads_gaps(tmp_path):
    files, _ = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    matrix = TimeSeriesMatrix.read(files, CHANNEL, pad=-3.0)

    assert matrix.shape == (1, 1, 40)
    np.testing.assert_allclose(matrix.view(np.ndarray)[0, 0, 16:24], -3.0)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesmatrix_multiple_files_gap_ignore_concatenates(tmp_path):
    files, expected = write_gwf_parts(tmp_path, starts=(1000.0, 1001.5))

    matrix = TimeSeriesMatrix.read(files, CHANNEL, gap="ignore")

    assert matrix.shape == (1, 1, len(expected))
    np.testing.assert_allclose(matrix.view(np.ndarray)[0, 0], expected)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseriesmatrix_multiple_files_gap_raise_rejects_outer_interval(
    tmp_path,
):
    files, _ = write_gwf_parts(tmp_path)

    with pytest.raises(ValueError, match="does not cover requested interval"):
        TimeSeriesMatrix.read(
            files,
            CHANNEL,
            start=999.5,
            end=1002.5,
            gap="raise",
        )


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_read_gwf_timeseries_with_autodetects_extension_and_channel():
    ts = TimeSeries.read(FIXTURE_DATA, CHANNEL)

    assert isinstance(ts, TimeSeries)
    assert ts.name == CHANNEL
    assert len(ts) > 0


def test_gwf_format_resolution_prefers_explicit_format_over_extension():
    path = FIXTURE_DATA

    assert _resolve_gwf_format(path, "gwf") == "gwf"
    assert _resolve_gwf_format(path, "frame") == "gwf"
    assert _resolve_gwf_format(path, "frameCPP") == "gwf.framecpp"
    assert _resolve_gwf_format(path, "FrAmEcPp") == "gwf.framecpp"
    assert _resolve_gwf_format(path, "GWF.FRAMECPP") == "gwf.framecpp"
    assert _resolve_gwf_format(path, "GWF.framel") == "gwf.framel"
    assert _resolve_gwf_format(path, "lalframe") == "gwf.lalframe"
    assert _resolve_gwf_format(path, "hdf5") is None
    assert _resolve_gwf_format(path, None) == "gwf"


def test_resolve_gwf_format_supports_non_empty_ordered_iterables():
    path = FIXTURE_DATA

    assert _resolve_gwf_format([path], None) == "gwf"
    assert _resolve_gwf_format((path,), None) == "gwf"
    assert _resolve_gwf_format([], None) is None
    assert _resolve_gwf_format([path, FIXTURE_DATA], None) == "gwf"
    assert _resolve_gwf_format([path, "not_gwf.txt"], None) is None
    assert _resolve_gwf_format({str(path)}, None) is None


def test_extract_gwf_read_args_supports_name_fallback_and_channel_aliases():
    channels, start, end, kw = _extract_gwf_read_args(
        (CHANNEL, 10.0, 20.0),
        {
            "format": "gwf",
            "name": "ignored-name",
            "start": 100.0,
        },
    )
    assert channels == [CHANNEL]
    assert start == 10.0
    assert end == 20.0
    assert "format" not in kw
    assert "name" not in kw


def test_extract_gwf_read_args_rejects_empty_channel_inputs():
    with pytest.raises(ValueError, match="No channels selected for GWF read"):
        _extract_gwf_read_args(([],), {"format": "gwf"})

    with pytest.raises(ValueError, match="No channels selected for GWF read"):
        _extract_gwf_read_args(((),), {"format": "gwf"})


def test_extract_gwf_read_args_no_start_end_leak_with_channel_alias():
    """start/end must be absent from gwf_kwargs when overridden by positional args.

    When positional start/end coexist with a channel alias keyword, the keyword
    start/end values must still be removed from gwf_kwargs so callers that forward
    start= and end= explicitly don't receive 'multiple values' TypeError.
    """
    channels, start, end, kw = _extract_gwf_read_args(
        (CHANNEL, 10.0, 20.0),
        {
            "format": "gwf",
            "name": "ignored",
            "start": 100.0,
            "end": 200.0,
        },
    )
    assert channels == [CHANNEL]
    assert start == 10.0
    assert end == 20.0
    assert "start" not in kw, f"'start' should not remain in gwf_kwargs, got {kw}"
    assert "end" not in kw, f"'end' should not remain in gwf_kwargs, got {kw}"


def test_extract_gwf_read_args_rejects_positional_keyword_overlap():
    with pytest.raises(
        TypeError,
        match="Cannot specify both positional and keyword 'start' for GWF read",
    ):
        _extract_gwf_read_args(
            ("K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ", 10.0),
            {"format": "gwf", "start": 0.0},
        )

    with pytest.raises(
        TypeError, match="Cannot specify both positional and keyword 'end' for GWF read"
    ):
        _extract_gwf_read_args(
            ("K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ", 10.0, 20.0),
            {"format": "gwf", "end": 30.0},
        )


def test_read_gwf_timeseries_with_single_channel_by_format_gwf():
    from gwpy.io.gwf.core import get_channel_names

    if not has_gwf_backend("framel"):
        pytest.skip("framel gwf backend not available")

    try:
        expected = get_channel_names(FIXTURE_DATA, backend="frameCPP")
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        expected = get_channel_names(FIXTURE_DATA)
    ts = TimeSeries.read(FIXTURE_DATA, CHANNEL, format="gwf.framel")
    assert isinstance(ts, TimeSeries)
    assert ts.name == CHANNEL
    if expected:
        assert ts.name in expected


def test_safe_get_reader_missing_format_returns_none_without_warning():
    import warnings

    from gwexpy.timeseries._gwf_io import _safe_get_reader, _safe_get_writer

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _safe_get_reader("gwexpy-no-such-format", TimeSeries) is None
        assert _safe_get_writer("gwexpy-no-such-format", TimeSeries) is None


def test_safe_get_reader_unexpected_error_warns(monkeypatch):
    from gwexpy.timeseries import _gwf_io

    def boom(*args, **kwargs):
        raise RuntimeError("registry broken")

    monkeypatch.setattr(_gwf_io.io_registry, "get_reader", boom)
    monkeypatch.setattr(_gwf_io.io_registry, "get_writer", boom)
    with pytest.warns(UserWarning, match="GWF alias registration skipped for 'gwf'"):
        assert _gwf_io._safe_get_reader("gwf", TimeSeries) is None
    with pytest.warns(UserWarning, match="GWF alias registration skipped for 'gwf'"):
        assert _gwf_io._safe_get_writer("gwf", TimeSeries) is None
