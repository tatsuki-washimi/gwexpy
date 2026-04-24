from pathlib import Path

import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries._gwf_io import _extract_gwf_read_args, _resolve_gwf_format

FIXTURE_DATA = Path(__file__).parent.parent / "fixtures" / "data" / "test.gwf"
CHANNEL = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"


def has_gwf_backend() -> bool:
    try:
        from gwpy.io.gwf.core import get_channel_names

        return bool(get_channel_names(FIXTURE_DATA))
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return False


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


def test_read_gwf_timeseries_with_autodetect_requires_backend():
    pytest.skip("Backend-dependent integration test for .gwf without explicit format")


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


def test_extract_gwf_read_args_rejects_positional_keyword_overlap():
    with pytest.raises(
        TypeError, match="Cannot specify both positional and keyword 'start' for GWF read"
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

    if not has_gwf_backend():
        pytest.skip("gwf backend not available")

    try:
        expected = get_channel_names(FIXTURE_DATA, backend="frameCPP")
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        expected = get_channel_names(FIXTURE_DATA)
    ts = TimeSeries.read(FIXTURE_DATA, CHANNEL, format="gwf.framel")
    assert isinstance(ts, TimeSeries)
    assert ts.name == CHANNEL
    if expected:
        assert ts.name in expected
