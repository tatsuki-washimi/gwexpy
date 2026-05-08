import os
from pathlib import Path

import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, _gwf_io
from gwexpy.timeseries._gwf_io import _extract_gwf_read_args, _resolve_gwf_format

FIXTURE_DATA = Path(__file__).parent.parent / "fixtures" / "data" / "test.gwf"
CHANNEL = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"
GWF_REQUIRE_FRAMEL_ENV = "GWEXPY_REQUIRE_GWF_FRAMEL"
GWF_FORMAT_BACKEND_CASES = (
    ("gwf", None),
    ("gwf.framecpp", "frameCPP"),
    ("gwf.lalframe", "lalframe"),
    ("gwf.framel", "framel"),
)
GWF_BACKEND_ERRORS = (
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _error_reason(prefix: str, exc: BaseException) -> str:
    return f"{prefix}: {type(exc).__name__}: {exc}"


def _framel_required() -> bool:
    return os.environ.get(GWF_REQUIRE_FRAMEL_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def gwf_discovery_available(backend: str | None = None) -> tuple[bool, str]:
    """Return availability for gwpy.io.gwf channel discovery only."""
    try:
        from gwpy.io.gwf.core import get_channel_names

        kwargs = {"backend": backend} if backend is not None else {}
        channels = get_channel_names(FIXTURE_DATA, **kwargs)
    except GWF_BACKEND_ERRORS as exc:
        return False, _error_reason(
            f"{backend or 'default'} gwf discovery unavailable", exc
        )
    if not channels:
        return False, f"{backend or 'default'} gwf discovery returned no channels"
    return True, f"{backend or 'default'} gwf discovery available"


def gwf_explicit_read_available(
    fmt: str, backend: str | None = None
) -> tuple[bool, str]:
    """Return availability for explicit TimeSeries GWF read dispatch."""
    try:
        TimeSeries.read(FIXTURE_DATA, CHANNEL, format=fmt)
    except GWF_BACKEND_ERRORS as exc:
        return False, _error_reason(
            f"{backend or 'default'} explicit {fmt} read dispatch unavailable", exc
        )
    return True, f"{backend or 'default'} explicit {fmt} read dispatch available"


def has_gwf_discovery_backend(backend: str | None = None) -> bool:
    available, _ = gwf_discovery_available(backend)
    return available


def _require_or_skip_gwf_explicit_read(fmt: str, backend: str | None = None) -> None:
    _, discovery_reason = gwf_discovery_available(backend)
    read_available, read_reason = gwf_explicit_read_available(fmt, backend)
    if read_available:
        return

    reason = f"{discovery_reason}; {read_reason}"
    if backend == "framel" and _framel_required():
        pytest.fail(
            f"{GWF_REQUIRE_FRAMEL_ENV}=1 requires FrameL GWF explicit read support; "
            f"{reason}"
        )
    pytest.skip(reason)


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(
    not has_gwf_discovery_backend(), reason="default gwf discovery not available"
)
def test_read_gwf_timeseriesdict_single_channel_string():
    tsd = TimeSeriesDict.read(FIXTURE_DATA, CHANNEL, format="gwf")

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert tsd[CHANNEL].name == CHANNEL
    assert len(tsd[CHANNEL]) > 0


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(
    not has_gwf_discovery_backend(), reason="default gwf discovery not available"
)
def test_read_gwf_timeseriesdict_channels_kwarg():
    tsd = TimeSeriesDict.read(FIXTURE_DATA, channels=[CHANNEL], format="gwf")

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(
    not has_gwf_discovery_backend(), reason="default gwf discovery not available"
)
def test_read_gwf_timeseriesdict_autodetects_extension_and_channels():
    tsd = TimeSeriesDict.read(FIXTURE_DATA)

    assert isinstance(tsd, TimeSeriesDict)
    assert CHANNEL in tsd


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(
    not has_gwf_discovery_backend(), reason="default gwf discovery not available"
)
def test_read_gwf_timeseriesdict_autodetects_list_source_with_nproc_alias():
    tsd = TimeSeriesDict.read([FIXTURE_DATA], CHANNEL, nproc=1)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(
    not has_gwf_discovery_backend(), reason="default gwf discovery not available"
)
def test_read_gwf_timeseriesdict_explicit_format_drops_unsupported_nproc():
    tsd = TimeSeriesDict.read([FIXTURE_DATA], CHANNEL, format="lalframe", nproc=1)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]


@pytest.mark.parametrize(("fmt", "backend"), GWF_FORMAT_BACKEND_CASES)
@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
def test_read_gwf_timeseriesdict_real_file_backend_matrix(fmt, backend):
    _require_or_skip_gwf_explicit_read(fmt, backend)

    tsd = TimeSeriesDict.read(FIXTURE_DATA, CHANNEL, format=fmt)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert len(tsd[CHANNEL]) > 0


@pytest.mark.parametrize(("fmt", "backend"), GWF_FORMAT_BACKEND_CASES)
@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
def test_read_gwf_timeseriesdict_real_file_list_source_backend_matrix(fmt, backend):
    _require_or_skip_gwf_explicit_read(fmt, backend)

    tsd = TimeSeriesDict.read([FIXTURE_DATA], CHANNEL, format=fmt)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd) == [CHANNEL]
    assert len(tsd[CHANNEL]) > 0


@pytest.mark.parametrize(("fmt", "backend"), GWF_FORMAT_BACKEND_CASES)
@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
def test_read_gwf_timeseries_real_file_backend_matrix(fmt, backend):
    _require_or_skip_gwf_explicit_read(fmt, backend)

    ts = TimeSeries.read(FIXTURE_DATA, CHANNEL, format=fmt)

    assert isinstance(ts, TimeSeries)
    assert ts.name == CHANNEL
    assert len(ts) > 0


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
    assert _resolve_gwf_format([path], None) == "gwf"


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


def test_filter_gwf_reader_kwargs_preserves_unknown_kwargs_except_parallel_aliases():
    def reader(source, channels, *, parallel=None, scaled=None):
        return None

    filtered = _gwf_io._filter_gwf_reader_kwargs(
        reader,
        {"nproc": 2, "scaled": True, "custom": "kept"},
    )

    assert filtered == {"parallel": 2, "scaled": True, "custom": "kept"}


def test_filter_gwf_reader_kwargs_drops_parallel_aliases_when_reader_rejects_them():
    def reader(source, channels, *, scaled=None):
        return None

    filtered = _gwf_io._filter_gwf_reader_kwargs(
        reader,
        {"nproc": 2, "parallel": 3, "scaled": True, "custom": "kept"},
    )

    assert filtered == {"scaled": True, "custom": "kept"}


def test_read_gwf_timeseries_with_single_channel_by_format_gwf():
    _require_or_skip_gwf_explicit_read("gwf.framel", "framel")

    ts = TimeSeries.read(FIXTURE_DATA, CHANNEL, format="gwf.framel")
    assert isinstance(ts, TimeSeries)
    assert ts.name == CHANNEL
