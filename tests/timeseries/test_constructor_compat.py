"""GWpy-compatible construction contracts for :class:`TimeSeries`."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries


class UTCDateTime:
    """Minimal float-compatible stand-in for ObsPy's supported scalar type."""

    datetime = datetime(2017, 1, 1)

    def __float__(self) -> float:
        return 1000.25


def _assert_constructor_result_matches_gwpy(
    expected: GwpyTimeSeries, observed: TimeSeries
) -> None:
    np.testing.assert_array_equal(observed.value, expected.value)
    np.testing.assert_array_equal(observed.times.value, expected.times.value)
    assert observed.shape == expected.shape
    assert observed.dtype == expected.dtype
    assert observed.unit == expected.unit
    assert observed.t0 == expected.t0
    assert observed.dt == expected.dt
    assert observed.times.unit == expected.times.unit


def test_numeric_keyword_epoch_preserves_gwpy_axis_unit_semantics() -> None:
    expected = GwpyTimeSeries([1.0, 2.0], None, 1000, 1 * u.ms)
    expected_keyword = GwpyTimeSeries([1.0, 2.0], t0=1000, dt=1 * u.ms)

    observed = TimeSeries([1.0, 2.0], t0=1000, dt=1 * u.ms)
    observed_positional = TimeSeries([1.0, 2.0], None, 1000, 1 * u.ms)
    observed_epoch = TimeSeries([1.0, 2.0], epoch=1000, dt=1 * u.ms)

    _assert_constructor_result_matches_gwpy(expected, expected_keyword)
    _assert_constructor_result_matches_gwpy(expected, observed)
    _assert_constructor_result_matches_gwpy(expected, observed_positional)
    _assert_constructor_result_matches_gwpy(expected, observed_epoch)


@pytest.mark.parametrize(
    "epoch",
    [
        "1000.25",
        Decimal("1000.25"),
        1000.25 * u.s,
        Time(1000.25, format="gps"),
        LIGOTimeGPS(1000, 250_000_000),
        np.datetime64("2017-01-01T00:00:00.123456789", "ns"),
        UTCDateTime(),
    ],
)
def test_gwpy_supported_keyword_epoch_types_pass_through_unchanged(
    epoch: object,
) -> None:
    expected = GwpyTimeSeries([1.0, 2.0], t0=epoch, dt=1 * u.ms)

    observed = TimeSeries([1.0, 2.0], t0=epoch, dt=1 * u.ms)

    _assert_constructor_result_matches_gwpy(expected, observed)


@pytest.mark.parametrize(
    "epoch",
    [
        "2017-01-01T00:00:00",
        datetime(2017, 1, 1),
        (2017, 1, 1),
        [2017, 1, 1],
    ],
)
def test_gwexpy_only_epoch_uses_same_positional_and_keyword_route(
    epoch: object,
) -> None:
    positional = TimeSeries([1.0, 2.0], None, epoch, 1 * u.ms)

    keyword = TimeSeries([1.0, 2.0], unit=None, t0=epoch, dt=1 * u.ms)

    _assert_constructor_result_matches_gwpy(keyword, positional)


@pytest.mark.parametrize(
    ("epoch", "expected_gps"),
    [
        ((2017, 1, 1), 1_167_264_018.0),
        ([2017, 1, 1, 0, 0, 0, 123_456], 1_167_264_018.123_456),
    ],
)
@pytest.mark.parametrize("form", ["positional", "keyword"])
def test_date_component_epoch_uses_scalar_gwpy_date_parsing(
    epoch: object, expected_gps: float, form: str
) -> None:
    if form == "positional":
        series = TimeSeries([1.0, 2.0], None, epoch, 1 * u.s)
    else:
        series = TimeSeries([1.0, 2.0], t0=epoch, dt=1 * u.s)

    assert series.t0.to_value(u.s) == pytest.approx(expected_gps, abs=1e-7)


@pytest.mark.parametrize("epoch", [(2017, 13, 1), [2017, 13, 1]])
def test_invalid_date_component_epoch_preserves_scalar_parser_error(
    epoch: object,
) -> None:
    with pytest.raises(ValueError):
        TimeSeries([1.0, 2.0], t0=epoch, dt=1 * u.s)


@pytest.mark.parametrize(
    ("epoch", "error"),
    [
        ("2017-01-01T00:00:00", ValueError),
        (datetime(2017, 1, 1), TypeError),
        ((2017, 1, 1), TypeError),
        ([2017, 1, 1], TypeError),
    ],
)
def test_gwexpy_only_epoch_types_are_not_successful_gwpy_constructor_inputs(
    epoch: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        GwpyTimeSeries([1.0, 2.0], t0=epoch, dt=1 * u.ms)


@pytest.mark.parametrize(
    "epoch",
    [
        1000,
        "1000.25",
        Decimal("1000.25"),
        1000.25 * u.s,
        Time(1000.25, format="gps"),
        LIGOTimeGPS(1000, 250_000_000),
        np.datetime64("2017-01-01T00:00:00.123456789", "ns"),
        UTCDateTime(),
    ],
)
@pytest.mark.parametrize("form", ["positional", "keyword"])
def test_gwpy_supported_epoch_reaches_parent_once_and_unchanged(
    epoch: object, form: str
) -> None:
    parent_result = np.array([1.0, 2.0]).view(TimeSeries)
    with patch(
        "gwpy.timeseries.TimeSeries.__new__", return_value=parent_result
    ) as parent:
        if form == "positional":
            TimeSeries([1.0, 2.0], None, epoch, 1 * u.ms)
        else:
            TimeSeries([1.0, 2.0], t0=epoch, dt=1 * u.ms)

    parent.assert_called_once()
    call = parent.call_args
    if form == "positional":
        assert call.args[3] is epoch
    else:
        assert call.kwargs["t0"] is epoch


def test_parent_constructor_failure_is_not_retried() -> None:
    class ParentFailure(RuntimeError):
        pass

    with patch(
        "gwpy.timeseries.TimeSeries.__new__", side_effect=ParentFailure("sentinel")
    ) as parent:
        with pytest.raises(ParentFailure, match="sentinel"):
            TimeSeries([1.0, 2.0], t0=1000, dt=1 * u.ms)

    parent.assert_called_once()


def test_t0_ns_rejects_xindex_as_a_competing_epoch_authority() -> None:
    with pytest.raises(TypeError, match=r"t0_ns.*xindex"):
        TimeSeries(
            [1.0, 2.0],
            t0_ns=1_167_264_018_000_000_000,
            dt=1 * u.s,
            xindex=[100.0, 101.0] * u.s,
        )


def test_iso_epoch_normalization_prefers_explicit_xunit_to_dt_unit() -> None:
    series = TimeSeries(
        [1.0, 2.0],
        t0="2017-01-01T00:00:00",
        dt=1 * u.ms,
        xunit=u.s,
    )

    assert series.times.unit == u.s
    assert series.dt.to_value(u.s) == pytest.approx(0.001)
    assert series.t0.to_value(u.s) == pytest.approx(1_167_264_018.0, abs=1e-7)


def test_t0_ns_visible_epoch_prefers_explicit_xunit_to_dt_unit() -> None:
    exact_t0_ns = 1_167_264_018_123_456_789
    series = TimeSeries(
        [1.0, 2.0],
        t0_ns=exact_t0_ns,
        dt=1 * u.ms,
        xunit=u.s,
    )

    assert series.times.unit == u.s
    assert series.dt.to_value(u.s) == pytest.approx(0.001)
    assert series.t0.to_value(u.s) == pytest.approx(
        exact_t0_ns / 1_000_000_000, rel=0, abs=1e-6
    )
    assert series.t0_gps_ns == exact_t0_ns
