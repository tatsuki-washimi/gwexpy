"""GWpy-compatible construction contracts for :class:`TimeSeries`."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries

_POSITIONAL_PREFIX = (
    u.V,
    1000,
    0.25,
    None,
    None,
    "H1:CONSTRUCTOR",
    "constructor-prefix",
)


def _exception_class(call: Callable[[], Any]) -> type[BaseException] | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - exception class is the oracle
        return type(exc)
    return None


class UTCDateTime:
    """Minimal float-compatible stand-in for ObsPy's supported scalar type."""

    datetime = datetime(2017, 1, 1)

    def __float__(self) -> float:
        return 1000.25


def test_constructor_exposes_gwpy_prefix_and_keyword_only_exact_extension() -> None:
    expected = inspect.signature(GwpyTimeSeries.__new__).parameters
    actual = inspect.signature(TimeSeries.__new__).parameters
    parent_names = list(expected)

    assert list(actual) == [*parent_names[:-1], "t0_ns", parent_names[-1]]
    for name in parent_names[:-1]:
        assert actual[name].kind is expected[name].kind
        assert actual[name].default == expected[name].default
    assert actual["t0_ns"].kind is inspect.Parameter.KEYWORD_ONLY
    assert actual["t0_ns"].default is None
    assert actual[parent_names[-1]].kind is inspect.Parameter.VAR_KEYWORD


@pytest.mark.parametrize("prefix_length", range(len(_POSITIONAL_PREFIX) + 1))
def test_constructor_every_gwpy_positional_prefix_matches(
    prefix_length: int,
) -> None:
    prefix = _POSITIONAL_PREFIX[:prefix_length]

    expected = GwpyTimeSeries([1.0, 2.0], *prefix)
    actual = TimeSeries([1.0, 2.0], *prefix)

    _assert_constructor_result_matches_gwpy(expected, actual)


@pytest.mark.parametrize(
    ("name", "prefix_length"),
    [
        ("unit", 1),
        ("t0", 2),
        ("dt", 3),
        ("sample_rate", 4),
        ("times", 5),
        ("channel", 6),
        ("name", 7),
    ],
)
def test_constructor_duplicate_positional_keyword_outcome_matches_gwpy(
    name: str, prefix_length: int
) -> None:
    prefix = _POSITIONAL_PREFIX[:prefix_length]
    duplicate = {name: _POSITIONAL_PREFIX[prefix_length - 1]}

    expected_error = _exception_class(
        lambda: GwpyTimeSeries([1.0, 2.0], *prefix, **duplicate)
    )
    actual_error = _exception_class(
        lambda: TimeSeries([1.0, 2.0], *prefix, **duplicate)
    )

    assert expected_error is TypeError
    assert actual_error is expected_error


def test_constructor_excess_positional_outcome_matches_gwpy() -> None:
    args = (*_POSITIONAL_PREFIX, "excess")

    expected_error = _exception_class(lambda: GwpyTimeSeries([1.0], *args))
    actual_error = _exception_class(lambda: TimeSeries([1.0], *args))

    assert expected_error is TypeError
    assert actual_error is expected_error


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
    # The explicit GWpy-compatible wrapper binds both public calling forms
    # before one canonical positional delegation.  The supported object must
    # still reach the parent unchanged and exactly once.
    assert call.args[3] is epoch


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
