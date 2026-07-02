from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _channel_label(channel: object) -> str | None:
    if channel is None:
        return None
    return str(channel)


def assert_timeseries_close(
    actual,
    expected_values: Sequence[float] | np.ndarray | None = None,
    *,
    sample_rate: float | None = None,
    t0: float | None = None,
    name: str | None = None,
    channel: str | None = None,
    unit: str | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    if expected_values is not None:
        expected = _as_array(expected_values)
        actual_values = _as_array(actual.value)
        assert actual_values.shape == expected.shape
        np.testing.assert_allclose(actual_values, expected, rtol=rtol, atol=atol)

    if sample_rate is not None:
        np.testing.assert_allclose(float(actual.sample_rate.value), sample_rate)

    if t0 is not None:
        np.testing.assert_allclose(float(actual.t0.value), t0)

    if name is not None:
        assert getattr(actual, "name", None) == name

    if channel is not None:
        assert _channel_label(getattr(actual, "channel", None)) == channel

    if unit is not None:
        assert str(getattr(actual, "unit", "")) == unit


def assert_timeseriesdict_close(
    actual,
    expected_values: Mapping[str, Sequence[float] | np.ndarray],
    *,
    sample_rate: float,
    t0: float,
    unit_by_key: Mapping[str, str] | None = None,
    check_channel: bool = True,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    assert tuple(actual.keys()) == tuple(expected_values.keys())

    for key, values in expected_values.items():
        assert_timeseries_close(
            actual[key],
            values,
            sample_rate=sample_rate,
            t0=t0,
            name=key,
            channel=key if check_channel else None,
            unit=None if unit_by_key is None else unit_by_key[key],
            rtol=rtol,
            atol=atol,
        )
