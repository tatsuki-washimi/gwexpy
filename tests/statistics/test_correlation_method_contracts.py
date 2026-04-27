"""Current scalar TimeSeries correlation method contracts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from gwexpy.timeseries import TimeSeries


def _linear_pair() -> tuple[TimeSeries, TimeSeries]:
    x = TimeSeries([1.0, 2.0, 3.0, 4.0], dt=1, t0=100)
    y = TimeSeries([2.0, 4.0, 6.0, 8.0], dt=1, t0=900)
    return x, y


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([controls, np.ones(len(controls))])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def test_correlation_dispatches_supported_scalar_methods() -> None:
    x, y = _linear_pair()

    assert x.correlation(y, method="pearson") == pytest.approx(1.0)
    assert x.correlation(y, method="kendall") == pytest.approx(1.0)
    assert x.correlation(y, method="fastmi", grid_size=16) >= 0.0


def test_mic_dispatch_uses_minepy_when_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeMINE:
        def __init__(self, *, alpha: float, c: int, est: str) -> None:
            calls["init"] = {"alpha": alpha, "c": c, "est": est}

        def compute_score(self, x: np.ndarray, y: np.ndarray) -> None:
            calls["score"] = (x.copy(), y.copy())

        def mic(self) -> float:
            return 0.82

    monkeypatch.setitem(sys.modules, "minepy", SimpleNamespace(MINE=FakeMINE))
    x, _ = _linear_pair()
    y = TimeSeries([2.0, 4.0, 6.0, 8.0, -100.0, -200.0], dt=1)

    assert x.correlation(y, method="mic", alpha=0.7, c=9, est="mic_e") == 0.82
    assert calls["init"] == {"alpha": 0.7, "c": 9, "est": "mic_e"}
    np.testing.assert_array_equal(calls["score"][0], x.value)
    np.testing.assert_array_equal(calls["score"][1], y.value[: len(x)])


def test_distance_dispatch_and_alias_use_dcor_when_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
        calls.append((x.copy(), y.copy()))
        return 0.75

    monkeypatch.setitem(
        sys.modules,
        "dcor",
        SimpleNamespace(distance_correlation=fake_distance_correlation),
    )
    x, _ = _linear_pair()
    y = TimeSeries([2.0, 4.0, 6.0, 8.0, -100.0, -200.0], dt=1)

    assert x.correlation(y, method="distance") == 0.75
    assert x.distance_correlation(y) == 0.75
    assert len(calls) == 2
    for call_x, call_y in calls:
        np.testing.assert_array_equal(call_x, x.value)
        np.testing.assert_array_equal(call_y, y.value[: len(x)])


def test_aliases_match_their_correlation_methods() -> None:
    x, y = _linear_pair()

    assert x.pcc(y) == pytest.approx(x.correlation(y, method="pearson"))
    assert x.ktau(y) == pytest.approx(x.correlation(y, method="kendall"))
    assert x.fastmi(y, grid_size=16) == pytest.approx(
        x.correlation(y, method="fastmi", grid_size=16)
    )


def test_unknown_correlation_method_error_records_spearman_unsupported() -> None:
    x, y = _linear_pair()

    with pytest.raises(ValueError, match="Unknown correlation method: spearman"):
        x.correlation(y, method="spearman")


def test_correlation_crops_by_position_without_t0_alignment() -> None:
    x = TimeSeries([1.0, 2.0, 3.0, 4.0], dt=1, t0=0)
    y = TimeSeries([2.0, 4.0, 6.0, 8.0, -100.0], dt=1, t0=10_000)

    assert x.pcc(y) == pytest.approx(1.0)


def test_sample_rate_mismatch_warns_and_resamples_other_series() -> None:
    coarse_times = np.arange(40, dtype=float)
    coarse_values = np.sin(coarse_times / 5.0)
    fine_times = np.arange(0.0, 40.0, 0.5)
    fine_values = np.interp(fine_times, coarse_times, coarse_values)
    x = TimeSeries(coarse_values, dt=1)
    y = TimeSeries(fine_values, dt=0.5)

    with pytest.warns(UserWarning, match="Sample rates do not match"):
        assert x.pcc(y) > 0.99


def test_scalar_timeseries_rejects_multidimensional_data_before_correlation() -> None:
    with pytest.raises(ValueError, match="2-dimensional data"):
        TimeSeries([[1.0, 2.0], [3.0, 4.0]], dt=1)


def test_constant_pearson_returns_nan_with_scipy_warning() -> None:
    constant = TimeSeries([1.0, 1.0, 1.0, 1.0], dt=1)
    varying = TimeSeries([1.0, 2.0, 3.0, 4.0], dt=1)

    with pytest.warns(stats.ConstantInputWarning):
        assert np.isnan(constant.pcc(varying))


def test_nan_policy_records_current_method_specific_behavior() -> None:
    x = TimeSeries([1.0, np.nan, 3.0, 4.0], dt=1)
    y = TimeSeries([1.0, 2.0, 3.0, 4.0], dt=1)

    try:
        pearson = x.pcc(y)
    except ValueError as exc:
        assert any(
            token in str(exc).lower()
            for token in ("nan", "non-finite", "nonfinite", "inf")
        )
    else:
        assert np.isnan(pearson)
    assert np.isnan(x.ktau(y))
    with pytest.warns(RuntimeWarning):
        with pytest.raises(Exception):
            x.fastmi(y, grid_size=16)


def test_partial_correlation_no_controls_falls_back_to_pearson_even_for_unknown_method() -> (
    None
):
    x, y = _linear_pair()

    assert x.partial_correlation(y) == pytest.approx(x.pcc(y))
    assert x.partial_correlation(y, method="not-a-method") == pytest.approx(x.pcc(y))


def test_partial_correlation_residual_and_precision_methods_match_current_formulae() -> (
    None
):
    control_values = np.arange(8, dtype=float)
    x_values = 2.0 * control_values + np.array(
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
    )
    y_values = -control_values + np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    x = TimeSeries(x_values, dt=1)
    y = TimeSeries(y_values, dt=1)
    control = TimeSeries(control_values, dt=1)

    residual_expected = stats.pearsonr(
        _residualize(x_values, control_values),
        _residualize(y_values, control_values),
    )[0]
    data = np.column_stack([x_values, y_values, control_values])
    precision = np.linalg.inv(np.cov(data, rowvar=False))
    precision_expected = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])

    assert x.partial_correlation(
        y, controls=control, method="residual"
    ) == pytest.approx(residual_expected)
    assert x.partial_correlation(
        y, controls=control, method="precision"
    ) == pytest.approx(precision_expected)


def test_partial_correlation_crops_controls_by_position() -> None:
    control_values = np.arange(6, dtype=float)
    x_values = np.array([0.0, 2.0, 4.0, 5.0, 8.0, 11.0, 99.0])
    y_values = np.array([1.0, 1.0, 1.0, -3.0, -3.0, -5.0, -99.0])
    x = TimeSeries(x_values, dt=1)
    y = TimeSeries(y_values, dt=1)
    control = TimeSeries(control_values, dt=1)

    expected = stats.pearsonr(
        _residualize(x_values[:6], control_values),
        _residualize(y_values[:6], control_values),
    )[0]
    data = np.column_stack([x_values[:6], y_values[:6], control_values])
    precision = np.linalg.inv(np.cov(data, rowvar=False))
    precision_expected = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])

    assert x.partial_correlation(
        y, controls=control, method="residual"
    ) == pytest.approx(expected)
    assert x.partial_correlation(
        y, controls=control, method="precision"
    ) == pytest.approx(precision_expected)


def test_partial_correlation_unknown_method_errors_only_after_controls_are_present() -> (
    None
):
    x, y = _linear_pair()
    control = TimeSeries([0.0, 1.0, 0.0, 1.0], dt=1)

    with pytest.raises(ValueError, match="Unknown partial correlation method: bad"):
        x.partial_correlation(y, controls=control, method="bad")
