from __future__ import annotations

import numpy as np
import pytest

from gwexpy.numerics import AutoScaler, safe_epsilon, safe_log_scale


def test_safe_epsilon_nonfinite_current_contract_falls_back_to_abs_tol():
    with pytest.warns(RuntimeWarning):
        eps = safe_epsilon([1.0, np.inf], abs_tol=1e-12)

    assert eps == pytest.approx(1e-12)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AutoScaler currently does not reject or regularize NaN input; "
        "changing this is a #286 statistical/numerical behavior decision."
    ),
)
def test_autoscaler_nonfinite_input_produces_finite_regularized_scale():
    scaler = AutoScaler([1.0, np.nan])

    assert np.isfinite(scaler.scale)
    assert np.all(np.isfinite(scaler.normalize()))


def test_safe_log_scale_uses_absolute_values_and_dynamic_zero_floor():
    values = np.array([-10.0, 0.0, 10.0])

    scaled = safe_log_scale(values, dynamic_range_db=40.0)

    assert scaled[0] == pytest.approx(scaled[2])
    assert scaled[1] == pytest.approx(10.0 * np.log10(10.0 * 1e-4))
    assert np.all(np.isfinite(scaled))
