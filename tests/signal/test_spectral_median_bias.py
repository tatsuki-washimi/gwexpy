from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from gwexpy import signal
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.signal.spectral import median_bias
from gwexpy.timeseries import TimeSeries


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 1.0),
        (2, 1.0),
        (3, 5.0 / 6.0),
        (4, 5.0 / 6.0),
        (5, 47.0 / 60.0),
        (6, 47.0 / 60.0),
        (7, 319.0 / 420.0),
        (8, 319.0 / 420.0),
        (9, 1879.0 / 2520.0),
        (10, 1879.0 / 2520.0),
    ],
)
def test_median_bias_uses_independent_exponential_ordinates(n: int, expected: float):
    assert median_bias(n) == pytest.approx(expected)


def test_median_bias_large_n_tends_to_log_two() -> None:
    expected = sum((-1) ** (k + 1) / k for k in range(1, 50))
    assert median_bias(50) == pytest.approx(expected)
    assert median_bias(50) > np.log(2.0)


def test_median_bias_uses_stable_constant_time_large_n_formula() -> None:
    n = 10**12 + 1
    expected = np.log(2.0) + 1.0 / (2.0 * n) - 1.0 / (4.0 * n**2)

    assert median_bias(n) == pytest.approx(expected, abs=1e-15)


def test_median_bias_returns_limit_for_huge_integral_n() -> None:
    assert median_bias(10**1000) == np.log(2.0)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), 1.5, "3", None])
def test_median_bias_rejects_non_integral_values(value: object) -> None:
    with pytest.raises(TypeError):
        median_bias(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_median_bias_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError):
        median_bias(value)


def test_median_bias_documents_findchirp_scope() -> None:
    doc = median_bias.__doc__ or ""
    assert "FINDCHIRP" in doc
    assert "B12" in doc
    assert "6.3b" in doc
    assert "overlap" in doc.lower()


def test_median_bias_is_available_from_gwexpy_signal_namespace() -> None:
    assert signal.spectral.median_bias is median_bias


def test_signal_import_defers_spectral_and_scipy_special() -> None:
    checkout = Path(__file__).resolve().parents[2]
    script = """
import importlib
import sys

import gwexpy.signal

gwexpy = sys.modules["gwexpy"]

assert "scipy.special" not in sys.modules
assert "gwexpy.signal.spectral" not in sys.modules
assert "gwexpy.spectral" not in sys.modules
assert "spectral" not in gwexpy.__dict__

signal_spectral = importlib.import_module("gwexpy.signal.spectral")
assert signal_spectral is importlib.import_module("gwexpy.signal.spectral")
assert "scipy.special" in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )


def test_root_spectral_access_is_a_separate_import_boundary() -> None:
    checkout = Path(__file__).resolve().parents[2]
    script = """
import importlib
import sys

import gwexpy

assert "gwexpy.signal" not in sys.modules
assert "gwexpy.spectral" not in sys.modules
assert "scipy.special" not in sys.modules

root_spectral = gwexpy.spectral
assert root_spectral is importlib.import_module("gwexpy.spectral")
from gwexpy import spectral as imported_spectral

assert imported_spectral is root_spectral
assert "spectral" in gwexpy.__all__
assert "spectral" in dir(gwexpy)
assert "scipy.special" in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )


def test_root_surface_is_lazy_and_explicit_bootstrap_is_idempotent() -> None:
    checkout = Path(__file__).resolve().parents[2]
    script = """
import importlib
import sys

import gwexpy

assert "gwexpy.signal" not in sys.modules
assert "gwexpy.spectral" not in sys.modules
assert "gwpy.io" not in sys.modules
assert "scipy.special" not in sys.modules
assert "spectral" not in gwexpy.__dict__
assert "signal" in gwexpy.__all__
assert "spectral" in gwexpy.__all__
assert "signal" in dir(gwexpy)
assert "spectral" in dir(gwexpy)

root_spectral = gwexpy.spectral
assert root_spectral is importlib.import_module("gwexpy.spectral")
from gwexpy import spectral

assert spectral is root_spectral
from gwexpy import TimeSeries

assert TimeSeries.__name__ == "TimeSeries"

gwexpy.register_all()
gwexpy.register_all()
from gwexpy.interop._registry import ConverterRegistry

assert ConverterRegistry.has_constructor("TimeSeries")
assert ConverterRegistry.has_constructor("Plot")
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("method", "expected_unit"),
    [("psd", u.V**2 / u.Hz), ("asd", u.V / u.Hz**0.5)],
)
def test_gwpy_median_mean_is_exposed_with_gwexpy_metadata(
    method: str, expected_unit: u.UnitBase
) -> None:
    sample_rate = 64.0
    fftlength = 2.0
    series = TimeSeries(
        np.arange(256.0),
        sample_rate=sample_rate,
        unit=u.V,
        name="deterministic-source",
    )

    result = getattr(series, method)(
        fftlength=fftlength,
        overlap=0.0,
        method="median-mean",
    )

    assert isinstance(result, FrequencySeries)
    assert result.unit == expected_unit
    np.testing.assert_allclose(
        result.xindex.value,
        np.arange(0.0, sample_rate / 2.0 + 0.5, 0.5),
    )
    assert result.xindex.unit == u.Hz
    assert result.name == "deterministic-source"
