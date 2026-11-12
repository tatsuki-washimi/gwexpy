"""Warning call sites, structural ``rng`` typing and ``window`` guard (#513)."""

from __future__ import annotations

import inspect
import typing
import warnings

import numpy as np
import pytest

from gwexpy.statistics import gauch as gauch_module
from gwexpy.statistics import rayleigh_test as rayleigh_module
from gwexpy.statistics import student_t_indicator as student_module
from gwexpy.statistics._common import UniformRandomSource
from gwexpy.statistics.gauch import compute_gauch
from gwexpy.statistics.rayleigh_test import rayleigh_pvalue
from gwexpy.statistics.student_t_indicator import compute_student_t_nu
from gwexpy.timeseries import TimeSeries

THIS_FILE = __file__


class _StubRandom:
    """Minimal structural stand-in for `numpy.random.Generator`."""

    def __init__(self):
        self._rng = np.random.default_rng(513)

    def random(self, size):
        return self._rng.random(size)


@pytest.fixture
def series():
    rng = np.random.default_rng(513)
    return TimeSeries(rng.normal(size=512), sample_rate=32, unit="m")


def _record(call):
    """Run ``call`` and return ``(recorded warnings, line of the call)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        call()
    return caught, line


def _only(caught, fragment):
    matches = [w for w in caught if fragment in str(w.message)]
    assert matches, f"no warning containing {fragment!r}"
    return matches


# --- 1. warnings point at the user's call -------------------------------------


def test_rayleigh_direct_call_warning_points_at_caller(series):
    spec = series.rayleigh_spectrogram(fftlength=1, stride=4)

    def call():
        rayleigh_pvalue(spec, 7, n_monte_carlo=50, rng=_StubRandom(), seed=1)

    caught, _ = _record(call)

    for warning in _only(caught, "rayleigh_pvalue"):
        assert warning.filename == THIS_FILE


def test_rayleigh_wrapper_warning_points_at_caller(series):
    def call():
        series.rayleigh_test(1, 4, n_monte_carlo=50, rng=_StubRandom(), seed=1)

    caught, _ = _record(call)

    matches = _only(caught, "rayleigh_pvalue")
    assert {str(w.category.__name__) for w in matches} >= {"UserWarning"}
    for warning in matches:
        assert warning.filename == THIS_FILE


def test_gauch_direct_call_warning_points_at_caller(series):
    def call():
        compute_gauch(series, 1, window=8, n_monte_carlo=20, rng=_StubRandom(), seed=1)

    caught, _ = _record(call)

    for warning in _only(caught, "compute_gauch"):
        assert warning.filename == THIS_FILE


def test_gauch_wrapper_warning_points_at_caller(series):
    def call():
        series.gauch(1, window=8, n_monte_carlo=20, rng=_StubRandom(), seed=1)

    caught, _ = _record(call)

    for warning in _only(caught, "compute_gauch"):
        assert warning.filename == THIS_FILE


def _failing_fit(*args, **kwargs):
    raise RuntimeError("fit failure injected by the test")


def test_student_t_direct_call_warning_points_at_caller(series, monkeypatch):
    monkeypatch.setattr(student_module.stats.t, "fit", _failing_fit)

    def call():
        compute_student_t_nu(series, 1, window=4, frange=(10, 12))

    caught, _ = _record(call)

    for warning in _only(caught, "compute_student_t_nu"):
        assert warning.filename == THIS_FILE


def test_student_t_wrapper_warning_points_at_caller(series, monkeypatch):
    monkeypatch.setattr(student_module.stats.t, "fit", _failing_fit)

    def call():
        series.student_t_spectrogram(1, window=4, frange=(10, 12))

    caught, _ = _record(call)

    for warning in _only(caught, "compute_student_t_nu"):
        assert warning.filename == THIS_FILE


def test_warning_line_is_the_calling_statement(series):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        series.gauch(1, window=8, n_monte_carlo=20, rng=_StubRandom(), seed=1)

    (warning,) = _only(caught, "both rng and seed")
    assert (warning.filename, warning.lineno) == (THIS_FILE, line)


# --- 2. rng is typed structurally ------------------------------------------------


@pytest.mark.parametrize(
    "function",
    [
        rayleigh_pvalue,
        rayleigh_module._get_rayleigh_stat_null_distribution,
        compute_gauch,
        gauch_module._get_lilliefors_null_distribution,
    ],
    ids=["rayleigh_pvalue", "rayleigh_null", "compute_gauch", "lilliefors_null"],
)
def test_rng_annotation_is_the_structural_protocol(function):
    # Annotations are strings here (``from __future__ import annotations``).
    annotation = inspect.signature(function).parameters["rng"].annotation

    assert annotation == f"{UniformRandomSource.__name__} | None"


def test_generator_and_stub_satisfy_the_protocol():
    @typing.runtime_checkable
    class _Checkable(UniformRandomSource, typing.Protocol):
        pass

    assert isinstance(np.random.default_rng(0), _Checkable)
    assert isinstance(_StubRandom(), _Checkable)
    assert not isinstance(object(), _Checkable)


def test_stub_and_generator_give_identical_results(series):
    spec = series.rayleigh_spectrogram(fftlength=1, stride=4)

    class Stub:
        def __init__(self):
            self._rng = np.random.default_rng(7)

        def random(self, size):
            return self._rng.random(size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_stub = rayleigh_pvalue(spec, 7, n_monte_carlo=50, rng=Stub())
        by_generator = rayleigh_pvalue(
            spec, 7, n_monte_carlo=50, rng=np.random.default_rng(7)
        )

    np.testing.assert_array_equal(by_stub.value, by_generator.value)


# --- 3. boolean window is not an integer count -------------------------------------


@pytest.mark.parametrize(
    "window",
    [True, False, np.True_, np.False_],
    ids=["True", "False", "np.True_", "np.False_"],
)
def test_boolean_window_is_rejected(series, window):
    with pytest.raises(ValueError, match="window must be a positive integer"):
        compute_student_t_nu(series, 1, window=window)


@pytest.mark.parametrize("window", [4, np.int64(4), np.int32(4)])
def test_integer_window_is_accepted_and_equivalent(series, window):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_student_t_nu(series, 1, window=window, frange=(10, 12))
        reference = compute_student_t_nu(series, 1, window=4, frange=(10, 12))

    np.testing.assert_array_equal(result.value, reference.value)


@pytest.mark.parametrize("window", [0, -3, 2.0, "4", None])
def test_other_invalid_windows_still_rejected(series, window):
    with pytest.raises(ValueError, match="window must be a positive integer"):
        compute_student_t_nu(series, 1, window=window)
