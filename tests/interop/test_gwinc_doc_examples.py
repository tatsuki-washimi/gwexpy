"""Executable contracts for the public GWinc interop docstring example."""

from __future__ import annotations

import doctest
import inspect
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.interop.gwinc_ import from_gwinc_budget


class _FakeTrace:
    """Minimal stand-in for ``gwinc.trace.BudgetTrace``.

    Deliberately *not* a ``dict`` subclass. The real class is a plain object
    that implements ``keys()`` and ``__getitem__`` itself, so modelling the
    fake as a dict would let it accept usage the real type rejects and hide a
    regression behind a green test. ``keys()``, ``__getitem__`` and ``.psd``
    are the entire surface :func:`from_gwinc_budget` touches.
    """

    def __init__(self, psd: np.ndarray, subtraces: dict | None = None) -> None:
        self.psd = psd
        self._subtraces = dict(subtraces or {})

    def keys(self):
        return self._subtraces.keys()

    def __getitem__(self, key):
        return self._subtraces[key]


def test_gwinc_docstring_example_executes_against_mocked_gwinc(monkeypatch) -> None:
    """Keep the documented total-ASD workflow executable without GWinc I/O."""
    frequencies = np.array([10.0, 100.0, 1000.0])
    budget = MagicMock()
    budget.name = "aLIGO"
    budget.run.return_value = _FakeTrace(
        np.full(frequencies.shape, 4e-46),
        {
            "Quantum": _FakeTrace(np.full(frequencies.shape, 1e-46)),
            "Seismic": _FakeTrace(np.full(frequencies.shape, 3e-46)),
        },
    )

    gwinc_mock = ModuleType("gwinc")
    gwinc_mock.load_budget = MagicMock(return_value=budget)
    monkeypatch.setitem(sys.modules, "gwinc", gwinc_mock)

    doctests = doctest.DocTestFinder().find(from_gwinc_budget)
    assert len(doctests) == 1
    result = doctest.DocTestRunner().run(doctests[0])

    assert result.failed == 0
    load_args, load_kwargs = gwinc_mock.load_budget.call_args
    assert load_args == ("aLIGO",)
    np.testing.assert_array_equal(load_kwargs["freq"], frequencies)
    run_args, run_kwargs = budget.run.call_args
    assert run_args == ()
    np.testing.assert_array_equal(run_kwargs["freq"], frequencies)


def test_gwinc_docstring_does_not_claim_classmethod_bindings() -> None:
    """Prevent the documented API from regressing to nonexistent methods."""
    docstring = inspect.getdoc(from_gwinc_budget)
    assert docstring is not None

    assert not hasattr(FrequencySeries, "from_gwinc_budget")
    assert not hasattr(FrequencySeriesDict, "from_gwinc_budget")
    assert "FrequencySeries.from_gwinc_budget" not in docstring
    assert "FrequencySeriesDict.from_gwinc_budget" not in docstring


def test_fake_trace_matches_the_real_gwinc_trace_surface() -> None:
    """Keep the mock honest about what a real ``BudgetTrace`` is.

    The mocked doctest above is only meaningful if the fake behaves like the
    real object. Pin the two properties that matter: the real trace is not a
    ``dict`` subclass, and the attributes :func:`from_gwinc_budget` uses
    (``keys``, ``__getitem__``, ``psd``) exist on both.
    """
    # gwinc.trace is not re-exported on the package, so import the submodule.
    trace_module = pytest.importorskip(
        "gwinc.trace", reason="real GWinc needed to check the mock"
    )

    real = trace_module.BudgetTrace
    assert not issubclass(real, dict), (
        "BudgetTrace became a dict subclass; _FakeTrace can now model it as one"
    )

    # ``psd`` is a property on the real class but a plain instance attribute on
    # the fake, so compare against an instance rather than the class.
    fake = _FakeTrace(np.zeros(3))
    for attribute in ("keys", "__getitem__", "psd"):
        assert hasattr(real, attribute), f"BudgetTrace no longer exposes {attribute}"
        assert hasattr(fake, attribute), f"_FakeTrace no longer exposes {attribute}"


def test_gwinc_docstring_documents_every_public_parameter() -> None:
    """Fail when a new parameter is added without documenting it.

    Each parameter is a lever on what the function returns, so an undocumented
    one is an undocumented return shape. Deriving the list from
    :func:`inspect.signature` ties this to the implementation instead of to a
    hand-maintained list that silently goes stale.
    """
    docstring = inspect.getdoc(from_gwinc_budget)
    assert docstring is not None

    undocumented = [
        name
        for name in inspect.signature(from_gwinc_budget).parameters
        if f"{name} :" not in docstring
    ]
    assert not undocumented, f"undocumented parameters: {undocumented}"


def test_gwinc_docstring_keeps_the_three_documented_return_shapes() -> None:
    """Regression guard for the three examples #608 restored.

    This checks that the specific examples removed by the #608 defect are
    still present; it does not enumerate the return shapes from the
    implementation. Parameter-level coverage is asserted by the test above.
    """
    examples = [
        example.source
        for test in doctest.DocTestFinder().find(from_gwinc_budget)
        for example in test.examples
    ]
    joined = "".join(examples)

    # Total-only FrequencySeries, the full FrequencySeriesDict, and a single
    # sub-trace via trace_name are the three documented return shapes.
    assert 'quantity="asd"' in joined
    assert "FrequencySeriesDict, budget" in joined
    assert 'trace_name="Quantum"' in joined
