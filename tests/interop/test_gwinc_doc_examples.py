"""Executable contracts for the public GWinc interop docstring example."""

from __future__ import annotations

import doctest
import inspect
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.interop.gwinc_ import from_gwinc_budget


class _FakeTrace(dict):
    """Minimal stand-in for ``gwinc.BudgetTrace``.

    Real traces expose their PSD as ``.psd`` and their sub-traces through
    dict-like access, which is all :func:`from_gwinc_budget` relies on.
    """

    def __init__(self, psd: np.ndarray, subtraces: dict | None = None) -> None:
        super().__init__(subtraces or {})
        self.psd = psd


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


def test_gwinc_docstring_covers_every_documented_return_shape() -> None:
    """#608 removed three broken examples; keep all three paths demonstrated."""
    docstring = inspect.getdoc(from_gwinc_budget)
    assert docstring is not None

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
