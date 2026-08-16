"""Focused contracts for the TimeSeries raw-doctest convergence lane."""

import ast
import doctest
import inspect

import numpy as np
import pytest

from gwexpy.frequencyseries.collections import FrequencySeriesDict
from gwexpy.histogram.collections import HistogramDict
from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries._signal import TimeSeriesSignalMixin
from gwexpy.timeseries._spectral_special import TimeSeriesSpectralSpecialMixin
from gwexpy.timeseries.pipeline import Pipeline, Transform


def _doc_examples(obj):
    doc = inspect.getdoc(obj)
    assert doc is not None
    return doctest.DocTestParser().get_examples(doc)


@pytest.mark.parametrize(
    "method_name",
    ["hilbert", "instantaneous_phase", "instantaneous_frequency", "baseband"],
)
def test_signal_examples_import_numpy_in_each_time_series_setup(method_name):
    method = getattr(TimeSeriesSignalMixin, method_name)
    doc = inspect.getdoc(method)
    assert doc is not None

    assert "import numpy as np" in doc


@pytest.mark.parametrize(
    ("obj", "required_imports"),
    [
        (Transform, ["from gwexpy.timeseries import ImputeTransform"]),
        (
            Pipeline,
            [
                "from gwexpy.timeseries import ImputeTransform",
                "from gwexpy.timeseries import Pipeline",
                "from gwexpy.timeseries import StandardizeTransform",
            ],
        ),
    ],
)
def test_pipeline_examples_import_public_transform_names(obj, required_imports):
    doc = inspect.getdoc(obj)
    assert doc is not None

    assert "ImputeTransform" in doc
    for required_import in required_imports:
        assert required_import in doc


@pytest.mark.parametrize("method_name", ["emd", "hht"])
def test_pyemd_examples_execute_deterministic_setup_before_optional_call(method_name):
    method = getattr(TimeSeriesSpectralSpecialMixin, method_name)
    examples = _doc_examples(method)
    doc = inspect.getdoc(method)
    assert doc is not None
    setup_sources = [
        example.source for example in examples if "TimeSeries(" in example.source
    ]

    assert "import numpy as np" in doc
    assert "from gwexpy.timeseries import TimeSeries" in doc
    assert "data =" in doc
    assert setup_sources
    assert all("doctest: +SKIP" not in source for source in setup_sources)
    assert ".emd(" in doc or ".hht(" in doc
    assert ".. code-block:: python" in doc
    assert "doctest: +SKIP" not in doc


@pytest.mark.parametrize(
    ("obj", "sample_count", "grid_expression", "metadata_expression"),
    [
        (
            TimeSeriesSignalMixin.hilbert,
            1000,
            "np.arange(1000) * 0.001",
            "dt=0.001",
        ),
        (
            TimeSeriesSignalMixin.instantaneous_phase,
            1000,
            "np.arange(1000) * 0.001",
            "dt=0.001",
        ),
        (
            TimeSeriesSignalMixin.instantaneous_frequency,
            1000,
            "np.arange(1000) * 0.001",
            "dt=0.001",
        ),
        (
            TimeSeriesSignalMixin.baseband,
            1000,
            "np.arange(1000) * 0.001",
            "dt=0.001",
        ),
        (
            TimeSeriesSignalMixin.lock_in,
            163840,
            "np.arange(163840) / 16384",
            "sample_rate=16384",
        ),
        (
            TimeSeriesSpectralSpecialMixin.hilbert_analysis,
            1000,
            "np.arange(1000) * 0.001",
            "dt=0.001",
        ),
    ],
)
def test_numerical_examples_use_exclusive_grids_matching_metadata(
    obj, sample_count, grid_expression, metadata_expression
):
    doc = inspect.getdoc(obj)
    assert doc is not None
    assert "np.linspace(" not in doc
    assert grid_expression in doc
    assert metadata_expression in doc

    tree = ast.parse("\n".join(example.source for example in _doc_examples(obj)))
    arange_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "arange"
    ]
    assert any(
        node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == sample_count
        for node in arange_calls
    )


def test_instantaneous_frequency_example_uses_strict_executable_assertion():
    doc = inspect.getdoc(TimeSeriesSignalMixin.instantaneous_frequency)
    assert doc is not None

    assert "# doctest: +SKIP" not in doc
    assert "rtol=2e-3" not in doc
    assert "np.testing.assert_allclose" in doc
    assert "rtol=1e-12" in doc
    assert "atol=1e-12" in doc


@pytest.mark.parametrize(
    ("obj", "required_semantics"),
    [
        (
            FrequencySeriesDict,
            [
                'list(fsd.keys()) == ["H1"]',
                'type(fsd["H1"]).__name__ == "FrequencySeries"',
            ],
        ),
        (
            HistogramDict,
            [
                'list(hd.keys()) == ["H1"]',
                'type(hd["H1"]).__name__ == "Histogram"',
            ],
        ),
    ],
)
def test_collection_examples_assert_stable_public_semantics(obj, required_semantics):
    doc = inspect.getdoc(obj)
    assert doc is not None

    for semantic_assertion in required_semantics:
        assert semantic_assertion in doc
    assert "FrequencySeriesDict([(" not in doc
    assert "HistogramDict([(" not in doc


def test_rfft_matches_the_public_gwpy_normalized_fft_contract():
    """The documented rfft alias remains available after the GWpy API change."""
    ts = TimeSeries([1.0, 2.0, 3.0, 4.0], sample_rate=1.0)

    result = ts.rfft()
    expected = ts.fft()

    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.frequencies.value, expected.frequencies.value)
    assert result.unit == expected.unit


def test_rfft_positional_nfft_matches_fft():
    ts = TimeSeries(np.arange(5.0), sample_rate=2.0)

    result = ts.rfft(8)
    expected = ts.fft(nfft=8)

    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.frequencies.value, expected.frequencies.value)


def test_rfft_keyword_nfft_matches_fft():
    ts = TimeSeries(np.arange(5.0), sample_rate=2.0)

    result = ts.rfft(nfft=8)
    expected = ts.fft(nfft=8)

    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.frequencies.value, expected.frequencies.value)


def test_rfft_rejects_two_positional_arguments():
    ts = TimeSeries(np.arange(5.0), sample_rate=2.0)

    with pytest.raises(TypeError, match="at most 1 positional argument"):
        ts.rfft(8, 4)
