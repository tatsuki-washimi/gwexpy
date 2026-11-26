"""Tests of explicit Differometor array conversions."""

import os

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.interop import (
    from_differometor_noise_budget,
    from_differometor_response,
    from_differometor_sensitivity,
)


def test_sensitivity_preserves_values_axis_and_unit():
    result = from_differometor_sensitivity(
        FrequencySeries, [10, 20, 40], [1, 2, 3], unit=u.m / u.Hz**0.5, name="ASD"
    )
    np.testing.assert_array_equal(result.value, [1, 2, 3])
    np.testing.assert_array_equal(result.frequencies.value, [10, 20, 40])
    assert result.unit == u.m / u.Hz**0.5
    assert result.name == "ASD"


def test_caller_supplied_noise_components():
    result = from_differometor_noise_budget(
        FrequencySeriesDict, [1, 2], {"thermal": [3, 4], "quantum": [5, 6]}, unit="m"
    )
    assert set(result) == {"thermal", "quantum"}
    np.testing.assert_array_equal(result["quantum"].value, [5, 6])
    np.testing.assert_array_equal(result["thermal"].frequencies.value, [1, 2])
    assert result["thermal"].unit == u.m
    via_class = FrequencySeriesDict.from_differometor_noise_budget(
        [1, 2], {"thermal": [3, 4]}, unit="m"
    )
    np.testing.assert_array_equal(via_class["thermal"].value, [3, 4])


@pytest.mark.parametrize("values", [[1], [[1, 2]]])
def test_reject_mismatched_or_multidimensional_arrays(values):
    with pytest.raises(ValueError):
        from_differometor_sensitivity(FrequencySeries, [1, 2], values, unit="m")


def test_unit_is_required_even_for_empty_budget():
    with pytest.raises(ValueError):
        from_differometor_noise_budget(FrequencySeriesDict, [1], {}, unit=None)


def test_response_selects_complex_upper_sideband_row():
    signal = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    result = from_differometor_response(
        FrequencySeries, [1, 2], signal, {"port": 1}, output="port", unit="m"
    )
    np.testing.assert_array_equal(result.value, signal[1])
    np.testing.assert_array_equal(result.frequencies.value, [1, 2])
    assert result.unit == u.m
    with pytest.raises(KeyError):
        from_differometor_response(
            FrequencySeries, [1, 2], signal, {}, output="port", unit="m"
        )
    matrix = FrequencySeries.from_differometor_response(
        [1, 2], signal, {"first": 0, "second": 1}, outputs=["first", "second"], unit="m"
    )
    np.testing.assert_array_equal(matrix.value[:, 0, :], signal)
    np.testing.assert_array_equal(matrix.frequencies.value, [1, 2])
    assert all(unit == u.m for unit in matrix.units.flat)
    np.testing.assert_array_equal(matrix.names[:, 0], ["first", "second"])
    with pytest.raises(NotImplementedError):
        FrequencySeries.from_differometor_response(
            [1, 2], signal, {"first": 0}, output="first", inputs=["drive"], unit="m"
        )


def test_real_differometor_noise_when_fixture_supplied():
    path = os.environ.get("GWEXPY_DIFFEROMETOR_FIXTURE")
    if not path:
        pytest.skip("set GWEXPY_DIFFEROMETOR_FIXTURE to a real producer output")
    with np.load(path) as producer:
        result = FrequencySeries.from_differometor_sensitivity(
            producer["frequencies"], producer["noise"], unit="m"
        )
        np.testing.assert_array_equal(result.value, producer["noise"])
        np.testing.assert_array_equal(result.frequencies.value, producer["frequencies"])
        assert result.unit == u.m
        port = str(producer["port"])
        row = int(producer["row"])
        response = FrequencySeries.from_differometor_response(
            producer["frequencies"],
            producer["signal"],
            {port: row},
            output=port,
            unit="m",
        )
        np.testing.assert_array_equal(response.value, producer["signal"][row])
