"""Historical ``filterba`` call-shape compatibility tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries

from gwexpy.frequencyseries import FrequencySeries


def test_gwpy4_has_no_filterba_parent() -> None:
    assert not hasattr(GwpyFrequencySeries, "filterba")


def test_filterba_warns_and_delegates_arguments_unchanged() -> None:
    series = FrequencySeries([1.0, 2.0, 3.0], df=1.0)
    expected = object()
    numerator = np.array([1.0, 0.5])
    denominator = np.array([1.0, -0.25])

    with patch.object(FrequencySeries, "filter", return_value=expected) as delegate:
        with pytest.warns(DeprecationWarning, match="use FrequencySeries.filter"):
            result = series.filterba(
                numerator,
                denominator,
                analog=False,
                inplace=False,
            )

    assert result is expected
    delegate.assert_called_once_with(
        numerator,
        denominator,
        analog=False,
        inplace=False,
    )


def test_filterba_preserves_filter_defaults_values_and_metadata() -> None:
    series = FrequencySeries(
        [1.0, 2.0, 3.0, 4.0],
        f0=0,
        df=1.0,
        unit="V",
        name="compat",
        epoch=1000,
    )
    numerator = np.array([1.0])
    denominator = np.array([1.0])

    filt = (numerator, denominator)
    expected = series.filter(filt)
    with pytest.warns(DeprecationWarning):
        observed = series.filterba(filt)

    assert observed is not series
    np.testing.assert_array_equal(observed.value, expected.value)
    np.testing.assert_array_equal(
        observed.frequencies.value, expected.frequencies.value
    )
    assert observed.unit == expected.unit
    assert observed.f0 == expected.f0
    assert observed.df == expected.df
    assert observed.epoch == expected.epoch
    assert observed.name == expected.name
