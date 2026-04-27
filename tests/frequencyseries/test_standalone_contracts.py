from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.units import Quantity
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.types.series import Series as GwpySeries

from gwexpy.frequencyseries import BifrequencyMap, FrequencySeries


def _make_frequencyseries() -> FrequencySeries:
    return FrequencySeries(
        [1 + 1j, 2 + 0j, 3 - 1j],
        frequencies=[0, 1, 2] * u.Hz,
        unit=u.m,
        name="sig",
        channel="H1:TEST",
        epoch=1234567890,
    )


def _make_bifrequency_map() -> BifrequencyMap:
    return BifrequencyMap.from_points(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
        f2=[10, 20],
        f1=[1, 2, 3],
        unit=u.s,
        name="map",
    )


def test_frequencyseries_constructor_filters_noise_range_kwargs():
    fs = FrequencySeries(
        [1, 2, 3],
        df=1 * u.Hz,
        fmin=10,
        fmax=20,
        unit=u.m,
    )

    assert isinstance(fs, FrequencySeries)
    assert fs.unit == u.m


def test_frequencyseries_indexing_views_and_gwex_attrs_contract():
    fs = _make_frequencyseries()
    fs._gwex_marker = {"source": "contract"}  # noqa: SLF001

    scalar = fs[1]
    sliced = fs[1:]
    ndarray_view = fs.view(np.ndarray)
    typed_view = fs.view(type(fs))

    assert isinstance(scalar, Quantity)
    assert scalar.unit == u.m
    assert isinstance(sliced, FrequencySeries)
    assert sliced._gwex_marker == {"source": "contract"}  # noqa: SLF001
    assert type(ndarray_view) is np.ndarray
    assert isinstance(typed_view, FrequencySeries)
    assert np.shares_memory(fs.value, typed_view.value)
    assert typed_view.name == "sig"
    assert str(typed_view.channel) == "H1:TEST"
    assert typed_view.epoch == fs.epoch


def test_frequencyseries_phase_db_rebin_and_quadrature_metadata_contracts():
    fs = _make_frequencyseries()

    phase = fs.phase()
    degree = fs.degree()
    decibel = fs.to_db()

    assert isinstance(phase, FrequencySeries)
    assert phase.unit == u.rad
    assert phase.name == "sig_phase"
    assert str(phase.channel) == "H1:TEST"
    assert phase.epoch == fs.epoch
    assert degree.unit == u.deg
    assert degree.name == "sig_phase_deg"
    assert decibel.unit == u.dB
    assert decibel.name == "sig_db"

    rebinned = FrequencySeries(
        [1, 2, 3, 4, 5],
        frequencies=[0, 1, 2, 3, 4] * u.Hz,
        unit=u.m,
        name="rb",
        channel="H1:TEST",
        epoch=fs.epoch,
    ).rebin(2 * u.Hz)
    assert isinstance(rebinned, FrequencySeries)
    assert rebinned.value.tolist() == [1.5, 3.5]
    assert rebinned.frequencies.value.tolist() == [0.5, 2.5]
    assert rebinned.frequencies.unit == u.Hz
    assert rebinned.unit == u.m
    assert rebinned.name == "rb"
    assert str(rebinned.channel) == "H1:TEST"

    other = FrequencySeries(
        [1, 1, 1],
        frequencies=fs.frequencies,
        unit=u.m,
        name="other",
        channel="H1:OTHER",
        epoch=fs.epoch,
    )
    combined = fs.quadrature_sum(other)
    assert isinstance(combined, FrequencySeries)
    assert combined.unit == u.m
    assert combined.epoch == fs.epoch
    assert combined.channel is None


def test_bifrequencymap_axes_indexing_and_crop_contracts():
    bmap = _make_bifrequency_map()

    assert bmap.frequency2.value.tolist() == [10, 20]
    assert bmap.frequency2.unit == u.Hz
    assert bmap.frequency1.value.tolist() == [1, 2, 3]
    assert bmap.frequency1.unit == u.Hz
    assert isinstance(bmap[0, 0], Quantity)
    assert bmap[0, 0].unit == u.s
    assert isinstance(bmap[:, :], BifrequencyMap)
    assert type(bmap[0, :]) is GwpySeries

    cropped = bmap.crop(low=1, high=2, low2=20, high2=20)
    assert isinstance(cropped, BifrequencyMap)
    assert cropped.shape == (1, 2)
    assert cropped.frequency1.value.tolist() == [1, 2]
    assert cropped.frequency1.unit == u.Hz
    assert cropped.frequency2.value.tolist() == [20]
    assert cropped.frequency2.unit == u.Hz
    assert cropped.unit == u.s
    assert cropped.name == "map"


def test_bifrequencymap_projection_helpers_return_gwpy_frequencyseries():
    bmap = _make_bifrequency_map()
    input_spectrum = FrequencySeries(
        [1, 1, 1],
        frequencies=[100, 200, 300] * u.Hz,
        unit=u.m,
        name="input",
        channel="H1:IN",
        epoch=123,
    )

    projected = bmap.propagate(input_spectrum, interpolate=False)

    assert type(projected) is GwpyFrequencySeries
    assert projected.value.tolist() == [6, 15]
    assert projected.frequencies.value.tolist() == [10, 20]
    assert projected.frequencies.unit == u.Hz
    assert projected.unit == u.m * u.s
    assert projected.name == "Projected: map x input"
    assert projected.channel is None
    assert projected.epoch is None

    diagonal = bmap.diagonal()
    assert type(diagonal) is GwpyFrequencySeries
    assert diagonal.unit == u.s
    assert diagonal.name == "map (diagonal mean)"

    convolved = bmap.convolute(input_spectrum, interpolate=False)
    assert type(convolved) is GwpyFrequencySeries
    assert convolved.value.tolist() == [6, 15]
    assert convolved.frequencies.value.tolist() == [10, 20]
    assert convolved.unit == u.Hz * u.m * u.s
    assert convolved.channel is None
    assert convolved.epoch is None

    nearest = bmap.get_slice(2 * u.kHz, axis="f1")
    assert type(nearest) is GwpyFrequencySeries
    assert nearest.value.tolist() == [2, 5]
    assert nearest.frequencies.value.tolist() == [10, 20]
    assert nearest.frequencies.unit == u.Hz
    assert nearest.name == "map (at f1=2 Hz)"

    with pytest.raises(ValueError, match="axis must be"):
        bmap.get_slice(1, axis="bad")
