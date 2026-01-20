
import numpy as np
import pytest
from astropy import units as u
from astropy.units import Quantity
from gwpy.types.index import Index

from gwexpy.types.axis import AxisDescriptor


def test_axis_descriptor_irregular():
    idx = Quantity([0, 1, 1.5, 3], "Hz")
    desc = AxisDescriptor("freq", idx)

    assert desc.name == "freq"
    assert desc.size == 4
    assert desc.unit == u.Hz
    assert not desc.regular
    assert desc.delta is None

def test_axis_descriptor_regular():
    idx = Index(np.linspace(0, 10, 101), unit="s")
    # depending on Index implementation, it might have .regular=True or we calculate it
    desc = AxisDescriptor("time", idx)

    assert desc.regular
    assert np.isclose(desc.delta.value, 0.1)
    assert desc.delta.unit == u.s

def test_iloc_nearest():
    idx = Quantity([0, 1, 2, 5, 10], "m")
    desc = AxisDescriptor("pos", idx)

    # 0 -> 0
    assert desc.iloc_nearest(0 * u.m) == 0
    # 1.4 -> 1 (diff 0.4 vs 2 is 0.6)
    assert desc.iloc_nearest(1.4 * u.m) == 1
    # 8 -> 4 (10) (diff 2 vs 5 is 3) or 3 (5)? 8-5=3, 10-8=2. So 10 is nearer.
    assert desc.iloc_nearest(8 * u.m) == 4

def test_iloc_slice():
    idx = Quantity([0, 10, 20, 30, 40], "s")
    desc = AxisDescriptor("time", idx)

    # slice(15, 35) -> 20, 30. Indices 2, 3?
    # searchsorted 'left': 15 -> idx 2 (20).
    # 35 -> idx 4 (40).
    # slice(2, 4) -> indices 2, 3. Values 20, 30. Correct.

    s = slice(15*u.s, 35*u.s)
    res = desc.iloc_slice(s)
    assert res.start == 2
    assert res.stop == 4
    assert res.step is None

def test_iloc_slice_irregular_step_error():
    idx = Quantity([0, 1, 4, 9], "m")
    desc = AxisDescriptor("pos", idx)

    with pytest.raises(ValueError):
        desc.iloc_slice(slice(0, 4, 1*u.m))
