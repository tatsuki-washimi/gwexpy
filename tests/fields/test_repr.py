# tests/fields/test_repr.py
import re
import numpy as np
from gwexpy.fields import (
    ScalarField,
    VectorField,
    TensorField,
    FieldList,
    FieldDict,
)

RE_SHAPE = r"\(\s*\d+(?:\s*,\s*\d+)*\s*\)"

def assert_has_repr_shape_and_domain(obj, clsname):
    r = repr(obj)
    assert f"<{clsname}" in r, f"{clsname} missing in repr: {r}"
    assert re.search(RE_SHAPE, r), f"shape pattern missing in repr: {r}"
    assert "@" in r, f"domain marker '@' missing in repr: {r}"

def test_scalarfield_repr():
    data = np.ones((2, 2, 2, 2))
    sf = ScalarField(data)
    assert_has_repr_shape_and_domain(sf, "ScalarField")
    # check concrete shape is included
    assert re.search(r"\(\s*2\s*,\s*2\s*,\s*2\s*,\s*2\s*\)", repr(sf))

def test_vectorfield_repr():
    v = VectorField(np.ones((2, 2, 2, 2, 3)))
    assert_has_repr_shape_and_domain(v, "VectorField")
    assert re.search(r"\(\s*2\s*,\s*2\s*,\s*2\s*,\s*2\s*,\s*3\s*\)", repr(v))

def test_tensorfield_repr():
    t = TensorField(np.ones((2, 2, 2, 2, 3, 3)))
    assert_has_repr_shape_and_domain(t, "TensorField")
    assert re.search(r"\(\s*2\s*,\s*2\s*,\s*2\s*,\s*2\s*,\s*3\s*,\s*3\s*\)", repr(t))

def test_fieldlist_fielddict_repr():
    sf = ScalarField(np.ones((2, 2, 2, 2)))
    fl = FieldList([sf])
    fd = FieldDict({"E": sf})
    # repr of containers should include a ScalarField repr inside
    assert "<ScalarField" in repr(fl)
    assert "<ScalarField" in repr(fd)
