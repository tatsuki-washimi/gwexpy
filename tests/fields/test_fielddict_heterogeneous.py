"""Tests for #521 heterogeneous FieldDict and #520 standardized Field I/O dispatch."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, TensorField, VectorField, list_fields
from gwexpy.fields.io import register_field_format


def _make_heterogeneous_fields():
    t4 = np.arange(4, dtype=float) * 0.5 * u.s
    t8 = np.arange(8, dtype=float) * 0.25 * u.s
    sf = ScalarField(
        np.ones((4, 2, 2, 1)) * 3.0,
        unit=u.m,
        axis0=t4,
        axis1=np.array([0.0, 1.0]) * u.m,
        axis2=np.array([0.0, 1.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
        name="disp",
    )
    vf_x = ScalarField(
        np.ones((8, 3, 2, 1)) * 2.0,
        unit=u.V,
        axis0=t8,
        axis1=np.array([0.0, 1.0, 2.0]) * u.m,
        axis2=np.array([0.0, 1.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    vf = VectorField({"x": vf_x})
    tf = TensorField({"xx": sf * (1.0 * u.Pa / u.m)})
    return sf, vf, tf


def test_fielddict_accepts_heterogeneous_fields_and_preserves_order():
    sf, vf, tf = _make_heterogeneous_fields()
    fd = FieldDict([("zeta", sf), ("alpha", vf), ("sigma", tf)])
    assert list(fd.keys()) == ["zeta", "alpha", "sigma"]
    assert isinstance(fd["zeta"], ScalarField)
    assert isinstance(fd["alpha"], VectorField)
    assert isinstance(fd["sigma"], TensorField)
    assert fd["zeta"].unit == u.m
    assert fd["alpha"]["x"].unit == u.V
    assert fd["sigma"]["xx"].unit == u.Pa


def test_fielddict_rejects_non_field_values_on_all_mutators():
    sf, _, _ = _make_heterogeneous_fields()
    with pytest.raises(TypeError, match="Expected a GWexpy field"):
        FieldDict({"bad": np.ones((2, 2))})  # type: ignore[dict-item]

    fd = FieldDict({"ok": sf})
    with pytest.raises(TypeError):
        fd["bad"] = 42  # type: ignore[assignment]
    with pytest.raises(TypeError):
        fd.update({"bad": [1, 2, 3]})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        fd.setdefault("bad", "not a field")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fd |= {"bad": object()}  # type: ignore[operator]


def test_fielddict_explicit_validation_and_backward_compat():
    sf, vf, _ = _make_heterogeneous_fields()
    fd = FieldDict({"sf": sf, "vf": vf})

    # Heterogeneous shapes fail explicit alignment check
    with pytest.raises(ValueError, match="mismatch"):
        fd.validate_alignment()

    # Compatible pair passes validate_alignment()
    fd_aligned = FieldDict({"a": sf, "b": sf * 2.0 * (u.V / u.m)})
    fd_aligned.validate_alignment()

    # Grouping by physical units
    fd_units = FieldDict({"dx": sf, "dy": sf.copy(), "ex": vf["x"]})
    fd_units.validate_units(groups=[["dx", "dy"], ["ex"]])
    with pytest.raises(ValueError, match="Inconsistent unit"):
        fd_units.validate_units(groups=[["dx", "ex"]])
    with pytest.raises(ValueError, match="Inconsistent unit"):
        fd_units.validate_units()

    # Legacy strict validate=True continues to reject mismatched units
    with pytest.raises(ValueError):
        FieldDict({"dx": sf, "ex": vf["x"]}, validate=True)


def test_fielddict_heterogeneous_arithmetic_and_batch_ops():
    t = np.arange(8, dtype=float) * 0.1 * u.s
    x = np.array([0.0, 1.0]) * u.m
    y = np.array([0.0, 1.0]) * u.m
    z = np.array([0.0]) * u.m
    sf = ScalarField(
        np.cos(2 * np.pi * 2.0 * t.value)[:, None, None, None] * np.ones((1, 2, 2, 1)),
        unit=u.m,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
    )
    vf = VectorField({"x": sf * 2.0})
    fd = FieldDict({"s": sf, "v": vf})

    scaled = fd * 3.0
    assert scaled["s"].value[0, 0, 0, 0] == pytest.approx(3.0)
    assert scaled["v"]["x"].value[0, 0, 0, 0] == pytest.approx(6.0)

    freq_fd = fd.fft_time_all()
    assert isinstance(freq_fd["s"], ScalarField)
    assert isinstance(freq_fd["v"], VectorField)
    assert freq_fd["s"].axis0_domain == "frequency"
    assert freq_fd["v"]["x"].axis0_domain == "frequency"


def test_standardized_field_io_dispatch_contract():
    fmt = "_test_field_io_contract_tmp"
    t = np.arange(6, dtype=float) * u.s
    f1 = ScalarField(
        np.arange(6, dtype=float).reshape(6, 1, 1, 1), unit=u.m, axis0=t, name="a"
    )
    f2 = ScalarField(
        (np.arange(6, dtype=float) * 2).reshape(6, 1, 1, 1), unit=u.V, axis0=t, name="b"
    )
    vf = VectorField({"x": f1})
    tf = TensorField({"xx": f2})
    store = {"a": f1, "b": f2}

    def _lister(src, **kw):
        return list(store.keys())

    def _read_scalar(src, name=None, **kw):
        key = name or "a"
        if key not in store:
            raise KeyError(key)
        return store[key].copy()

    def _read_vector(src, name=None, **kw):
        return vf.copy()

    def _read_tensor(src, name=None, **kw):
        return tf.copy()

    def _read_dict(src, names=None, **kw):
        if names is None:
            keys = list(store.keys())
        elif isinstance(names, str):
            keys = [names]
        else:
            keys = list(names)
        missing = [k for k in keys if k not in store]
        if missing:
            raise KeyError(missing)
        return FieldDict([(k, store[k].copy()) for k in keys])

    register_field_format(
        fmt,
        reader_scalar=_read_scalar,
        reader_vector=_read_vector,
        reader_tensor=_read_tensor,
        reader_dict=_read_dict,
        lister=_lister,
    )

    assert list_fields("mem://x", format=fmt) == ["a", "b"]
    got_s = ScalarField.read("mem://x", "b", format=fmt, start=1 * u.s, end=4 * u.s)
    assert got_s.shape[0] == 4
    assert got_s._axis0_index.value[0] == pytest.approx(1.0)
    assert got_s._axis0_index.value[-1] == pytest.approx(4.0)
    assert got_s.unit == u.V

    with pytest.raises(KeyError):
        ScalarField.read("mem://x", "nonexistent", format=fmt)

    assert isinstance(VectorField.read("mem://x", format=fmt), VectorField)
    assert isinstance(TensorField.read("mem://x", format=fmt), TensorField)

    got_d = FieldDict.read("mem://x", ["b", "a"], format=fmt)
    assert list(got_d.keys()) == ["b", "a"]

    for forbidden in ("field", "fields", "profile", "times", "grid_mode", "selectors"):
        with pytest.raises(TypeError, match="Unsupported Field I/O"):
            ScalarField.read("mem://x", format=fmt, **{forbidden: "x"})
