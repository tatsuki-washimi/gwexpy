from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.fields.tensor import TensorField

TensorFieldUnaryOp = Callable[[TensorField], TensorField]
_INV_METHOD: TensorFieldUnaryOp | None = getattr(TensorField, "inv", None)
_ANTISYM_METHOD: TensorFieldUnaryOp | None = getattr(TensorField, "antisymmetrize", None)


def test_tensorfield_init():
    data = np.zeros((10, 4, 4, 4))
    f11 = ScalarField(data + 1, unit=u.Pa)
    f22 = ScalarField(data + 2, unit=u.Pa)

    components_init: dict[tuple[int, ...], ScalarField] = {(0, 0): f11, (1, 1): f22}
    tf = TensorField(components_init, rank=2)
    assert tf.rank == 2
    assert tf[(0,0)] is f11
    assert tf[(1,1)] is f22

def test_tensorfield_trace():
    data = np.zeros((10, 4, 4, 4))
    f11 = ScalarField(data + 1, unit=u.Pa)
    f22 = ScalarField(data + 2, unit=u.Pa)
    f33 = ScalarField(data + 3, unit=u.Pa)

    trace_components: dict[tuple[int, ...], ScalarField] = {
        (0, 0): f11,
        (1, 1): f22,
        (2, 2): f33,
    }
    tf = TensorField(trace_components, rank=2)
    tr = tf.trace()

    assert isinstance(tr, ScalarField)
    # 1 + 2 + 3 = 6
    assert np.all(tr.value == 6)
    assert tr.unit == u.Pa

def test_tensorfield_symmetrize():
    data = np.zeros((10, 4, 4, 4))
    # T_01 = 1, T_10 = 3 => S_01 = S_10 = (1 + 3) / 2 = 2
    f01 = ScalarField(data + 1, unit=u.Pa)
    f10 = ScalarField(data + 3, unit=u.Pa)

    sym_components: dict[tuple[int, ...], ScalarField] = {(0, 1): f01, (1, 0): f10}
    tf = TensorField(sym_components, rank=2)
    tf_sym = tf.symmetrize()

    assert np.all(tf_sym[(0,1)].value == 2)
    assert np.all(tf_sym[(1,0)].value == 2)

def test_tensorfield_matmul_vector():
    from gwexpy.fields.vector import VectorField
    f_1 = ScalarField(np.ones((10, 4, 4, 4)))
    # Matrix [[1, 2], [0, 1]]
    t_components: dict[tuple[int, ...], ScalarField] = {
        (0, 0): f_1,
        (0, 1): cast(ScalarField, f_1 * 2),
        (1, 0): cast(ScalarField, f_1 * 0),
        (1, 1): f_1,
    }
    t = TensorField(t_components, rank=2)
    # Vector [1, 1]
    v = VectorField({'x': f_1, 'y': f_1})

    # [1*1 + 2*1, 0*1 + 1*1] = [3, 1]
    res = t @ v
    assert isinstance(res, VectorField)
    assert np.all(res['x'].value == 3)
    assert np.all(res['y'].value == 1)

def test_tensorfield_matmul_tensor():
    f_1 = ScalarField(np.ones((10, 4, 4, 4)))
    # A = [[1, 1], [0, 1]]
    a_components: dict[tuple[int, ...], ScalarField] = {
        (0, 0): f_1,
        (0, 1): f_1,
        (1, 0): cast(ScalarField, f_1 * 0),
        (1, 1): f_1,
    }
    a = TensorField(a_components, rank=2)
    # B = [[1, 0], [1, 1]]
    b_components: dict[tuple[int, ...], ScalarField] = {
        (0, 0): f_1,
        (0, 1): cast(ScalarField, f_1 * 0),
        (1, 0): f_1,
        (1, 1): f_1,
    }
    b = TensorField(b_components, rank=2)

    # A @ B = [[1*1 + 1*1, 1*0 + 1*1], [0*1 + 1*1, 0*0 + 1*1]] = [[2, 1], [1, 1]]
    res = a @ b
    assert isinstance(res, TensorField)
    assert np.all(res[(0, 0)].value == 2)
    assert np.all(res[(0, 1)].value == 1)
    assert np.all(res[(1, 1)].value == 1)

def test_tensorfield_det():
    f_1 = ScalarField(np.ones((10, 4, 4, 4)), unit=u.m)
    # A = [[2, 0], [0, 3]] -> det = 6
    det_components: dict[tuple[int, ...], ScalarField] = {
        (0, 0): cast(ScalarField, f_1 * 2),
        (0, 1): cast(ScalarField, f_1 * 0),
        (1, 0): cast(ScalarField, f_1 * 0),
        (1, 1): cast(ScalarField, f_1 * 3),
    }
    a = TensorField(det_components, rank=2)

    d = a.det()
    assert isinstance(d, ScalarField)
    assert np.all(d.value == 6)
    assert d.unit == u.m**2


def test_tensorfield_det_3x3_metadata_units():
    axis0 = np.linspace(0, 1, 5) * u.s
    base = ScalarField(np.ones((5, 2, 2, 2)), axis0=axis0, unit=u.m)

    tf_components: dict[tuple[int, ...], ScalarField] = {
        (0, 0): cast(ScalarField, base * 2),
        (1, 1): cast(ScalarField, base * 3),
        (2, 2): cast(ScalarField, base * 4),
    }
    tf = TensorField(tf_components, rank=2)

    det = tf.det()
    assert isinstance(det, ScalarField)
    assert np.allclose(det.value, 24)
    assert det.unit == u.m**3
    assert det.axis0_domain == base.axis0_domain
    assert det.space_domains == base.space_domains


@pytest.mark.skipif(_INV_METHOD is None, reason="TensorField.inv is not implemented yet")
def test_tensorfield_inverse_diagonal_units():
    axis0 = np.linspace(0, 1, 3) * u.s
    base = ScalarField(np.ones((3, 1, 1, 1)), axis0=axis0, unit=u.m)
    diag_components: dict[tuple[int, ...], ScalarField] = {
        (0, 0): cast(ScalarField, base * 2),
        (1, 1): cast(ScalarField, base * 3),
        (2, 2): cast(ScalarField, base * 4),
    }
    diag = TensorField(diag_components, rank=2)

    inv_method = _INV_METHOD
    assert inv_method is not None
    inv_tensor = inv_method(diag)
    assert isinstance(inv_tensor, TensorField)
    assert np.allclose(inv_tensor[(0, 0)].value, 0.5)
    assert np.allclose(inv_tensor[(1, 1)].value, 1 / 3)
    assert np.allclose(inv_tensor[(2, 2)].value, 0.25)
    assert inv_tensor[(0, 0)].unit == u.m ** -1
    assert inv_tensor[(1, 1)].unit == u.m ** -1
    assert inv_tensor[(2, 2)].unit == u.m ** -1
    assert inv_tensor[(0, 0)].axis0_domain == base.axis0_domain


@pytest.mark.skipif(
    _ANTISYM_METHOD is None,
    reason="TensorField.antisymmetrize is not implemented yet",
)
def test_tensorfield_antisymmetrize_behavior():
    axis0 = np.linspace(0, 1, 2) * u.s
    base = ScalarField(np.ones((2, 1, 1, 1)), axis0=axis0, unit=u.Pa)

    antisym_components: dict[tuple[int, ...], ScalarField] = {
        (0, 1): cast(ScalarField, base * 2),
        (1, 0): cast(ScalarField, base * -3),
    }
    tf = TensorField(antisym_components, rank=2)

    antisym_method = _ANTISYM_METHOD
    assert antisym_method is not None
    antisym = antisym_method(tf)
    assert np.allclose(antisym[(0, 1)].value, 2.5)
    assert np.allclose(antisym[(1, 0)].value, -2.5)
    assert antisym[(0, 1)].unit == u.Pa
    assert antisym[(1, 0)].unit == u.Pa

def test_tensorfield_errors_rank():
    f = ScalarField(np.ones((10, 4, 4, 4)), unit=u.Pa)
    tf1 = TensorField({(0,): f}, rank=1)

    with pytest.raises(ValueError, match="only defined for rank-2"):
        tf1.trace()

    with pytest.raises(ValueError, match="is only defined for rank-2"):
        tf1 @ tf1

    with pytest.raises(ValueError, match="only defined for rank-2"):
        tf1.det()

    with pytest.raises(ValueError, match="only defined for rank-2"):
        tf1.symmetrize()


def test_tensorfield_symmetrize_missing_component():
    # Only T_01 provided, T_10 missing
    f = ScalarField(np.ones((10, 4, 4, 4)))
    tf = TensorField({(0, 1): f}, rank=2)

    # S_01 = (T_01 + 0) / 2 = 0.5
    # S_10 = (0 + T_01) / 2 = 0.5
    ts = tf.symmetrize()
    assert np.all(ts[(0, 1)].value == 0.5)
    assert np.all(ts[(1, 0)].value == 0.5)


def test_tensorfield_to_array_order():
    f = ScalarField(np.ones((10, 4, 4, 4)))
    tf = TensorField({(0, 0): cast(ScalarField, f * 1), (1, 1): cast(ScalarField, f * 2)}, rank=2)

    # Order 'last' (default)
    arr_last = tf.to_array(order="last")
    assert arr_last.shape == (10, 4, 4, 4, 2, 2)
    assert arr_last[0, 0, 0, 0, 0, 0] == 1
    assert arr_last[0, 0, 0, 0, 1, 1] == 2

    # Order 'first'
    arr_first = tf.to_array(order="first")
    assert arr_first.shape == (2, 2, 10, 4, 4, 4)
    assert arr_first[0, 0, 0, 0, 0, 0] == 1
    assert arr_first[1, 1, 0, 0, 0, 0] == 2
