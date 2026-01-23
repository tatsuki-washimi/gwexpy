import pytest
import numpy as np
from astropy import units as u
from gwexpy.fields import ScalarField
from gwexpy.fields.tensor import TensorField

def test_tensorfield_init():
    data = np.zeros((10, 4, 4, 4))
    f11 = ScalarField(data + 1, unit=u.Pa)
    f22 = ScalarField(data + 2, unit=u.Pa)
    
    tf = TensorField({(0,0): f11, (1,1): f22}, rank=2)
    assert tf.rank == 2
    assert tf[(0,0)] is f11
    assert tf[(1,1)] is f22

def test_tensorfield_trace():
    data = np.zeros((10, 4, 4, 4))
    f11 = ScalarField(data + 1, unit=u.Pa)
    f22 = ScalarField(data + 2, unit=u.Pa)
    f33 = ScalarField(data + 3, unit=u.Pa)
    
    tf = TensorField({(0,0): f11, (1,1): f22, (2,2): f33}, rank=2)
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
    
    tf = TensorField({(0,1): f01, (1,0): f10}, rank=2)
    tf_sym = tf.symmetrize()
    
    assert np.all(tf_sym[(0,1)].value == 2)
    assert np.all(tf_sym[(1,0)].value == 2)

def test_tensorfield_matmul_vector():
    from gwexpy.fields.vector import VectorField
    f_1 = ScalarField(np.ones((10, 4, 4, 4)))
    # Matrix [[1, 2], [0, 1]]
    t = TensorField({
        (0, 0): f_1,
        (0, 1): f_1 * 2,
        (1, 0): f_1 * 0,
        (1, 1): f_1
    }, rank=2)
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
    a = TensorField({
        (0, 0): f_1, (0, 1): f_1,
        (1, 0): f_1 * 0, (1, 1): f_1
    }, rank=2)
    # B = [[1, 0], [1, 1]]
    b = TensorField({
        (0, 0): f_1, (0, 1): f_1 * 0,
        (1, 0): f_1, (1, 1): f_1
    }, rank=2)
    
    # A @ B = [[1*1 + 1*1, 1*0 + 1*1], [0*1 + 1*1, 0*0 + 1*1]] = [[2, 1], [1, 1]]
    res = a @ b
    assert isinstance(res, TensorField)
    assert np.all(res[(0, 0)].value == 2)
    assert np.all(res[(0, 1)].value == 1)
    assert np.all(res[(1, 1)].value == 1)

def test_tensorfield_det():
    f_1 = ScalarField(np.ones((10, 4, 4, 4)), unit=u.m)
    # A = [[2, 0], [0, 3]] -> det = 6
    a = TensorField({
        (0, 0): f_1 * 2, (0, 1): f_1 * 0,
        (1, 0): f_1 * 0, (1, 1): f_1 * 3
    }, rank=2)
    
    d = a.det()
    assert isinstance(d, ScalarField)
    assert np.all(d.value == 6)
    assert d.unit == u.m**2
