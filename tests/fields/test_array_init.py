import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField, TensorField, VectorField


def test_vectorfield_array_init():
    # 5D array initialization
    arr = np.ones((2, 3, 4, 5, 3))
    v = VectorField(arr)

    # Check components created successfully (x, y, z)
    assert set(v.keys()) == {'x', 'y', 'z'}
    assert v['x'].shape == (2, 3, 4, 5)

    # Value check
    assert np.allclose(v['y'].value, 1.0)

    # Dictionary initialization should still work (backward compatibility)
    v2 = VectorField({'x': ScalarField(np.ones((2, 3, 4, 5)))})
    assert set(v2.keys()) == {'x'}

    # Invalid dimension -> ValueError
    with pytest.raises(ValueError, match="expected 5D array"):
        VectorField(np.ones((2, 3, 4, 5)))


def test_vectorfield_array_init_rejects_more_than_three_components():
    with pytest.raises(ValueError, match="supports 1, 2, or 3 components"):
        VectorField(np.ones((2, 3, 4, 5, 4)))

def test_tensorfield_array_init():
    # 6D array initialization
    arr = np.ones((2, 3, 4, 5, 3, 3))
    t = TensorField(arr)

    # Check rank inference
    assert t.rank == 2

    # Check component keys are valid (i, j)
    expected_keys = {(i, j) for i in range(3) for j in range(3)}
    assert set(t.keys()) == expected_keys

    assert t[(0, 1)].shape == (2, 3, 4, 5)

    # Value check
    assert np.allclose(t[(2, 2)].value, 1.0)

    # Dictionary initialization check (backward compatibility)
    t2 = TensorField({(0, 0): ScalarField(np.ones((2, 3, 4, 5)))})
    assert set(t2.keys()) == {(0, 0)}
    assert t2.rank == 2

    # Invalid dimension -> ValueError
    with pytest.raises(ValueError, match="expects 6D array"):
        TensorField(np.ones((2, 3, 4, 5, 3)))
