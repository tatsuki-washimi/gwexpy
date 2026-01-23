
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from astropy import units as u

from gwexpy.fields import ScalarField, VectorField, TensorField
from gwexpy.plot.field import FieldPlot
from matplotlib.animation import FuncAnimation

@pytest.fixture
def scalar_field():
    # 4D field: t, x, y, z
    data = np.random.randn(5, 4, 4, 4)
    times = np.arange(5) * u.s
    # Axis names default: t, x, y, z
    return ScalarField(data, axis0=times, unit=u.V)

@pytest.fixture
def vector_field(scalar_field):
    return VectorField({'x': scalar_field, 'y': scalar_field})

@pytest.fixture
def tensor_field(scalar_field):
    return TensorField({(0, 0): scalar_field, (1, 1): scalar_field}, rank=2)

def test_get_slice(scalar_field):
    # Test auto slice
    # get_slice tries to find 2 free axes.
    # 4 axes total. Need to fix 2.
    # If we provide no args, it should fix 2 axes automatically (likely idx 0).
    sliced, xidx, yidx, xname, yname = scalar_field.get_slice()
    assert sliced.ndim == 2
    assert sliced.unit == scalar_field.unit
    
    # Test specific slice
    sliced, xidx, yidx, xname, yname = scalar_field.get_slice(x_axis='x', y_axis='y', t=0, z=0)
    assert xname == 'x'
    assert yname == 'y'
    assert len(xidx) == 4
    assert len(yidx) == 4

def test_field_plot_scalar(scalar_field):
    fp = FieldPlot()
    mesh = fp.add_scalar(scalar_field, x='x', y='y', slice_kwargs={'t': 0, 'z': 0})
    assert mesh is not None
    plt.close(fp.figure)

def test_scalar_field_plot_method(scalar_field):
    fp = scalar_field.plot(x='x', y='y', t=0, z=0)
    assert isinstance(fp, FieldPlot)
    plt.close(fp.figure)

def test_vector_field_plots(vector_field):
    # Magnitude
    fp = vector_field.plot_magnitude(x='x', y='y', t=0, z=0)
    assert isinstance(fp, FieldPlot)
    plt.close(fp.figure)
    
    # Quiver
    fp = vector_field.quiver(x='x', y='y', t=0, z=0)
    assert isinstance(fp, FieldPlot)
    plt.close(fp.figure)
    
    # Streamline
    fp = vector_field.streamline(x='x', y='y', t=0, z=0)
    assert isinstance(fp, FieldPlot)
    plt.close(fp.figure)
    
    # Combined plot
    fp = vector_field.plot(x='x', y='y', t=0, z=0)
    assert isinstance(fp, FieldPlot)
    plt.close(fp.figure)

def test_tensor_field_plot(tensor_field):
    # We might need explicit rank? Fixture sets rank=2
    fp = tensor_field.plot_components(x='x', y='y', t=0, z=0)
    assert isinstance(fp, FieldPlot)
    plt.close(fp.figure)

def test_animate(scalar_field):
    ani = scalar_field.animate(x='x', y='y', axis='t', z=0)
    assert isinstance(ani, FuncAnimation)
