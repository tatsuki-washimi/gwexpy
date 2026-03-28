"""Coverage patch for ScalarField and SignalField.
Covers attribute propagation, serialization, and complex transforms.
"""

import pickle
import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import ScalarField
from gwexpy.fields.signal import spectral_density, coherence_map


@pytest.fixture
def base_field():
    """Create a 4D ScalarField for testing."""
    nt, nx, ny, nz = 64, 4, 4, 4
    data = np.random.randn(nt, nx, ny, nz)
    t = np.arange(nt) * 0.1 * u.s
    x = np.arange(nx) * 1.0 * u.m
    y = np.arange(ny) * 1.0 * u.m
    z = np.arange(nz) * 1.0 * u.m
    field = ScalarField(
        data,
        unit=u.V,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real",
    )
    # Add custom attributes
    field._gwex_custom_attr = "top_secret"
    field._gwex_meta_info = {"version": "2.0", "status": "active"}
    return field


def test_scalarfield_attribute_propagation(base_field):
    """Test that _gwex_ attributes are preserved across various operations."""
    
    # 1. Slicing
    sliced = base_field[0:10, :, :, :]
    assert sliced._gwex_custom_attr == "top_secret"
    assert sliced._gwex_meta_info["version"] == "2.0"
    
    # 2. isel
    iseled = base_field.isel(x=0)
    assert iseled._gwex_custom_attr == "top_secret"
    
    # 3. fft_time
    fft_t = base_field.fft_time()
    # Note: currently ScalarField doesn't explicitly propagate in fft_time constructor call 
    # unless we fixed it in the base class finalize or the method itself.
    # Let's check if our SeriesMatrix fix handles this if ScalarField uses it.
    # ScalarField inherits FieldBase which might not inherit SeriesMatrix.
    assert hasattr(fft_t, "_gwex_custom_attr"), "Attribute lost in fft_time"
    assert fft_t._gwex_custom_attr == "top_secret"
    
    # 4. fft_space
    fft_s = base_field.fft_space(axes=["x"])
    assert fft_s._gwex_custom_attr == "top_secret"


def test_scalarfield_pickle_serialization(base_field):
    """Test pickle serialization for ScalarField."""
    
    # Real-space field
    dumped = pickle.dumps(base_field)
    loaded = pickle.loads(dumped)
    
    assert isinstance(loaded, ScalarField)
    assert_allclose(loaded.value, base_field.value)
    assert loaded.unit == base_field.unit
    assert loaded.axis0_domain == "time"
    assert loaded.space_domains == base_field.space_domains
    assert loaded._gwex_custom_attr == "top_secret"
    
    # K-space/Frequency-space field
    k_field = base_field.fft_time().fft_space()
    dumped_k = pickle.dumps(k_field)
    loaded_k = pickle.loads(dumped_k)
    
    assert loaded_k.axis0_domain == "frequency"
    assert "kx" in loaded_k.space_domains
    assert loaded_k.space_domains["kx"] == "k"
    assert loaded_k._gwex_custom_attr == "top_secret"


def test_scalarfield_complex_transforms(base_field):
    """Test composition of time and space transforms."""
    
    # FFT Time -> FFT Space
    field_tf = base_field.fft_time().fft_space(axes=["x", "y"])
    
    assert field_tf.axis0_domain == "frequency"
    assert field_tf.space_domains["kx"] == "k"
    assert field_tf.space_domains["ky"] == "k"
    assert field_tf.space_domains["z"] == "real"
    assert field_tf.axis_names == ("f", "kx", "ky", "z")
    
    # Reversibility in space
    recovered_space = field_tf.ifft_space(axes=["kx", "ky"])
    assert recovered_space.space_domains["x"] == "real"
    assert recovered_space.space_domains["y"] == "real"
    assert recovered_space.axis_names == ("f", "x", "y", "z")
    
    # Verify values match (at least shape and rough magnitude)
    assert recovered_space.shape == base_field.fft_time().shape


def test_signal_spectral_density_params(base_field):
    """Test spectral_density with various parameters."""
    
    # Welch with custom parameters
    res_welch = spectral_density(
        base_field, 
        axis=0, 
        method="welch", 
        nfft=32, 
        noverlap=8,  # samples
        window="hamming"
    )
    assert res_welch.axis0_domain == "frequency"
    assert res_welch.shape[0] == 17  # (32/2 + 1)
    
    # Spectral density along spatial axis
    res_spatial = spectral_density(base_field, axis="x", method="fft")
    assert res_spatial.space_domains["kx"] == "k"
    assert res_spatial.axis_names[1] == "kx"


def test_signal_coherence_map_resolved(base_field):
    """Test coherence_map without frequency banding."""
    
    # Frequency-resolved coherence map
    coh_map = coherence_map(
        base_field,
        ref_point=(0*u.m, 0*u.m, 0*u.m),
        plane="xy",
        at={"z": 0 * u.m},
        band=None,  # Do not average over frequency
        nfft=32
    )
    
    assert coh_map.axis0_domain == "frequency"
    assert coh_map.shape[0] == 17
    assert coh_map.shape[1] == base_field.shape[1]
    assert coh_map.unit == u.dimensionless_unscaled
    assert coh_map.axis_names[0] == "f"
