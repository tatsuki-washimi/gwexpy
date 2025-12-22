import numpy as np
from astropy.units import Quantity

def to_plain_array(data, copy=False):
    """
    Extract a plain numpy array from various wrappers (TimeSeries, Quantity, etc).
    """
    if hasattr(data, "value"):
        data = data.value
    
    if isinstance(data, Quantity):
        data = data.value
        
    return np.array(data, copy=copy)

def from_plain_array(cls, array, t0, dt, unit=None, **kwargs):
    """
    Reconstruct a gwexpy object from a plain array.
    """
    # Ensure data is numpy
    if hasattr(array, "numpy"): # torch/tf
         array = array.numpy()
    elif hasattr(array, "get"): # cupy
         array = array.get()
         
    return cls(array, t0=t0, dt=dt, unit=unit, **kwargs)
