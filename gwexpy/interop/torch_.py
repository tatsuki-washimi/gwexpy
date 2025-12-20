
from ._optional import require_optional
import numpy as np

def to_torch(ts, device=None, dtype=None, requires_grad=False, copy=False):
    """
    Convert TimeSeries to torch.Tensor.
    """
    torch = require_optional("torch")
    
    # Use .value if ts is a Quantity (which it usually is in GWpy/gwexpy)
    # However, ts itself might be the TimeSeries/FrequencySeries object.
    # GWpy Series objects have a .value property (inherited from ndarray view).
    data = ts.value
    if hasattr(data, 'value'):
        # Just in case data is still an astropy Quantity
        data = data.value
    
    if copy:
        tensor = torch.tensor(data, device=device, dtype=dtype, requires_grad=requires_grad)
    else:
        tensor = torch.as_tensor(data, device=device, dtype=dtype)
        if requires_grad:
            tensor.requires_grad_(True)
            
    return tensor

def from_torch(cls, tensor, t0, dt, unit=None):
    """
    Create TimeSeries from torch.Tensor.
    Handles CPU/GPU and detach.
    """
    # Ensure tensor is on cpu, detached, numpy
    data = tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
    
    return cls(data, t0=t0, dt=dt, unit=unit)
