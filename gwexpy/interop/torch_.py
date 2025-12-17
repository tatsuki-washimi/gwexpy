
from ._optional import require_optional
import numpy as np

def to_torch(ts, device=None, dtype=None, requires_grad=False, copy=False):
    """
    Convert TimeSeries to torch.Tensor.
    """
    torch = require_optional("torch")
    
    data = ts.value
    # torch.tensor(data) copies by default? 
    # torch.as_tensor(data) avoids copy if possible.
    # Spec says: copy=False (default).
    
    if copy:
        tensor = torch.tensor(data, device=device, dtype=dtype, requires_grad=requires_grad)
    else:
        # as_tensor doesn't support requires_grad directly in all versions, 
        # but safely creates tensor sharing memory if possible (NumPy bridge).
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
