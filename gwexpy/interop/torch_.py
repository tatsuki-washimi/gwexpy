"""
gwexpy.interop.torch_
----------------------

Interoperability with PyTorch tensors.
"""

from ._optional import require_optional
import numpy as np

__all__ = ["to_torch", "from_torch"]


def to_torch(series, device=None, dtype=None, requires_grad=False, copy=False):
    """
    Convert a series to a PyTorch tensor.

    Parameters
    ----------
    series : TimeSeries or array-like
        Input data.
    device : str or torch.device, optional
        Target device.
    dtype : torch.dtype, optional
        Target dtype.
    requires_grad : bool, optional
        Whether to track gradients.
    copy : bool, optional
        If True, always copy data; otherwise share memory if possible.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    torch = require_optional("torch")

    data = series
    if hasattr(data, "value"):
        data = data.value

    try:
        from astropy.units import Quantity
    except ImportError:
        Quantity = ()

    if Quantity and isinstance(data, Quantity):
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
    Create a TimeSeries from a PyTorch tensor.

    Parameters
    ----------
    cls : type
        TimeSeries class to instantiate.
    tensor : torch.Tensor
        Input tensor.
    t0 : Quantity or float
        Start time.
    dt : Quantity or float
        Sample interval.
    unit : str or Unit, optional
        Data unit.

    Returns
    -------
    TimeSeries
        The created time series.
    """
    data = tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
    return cls(data, t0=t0, dt=dt, unit=unit)
