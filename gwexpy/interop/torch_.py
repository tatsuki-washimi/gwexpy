"""gwexpy.interop.torch_
----------------------

Interoperability with PyTorch tensors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

from ._optional import require_optional

if TYPE_CHECKING:
    import torch

    from gwexpy.timeseries import TimeSeries

T = TypeVar("T", bound="TimeSeries")

__all__ = ["to_torch", "from_torch"]


def to_torch(
    series: Union[TimeSeries, Any],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False,
    copy: bool = False,
) -> torch.Tensor:
    """Convert a series to a PyTorch tensor.

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

    from .base import to_plain_array

    data = to_plain_array(series, copy=copy)

    if copy:
        tensor = torch.tensor(
            data, device=device, dtype=dtype, requires_grad=requires_grad
        )
    else:
        tensor = torch.as_tensor(data, device=device, dtype=dtype)
        if requires_grad:
            tensor.requires_grad_(True)

    return tensor


def from_torch(
    cls: type[T],
    tensor: torch.Tensor,
    t0: Any,
    dt: Any,
    unit: Optional[str] = None,
) -> T:
    """Create a TimeSeries from a PyTorch tensor.

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
    # Safe handle for conjugate and negative resolving which might not be present in all versions
    # or relevant for all tensor types.
    t = tensor.detach().cpu()
    if hasattr(t, "resolve_conj"):
        t = t.resolve_conj()
    if hasattr(t, "resolve_neg"):
        t = t.resolve_neg()
    data = t.numpy()
    return cls(data, t0=t0, dt=dt, unit=unit)
