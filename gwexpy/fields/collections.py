"""Collections for ScalarField objects in the `gwexpy.fields` namespace."""

import numpy as np

from .scalar import ScalarField

__all__ = ["FieldList", "FieldDict"]

# Tolerance for axis coordinate comparison
_AXIS_RTOL = 1e-9
_AXIS_ATOL = 1e-12


class FieldList(list):
    """List-like collection for `ScalarField` objects with batch operations."""

    def __init__(self, items=None, validate=False):
        if items is None:
            items = []
        super().__init__(items)
        if validate:
            self._validate()

    def _validate(self):
        """Validate that all items are ScalarField with compatible metadata."""
        if not self:
            return

        first = self[0]
        if not isinstance(first, ScalarField):
            raise TypeError(f"Expected ScalarField, got {type(first)}")

        ref_unit = first.unit
        ref_axis_names = first.axis_names
        ref_axis0_domain = first.axis0_domain
        ref_space_domains = first.space_domains

        ref_axes = [
            first._axis0_index,
            first._axis1_index,
            first._axis2_index,
            first._axis3_index,
        ]

        for i, item in enumerate(self[1:], 1):
            if not isinstance(item, ScalarField):
                raise TypeError(f"Item {i}: Expected ScalarField, got {type(item)}")
            if item.unit != ref_unit:
                raise ValueError(
                    f"Item {i}: Inconsistent unit. "
                    f"Expected {ref_unit}, got {item.unit}"
                )
            if item.axis_names != ref_axis_names:
                raise ValueError(
                    f"Item {i}: Inconsistent axis_names. "
                    f"Expected {ref_axis_names}, got {item.axis_names}"
                )
            if item.axis0_domain != ref_axis0_domain:
                raise ValueError(
                    f"Item {i}: Inconsistent axis0_domain. "
                    f"Expected {ref_axis0_domain}, got {item.axis0_domain}"
                )
            if item.space_domains != ref_space_domains:
                raise ValueError(
                    f"Item {i}: Inconsistent space_domains. "
                    f"Expected {ref_space_domains}, got {item.space_domains}"
                )

            item_axes = [
                item._axis0_index,
                item._axis1_index,
                item._axis2_index,
                item._axis3_index,
            ]
            for ax_idx, (ref_ax, item_ax) in enumerate(zip(ref_axes, item_axes)):
                if ref_ax.shape != item_ax.shape:
                    raise ValueError(
                        f"Item {i}: Axis {ax_idx} shape mismatch. "
                        f"Expected {ref_ax.shape}, got {item_ax.shape}"
                    )
                if ref_ax.unit != item_ax.unit:
                    raise ValueError(
                        f"Item {i}: Axis {ax_idx} unit mismatch. "
                        f"Expected {ref_ax.unit}, got {item_ax.unit}"
                    )
                if not np.allclose(
                    ref_ax.value, item_ax.value, rtol=_AXIS_RTOL, atol=_AXIS_ATOL
                ):
                    raise ValueError(
                        f"Item {i}: Axis {ax_idx} coordinate mismatch. "
                        f"Axis values differ beyond tolerance."
                    )

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields, returning FieldList."""
        return self.__class__([f.fft_time(**kwargs) for f in self])

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields, returning FieldList."""
        return self.__class__([f.ifft_time(**kwargs) for f in self])

    def fft_space_all(self, axes=None, **kwargs):
        """Apply fft_space to all fields, returning FieldList."""
        return self.__class__([f.fft_space(axes=axes, **kwargs) for f in self])

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space to all fields, returning FieldList."""
        return self.__class__([f.ifft_space(axes=axes, **kwargs) for f in self])

    def resample_all(self, rate, **kwargs):
        """Apply resample to all fields, returning FieldList."""
        return self.__class__([f.resample(rate, **kwargs) for f in self])

    def filter_all(self, *args, **kwargs):
        """Apply filter to all fields, returning FieldList."""
        return self.__class__([f.filter(*args, **kwargs) for f in self])

    def sel_all(self, **kwargs):
        """Apply sel to all fields, returning FieldList."""
        return self.__class__([f.sel(**kwargs) for f in self])

    def isel_all(self, **kwargs):
        """Apply isel to all fields, returning FieldList."""
        return self.__class__([f.isel(**kwargs) for f in self])


class FieldDict(dict):
    """Dict-like collection for `ScalarField` objects with batch operations."""

    def __init__(self, items=None, validate=False):
        if items is None:
            items = {}
        super().__init__(items)
        if validate:
            self._validate()

    def copy(self) -> "FieldDict":
        """Return a copy of this FieldDict."""
        return self.__class__({k: v.copy() for k, v in self.items()})

    def __mul__(self, other):
        if np.isscalar(other):
            return self.__class__({k: v * other for k, v in self.items()})
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if np.isscalar(other):
            return self.__class__({k: v + other for k, v in self.items()})
        # Note: If other is FieldDict, we might want to Zip add?
        # But for now, stick to scalar as per plan Phase 2.
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            return self.__class__({k: v - other for k, v in self.items()})
        return NotImplemented

    def __rsub__(self, other):
        if np.isscalar(other):
            return self.__class__({k: other - v for k, v in self.items()})
        return NotImplemented

    def _validate(self):
        """Validate that all values are ScalarField with compatible metadata."""
        if not self:
            return

        values = list(self.values())
        first = values[0]
        if not isinstance(first, ScalarField):
            raise TypeError(f"Expected ScalarField, got {type(first)}")

        ref_unit = first.unit
        ref_axis_names = first.axis_names
        ref_axis0_domain = first.axis0_domain
        ref_space_domains = first.space_domains

        ref_axes = [
            first._axis0_index,
            first._axis1_index,
            first._axis2_index,
            first._axis3_index,
        ]

        for key, item in list(self.items())[1:]:
            if not isinstance(item, ScalarField):
                raise TypeError(f"Key '{key}': Expected ScalarField, got {type(item)}")
            if item.unit != ref_unit:
                raise ValueError(
                    f"Key '{key}': Inconsistent unit. "
                    f"Expected {ref_unit}, got {item.unit}"
                )
            if item.axis_names != ref_axis_names:
                raise ValueError(
                    f"Key '{key}': Inconsistent axis_names. "
                    f"Expected {ref_axis_names}, got {item.axis_names}"
                )
            if item.axis0_domain != ref_axis0_domain:
                raise ValueError(
                    f"Key '{key}': Inconsistent axis0_domain. "
                    f"Expected {ref_axis0_domain}, got {item.axis0_domain}"
                )
            if item.space_domains != ref_space_domains:
                raise ValueError(
                    f"Key '{key}': Inconsistent space_domains. "
                    f"Expected {ref_space_domains}, got {item.space_domains}"
                )

            item_axes = [
                item._axis0_index,
                item._axis1_index,
                item._axis2_index,
                item._axis3_index,
            ]
            for ax_idx, (ref_ax, item_ax) in enumerate(zip(ref_axes, item_axes)):
                if ref_ax.shape != item_ax.shape:
                    raise ValueError(
                        f"Key '{key}': Axis {ax_idx} shape mismatch. "
                        f"Expected {ref_ax.shape}, got {item_ax.shape}"
                    )
                if ref_ax.unit != item_ax.unit:
                    raise ValueError(
                        f"Key '{key}': Axis {ax_idx} unit mismatch. "
                        f"Expected {ref_ax.unit}, got {item_ax.unit}"
                    )
                if not np.allclose(
                    ref_ax.value, item_ax.value, rtol=_AXIS_RTOL, atol=_AXIS_ATOL
                ):
                    raise ValueError(
                        f"Key '{key}': Axis {ax_idx} coordinate mismatch. "
                        f"Axis values differ beyond tolerance."
                    )

    def fft_time_all(self, **kwargs):
        """Apply fft_time to all fields, returning FieldDict."""
        return self.__class__({k: v.fft_time(**kwargs) for k, v in self.items()})

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time to all fields, returning FieldDict."""
        return self.__class__({k: v.ifft_time(**kwargs) for k, v in self.items()})

    def fft_space_all(self, axes=None, **kwargs):
        """Apply fft_space to all fields, returning FieldDict."""
        return self.__class__({k: v.fft_space(axes=axes, **kwargs) for k, v in self.items()})

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space to all fields, returning FieldDict."""
        return self.__class__({k: v.ifft_space(axes=axes, **kwargs) for k, v in self.items()})

    def resample_all(self, rate, **kwargs):
        """Apply resample to all fields, returning FieldDict."""
        return self.__class__({k: v.resample(rate, **kwargs) for k, v in self.items()})

    def filter_all(self, *args, **kwargs):
        """Apply filter to all fields, returning FieldDict."""
        return self.__class__({k: v.filter(*args, **kwargs) for k, v in self.items()})

    def sel_all(self, **kwargs):
        """Apply sel to all fields, returning FieldDict."""
        return self.__class__({k: v.sel(**kwargs) for k, v in self.items()})

    def isel_all(self, **kwargs):
        """Apply isel to all fields, returning FieldDict."""
        return self.__class__({k: v.isel(**kwargs) for k, v in self.items()})
