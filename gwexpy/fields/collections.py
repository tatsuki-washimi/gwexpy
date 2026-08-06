"""Collections for ScalarField objects in the `gwexpy.fields` namespace."""

from __future__ import annotations

import copy
import operator

import numpy as np
from astropy import units as u

from .base import _axis_coordinates_close
from .scalar import ScalarField

__all__ = ["FieldList", "FieldDict"]


class _FieldCollectionArithmetic:
    """Fail-closed scalar, Quantity, and Unit arithmetic for collections."""

    # Make reflected Python operators win over Astropy Quantity's ndarray
    # dispatch. Collection-level ufuncs have no metadata-preserving contract.
    __array_priority__ = 10_001
    __array_ufunc__ = None

    def _new_from_components(self, components):
        raise NotImplementedError

    def _replace_components(self, components) -> None:
        raise NotImplementedError

    def _component_items(self):
        raise NotImplementedError

    @staticmethod
    def _is_numeric_scalar(value) -> bool:
        return np.isscalar(value) and not isinstance(value, (str, bytes))

    @staticmethod
    def _is_scalar_quantity(value) -> bool:
        return isinstance(value, u.Quantity) and value.isscalar

    @staticmethod
    def _is_unit(value) -> bool:
        return isinstance(value, u.UnitBase)

    def _require_operand(self, other, *, additive: bool) -> bool:
        is_scalar = self._is_numeric_scalar(other)
        is_quantity = self._is_scalar_quantity(other)
        is_unit = self._is_unit(other)
        if not (is_scalar or is_quantity or (is_unit and not additive)):
            return False
        if additive and is_scalar:
            for _, field in self._component_items():
                unit = field.unit or u.dimensionless_unscaled
                if not unit.is_equivalent(u.dimensionless_unscaled):
                    raise TypeError(
                        "bare scalar addition and subtraction require "
                        "dimensionless Field components"
                    )
        if additive and is_quantity:
            for _, field in self._component_items():
                unit = field.unit or u.dimensionless_unscaled
                if not unit.is_equivalent(other.unit):
                    raise u.UnitConversionError(
                        f"cannot add or subtract {other.unit} and {unit}"
                    )
        return True

    def _binary_operation(self, other, operation, *, reflected: bool, additive: bool):
        if not self._require_operand(other, additive=additive):
            return NotImplemented
        components = []
        for key, field in self._component_items():
            value = operation(other, field) if reflected else operation(field, other)
            self._detach_component_metadata(field, value)
            components.append((key, value))
        return self._new_from_components(components)

    @staticmethod
    def _detach_component_metadata(source, result) -> None:
        """Make operation-result metadata independent of its source component."""
        for axis in range(4):
            attribute = f"_axis{axis}_index"
            setattr(result, attribute, getattr(source, attribute).copy())
        result._axis_names = list(source.axis_names)
        result._axis0_domain = source.axis0_domain
        result._space_domains = dict(source.space_domains)
        result._axis0_offset = copy.deepcopy(source._axis0_offset)
        for attribute in ("name", "epoch", "channel"):
            if hasattr(source, attribute):
                setattr(result, attribute, copy.deepcopy(getattr(source, attribute)))
        for attribute, value in source.__dict__.items():
            if attribute.startswith("_gwex_"):
                setattr(result, attribute, copy.deepcopy(value))

    def _inplace_operation(self, other, operation, *, additive: bool):
        # Compute every component first.  A unit/domain error therefore leaves
        # the original collection and every component untouched.
        if not self._require_operand(other, additive=additive):
            raise TypeError(
                f"unsupported operand type for in-place operation: "
                f"{type(other).__name__}"
            )
        updated = self._binary_operation(
            other, operation, reflected=False, additive=additive
        )
        self._replace_components(updated._component_items())
        return self

    def __mul__(self, other):
        return self._binary_operation(
            other, operator.mul, reflected=False, additive=False
        )

    def __rmul__(self, other):
        return self._binary_operation(
            other, operator.mul, reflected=True, additive=False
        )

    def __truediv__(self, other):
        return self._binary_operation(
            other, operator.truediv, reflected=False, additive=False
        )

    def __rtruediv__(self, other):
        return self._binary_operation(
            other, operator.truediv, reflected=True, additive=False
        )

    def __add__(self, other):
        return self._binary_operation(
            other, operator.add, reflected=False, additive=True
        )

    def __radd__(self, other):
        return self._binary_operation(
            other, operator.add, reflected=True, additive=True
        )

    def __sub__(self, other):
        return self._binary_operation(
            other, operator.sub, reflected=False, additive=True
        )

    def __rsub__(self, other):
        return self._binary_operation(
            other, operator.sub, reflected=True, additive=True
        )

    def __imul__(self, other):
        return self._inplace_operation(other, operator.mul, additive=False)

    def __itruediv__(self, other):
        return self._inplace_operation(other, operator.truediv, additive=False)

    def __iadd__(self, other):
        return self._inplace_operation(other, operator.add, additive=True)

    def __isub__(self, other):
        return self._inplace_operation(other, operator.sub, additive=True)


class FieldList(_FieldCollectionArithmetic, list):
    """A list of ScalarField objects with batch operations.

    `FieldList` provides a container for multiple fields, supporting
    arithmetic and batch processing across the list.

    Parameters
    ----------
    items : list, optional
        A list of `ScalarField` objects.

    validate : bool, optional
        If True, ensures that all fields have matching axes.

    Examples
    --------
    >>> import numpy as np
    >>> from gwexpy.fields import ScalarField, FieldList
    >>> sf = ScalarField(np.ones((2, 2, 2, 2)))
    >>> fl = FieldList([sf])
    >>> fl
    [<ScalarField(2, 2, 2, 2)@time, 1.0>]

    """

    def __init__(self, items=None, validate=False):
        if items is None:
            items = []
        super().__init__(items)
        if validate:
            self._validate()

    def _component_items(self):
        return enumerate(self)

    def _new_from_components(self, components):
        return self.__class__([value for _, value in components], validate=False)

    def _replace_components(self, components) -> None:
        self[:] = [value for _, value in components]

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
            u_item = item.unit if item.unit is not None else u.dimensionless_unscaled
            u_ref = ref_unit if ref_unit is not None else u.dimensionless_unscaled
            if not u_item.is_equivalent(u_ref):
                raise ValueError(
                    f"Item {i}: Inconsistent unit. Expected equivalent to {ref_unit}, got {item.unit}"
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
                if not _axis_coordinates_close(ref_ax, item_ax):
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


class FieldDict(_FieldCollectionArithmetic, dict):
    """A dictionary of ScalarField objects with batch operations.

    `FieldDict` provides a labeled container for multiple fields,
    supporting arithmetic and batch processing.

    Parameters
    ----------
    items : dict, optional
        A dictionary mapping labels to `ScalarField` objects.

    validate : bool, optional
        If True, ensures that all fields have matching axes.

    Examples
    --------
    >>> import numpy as np
    >>> from gwexpy.fields import ScalarField, FieldDict
    >>> f = ScalarField(np.ones((2, 2, 2, 2)))
    >>> fd = FieldDict({'E': f})
    >>> fd
    {'E': <ScalarField(2, 2, 2, 2)@time, 1.0>}

    """

    def __init__(self, items=None, validate=False):
        if items is None:
            items = {}
        super().__init__(items)
        if validate:
            self._validate()

    def copy(self) -> FieldDict:
        """Return a copy of this FieldDict."""
        return self.__class__({k: v.copy() for k, v in self.items()})

    def _component_items(self):
        return self.items()

    def _new_from_components(self, components) -> FieldDict:
        """Return this collection's subclass with replacement components."""
        result = self.copy()
        result.clear()
        result.update(components)
        return result

    def _replace_components(self, components) -> None:
        self.clear()
        self.update(components)

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
            u_item = item.unit if item.unit is not None else u.dimensionless_unscaled
            u_ref = ref_unit if ref_unit is not None else u.dimensionless_unscaled
            if not u_item.is_equivalent(u_ref):
                raise ValueError(
                    f"Key '{key}': Inconsistent unit. "
                    f"Expected equivalent to {ref_unit}, got {item.unit}"
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
                if not _axis_coordinates_close(ref_ax, item_ax):
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
        return self.__class__(
            {k: v.fft_space(axes=axes, **kwargs) for k, v in self.items()}
        )

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space to all fields, returning FieldDict."""
        return self.__class__(
            {k: v.ifft_space(axes=axes, **kwargs) for k, v in self.items()}
        )

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
