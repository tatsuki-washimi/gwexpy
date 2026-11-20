"""Collections for physical field objects in the `gwexpy.fields` namespace."""

from __future__ import annotations

import copy
import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from astropy import units as u

from ..types.axis import AxisDescriptor
from .base import FieldBase, _axis_coordinates_close
from .scalar import ScalarField

__all__ = ["FieldList", "FieldDict"]


def _is_field_value(value: Any) -> bool:
    """Return True when *value* is a GWexpy physical field object."""
    if isinstance(value, FieldBase):
        return True
    return bool(getattr(type(value), "_is_gwexpy_field", False))


def _iter_leaf_scalar_fields(key: Any, value: Any) -> Iterable[tuple[str, Any]]:
    """Yield ``(label, scalar_field)`` pairs from scalar or dict-based fields."""
    if isinstance(value, FieldBase):
        yield (repr(key), value)
        return
    if isinstance(value, dict):
        for comp_key, comp_val in value.items():
            yield from _iter_leaf_scalar_fields(f"{key!r}[{comp_key!r}]", comp_val)
        return
    raise TypeError(
        f"Key {key!r}: field of type {type(value).__name__} does not expose "
        "4D rectilinear axes for alignment/unit inspection."
    )


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
            for key, field in self._component_items():
                for _, leaf in _iter_leaf_scalar_fields(key, field):
                    unit = leaf.unit or u.dimensionless_unscaled
                    if not unit.is_equivalent(u.dimensionless_unscaled):
                        raise TypeError(
                            "bare scalar addition and subtraction require "
                            "dimensionless Field components"
                        )
        if additive and is_quantity:
            for key, field in self._component_items():
                for _, leaf in _iter_leaf_scalar_fields(key, field):
                    unit = leaf.unit or u.dimensionless_unscaled
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
            if isinstance(field, FieldBase) and isinstance(value, FieldBase):
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
    """A labeled dictionary of physical field objects with batch operations.

    ``FieldDict`` stores heterogeneous GWexpy physical fields
    (:class:`~gwexpy.fields.ScalarField`, :class:`~gwexpy.fields.VectorField`,
    :class:`~gwexpy.fields.TensorField`, and compatible mesh-aware field
    types), preserving insertion order while allowing distinct physical units,
    shapes, and coordinate grids by default.

    Explicit :meth:`validate_alignment` and :meth:`validate_units` methods are
    provided when uniform grid alignment or grouped unit consistency is
    required. Passing ``validate=True`` preserves strict homogeneous
    ``ScalarField`` validation for backward compatibility.

    Parameters
    ----------
    items : mapping or iterable of (key, value), optional
        Initial labeled fields. Every value must be a GWexpy field object.
    validate : bool, optional
        If ``True``, enforce strict homogeneous ``ScalarField`` shape, axis,
        domain, and unit equality. Default is ``False``.

    """

    def __init__(self, items=None, validate: bool = False):
        if items is None:
            items = {}
        super().__init__(items)
        if validate:
            self._validate()
        else:
            for key, value in self.items():
                self._validate_entry(key, value)

    @staticmethod
    def _validate_entry(key: Any, value: Any) -> None:
        if not _is_field_value(value):
            raise TypeError(
                f"Key {key!r}: Expected a GWexpy field object "
                f"(ScalarField, VectorField, or TensorField), "
                f"got {type(value).__name__}."
            )

    def __setitem__(self, key: Any, value: Any) -> None:
        self._validate_entry(key, value)
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update dictionary in place after validating all values are field objects."""
        pending = dict(*args, **kwargs)
        for key, value in pending.items():
            self._validate_entry(key, value)
        super().update(pending)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Return value for *key* if present, else validate and insert *default*."""
        if key in self:
            return self[key]
        self._validate_entry(key, default)
        return super().setdefault(key, default)

    def __ior__(self, other: Any) -> FieldDict:  # type: ignore[override,misc]
        """Merge *other* into this :class:`FieldDict` after validating values."""
        self.update(other)
        return self

    @classmethod
    def read(
        cls,
        source: Any,
        names: Sequence[str] | str | None = None,
        *,
        format: str | None = None,
        start: Any = None,
        end: Any = None,
        **kwargs: Any,
    ) -> FieldDict:
        """Read multiple named physical fields from *source* into a :class:`FieldDict`.

        Parameters
        ----------
        source : str, Path, file-like, or sequence of paths
            Input data source.
        names : sequence of str or str
            Field names to load. Returned keys preserve the requested order.
        format : str, optional
            Format identifier.
        start, end : optional
            Continuous range selection along axis 0 where supported.
        **kwargs
            Format-specific reader options.

        """
        from .io.core import read_fielddict

        return read_fielddict(
            source,
            names,
            format=format,
            start=start,
            end=end,
            **kwargs,
        )

    def write(
        self,
        target: Any,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write this :class:`FieldDict` to *target*.

        Parameters
        ----------
        target : str, Path, file-like, or h5py.Group
            Destination file path, stream, or HDF5 group.
        format : str, optional
            Output format identifier (e.g. ``'hdf5'``).
        **kwargs
            Format-specific writer options.

        """
        from .io.core import write_fielddict

        return write_fielddict(self, target, format=format, **kwargs)

    def copy(self) -> FieldDict:
        """Return a copy of this FieldDict preserving key order and field metadata."""
        return self.__class__({k: v.copy() for k, v in self.items()})

    def _component_items(self):
        return self.items()

    def _new_from_components(self, components) -> FieldDict:
        """Return this collection's subclass with replacement components."""
        if type(self) is FieldDict:
            return self.__class__(dict(components))
        result = self.copy()
        result.clear()
        result.update(components)
        return result

    def _replace_components(self, components) -> None:
        items = list(components)
        for key, value in items:
            self._validate_entry(key, value)
        self.clear()
        super().update(items)

    def validate_alignment(
        self,
        *,
        axis0_only: bool = False,
        require_regular: bool = False,
        strict_metadata: bool = False,
    ) -> None:
        """Validate coordinate-grid alignment across contained fields.

        Parameters
        ----------
        axis0_only : bool, default False
            When ``True``, check only axis 0 domain, length, coordinates, and
            time-reference compatibility (allowing distinct spatial grids).
        require_regular : bool, default False
            When ``True``, also require every checked axis to pass
            :attr:`AxisDescriptor.regular`.
        strict_metadata : bool, default False
            When ``True``, also check matching ``epoch``, ``axis_names`` (even
            when ``axis0_only=True``), and ``geo.crs`` when present.

        Raises
        ------
        ValueError
            If any contained field violates the requested alignment contract.
        TypeError
            If any contained field lacks rectilinear 4D axes.

        """
        if not self:
            return

        leaves: list[tuple[str, Any]] = []
        for key, field in self.items():
            leaves.extend(_iter_leaf_scalar_fields(key, field))
        if not leaves:
            return

        axes_to_check = (0,) if axis0_only else (0, 1, 2, 3)

        if require_regular:
            for label, leaf in leaves:
                for ax_idx in axes_to_check:
                    ax_name = leaf.axis_names[ax_idx]
                    ax_index = getattr(leaf, f"_axis{ax_idx}_index")
                    if not AxisDescriptor(ax_name, ax_index).regular:
                        raise ValueError(
                            f"Field {label}: Axis {ax_idx} ('{ax_name}') is not "
                            "regularly spaced."
                        )

        ref_label, ref = leaves[0]
        for label, item in leaves[1:]:
            if item.time_reference != ref.time_reference:
                raise ValueError(
                    f"Field {label}: time_reference mismatch vs {ref_label} "
                    f"({item.time_reference} vs {ref.time_reference})."
                )
            if item.axis0_domain != ref.axis0_domain:
                raise ValueError(
                    f"Field {label}: Inconsistent axis0_domain vs {ref_label}. "
                    f"Expected {ref.axis0_domain!r}, got {item.axis0_domain!r}."
                )
            if strict_metadata and item.epoch != ref.epoch:
                raise ValueError(
                    f"Field {label}: epoch mismatch vs {ref_label} "
                    f"({item.epoch!r} vs {ref.epoch!r})."
                )
            if not axis0_only:
                if item.axis_names != ref.axis_names:
                    raise ValueError(
                        f"Field {label}: Inconsistent axis_names vs {ref_label}. "
                        f"Expected {ref.axis_names}, got {item.axis_names}."
                    )
                if item.space_domains != ref.space_domains:
                    raise ValueError(
                        f"Field {label}: Inconsistent space_domains vs {ref_label}. "
                        f"Expected {ref.space_domains}, got {item.space_domains}."
                    )
                if strict_metadata and ref.geo is not None and item.geo is not None:
                    if item.geo.crs != ref.geo.crs:
                        raise ValueError(
                            f"Field {label}: CRS mismatch vs {ref_label} "
                            f"({item.geo.crs!r} vs {ref.geo.crs!r})."
                        )
            elif strict_metadata and item.axis_names[0] != ref.axis_names[0]:
                raise ValueError(
                    f"Field {label}: axis 0 name mismatch vs {ref_label} "
                    f"({item.axis_names[0]!r} vs {ref.axis_names[0]!r})."
                )

            for ax_idx in axes_to_check:
                ref_ax = getattr(ref, f"_axis{ax_idx}_index")
                item_ax = getattr(item, f"_axis{ax_idx}_index")
                if ref_ax.shape != item_ax.shape:
                    raise ValueError(
                        f"Field {label}: Axis {ax_idx} shape mismatch vs {ref_label}. "
                        f"Expected {ref_ax.shape}, got {item_ax.shape}."
                    )
                if not _axis_coordinates_close(ref_ax, item_ax):
                    raise ValueError(
                        f"Field {label}: Axis {ax_idx} coordinate mismatch vs {ref_label}."
                    )

    def validate_units(
        self,
        groups: Mapping[str, Iterable[Any]] | Sequence[Iterable[Any]] | None = None,
    ) -> None:
        """Validate physical unit equivalence across fields or within named groups.

        Parameters
        ----------
        groups : mapping or sequence of key iterables, optional
            When ``None``, every entry in the collection must share equivalent
            units. When a mapping ``{group_name: [key1, key2, ...]}`` or a
            sequence of key groups is given, entries within each group must
            share equivalent units while different groups remain independent.

        Raises
        ------
        KeyError
            If a key specified in *groups* is not present in this collection.
        ValueError
            If any group contains incompatible physical units.

        """

        def _check_group(group_label: str, keys: Iterable[Any]) -> None:
            leaves: list[tuple[str, Any]] = []
            for key in keys:
                if key not in self:
                    raise KeyError(
                        f"Key {key!r} not found in FieldDict "
                        f"(available: {list(self.keys())})."
                    )
                leaves.extend(_iter_leaf_scalar_fields(key, self[key]))
            if not leaves:
                return
            ref_label, ref_field = leaves[0]
            ref_unit = (
                ref_field.unit
                if ref_field.unit is not None
                else u.dimensionless_unscaled
            )
            for label, leaf in leaves[1:]:
                item_unit = (
                    leaf.unit if leaf.unit is not None else u.dimensionless_unscaled
                )
                if not item_unit.is_equivalent(ref_unit):
                    raise ValueError(
                        f"Inconsistent unit in {group_label}: {label} has unit "
                        f"{item_unit!r}, expected equivalent to {ref_unit!r} "
                        f"(from {ref_label})."
                    )

        if groups is None:
            _check_group("FieldDict", self.keys())
        elif isinstance(groups, Mapping):
            for group_name, group_keys in groups.items():
                if isinstance(group_keys, (str, bytes)):
                    group_keys = [group_keys]
                _check_group(f"group {group_name!r}", group_keys)
        elif isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
            for idx, group_keys in enumerate(groups):
                if isinstance(group_keys, (str, bytes)):
                    group_keys = [group_keys]
                _check_group(f"group[{idx}]", group_keys)
        else:
            raise TypeError(
                "groups must be a mapping, a sequence of key iterables, or None."
            )

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

    def _apply_batch(
        self, scalar_method: str, collection_method: str, *args: Any, **kwargs: Any
    ):
        """Apply a batch method across heterogeneous field entries, failing clearly."""
        out: list[tuple[Any, Any]] = []
        for key, field in self.items():
            if hasattr(field, scalar_method):
                fn = getattr(field, scalar_method)
                out.append((key, fn(*args, **kwargs)))
            elif hasattr(field, collection_method):
                fn = getattr(field, collection_method)
                out.append((key, fn(*args, **kwargs)))
            else:
                raise TypeError(
                    f"Key {key!r}: field of type {type(field).__name__} "
                    f"does not support '{scalar_method}'."
                )
        return self.__class__(dict(out))

    def fft_time_all(self, **kwargs):
        """Apply fft_time across contained fields, returning the collection type."""
        return self._apply_batch("fft_time", "fft_time_all", **kwargs)

    def ifft_time_all(self, **kwargs):
        """Apply ifft_time across contained fields, returning the collection type."""
        return self._apply_batch("ifft_time", "ifft_time_all", **kwargs)

    def fft_space_all(self, axes=None, **kwargs):
        """Apply fft_space across contained fields, returning the collection type."""
        return self._apply_batch("fft_space", "fft_space_all", axes=axes, **kwargs)

    def ifft_space_all(self, axes=None, **kwargs):
        """Apply ifft_space across contained fields, returning the collection type."""
        return self._apply_batch("ifft_space", "ifft_space_all", axes=axes, **kwargs)

    def resample_all(self, rate, **kwargs):
        """Apply resample across contained fields, returning the collection type."""
        return self._apply_batch("resample", "resample_all", rate, **kwargs)

    def filter_all(self, *args, **kwargs):
        """Apply filter across contained fields, returning the collection type."""
        return self._apply_batch("filter", "filter_all", *args, **kwargs)

    def sel_all(self, **kwargs):
        """Apply sel across contained fields, returning the collection type."""
        return self._apply_batch("sel", "sel_all", **kwargs)

    def isel_all(self, **kwargs):
        """Apply isel across contained fields, returning the collection type."""
        return self._apply_batch("isel", "isel_all", **kwargs)
