from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from astropy import units as u
from gwpy.types.index import Index

if TYPE_CHECKING:
    from .metadata import MetaDataDict, MetaDataMatrix


class SeriesMatrixAnalysisMixin:
    """Mixin for SeriesMatrix spectral analysis, cropping, and interpolation."""

    if TYPE_CHECKING:
        _value: np.ndarray
        value: np.ndarray
        xspan: Any
        dx: Any
        epoch: float | int | None
        name: str | None
        rows: MetaDataDict
        cols: MetaDataDict
        meta: MetaDataMatrix
        attrs: dict[str, Any] | None
        dtype: np.dtype[Any]
        ndim: int
        shape: tuple[int, ...]

        @property
        def xindex(self) -> np.ndarray | u.Quantity | Index | None: ...

        @xindex.setter
        def xindex(self, value: np.ndarray | u.Quantity | Index | None) -> None: ...

        @property
        def xunit(self) -> u.Unit | None: ...

        @property
        def _x_axis_index(self) -> int: ...

        def row_index(self, key: Any) -> int: ...
        def col_index(self, key: Any) -> int: ...
        def is_contiguous(self, other: Any, tol: float = ...) -> int: ...
        def _get_meta_for_constructor(
            self, data: np.ndarray, xindex: Any
        ) -> dict[str, Any]: ...

    def _get_axis_slice(self, axis, sl):
        """Helper to create a slice tuple for a specific axis."""
        s = [slice(None)] * self.ndim
        s[axis] = sl
        return tuple(s)

    @property
    def _x_axis_norm(self):
        """Normalized x-axis index."""
        idx = getattr(self, "_x_axis_index", -1)
        if idx < 0:
            idx += self.ndim
        return idx

    def crop(self, start=None, end=None, copy=False):
        """Crop the matrix to a specified range along the sample axis."""
        xindex = cast(np.ndarray | u.Quantity | Index, self.xindex)
        xunit = getattr(xindex, "unit", None)
        xvalues = getattr(xindex, "value", np.asarray(xindex))

        def _as_base(val):
            if val is None:
                return None
            if xunit is not None:
                return u.Quantity(val, xunit).to_value(xunit)
            return u.Quantity(val).value if isinstance(val, u.Quantity) else val

        start_val = _as_base(start)
        end_val = _as_base(end)

        idx0 = (
            np.searchsorted(xvalues, start_val, side="left")
            if start_val is not None
            else 0
        )
        idx1 = (
            np.searchsorted(xvalues, end_val, side="left")
            if end_val is not None
            else len(xvalues)
        )

        sl = slice(idx0, idx1)
        new_data = self.value[self._get_axis_slice(self._x_axis_norm, sl)]

        if copy:
            new_data = np.array(new_data, copy=True)
        new_xindex = xindex[idx0:idx1]

        return self.__class__(**self._get_meta_for_constructor(new_data, new_xindex))

    def append(self, other, inplace=True, pad=None, gap=None, resize=True):
        """Append another matrix along the sample axis."""
        target = self
        axis = target._x_axis_norm

        if gap is None:
            gap = "raise" if pad is None else "pad"
        if pad is None and gap == "pad":
            pad = 0.0

        base_unit = getattr(target.xindex, "unit", getattr(other.xindex, "unit", None))

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return u.Quantity(val, base_unit).to_value(base_unit)

        def _is_regular(xidx):
            if hasattr(xidx, "regular"):
                return xidx.regular
            try:
                arr = np.asarray(xidx, dtype=float)
                if arr.size < 2:
                    return True
                diffs = np.diff(arr)
                return np.allclose(diffs, diffs[0])
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                return False

        def _concat_ignore(a, b):
            new_data = np.concatenate([a.value, b.value], axis=axis)
            if base_unit is None:
                new_xindex = np.concatenate(
                    [np.asarray(a.xindex), np.asarray(b.xindex)]
                )
            else:
                ax = u.Quantity(a.xindex).to_value(base_unit)
                bx = u.Quantity(b.xindex).to_value(base_unit)
                new_xindex = np.concatenate([ax, bx]) * base_unit
            return self.__class__(
                **self._get_meta_for_constructor(new_data, new_xindex)
            )

        cont = target.is_contiguous(other)
        if cont != 1:
            _gap_is_numeric = isinstance(gap, (int, float, np.number, u.Quantity))
            _use_pad = (gap == "pad") or (_gap_is_numeric and pad is not None)
            if _use_pad:
                if not _is_regular(target.xindex):
                    raise ValueError("Padding gap requires regular xindex")
                s0, s1 = target.xspan
                o0, _ = other.xspan
                s0b = _to_base(s0)
                s1b = _to_base(s1)
                o0b = _to_base(o0)
                if s0b <= o0b < s1b:
                    raise ValueError("Cannot append overlapping SeriesMatrix")
                if o0b < s0b:
                    raise ValueError("Cannot append that starts before this one")

                gap_base = np.inf if gap == "pad" else _to_base(gap)
                out_full = target.append_exact(
                    other, inplace=False, pad=pad, gap=gap_base
                )
            elif gap == "ignore":
                out_full = _concat_ignore(target, other)
            else:
                s0, s1 = target.xspan
                o0, _ = other.xspan
                if _to_base(s0) < _to_base(o0) < _to_base(s1):
                    raise ValueError("Cannot append overlapping SeriesMatrix")
                raise ValueError("Cannot append discontiguous SeriesMatrix")
        else:
            out_full = target.append_exact(other, inplace=False, pad=None, gap=None)

        if not resize:
            orig_len = target.shape[axis]
            sl = slice(-orig_len, None)
            new_data = out_full.value[out_full._get_axis_slice(axis, sl)]
            out_xindex = cast(np.ndarray | u.Quantity | Index, out_full.xindex)
            new_xindex = out_xindex[sl]
            out_full = self.__class__(
                **self._get_meta_for_constructor(new_data, new_xindex)
            )

        if inplace:
            if out_full.shape == target.shape:
                target._value[:] = out_full._value
                target.xindex = out_full.xindex
                target.epoch = out_full.epoch
                return target
            # If shape differs, inplace fails for ndarray subclass usually?
            # SeriesMatrixAnalysisMixin implies it might fail or resize?
            # The original code:
            # if self.shape != out_data.shape: self.resize(...)
            # But resize is destructive.
            # Here we just return out_full if not same shape?
            pass

        return out_full

    def append_exact(self, other, inplace=False, pad=None, gap=None, tol=1 / 2.0**18):
        """Append another matrix with strict contiguity checking."""
        # Shape check: all dims except x-axis must match
        axis = self._x_axis_norm

        s_shape = list(self.shape)
        o_shape = list(other.shape)
        s_shape.pop(axis)
        o_shape.pop(axis)

        if s_shape != o_shape:
            raise ValueError(
                f"Matrix shapes mismatch (excluding append axis): {self.shape} vs {other.shape}"
            )

        base_unit = getattr(self.xindex, "unit", getattr(other.xindex, "unit", None))

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return u.Quantity(val, base_unit).to_value(base_unit)

        x1 = _to_base(self.xspan[1])
        x2 = _to_base(other.xspan[0])
        diff = x2 - x1

        if abs(diff) > tol:
            if gap is None:
                raise ValueError(
                    f"Matrices are not contiguous (gap={diff}) and gap handling is not specified"
                )
            if isinstance(gap, (int, float, np.number, u.Quantity)):
                gap_val = _to_base(gap)
                if abs(diff) > gap_val:
                    raise ValueError(
                        f"Matrices are not contiguous (gap={diff}) and gap exceeds tolerance {gap_val}"
                    )

            if pad is None:
                raise ValueError("Gap detected but pad value not provided")

            # Padding logic
            dx = _to_base(self.dx)
            n_pad = int(round(diff / dx))
            if n_pad < 0:
                raise ValueError(f"Matrices overlap: end={x1}, start={x2}")

            # Create padding data
            pad_shape = list(self.shape)
            pad_shape[axis] = n_pad
            pad_data = np.full(pad_shape, pad, dtype=self.dtype)
            pad_x = x1 + np.arange(n_pad) * dx
            if base_unit:
                pad_x = pad_x * base_unit

            new_data = np.concatenate([self.value, pad_data, other.value], axis=axis)
            new_xindex = np.concatenate(
                [np.asarray(self.xindex), np.asarray(pad_x), np.asarray(other.xindex)]
            )
            if base_unit:
                new_xindex = new_xindex * base_unit
        else:
            new_data = np.concatenate([self.value, other.value], axis=axis)
            new_xindex = np.concatenate(
                [np.asarray(self.xindex), np.asarray(other.xindex)]
            )
            if base_unit:
                new_xindex = new_xindex * base_unit

        res = self.__class__(**self._get_meta_for_constructor(new_data, new_xindex))

        if inplace:
            # Resizing inplace ndarray is tricky/impossible if size changes.
            # But SeriesMatrix logic tried to do self._value[:] = ...
            pass  # We return res mostly.

        return res

    def prepend(self, other, inplace=True, pad=None, gap=None, resize=True):
        """Prepend another matrix at the beginning along the sample axis."""
        res = other.append(self, inplace=False, pad=pad, gap=gap, resize=resize)
        # Inplace prepend implies changing self to result
        # If we can't mutate self shape, we return result.
        return res

    def prepend_exact(self, other, inplace=False, pad=None, gap=None, tol=1 / 2.0**18):
        """Prepend another matrix with strict contiguity checking."""
        return other.append_exact(self, inplace=inplace, pad=pad, gap=gap, tol=tol)

    def update(self, other, inplace=True, pad=None, gap=None):
        """Update matrix by appending without resizing (rolling buffer style)."""
        return self.append(other, inplace=inplace, pad=pad, gap=gap, resize=False)

    def diff(self, n=1, axis=None):
        """Calculate the n-th discrete difference along the sample axis."""
        target_axis = self._x_axis_norm if axis is None else axis

        new_data = np.diff(self.value, n=n, axis=target_axis)

        if target_axis == self._x_axis_norm:
            xindex = cast(np.ndarray | u.Quantity | Index, self.xindex)
            new_xindex = xindex[n:]
        else:
            new_xindex = self.xindex

        # Name update handled by caller usually or simple append
        new_inst = self.__class__(
            **self._get_meta_for_constructor(new_data, new_xindex)
        )
        if self.name:
            new_inst.name = f"diff({self.name}, n={n})"
        return new_inst

    def value_at(self, x):
        """Get the matrix values at a specific x-axis location."""
        xvalues = getattr(self.xindex, "value", np.asarray(self.xindex))
        xunit = getattr(self.xindex, "unit", None)
        if xunit:
            target = u.Quantity(x, xunit).to_value(xunit)
        else:
            target = x

        idx = np.searchsorted(xvalues, target)
        if idx >= len(xvalues) or not np.isclose(xvalues[idx], target):
            raise ValueError(f"Value {x} not found in xindex")

        return self.value[self._get_axis_slice(self._x_axis_norm, idx)]

    def pad(self, pad_width, **kwargs):
        """Pad the matrix along the sample axis."""

        axis = self._x_axis_norm

        if isinstance(pad_width, int):
            pw = (pad_width, pad_width)
        else:
            pw = pad_width

        # Create full pad tuple for np.pad
        # ((0,0), ..., (pw_before, pw_after), ..., (0,0))
        full_pad: list[tuple[int, int]] = [(0, 0)] * self.ndim
        full_pad[axis] = pw
        full_pad_tuple: tuple[tuple[int, int], ...] = tuple(full_pad)

        kwargs.setdefault("mode", "constant")
        new_data = np.pad(self.value, full_pad_tuple, **kwargs)

        # Update xindex
        dx = self.dx
        xindex = cast(np.ndarray | u.Quantity | Index, self.xindex)
        x0 = xindex[0]
        n_before, n_after = pw
        new_x0 = x0 - n_before * dx
        total_len = new_data.shape[axis]
        from .seriesmatrix_validation import build_index_if_needed

        new_xindex = build_index_if_needed(
            None, dx=dx, x0=new_x0, xunit=self.xunit, length=total_len
        )

        return self.__class__(**self._get_meta_for_constructor(new_data, new_xindex))

    def shift(self, delta):
        """Shift the sample axis by a constant offset."""
        self.xindex = self.xindex + delta
        return self

    def interpolate(self, xindex, **kwargs):
        """Interpolate the matrix to a new sample axis."""
        from scipy.interpolate import interp1d

        # Prepare interpolator
        x_old = getattr(self.xindex, "value", np.asarray(self.xindex))

        axis = self._x_axis_norm
        y_old = np.moveaxis(self.value, axis, 0)  # Move x-axis to 0

        f = interp1d(x_old, y_old, axis=0, **kwargs)

        x_new = getattr(xindex, "value", np.asarray(xindex))
        y_new_stack = f(x_new)
        y_new = np.moveaxis(y_new_stack, 0, axis)  # Move back

        return self.__class__(**self._get_meta_for_constructor(y_new, xindex))
