from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .metadata import MetaData, MetaDataDict, MetaDataMatrix

if TYPE_CHECKING:
    from astropy import units as u
    from gwpy.types.index import Index


class SeriesMatrixStructureMixin:
    """Mixin for SeriesMatrix structural transformations (reshape, transpose, cast)."""

    if TYPE_CHECKING:
        _value: np.ndarray
        value: np.ndarray
        rows: MetaDataDict
        cols: MetaDataDict
        meta: MetaDataMatrix
        name: str | None
        epoch: float | int | None
        attrs: dict[str, Any] | None

        @property
        def xindex(self) -> np.ndarray | u.Quantity | Index | None: ...

        @xindex.setter
        def xindex(self, value: np.ndarray | u.Quantity | Index | None) -> None: ...

        def view(self, dtype: Any = ..., type: type[Any] | None = ...) -> Any: ...

    def copy(self, order="C"):
        """Create a deep copy of this matrix."""

        def _copy_meta_dict(md: MetaDataDict, prefix: str):
            items = OrderedDict()
            for k, v in md.items():
                items[k] = MetaData(**dict(v))
            return MetaDataDict(items, expected_size=len(md), key_prefix=prefix)

        new_val = np.array(self.view(np.ndarray), copy=True, order=order)
        new_meta = MetaDataMatrix(deepcopy(np.asarray(self.meta)))
        new_rows = _copy_meta_dict(self.rows, "row")
        new_cols = _copy_meta_dict(self.cols, "col")
        xindex = self.xindex
        if xindex is None:
            new_xindex = None
        else:
            try:
                new_xindex = xindex.copy()
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                new_xindex = deepcopy(xindex)

        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            new_val,
            meta=new_meta,
            rows=new_rows,
            cols=new_cols,
            xindex=new_xindex,
            name=self.name,
            epoch=self.epoch,
            attrs=deepcopy(self.attrs),
        )

    def astype(
        self,
        dtype,
        order="K",
        casting="unsafe",
        subok=True,
        copy=True,
    ):
        """Cast matrix data to a specified type."""
        new_val = self.value.astype(  # type: ignore[arg-type]
            dtype,
            order=order,
            casting=casting,
            subok=subok,
            copy=cast(Any, copy),
        )
        if not copy and new_val is self.value:
            return self
        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            new_val,
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )

    @property
    def real(self):
        """Real part of the matrix."""
        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            self.value.real,
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=f"{self.name}.real" if self.name else "",
            epoch=self.epoch,
            attrs=self.attrs,
        )

    @real.setter
    def real(self, value) -> None:
        self.value.real = value

    @property
    def imag(self):
        """Imaginary part of the matrix."""
        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            self.value.imag,
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=f"{self.name}.imag" if self.name else "",
            epoch=self.epoch,
            attrs=self.attrs,
        )

    @imag.setter
    def imag(self, value) -> None:
        self.value.imag = value

    def conj(self):
        """Complex conjugate of the matrix."""
        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            np.conjugate(self.value),
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=f"{self.name}.conj" if self.name else "",
            epoch=self.epoch,
            attrs=self.attrs,
        )

    @property
    def T(self):
        """Transpose of the matrix (rows and columns swapped)."""
        return self.transpose()

    def transpose(self, *axes):
        """Transpose rows and columns, preserving sample axis as 2."""
        if not axes:
            # Default matrix transpose (0, 1, 2) -> (1, 0, 2)
            new_val = np.transpose(self.value, (1, 0, 2))
            new_meta = self.meta.T
            matrix_cls = cast(type[Any], self.__class__)
            return matrix_cls(
                new_val,
                xindex=self.xindex,
                rows=self.cols,
                cols=self.rows,
                meta=new_meta,
                name=f"{self.name}.T" if self.name else "",
                epoch=self.epoch,
                attrs=self.attrs,
            )
        else:
            # Custom axes transpose
            new_val = np.transpose(self.value, axes)
            return new_val

    def reshape(
        self,
        *shape,
        order: Literal["A", "C", "F"] | None = "C",
        copy: bool | None = None,
    ):
        """Reshape the matrix dimensions."""
        # For SeriesMatrix, reshape usually only makes sense on (rows, cols)
        # while keeping sample axis size.
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        nsamp = self._value.shape[2]
        if len(shape) == 2:
            target_shape = (shape[0], shape[1], nsamp)
        elif len(shape) == 3:
            if shape[2] != nsamp:
                raise ValueError(
                    f"Cannot reshape sample axis: expected {nsamp}, got {shape[2]}"
                )
            target_shape = shape
        else:
            raise ValueError("Reshape target must be 2D or 3D")

        target_shape_int = tuple(int(s) for s in target_shape)
        if copy is None:
            new_val = self._value.reshape(target_shape_int, order=order)
        else:
            reshaped = self._value.reshape(target_shape_int, order=order)
            new_val = np.array(reshaped, copy=copy)
        # Metadata must also be reshaped
        new_meta = np.asarray(self.meta).reshape(target_shape_int[:2], order=order)

        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            new_val,
            xindex=self.xindex,
            meta=MetaDataMatrix(new_meta),
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )
