from __future__ import annotations

from copy import deepcopy
from typing import OrderedDict

import numpy as np

from .metadata import MetaData, MetaDataDict, MetaDataMatrix


class SeriesMatrixStructureMixin:
    """Mixin for SeriesMatrix structural transformations (reshape, transpose, cast)."""

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
        try:
            new_xindex = self.xindex.copy()
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            new_xindex = deepcopy(self.xindex)

        return self.__class__(
            new_val,
            meta=new_meta,
            rows=new_rows,
            cols=new_cols,
            xindex=new_xindex,
            name=self.name,
            epoch=self.epoch,
            attrs=deepcopy(self.attrs),
        )

    def astype(self, dtype, copy=True):
        """Cast matrix data to a specified type."""
        new_val = self.value.astype(dtype, copy=copy)
        if not copy and new_val is self.value:
            return self
        return self.__class__(
            new_val,
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )

    def real(self):
        """Real part of the matrix."""
        return self.__class__(
            self.value.real,
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=f"{self.name}.real" if self.name else "",
            epoch=self.epoch,
            attrs=self.attrs,
        )

    def imag(self):
        """Imaginary part of the matrix."""
        return self.__class__(
            self.value.imag,
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=f"{self.name}.imag" if self.name else "",
            epoch=self.epoch,
            attrs=self.attrs,
        )

    def conj(self):
        """Complex conjugate of the matrix."""
        return self.__class__(
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
            return self.__class__(
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

    def reshape(self, shape, order="C"):
        """Reshape the matrix dimensions."""
        # For SeriesMatrix, reshape usually only makes sense on (rows, cols)
        # while keeping sample axis size.
        nsamp = self._value.shape[2]
        if len(shape) == 2:
            target_shape = (shape[0], shape[1], nsamp)
        elif len(shape) == 3:
            if shape[2] != nsamp:
                raise ValueError(f"Cannot reshape sample axis: expected {nsamp}, got {shape[2]}")
            target_shape = shape
        else:
            raise ValueError("Reshape target must be 2D or 3D")

        new_val = self._value.reshape(target_shape, order=order)
        # Metadata must also be reshaped
        new_meta = self.meta.value.reshape(target_shape[:2], order=order)

        return self.__class__(
            new_val,
            xindex=self.xindex,
            meta=MetaDataMatrix(new_meta),
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )
