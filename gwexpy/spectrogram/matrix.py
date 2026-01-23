from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from gwexpy.types.mixin import PhaseMethodsMixin
from gwexpy.types.seriesmatrix import SeriesMatrix

from .collections import SpectrogramDict, SpectrogramList
from .matrix_analysis import SpectrogramMatrixAnalysisMixin
from .matrix_core import SpectrogramMatrixCoreMixin
from .spectrogram import Spectrogram


class SpectrogramMatrix(
    PhaseMethodsMixin,
    SpectrogramMatrixCoreMixin,
    SpectrogramMatrixAnalysisMixin,
    SeriesMatrix,
):
    """
    Evaluation Matrix for Spectrograms (Time-Frequency maps).

    This class represents a collection of Spectrograms, structured as either:

    - 3D: (Batch, Time, Frequency)
    - 4D: (Row, Col, Time, Frequency)

    It inherits from SeriesMatrix, providing powerful indexing, metadata management,
    and analysis capabilities (slicing, interpolation, statistics).

    Serialization (Known Limitation)
    --------------------------------
    Pickle round-trip (`pickle.dumps` / `pickle.loads`) preserves the array shape
    and values, but **axis metadata (xindex/times, frequencies) may not be fully
    restored** due to numpy ndarray subclass serialization constraints.

    This is a known limitation. If you require full metadata preservation, consider
    using HDF5 I/O methods (e.g., `to_hdf5` / `from_hdf5`) instead of pickle.

    TODO: Implement `__reduce_ex__` or `__getstate__/__setstate__` to enable
    complete metadata preservation in pickle round-trips.
    """


    series_class = Spectrogram
    dict_class = SpectrogramDict
    list_class = SpectrogramList

    def __new__(
        cls,
        data,
        times=None,
        frequencies=None,
        unit=None,
        name=None,
        rows=None,
        cols=None,
        meta=None,
        **kwargs,
    ):
        # Handle alias
        if times is None:
            times = kwargs.get("xindex")

        # SeriesMatrix expects 'xindex' and 'xunit' etc.
        # We assume 'data' might be (N, M, Time, Freq) or (N, Time, Freq).
        # We pass xindex=times.

        # We first let SeriesMatrix handle normalization of N, M and MetaDataMatrix.
        # However, SeriesMatrix.__new__ behavior for 4D/nD data depends on validation.
        # SeriesMatrix validation usually assumes 3D.
        # We may need to bypass or adjust SeriesMatrix.__new__ validation for 4D data if it's too strict.

        # For now, we try to perform basic setup and call super().__new__ via np.ndarray mechanism
        # but SeriesMatrix does a lot of heavy lifting in __new__.

        # Strategy: adapt arguments to SeriesMatrix signature
        # times -> xindex

        # Note: SeriesMatrix input normalization might flatten extra dims if not careful.
        # Check gwexpy/types/series_matrix_validation.py: _normalize_input handles 3D.
        # For 4D specific handling, we might need to manually prep or rely on SeriesMatrix letting it pass?
        # Actually SeriesMatrixValidationMixin _normalize_input mainly handles 1D, 2D, 3D.

        # If data is 4D, SeriesMatrix _normalize_input might fail or treat it oddly.
        # Let's verify _normalize_input logic (Step 1167).
        # It has blocks for Scalar, Series, Array, 1D/2D, 3D. It does NOT explicitly handle 4D.
        # So we might need to override behavior or pre-process data to be SeriesMatrix-compatible (stored as object array?)
        # NO, we want numeric array.

        # If data is 4D (N, M, T, F), SeriesMatrix assumes (Row, Col, Sample).
        # If we want to use SeriesMatrix infrastructure, we must respect the 3-axis structure `(Row, Col, X)`?
        # Integrating 4D directly into SeriesMatrix (nd=4) might break many assumptions in `series_matrix_core` (e.g. shape3D return).

        # ALTERNATIVE: Use Object Array of Spectrograms? No, expensive.
        # ALTERNATIVE: Treat Freq axis as implicit?

        # If we invoke SeriesMatrix, it calls `_normalize_input`.
        # If we just call `np.array(data).view(cls)`, we bypass SeriesMatrix.__new__ logic entirely?
        # But we want mixins.

        # Since we inherit SeriesMatrix, calling SeriesMatrix(data...) creates a new object using SeriesMatrix.__new__.

        # Let's implement a custom __new__ that handles the 4D init, sets properties, and returns the view,
        # mimicking SeriesMatrix.__new__ but tailored for 4D.

        # ... Wait, if we inherit SeriesMatrix, `super()` refers to SeriesMatrix.
        # If we don't call `super().__new__`, we skip its logic. That's fine if we replicate what we need.

        obj = np.asarray(data).view(cls)

        # Set Spectrogram-specific props
        obj.times = times  # sets xindex via CoreMixin
        obj.frequencies = frequencies

        # Set metadata manually or via helpers if available.
        # Only do basic setup here to replicate old behavior + SeriesMatrix props

        obj.name = name
        obj.unit = unit  # logic for unit array vs scalar unit needed?

        # Setup MetaDataMatrix using rows/cols logic from previous implementation
        from gwexpy.types.metadata import MetaDataDict, MetaDataMatrix

        def _entries_len(entries):
            return len(entries) if entries is not None else None

        if obj.ndim == 3:  # (Batch, Time, Freq)
            N = obj.shape[0]
            # ... (same logic as before for rows/cols) ...
            # Simplify for brevity or reuse logic?
            row_len = _entries_len(rows)
            col_len = _entries_len(cols)
            use_grid = row_len and col_len and row_len * col_len == N

            if use_grid:
                obj.rows = MetaDataDict(rows, expected_size=row_len, key_prefix="row")
                obj.cols = MetaDataDict(cols, expected_size=col_len, key_prefix="col")
                obj.meta = MetaDataMatrix(meta, shape=(row_len, col_len))
            else:
                obj.rows = MetaDataDict(rows, expected_size=N, key_prefix="batch")
                obj.cols = MetaDataDict(None, expected_size=1, key_prefix="col")
                obj.meta = MetaDataMatrix(meta, shape=(N, 1))

        elif obj.ndim == 4:  # (Row, Col, Time, Freq)
            nrow, ncol = obj.shape[:2]
            obj.rows = MetaDataDict(rows, expected_size=nrow, key_prefix="row")
            obj.cols = MetaDataDict(cols, expected_size=ncol, key_prefix="col")
            obj.meta = MetaDataMatrix(meta, shape=(nrow, ncol))
        else:
            # Fallback
            obj.rows = None
            obj.cols = None
            obj.meta = None

        # Apply unit to metadata if needed (only if not explicitly set in meta)
        if unit is not None and getattr(obj, "meta", None) is not None:
            for m in obj.meta.flat:
                # MetaData defaults to dimensionless_unscaled, so check for that too
                if m.unit is None or m.unit == u.dimensionless_unscaled:
                    m.unit = unit

        # If no global unit was provided, infer it from metadata if consistent
        if obj.unit is None and getattr(obj, "meta", None) is not None:
            meta_units = {m.unit for m in obj.meta.flat if m is not None}
            if len(meta_units) == 1:
                obj.unit = next(iter(meta_units))

        obj.epoch = kwargs.get("epoch", 0.0)
        obj._value = obj.view(np.ndarray)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        super().__array_finalize__(obj)
        self.frequencies = getattr(obj, "frequencies", None)
        if not hasattr(self, "_value"):
            self._value = self.view(np.ndarray)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override SeriesMatrix.__array_ufunc__ to correctly handle SpectrogramMatrix structure
        (Batch, Time, Freq) or (Row, Col, Time, Freq).

        Per-element units are preserved in MetaDataMatrix:
        - Scalar operations: apply ufunc to each element's unit individually
        - Binary matrix operations: check per-element unit compatibility and raise
          UnitConversionError if any pair is incompatible
        """
        from gwexpy.types.metadata import MetaData, MetaDataMatrix

        if method != "__call__":
            # Defer to ndarray (e.g. at, reduce) - might lose metadata but SeriesMatrix does too
            args = [
                inp.view(np.ndarray) if isinstance(inp, SpectrogramMatrix) else inp
                for inp in inputs
            ]
            return super(SeriesMatrix, self).__array_ufunc__(
                ufunc, method, *args, **kwargs
            )

        # Identify ufunc category for unit handling
        _ADD_SUB_UFUNCS = {np.add, np.subtract}
        _COMPARISON_UFUNCS = {
            np.less, np.less_equal, np.equal, np.not_equal,
            np.greater, np.greater_equal
        }
        _MUL_DIV_UFUNCS = {np.multiply, np.divide, np.floor_divide, np.true_divide}

        # 1. Unpack inputs
        args = []
        sgm_inputs = []  # SpectrogramMatrix instances
        scalar_inputs = []  # Scalars/units for unit arithmetic
        for inp in inputs:
            if isinstance(inp, SpectrogramMatrix):
                args.append(inp.view(np.ndarray))
                sgm_inputs.append(inp)
            elif isinstance(inp, (u.Quantity, np.ndarray, float, int, complex)):
                val = getattr(inp, "value", inp)
                args.append(np.asarray(val))
                scalar_inputs.append(inp)
            elif isinstance(inp, u.UnitBase):
                args.append(1.0)  # Unit acts as multiplier
                scalar_inputs.append(inp)
            else:
                return NotImplemented

        if not sgm_inputs:
            return NotImplemented

        main = sgm_inputs[0]

        # 2. Compute Data
        try:
            result_data = ufunc(*args, **kwargs)
        except (TypeError, ValueError, u.UnitConversionError):
            return NotImplemented

        # 3. Handle per-element unit propagation
        # Determine if this is a scalar op (1 matrix) or binary matrix op (2+ matrices)
        is_scalar_op = len(sgm_inputs) == 1
        is_binary_matrix_op = len(sgm_inputs) >= 2

        new_meta = None
        if main.meta is not None:
            meta_shape = main.meta.shape
            new_meta_arr = np.empty(meta_shape, dtype=object)

            if is_scalar_op:
                # Scalar operation: apply ufunc to each element's unit
                # Get scalar unit(s)
                scalar_unit = u.dimensionless_unscaled
                for sc in scalar_inputs:
                    if isinstance(sc, u.UnitBase):
                        scalar_unit = sc
                    elif isinstance(sc, u.Quantity):
                        scalar_unit = sc.unit
                    # else: dimensionless

                for idx in np.ndindex(meta_shape):
                    old_meta = main.meta[idx]
                    old_unit = old_meta.unit if old_meta.unit else u.dimensionless_unscaled

                    try:
                        # Apply ufunc to units
                        q_result = ufunc(
                            u.Quantity(1, old_unit),
                            u.Quantity(1, scalar_unit)
                        ) if len(inputs) == 2 else ufunc(u.Quantity(1, old_unit))
                        new_unit = q_result.unit if hasattr(q_result, "unit") else old_unit
                    except (TypeError, ValueError, u.UnitConversionError):
                        new_unit = old_unit

                    new_meta_arr[idx] = MetaData(
                        name=old_meta.name,
                        channel=old_meta.channel,
                        unit=new_unit,
                    )

            elif is_binary_matrix_op:
                # Binary matrix operation: check per-element unit compatibility
                other_sgm = sgm_inputs[1] if len(sgm_inputs) > 1 else None

                if other_sgm is not None and other_sgm.meta is not None:
                    # Check shape compatibility
                    if main.meta.shape != other_sgm.meta.shape:
                        raise ValueError(
                            f"Metadata shape mismatch: {main.meta.shape} vs {other_sgm.meta.shape}"
                        )

                    for idx in np.ndindex(meta_shape):
                        m1 = main.meta[idx]
                        m2 = other_sgm.meta[idx]
                        u1 = m1.unit if m1.unit else u.dimensionless_unscaled
                        u2 = m2.unit if m2.unit else u.dimensionless_unscaled

                        # Check strict unit equality for add/sub/comparison
                        # Following SeriesMatrix check_add_sub_compatibility:
                        # u0 != uk raises UnitConversionError (even for equivalent units like m vs cm)
                        if ufunc in _ADD_SUB_UFUNCS or ufunc in _COMPARISON_UFUNCS:
                            if u1 != u2:
                                raise u.UnitConversionError(
                                    f"Unit mismatch at element {idx}: {u1} vs {u2}"
                                )
                            new_unit = u1  # Preserve first unit for add/sub
                            if ufunc in _COMPARISON_UFUNCS:
                                new_unit = u.dimensionless_unscaled
                        elif ufunc in _MUL_DIV_UFUNCS:
                            if ufunc == np.multiply:
                                new_unit = u1 * u2
                            else:
                                new_unit = u1 / u2
                        else:
                            # Default: try to compute
                            try:
                                q_result = ufunc(u.Quantity(1, u1), u.Quantity(1, u2))
                                new_unit = q_result.unit if hasattr(q_result, "unit") else u1
                            except (TypeError, ValueError, u.UnitConversionError) as e:
                                if isinstance(e, u.UnitConversionError):
                                    raise
                                new_unit = u1

                        new_meta_arr[idx] = MetaData(
                            name=m1.name,
                            channel=m1.channel,
                            unit=new_unit,
                        )
                else:
                    # Other matrix has no meta; keep main's meta
                    new_meta_arr = main.meta.copy()

            new_meta = MetaDataMatrix(new_meta_arr)

        def _infer_unit(meta):
            if meta is None:
                return None
            meta_units = {m.unit for m in meta.flat if m is not None}
            if len(meta_units) == 1:
                return next(iter(meta_units))
            return None

        # Reconstruct SpectrogramMatrix
        if result_data.shape == main.shape:
            obj = self.__class__(
                result_data,
                times=main.times,
                frequencies=main.frequencies,
                rows=main.rows,
                cols=main.cols,
                meta=new_meta,
                name=main.name,
                unit=_infer_unit(new_meta),
            )
            return obj

        return result_data

    def __mul__(self, other):
        """Multiply by scalar, unit, or matrix."""
        # Explicitly handle u.UnitBase to avoid astropy ufunc precedence issues
        if isinstance(other, u.UnitBase):
            return np.multiply(self, u.Quantity(1, other))
        return np.multiply(self, other)

    def __rmul__(self, other):
        """Right multiply by scalar, unit, or matrix."""
        if isinstance(other, u.UnitBase):
            return np.multiply(u.Quantity(1, other), self)
        return np.multiply(other, self)

    def __truediv__(self, other):
        """Divide by scalar, unit, or matrix."""
        if isinstance(other, u.UnitBase):
            return np.divide(self, u.Quantity(1, other))
        return np.divide(self, other)

    def __rtruediv__(self, other):
        """Right divide by scalar, unit, or matrix."""
        if isinstance(other, u.UnitBase):
            return np.divide(u.Quantity(1, other), self)
        return np.divide(other, self)

    def __add__(self, other):
        """Add scalar/quantity or matrix."""
        return np.add(self, other)

    def __radd__(self, other):
        """Right add."""
        return np.add(other, self)

    def __sub__(self, other):
        """Subtract scalar/quantity or matrix."""
        return np.subtract(self, other)

    def __rsub__(self, other):
        """Right subtract."""
        return np.subtract(other, self)

    def row_keys(self):
        return tuple(self.rows.keys()) if self.rows else tuple()

    def col_keys(self):
        return tuple(self.cols.keys()) if self.cols else tuple()

    def is_compatible(self, other: Any) -> bool:
        """
        Check compatibility with another SpectrogramMatrix/object.
        Overrides SeriesMatrix.is_compatible to avoid loop range issues due to
        mismatch between data shape (Time axis) and metadata shape (Batch/Col).
        """
        # 1. Type check
        if not isinstance(other, type(self)):
            # Fallback or strict check? SeriesMatrix falls back to array shape check.
            if hasattr(other, "shape") and np.shape(self) != np.shape(other):
                raise ValueError(
                    f"shape does not match: {self.shape} vs {np.shape(other)}"
                )
            return True  # assume compatible if shapes match and not SpectrogramMatrix

        # 2. Shape check
        if self.shape != other.shape:
            raise ValueError(
                f"matrix shape does not match: {self.shape} vs {other.shape}"
            )

        # 3. Times/Xindex Check
        # Check units
        t_unit_self = getattr(self.times, "unit", None)
        t_unit_other = getattr(other.times, "unit", None)
        if (
            t_unit_self != t_unit_other
        ):  # Simple equality check sufficient for same implementation
            # Try convert? SeriesMatrix logic is strict about unit object equality or equivalence
            if t_unit_self is not None and t_unit_other is not None:
                if not u.Unit(t_unit_self).is_equivalent(u.Unit(t_unit_other)):
                    raise ValueError(
                        f"times unit does not match: {t_unit_self} vs {t_unit_other}"
                    )

        # Check dx/content (for contiguous check, usually handled by caller, but is_compatible checks xindex content equality?)
        # SeriesMatrix.is_compatible checks xindex equality if dx matches or fallback.
        # But we only need unit compatibility usually for ops?
        # is_contiguous calls is_compatible.
        # Let's keep it simple: check units match. Content matching is handled by append logic (overlap check etc).

        # 4. Meta/Channel Unit consistency
        if self.meta.shape != other.meta.shape:
            # Should match if shapes match (unless metadata structure differs profoundly)
            # But let's proceed to loop over valid meta range
            raise ValueError(
                f"metadata shape mismatch: {self.meta.shape} vs {other.meta.shape}"
            )

        for i in range(self.meta.shape[0]):
            for j in range(self.meta.shape[1]):
                u1 = self.meta[i, j].unit
                u2 = other.meta[i, j].unit
                if u1 != u2:
                    # Allow None vs None
                    if u1 is None and u2 is None:
                        continue
                    if u1 is None or u2 is None:
                        raise ValueError(
                            f"Unit mismatch at meta ({i},{j}): {u1} vs {u2}"
                        )
                    if not u1.is_equivalent(u2):
                        raise ValueError(
                            f"Unit mismatch at meta ({i},{j}): {u1} vs {u2}"
                        )

        return True

    def row_index(self, key):
        if not self.rows:
            raise KeyError(f"Invalid row key: {key}")
        try:
            return list(self.row_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key):
        if not self.cols:
            raise KeyError(f"Invalid column key: {key}")
        try:
            return list(self.col_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid column key: {key}")

    def __getitem__(self, key):
        from gwexpy.types.seriesmatrix_validation import _slice_metadata_dict

        # Handle label-based indexing
        if isinstance(key, str):
            try:
                key = self.row_index(key)
            except KeyError:
                raise

        if (
            isinstance(key, (list, np.ndarray))
            and len(key) > 0
            and isinstance(key[0], str)
        ):
            key = [self.row_index(k) for k in key]

        # Handle tuple keys (Row, Col) or (Row, Col, Time, Freq)
        if isinstance(key, tuple):
            new_key = list(key)
            if isinstance(new_key[0], str):
                new_key[0] = self.row_index(new_key[0])
            if len(new_key) > 1 and isinstance(new_key[1], str):
                try:
                    new_key[1] = self.col_index(new_key[1])
                except (KeyError, IndexError):
                    # Maybe it's not a column key but a slice or index for Time/Freq?
                    pass
            key = tuple(new_key)

        # Access raw data
        raw_data = self.view(np.ndarray)[key]

        # Check for scalar element extraction (returning Spectrogram)
        is_single_element = False
        r_idx, c_idx = 0, 0

        if self.ndim == 3:  # (Batch, Time, Freq)
            if isinstance(key, (int, np.integer)):
                is_single_element = True
                r_idx = key
                c_idx = 0
        elif self.ndim == 4:  # (Row, Col, Time, Freq)
            if isinstance(key, tuple) and len(key) >= 2:
                r, c = key[0], key[1]
                if isinstance(r, (int, np.integer)) and isinstance(
                    c, (int, np.integer)
                ):
                    is_single_element = True
                    r_idx, c_idx = r, c

        if is_single_element:
            # Return Spectrogram
            meta = self.meta[r_idx, c_idx] if self.meta is not None else None
            unit = meta.unit if meta else self.unit
            name = meta.name if meta and meta.name else self.name
            channel = meta.channel if meta else None

            # raw_data should be (Time, Freq)
            if raw_data.ndim != 2:
                # Should not happen if indices are correct for 3D/4D
                raise ValueError(
                    f"Extracted data has wrong dimension for Spectrogram: {raw_data.ndim} (expected 2)"
                )

            return self.series_class(
                raw_data,
                times=self.times,
                frequencies=self.frequencies,
                unit=unit,
                name=name,
                channel=channel,
                epoch=getattr(self, "epoch", None),
            )

        # Return Sub-Matrix
        # We assume raw_data is ndarray. View as SpectrogramMatrix.
        ret = np.asarray(raw_data).view(type(self))
        ret._value = ret.view(np.ndarray)

        # Propagate global props
        ret.times = getattr(self, "times", None)
        ret.frequencies = getattr(self, "frequencies", None)
        ret.unit = getattr(self, "unit", None)
        ret.epoch = getattr(self, "epoch", None)

        # Propagate/Slice Metadata (Rows, Cols, Meta)
        # This is complex for general slicing. Simplification:
        # If ndim preserved, try to slice rows/cols.
        # If ndim reduced (e.g. 4D -> 3D), adjust.
        # Basic case: Batch slicing on 3D or Row slicing on 4D

        main_key = key[0] if isinstance(key, tuple) else key

        # 3D: (Batch, T, F) -> Slice batch
        if self.ndim == 3 and ret.ndim == 3:
            if self.rows:
                ret.rows = _slice_metadata_dict(self.rows, main_key, "row")
            if self.meta is not None:
                # meta is (N, 1)
                try:
                    ret.meta = self.meta[main_key]
                except (IndexError, TypeError):
                    pass
        # 4D: (Row, Col, T, F) -> Slice row, maybe col
        elif self.ndim == 4 and ret.ndim == 4:
            r_key = key[0] if isinstance(key, tuple) else key
            c_key = key[1] if isinstance(key, tuple) and len(key) > 1 else slice(None)

            if self.rows:
                ret.rows = _slice_metadata_dict(self.rows, r_key, "row")
            if self.cols:
                ret.cols = _slice_metadata_dict(self.cols, c_key, "col")
            if self.meta is not None:
                try:
                    # meta is (Row, Col)
                    # If key is simple tuple (slice, slice)
                    if isinstance(key, tuple) and len(key) <= 2:
                        ret.meta = self.meta[key]
                    else:
                        # complex slicing?
                        pass
                except (IndexError, TypeError, KeyError):
                    pass

        return ret

    def to_series_2Dlist(self):
        """Convert matrix to a 2D nested list of Spectrogram objects."""
        r_keys = self.row_keys()
        c_keys = self.col_keys()
        if self.ndim == 3:
            return [[self[i] for _ in range(1)] for i in range(len(r_keys))]
        return [[self[i, j] for j in range(len(c_keys))] for i in range(len(r_keys))]

    def to_series_1Dlist(self):
        """Convert matrix to a flat 1D list of Spectrogram objects."""
        r_keys = self.row_keys()
        c_keys = self.col_keys()
        results = []
        if self.ndim == 3:
            for i in range(len(r_keys)):
                results.append(self[i])
        elif self.ndim == 4:
            for i in range(len(r_keys)):
                for j in range(len(c_keys)):
                    results.append(self[i, j])
        else:
            raise ValueError(f"Unsupported SpectrogramMatrix dimension: {self.ndim}")
        return results

    def to_list(self):
        """Convert to SpectrogramList."""
        from .collections import SpectrogramList

        return SpectrogramList(self.to_series_1Dlist())

    def to_dict(self):
        """Convert to SpectrogramDict."""
        from .collections import SpectrogramDict

        r_keys = self.row_keys()
        c_keys = self.col_keys()
        results = SpectrogramDict()
        if self.ndim == 3:
            for i, rk in enumerate(r_keys):
                results[rk] = self[i]
        elif self.ndim == 4:
            for i, rk in enumerate(r_keys):
                for j, ck in enumerate(c_keys):
                    if len(c_keys) == 1:
                        results[rk] = self[i, j]
                    else:
                        results[(rk, ck)] = self[i, j]
        return results

    def _all_element_units_equivalent(self):
        """Check whether all element units are mutually equivalent."""
        if self.meta is None:
            return True, self.unit
        ref_unit = self.meta[0, 0].unit
        for m in self.meta.flat:
            if m.unit is None:
                continue
            if not m.unit.is_equivalent(ref_unit):
                return False, ref_unit
        return True, ref_unit

    @property
    def shape3D(self):
        # Override Base logic to return relevant 3D view (Batch, Time, Freq) for display?
        # Or (Row, Col, Time) if we treat Freq as hidden dim?
        # For uniformity with SeriesMatrix which is (Row, Col, Sample),
        # if we are 4D (Row, Col, Time, Freq), we might want to return (Row, Col, Time) as 'main' shape with _x_axis_index logic?
        # But core checks shape[-1].
        return self.shape

    def plot(self, **kwargs):
        """Plot the matrix data using gwexpy.plot.Plot."""
        from gwexpy.plot import Plot

        return Plot(self, **kwargs)

    def plot_summary(self, **kwargs):
        """
        Plot Matrix as side-by-side Spectrograms and percentile summaries.
        """
        from gwexpy.plot.plot import plot_summary

        return plot_summary(self, **kwargs)
