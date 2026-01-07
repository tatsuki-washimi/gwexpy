from __future__ import annotations
import numpy as np
from typing import Any
from astropy import units as u

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.mixin import PhaseMethodsMixin
from .matrix_analysis import SpectrogramMatrixAnalysisMixin
from .matrix_core import SpectrogramMatrixCoreMixin
from .collections import SpectrogramList, SpectrogramDict
from .spectrogram import Spectrogram


class SpectrogramMatrix(
    PhaseMethodsMixin,
    SpectrogramMatrixCoreMixin,
    SpectrogramMatrixAnalysisMixin,
    SeriesMatrix
):
    """
    Evaluation Matrix for Spectrograms (Time-Frequency maps).

    This class represents a collection of Spectrograms, structured as either:
    - 3D: (Batch, Time, Frequency)
    - 4D: (Row, Col, Time, Frequency)

    It inherits from SeriesMatrix, providing powerful indexing, metadata management,
    and analysis capabilities (slicing, interpolation, statistics).
    """
    series_class = Spectrogram
    dict_class = SpectrogramDict
    list_class = SpectrogramList

    def __new__(cls, data, times=None, frequencies=None, unit=None, name=None,
                rows=None, cols=None, meta=None, **kwargs):

        # Handle alias
        if times is None:
             times = kwargs.get('xindex')

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
        obj.unit = unit # logic for unit array vs scalar unit needed?

        # Setup MetaDataMatrix using rows/cols logic from previous implementation
        from gwexpy.types.metadata import MetaDataDict, MetaDataMatrix

        def _entries_len(entries):
            return len(entries) if entries is not None else None

        if obj.ndim == 3: # (Batch, Time, Freq)
             N = obj.shape[0]
             # ... (same logic as before for rows/cols) ...
             # Simplify for brevity or reuse logic?
             row_len = _entries_len(rows)
             col_len = _entries_len(cols)
             use_grid = (row_len and col_len and row_len * col_len == N)

             if use_grid:
                 obj.rows = MetaDataDict(rows, expected_size=row_len, key_prefix='row')
                 obj.cols = MetaDataDict(cols, expected_size=col_len, key_prefix='col')
                 obj.meta = MetaDataMatrix(meta, shape=(row_len, col_len))
             else:
                 obj.rows = MetaDataDict(rows, expected_size=N, key_prefix='batch')
                 obj.cols = MetaDataDict(None, expected_size=1, key_prefix='col')
                 obj.meta = MetaDataMatrix(meta, shape=(N, 1))

        elif obj.ndim == 4: # (Row, Col, Time, Freq)
             nrow, ncol = obj.shape[:2]
             obj.rows = MetaDataDict(rows, expected_size=nrow, key_prefix='row')
             obj.cols = MetaDataDict(cols, expected_size=ncol, key_prefix='col')
             obj.meta = MetaDataMatrix(meta, shape=(nrow, ncol))
        else:
             # Fallback
             obj.rows = None
             obj.cols = None
             obj.meta = None

        # Apply unit to metadata if needed
        if unit is not None and getattr(obj, 'meta', None) is not None:
             for m in obj.meta.flat:
                  if m.unit is None:
                       m.unit = unit

        obj.epoch = kwargs.get('epoch', 0.0)
        obj._value = obj.view(np.ndarray)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        super().__array_finalize__(obj)
        self.frequencies = getattr(obj, 'frequencies', None)
        if not hasattr(self, '_value'):
             self._value = self.view(np.ndarray)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override SeriesMatrix.__array_ufunc__ to correctly handle SpectrogramMatrix structure
        (Batch, Time, Freq) or (Row, Col, Time, Freq).
        """
        if method != '__call__':
            # Defer to ndarray (e.g. at, reduce) - might lose metadata but SeriesMatrix does too
            args = [inp.view(np.ndarray) if isinstance(inp, SpectrogramMatrix) else inp for inp in inputs]
            return super(SeriesMatrix, self).__array_ufunc__(ufunc, method, *args, **kwargs)

        # 1. Unpack inputs and checking units/meta
        args = []
        metas = []
        shapes = []
        for inp in inputs:
            if isinstance(inp, SpectrogramMatrix):
                args.append(inp.view(np.ndarray))
                metas.append(inp.meta)
                shapes.append(inp.shape)
            elif isinstance(inp, (u.Quantity, np.ndarray, float, int, complex)):
                 # Wrap as ndarray
                 val = getattr(inp, 'value', inp)
                 args.append(np.asarray(val))
                 # Dummy meta? Scalar has no meta. Implicitly handled by ufunc logic if skipped?
                 metas.append(None)
                 shapes.append(np.shape(val))
            else:
                 return NotImplemented

        # 2. Compute Data
        try:
             result_data = ufunc(*args, **kwargs)
        except Exception:
             return NotImplemented

        # 3. Propagate Metadata (Units)
        # We assume result has same shape structure (or broadcasted).
        # We take self's (this instance's) non-data props (times, freqs, rows, cols).
        # If ufunc allows, units might change.

        # Simple logic: if preserving unit or changing unit, update meta/unit property.
        # This is complex to do perfectly (SeriesMatrix tries hard).
        # For now, we take unit from first operand if it's SpectrogramMatrix, and apply ufunc to unit?

        main = [x for x in inputs if isinstance(x, SpectrogramMatrix)][0]

        # Attempt to compute new global unit
        unit_args = []
        for inp in inputs:
             if hasattr(inp, 'unit'):
                 unit_args.append(inp.unit if inp.unit else u.dimensionless_unscaled)
             else:
                 unit_args.append(u.dimensionless_unscaled)
        q_args = []
        for x in unit_args:
             if x is None: q_args.append(1)
             else: q_args.append(u.Quantity(1, x))

        try:
             res_q = ufunc(*q_args)
             new_unit = res_q.unit if hasattr(res_q, 'unit') else None
        except (TypeError, ValueError, AttributeError) as e:
             if isinstance(e, u.UnitConversionError):
                  raise e
             new_unit = None

        # Reconstruct SpectrogramMatrix
        # If shape matches main, keep rows/cols/times/freqs
        # If shape changed (e.g. reduction), this ufunc shouldn't have been handled here (method != call usually?)
        # Standard ufuncs like add/mul preserve shape.

        if result_data.shape == main.shape:
             # Propagate meta if possible
             new_meta = None
             if main.meta is not None:
                  # Deep copy meta and update units
                  # Or create new MetaDataMatrix with new units
                  if new_unit is not None:
                       # Update all cells?
                       # Doing shallow copy of meta matrix
                       pass
                  new_meta = main.meta # Simplification: keep meta structure

             obj = self.__class__(
                 result_data,
                 times=main.times,
                 frequencies=main.frequencies,
                 rows=main.rows,
                 cols=main.cols,
                 meta=new_meta,
                 name=main.name,
                 unit=new_unit
             )
             return obj

        return result_data

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
             if hasattr(other, 'shape') and np.shape(self) != np.shape(other):
                  raise ValueError(f"shape does not match: {self.shape} vs {np.shape(other)}")
             return True # assume compatible if shapes match and not SpectrogramMatrix

        # 2. Shape check
        if self.shape != other.shape:
            raise ValueError(f"matrix shape does not match: {self.shape} vs {other.shape}")

        # 3. Times/Xindex Check
        # Check units
        t_unit_self = getattr(self.times, "unit", None)
        t_unit_other = getattr(other.times, "unit", None)
        if t_unit_self != t_unit_other: # Simple equality check sufficient for same implementation
             # Try convert? SeriesMatrix logic is strict about unit object equality or equivalence
             if t_unit_self is not None and t_unit_other is not None:
                 if not u.Unit(t_unit_self).is_equivalent(u.Unit(t_unit_other)):
                      raise ValueError(f"times unit does not match: {t_unit_self} vs {t_unit_other}")

        # Check dx/content (for contiguous check, usually handled by caller, but is_compatible checks xindex content equality?)
        # SeriesMatrix.is_compatible checks xindex equality if dx matches or fallback.
        # But we only need unit compatibility usually for ops?
        # is_contiguous calls is_compatible.
        # Let's keep it simple: check units match. Content matching is handled by append logic (overlap check etc).

        # 4. Meta/Channel Unit consistency
        if self.meta.shape != other.meta.shape:
             # Should match if shapes match (unless metadata structure differs profoundly)
             # But let's proceed to loop over valid meta range
             raise ValueError(f"metadata shape mismatch: {self.meta.shape} vs {other.meta.shape}")

        for i in range(self.meta.shape[0]):
             for j in range(self.meta.shape[1]):
                  u1 = self.meta[i, j].unit
                  u2 = other.meta[i, j].unit
                  if u1 != u2:
                       # Allow None vs None
                       if u1 is None and u2 is None: continue
                       if u1 is None or u2 is None:
                            raise ValueError(f"Unit mismatch at meta ({i},{j}): {u1} vs {u2}")
                       if not u1.is_equivalent(u2):
                            raise ValueError(f"Unit mismatch at meta ({i},{j}): {u1} vs {u2}")

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

        if isinstance(key, (list, np.ndarray)) and len(key) > 0 and isinstance(key[0], str):
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


        if self.ndim == 3: # (Batch, Time, Freq)
            if isinstance(key, (int, np.integer)):
                is_single_element = True
                r_idx = key
                c_idx = 0
        elif self.ndim == 4: # (Row, Col, Time, Freq)
            if isinstance(key, tuple) and len(key) >= 2:
                r, c = key[0], key[1]
                if isinstance(r, (int, np.integer)) and isinstance(c, (int, np.integer)):
                     is_single_element = True
                     r_idx, c_idx = r, c

        if is_single_element:
             # Return Spectrogram
             meta = self.meta[r_idx, c_idx] if self.meta is not None else None
             unit = meta.unit if meta else self.unit
             name = (meta.name if meta and meta.name else self.name)
             channel = meta.channel if meta else None

             # raw_data should be (Time, Freq)
             if raw_data.ndim != 2:
                 # Should not happen if indices are correct for 3D/4D
                 raise ValueError(f"Extracted data has wrong dimension for Spectrogram: {raw_data.ndim} (expected 2)")

             return self.series_class(
                 raw_data,
                 times=self.times,
                 frequencies=self.frequencies,
                 unit=unit,
                 name=name,
                 channel=channel,
                 epoch=getattr(self, 'epoch', None)
             )

        # Return Sub-Matrix
        # We assume raw_data is ndarray. View as SpectrogramMatrix.
        ret = np.asarray(raw_data).view(type(self))
        ret._value = ret.view(np.ndarray)

        # Propagate global props
        ret.times = getattr(self, 'times', None)
        ret.frequencies = getattr(self, 'frequencies', None)
        ret.unit = getattr(self, 'unit', None)
        ret.epoch = getattr(self, 'epoch', None)

        # Propagate/Slice Metadata (Rows, Cols, Meta)
        # This is complex for general slicing. Simplification:
        # If ndim preserved, try to slice rows/cols.
        # If ndim reduced (e.g. 4D -> 3D), adjust.
        # Basic case: Batch slicing on 3D or Row slicing on 4D

        main_key = key[0] if isinstance(key, tuple) else key

        # 3D: (Batch, T, F) -> Slice batch
        if self.ndim == 3 and ret.ndim == 3:
             if self.rows:
                 ret.rows = _slice_metadata_dict(self.rows, main_key, 'row')
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

             if self.rows: ret.rows = _slice_metadata_dict(self.rows, r_key, 'row')
             if self.cols: ret.cols = _slice_metadata_dict(self.cols, c_key, 'col')
             if self.meta is not None:
                  try:
                      # meta is (Row, Col)
                      # If key is simple tuple (slice, slice)
                      if isinstance(key, tuple) and len(key) <= 2:
                           ret.meta = self.meta[key]
                      else:
                           # complex slicing?
                           pass
                  except Exception:
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
