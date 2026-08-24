from __future__ import annotations

from typing import Any, cast

import numpy as np
from astropy import units as u

from gwexpy.types.metadata import MetaDataDict, MetaDataMatrix
from gwexpy.types.mixin import PhaseMethodsMixin
from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.seriesmatrix_base import _copy_metadata_dict, _scalar_power_exponent
from gwexpy.types.typing import ArrayLike, IndexLike, MetaDataCollectionType, UnitLike

from .collections import SpectrogramDict, SpectrogramList
from .matrix_analysis import SpectrogramMatrixAnalysisMixin
from .matrix_core import SpectrogramMatrixCoreMixin
from .spectrogram import Spectrogram


class SpectrogramMatrix(  # type: ignore[misc]
    PhaseMethodsMixin,
    SpectrogramMatrixCoreMixin,
    SpectrogramMatrixAnalysisMixin,
    SeriesMatrix,
):
    """Evaluation Matrix for Spectrograms (Time-Frequency maps).

    `SpectrogramMatrix` represents a collection of Spectrograms,
    structured as a multivariate matrix with dimensions either:

    - 3D: ``(Batch, Time, Frequency)``
    - 4D: ``(Row, Col, Time, Frequency)``

    It extends the core `~gwexpy.types.seriesmatrix.SeriesMatrix` with
    spectrogram-specific axes (times and frequencies) and analysis
    methods.

    Parameters
    ----------
    data : array-like
        The data values for the matrix. Should be 3D or 4D.

    times : array-like, optional
        The time values corresponding to each row.

    frequencies : array-like, optional
        The frequency values corresponding to each column.

    unit : `str`, `~astropy.units.Unit`, optional
        Physical unit of the data.

    **kwargs
        Additional keyword arguments passed to the
        `~gwexpy.types.seriesmatrix.SeriesMatrix` constructor.

    Notes
    -----
    Serialization is supported via HDF5 and Pickle. Metadata is
    preserved per-element in the `meta` attribute.

    Key methods:

    .. autosummary::

       ~SpectrogramMatrix.plot_summary
       ~SpectrogramMatrix.to_dict
       ~SpectrogramMatrix.to_list
       ~SpectrogramMatrix.radian

    Examples
    --------
    >>> from gwexpy.spectrogram import SpectrogramMatrix
    >>> import numpy as np
    >>> data = np.ones((1, 2, 2))
    >>> sm = SpectrogramMatrix(data, times=[0, 1], frequencies=[10, 20])
    >>> sm
    <SeriesMatrix shape=(1, 2, 2) rows=('batch0',) cols=('col0',)>

    """

    series_class = Spectrogram
    dict_class = SpectrogramDict
    list_class = SpectrogramList

    def __new__(
        cls,
        data: ArrayLike | SpectrogramMatrix,
        times: IndexLike | None = None,
        frequencies: IndexLike | None = None,
        unit: UnitLike = None,
        name: str | None = None,
        rows: MetaDataCollectionType = None,
        cols: MetaDataCollectionType = None,
        meta: Any = None,
        **kwargs: Any,
    ) -> SpectrogramMatrix:
        """Create a new `SpectrogramMatrix` instance."""
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
            obj.rows = None  # type: ignore[assignment]
            obj.cols = None  # type: ignore[assignment]
            obj.meta = None  # type: ignore[assignment]

        # Apply unit to metadata if needed (only if not explicitly set in meta)
        if unit is not None and obj.meta is not None:
            for m in obj.meta.reshape(-1):
                # MetaData defaults to dimensionless_unscaled, so check for that too
                if m.unit is None or m.unit == u.dimensionless_unscaled:
                    m.unit = unit

        # If no global unit was provided, infer it from metadata if consistent
        if obj.unit is None and obj.meta is not None:
            meta_units = {m.unit for m in obj.meta.reshape(-1) if m is not None}
            if len(meta_units) == 1:
                obj.unit = next(iter(meta_units))

        obj.epoch = kwargs.get("epoch", 0.0)
        obj._value = obj.view(np.ndarray)
        return obj

    @staticmethod
    def _resupplied_frequencies(frequencies: Any) -> Any:
        """Return an independent copy of *frequencies* for a rebuilt matrix."""
        if frequencies is None:
            return None
        try:
            return frequencies.copy()
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            from copy import deepcopy

            return deepcopy(frequencies)

    def copy(self, order="C"):
        """Create a deep copy of this matrix, including the frequency axis.

        The inherited `~gwexpy.types.series_matrix_structure.SeriesMatrixStructureMixin.copy`
        only knows about the row/col/xindex metadata shared by every
        `~gwexpy.types.seriesmatrix.SeriesMatrix`; it does not resupply
        `frequencies` -- a `SpectrogramMatrix`-specific axis -- so a bare call
        silently dropped `frequencies`/`f0`/`df` (and anything derived from
        them, such as `clip`/`round`, which rebuild via `copy`).
        """
        new = super().copy(order=order)
        new.frequencies = self._resupplied_frequencies(self.frequencies)
        return new

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        """Cast matrix data to *dtype*, including the frequency axis.

        `_rebuild_with_values` (used by `clip`/`round`) falls back to `astype`
        instead of `copy` whenever the operation changes dtype -- e.g.
        clipping an integer-valued matrix against float or `Quantity` bounds.
        The inherited
        `~gwexpy.types.series_matrix_structure.SeriesMatrixStructureMixin.astype`
        does not resupply `frequencies` either, so that path silently dropped
        it the same way the un-overridden `copy` used to.
        """
        new = super().astype(
            dtype, order=order, casting=casting, subok=subok, copy=copy
        )
        if new is self:
            return new
        new.frequencies = self._resupplied_frequencies(self.frequencies)
        return new

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        super().__array_finalize__(obj)
        self.frequencies = getattr(obj, "frequencies", None)

        # Propagate custom attributes (similar to TimeSeriesCore)
        for key in getattr(obj, "__dict__", {}):
            if key.startswith("_gwex_"):
                setattr(self, key, getattr(obj, key))

        if not hasattr(self, "_value"):
            self._value = self.view(np.ndarray)

    def _cell_indices(self):
        """Yield ``(meta_index, data_index)`` pairs for every spectrogram cell.

        A 3-D matrix is ``(Batch, Time, Freq)`` with ``(N, 1)`` metadata, so
        batch ``i`` maps to metadata cell ``(i, 0)``.  A 4-D matrix is
        ``(Row, Col, Time, Freq)`` and the two indices coincide.
        """
        if self.meta is None:
            return
        if self.ndim == 3:
            for i in range(self.meta.shape[0]):
                yield (i, 0), (i,)
        else:
            for i in range(self.meta.shape[0]):
                for j in range(self.meta.shape[1]):
                    yield (i, j), (i, j)

    def _add_sub_reference_unit(self, inputs, meta_idx):
        """Return the unit that cell *meta_idx* is expressed in for add/sub.

        The leftmost operand fixes the result unit, matching both astropy and
        `SeriesMatrix`: ``1 * u.cm + sgm(unit=m)`` comes out in centimetres,
        ``sgm(unit=m) + 1 * u.cm`` in metres.
        """
        first = inputs[0]
        if isinstance(first, u.Quantity):
            return first.unit
        if isinstance(first, u.UnitBase):
            return first
        return self.meta[meta_idx].unit or u.dimensionless_unscaled

    def _convert_operand_for_add_sub(self, operand, values, inputs):
        """Rescale *values* into the leftmost operand's per-cell units.

        Parameters
        ----------
        operand : SpectrogramMatrix, Quantity or other
            The original operand the values came from; it supplies the source
            unit (per cell for a matrix, globally for a ``Quantity``).
        values : numpy.ndarray
            The raw values extracted from *operand*.
        inputs : tuple
            The full ufunc operand tuple, used to locate the reference unit.

        Returns
        -------
        numpy.ndarray
            *values* when no cell needs rescaling, otherwise a converted copy.

        Notes
        -----
        Before gwexpy 0.1.13 the value was taken straight from
        ``Quantity.value`` and only the *units* were combined, so
        ``sgm(unit=m) + 1 * u.cm`` added 1 m instead of 0.01 m (issue #576).
        Plain numbers stay unconverted: they are treated as already being in
        the reference unit, which is the long-standing behaviour.

        """
        if isinstance(operand, SpectrogramMatrix):

            def source_unit_at(idx):
                return operand.meta[idx].unit

        elif isinstance(operand, u.Quantity):

            def source_unit_at(idx):
                return operand.unit

        else:
            return values

        converted = None
        broadcast = np.shape(values) != np.shape(self)
        for meta_idx, data_idx in self._cell_indices():
            target = self._add_sub_reference_unit(inputs, meta_idx)
            source = source_unit_at(meta_idx) or u.dimensionless_unscaled
            if source == target:
                continue
            if not source.is_equivalent(target):
                raise u.UnitConversionError(
                    f"Unit mismatch at element {meta_idx}: {source} vs {target}"
                )
            if converted is None:
                base = np.broadcast_to(values, np.shape(self)) if broadcast else values
                converted = np.array(
                    base, dtype=np.result_type(np.asarray(values).dtype, np.float64)
                )
            converted[data_idx] = u.Quantity(converted[data_idx], source).to_value(
                target
            )
        return values if converted is None else converted

    def _ufunc_dispatch(self, ufunc, method, *inputs, **kwargs):
        """Apply *ufunc* to this matrix, propagating per-element units.

        Handles both `SpectrogramMatrix` layouts -- ``(Batch, Time, Freq)`` and
        ``(Row, Col, Time, Freq)`` -- which is why this overrides the
        `SeriesMatrix` implementation instead of reusing it.

        Per-element units live in the `MetaDataMatrix`:

        - Scalar operations apply the ufunc to each element's unit.
        - Binary matrix operations convert the right operand into the left
          operand's per-element units and raise ``UnitConversionError`` when a
          pair is dimensionally incompatible.

        Like `SeriesMatrix`, this is reached only through the explicit
        operators: ``__array_ufunc__`` is ``None``, so NumPy never calls it and
        ufunc methods other than ``__call__`` are rejected rather than silently
        dropping metadata.
        """
        from gwexpy.types.metadata import MetaData, MetaDataMatrix

        if method != "__call__":
            raise TypeError(
                f"SpectrogramMatrix does not support ufunc method {method!r}; "
                "only '__call__' propagates per-element metadata"
            )
        if kwargs.get("out") is not None:
            raise TypeError(
                "SpectrogramMatrix does not support the 'out' argument; "
                "use an in-place operator (e.g. 'a *= b') instead"
            )
        if kwargs.get("where", True) is not True:
            raise TypeError("SpectrogramMatrix does not support the 'where' argument")

        # Identify ufunc category for unit handling
        _ADD_SUB_UFUNCS = {np.add, np.subtract}
        _COMPARISON_UFUNCS = {
            np.less,
            np.less_equal,
            np.equal,
            np.not_equal,
            np.greater,
            np.greater_equal,
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
            elif isinstance(
                inp, (u.Quantity, np.ndarray, float, int, complex, np.number)
            ):
                # np.number (e.g. np.int64) does not subclass Python int, so
                # it needs its own branch here even though it is already
                # accepted by ``_scalar_power_exponent`` downstream -- without
                # this, ``matrix * np.int64(2)`` regressed to a bare
                # ``TypeError`` from ``__array_ufunc__=None`` on this class
                # (the sibling ``SeriesMatrix`` implementation in
                # ``seriesmatrix_base.py`` accepts it via the same check).
                val = getattr(inp, "value", inp)
                args.append(np.asarray(val))
                scalar_inputs.append(inp)
            elif isinstance(inp, u.UnitBase) and ufunc in _MUL_DIV_UFUNCS:
                # A bare Unit carries no values, so it is only meaningful as a
                # multiplier; ``sgm + u.s`` has no defensible reading.
                args.append(1.0)
                scalar_inputs.append(inp)
            else:
                return NotImplemented

        if not sgm_inputs:
            return NotImplemented

        main = sgm_inputs[0]

        # A raw ndarray has no unit metadata.  Astropy-compatible add/sub
        # therefore refuses it for a dimensional spectrogram instead of
        # treating its values as if they were already expressed in each cell
        # unit.  The check happens before any result allocation so the same
        # refusal is atomic for `+=`/`-=` through the shared in-place path.
        if (
            ufunc in _ADD_SUB_UFUNCS
            and any(
                isinstance(inp, np.ndarray)
                and not isinstance(inp, (SpectrogramMatrix, u.Quantity))
                for inp in inputs
            )
            and not main._matrix_is_strictly_dimensionless(main)
        ):
            raise TypeError(
                "SpectrogramMatrix add/sub with a raw ndarray requires "
                "strictly dimensionless matrix cells"
            )

        # 1b. Bring every other operand into `main`'s per-element units before
        #     the values are combined (issue #576).
        if ufunc in _ADD_SUB_UFUNCS or ufunc in _COMPARISON_UFUNCS:
            args = [
                main._convert_operand_for_add_sub(inp, np.asarray(arg), inputs)
                for inp, arg in zip(inputs, args)
            ]

        if ufunc in {np.divide, np.true_divide, np.floor_divide, np.mod, np.remainder}:
            if len(args) == 2 and np.any(np.asarray(args[1]) == 0):
                raise ZeroDivisionError(
                    f"{ufunc.__name__} by zero in SpectrogramMatrix operation"
                )

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
                # Scalar operation: apply the ufunc to each element's unit,
                # preserving the operand order so that reflected operators
                # (``2 * u.s / sgm``) compose their units the right way round.
                for idx in np.ndindex(meta_shape):
                    old_meta = cast(MetaData, main.meta[idx])
                    old_unit = (
                        old_meta.unit if old_meta.unit else u.dimensionless_unscaled
                    )
                    if ufunc in _COMPARISON_UFUNCS:
                        new_unit = u.dimensionless_unscaled
                    elif ufunc in _ADD_SUB_UFUNCS:
                        # The values were rescaled into this unit above, so the
                        # result must be labelled with it -- astropy's own
                        # "left operand wins" rule is already baked into
                        # ``_add_sub_reference_unit``.
                        new_unit = main._add_sub_reference_unit(inputs, idx)
                    else:
                        new_unit = self._scalar_result_unit(
                            ufunc, inputs, main, old_unit
                        )
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
                        m1 = cast(MetaData, main.meta[idx])
                        m2 = cast(MetaData, other_sgm.meta[idx])
                        u1 = m1.unit if m1.unit else u.dimensionless_unscaled
                        u2 = m2.unit if m2.unit else u.dimensionless_unscaled

                        # Add/sub/comparison only need *equivalent* units: the
                        # right operand's values were already rescaled into
                        # `main`'s units above, so `m` and `cm` now agree.
                        if ufunc in _ADD_SUB_UFUNCS or ufunc in _COMPARISON_UFUNCS:
                            if not u1.is_equivalent(u2):
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
                                new_unit = (
                                    q_result.unit if hasattr(q_result, "unit") else u1
                                )
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
            meta_units = {m.unit for m in meta.reshape(-1) if m is not None}
            if len(meta_units) == 1:
                return next(iter(meta_units))
            return None

        # Reconstruct SpectrogramMatrix.  Row/column metadata is deep-copied so
        # the result never shares MetaData instances with its operands.
        if result_data.shape == main.shape:
            obj = self.__class__(
                result_data,
                times=main.times,
                frequencies=main.frequencies,
                rows=_copy_metadata_dict(main.rows, "row") if main.rows else main.rows,
                cols=_copy_metadata_dict(main.cols, "col") if main.cols else main.cols,
                meta=new_meta,
                name=main.name,
                unit=_infer_unit(new_meta),
            )
            return obj

        return result_data

    @staticmethod
    def _unit_probe(operand, matrix, matrix_unit):
        """Return a unit-carrying stand-in for *operand* in a unit calculation."""
        if operand is matrix:
            return u.Quantity(1, matrix_unit)
        if isinstance(operand, u.UnitBase):
            return u.Quantity(1, operand)
        if isinstance(operand, u.Quantity):
            return u.Quantity(1, operand.unit)
        return u.Quantity(1, u.dimensionless_unscaled)

    def _scalar_result_unit(self, ufunc, inputs, main, cell_unit):
        """Compute the result unit of a scalar operation on one element.

        ``power`` is special-cased because the exponent is a plain number: the
        generic ``ufunc(Quantity(1, base), Quantity(1, dimensionless))`` probe
        would return the base unit unchanged instead of ``base ** exponent``.
        """
        if ufunc is np.power and len(inputs) == 2:
            exponent = _scalar_power_exponent(inputs[1])
            if exponent is None:
                raise u.UnitConversionError(
                    "power with a non-scalar exponent is not supported for "
                    "SpectrogramMatrix"
                )
            return cell_unit**exponent
        probes = [self._unit_probe(inp, main, cell_unit) for inp in inputs]
        try:
            q_result = ufunc(*probes)
        except (TypeError, ValueError, u.UnitConversionError):
            return cell_unit
        return getattr(q_result, "unit", cell_unit)

    def row_keys(self):
        """Return the row metadata keys."""
        return tuple(self.rows.keys()) if self.rows else tuple()

    def col_keys(self):
        """Return the column metadata keys."""
        return tuple(self.cols.keys()) if self.cols else tuple()

    def is_compatible(self, other: Any) -> bool:
        """Check compatibility with another SpectrogramMatrix/object.

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
        if self.meta is None and other.meta is None:
            return True
        if self.meta is None or other.meta is None:
            raise ValueError("Metadata mismatch: one has metadata, the other does not")

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
        """Return the integer index for a row key."""
        if not self.rows:
            raise KeyError(f"Invalid row key: {key}")
        try:
            return list(self.row_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key):
        """Return the integer index for a column key."""
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

            # Row index (0)
            if len(new_key) > 0:
                if isinstance(new_key[0], str):
                    new_key[0] = self.row_index(new_key[0])
                elif (
                    isinstance(new_key[0], (list, np.ndarray))
                    and len(new_key[0]) > 0
                    and isinstance(new_key[0][0], str)
                ):
                    new_key[0] = [self.row_index(k) for k in new_key[0]]

            # Col index (1) - only if we have at least 2 dims relevant to metadata (4D case or 3D with abuse?)
            # SpectrogramMatrix 4D: (Row, Col, Time, Freq). 3D: (Batch, Time, Freq).
            # For 3D, col index is not applicable in the same way, but let's assume standard behavior.
            if len(new_key) > 1:
                # Check if second element is string
                if isinstance(new_key[1], str):
                    try:
                        new_key[1] = self.col_index(new_key[1])
                    except (KeyError, IndexError):
                        # If columns are not defined or key not found, it might be a time-slice?
                        # But for 4D matrix, dim 1 IS Col.
                        if self.ndim == 4:
                            raise
                        pass
                elif (
                    isinstance(new_key[1], (list, np.ndarray))
                    and len(new_key[1]) > 0
                    and isinstance(new_key[1][0], str)
                ):
                    if self.ndim == 4:
                        new_key[1] = [self.col_index(k) for k in new_key[1]]

            key = tuple(new_key)

        # Access raw data
        raw_data = self.view(np.ndarray)[key]

        # Check for scalar element extraction (returning Spectrogram)
        is_single_element = False
        r_idx, c_idx = 0, 0

        if self.ndim == 3:  # (Batch, Time, Freq)
            if isinstance(key, (int, np.integer)):
                is_single_element = True
                r_idx = int(key)
                c_idx = 0
        elif self.ndim == 4:  # (Row, Col, Time, Freq)
            if isinstance(key, tuple) and len(key) >= 2:
                r, c = key[0], key[1]
                if isinstance(r, (int, np.integer)) and isinstance(
                    c, (int, np.integer)
                ):
                    is_single_element = True
                    r_idx, c_idx = int(r), int(c)

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
                    sliced = self.meta[main_key]
                    if isinstance(sliced, np.ndarray):
                        ret.meta = sliced.view(MetaDataMatrix)  # type: ignore[assignment]
                    else:
                        ret.meta = sliced
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
                        sliced = self.meta[key]
                        if isinstance(sliced, np.ndarray):
                            ret.meta = sliced.view(MetaDataMatrix)  # type: ignore[assignment]
                        else:
                            ret.meta = sliced
                    else:
                        # complex slicing?
                        pass
                except (IndexError, TypeError, KeyError):
                    pass
        # 4D -> 3D reduction (e.g. slice out one Row or one Col)
        elif self.ndim == 4 and ret.ndim == 3:
            # Case A: row is scalar, col is slice -> result Batch is Col
            # Case B: row is slice, col is scalar -> result Batch is Row
            r_idx = key[0] if isinstance(key, tuple) else key
            col_idx: Any = (
                key[1] if isinstance(key, tuple) and len(key) > 1 else slice(None)
            )

            is_row_scalar = isinstance(r_idx, (int, np.integer))
            is_col_scalar = isinstance(col_idx, (int, np.integer))

            if is_row_scalar:
                # Batch = Col
                if self.cols:
                    ret.rows = _slice_metadata_dict(self.cols, col_idx, "row")
            elif is_col_scalar:
                # Batch = Row
                if self.rows:
                    ret.rows = _slice_metadata_dict(self.rows, r_idx, "row")

            if self.meta is not None:
                try:
                    sliced = self.meta[r_idx, col_idx]
                    if isinstance(sliced, np.ndarray):
                        # Ensure meta for 3D is (N, 1)
                        ret.meta = sliced.reshape(-1, 1).view(MetaDataMatrix)
                    else:
                        # Scalar meta?
                        pass
                except (IndexError, TypeError, KeyError):
                    pass

        # Propagate custom attributes
        for k, v in getattr(self, "__dict__", {}).items():
            if k.startswith("_gwex_"):
                ret.__dict__[k] = v

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
        for m in self.meta.reshape(-1):
            if m.unit is None:
                continue
            if not m.unit.is_equivalent(ref_unit):
                return False, ref_unit
        return True, ref_unit

    @property
    def shape3D(self):
        """Return the display-oriented 3D shape view."""
        # Override Base logic to return relevant 3D view (Batch, Time, Freq) for display?
        # Or (Row, Col, Time) if we treat Freq as hidden dim?
        # For uniformity with SeriesMatrix which is (Row, Col, Sample),
        # if we are 4D (Row, Col, Time, Freq), we might want to return (Row, Col, Time) as 'main' shape with _x_axis_index logic?
        # But core checks shape[-1].
        return self.shape

    def plot_summary(self, **kwargs):
        """Plot the matrix as side-by-side spectrograms and percentile summaries."""
        from gwexpy.plot.plot import plot_summary

        return plot_summary(self, **kwargs)

    def __reduce__(self):
        """Customize pickle serialization to ensure metadata preservation.

        Returns standard numpy reduce tuple but appends __dict__ only if not automatically handled.
        """
        picked = list(super().__reduce__())
        # picked is [func, args, state]
        # state is (version, shape, dtype, isFortran, rawdata)
        state = picked[2]

        # Append our __dict__ to state tuple to ensure it's saved
        full_state = state + (self.__dict__,)
        picked[2] = full_state
        return tuple(picked)

    def __setstate__(self, state):
        """Restore state from pickle."""
        # The last element contains our __dict__
        my_dict = state[-1]

        # The rest is for numpy
        np_state = state[:-1]

        super().__setstate__(np_state)
        self.__dict__.update(my_dict)
