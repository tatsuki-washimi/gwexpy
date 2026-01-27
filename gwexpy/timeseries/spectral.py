"""
Spectral matrix calculation helpers for TimeSeries collections.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix


def _get_series_list_and_names(collection):
    if isinstance(collection, dict):
        # TimeSeriesDict-like
        return list(collection.values()), list(collection.keys())
    elif isinstance(collection, list):
        # TimeSeriesList-like
        names = []
        for i, item in enumerate(collection):
            name = getattr(item, "name", None)
            if not name:
                name = f"ch{i}"
            names.append(name)
        return collection, names
    else:
        raise TypeError(f"Unsupported collection type: {type(collection)}")


def _normalize_dt_seconds(dt, method_name):
    if dt is None:
        raise ValueError(f"{method_name} requires TimeSeries with dt set")
    if isinstance(dt, u.Quantity):
        unit = dt.unit
        if (
            unit is None
            or unit == u.dimensionless_unscaled
            or getattr(unit, "physical_type", None) == "dimensionless"
        ):
            return float(np.asarray(dt.value).flat[0])
        try:
            return float(dt.to(u.s).value)
        except u.UnitConversionError as exc:
            raise ValueError(f"{method_name} requires dt with time units") from exc
    try:
        return float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{method_name} requires numeric dt") from exc


def _validate_dt_and_fftlength(series_rows, series_cols, fftlength, method_name):
    if fftlength is None:
        raise ValueError(f"{method_name} requires fftlength")
    if not series_rows and not series_cols:
        return
    series_all = list(series_rows) + list(series_cols)
    ref_dt = None
    for ts in series_all:
        dt = getattr(ts, "dt", None)
        dt_sec = _normalize_dt_seconds(dt, method_name)
        if ref_dt is None:
            ref_dt = dt_sec
            continue
        if not np.isclose(dt_sec, ref_dt, rtol=1e-12, atol=0.0):
            raise ValueError(f"{method_name} requires common dt; mismatch in dt")


def csd_matrix_from_collection(
    collection,
    other=None,
    *,
    fftlength=None,
    overlap=None,
    window="hann",
    hermitian=True,
    include_diagonal=True,
    **kwargs,
):
    """
    Compute Cross-Spectral Density (CSD) matrix for a TimeSeries collection.

    For self-matrices (other is None), the diagonal is always the PSD (auto PSD)
    and must be computed; include_diagonal=False is not allowed.
    Uncomputed elements are represented as complex NaN.
    The frequency axis is taken from the first computed element without
    frequency alignment/truncation. Instead, dt and fftlength consistency is
    enforced before computation and mismatches raise ValueError.
    """
    series_rows, names_rows = _get_series_list_and_names(collection)

    if other is None:
        series_cols = series_rows
        names_cols = names_rows
        is_symmetric_input = True
    else:
        series_cols, names_cols = _get_series_list_and_names(other)
        is_symmetric_input = False

    if include_diagonal is False:
        raise ValueError(
            "CSD matrix requires diagonal PSD; include_diagonal=False is not allowed"
        )
    _validate_dt_and_fftlength(
        series_rows, series_cols, fftlength, "csd_matrix_from_collection"
    )

    n_rows = len(series_rows)
    n_cols = len(series_cols)

    matrix_data = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    ref_freqs = None
    ref_unit = None

    # Iterate
    for i in range(n_rows):
        start_j = 0
        if is_symmetric_input and hermitian:
            start_j = i

        for j in range(start_j, n_cols):
            ts_row = series_rows[i]
            ts_col = series_cols[j]

            if is_symmetric_input and i == j:
                val = ts_row.psd(
                    fftlength=fftlength, overlap=overlap, window=window, **kwargs
                )
            else:
                val = ts_row.csd(
                    ts_col,
                    fftlength=fftlength,
                    overlap=overlap,
                    window=window,
                    **kwargs,
                )

            if ref_freqs is None:
                ref_freqs = val.frequencies
                ref_unit = val.unit

            matrix_data[i][j] = val

            if is_symmetric_input and hermitian and i != j:
                if (
                    i < n_cols and j < n_rows
                ):  # Bound check (always true if symmetric input)
                    val_conj = val.conjugate()
                    if val_conj.name:
                        val_conj.name = val.name
                    matrix_data[j][i] = val_conj

    if ref_freqs is None:
        if n_rows > 0 and n_cols > 0:
            # Fallback compute
            idx_row = 0
            idx_col = 0
            if is_symmetric_input and n_cols > 0:
                val = series_rows[idx_row].psd(
                    fftlength=fftlength, overlap=overlap, window=window, **kwargs
                )
            else:
                val = series_rows[idx_row].csd(
                    series_cols[idx_col],
                    fftlength=fftlength,
                    overlap=overlap,
                    window=window,
                    **kwargs,
                )
            ref_freqs = val.frequencies
            ref_unit = val.unit
        else:
            return FrequencySeriesMatrix(np.zeros((0, 0, 0)), rows=[], cols=[])

    freq_len = len(ref_freqs)
    dtype = np.complex128

    out_value = np.full((n_rows, n_cols, freq_len), np.nan + 1j * np.nan, dtype=dtype)
    meta_array = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
        for j in range(n_cols):
            item = matrix_data[i][j]
            if item is None:
                out_value[i, j, :] = np.nan + 1j * np.nan
                meta_array[i][j] = MetaData(unit=ref_unit, name="", channel=None)
            else:
                # Verify frequency alignment could be added here
                out_value[i, j, :] = item.value
                meta_array[i][j] = MetaData(
                    unit=item.unit, name=item.name, channel=item.channel
                )

    meta_mat = MetaDataMatrix(meta_array)
    return FrequencySeriesMatrix(
        out_value,
        frequencies=ref_freqs,
        meta=meta_mat,
        rows=names_rows,
        cols=names_cols,
    )


def coherence_matrix_from_collection(
    collection,
    other=None,
    *,
    fftlength=None,
    overlap=None,
    window="hann",
    symmetric=True,
    include_diagonal=True,
    diagonal_value=1.0,
    **kwargs,
):
    """
    Compute coherence matrix for a TimeSeries collection.

    If include_diagonal is True and diagonal_value is not None, the diagonal is
    filled with that value without computing coherence. If diagonal_value is
    None, the diagonal is computed via ts.coherence(ts). If include_diagonal is
    False, the diagonal remains uncomputed (NaN).
    Uncomputed elements are represented as NaN.
    The frequency axis is taken from the first computed element without
    frequency alignment/truncation. Instead, dt and fftlength consistency is
    enforced before computation and mismatches raise ValueError.
    """
    series_rows, names_rows = _get_series_list_and_names(collection)

    if other is None:
        series_cols = series_rows
        names_cols = names_rows
        is_symmetric_input = True
    else:
        series_cols, names_cols = _get_series_list_and_names(other)
        is_symmetric_input = False

    _validate_dt_and_fftlength(
        series_rows, series_cols, fftlength, "coherence_matrix_from_collection"
    )

    n_rows = len(series_rows)
    n_cols = len(series_cols)

    matrix_data = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    ref_freqs = None

    for i in range(n_rows):
        start_j = 0
        if is_symmetric_input and symmetric:
            start_j = i

        for j in range(start_j, n_cols):
            if is_symmetric_input and i == j:
                if not include_diagonal:
                    continue
                if diagonal_value is not None:
                    continue

            ts_row = series_rows[i]
            ts_col = series_cols[j]

            val = ts_row.coherence(
                ts_col, fftlength=fftlength, overlap=overlap, window=window, **kwargs
            )

            if ref_freqs is None:
                ref_freqs = val.frequencies

            matrix_data[i][j] = val

            if is_symmetric_input and symmetric and i != j:
                if i < n_cols and j < n_rows:
                    matrix_data[j][i] = val.copy()

    if ref_freqs is None:
        if n_rows > 0:
            val = series_rows[0].coherence(
                series_rows[0],
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                **kwargs,
            )
            ref_freqs = val.frequencies
        else:
            return FrequencySeriesMatrix(np.zeros((0, 0, 0)), rows=[], cols=[])

    freq_len = len(ref_freqs)
    out_value = np.full((n_rows, n_cols, freq_len), np.nan, dtype=float)
    meta_array = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
        for j in range(n_cols):
            item = matrix_data[i][j]
            if item is None:
                if (
                    is_symmetric_input
                    and i == j
                    and include_diagonal
                    and diagonal_value is not None
                ):
                    out_value[i, j, :] = diagonal_value
                    meta_array[i][j] = MetaData(
                        unit=u.dimensionless_unscaled, name="coherence", channel=None
                    )
                else:
                    meta_array[i][j] = MetaData(
                        unit=u.dimensionless_unscaled, name="", channel=None
                    )
            else:
                out_value[i, j, :] = item.value
                meta_array[i][j] = MetaData(
                    unit=item.unit, name=item.name, channel=item.channel
                )

    meta_mat = MetaDataMatrix(meta_array)
    return FrequencySeriesMatrix(
        out_value,
        frequencies=ref_freqs,
        meta=meta_mat,
        rows=names_rows,
        cols=names_cols,
    )
