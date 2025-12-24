"""
Spectral matrix calculation helpers for TimeSeries collections.
"""

import numpy as np
from astropy import units as u
from gwexpy.frequencyseries.frequencyseries import FrequencySeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix

def _get_series_list_and_names(collection):
    if isinstance(collection, dict):
        # TimeSeriesDict-like
        return list(collection.values()), list(collection.keys())
    elif isinstance(collection, list):
        # TimeSeriesList-like
        names = []
        for i, item in enumerate(collection):
            name = getattr(item, 'name', None)
            if not name:
                name = f"ch{i}"
            names.append(name)
        return collection, names
    else:
        raise TypeError(f"Unsupported collection type: {type(collection)}")

def csd_matrix_from_collection(
    collection,
    other=None,
    *,
    fftlength=None,
    overlap=None,
    window='hann',
    hermitian=True,
    include_diagonal=True,
    **kwargs
):
    series_rows, names_rows = _get_series_list_and_names(collection)

    if other is None:
        series_cols = series_rows
        names_cols = names_rows
        is_symmetric_input = True
    else:
        series_cols, names_cols = _get_series_list_and_names(other)
        is_symmetric_input = False

    n_rows = len(series_rows)
    n_cols = len(series_cols)

    matrix_data = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    ref_freqs = None

    # Iterate
    for i in range(n_rows):
        start_j = 0
        if is_symmetric_input and hermitian:
             start_j = i if include_diagonal else i + 1

        for j in range(start_j, n_cols):
             if is_symmetric_input and not include_diagonal and i == j:
                  continue

             # If hermitian symmetry, we might skip lower triangle computation
             # Logic above: start_j <= j. So we visit upper triangle (i <= j).
             # We only visit loop if j >= start_j.
             # So we compute upper triangle.

             ts_row = series_rows[i]
             ts_col = series_cols[j]

             # GWpy csd: correlation of (ts_row, ts_col)
             val = ts_row.csd(ts_col, fftlength=fftlength, overlap=overlap, window=window, **kwargs)

             if ref_freqs is None:
                  ref_freqs = val.frequencies

             matrix_data[i][j] = val

             if is_symmetric_input and hermitian and i != j:
                  if i < n_cols and j < n_rows: # Bound check (always true if symmetric input)
                      val_conj = val.conjugate()
                      # Ensure name is updated? GWpy csd name is usually "CSD".
                      if val_conj.name:
                          val_conj.name = val.name
                      matrix_data[j][i] = val_conj

    if ref_freqs is None:
         if n_rows > 0 and n_cols > 0:
              # Fallback compute
              idx_row = 0
              idx_col = 0 if not (is_symmetric_input and not include_diagonal) else (1 if n_cols > 1 else 0)
              if is_symmetric_input and not include_diagonal and n_cols <= 1:
                   # No off-diagonal exists
                   # Just compute diag for shape
                   idx_col = 0

              val = series_rows[idx_row].csd(series_cols[idx_col], fftlength=fftlength, overlap=overlap, window=window, **kwargs)
              ref_freqs = val.frequencies
         else:
              return FrequencySeriesMatrix(np.zeros((0,0,0)), rows=[], cols=[])

    freq_len = len(ref_freqs)
    dtype = np.complex128

    out_value = np.zeros((n_rows, n_cols, freq_len), dtype=dtype)
    meta_array = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
        for j in range(n_cols):
             item = matrix_data[i][j]
             if item is None:
                  out_value[i, j, :] = 0
                  meta_array[i][j] = MetaData(
                       unit=getattr(matrix_data[0][0] if matrix_data[0][0] else u.dimensionless_unscaled, 'unit', None),
                       name="",
                       channel=None
                  )
             else:
                  # Verify frequency alignment could be added here
                  out_value[i, j, :] = item.value
                  meta_array[i][j] = MetaData(
                       unit=item.unit,
                       name=item.name,
                       channel=item.channel
                  )

    meta_mat = MetaDataMatrix(meta_array)
    return FrequencySeriesMatrix(
         out_value,
         frequencies=ref_freqs,
         meta=meta_mat,
         rows=names_rows,
         cols=names_cols
    )

def coherence_matrix_from_collection(
    collection,
    other=None,
    *,
    fftlength=None,
    overlap=None,
    window='hann',
    symmetric=True,
    include_diagonal=True,
    diagonal_value=1.0,
    **kwargs
):
    series_rows, names_rows = _get_series_list_and_names(collection)

    if other is None:
        series_cols = series_rows
        names_cols = names_rows
        is_symmetric_input = True
    else:
        series_cols, names_cols = _get_series_list_and_names(other)
        is_symmetric_input = False

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

             val = ts_row.coherence(ts_col, fftlength=fftlength, overlap=overlap, window=window, **kwargs)

             if ref_freqs is None:
                  ref_freqs = val.frequencies

             matrix_data[i][j] = val

             if is_symmetric_input and symmetric and i != j:
                  if i < n_cols and j < n_rows:
                       matrix_data[j][i] = val

    if ref_freqs is None:
         if n_rows > 0:
              val = series_rows[0].coherence(series_rows[0], fftlength=fftlength, overlap=overlap, window=window, **kwargs)
              ref_freqs = val.frequencies
         else:
              return FrequencySeriesMatrix(np.zeros((0,0,0)), rows=[], cols=[])

    freq_len = len(ref_freqs)
    out_value = np.zeros((n_rows, n_cols, freq_len), dtype=float)
    meta_array = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
         for j in range(n_cols):
              item = matrix_data[i][j]
              if item is None:
                   if is_symmetric_input and i == j and include_diagonal and diagonal_value is not None:
                        out_value[i, j, :] = diagonal_value
                        meta_array[i][j] = MetaData(unit=u.dimensionless_unscaled, name="coherence", channel=None)
                   else:
                        out_value[i, j, :] = 0
                        meta_array[i][j] = MetaData(unit=None, name="", channel=None)
              else:
                   out_value[i, j, :] = item.value
                   meta_array[i][j] = MetaData(
                        unit=item.unit,
                        name=item.name,
                        channel=item.channel
                   )

    meta_mat = MetaDataMatrix(meta_array)
    return FrequencySeriesMatrix(
         out_value,
         frequencies=ref_freqs,
         meta=meta_mat,
         rows=names_rows,
         cols=names_cols
    )
