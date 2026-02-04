# TimeSeries

**Inherits from:** `gwpy.timeseries.TimeSeries`

Extended TimeSeries with full gwexpy functionality.

See {doc}`api/timeseries` for the API reference.

## Statistics / association

- `correlation(other, method="pearson"|"kendall"|"mic"|"distance")`
- `partial_correlation(other, controls=..., method="residual"|"precision")`
- `fastmi(other, grid_size=..., quantile=...)`
- `granger_causality(other, maxlag=..., test=...)`

## Pickle / shelve portability

.. warning::
   Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
   arbitrary code on load.

gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.
