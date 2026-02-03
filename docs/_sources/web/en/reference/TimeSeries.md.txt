# TimeSeries

**Inherits from:** `gwpy.timeseries.TimeSeries`

Extended TimeSeries with full gwexpy functionality.

See {doc}`api/timeseries` for the API reference.

## Pickle / shelve portability

.. warning::
   Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
   arbitrary code on load.

gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.
