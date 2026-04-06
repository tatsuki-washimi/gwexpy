# FrequencySeries

**Inherits from:** `gwpy.frequencyseries.FrequencySeries`

Extended FrequencySeries with gwexpy analysis and interop features.

See {doc}`api/frequencyseries` for the API reference.

## Pickle / shelve portability

.. warning::
   Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
   arbitrary code on load.

gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.
