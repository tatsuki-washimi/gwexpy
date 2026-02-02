# Spectrogram

**Inherits from:** `gwpy.spectrogram.Spectrogram`

Extended Spectrogram with gwexpy analysis and visualization helpers.

See {doc}`api/spectrogram` for the API reference.

## Pickle / shelve portability

.. warning::
   Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
   arbitrary code on load.

gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.
