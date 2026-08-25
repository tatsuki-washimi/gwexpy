Spectrogram
===========

.. note::
   Page role: Secondary API category

**Stability:** Stable

.. currentmodule:: gwexpy.spectrogram

Overview
--------

Provenance
----------

``Spectrogram.provenance`` is an optional detached mapping for analysis
results.  Its stable v0.2.0 form has
``schema="gwexpy.spectrogram.provenance"`` and ``schema_version=1``.
Only JSON-safe values are accepted; live objects such as random-number
generators are rejected.  The mapping is retained by copy, slicing,
Spectrogram-preserving arithmetic, pickle, and explicit HDF5 round-trips.
The HDF5 sidecar is also retained when GWpy infers HDF5 from a ``.h5`` or
``.hdf5`` filename.  A ``.hdf`` filename requires ``format="hdf5"``.
HDF5 stores it as a GWexpy file-level sidecar, so the native GWpy dataset
remains readable by GWpy.
The sidecar is validated before a write and limited to 1 MB.  Within a
process, provenance-aware reads and updates of the same physical file share a
per-file lock, so a reader does not observe replacement data with an older
sidecar.  An ordinary failed update restores the original dataset link and
sidecar state.  If restoration itself fails, the original is retained under a
named recovery artifact and the raised error reports both failures.  Path
replacement writes a complete sibling temporary HDF5 file before ``os.replace``.
The per-file lock and rollback do not provide a cross-process HDF5 transaction;
that guarantee is outside this v0.2.0 scope.  Pickles
without provenance remain GWpy-portable; unpickling a provenance-bearing
Spectrogram requires GWexpy.

.. note::
   Learning path:
   Use this page after the introductory spectrogram tutorial or when a time-frequency workflow needs exact API details.

.. seealso::

   :doc:`../../user_guide/tutorials/index`
      Tutorial hub for feature-first learning paths.
   :doc:`../../user_guide/tutorials/intro_spectrogram`
      Basic ``Spectrogram`` walkthrough before API lookup.
   :doc:`../FFT_Conventions`
      Fourier normalization and axis conventions used by GWexpy.
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      Time-frequency case study that maps back to ``Spectrogram`` operations.
   :doc:`../../user_guide/numerical_stability`
      Stability considerations for FFT-driven time-frequency analysis.
   :doc:`../topics`
      Theory/concept landing for convention-heavy and advanced/theory questions.

.. autosummary::
   :toctree: _autosummary

   Spectrogram

Spectrogram Class
-----------------

.. autoclass:: Spectrogram
   :no-index:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: Methods

   .. autosummary::

      ~Spectrogram.plot
      ~Spectrogram.crop
      ~Spectrogram.percentile
      ~Spectrogram.ratio
      ~Spectrogram.filter

Module Contents
---------------

.. automodule:: gwexpy.spectrogram
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Spectrogram
