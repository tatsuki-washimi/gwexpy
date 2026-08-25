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
sidecar.  Its entries use canonical absolute HDF5 dataset names (for example
``/group/disk``), matching ``Dataset.name`` exactly.  An ordinary failed
update restores the original dataset link and sidecar state.  If restoration
or rollback cleanup fails, GWexpy attempts to
retain a named recovery artifact and reports the operation plus every
restoration, preservation, and cleanup failure.  A reported artifact contains
an actionable saved dataset or prior-sidecar snapshot; an empty or unusable
group is reported as unavailable.  A sidecar-only artifact is actionable only
when its exact boolean absence marker, or its bounded serialized prior sidecar,
passes the normal JSON, path, and provenance-schema validation; inspection does
not mutate either artifact or public state.  Recovery errors are listed in the
order they occurred while their restoration, preservation, and cleanup
categories remain available separately.  Error messages use bounded safe
descriptions, retaining the original exception objects without invoking
untrusted exception formatting.  Rollback errors always retain at least one
causal exception; an invalid internal construction records a synthetic
invariant error rather than exposing an empty or misleading rollback state.
Valid internal rollback states use the exact identity-preserving event order:
operation then restoration or cleanup then preservation; a committed write has
cleanup/preservation failures only.
If artifact creation also fails, the error
reports that recovery is unavailable after retrying the prior sidecar snapshot
directly.  If data and sidecar commit before rollback cleanup
fails, the write remains committed and its structured error marks
``operation_committed=True``.  Path
replacement writes a complete sibling temporary HDF5 file before ``os.replace``.
For an existing regular file, its permission bits are applied to the temporary
replacement.  Symbolic-link targets are rejected; ownership, ACLs, and
extended attributes are not preserved by this operation.
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
