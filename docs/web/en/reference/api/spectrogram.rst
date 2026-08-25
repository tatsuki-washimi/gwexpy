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
cleanup/preservation failures only.  Direct internal construction may omit its
optional event sequence, in which case GWexpy derives that exact order from the
validated phase fields; an explicitly supplied sequence is checked strictly.
If artifact creation also fails, the error
reports that recovery is unavailable after retrying the prior sidecar snapshot
directly.  If data and sidecar commit before rollback cleanup
fails, the write remains committed and its structured error marks
``operation_committed=True``.  For a pathname, ``overwrite=True`` with the default ``append=False`` has GWpy's
whole-file replacement semantics: GWexpy writes only the requested dataset and
its sidecar entry to a complete sibling temporary HDF5 file before ``os.replace``.
It intentionally removes unrelated datasets and provenance entries.  In
contrast, ``append=True`` mutates the existing file under the transaction lock:
different dataset paths and their sidecar entries are retained, while an
existing same path is replaced only when ``overwrite=True``.  Passing an open
``h5py.File`` or ``h5py.Group`` uses that same in-file, per-dataset mutation
behavior.  Thus two pathname replacement writers are serialized but
last-writer-wins at file scope; preservation across distinct paths requires
``append=True`` or an open container.
For an existing regular file, its permission bits are applied to the temporary
replacement.  Symbolic-link targets are rejected; ownership, ACLs, and
extended attributes are not preserved by this operation.
Provenance-aware pathname reads and writes also acquire a bounded (10 second)
POSIX advisory transaction lock on a durable sibling lock file.  Relative,
absolute, and symbolic-link pathname aliases resolve to the same lock.  The
lock covers the complete data-plus-sidecar update and recovery path, so another
GWexpy provenance-aware pathname reader or writer observes either the prior or
the committed pair for the selected operation scope.  Lock acquisition fails closed with a structured
``CrossProcessHDF5LockError``; lock files are deliberately retained and are
never treated as stale or stolen, while the operating system releases a held
lock when a process terminates.  This requires POSIX ``flock`` and a local
filesystem that honors it; unsupported platforms and anonymous file objects
fail closed.  Caller-owned ``h5py`` handles participate only while passed to a
GWexpy provenance-aware operation; independently opened or mutated handles
remain outside the transaction.  No distributed or network-filesystem
guarantee is claimed.  Pickles
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
