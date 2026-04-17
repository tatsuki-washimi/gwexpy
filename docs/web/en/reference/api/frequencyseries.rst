Frequency Series
================

**Stability:** Stable

.. currentmodule:: gwexpy.frequencyseries

Overview
--------

.. seealso::

   :doc:`../../user_guide/physics_models`
      Background on frequency-domain modeling and spectral interpretation.
   :doc:`../FFT_Conventions`
      Fourier normalization and axis conventions used by GWexpy.
   :doc:`../Spectral`
      PSD / ASD estimation helpers that commonly produce or consume ``FrequencySeries`` objects.
   :doc:`../../user_guide/tutorials/case_bootstrap_gls_fitting`
      Frequency-domain fitting workflow that maps back to this API.

.. autosummary::
   :toctree: _autosummary

   FrequencySeries

FrequencySeries Class
---------------------

.. autoclass:: FrequencySeries
   :no-index:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: Methods

   .. autosummary::

      ~FrequencySeries.ifft
      ~FrequencySeries.idct
      ~FrequencySeries.rms
      ~FrequencySeries.abs
      ~FrequencySeries.angle
      ~FrequencySeries.phase
      ~FrequencySeries.filter

Module Contents
---------------

.. automodule:: gwexpy.frequencyseries
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: FrequencySeries
