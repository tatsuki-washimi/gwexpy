Frequency Series
================

.. note::
   Page role: Secondary API category

**Stability:** Stable

.. currentmodule:: gwexpy.frequencyseries

Overview
--------

.. note::
   Learning path:
   Use this page after the basic ``FrequencySeries`` tutorial or when a fitting/spectral workflow sends you back to the exact API surface.

.. seealso::

   :doc:`../../user_guide/tutorials/index`
      Tutorial hub for feature-first learning paths.
   :doc:`../../user_guide/tutorials/intro_frequencyseries`
      Basic ``FrequencySeries`` walkthrough before API lookup.
   :doc:`../../user_guide/physics_models`
      Background on frequency-domain modeling and spectral interpretation.
   :doc:`../FFT_Conventions`
      Fourier normalization and axis conventions used by GWexpy.
   :doc:`../Spectral`
      PSD / ASD estimation helpers that commonly produce or consume ``FrequencySeries`` objects.
   :doc:`../../user_guide/tutorials/case_bootstrap_gls_fitting`
      Frequency-domain fitting workflow that maps back to this API.
   :doc:`../topics`
      Theory/concept landing for validation assumptions and convention-heavy questions.

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
