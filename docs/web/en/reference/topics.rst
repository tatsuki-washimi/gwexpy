Reference Topics
================

.. note::
   Page role: Theory and concept landing

**Stability:** Stable

Use this page when you want to browse the reference by concept rather than by class or module name.

**Audience:** Users who know the analytical question they are asking, but need the matching convention, theory note, or helper surface.
**Use this page for:** Starting from concepts such as Fourier conventions, validation assumptions, spectral estimation, or compatibility layers.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Topic
     - Stability
     - Start here
   * - Theory and conventions
     - Stable
     - :doc:`FFT_Conventions`, :doc:`../user_guide/prerequisites_and_conventions`
   * - Spectral and fitting helpers
     - Stable
     - :doc:`Spectral`, :doc:`fitting`, :doc:`../user_guide/tutorials/case_bootstrap_gls_fitting`
   * - Validation and audit notes
     - Stable
     - :doc:`../user_guide/validated_algorithms`, :doc:`../user_guide/numerical_stability`
   * - Noise modeling helpers
     - Stable
     - :doc:`Noise`
   * - Compatibility and extra APIs
     - Stable
     - :doc:`api/extra`

Concept Guides
--------------

- :doc:`FFT_Conventions` for Fourier normalization, axis conventions, and API mappings.
- :doc:`../user_guide/prerequisites_and_conventions` for shared assumptions about time systems, FFT conventions, and physical interpretation.
- :doc:`Spectral` for PSD, ASD, and bootstrap-oriented estimators.
- :doc:`fitting` for least-squares, GLS, and MCMC-oriented fitting helpers.
- :doc:`../user_guide/validated_algorithms` for audit-backed assumptions, evidence, and exact API cross-links.
- :doc:`../user_guide/numerical_stability` for stabilization choices such as adaptive whitening.
- :doc:`Noise` for synthetic detector-noise and surrogate generation helpers.

Bridge Pages
------------

- :doc:`api/extra` for compatibility-oriented API entry points.
- :doc:`../user_guide/gwexpy_for_gwpy_users_en` for GWpy migration guidance.
- :doc:`../user_guide/gwpy_added_api_index_en` for GWpy difference-oriented API browsing.
- :doc:`../user_guide/tutorials/field_scalar_intro` and :doc:`../user_guide/tutorials/advanced_arima` for exact tutorial entry points that map back to the reference.

.. seealso::
   Cross-navigation:

   - :doc:`index` for the main reference hub
   - :doc:`api/index` for module/category-first API browsing
   - :doc:`../user_guide/tutorials/index` for tutorial-first navigation
   - :doc:`../user_guide/validated_algorithms` for the explicit advanced/theory user-guide landing

.. toctree::
   :maxdepth: 1

   FFT_Conventions
   Spectral
   fitting
   Noise
   api/extra
