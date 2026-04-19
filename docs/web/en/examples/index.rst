.. meta::
   :description: Browse the canonical GWexpy case-study gallery by workflow theme, from calibration and interoperability to ML and noise hunting.

.. _examples-en-gallery-entry:

Case Studies (Goal-Oriented Workflows)
======================================

.. note::
   Page role: Guide index

A collection of goal-oriented demonstrations that combine multiple GWexpy features into practical workflows.
To learn feature-oriented examples first, refer to :doc:`../user_guide/tutorials/index`.

**Audience:** Users who already know the basics and want end-to-end examples mapped to real analysis tasks.
**Prerequisites:** Familiarity with the relevant core classes, especially from :doc:`../user_guide/tutorials/index` and :doc:`../user_guide/getting_started`.
**Use this page for:** Finding the canonical public gallery of GWexpy goal-oriented case studies by workflow theme.
**Search hints:** case studies, gallery, end-to-end workflow, calibration, interoperability, noise hunting, ML

.. note::
   `Case Studies` are goal-oriented workflow demonstrations, while `Tutorials` are feature-oriented class/capability examples.
   This page is the canonical public index for all `case_*` notebooks.

.. note::
   On this page:
   Section I focuses on calibration and control, II on interoperability and reproducibility, III on statistical and ML workflows, and IV on noise hunting and detector diagnostics.

.. note::
   Example framing:
   Goal: pick a workflow family that matches your task.
   Inputs: familiarity with the relevant GWexpy objects and the notebook environment.
   Outputs: a case-study notebook path you can read or run locally.

.. _examples-en-featured-gallery:

Featured Gallery
----------------

.. note::
   These featured cards reuse the same three teaser thumbnails shown on the homepage.
   This page remains the canonical gallery: the homepage is a short visual preview, while the full categorized list below is the source of truth.

.. note::
   Minimal visual index:

   - `Noise Budget` thumbnail -> start in :ref:`section-iv-noise-hunting-and-detector-diagnostics`
   - `Transfer Function Estimation` thumbnail -> start in :ref:`section-i-calibration-response-and-control`
   - `Active Damping` thumbnail -> start in :ref:`section-i-calibration-response-and-control`

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Noise Budget
      :img-top: /_static/images/case_noise_budget_thumb.png
      :img-alt: Thumbnail of the Bruco noise budget example
      :link: ../user_guide/tutorials/case_noise_budget
      :link-type: doc
      :text-align: center

      Identify dominant couplings and compare mitigation directions in a detector-noise workflow.

   .. grid-item-card:: Transfer Function Estimation
      :img-top: /_static/images/case_transfer_function_thumb.png
      :img-alt: Thumbnail of the transfer function estimation example
      :link: ../user_guide/tutorials/case_transfer_function
      :link-type: doc
      :text-align: center

      Estimate coherence, fit responses, and connect measured transfer paths back to the API.

   .. grid-item-card:: Active Damping
      :img-top: /_static/images/case_active_damping_thumb.png
      :img-alt: Thumbnail of the active damping example
      :link: ../user_guide/tutorials/case_active_damping
      :link-type: doc
      :text-align: center

      Follow a multi-input multi-output control workflow for a 6-DOF isolation system.

.. _examples-en-canonical-list:

.. _section-i-calibration-response-and-control:

I. Calibration, Response, and Control
-------------------------------------

- :doc:`Active Damping: MIMO control for a 6-DOF isolation system <../user_guide/tutorials/case_active_damping>`
- :doc:`Transfer Function Measurement: estimation, coherence, and fitting <../user_guide/tutorials/case_transfer_function>`
- :doc:`Calibration Pipeline: counts-to-strain conversion <../user_guide/tutorials/case_calibration_pipeline>`
- :doc:`DTT XML Workflow: loading and reusing measured response data <../user_guide/tutorials/case_dttxml_calibration>`

.. _section-ii-interoperability-io-and-reproducibility:

II. Interoperability, I/O, and Reproducibility
----------------------------------------------

- :doc:`Finesse 3 Interoperability: simulation vs. measurement <../user_guide/tutorials/case_finesse_optics>`
- :doc:`ObsPy Interoperability: ingesting and analyzing seismic data <../user_guide/tutorials/case_seismic_obspy>`
- :doc:`GBD Format I/O: round-tripping detector data products <../user_guide/tutorials/case_gbd_format>`
- :doc:`HDF5 Provenance: reproducible metadata management <../user_guide/tutorials/case_hdf5_provenance>`
- :doc:`PyCBC Interoperability: from gwexpy preprocessing to search <../user_guide/tutorials/case_pycbc_search>`

.. _section-iii-statistical-and-ml-workflows:

III. Statistical and ML Workflows
---------------------------------

- :doc:`Bootstrap PSD and GLS Fitting <../user_guide/tutorials/case_bootstrap_gls_fitting>`
- :doc:`ML Preprocessing Pipeline: feature engineering and comparison <../user_guide/tutorials/case_ml_preprocessing>`
- :doc:`Event-Synchronized Analysis: SegmentTable-driven window selection <../user_guide/tutorials/case_segment_analysis>`
- :doc:`Physical Validity Checking: units, floors, and sanity tests <../user_guide/tutorials/case_physics_validation>`
- :doc:`ARIMA-Based Burst Detection <../user_guide/tutorials/case_arima_burst_search>`
- :doc:`Signal Extraction: weak signal recovery from colored noise <../user_guide/tutorials/case_signal_extraction>`

.. _section-iv-noise-hunting-and-detector-diagnostics:

IV. Noise Hunting and Detector Diagnostics
------------------------------------------

- :doc:`Noise Budgeting: identifying dominant noise couplings <../user_guide/tutorials/case_noise_budget>`
- :doc:`Lock-in Detection: recovering weak AM/FM structure <../user_guide/tutorials/case_lockin_detection>`
- :doc:`Wiener Filtering: coherent noise subtraction <../user_guide/tutorials/case_wiener_filter>`
- :doc:`Coupling Analysis: estimating transfer paths between channels <../user_guide/tutorials/case_coupling_analysis>`
- :doc:`Bruco and ICA Noise Reduction: witness selection to subtraction <../user_guide/tutorials/case_bruco_ica_denoising>`
- :doc:`Bruco Advanced: bilinear coupling and AM/FM failure modes <../user_guide/tutorials/case_bruco_advanced>`
- :doc:`Violin Mode Analysis: fitting and tracking resonance families <../user_guide/tutorials/case_violin_mode>`
- :doc:`Schumann Resonance Analysis <../user_guide/tutorials/case_schumann_resonance>`
- :doc:`Glitch Analysis: Q-transform and Omega-scan <../user_guide/tutorials/case_glitch_analysis>`

.. note::
   For full API details (arguments, return values, class listings), see :doc:`../reference/index`.

.. seealso::
   Next to read:

   - :doc:`../user_guide/tutorials/index` to build class- and feature-level foundations first
   - :doc:`../reference/index` to look up API details behind a case-study workflow
   - :doc:`../user_guide/io_formats` if your next question is about supported formats and read/write paths

.. toctree::
   :hidden:

   ../user_guide/tutorials/case_active_damping
   ../user_guide/tutorials/case_transfer_function
   ../user_guide/tutorials/case_calibration_pipeline
   ../user_guide/tutorials/case_dttxml_calibration
   ../user_guide/tutorials/case_finesse_optics
   ../user_guide/tutorials/case_seismic_obspy
   ../user_guide/tutorials/case_gbd_format
   ../user_guide/tutorials/case_hdf5_provenance
   ../user_guide/tutorials/case_pycbc_search
   ../user_guide/tutorials/case_bootstrap_gls_fitting
   ../user_guide/tutorials/case_ml_preprocessing
   ../user_guide/tutorials/case_segment_analysis
   ../user_guide/tutorials/case_physics_validation
   ../user_guide/tutorials/case_arima_burst_search
   ../user_guide/tutorials/case_signal_extraction
   ../user_guide/tutorials/case_noise_budget
   ../user_guide/tutorials/case_lockin_detection
   ../user_guide/tutorials/case_wiener_filter
   ../user_guide/tutorials/case_coupling_analysis
   ../user_guide/tutorials/case_bruco_ica_denoising
   ../user_guide/tutorials/case_bruco_advanced
   ../user_guide/tutorials/case_violin_mode
   ../user_guide/tutorials/case_schumann_resonance
   ../user_guide/tutorials/case_glitch_analysis
