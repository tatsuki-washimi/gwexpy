:orphan:

.. meta::
   :description: GWexpy documentation hub for installation, quickstart, tutorials, case studies, and API reference entry points.

.. _hub-en-hero:

GWexpy Documentation
==============================

GWexpy extends GWpy with new containers and numerical utilities for time-series and frequency-series data analysis.

v\ |release| · Python ≥ 3.9 · Last updated: |today|

.. raw:: html

   <div class="gw-hub-lang-switch">
     <div id="lang-pill">
       <span>🌐</span>
       <a href="../ja/index.html">日本語</a>
       <span class="lang-sep">|</span>
       <span class="lang-current">English</span>
     </div>
   </div>

..   <section class="gw-hub-hero">
     <div class="gw-hub-panel gw-hub-copy">
       <p class="gw-hub-eyebrow">Documentation Hub</p>
       <h2 class="gw-hub-title">Explore features by analysis workflow</h2>
       <p class="gw-hub-lede">
         A goal-oriented entry point for matrices, fields, fitting, and advanced signal processing.
         Start with the Quick Start, or browse the categories below to find tools that match your analysis task.
       </p>
       <dl class="gw-hub-meta">
         <div>
           <dt>Containers</dt>
           <dd>Matrix / Field / Series</dd>
         </div>
         <div>
           <dt>Analysis</dt>
           <dd>Fitting, BrUCo, MCMC</dd>
         </div>
         <div>
           <dt>Install</dt>
           <dd>Git checkout recommended</dd>
         </div>
       </dl>
     </div>
     <figure class="gw-hub-panel gw-hub-figure">
       <img src="../../_static/images/phase3/gateway_hero_scientific.png" alt="FrequencySeriesMatrix overview with a resonance fit">
       <p>
         FrequencySeriesMatrix visualization featuring a Lorentzian fit on a single channel, 
         demonstrating the integrated container and analysis workflow.
       </p>
     </figure>
   </section>

.. raw:: html

   <section class="gw-hub-duo">
     <section class="gw-hub-panel">
       <p>Quick installation</p>
       <pre><code>git clone https://github.com/tatsuki-washimi/gwexpy.git
    cd gwexpy
    pip install -e .</code></pre>
     </section>
     <section class="gw-hub-panel">
       <p>3-line demo</p>
       <pre><code>from gwexpy.timeseries import FrequencySeriesMatrix
    fsmtx = FrequencySeriesMatrix.read("data.hdf5")
    fsmtx[2, 0].fit(model="lorentzian").plot()</code></pre>
     </section>
   </section>

.. note::

   **Page Role**: Documentation Landing Page
   **Audience**: First-time users, GWpy analysts, and developers looking for integrated workflows.
   **Search Hints**: ``quickstart``, ``installation``, ``tutorials``, ``examples``, ``ScalarField``, ``TimeSeriesMatrix``

.. button-ref:: user_guide/quickstart
    :ref-type: doc
    :color: primary
    :shadow:
    :expand:

    🚀 Quick Start (Learn the basics in 5 minutes)

.. _hub-en-workflow-entry:

----

Select your guide
-----------------

.. note::

   **Tutorials** are feature-oriented paths for learning individual classes or capabilities.
   **Case Studies** are workflow-oriented examples combining multiple features for practical analysis.

.. grid:: 3
    :gutter: 3
    :class-container: grid-container

    .. grid-item-card:: 🎓 New to GWexpy?
        :link: user_guide/getting_started
        :link-type: doc

        Foundation
        
        * Installation Guide
        * Feature-oriented Tutorials

    .. grid-item-card:: 🔬 For Analysts
        :link: examples/index
        :link-type: doc

        Applications

        * Goal-oriented Case Studies
        * Advanced Signal Processing

    .. grid-item-card:: 🔄 For GWpy Users
        :link: user_guide/gwexpy_for_gwpy_users_en
        :link-type: doc

        Interoperability

        * Migration recipes
        * Added API index

----

Browse by category
------------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 📈 Time-Series Analysis
        :link: user_guide/tutorials/intro_timeseries
        :link-type: doc

        Load, plot, and filter time-series data.

    .. grid-item-card:: 📊 Spectrogram Analysis
        :link: user_guide/tutorials/intro_spectrogram
        :link-type: doc

        STFT, Q-scans, and time-frequency maps.

    .. grid-item-card:: 🌊 Field Data Operations
        :link: user_guide/tutorials/field_scalar_intro
        :link-type: doc

        ScalarField / VectorField / TensorField logic.

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 🔧 Noise Budget (BrUCo)
        :link: user_guide/tutorials/advanced_bruco
        :link-type: doc

        Decompose and visualize noise contributions.

    .. grid-item-card:: 📐 Fitting & MCMC
        :link: user_guide/tutorials/intro_fitting
        :link-type: doc

        GLS, Bayesian fitting, and MCMC sampling.

    .. grid-item-card:: 🧮 ML Pre-processing
        :link: user_guide/tutorials/ml_preprocessing_methods
        :link-type: doc

        Feature extraction and scikit-learn integration.

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 📁 Multi-Format I/O
        :link: user_guide/tutorials/case_gbd_format
        :link-type: doc
        :class-card: gw-highlight

        HDF5 / GBD / Frame / Zarr support.

        **Core Strengths**

    .. grid-item-card:: 🔗 Interoperability
        :link: user_guide/tutorials/case_seismic_obspy
        :link-type: doc
        :class-card: gw-highlight

        ObsPy / LAL / Finesse / PyCBC tools.

        **Core Strengths**

    .. grid-item-card:: 🔢 Matrix Operations
        :link: user_guide/tutorials/matrix_timeseries
        :link-type: doc

        Vectorized processing for TimeSeriesMatrix.
        TimeSeriesMatrix / FrequencySeriesMatrix batch processing.

----

.. _hub-en-gallery-entry:

Visual Examples
---------------

These cards are a short preview of the canonical goal-oriented case-study gallery.

.. grid:: 3
    :gutter: 3

    .. grid-item-card::
        :img-top: /_static/images/case_noise_budget_thumb.png
        :img-alt: Thumbnail of the Bruco noise budget example
        :link: user_guide/tutorials/case_noise_budget
        :link-type: doc
        :text-align: center

        Noise Budget

    .. grid-item-card::
        :img-top: /_static/images/case_transfer_function_thumb.png
        :img-alt: Thumbnail of the transfer function estimation example
        :link: user_guide/tutorials/case_transfer_function
        :link-type: doc
        :text-align: center

        Transfer Function Estimation

    .. grid-item-card::
        :img-top: /_static/images/case_active_damping_thumb.png
        :img-alt: Thumbnail of the active damping example
        :link: user_guide/tutorials/case_active_damping
        :link-type: doc
        :text-align: center

        Active Damping

.. button-ref:: examples/index
    :ref-type: doc
    :color: secondary
    :expand:

    Browse the full goal-oriented case-study gallery

.. _hub-en-reference-entry:

----

Learn the basics of GWpy
-------------------------

GWexpy is built on top of GWpy. For GWpy fundamentals, see the official docs:

`gwpy.readthedocs.io/en/stable/ <https://gwpy.readthedocs.io/en/stable/>`_

----

Suggested starting points
-------------------------

- Run code immediately: :doc:`user_guide/quickstart`
- Choose a learning path: :doc:`user_guide/getting_started`
- Review shared assumptions first: :doc:`user_guide/prerequisites_and_conventions`

----

.. toctree::
   :maxdepth: 2
   :caption: 🚀 Start

   user_guide/installation
   user_guide/quickstart
   user_guide/getting_started

.. toctree::
   :maxdepth: 2
   :caption: 📖 Guide

   user_guide/prerequisites_and_conventions
   user_guide/interop
   user_guide/time_utilities
   user_guide/numerical_stability
   user_guide/scalarfield_slicing
   user_guide/gwexpy_for_gwpy_users_en
   user_guide/gwpy_added_api_index_en

.. toctree::
   :maxdepth: 2
   :caption: 🎓 Learn

   user_guide/tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: 🗂 Formats & I/O

   user_guide/io_formats

.. toctree::
   :maxdepth: 2
   :caption: 📚 Reference

   reference/index
   user_guide/validated_algorithms
   user_guide/architecture
   user_guide/physics_models
   user_guide/glossary

.. toctree::
   :maxdepth: 1
   :caption: ℹ️ Info

   user_guide/roadmap
   user_guide/troubleshooting
   user_guide/verification_and_quality
   user_guide/citation
   user_guide/changelog
   user_guide/license

.. toctree::
   :hidden:

   user_guide/cli
   user_guide/gui
