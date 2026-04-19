:orphan:

.. meta::
   :description: GWexpy documentation hub for installation, quickstart, tutorials, case studies, and API reference entry points.

.. _hub-en-hero:

GWexpy Documentation
==============================

GWexpy extends GWpy with new containers and numerical utilities for time-series and frequency-series data analysis.

v\ |release| · Python ≥ 3.9 · Last updated: |today|

.. raw:: html

   <style>
   .gw-hub-hero {
     display: grid;
     grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
     gap: 1.25rem;
     margin: 1.5rem 0 2rem;
     align-items: stretch;
   }
   .gw-hub-panel {
     background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
     border: 1px solid #d7e3ef;
     border-radius: 20px;
     box-shadow: 0 16px 40px rgba(15, 42, 74, 0.08);
   }
   .gw-hub-copy {
     padding: 1.5rem;
   }
   .gw-hub-eyebrow {
     margin: 0 0 0.65rem;
     color: #0f6cbd;
     font-size: 0.78rem;
     font-weight: 700;
     letter-spacing: 0.08em;
     text-transform: uppercase;
   }
   .gw-hub-title {
     margin: 0;
     font-size: 2.35rem;
     line-height: 1.02;
     letter-spacing: -0.03em;
     color: #10243a;
   }
   .gw-hub-lede {
     margin: 0.9rem 0 0;
     color: #566779;
     line-height: 1.8;
   }
   .gw-hub-meta {
     display: grid;
     grid-template-columns: repeat(3, minmax(0, 1fr));
     gap: 0.75rem;
     margin-top: 1.1rem;
   }
   .gw-hub-meta div {
     padding: 0.85rem 0.9rem;
     background: rgba(255,255,255,0.82);
     border: 1px solid #d7e3ef;
     border-radius: 14px;
   }
   .gw-hub-meta dt {
     margin: 0;
     color: #61758a;
     font-size: 0.72rem;
     font-weight: 700;
     letter-spacing: 0.08em;
     text-transform: uppercase;
   }
   .gw-hub-meta dd {
     margin: 0.35rem 0 0;
     color: #10243a;
     font-size: 0.92rem;
     font-weight: 700;
   }
   .gw-hub-figure {
     padding: 1rem;
   }
   .gw-hub-figure img {
     width: 100%;
     height: auto;
     display: block;
     border-radius: 14px;
     border: 1px solid #d7e3ef;
     background: #fff;
   }
   .gw-hub-figure p {
     margin: 0.75rem 0 0;
     color: #5c6d80;
     font-size: 0.88rem;
     line-height: 1.65;
   }
   .gw-hub-duo {
     display: grid;
     grid-template-columns: repeat(2, minmax(0, 1fr));
     gap: 1rem;
     margin: 0 0 2rem;
   }
   .gw-hub-duo section {
     padding: 1.15rem 1.2rem;
   }
   .gw-hub-duo p {
     margin: 0 0 0.65rem;
     color: #0f6cbd;
     font-size: 0.76rem;
     font-weight: 700;
     letter-spacing: 0.08em;
     text-transform: uppercase;
   }
   .gw-hub-duo pre {
     margin: 0;
     padding: 0.95rem 1rem;
     border: 1px solid #d7e3ef;
     border-radius: 14px;
     background: #f8fbfe;
     overflow-x: auto;
   }
   @media (max-width: 900px) {
     .gw-hub-hero,
     .gw-hub-duo {
       grid-template-columns: 1fr;
     }
     .gw-hub-meta {
       grid-template-columns: 1fr;
     }
   }
   </style>

.. raw:: html

   <section class="gw-hub-hero">
     <div class="gw-hub-panel gw-hub-copy">
       <p class="gw-hub-eyebrow">Documentation Hub</p>
       <h2 class="gw-hub-title">Choose a workflow, not just a page</h2>
       <p class="gw-hub-lede">
         This top page groups matrix containers, field operations, fitting, and signal-processing
         paths into task-oriented entry points. Start with Quick Start, or jump directly to the
         card that matches the analysis you want to run.
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
           <dd>Git checkout first</dd>
         </div>
       </dl>
     </div>
     <figure class="gw-hub-panel gw-hub-figure">
       <img src="/_static/images/phase3/gateway_hero_scientific.png" alt="FrequencySeriesMatrix overview with a resonance fit">
       <p>
         A matrix-level frequency response overview paired with a resonance fit for one extracted
         channel, showing the container model and analysis workflow in one figure.
       </p>
     </figure>
   </section>

.. raw:: html

   <section class="gw-hub-duo">
     <section class="gw-hub-panel">
       <p>Quick install</p>
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

   **Page Role**: documentation landing page
   **Audience**: first-time users, GWpy users, and analysts looking for practical workflows
   **Search Hints**: ``quickstart``, ``installation``, ``tutorials``, ``examples``, ``ScalarField``, ``TimeSeriesMatrix``

.. button-ref:: user_guide/quickstart
    :ref-type: doc
    :color: primary
    :shadow:
    :expand:

    🚀 Quick Start (Get started in 5 minutes)

.. _hub-en-workflow-entry:

----

Find the right guide for you
-----------------------------

.. note::

   `Tutorials` are feature-oriented learning paths that teach one class or capability at a time.
   `Case Studies` are goal-oriented workflows that combine multiple features around one analysis task.

.. grid:: 3
    :gutter: 3
    :class-container: grid-container

    .. grid-item-card:: 🎓 New to GWexpy?
        :link: user_guide/getting_started
        :link-type: doc

        Learn the basics

        * Installation Guide
        * Feature-oriented Tutorials

    .. grid-item-card:: 🔬 For Analysts
        :link: examples/index
        :link-type: doc

        Practical examples

        * Goal-oriented Case Studies
        * Advanced Signal Processing

    .. grid-item-card:: 🔄 For GWpy Users
        :link: user_guide/gwexpy_for_gwpy_users_en
        :link-type: doc

        Migration and compatibility

        * Difference recipes and compatibility
        * Path to the added-API index

----

Find by what you want to do
----------------------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 📈 Time-Series Analysis
        :link: user_guide/tutorials/intro_timeseries
        :link-type: doc

        Read, plot, and filter time-series data

    .. grid-item-card:: 📊 Spectrogram Analysis
        :link: user_guide/tutorials/intro_spectrogram
        :link-type: doc

        STFT, Q-scan, and time-frequency representation

    .. grid-item-card:: 🌊 Field Data Operations
        :link: user_guide/tutorials/field_scalar_intro
        :link-type: doc

        ScalarField / VectorField / TensorField basics

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 🔧 Noise Budget (BrUCo)
        :link: user_guide/tutorials/advanced_bruco
        :link-type: doc

        Decompose and visualise noise contributions

    .. grid-item-card:: 📐 Fitting & MCMC
        :link: user_guide/tutorials/intro_fitting
        :link-type: doc

        GLS, Bayesian fitting, MCMC sampling

    .. grid-item-card:: 🧮 Pre-processing & ML Pipeline
        :link: user_guide/tutorials/ml_preprocessing_methods
        :link-type: doc

        Feature extraction, normalisation, scikit-learn

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 📁 Multi-Format I/O
        :link: user_guide/tutorials/case_gbd_format
        :link-type: doc
        :class-card: gw-highlight

        HDF5 / GBD / Frame / Zarr and more

        **GWexpy differentiator**

    .. grid-item-card:: 🔗 Library Interoperability
        :link: user_guide/tutorials/case_seismic_obspy
        :link-type: doc
        :class-card: gw-highlight

        ObsPy / LAL / Finesse / PyCBC integration

        **GWexpy differentiator**

    .. grid-item-card:: 🔢 Matrix Operations
        :link: user_guide/tutorials/matrix_timeseries
        :link-type: doc

        TimeSeriesMatrix / FrequencySeriesMatrix batch processing

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
   user_guide/citation
   user_guide/changelog
   user_guide/license

.. toctree::
   :hidden:

   user_guide/cli
   user_guide/gui
