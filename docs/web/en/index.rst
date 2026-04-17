:orphan:

GWexpy Documentation
==============================

GWexpy extends GWpy with new containers and numerical utilities for time-series and frequency-series data analysis.

v\ |release| · Python ≥ 3.9 · Last updated: |today|

.. button-ref:: user_guide/quickstart
    :ref-type: doc
    :color: primary
    :shadow:
    :expand:

    🚀 Quick Start (Get started in 5 minutes)

----

Find the right guide for you
-----------------------------

.. grid:: 3
    :gutter: 3
    :class-container: grid-container

    .. grid-item-card:: 🎓 New to GWexpy?
        :link: user_guide/getting_started
        :link-type: doc

        Learn the basics

        * Installation Guide
        * Basic Tutorials

    .. grid-item-card:: 🔬 For Analysts
        :link: examples/index
        :link-type: doc

        Practical examples

        * Case Studies
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

Visual Examples
---------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card::
        :img-top: /_static/images/case_noise_budget_thumb.png
        :link: user_guide/tutorials/advanced_bruco
        :link-type: doc
        :text-align: center

        Noise Budget

    .. grid-item-card::
        :img-top: /_static/images/case_transfer_function_thumb.png
        :link: user_guide/tutorials/case_transfer_function
        :link-type: doc
        :text-align: center

        Transfer Function Estimation

    .. grid-item-card::
        :img-top: /_static/images/case_active_damping_thumb.png
        :link: user_guide/tutorials/case_active_damping
        :link-type: doc
        :text-align: center

        Active Damping

----

Learn the basics of GWpy
-------------------------

GWexpy is built on top of GWpy. For GWpy fundamentals, see the official docs:

`gwpy.github.io/docs/stable/ <https://gwpy.github.io/docs/stable/>`_

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

   user_guide/io_formats
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
   :caption: 📚 Reference

   reference/index
   user_guide/validated_algorithms
   user_guide/architecture
   user_guide/physics_models
   user_guide/glossary

.. toctree::
   :maxdepth: 1
   :caption: ℹ️ Info

   user_guide/troubleshooting
   user_guide/citation
   user_guide/changelog
   user_guide/license

.. toctree::
   :hidden:

   user_guide/cli
   user_guide/gui
