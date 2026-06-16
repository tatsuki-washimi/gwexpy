:orphan:

.. meta::
   :description: GWexpy documentation hub for installation, quickstart, tutorials, case studies, and API reference entry points.

GWexpy Documentation
==============================

GWexpy extends GWpy with new containers and numerical utilities for time-series and frequency-series data analysis.

v\ |release| · Python ≥ 3.11 · Last updated: |today|

.. raw:: html

   <section class="gw-hub-duo">
     <section class="gw-hub-panel">
       <p>Quick installation</p>
       <pre><code>git clone https://github.com/tatsuki-washimi/gwexpy.git
    cd gwexpy
    python -m pip install -e .</code></pre>
     </section>
     <section class="gw-hub-panel">
       <p>3-line demo</p>
       <pre><code>from gwexpy.frequencyseries import FrequencySeriesMatrix
    fsmtx = FrequencySeriesMatrix.read("data.hdf5")
    fsmtx[2, 0].fit(model="lorentzian").plot()</code></pre>
     </section>
   </section>

.. button-ref:: user_guide/quickstart
    :ref-type: doc
    :color: primary
    :shadow:
    :expand:

    🚀 Quick Start (Learn the basics in 5 minutes)

.. note::

   Found a bug or want to request a feature? Use the
   `lightweight feedback form <https://forms.gle/c8jJaf9UCs5tb5cC8>`_.
   For security reports, do not use this form; follow the repository security policy.

----

Select your guide
-----------------

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

.. button-ref:: examples/index
    :ref-type: doc
    :color: secondary
    :expand:

    Browse the full goal-oriented case-study gallery

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
   user_guide/prerequisites_and_conventions

.. toctree::
   :maxdepth: 2
   :caption: 🎓 Learn

   user_guide/tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: 📚 Reference

   reference/index
   user_guide/io_formats
   user_guide/interop
   user_guide/time_utilities
   user_guide/numerical_stability
   user_guide/scalarfield_slicing
   user_guide/gwexpy_for_gwpy_users_en
   user_guide/gwpy_added_api_index_en
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
