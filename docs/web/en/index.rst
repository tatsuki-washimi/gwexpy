GWexpy Documentation
====================

GWexpy extends GWpy with additional containers and analysis utilities for time and frequency series data analysis.

**Key Features:**

* TimeSeries/FrequencySeries matrix support
* High-dimensional Field API (ScalarField, VectorField, TensorField)
* Numerical stability enhancements
* Utilities for experimental data analysis

.. admonition:: Quick Start
   :class: tip

   Start with :doc:`Installation <user_guide/installation>` to set up your environment, then proceed to :doc:`Quick Start <user_guide/quickstart>` to learn the basics.


Choose Your Guide
-----------------

.. grid:: 3

    .. grid-item-card:: For Beginners
        :link: user_guide/getting_started
        :link-type: doc

        Learn from the basics

        * Installation Guide
        * Quick Start
        * Basic Tutorials

    .. grid-item-card:: For Experimental Data Analysts
        :link: examples/index
        :link-type: doc

        Learn from examples

        * Basic usage examples
        * Advanced signal processing
        * Real-world case studies

    .. grid-item-card:: For GWpy Users
        :link: user_guide/getting_started
        :link-type: doc

        Migration & Upgrades

        * Differences from GWpy
        * New Feature Highlights
        * Compatibility Information


Visual Examples
---------------

.. figure:: ../../_static/images/hero_plot.png
   :align: center
   :width: 90%
   :alt: GWexpy time series visualization example

   Time series visualization example with GWexpy


Example Gallery
---------------

Representative case studies:

.. grid:: 3

    .. grid-item-card:: Noise Budget Analysis
        :link: examples/index
        :link-type: doc
        :img-top: ../../_static/images/case_noise_budget_thumb.png

        Multi-channel correlation and noise source identification

    .. grid-item-card:: Transfer Function Measurement
        :link: examples/index
        :link-type: doc
        :img-top: ../../_static/images/case_transfer_function_thumb.png

        Bode plot and model fitting

    .. grid-item-card:: Active Damping Control
        :link: examples/index
        :link-type: doc
        :img-top: ../../_static/images/case_active_damping_thumb.png

        6-DOF MIMO control simulation


Learning Path by User Level
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - User Level
     - Start Here
     - Next Steps
   * - Beginners
     - :doc:`Installation <user_guide/installation>`, :doc:`Quick Start <user_guide/quickstart>`
     - :doc:`Getting Started <user_guide/getting_started>`, :doc:`Tutorials <user_guide/tutorials/index>`
   * - Researchers
     - :doc:`Examples <examples/index>`
     - :doc:`Advanced Signal Processing <user_guide/tutorials/index>`, :doc:`API Reference <reference/index>`
   * - GWpy Users
     - :doc:`Migration Guide <user_guide/getting_started>`
     - :doc:`New Feature Tutorials <user_guide/tutorials/index>`


Main Documentation
------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/getting_started
   user_guide/tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index


Next Steps
----------

To continue learning:

* :doc:`Getting Started <user_guide/getting_started>` - Detailed learning path
* :doc:`Tutorials <user_guide/tutorials/index>` - All tutorials
* :doc:`Examples <examples/index>` - Real-world case studies
* :doc:`API Reference <reference/index>` - Class and function details

**Advanced Topics:**

* :doc:`ScalarField Slicing <user_guide/scalarfield_slicing>` - Multi-dimensional data manipulation
* :doc:`Numerical Stability & Accuracy <user_guide/numerical_stability>` - Computational reliability
* :doc:`Validated Algorithms <user_guide/validated_algorithms>` - Algorithm verification reports


Language
--------

* :doc:`日本語 (Japanese) <../ja/index>`
