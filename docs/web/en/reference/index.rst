Reference
=========

Detailed specifications of GWexpy classes, functions, and APIs.

Key Data Structures
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :doc:`ScalarField <../user_guide/scalarfield_slicing>`
     - High-dimensional data container for 4D (time, space) fields.
   * - :class:`TimeSeriesMatrix <gwexpy.timeseries.TimeSeriesMatrix>`
     - Matrix-format container for batch processing of multiple time series.
   * - :class:`FrequencySeriesMatrix <gwexpy.frequencyseries.FrequencySeriesMatrix>`
     - Container for multi-channel data in the frequency domain.

API Reference
-------------

For detailed methods and properties of each module, please refer to the links below.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🧩 API Index
        :link: api/index
        :link-type: doc

        List of all modules and functions

    .. grid-item-card:: 🏗️ Class List
        :link: classes
        :link-type: doc

        Properties and methods of major classes

.. note::
   For the design intent and usage examples of `ScalarField`, see :doc:`../user_guide/scalarfield_slicing`. For module-by-module API pages, start from :doc:`api/index`.

.. toctree::
   :hidden:

   api/index
   classes
