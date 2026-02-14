Tutorials
=========

Learn how to use **GWexpy** through interactive examples.

.. tip::
   If you're new to GWexpy, we recommend starting with :doc:`../getting_started`.

.. note::
   These tutorials are generated from Jupyter Notebooks. Use "Edit on GitHub" on each page to download the original `.ipynb` and run it locally.

I. Core Data Structures
-----------------------
Fundamental data containers and basic operations.

.. toctree::
   :maxdepth: 1
   :caption: Core Data Structures

   TimeSeries Basics <intro_timeseries>
   FrequencySeries Basics <intro_frequencyseries>
   Spectrogram Basics <intro_spectrogram>
   Plotting Basics <intro_plotting>
   Map Plotting Basics <intro_mapplotting>
   Interoperability Basics <intro_interop>

II. Multi-channel & Matrix Containers
-------------------------------------
Handling multiple channels efficiently using Matrix classes.

.. toctree::
   :maxdepth: 1
   :caption: Matrix Containers

   TimeSeriesMatrix <matrix_timeseries>
   FrequencySeriesMatrix <matrix_frequencyseries>
   SpectrogramMatrix <matrix_spectrogram>

III. High-dimensional Fields
----------------------------
Next-generation API for scalar, vector, and tensor fields in 4D spacetime.

.. toctree::
   :maxdepth: 1
   :caption: Fields API

   Scalar Field Basics <field_scalar_intro>
   Vector Field Basics <field_vector_intro>
   Tensor Field Basics <field_tensor_intro>
   Scalar Field Signals <field_scalar_signal>
   Field × Advanced Analysis Workflow <field_advanced_workflow>

IV. Advanced Signal Processing
------------------------------
Statistical analysis and advanced transforms.

.. toctree::
   :maxdepth: 1
   :caption: Advanced Analysis

   Advanced Fitting <advanced_fitting>
   Bootstrap PSD & GLS Fitting <case_bootstrap_gls_fitting>
   Peak Detection <advanced_peak_detection>
   HHT Analysis <advanced_hht>
   Time-Frequency Analysis: Interactive Examples <time_frequency_analysis_comparison>
   Time-Frequency Methods: Theory Guide <time_frequency_comparison>
   ARIMA Forecasting <advanced_arima>
   Nonlinear Correlation <advanced_correlation>
   ML Preprocessing Methods <ml_preprocessing_methods>
   Linear Algebra for GW Analysis <advanced_linear_algebra>
   Field × Advanced Analysis Integration <field_advanced_integration>

V. Specialized Tools
--------------------
Tools for specific noise hunting and diagnostics tasks.

.. toctree::
   :maxdepth: 1
   :caption: Specialized Tools

   Noise Hunting with Bruco <advanced_bruco>

.. note::
   Practical case studies and applications are consolidated in :doc:`../examples/index`.
   See the examples section for real-world applications like noise budget analysis,
   transfer function calculations, and active damping.
