Tutorials
=========

Learn how to use **GWexpy** through interactive examples.

.. tip::
   If you're new to GWexpy, we recommend starting with :doc:`../getting_started`.

These tutorials are generated from Jupyter Notebook source files. To run them locally, download the corresponding ``.ipynb`` from ``docs/web/en/user_guide/tutorials/`` in the repository.

.. note::
   Naming convention: tutorial titles follow the form "Feature: Task".
   English pages keep titles English-only, except for API identifiers and library names.

.. note::
   Each tutorial shows its **difficulty**, **estimated time**, and **primary audience**.
   Audience labels are normalized to `Beginners`, `GWpy Users`, and `Analysis Practitioners`.

.. note::
   `Tutorials` are class- and feature-oriented examples.
   Theme-driven demonstrations that combine multiple features are collected in :doc:`../../examples/index` under `Case Studies`.

I. Core Data Structures
-----------------------
Fundamental data containers and basic operations.

- :doc:`TimeSeries: Basics <intro_timeseries>` :bdg-primary:`Beginner` :bdg-secondary:`15 min` :bdg-info:`Beginners`
- :doc:`FrequencySeries: Basics <intro_frequencyseries>` :bdg-primary:`Beginner` :bdg-secondary:`15 min` :bdg-info:`Beginners`
- :doc:`Spectrogram: Basics <intro_spectrogram>` :bdg-primary:`Beginner` :bdg-secondary:`15 min` :bdg-info:`Beginners`
- :doc:`Noise Generation: Basics <intro_noise>` :bdg-primary:`Beginner` :bdg-secondary:`15 min` :bdg-info:`Beginners`
- :doc:`Plotting: Basics <intro_plotting>` :bdg-primary:`Beginner` :bdg-secondary:`15 min` :bdg-info:`Beginners`
- :doc:`Map Plotting: Basics <intro_mapplotting>` :bdg-primary:`Beginner` :bdg-secondary:`20 min` :bdg-info:`GWpy Users`
- :doc:`Histogram: Basics <intro_histogram>` :bdg-primary:`Beginner` :bdg-secondary:`15 min` :bdg-info:`Beginners`

II. Multi-channel & Matrix Containers
-------------------------------------
Handling multiple channels efficiently using Matrix classes.

- :doc:`TimeSeriesMatrix: Matrix Basics <matrix_timeseries>` :bdg-primary:`Beginner` :bdg-secondary:`20 min` :bdg-info:`GWpy Users`
- :doc:`FrequencySeriesMatrix: Matrix Basics <matrix_frequencyseries>` :bdg-primary:`Beginner` :bdg-secondary:`20 min` :bdg-info:`GWpy Users`
- :doc:`SpectrogramMatrix: Matrix Basics <matrix_spectrogram>` :bdg-primary:`Beginner` :bdg-secondary:`20 min` :bdg-info:`GWpy Users`

III. High-dimensional Fields
----------------------------
Next-generation API for scalar, vector, and tensor fields in 4D spacetime.

- :doc:`Field API: ScalarField Basics <field_scalar_intro>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Field API: VectorField Basics <field_vector_intro>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Field API: TensorField Basics <field_tensor_intro>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Field API: ScalarField Signal Processing <field_scalar_signal>` :bdg-primary:`Intermediate` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`Field API: Advanced Analysis Workflow <field_advanced_workflow>` :bdg-primary:`Intermediate` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`Field API: Advanced Analysis Integration <field_advanced_integration>` :bdg-primary:`Intermediate` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`

IV. Advanced Signal Processing
------------------------------
Statistical analysis and advanced transforms.

- :doc:`Fitting: Basics <intro_fitting>` :bdg-primary:`Beginner` :bdg-secondary:`20 min` :bdg-info:`GWpy Users`
- :doc:`Fitting: Spectral Line Analysis <advanced_fitting>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Spectrogram: Normalization and Cleaning <advanced_spectrogram_processing>` :bdg-primary:`Intermediate` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`Peak Detection: Basics <advanced_peak_detection>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Peak Tracking: Time Evolution <advanced_peak_tracking>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`HHT: Analysis <advanced_hht>` :bdg-primary:`Advanced` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`Time-Frequency Analysis: Interactive Comparison <time_frequency_analysis_comparison>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Time-Frequency Analysis: Method Selection Guide <time_frequency_comparison>` :bdg-primary:`Intermediate` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`ARIMA: Time Series Forecasting <advanced_arima>` :bdg-primary:`Advanced` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`Correlation Analysis: Statistical Methods <advanced_correlation>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`ML Preprocessing: Individual Methods <ml_preprocessing_methods>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Linear Algebra: Gravitational Wave Analysis <advanced_linear_algebra>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Non-Gaussian Noise Analysis: Rayleigh and Gaussian-Chi <rayleigh_gauch_tutorial>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Coupling Analysis: Multi-Channel Coupling <advanced_coupling>` :bdg-primary:`Advanced` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`Decomposition Analysis: PCA, ICA, and Eigenmodes <advanced_decomposition>` :bdg-primary:`Advanced` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`

V. Data I/O & Interoperability
------------------------------
Tutorials for file ingest, read/write workflows, and conversions with external libraries.

- :doc:`Interoperability: Basics <intro_interop>` :bdg-primary:`Beginner` :bdg-secondary:`20 min` :bdg-info:`GWpy Users`
- :doc:`Case Study: GBD Format I/O <case_gbd_format>` :bdg-primary:`Intermediate` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`
- :doc:`Case Study: Seismic Analysis with ObsPy <case_seismic_obspy>` :bdg-primary:`Intermediate` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`

VI. Noise Hunting & Specialized Tools
-------------------------------------
Tools for specific noise hunting and diagnostics tasks.

- :doc:`Bruco: Basics <advanced_bruco>` :bdg-primary:`Advanced` :bdg-secondary:`45 min` :bdg-info:`Analysis Practitioners`

VII. Segment Analysis
---------------------
Table-based analysis for time segments.

- :doc:`SegmentTable: Basics <intro_segment_table>` :bdg-primary:`Beginner` :bdg-secondary:`15 min` :bdg-info:`Beginners`
- :doc:`Segment Analysis: Basic Pipeline <intro_table>` :bdg-primary:`Beginner` :bdg-secondary:`20 min` :bdg-info:`GWpy Users`
- :doc:`ASD Analysis: Pipeline <segment_asd_pipeline>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`
- :doc:`Segment Analysis: Visualization <segment_visualization>` :bdg-primary:`Intermediate` :bdg-secondary:`30 min` :bdg-info:`Analysis Practitioners`

.. note::
   Theme-driven demonstrations live in :doc:`../../examples/index`.
   See the case studies section for workflows such as noise budgeting,
   transfer function measurement, ML preprocessing, and event-synchronized analysis.

.. toctree::
   :hidden:
   :maxdepth: 1

   intro_timeseries
   intro_frequencyseries
   intro_spectrogram
   intro_noise
   intro_plotting
   intro_mapplotting
   intro_histogram
   matrix_timeseries
   matrix_frequencyseries
   matrix_spectrogram
   field_scalar_intro
   field_vector_intro
   field_tensor_intro
   field_scalar_signal
   field_advanced_workflow
   intro_fitting
   advanced_fitting
   advanced_spectrogram_processing
   advanced_peak_detection
   advanced_peak_tracking
   advanced_hht
   time_frequency_analysis_comparison
   time_frequency_comparison
   advanced_arima
   advanced_correlation
   ml_preprocessing_methods
   advanced_linear_algebra
   field_advanced_integration
   rayleigh_gauch_tutorial
   advanced_coupling
   advanced_decomposition
   intro_interop
   advanced_bruco
   intro_segment_table
   intro_table
   segment_asd_pipeline
   segment_visualization
