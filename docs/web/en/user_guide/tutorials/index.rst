Tutorials
=========

Learn how to use **GWexpy** through interactive examples.

.. tip::
   If you're new to GWexpy, we recommend starting with :doc:`../getting_started`.

These tutorials are generated from Jupyter Notebook source files. To run them locally, download the corresponding ``.ipynb`` from ``docs/web/en/user_guide/tutorials/`` in the repository.

.. note::
   Naming convention: tutorial titles follow the form "Feature: Task".
   English pages keep titles English-only, except for API identifiers and library names.

I. Core Data Structures
-----------------------
Fundamental data containers and basic operations.

II. Multi-channel & Matrix Containers
-------------------------------------
Handling multiple channels efficiently using Matrix classes.

III. High-dimensional Fields
----------------------------
Next-generation API for scalar, vector, and tensor fields in 4D spacetime.

IV. Advanced Signal Processing
------------------------------
Statistical analysis and advanced transforms.

V. Specialized Tools
--------------------
Tools for specific noise hunting and diagnostics tasks.

VI. Segment Analysis
--------------------
Table-based analysis for time segments.

.. toctree::
   :maxdepth: 1

   TimeSeries: Basics <intro_timeseries>
   FrequencySeries: Basics <intro_frequencyseries>
   Spectrogram: Basics <intro_spectrogram>
   Noise Generation: Basics <intro_noise>
   Plotting: Basics <intro_plotting>
   Map Plotting: Basics <intro_mapplotting>
   Interoperability: Basics <intro_interop>
   Histogram: Basics <intro_histogram>
   TimeSeriesMatrix: Matrix Basics <matrix_timeseries>
   FrequencySeriesMatrix: Matrix Basics <matrix_frequencyseries>
   SpectrogramMatrix: Matrix Basics <matrix_spectrogram>
   Field API: ScalarField Basics <field_scalar_intro>
   Field API: VectorField Basics <field_vector_intro>
   Field API: TensorField Basics <field_tensor_intro>
   Field API: ScalarField Signal Processing <field_scalar_signal>
   Field API: Advanced Analysis Workflow <field_advanced_workflow>
   Fitting: Basics <intro_fitting>
   Fitting: Spectral Line Analysis <advanced_fitting>
   Spectrogram: Normalization and Cleaning <advanced_spectrogram_processing>
   Case Study: Bootstrap PSD and GLS Fitting <case_bootstrap_gls_fitting>
   Peak Detection: Basics <advanced_peak_detection>
   Peak Tracking: Time Evolution <advanced_peak_tracking>
   HHT: Analysis <advanced_hht>
   Time-Frequency Analysis: Interactive Comparison <time_frequency_analysis_comparison>
   Time-Frequency Analysis: Method Selection Guide <time_frequency_comparison>
   ARIMA: Time Series Forecasting <advanced_arima>
   Correlation Analysis: Statistical Methods <advanced_correlation>
   ML Preprocessing: Individual Methods <ml_preprocessing_methods>
   Case Study: ML Preprocessing Pipeline <case_ml_preprocessing>
   Linear Algebra: Gravitational Wave Analysis <advanced_linear_algebra>
   Field API: Advanced Analysis Integration <field_advanced_integration>
   Non-Gaussian Noise Analysis: Rayleigh and Gaussian-Chi <rayleigh_gauch_tutorial>
   Coupling Analysis: Multi-Channel Coupling <advanced_coupling>
   Decomposition Analysis: PCA, ICA, and Eigenmodes <advanced_decomposition>
   Case Study: Seismic Analysis with ObsPy <case_seismic_obspy>
   Case Study: GBD Format I/O <case_gbd_format>
   BruCo: Basics <advanced_bruco>
   Case Study: BruCo and ICA Noise Reduction <case_bruco_ica_denoising>
   BruCo: Bilinear Coupling and AM/FM Demodulation <case_bruco_advanced>
   Case Study: Violin Mode Analysis <case_violin_mode>
   Case Study: Schumann Resonance Analysis <case_schumann_resonance>
   SegmentTable: Basics <intro_segment_table>
   Segment Analysis: Basic Pipeline <intro_table>
   ASD Analysis: Pipeline <segment_asd_pipeline>
   Segment Analysis: Visualization <segment_visualization>
   Case Study: Event-Synchronized Analysis <case_segment_analysis>

.. note::
   Practical case studies and applications are consolidated in :doc:`../../examples/index`.
   See the examples section for real-world applications like noise budget analysis,
   transfer function calculations, and active damping.
