Tutorials
=========

Learn how to use **GWexpy** through interactive examples.

.. tip::
   If you're new to GWexpy, we recommend starting with :doc:`../getting_started`.

These tutorials are generated from Jupyter Notebook source files. To run them locally, download the corresponding `.ipynb` from `docs/web/en/user_guide/tutorials/` in the repository.

I. Core Data Structures
-----------------------
Fundamental data containers and basic operations.

.. toctree::
   :maxdepth: 1
   :caption: Core Data Structures

   TimeSeries Basics <intro_timeseries>
   FrequencySeries Basics <intro_frequencyseries>
   Spectrogram Basics <intro_spectrogram>
   Noise Generation Basics <intro_noise>
   Plotting Basics <intro_plotting>
   Map Plotting Basics <intro_mapplotting>
   Interoperability Basics <intro_interop>
   Histogram Basics <intro_histogram>

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

   Spectral Fitting Basics <intro_fitting>
   Advanced Fitting & Spectral Lines <advanced_fitting>
   Spectrogram Processing: Normalization & Cleaning <advanced_spectrogram_processing>
   Bootstrap PSD & GLS Fitting <case_bootstrap_gls_fitting>
   Peak Detection <advanced_peak_detection>
   Peak / Line Time Tracking <advanced_peak_tracking>
   HHT Analysis <advanced_hht>
   Time-Frequency Analysis: Interactive Examples <time_frequency_analysis_comparison>
   Time-Frequency Methods: Theory Guide <time_frequency_comparison>
   ARIMA Forecasting <advanced_arima>
   Nonlinear Correlation <advanced_correlation>
   ML Preprocessing Methods <ml_preprocessing_methods>
   ML Preprocessing Pipeline <case_ml_preprocessing>
   Linear Algebra for GW Analysis <advanced_linear_algebra>
   Field × Advanced Analysis Integration <field_advanced_integration>
   Non-Gaussian Noise Analysis: Rayleigh & Gaussian-Chi <rayleigh_gauch_tutorial>
   Multi-channel Coupling <advanced_coupling>
   Modal Decomposition <advanced_decomposition>
   Seismic Analysis (ObsPy) <case_seismic_obspy>
   GBD Format (GRAPHTEC) <case_gbd_format>

V. Specialized Tools
--------------------
Tools for specific noise hunting and diagnostics tasks.

.. toctree::
   :maxdepth: 1
   :caption: Specialized Tools

   BruCo Basics <advanced_BruCo>
   BruCo + ICA End-to-End Denoising <case_BruCo_ica_denoising>
   Advanced BruCo: Bilinear Coupling & AM/FM Demodulation <case_BruCo_advanced>
   Violin Mode Analysis <case_violin_mode>
   Schumann Resonance Analysis <case_schumann_resonance>

VI. Segment Analysis
--------------------
Table-based analysis for time segments.

.. toctree::
   :maxdepth: 1
   :caption: Segment Analysis

   Segment Basics <intro_segment_table>
   Segment Analysis Pipeline (Advanced) <intro_table>
   ASD Analysis Pipeline <segment_asd_pipeline>
   Visualization Deep Dive <segment_visualization>
   Event Matching Case Study <case_segment_analysis>

.. note::
   Practical case studies and applications are consolidated in :doc:`../../examples/index`.
   See the examples section for real-world applications like noise budget analysis,
   transfer function calculations, and active damping.
