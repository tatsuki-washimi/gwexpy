.. _glossary:

Glossary
========

Definitions of key terms and concepts used throughout the GWexpy documentation.

.. glossary::

   ScalarField
      The primary 4D field container (time, x, y, z) in GWexpy. It maintains domain information and axis metadata.

   TimeSeries
      A single-channel time-domain object (compatible with GWpy). The fundamental data type representing time-ordered values.

   TimeSeriesMatrix
      A matrix-style container for processing multiple time-series data streams simultaneously.

   FrequencySeriesMatrix
      A matrix-style container for processing multi-channel frequency-domain data.

   FieldList
   FieldDict
      Collection types for storing and performing batch operations on multiple fields (e.g., ``ScalarField``).

   SeriesMatrix
      A generic term for containers that hold multi-channel signals (either TimeSeries or FrequencySeries).

   ASD
      Amplitude Spectral Density. A representation of a signal's amplitude components as a density across frequencies.

   PSD
      Power Spectral Density. A representation of a signal's power (amplitude squared) components as a density across frequencies.

   CSD
      Cross Spectral Density. A metric used to evaluate the correlation between two signals in the frequency domain.

   Whitening
      The process of making a signal's power spectrum flat (white), typically used to normalize noise characteristics.

   Adaptive Whitening
      An advanced whitening technique that dynamically regularizes or homogenizes signal noise by tracking local statistical changes.

   NaN/Inf propagation
      (Legacy: Death Floats) The phenomenon where Not-a-Number (NaN) or Infinity (Inf) values produced during a calculation spread to subsequent results, invalidating the entire analysis.

   VIF
      Variance Inflation Factor. A diagnostic for multicollinearity. It measures how much the variance of an estimated regression coefficient is increased due to correlation between predictors.

   BruCo
      An analysis method or module implementation for estimating and removing noise in a primary channel using auxiliary channels, often via linear combinations or ICA.

   GPS時刻ユーティリティ関数
      A suite of functions (e.g., ``tconvert``, ``to_gps``, ``from_gps``) used for interconverting between GPS seconds, UTC, datetime, and ISO strings.

   Safe Log
      A technique used in logarithmic calculations to prevent divergence to negative infinity when the input is close to zero, typically by applying a lower bound (floor).

   SegmentTable
      A table structure designed to efficiently manage metadata and status flags for time intervals (segments) target for analysis.

   Time-Plane Transform
      The process of mapping time-series data into a time-frequency plane (e.g., Spectrogram) using methods like Q-transform or CWT.

   Pickle
      A standard Python protocol for serializing and de-serializing objects. Loading pickles from untrusted sources entails security risks.

   GWOSC
      Gravitational Wave Open Science Center. A platform providing public access to LIGO, Virgo, and KAGRA observation data and catalogs.

   Stability Labels
      Indicators of API maturity (Stable / Experimental / Deprecated).
