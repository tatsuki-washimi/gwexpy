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
      A matrix-style container for processing multi-channel time-series data streams simultaneously.

   FrequencySeriesMatrix
      A matrix-style container for processing multi-channel frequency-domain data.

   FieldList
   FieldDict
      Collection types for storing and performing batch operations on multiple fields (e.g., ``ScalarField``).

   SeriesMatrix
      A generic term for containers that hold multi-channel signals (either TimeSeries or FrequencySeries). They may have associated aliases.

   ASD
      Amplitude Spectral Density. A representation of a signal's amplitude components as a density across frequencies.

   PSD
      Power Spectral Density. A representation of a signal's power (amplitude squared) components as a density across frequencies.

   CSD
      Cross Spectral Density. A metric used to evaluate the correlation between two signals in the frequency domain.

   FFT / STFT / CWT / HHT
      Various time-frequency transform methods, referring to Fast Fourier Transform, Short-Time Fourier Transform, Continuous Wavelet Transform, and Hilbert-Huang Transform, respectively.

   Whitening
      The process of making a signal's power spectrum flat (white), typically used to normalize noise characteristics.

   Adaptive Whitening
   AD-Whitening
      An advanced whitening technique that dynamically regularizes or homogenizes signal noise by tracking local statistical changes.

   NaN/Inf propagation
      (Legacy: Death Floats) The phenomenon where Not-a-Number (NaN) or Infinity (Inf) values produced during a calculation spread to subsequent results, invalidating the entire analysis.

   VIF
      Variance Inflation Factor. A diagnostic for multicollinearity. It measures how much the variance of an estimated regression coefficient is increased due to correlation between predictors.

   ``BruCo``
      BruCo is a coherence/correlation-based noise analysis framework. The documentation follows the capitalization used in the implementation (``BruCo`` or ``Bruco``).

   Field API
      The public interface providing ``ScalarField`` and related operations.

   TimePlaneTransform
      A utility for mapping time-series data into a time-frequency plane (Spectrogram) using methods like Q-transform or CWT.

   Safe Log
      Lower-bound handling during logarithmic transformation. The default floor is 200 dB, which can be overridden by parameters.

   SegmentTable
      A table structure designed to manage time intervals (segments) used in analysis. Each row typically contains `t0`, `t1`, `label`, and `quality_flag`.

   CITATION.cff
      A metadata file containing citation information for papers or software in a standardized format.

   GWOSC
      Gravitational Wave Open Science Center. A platform providing public access to LIGO, Virgo, and KAGRA observation data and catalogs.

   miniSEED / GWF / GBD / MTH5 / TDMS / Zarr
      Various input/output file formats supported by GWexpy.

   Pickle
      (Pickle — Security Warning) A standard Python protocol for serializing objects. Loading pickles can lead to arbitrary code execution, so alternatives like HDF5 or Zarr are recommended for untrusted data.

   tconvert / to_gps / from_gps
      A suite of GPS time utility functions used for interconverting between GPS seconds, UTC, datetime, and ISO strings.

   Leap second
      A one-second adjustment applied to Coordinated Universal Time (UTC) to keep it in sync with the Earth's rotation. Critical for accurate time conversion.

   GPS time
      Seconds since the GPS epoch (January 6, 1980). A monotonic time scale that does not include leap seconds.

   UTC
      Coordinated Universal Time. The primary time standard by which the world regulates clocks and time, adjusted by leap seconds.

   MCMC
      Markov chain Monte Carlo. A class of algorithms for sampling from a probability distribution, often used in Bayesian inference.

   ICA / PCA
      Independent Component Analysis / Principal Component Analysis. Statistical methods used for signal separation or dimensionality reduction.

   Robust ICA
      An implementation of Independent Component Analysis designed to be robust against outliers and noise.

   ASPIRE / ICRR / LALSuite / PyCBC / Bilby
      Names of external tools, libraries, or organizations used in gravitational-wave analysis and related research.

   Stability Labels
      Indicators of API maturity (**Stable**, **Experimental**, or **Deprecated**).

