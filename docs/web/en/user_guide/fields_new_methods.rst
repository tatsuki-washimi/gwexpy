New Signal Processing Methods
==============================

.. currentmodule:: gwexpy.fields

This guide covers the new signal processing methods added to :class:`ScalarField` and :class:`FieldDict` based on GWpy's ``TimeSeries`` and ``FrequencySeries`` APIs.

Overview
--------

We have implemented **23 new methods** across three priority levels:

- **High Priority (10 methods)**: Core signal processing essential for gravitational wave data analysis
- **Low Priority (12 methods)**: Utility and mathematical operations
- **Medium Priority (9 methods)**: Advanced analysis methods (deferred for future implementation)

All methods are available on both :class:`ScalarField` and :class:`FieldDict` (which applies operations to all components).

Basic Preprocessing
-------------------

Detrending
~~~~~~~~~~

Remove polynomial trends that can distort spectral analysis::

    # Remove linear trend
    detrended = field.detrend('linear')

    # Remove DC offset only
    detrended = field.detrend('constant')

Tapering
~~~~~~~~

Apply window functions to suppress FFT ringing artifacts::

    from astropy import units as u

    # Taper both ends with 1 second
    tapered = field.taper(duration=1.0*u.s)

    # Taper left side only with 100 samples
    tapered = field.taper(side='left', nsamples=100)

Cropping and Padding
~~~~~~~~~~~~~~~~~~~~

Extract time segments or extend data::

    # Extract segment from 10s to 20s
    segment = field.crop(start=10*u.s, end=20*u.s)

    # Pad 100 samples on each end with zeros
    padded = field.pad(100)

    # Pad asymmetrically with edge values
    padded = field.pad((50, 150), mode='edge')

Mathematical Operations
-----------------------

Statistical Operations
~~~~~~~~~~~~~~~~~~~~~~

Compute statistics along any axis::

    # Global statistics
    mean_val = field.mean()
    median_val = field.median()
    std_val = field.std()
    rms_val = field.rms()

    # Time-axis statistics (reduces to spatial field)
    time_mean = field.mean(axis=0)
    time_rms = field.rms(axis=0)

    # Spatial statistics
    x_profile = field.mean(axis=1)

Element-wise Operations
~~~~~~~~~~~~~~~~~~~~~~~

::

    # Absolute value
    abs_field = field.abs()

    # Square root
    sqrt_field = field.sqrt()

Advanced Signal Processing
---------------------------

Whitening
~~~~~~~~~

Normalize the amplitude spectral density to flatten the spectrum::

    # Whiten using 2-second segments with 1-second overlap
    whitened = field.whiten(fftlength=2.0, overlap=1.0)

This is essential preprocessing before matched filtering, as it emphasizes weak signals buried in colored noise.

Convolution
~~~~~~~~~~~

Apply FIR filters via time-domain convolution::

    import numpy as np

    # Simple matched filter template
    template = np.array([1, 2, 3, 2, 1]) / 9.0
    matched = field.convolve(template, mode='same')

Signal Injection
~~~~~~~~~~~~~~~~

Add simulated signals for testing detection pipelines::

    # Create a simulated plane wave
    signal = ScalarField.simulate('plane_wave',
                                   shape=(1000, 10, 10, 10),
                                   frequency=100,
                                   amplitude=1e-21)

    # Inject with scaling factor
    injected = field.inject(signal, alpha=0.5)

Filtering Methods
~~~~~~~~~~~~~~~~~

Zero-Pole-Gain (ZPK) filter::

    # Custom IIR filter
    zeros = [0]
    poles = [-1, -1+1j, -1-1j]
    gain = 1.0
    filtered = field.zpk(zeros, poles, gain)

Cross-Spectral Analysis
-----------------------

Cross-Spectral Density
~~~~~~~~~~~~~~~~~~~~~~

Analyze relationships between different channels or spatial points::

    # CSD between two fields
    csd_result = field1.csd(field2, fftlength=2.0, overlap=1.0)

Coherence
~~~~~~~~~

Compute frequency-coherence (values 0-1) indicating correlation at each frequency::

    # Identify correlated noise sources
    coh = field1.coherence(field2, fftlength=2.0, overlap=1.0)

    # High coherence (>0.8) suggests correlated signals/noise
    correlated_freqs = coh.value > 0.8

Spectrogram
~~~~~~~~~~~

Generate time-frequency representations::

    # Spectrogram with 1s stride and 2s FFT length
    spec = field.spectrogram(stride=1.0, fftlength=2.0, overlap=1.0)

Time Series Utilities
----------------------

Compatibility Checking
~~~~~~~~~~~~~~~~~~~~~~

Validate if fields can be combined::

    if field1.is_compatible(field2):
        combined = field1.append(field2)

    if field1.is_contiguous(field2):
        # Fields are adjacent in time
        combined = field1.append(field2, gap='ignore')

Concatenation
~~~~~~~~~~~~~

Append or prepend time segments::

    # Simple append
    combined = field1.append(field2)

    # Handle gaps by padding
    combined = field1.append(field2, gap='pad', pad=0.0)

    # Prepend (field2 comes before field1)
    combined = field1.prepend(field2)

Value Extraction
~~~~~~~~~~~~~~~~

Extract field values at specific times::

    # Single time point (returns 3D spatial array)
    values_3d = field.value_at(5.0 * u.s)

    # Multiple time points (returns ScalarField)
    times = [1.0, 2.0, 3.0] * u.s
    subset = field.value_at(times)

FieldDict Operations
--------------------

All methods work on :class:`FieldDict` by applying operations to each component::

    from gwexpy.fields import FieldDict

    # Create a vector field as FieldDict
    vector_field = FieldDict({
        'x': Ex_field,
        'y': Ey_field,
        'z': Ez_field
    })

    # Preprocess all components
    detrended = vector_field.detrend('linear')
    whitened = vector_field.whiten(fftlength=2.0)

    # Cross-spectral analysis between components
    csd_xy = vector_field['x'].csd(vector_field['y'])
    coh_xy = vector_field['x'].coherence(vector_field['y'])

    # Statistical operations on all components
    rms_dict = vector_field.rms(axis=0)
    mean_dict = vector_field.mean(axis=0)

Medium Priority Methods
------------------------

The following advanced analysis methods have also been implemented:

Correlation Analysis
~~~~~~~~~~~~~~~~~~~~

Autocorrelation Function
^^^^^^^^^^^^^^^^^^^^^^^^^

Reveals periodic structures and characteristic timescales::

    # Compute autocorrelation
    acf = field.autocorrelation(maxlag=100)

    # Peak at non-zero lag indicates periodicity
    # acf.value[maxlag] is always 1.0 (zero-lag correlation)

Cross-Correlation
^^^^^^^^^^^^^^^^^

Time-domain correlation for time-delay estimation::

    # Cross-correlation between two fields
    xcf = field1.correlate(field2, maxlag=100)

    # Find time delay
    lag_idx = np.argmax(xcf.value[:, 0, 0, 0])
    time_delay = xcf._axis0_index[lag_idx]

Interpolation Resampling
~~~~~~~~~~~~~~~~~~~~~~~~~

High-quality resampling using interpolation (better than FFT for calibrated data)::

    # Cubic interpolation resampling
    resampled = field.interpolate(4096, kind='cubic')

    # Other interpolation methods
    resampled = field.interpolate(4096, kind='linear')  # Faster
    resampled = field.interpolate(4096, kind='quintic')  # Higher order

Rayleigh Statistics
~~~~~~~~~~~~~~~~~~~~

Detect non-Gaussian spectral features and spectral lines.

Rayleigh Spectrum
^^^^^^^^^^^^^^^^^

Ratio of maximum to mean bin power as a function of frequency::

    # Compute Rayleigh spectrum
    ray_spec = field.rayleigh_spectrum(fftlength=2.0, overlap=1.0)

    # Values >> 2 indicate non-Gaussian features
    # R ≈ 2 for Gaussian noise
    non_gaussian_freqs = ray_spec.value > 4.0

Rayleigh Spectrogram
^^^^^^^^^^^^^^^^^^^^

Time-frequency Rayleigh statistic for transient feature detection::

    # Time-frequency Rayleigh analysis
    ray_spec = field.rayleigh_spectrogram(
        stride=1.0, fftlength=2.0, overlap=1.0
    )

    # Useful for glitch classification and data quality

Examples
--------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical gravitational wave data analysis workflow::

    import numpy as np
    from astropy import units as u
    from gwexpy.fields import ScalarField

    # 1. Load/create field data
    field = ScalarField(data, unit=u.m/u.s, axis0=times, ...)

    # 2. Preprocessing
    field = field.detrend('linear')        # Remove trends
    field = field.taper(duration=1.0*u.s)  # Suppress edge effects
    field = field.highpass(10)             # Remove low frequencies

    # 3. Whitening for matched filtering
    whitened = field.whiten(fftlength=2.0, overlap=1.0)

    # 4. Matched filtering with template
    template = create_template()  # Your template function
    matched = whitened.convolve(template, mode='same')

    # 5. Analysis
    snr_map = matched.abs()
    peak_time = times[np.argmax(snr_map.value[:, 0, 0, 0])]

    # 6. Spectral analysis
    psd = field.psd(axis=0, fftlength=2.0)
    spec = field.spectrogram(stride=0.5, fftlength=1.0)

Multi-Detector Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Analyzing data from multiple detectors::

    # Create FieldDict for multiple detectors
    detectors = FieldDict({
        'H1': h1_field,  # LIGO Hanford
        'L1': l1_field,  # LIGO Livingston
        'V1': v1_field   # Virgo
    })

    # Preprocess all detectors identically
    detectors = detectors.detrend('linear')
    detectors = detectors.bandpass(30, 300)
    detectors = detectors.whiten(fftlength=4.0)

    # Cross-coherence between detectors
    coh_HL = detectors['H1'].coherence(detectors['L1'], fftlength=4.0)
    coh_HV = detectors['H1'].coherence(detectors['V1'], fftlength=4.0)

    # Inject test signal into all detectors
    signal = ScalarField.simulate('plane_wave', ...)
    injected = detectors.inject(signal, alpha=1.0)

See Also
--------

- :doc:`/web/en/reference/api/fields` - Complete API reference
- :doc:`/web/en/examples/index` - More examples
- `GWpy TimeSeries Documentation <https://gwpy.github.io/docs/stable/timeseries/>`_ - Reference implementation
