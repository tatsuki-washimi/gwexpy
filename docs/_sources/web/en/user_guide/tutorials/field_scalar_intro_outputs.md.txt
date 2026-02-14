# ScalarField Tutorial - Expected Outputs

This document describes the expected visual outputs and results from the ScalarField tutorial notebook (`field_scalar_intro.ipynb`).

## Note on Execution

The ScalarField tutorial notebook now includes embedded outputs for implemented sections.
You can view the main plots directly on the documentation page.

Run locally if you want to regenerate outputs or modify parameters:

1. **Download the notebook** from the documentation page
2. **Set up your environment**:
   ```bash
   pip install gwexpy matplotlib numpy astropy
   ```
3. **Run the notebook** in Jupyter:
   ```bash
   jupyter notebook field_scalar_intro.ipynb
   ```

## Expected Visualizations

### Section 3: Time-Frequency Transformation

**Plot 1: FFT Spectrum**
- X-axis: Frequency (Hz), range 0-50 Hz
- Y-axis: Amplitude (V)
- Shows a clear peak at 10 Hz (the input signal frequency)
- Red dashed line marks the expected peak position
- Demonstrates that ScalarField.fft_time() correctly transforms time domain to frequency domain

**Output text**:
```
Peak frequency: 10.00 Hz (expected: 10.0 Hz)
```

**Plot 2: Inverse FFT Reconstruction**
- X-axis: Time (s)
- Y-axis: Amplitude (V)
- Blue solid line: Original signal
- Red dashed line: Reconstructed signal (overlaps perfectly with original)
- Demonstrates lossless round-trip transformation: time → frequency → time

**Output text**:
```
Max reconstruction error: ~1e-15 V
```

### Section 4: Real Space-K Space Transformation

**Plot 3: Spatial FFT Spectrum**
- X-axis: Wavenumber kx (rad/m)
- Y-axis: Amplitude
- Shows symmetric peaks at ±1.57 rad/m (corresponding to 4m wavelength)
- Green shaded region marks the expected wavenumber
- Demonstrates ScalarField.fft_space() correctly transforms spatial dimensions

**Output text**:
```
Peak wavenumber: ±1.57 rad/m (expected: ±1.57 rad/m)
```

### Section 6: Signal Processing

**Plot 4: Power Spectral Density (PSD)**
- X-axis: Frequency (Hz), logarithmic scale
- Y-axis: PSD, logarithmic scale
- Two curves:
  - Source position (x=1m): Higher PSD at 30 Hz
  - Far field (x=4m): Lower PSD at 30 Hz (1/r² falloff)
- Demonstrates ScalarField.psd() method for spectral analysis

**Plot 5: Frequency-Space Mapping**
- **Current status**: This dedicated visualization is not implemented yet.
- The notebook currently includes a "Coming Soon" section and a future usage example for `plot_freq_space()`.
- Available output today: PSD analysis (Plot 4) and accompanying text outputs.

**Plot 6: Cross-Correlation**
- **Current status**: `plot_cross_correlation()` visualization is planned for a future release.
- The notebook currently includes placeholder code and usage notes only.

**Plot 7: Coherence Map**
- **Current status**: `plot_coherence_map()` visualization is planned for a future release.
- The notebook currently includes placeholder code and usage notes only.

### Section 7: Numerical Invariants

**Output text**:
```
Time FFT Round-trip max error: ~1e-10
Time FFT invariant check: PASSED

Space FFT Round-trip max error: ~1e-10
Space FFT invariant check: PASSED
```

Confirms numerical stability and precision of FFT implementations.

## Key Takeaways from Visualizations

1. **FFT Accuracy**: All transformations are numerically precise (errors < 1e-10)
2. **4D Structure Preservation**: Slicing always maintains 4 dimensions
3. **Domain Tracking**: Metadata correctly tracks time/frequency and real/k-space domains
4. **Signal Processing**: PSD workflow is currently demonstrated with embedded outputs
5. **API Roadmap Clarity**: Frequency-space, cross-correlation, and coherence visualizations are marked as upcoming
6. **Physical Interpretation**: Current plots clearly show wave propagation and spatial dependence

## Generating These Plots

To regenerate all plots and outputs:

```bash
cd docs/web/en/user_guide/tutorials

# Execute notebook and save with outputs
jupyter nbconvert --to notebook --execute \
  field_scalar_intro.ipynb \
  --output field_scalar_intro.ipynb \
  --ExecutePreprocessor.timeout=300

# For Japanese version
cd ../../ja/user_guide/tutorials
jupyter nbconvert --to notebook --execute \
  field_scalar_intro.ipynb \
  --output field_scalar_intro.ipynb \
  --ExecutePreprocessor.timeout=300
```

## Technical Notes

- **Execution time**: ~30-60 seconds depending on hardware
- **Memory usage**: ~500 MB for full notebook execution
- **Dependencies**: numpy, scipy, matplotlib, astropy, gwpy, gwexpy

## Future Improvements

- [ ] Add inline plot annotations explaining key features
- [ ] Include 3D surface plots for spatial fields
- [ ] Add animation examples for time evolution
- [ ] Demonstrate multi-field batch processing with FieldList/FieldDict

---

**Last Updated**: 2026-02-14
**Status**: Main outputs are embedded; three visualization methods remain marked as coming soon
**Priority**: High (before PyPI release recommended, not required)
