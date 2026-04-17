# Case Study: Failure-Prone HHT Analysis Patterns

HHT is powerful for nonlinear, nonstationary signals, but it is also easy to over-interpret. Poor parameter choices can make HHT look less stable than STFT, even when the underlying signal is real. This page complements :doc:`advanced_hht` by documenting common failure modes in practice.

## What this case study covers

- mode mixing when plain EMD is used on noisy data
- endpoint artifacts that look like physical frequency changes
- assuming each IMF maps one-to-one to a physical mode
- comparing STFT and HHT under unfair conditions

## Prerequisites

- for the main API and SASI example, see :doc:`advanced_hht`
- for broader time-frequency comparisons, also see :doc:`time_frequency_comparison`

## Failure Mode 1: Using plain EMD for a weak track in noise

**Symptom**: the 130 Hz component is split across multiple IMFs and the instantaneous frequency jitters strongly.  
**Typical cause**: `emd_method="emd"` is used unchanged even though the data are noise-dominated.

```python
hht_plain = ts.hht(
    emd_method="emd",
    output="spectrogram",
)
```

Start by comparing against an EEMD-based result.

```python
hht_eemd = ts.hht(
    emd_method="eemd",
    eemd_trials=20,
    output="spectrogram",
)
```

**Checks**:

- is the same track split across multiple IMFs?
- does the main track stay in place when `eemd_trials` changes?
- are you tracking the signal, or just the loudest bursty noise?

## Failure Mode 2: Reading endpoint artifacts as physics

The Hilbert transform is unstable near the boundaries of the analysis window. That can create spurious instantaneous-frequency jumps at the start or end of the segment.

```python
hht_spec = ts.hht(
    emd_method="eemd",
    eemd_trials=20,
    hilbert_kwargs={"pad": 256},
    output="spectrogram",
)
```

**Operational rule**:

- exclude the first and last few hundred samples from interpretation
- compare padded and unpadded runs
- do not assign physical meaning to changes that only appear at the edges

## Failure Mode 3: Naming IMFs as physical modes too early

**Symptom**: conclusions like `IMF 2 = SASI`, `IMF 3 = convection` appear immediately after decomposition.  
**Typical cause**: the algorithmic decomposition is treated as a direct physical labeling.

IMFs are decomposition components, not guaranteed physical modes. Before naming them, verify at least the following:

1. the band and time evolution are consistent with theory or simulation
2. the same track remains visible when nearby IMFs are recombined
3. STFT or wavelet views show the same event timing

## Failure Mode 4: Comparing STFT and HHT with mismatched preprocessing

HHT often appears superior in time resolution, but the visual comparison can be misleading if the preprocessing and parameters are not aligned.

- feed the same preprocessed `TimeSeries` into both methods
- state the STFT window length and overlap explicitly
- state `emd_method`, `eemd_trials`, and `hilbert_kwargs` for HHT

```python
ts_white = ts.whiten(fftlength=1.0, overlap=0.5)
stft = ts_white.spectrogram2(fftlength=0.05, overlap=0.045)
hht = ts_white.hht(emd_method="eemd", eemd_trials=20, output="spectrogram")
```

If STFT misses a feature, that does not automatically validate HHT. If HHT looks noisy, that does not automatically validate STFT. They fail differently.

## Recommended workflow

1. reproduce the baseline SASI example from :doc:`advanced_hht`
2. move to EEMD or CEEMD early when the signal is noise-dominated
3. set padding and an ignored edge region before interpreting instantaneous frequency
4. compare single-IMF results against recombined IMFs and STFT
5. only assign physical labels after cross-checking against theory or simulation

## Related pages

- :doc:`advanced_hht`
- :doc:`case_glitch_analysis`
- :doc:`time_frequency_analysis_comparison`
- :doc:`time_frequency_comparison`
