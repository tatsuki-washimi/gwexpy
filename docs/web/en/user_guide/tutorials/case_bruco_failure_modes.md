# Case Study: Failure-Prone Bruco Analysis Patterns

Bruco is effective for scanning large auxiliary-channel sets, but it is also easy to misuse: you can get a strong coherence ranking and still fail to identify the real coupling path. This case study complements :doc:`advanced_bruco` by focusing on recurrent failure modes that appear in detector characterization work.

## What this case study covers

- missing nonlinear couplings when only linear channels are scanned
- ranking false positives caused by shared drift or low-frequency trends
- treating "high coherence" as proof of causality
- validating top-ranked channels before downstream subtraction

## Prerequisites

- for the basic API and minimal workflow, see :doc:`advanced_bruco`
- for class details, see :doc:`../../reference/Bruco`

## Failure Mode 1: Trusting a linear Bruco scan too early

**Symptom**: the expected witness channel does not appear near the top, while loosely related environmental channels do.  
**Typical cause**: the true coupling path is bilinear or amplitude-modulated, but only raw auxiliary channels were provided.

```python
from gwexpy.analysis import Bruco

bruco = Bruco(target_channel=target.name, aux_channels=aux_names)
linear_result = bruco.compute(
    start=target.t0.value,
    duration=target.duration.value,
    fftlength=4.0,
    overlap=2.0,
    aux_data=aux_matrix,
)
linear_result.to_dataframe(ranks=[0, 1, 2])
```

If the ranking remains unstable, promote explicit hypotheses into virtual auxiliary channels rather than scanning only the original sensors.

```python
virtual_aux = aux_matrix.copy()
virtual_aux["ASC_X2"] = aux_matrix["ASC_X"] ** 2
virtual_aux["PEM_ACC_TIMES_MIC"] = aux_matrix["PEM_ACC"] * aux_matrix["PEM_MIC"]
```

**Checks**:

- does the top ranking change after virtual channels are added?
- do the candidate bands line up with the problematic ASD features?
- do the strongest channels cluster around one physical subsystem?

## Failure Mode 2: Ranking drift instead of coupling

**Symptom**: channels with large low-frequency trends always rise to the top.  
**Typical cause**: coherence was computed before detrending or band-limiting, so shared drift is mistaken for a coupling path.

```python
preprocessed = {}
for name, ts in aux_matrix.items():
    preprocessed[name] = ts.detrend().highpass(5.0)

target_clean = target.detrend().highpass(5.0)
```

**What this prevents**:

- promoting ground motion or temperature drift as the main culprit
- making commissioning decisions from a single ranking table

## Failure Mode 3: Treating coherence as causality

Bruco measures how strongly the target and an auxiliary channel are related. It does not, by itself, prove a direct causal path. Two channels can rank highly because they share a third driver.

**Minimum validation loop**:

1. confirm that the band matches the actual problem band
2. test whether a residual or subtraction step improves the target ASD
3. check whether the candidate is consistent with the hardware layout

```python
top = linear_result.to_dataframe(ranks=[0]).iloc[0]
candidate = virtual_aux[top["channel"]]
residual = target_clean - candidate * 0.1
```

The coefficient here is intentionally simplified. The important point is to test whether the top-ranked channel actually reduces the problematic structure once it is used downstream.

## Recommended workflow

1. reproduce a baseline result from :doc:`advanced_bruco`
2. decide whether detrending, bandpass filtering, or whitening is required for the target band
3. add virtual channels only when linear witnesses do not explain the feature
4. validate top-ranked witnesses with residual ASD or subtraction tests
5. cross-check the result against known control paths and sensor placement

## Related pages

- :doc:`advanced_bruco`
- :doc:`case_bruco_advanced`
- :doc:`case_bruco_ica_denoising`
- :doc:`case_noise_budget`
