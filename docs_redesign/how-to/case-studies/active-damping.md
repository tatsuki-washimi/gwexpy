# Case study: active damping of a mechanical resonance

```{admonition} Goal
:class: tip
Reduce the quality factor of a lightly damped mechanical mode by closing a
feedback loop, and confirm the result in the measured spectrum.
```

## Problem

A suspension mode rings at $f_0 \approx 12\ \mathrm{Hz}$ with a high $Q$,
injecting narrow-band motion into the readout. We want to add electronic
damping without compromising broadband performance.

## Approach

1. **Measure** the open-loop transfer function from actuator to sensor.
2. **Model** the resonance and design a band-limited damping filter.
3. **Close** the loop and predict the suppressed spectrum.
4. **Validate** against a fresh measurement.

```{figure} ../../_static/case_active_damping_thumb.png
:width: 70%
:align: center

Open-loop (light) vs. damped (dark) amplitude spectral density around the
resonance.
```

## Result

The damped loop lowers the peak by roughly an order of magnitude while leaving
the noise floor away from $f_0$ untouched.

```{seealso}
The full runnable analysis lives in the `case_active_damping.ipynb` notebook;
related studies include transfer-function estimation and noise budgeting from
the {doc}`gallery <index>`.
```
