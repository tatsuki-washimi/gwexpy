# Notebook Physics Comment Rubric

Use this rubric when editing or reviewing code comments inside tutorial and case-study notebooks.

## When Physics-Intent Comments Are Required

Add explicit physical rationale when a code cell makes an analysis choice that changes scientific interpretation, for example:

- filters, whitening, detrending, or low/high-pass selection
- windows, segment selection, and FFT averaging choices
- coherence, projection, transfer-function, or coupling estimates
- fitting model choice, parameter initialization, and band cropping
- demodulation, lock-in detection, or baseband conversion
- modal transforms, damping control, or calibration steps

## What a Good Comment Explains

1. The physical role of the signal, channel, or parameter.
2. Why this operation is appropriate for the current measurement question.
3. What artifact, ambiguity, or failure mode the step is trying to avoid.

Good comments are short, but they should tell the reader why the step matters physically, not only what API is called next.

## What to Avoid

- Restating the function name with no added meaning.
- Explaining obvious syntax instead of the measurement logic.
- Long textbook paragraphs that interrupt the notebook flow.
- Over-claiming causality when the method only shows correlation or projection.

## Rewrite Examples

```python
# Bad: Calculate ASD for each segment
# Good: Estimate ASD per science-valid segment so non-stationary intervals do not smear transient glitches into the baseline noise floor.

# Bad: 2. Low-pass Filter
# Good: Low-pass after mixing removes the 2*f_c image term, leaving only the slow envelope and phase drift that carry the physical modulation.

# Bad: Closed-loop system
# Good: Close the loop here to test whether modal damping suppresses the suspension resonance without injecting excess cross-coupled motion into other modes.
```

## Review Questions

- If this comment were removed, would a reader still understand why the method choice is physically justified?
- Does the comment distinguish “explains part of the noise” from “proves the mechanism” when using coherence or projection?
- Does the comment mention stationarity, resonance, calibration, or coupling assumptions when those assumptions matter?
