# Calibration and Physical Units

Techniques for applying frequency-dependent calibration transfer functions, verifying physical dimensional units, and guarding against inversion and grid errors.

## Analysis Pathways

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Calibration Units Contract
:link: calibration_units_contract
:link-type: doc

Apply frequency-dependent response $C(f)$ to complex spectra and ASDs while enforcing astropy physical unit integrity.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Calibration Pipeline
:link: ../case-studies/case_calibration_pipeline
:link-type: doc

End-to-end calibration measurement, swept/stepped sine analysis, and model fitting.
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` DTT XML Calibration
:link: ../case-studies/case_dttxml_calibration
:link-type: doc

Load pre-measured calibration transfer functions directly from diagnostic test tool (DTT) XML files.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Physics Validation
:link: ../case-studies/case_physics_validation
:link-type: doc

Automated unit sanity checks and consistency guards for gravitational-wave data containers.
:::
::::

## Core Conventions

- **Coupling Function**: $C(f) = Y_{\text{count}}(f) / X_{\text{m}}(f)$ with unit $\text{ct}/\text{m}$.
- **Complex Spectrum Calibration**: $X(f) = Y(f) / C(f)$ restoring physical displacement.
- **Amplitude Spectral Density (ASD)**: $A_x(f) = A_y(f) / |C(f)|$ with unit $\text{m}/\sqrt{\text{Hz}}$.
- **Power Spectral Density (PSD)**: $P_x(f) = P_y(f) / |C(f)|^2$ with unit $\text{m}^2/\text{Hz}$.
- **Strain Conversion**: $h(f) = X(f) / L$ given arm length $L$.

```{toctree}
:hidden:

calibration_units_contract
```
