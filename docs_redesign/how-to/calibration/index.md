# Calibration and Physical Units

Techniques for applying frequency-dependent calibration transfer functions, verifying physical dimensional units, and guarding against inversion and grid errors.

## Analysis Pathways

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Calibration Units Contract
:link: calibration_units_contract
:link-type: doc

Core tutorial: Apply frequency response $C(f)$ [count / m] to complex spectra and spectral densities while enforcing Astropy physical unit contracts and defensive exception guards.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Calibration Pipeline
:link: ../case-studies/case_calibration_pipeline
:link-type: doc

Case study: End-to-end swept/stepped sine excitation analysis, coherence tracking, and parametric actuator transfer function fitting.
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` DTT XML Calibration
:link: ../case-studies/case_dttxml_calibration
:link-type: doc

Case study: Ingest and reuse pre-measured calibration transfer function products directly from LIGO Diagnostic Test Tool (DTT) XML files.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Physics Validation
:link: ../case-studies/case_physics_validation
:link-type: doc

Case study: Broad automated sanity checks, dimensional integrity auditing, and consistency validation across GWexpy containers.
:::
::::

## Role Separation and Scope

| Resource | Primary Scope | Typical Usage | Key Focus |
|---|---|---|---|
| **Calibration Units Contract** (Tutorial) | In-memory series calibration | Daily script analysis | Unit safety, wrong-direction guards, grid mismatch errors |
| **Calibration Pipeline** (Case Study) | Experimental measurement | Hardware commissioning | Swept sine analysis, actuator calibration, TF modeling |
| **DTT XML Calibration** (Case Study) | File I/O & legacy interop | Commissioning data reuse | Parsing XML transfer functions into GWexpy containers |
| **Physics Validation** (Case Study) | Container test suite | Quality assurance | Broad physics invariant assertions |

## Core Mathematical Conventions

- **Coupling Response**: $C(f) = Y_{\text{count}}(f) / X_{\text{m}}(f)$ with physical unit $\text{ct}/\text{m}$.
- **Complex Spectrum Calibration**: $X(f) = Y(f) / C(f)$ restoring physical displacement in meters.
- **Amplitude Spectral Density (ASD)**: $A_x(f) = A_y(f) / |C(f)|$ with unit $\text{m}/\sqrt{\text{Hz}}$.
- **Power Spectral Density (PSD)**: $P_x(f) = P_y(f) / |C(f)|^2$ with unit $\text{m}^2/\text{Hz}$.
- **Strain Sensitivity**: $h(f) = X(f) / L$ given detector arm length $L$.

```{toctree}
:hidden:

calibration_units_contract
```
