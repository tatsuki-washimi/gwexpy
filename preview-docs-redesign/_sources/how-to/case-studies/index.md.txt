# Case studies

Goal-oriented workflows that combine several GWexpy features into a complete,
real-world analysis you can adapt to your own data. For feature-by-feature
lessons, start with the [tutorials](../../tutorials/index.md) and the
[technique recipes](../index.md).

## Featured

::::{grid} 1 2 2 3
:gutter: 4

:::{grid-item-card} Noise budget
:img-top: ../../_static/images/case_noise_budget_thumb.png
:link: case_noise_budget
:link-type: doc

Decompose a target spectrum into its measured contributions with a
BrUCo-style budget.
:::

:::{grid-item-card} Transfer function
:img-top: ../../_static/images/case_transfer_function_thumb.png
:link: case_transfer_function
:link-type: doc

Estimate and plot a multi-channel transfer function from coherent excitations.
:::

:::{grid-item-card} Active damping
:img-top: ../../_static/images/case_active_damping_thumb.png
:link: case_active_damping
:link-type: doc

Design and validate a MIMO active-damping loop for a 6-DOF isolation system.
:::
::::

## I. Calibration, response, and control

- {doc}`Active damping: MIMO control for a 6-DOF isolation system <case_active_damping>`
- {doc}`Transfer function measurement: estimation, coherence, and fitting <case_transfer_function>`
- {doc}`Calibration pipeline: counts-to-strain conversion <case_calibration_pipeline>`
- {doc}`DTT XML workflow: loading and reusing measured response data <case_dttxml_calibration>`

## II. Interoperability, I/O, and reproducibility

- {doc}`Finesse 3 interoperability: simulation vs. measurement <case_finesse_optics>`
- {doc}`ObsPy interoperability: ingesting and analyzing seismic data <case_seismic_obspy>`
- {doc}`GBD format I/O: round-tripping detector data products <case_gbd_format>`
- {doc}`HDF5 provenance: reproducible metadata management <case_hdf5_provenance>`
- {doc}`PyCBC interoperability: from gwexpy preprocessing to search <case_pycbc_search>`

## III. Statistical and ML workflows

- {doc}`Bootstrap PSD and GLS fitting <case_bootstrap_gls_fitting>`
- {doc}`ML preprocessing pipeline: feature engineering and comparison <case_ml_preprocessing>`
- {doc}`Event-synchronized analysis: SegmentTable-driven window selection <case_segment_analysis>`
- {doc}`Physical validity checking: units, floors, and sanity tests <case_physics_validation>`
- {doc}`ARIMA-based burst detection <case_arima_burst_search>`
- {doc}`Signal extraction: weak signal recovery from colored noise <case_signal_extraction>`

## IV. Noise hunting and detector diagnostics

- {doc}`Noise budgeting: identifying dominant noise couplings <case_noise_budget>`
- {doc}`Lock-in detection: recovering weak AM/FM structure <case_lockin_detection>`
- {doc}`Wiener filtering: coherent noise subtraction <case_wiener_filter>`
- {doc}`Coupling analysis: estimating transfer paths between channels <case_coupling_analysis>`
- {doc}`Bruco and ICA noise reduction: witness selection to subtraction <case_bruco_ica_denoising>`
- {doc}`Bruco advanced: bilinear coupling and AM/FM failure modes <case_bruco_advanced>`
- {doc}`Violin mode analysis: fitting and tracking resonance families <case_violin_mode>`
- {doc}`Schumann resonance analysis <case_schumann_resonance>`
- {doc}`Glitch analysis: Q-transform and Omega-scan <case_glitch_analysis>`

:::{note}
Thumbnails are shown for the featured case studies above; the remaining
galleries are listed as links until executed-notebook thumbnails are generated
in CI.
:::

```{toctree}
:hidden:

case_active_damping
case_transfer_function
case_calibration_pipeline
case_dttxml_calibration
case_finesse_optics
case_seismic_obspy
case_gbd_format
case_hdf5_provenance
case_pycbc_search
case_bootstrap_gls_fitting
case_ml_preprocessing
case_segment_analysis
case_physics_validation
case_arima_burst_search
case_signal_extraction
case_noise_budget
case_lockin_detection
case_wiener_filter
case_coupling_analysis
case_bruco_ica_denoising
case_bruco_advanced
case_violin_mode
case_schumann_resonance
case_glitch_analysis
```
