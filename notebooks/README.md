# Notebooks for Paper

This directory contains executable marimo notebooks that reproduce the examples and figures from the GWexpy paper.

## Contents

These marimo notebooks demonstrate the key features of GWexpy as described in the SoftwareX manuscript:

1. **01_transfer_function_workflow.py** - Transfer function estimation (Figure 2)
   - DTT XML I/O workflow
   - Computing transfer functions using `TimeSeries.transfer_function()`
   - Bode plot visualization with `gwexpy.plot.Plot`
   - Exports: `docs/gwexpy-paper/figure2_transfer_function.{png,pdf}`

2. **02_coherence_ranking.py** - Coherence-based coupling analysis (Figure 3)
   - Multi-channel `TimeSeriesMatrix` construction
   - `coherence_ranking()` for BruCo-style noise hunting
   - Top-k channel identification with `topk()` and `plot_ranked()`
   - Exports: `docs/gwexpy-paper/figure3_coherence_ranking.{png,pdf}`

## Prerequisites

Install GWexpy with the required extras:

```bash
pip install -e ".[analysis,gw]"
```

Install marimo:

```bash
pip install marimo
```

## Running the Notebooks

### Interactive mode (browser-based)

```bash
marimo edit notebooks/01_transfer_function_workflow.py
```

### Script mode (headless)

```bash
marimo run notebooks/01_transfer_function_workflow.py
```

Or execute directly:

```bash
python notebooks/01_transfer_function_workflow.py
```

## Generated Figures

Running the notebooks will generate paper figures in `docs/gwexpy-paper/`:
- `figure2_transfer_function.{png,pdf}`
- `figure3_coherence_ranking.{png,pdf}`

## CI Testing

These marimo notebooks are standard Python scripts and can be executed in CI:

```bash
python notebooks/01_transfer_function_workflow.py
python notebooks/02_coherence_ranking.py
```

## Data

All notebooks use synthetic data to ensure reproducibility without requiring access to proprietary detector data.
