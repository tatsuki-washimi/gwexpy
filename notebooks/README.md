# Notebooks for Paper

This directory contains executable notebooks that reproduce the examples and figures from the GWexpy paper.

## Contents

These notebooks demonstrate the key features of GWexpy as described in the SoftwareX manuscript:

1. **DTT XML I/O** - Reading DTT XML files with the `products` parameter
2. **Transfer Function Workflow** - Computing transfer functions in steady and transient modes
3. **Multi-Format Integration** - Reading and integrating heterogeneous data formats

## Running the Notebooks

Install GWexpy with the required extras:

```bash
pip install -e ".[analysis,gw]"
```

Then run the notebooks:

```bash
jupyter notebook notebooks/
```

## CI Testing

These notebooks are automatically executed in CI using `nbmake` to ensure reproducibility:

```bash
pytest --nbmake notebooks/ --nbmake-timeout=600
```

## Data

All notebooks use synthetic or anonymized data to ensure reproducibility without requiring access to proprietary detector data.
