# Troubleshooting

Common issues encountered when using GWexpy and their solutions.

## Installation Issues

### 1. Cannot install `nds2` / `framel`
Binary libraries used in the `[gw]` extra cannot be installed via `pip`.

**Solution:**
Use Conda (Miniforge, etc.) to install the dependencies first.
```bash
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp
```

### 2. Errors on Apple Silicon (M1/M2/M3) Mac
Some GW analysis packages may be built for Intel (x86_64) and might not work out of the box.

**Solution:**
Native (arm64) support for packages from the `conda-forge` channel is improving. Always ensure you are using the latest versions.
```bash
conda update -c conda-forge --all
```

### 3. Compilation error for `minepy` (MIC calculation)
`pip install minepy` may fail due to C extension compilation issues.

**Solution:**
Run the automated build script included in the repository.
```bash
python scripts/install_minepy.py
```

## Plotting & Visualization Issues

### 4. Plots not displaying / `Tcl_AsyncDelete` error
There may be an inconsistency with the Matplotlib backend in Jupyter Notebook or GUI applications.

**Solution:**
Try explicitly specifying the backend:
```python
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Agg', 'TkAgg'
```

### 5. Map (`GeoMap`) not displaying
Verify the installation status of `pygmt` and ensure the GMT (Generic Mapping Tools) binary is in your path.

**Solution:**
We recommend reinstalling `pygmt` via Conda.
```bash
conda install -c conda-forge pygmt
```

---

## Still Not Working?

Please report the issue with the error logs on the GitHub [Issues](https://github.com/gwexpy/gwexpy/issues) page.
Including the following information will help us assist you faster:
* OS Version
* Python Version
* The command executed and the full Traceback
