# Bruco (Brute force coherence)

`Bruco` computes **brute force coherence** between a target channel (e.g., a gravitational-wave channel) and a large number of auxiliary channels to identify noise sources.
For a given frequency band and time segment, it ranks auxiliary channels by coherence with the target and estimates their noise contribution (Noise Projection).

The original Bruco implementation is available at:
- https://github.com/mikelovskij/bruco


## Key Features

1. **Batch Processing for Fast Scanning**: Efficiently processes thousands of channels by fetching and computing data in batches of a specified size.
2. **Top-N Rank Retention**: Retains the top N channels with the highest coherence **per frequency bin**, not just by overall maximum coherence.
3. **Parallel Processing**: Accelerates coherence computation using multiprocessing (`concurrent.futures`).
4. **Noise Projection**: Estimates how much each auxiliary channel contributes to the target channel's noise and displays it as a spectrum.
5. **HTML Report**: Generates an HTML report with images summarizing the analysis results.

---

## Usage

### 1. Initialization

Create a `Bruco` instance by specifying the target channel and a list of auxiliary channels to scan.

```python
from gwexpy.analysis.bruco import Bruco

# Target channel name
target = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"

# Auxiliary channel list (typically obtained from NDS, etc.)
aux_channels = [
    "K1:PEM-ACC_MC_TABLE_SENS_Z_OUT_DQ",
    "K1:PEM-MIC_BS_BOOTH_SENS_Z_OUT_DQ",
    "K1:IMC-MCL_SERVO_SUM_OUT_DQ",
    # ... and many more
]

# Optional: channels to exclude (e.g., calibration signals)
excluded = ["K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ", "K1:GRD-LSC_LOCK_STATE_N"]

# Create instance
bruco = Bruco(target, aux_channels, excluded_channels=excluded)
```

### 2. Running the Analysis (Compute)

Depending on your data source, you can use one of the following four patterns (Cases).

#### Case 1: Automatic Data Retrieval (Standard)
Simply specify channel names and time to automatically fetch data (via NDS2 or frame files) and run the analysis.

```python
# Analysis settings
start_gps = 1234567890
duration = 64  # seconds

# Run computation
result = bruco.compute(
    start=start_gps,
    duration=duration,
    batch_size=50,     # Number of channels to fetch at once
    nproc=4            # Number of parallel processes
)
```

#### Case 2: Direct Data Input (Manual)
Analyze using pre-prepared `TimeSeries` objects. Useful for simulation data or data obtained through special methods.
Since timing information (`t0`, `duration`) is taken from the data, the `start` and `duration` arguments can be omitted.

**A. Pass as a dictionary (all data pre-loaded)**
```python
aux_dict = TimeSeriesDict.read(..., channels=my_channels)
result = bruco.compute(
    aux_data=aux_dict  # Pass dictionary (start, duration auto-inferred)
)
```

**B. Pass as a generator (memory-efficient)**
When handling a large number of channels, you can pass data using a generator.
Note: When using a generator, specifying `start` and `duration` is recommended since auto-inference is difficult. However, if `target_data` is provided, they are inferred from it.

```python
def data_stream(channels):
    for ch in channels:
        yield TimeSeries.get(ch, start, end)

result = bruco.compute(
    start, duration,                   # Recommended when using generator
    aux_data=data_stream(my_channels), # Pass generator
    batch_size=100                     # Process 100 at a time, then release memory
)
```

#### Case 3: Automatic Retrieval + Preprocessing (Hybrid)
When you want `Bruco` to handle data retrieval but need to apply filtering or inter-channel operations before analysis, use a callback function.

```python
def my_preprocessing(batch_data: TimeSeriesDict) -> TimeSeriesDict:
    # Receive batch data, process, and return
    for ch, ts in batch_data.items():
        batch_data[ch] = ts.highpass(10)  # Example: 10 Hz highpass filter
    return batch_data

result = bruco.compute(
    start, duration,
    preprocess_batch=my_preprocessing  # Specify callback
)
```
This combines the convenience of automatic retrieval with the flexibility of custom processing, while still benefiting from parallel computation.

#### Case 4: Mixed Mode (NDS + Manual)
You can analyze channels specified at `Bruco` initialization (automatic retrieval) and data passed to `compute()` via `aux_data` (manual) **simultaneously**.
Both data sources are processed sequentially, and results are merged.

```python
# 1. Initialize with channels for automatic retrieval
bruco = Bruco(target, ["K1:NDS-CHANNEL-1", ...])

# 2. Create manual data dictionary
manual_dict = TimeSeriesDict(...)

# 3. Run with both sources
# Note: manual_dict timing must match start/duration.
result = bruco.compute(
    start, duration,
    aux_data=manual_dict
)
```
**Note**: If the timing (`t0`, `duration`) of data in `aux_data` does not fully cover the analysis interval specified by `start` and `duration`, a `ValueError` is raised.

### 3. Viewing and Saving Results

`compute()` returns a `BrucoResult` object. Use this object to visualize results and create reports.

#### Step 3.1: Plotting Coherence
Displays the channels with the highest coherence at each frequency, color-coded.

```python
fig_coh = result.plot_coherence()
fig_coh.show()
```

By default, coherence spectra are shown for the **top contributing channels (Top-K)**.
To display by rank as in the previous behavior, specify `ranks=[0, 1, ...]`.

#### Step 3.2: Plotting Noise Projection
Overlays the target ASD with the noise contribution (Noise Projection) from each auxiliary channel.
Use the `asd` argument (boolean) to switch between ASD (default) and PSD display.

```python
# Display as ASD (default, asd=True)
fig_proj = result.plot_projection()
fig_proj.show()

# Display as PSD
fig_proj_psd = result.plot_projection(asd=False)
fig_proj_psd.show()
```

#### Step 3.3: Generating HTML Report
Creates a directory and outputs an HTML report summarizing the results.

```python
# Output to 'bruco_report' directory
result.generate_report(output_dir="bruco_report")
```

---

## Architecture and Notes

- **Data Retrieval**: Uses `TimeSeriesDict.get()` to fetch data. If some channels in a batch fail to load, it automatically switches to individual retrieval mode and analyzes only valid channels.
- **Resampling**: When the target and auxiliary channels have different sampling rates, the data is automatically downsampled to match the slower rate.
- **Internal PSD Basis**: The internal representation is unified as PSD; ASD display is converted only at rendering time. `coherence_threshold` is applied on amplitude coherence basis when `asd=True`.
- **Memory Management**: When handling a very large number of channels, adjust `batch_size` and `block_size` to control memory usage.

### Top-N Update Block Size

`BrucoResult` Top-N updates process channels in blocks.
The block size is determined in the following order:

1. `block_size` argument (`int` or `"auto"`)
2. `GWEXPY_BRUCO_BLOCK_SIZE` environment variable (`int` or `"auto"`)
3. Default `256`

When set to `"auto"`, it uses `GWEXPY_BRUCO_BLOCK_BYTES` to estimate:

```
max_cols = (block_bytes // (n_bins * 8)) - top_n
block_size = clamp(max_cols, 16, 1024)
```

To determine a target `block_size`, use this as a guideline:

```
block_bytes ~= (top_n + block_size) * n_bins * 8
```

Example: `n_bins=20000`, `top_n=5`, `block_size=256`

```bash
export GWEXPY_BRUCO_BLOCK_SIZE=auto
export GWEXPY_BRUCO_BLOCK_BYTES=41760000
```

### Benchmark

Run a simple benchmark of `update_batch` with `scripts/bruco_bench.py`:

```bash
python scripts/bruco_bench.py --n-bins 20000 --n-channels 300 --top-n 5 --block-size auto
```

Reference values (environment-dependent):

```
elapsed_s=0.153
ru_maxrss_kb=627808
block_size_resolved=414
```

## API Reference

### `Bruco`

**`__init__(self, target_channel: str, aux_channels: List[str], excluded_channels: List[str] = None)`**
- `target_channel`: The main channel to analyze.
- `aux_channels`: List of auxiliary channel names for comparison.
- `excluded_channels`: List of channel names to exclude from analysis.

**`compute(self, start=None, duration=None, fftlength=2.0, overlap=1.0, nproc=4, batch_size=100, top_n=5, block_size=None, ...) -> BrucoResult`**
- `start`: GPS start time. Can be omitted if inferable from data (`target_data` or `aux_data` dict).
- `duration`: Length of analysis data (seconds). Can be omitted if inferable.
- `fftlength`: FFT length for spectral computation (seconds).
- `overlap`: Overlap length (seconds).
- `nproc`: Number of processes for parallel computation.
- `batch_size`: Number of channels to fetch at once.
- `top_n`: Number of top channels to retain per frequency bin.
- `block_size`: Block size for Top-N updates (`int` or `"auto"`).
- `target_data`: (`TimeSeries`) Pre-fetched target data.
- `aux_data`: (`TimeSeriesDict` or `Iterable`) Pre-fetched auxiliary channel data.
- `preprocess_batch`: (`Callable`) Callback function for batch preprocessing.

### `BrucoResult`

**`plot_coherence(self, asd=True, coherence_threshold=0.0, channels=None, ranks=None)`**
- Plots the coherence spectrum.
- **Default behavior**: When neither `channels` nor `ranks` is specified, automatically selects and plots the top contributing channels.
- `channels`: Specify particular channel names to plot.
- `ranks`: Specify particular ranks (0=highest) to plot (legacy behavior).
- `asd=True`: Display amplitude coherence ($\sqrt{C_{xy}^2}$).
- `asd=False`: Display squared coherence ($C_{xy}^2$).
- `coherence_threshold`: Displays a threshold line at the specified value (automatically converted to match the `asd` setting).

**`plot_projection(self, asd=True, coherence_threshold=0.0, channels=None, ranks=None)`**
- Plots the target ASD and noise projections.
- **Default behavior**: When neither `channels` nor `ranks` is specified, automatically selects and plots the top contributing channels.
- `channels`: Specify particular channel names to plot.
- `ranks`: Specify particular ranks to plot (legacy behavior).
- `asd=True`: Display as ASD (Amplitude Spectral Density). `False` for PSD.
- `coherence_threshold`: Masks contributions at frequencies with coherence below this value as `NaN` (appears as gaps in the plot).

**`generate_report(self, output_dir="bruco_report", asd=True)`**
- Generates a report (HTML, PNG, CSV) in the specified directory.
