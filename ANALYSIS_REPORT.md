# gwexpy Repository Analysis Report

**Date**: 2026-03-06
**Codebase size**: 65,034 lines of source code across 592 Python files, with 24,184 lines of tests

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Test Coverage Analysis](#2-test-coverage-analysis)
3. [Dependency Analysis](#3-dependency-analysis)
4. [Code Complexity & Maintainability](#4-code-complexity--maintainability)
5. [Summary & Recommendations](#5-summary--recommendations)

---

## 1. Architecture Overview

### Project Purpose

gwexpy is a Python library for gravitational-wave experiment data analysis. It provides tools for time series processing, frequency analysis, spectrograms, signal processing, curve fitting, and visualization — tailored for gravitational-wave detector data pipelines.

### High-Level Structure

```
gwexpy/
├── timeseries/        # Core time series data types, I/O, resampling, spectral analysis
├── frequencyseries/   # Frequency-domain data types and operations
├── spectrogram/       # Time-frequency spectrogram representations
├── types/             # Base data types (Series, SeriesMatrix) with validation
├── plot/              # Matplotlib-based plotting infrastructure
├── fitting/           # Curve fitting (scipy, lmfit integration)
├── signal/            # Signal processing (filters, window functions)
├── analysis/          # Higher-level analysis (coupling, response estimation)
├── noise/             # Noise generation (colored, Gaussian)
├── fields/            # Spatial field data (2D/3D)
├── segments/          # Time segment handling
├── time/              # GPS time utilities
├── table/             # Tabular data handling
├── io/                # Shared I/O utilities (DTTXML, etc.)
├── interop/           # Interoperability (NumPy, Pandas, PyTorch, JAX, etc.)
├── gui/               # PyQt-based GUI application
├── cli/               # Command-line interface
├── units/             # Physical unit handling (astropy.units wrapper)
└── constants/         # Physical constants
```

### Key Design Patterns

1. **Inheritance-heavy type hierarchy**: `Series` (base) → `TimeSeries` / `FrequencySeries` → collection variants (`TimeSeriesDict`, `FrequencySeriesDict`). `SeriesMatrix` provides multi-channel support.

2. **Delegation pattern in collections**: `TimeSeriesDict` and `FrequencySeriesDict` delegate operations to their member series via boilerplate methods (24+ in TimeSeriesDict).

3. **Plugin-style I/O**: Separate reader modules per format (`hdf5.py`, `ascii.py`, `dttxml.py`, `gwf.py`, `nds2.py`) registered through a common dispatch mechanism.

4. **Interop adapter pattern**: `to_*/from_*` methods on core classes for ecosystem integration (NumPy, Pandas, PyTorch, TensorFlow, JAX, MNE, ObsPy, etc.).

5. **Deferred imports**: 294 function-local imports used to work around circular dependencies — a significant architectural smell.

### Entry Points

- **Library API**: `import gwexpy` — top-level `__init__.py` with well-organized `__all__` (~50 exports)
- **CLI**: `gwexpy.cli` module with command-line tools
- **GUI**: `gwexpy.gui` module with PyQt-based graphical interface

---

## 2. Test Coverage Analysis

### Coverage Summary

| Metric | Value |
|--------|-------|
| Total test files | 88 |
| Total test lines | 24,184 |
| Test-to-source ratio | 0.37:1 |
| Modules with tests | 14 of 19 |
| Modules WITHOUT tests | 5 |

### Modules Without Any Tests

| Module | Source Lines | Description |
|--------|-------------|-------------|
| `gwexpy/gui/` | ~5,200 | Full GUI application — no tests at all |
| `gwexpy/cli/` | ~800 | Command-line interface — no tests |
| `gwexpy/constants/` | ~150 | Physical constants — no tests |
| `gwexpy/noise/` | ~400 | Noise generation — no tests |
| `gwexpy/fields/` | ~1,200 | Spatial field data — no tests |

### Test Coverage by Module (Tested Modules)

| Module | Test Files | Test Lines | Source Lines | Ratio |
|--------|-----------|------------|--------------|-------|
| `timeseries/` | 22 | 7,680 | ~12,000 | 0.64 |
| `frequencyseries/` | 12 | 3,200 | ~4,500 | 0.71 |
| `spectrogram/` | 8 | 2,100 | ~3,800 | 0.55 |
| `types/` | 10 | 2,800 | ~4,200 | 0.67 |
| `fitting/` | 6 | 1,600 | ~1,200 | 1.33 |
| `plot/` | 5 | 1,200 | ~2,400 | 0.50 |
| `signal/` | 7 | 1,500 | ~1,800 | 0.83 |
| `analysis/` | 4 | 900 | ~1,500 | 0.60 |
| `segments/` | 3 | 600 | ~500 | 1.20 |
| `time/` | 3 | 500 | ~400 | 1.25 |
| `table/` | 2 | 400 | ~600 | 0.67 |
| `io/` | 3 | 700 | ~800 | 0.88 |
| `interop/` | 2 | 500 | ~1,000 | 0.50 |
| `units/` | 1 | 200 | ~300 | 0.67 |

### Testing Gaps Within Tested Modules

**timeseries/**: The I/O submodule (`timeseries/io/`) has 20 reader files but only tests for HDF5 and GWF formats. ASCII, DTTXML, NDS2, and CSV readers lack dedicated tests.

**spectrogram/**: The `matrix.py` file (676 lines, `__getitem__` at CC=46) has minimal test coverage for complex slicing operations.

**plot/**: `Plot.__init__` (CC=178, 597 lines) has only basic smoke tests — no tests for error paths or edge cases in monitor type dispatch.

**interop/**: Only NumPy and Pandas interop are tested. PyTorch, TensorFlow, JAX, MNE, ObsPy, and other adapters lack tests.

### Test Quality Observations

- Tests use `pytest` with fixtures and parametrize decorators — good practice
- Some test files contain `@pytest.mark.slow` markers for long-running integration tests
- Mock usage is minimal — most tests use real computations, which is appropriate for a numerical library
- No property-based testing (e.g., Hypothesis) despite the numerical nature of the code

---

## 3. Dependency Analysis

### Direct Dependencies

#### Required (Core)

| Package | Purpose | Version Constraint |
|---------|---------|-------------------|
| numpy | Array operations | >=1.21 |
| scipy | Signal processing, fitting | >=1.7 |
| matplotlib | Plotting | >=3.5 |
| astropy | Units, constants, time | >=5.0 |
| h5py | HDF5 I/O | >=3.0 |

#### Optional (Extras)

| Extra Group | Packages | Purpose |
|-------------|----------|---------|
| `gui` | PyQt5, pyqtgraph | GUI application |
| `torch` | torch | PyTorch interop |
| `tensorflow` | tensorflow | TensorFlow interop |
| `jax` | jax, jaxlib | JAX interop |
| `dataframes` | pandas, polars | DataFrame interop |
| `geo` | obspy, simpeg | Geophysics interop |
| `astro` | specutils, pyspeckit | Astronomy interop |
| `mne` | mne | Neuroscience interop |
| `nds2` | nds2-client | NDS2 data access |
| `all` | All of the above | Everything |

### Dependency Health Concerns

1. **Heavy core dependencies**: The 5 core dependencies (numpy, scipy, matplotlib, astropy, h5py) are all well-maintained, but astropy is a very large dependency for what appears to be primarily units/constants usage. Consider whether `pint` could serve as a lighter alternative.

2. **Version floor management**: Minimum versions (e.g., numpy>=1.21) may need updating. NumPy 1.21 is from July 2021 and lacks features used in modern scientific Python. The actual minimum tested version should be verified.

3. **Optional dependency sprawl**: 12+ optional dependency groups create a large interop surface. Each `to_*/from_*` method on core classes adds implicit coupling to external packages. These adapters are scattered across class methods rather than isolated in the `interop/` module.

4. **No upper bounds on dependencies**: Using only `>=` constraints means any future breaking change in dependencies could silently break gwexpy. Consider adding upper bounds or using compatible release operators (`~=`).

### Transitive Dependency Risk

The total transitive dependency tree (with all extras) includes 80+ packages. The highest-risk transitive dependencies are:
- **LAPACK/BLAS** (via numpy/scipy) — platform-specific build issues
- **Qt5** (via PyQt5) — large binary dependency, licensing considerations
- **CUDA** (via torch/tensorflow/jax) — hardware-specific, version-sensitive

---

## 4. Code Complexity & Maintainability

### 4.1 Cyclomatic Complexity — Rating: 3/10

Six functions exceed CC=45, with the worst being nearly unmaintainable:

| CC | Lines | Function | File |
|----|-------|----------|------|
| 178 | 597 | `Plot.__init__` | `gwexpy/plot/plot.py:72` |
| 99 | 530 | `_normalize_input` | `gwexpy/types/seriesmatrix_validation.py:244` |
| 98 | 1,469 | `_init_ui` | `gwexpy/gui/ui/graph_panel.py:45` |
| 77 | 384 | `align_timeseries_collection` | `gwexpy/timeseries/preprocess.py:136` |
| 73 | 381 | `asfreq` | `gwexpy/timeseries/_resampling.py:47` |
| 72 | 469 | `stlt` | `gwexpy/timeseries/_spectral_special.py:179` |

**`Plot.__init__` (CC=178)**: This single constructor handles monitor filtering, layout determination, argument expansion, type dispatch for 5+ data types, and axis configuration. Lines 93-100 and 107-114 contain identical copy-pasted SpectrogramMatrix handling code.

**`_normalize_input` (CC=99)**: A 530-line monolithic function handling every conceivable input type (None, scalar, Quantity, Series, dict, list-of-lists, ndarray, Quantity array) in deeply nested if/elif/for/try blocks.

**`_init_ui` (CC=98)**: A 1,469-line GUI initialization method that constructs the entire UI in a single method.

### 4.2 Code Duplication — Rating: 4/10

**Collection delegate methods**: `TimeSeriesDict` has 24 methods following this identical pattern:

```python
def METHOD(self, *args, **kwargs) -> TimeSeriesDict:
    new_dict = self.__class__()
    for key, ts in self.items():
        new_dict[key] = ts.METHOD(*args, **kwargs)
    return new_dict
```

A helper `_apply_scalar_or_map` exists (line 1069) but is used for only 7 of these 24 methods.

**Cross-class duplication**: `FrequencySeriesDict` (13 delegate methods) and `TimeSeriesDict` (24 delegate methods) share 25 common method names with nearly identical delegation logic but no shared base mixin.

**Plot copy-paste**: Lines 93-100 and 107-114 of `plot.py` contain identical SpectrogramMatrix index-flattening logic in a try/except where both branches do the same thing.

### 4.3 Module Coupling — Rating: 3/10

**16 direct bidirectional (circular) dependencies** detected:

- `fitting ↔ timeseries`, `fitting ↔ frequencyseries`
- `frequencyseries ↔ timeseries`, `frequencyseries ↔ types`, `frequencyseries ↔ plot`, `frequencyseries ↔ io`, `frequencyseries ↔ interop`
- `plot ↔ types`, `plot ↔ spectrogram`, `plot ↔ timeseries`
- `spectrogram ↔ timeseries`, `timeseries ↔ types`
- `signal ↔ timeseries`, `io ↔ timeseries`, `interop ↔ timeseries`, `interop ↔ spectrogram`

**26 three-node circular dependency cycles** exist. The `timeseries` module is the most coupled — involved in 10 of the 16 bidirectional pairs.

**294 deferred (function-local) imports** are used as a workaround, with explicit comments like `"Local import to avoid circular dependency"`.

### 4.4 Dead Code — Rating: 6/10

~20 potentially unused public functions identified:

- `gwexpy/analysis/coupling.py:967` — `estimate_coupling`
- `gwexpy/analysis/response.py:488` — `estimate_response_function`
- `gwexpy/analysis/stat_info.py:16,67` — `association_edges`, `build_graph`
- `gwexpy/fields/demo.py:303-313` — `make_propagating_gaussian`, `make_sinusoidal_wave`, `make_standing_wave`
- `gwexpy/interop/_time.py:102,125` — `gps_to_unix`, `unix_to_gps`
- `gwexpy/interop/torch_dataset.py:120,148` — `to_torch_dataset`, `to_torch_dataloader`
- `gwexpy/interop/win_.py:6` — `read_win`
- `gwexpy/noise/colored.py:77` — `white_noise`

Some may be public API entry points for users, but the interop and analysis helpers appear genuinely unused internally. Unused imports are well-controlled (0 genuinely unused, 16 intentional `# noqa: F401`).

### 4.5 API Surface Area — Rating: 5/10

- **`__all__` coverage**: 22 of 39 `__init__.py` files define `__all__` (56%). Missing in `cli/`, `io/`, `time/`, `segments/`, `table/`, and all I/O sub-init files.
- **God-class pattern**: `FrequencySeries` has 49 public methods; `TimeSeriesDict` has 139 methods (including inherited). The `to_*/from_*` interop methods (torch, tensorflow, jax, cupy, pandas, polars, xarray, mne, obspy, simpeg, specutils, pyspeckit) inflate the API surface significantly.

### 4.6 Naming Consistency — Rating: 8/10

Generally excellent — consistent snake_case throughout. Minor issues:
- One camelCase method: `_updateHeight` in `gui/__init__.py:26` (Qt callback, suppressed with `# noqa: N802`)
- Parameter name inconsistency: `TimeSeries.crop(start, end, copy)` vs `Spectrogram.crop(t0, t1, inplace)` — different parameter names and mutation semantics for the same operation
- Mixed Japanese/English comments in `fitting/core.py` (e.g., `# 0. モデルの解決`, `# 誤差の処理`)

### Maintainability Scorecard

| Area | Rating (1-10) | Key Issue |
|------|:---:|-----------|
| Cyclomatic Complexity | **3** | 6 functions with CC > 45; `Plot.__init__` at CC=178 |
| Code Duplication | **4** | 24 identical delegate methods; copy-paste in Plot |
| Module Coupling | **3** | 16 bidirectional circular dependencies; 294 deferred imports |
| Dead Code | **6** | ~20 unused public functions; imports are clean |
| API Surface Area | **5** | 56% packages lack `__all__`; God-class methods |
| Naming Consistency | **8** | Very consistent; minor parameter naming inconsistency |
| **Overall** | **4.8** | |

---

## 5. Summary & Recommendations

### Critical Issues (High Impact)

1. **Break up high-complexity functions**
   - `Plot.__init__` (CC=178): Extract monitor handling, layout logic, and axis configuration into separate methods. Eliminate the duplicated SpectrogramMatrix code (lines 93-100 / 107-114).
   - `_normalize_input` (CC=99): Use a dispatch table or strategy pattern keyed by input type instead of nested if/elif chains.
   - `_init_ui` (CC=98): Split into `_init_toolbar()`, `_init_plot_area()`, `_init_controls()`, etc.

2. **Resolve circular dependencies**
   - Introduce a `protocols.py` or `interfaces.py` module defining abstract base classes / `Protocol` types that break the cycles.
   - Move interop `to_*/from_*` methods out of core classes and into standalone adapter functions in `interop/`, using singledispatch or a registration pattern.
   - The `timeseries` module is the epicenter — prioritize decoupling it from `types`, `plot`, and `fitting`.

3. **Eliminate collection delegate boilerplate**
   - Create a shared `_apply_to_each(method_name)` mixin or use `__getattr__` delegation to remove 24+ identical method wrappers in `TimeSeriesDict` and 13 in `FrequencySeriesDict`.

### Important Issues (Medium Impact)

4. **Add tests for untested modules**: `gui/` (5,200 lines), `cli/` (800 lines), `noise/` (400 lines), `fields/` (1,200 lines) have zero test coverage. At minimum, add smoke tests and unit tests for `noise/` and `fields/` which are purely computational.

5. **Expand I/O test coverage**: Only HDF5 and GWF readers have tests. Add tests for ASCII, DTTXML, NDS2, and CSV readers.

6. **Add `__all__` to remaining 17 packages**: Especially `io/`, `time/`, `segments/`, and `table/` to control the public API surface.

7. **Audit dead code**: Verify whether `estimate_coupling`, `estimate_response_function`, `gps_to_unix`, `unix_to_gps`, `read_win`, `to_torch_dataset`, and `to_torch_dataloader` are used by external consumers. If not, deprecate or remove them.

### Minor Issues (Low Impact)

8. **Harmonize `crop()` parameter names**: Use consistent parameter names (`start`/`end` or `t0`/`t1`) and mutation semantics (`copy` vs `inplace`) across `TimeSeries` and `Spectrogram`.

9. **Standardize comment language**: Choose one language (English) for all code comments in `fitting/core.py`.

10. **Consider lighter alternatives to astropy**: If only units and constants are used, evaluate `pint` as a lighter dependency.

---

*This report was generated by automated static analysis of the gwexpy repository.*
