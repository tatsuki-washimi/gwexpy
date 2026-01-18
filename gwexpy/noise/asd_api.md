# ASD API Reference

The `gwexpy.noise.asd` module provides functions for generating Amplitude Spectral Density (ASD).
All functions return `FrequencySeries` objects.

> [!NOTE]
> `gwexpy.noise` is separated into the `.asd` submodule for generating ASD and the `.wave` submodule for generating time-series waveforms.

---

## from_pygwinc

Retrieves ASD from pyGWINC detector noise models.

### Signature

```python
from_pygwinc(
    model: str,
    frequencies: np.ndarray | None = None,
    quantity: Literal["strain", "darm", "displacement"] = "strain",
    **kwargs
) -> FrequencySeries
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | pyGWINC model name (e.g., "aLIGO", "Aplus", "Voyager") |
| `frequencies` | array | None | Frequency array [Hz]. If None, generated from fmin/fmax/df |
| `quantity` | str | "strain" | Physical quantity (see below) |
| `fmin` | float | 10.0 | Minimum frequency [Hz] |
| `fmax` | float | 4000.0 | Maximum frequency [Hz] |
| `df` | float | 1.0 | Frequency step [Hz] |

### Allowed quantity values

| quantity | Unit | Description |
|----------|------|-------------|
| `"strain"` | 1/√Hz | Strain ASD |
| `"darm"` | m/√Hz | Differential Arm Length ASD |
| `"displacement"` | m/√Hz | Deprecated alias for `"darm"` |

### Exceptions

- `ValueError`: quantity is not an allowed value
- `ValueError`: `quantity="darm"` but arm length cannot be retrieved
- `ValueError`: `fmin >= fmax`
- `ImportError`: pygwinc is not installed

### Conversion Rule

`darm = strain × L` (where L is the IFO arm length `ifo.Infrastructure.Length`)

### Example

```python
from gwexpy.noise.asd import from_pygwinc

# Strain ASD
strain_asd = from_pygwinc("aLIGO", quantity="strain")
# unit: 1 / sqrt(Hz)

# DARM ASD
darm_asd = from_pygwinc("aLIGO", quantity="darm")
# unit: m / sqrt(Hz)
```

---

## from_obspy

Retrieves ASD from ObsPy seismic/infrasound noise models.

### Signature

```python
from_obspy(
    model: str,
    frequencies: np.ndarray | None = None,
    quantity: Literal["displacement", "velocity", "acceleration"] = "acceleration",
    **kwargs
) -> FrequencySeries
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model name: "NHNM", "NLNM", "IDCH", "IDCL" |
| `frequencies` | array | None | Frequency array [Hz]. If None, uses the model's original frequencies |
| `quantity` | str | "acceleration" | Physical quantity (see below) |

### Allowed quantity values (seismic models only)

| quantity | Unit | Description |
|----------|------|-------------|
| `"acceleration"` | m/(s²·√Hz) | Acceleration ASD |
| `"velocity"` | m/(s·√Hz) | Velocity ASD |
| `"displacement"` | m/√Hz | Displacement ASD |

> **⚠️ Important**: `quantity="strain"` is **NOT supported**. Seismic noise models do not have the concept of strain. Use `from_pygwinc` for strain ASD.

### Exceptions

- `ValueError`: `quantity="strain"` (not supported)
- `ValueError`: quantity is not an allowed value
- `ValueError`: model is unknown
- `ImportError`: obspy is not installed

### Conversion Rules (from acceleration)

- `velocity = acceleration / (2πf)`
- `displacement = acceleration / (2πf)²`
- **f=0 returns NaN** (not infinity)

### Example

```python
from gwexpy.noise.asd import from_obspy

# Acceleration ASD (default)
acc_asd = from_obspy("NLNM")
# unit: m / (s² · sqrt(Hz))

# Displacement ASD
disp_asd = from_obspy("NLNM", quantity="displacement")
# unit: m / sqrt(Hz)

# strain raises ValueError
# from_obspy("NLNM", quantity="strain")  # NG!
```

---

## quantity Compatibility Summary

### pyGWINC (`from_pygwinc`)

| quantity | Unit | Notes |
|----------|------|-------|
| strain | 1/√Hz | Default |
| darm | m/√Hz | = strain × L |
| displacement | m/√Hz | Alias for darm (deprecated) |
| velocity | - | **ValueError** |
| acceleration | - | **ValueError** |

### ObsPy (`from_obspy`)

| quantity | Unit | Notes |
|----------|------|-------|
| acceleration | m/(s²·√Hz) | Default |
| velocity | m/(s·√Hz) | = acc / (2πf) |
| displacement | m/√Hz | = acc / (2πf)² |
| strain | - | **ValueError** |
