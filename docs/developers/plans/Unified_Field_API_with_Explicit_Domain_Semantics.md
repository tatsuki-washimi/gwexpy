# Field API Refactoring and Domain Specification Plan

## Title
**Unified Field API with Explicit Domain Semantics  
(ScalarField / VectorField / TensorField)**

---

## 1. Background and Motivation

The current implementation uses a `ScalarField` class located under `gwexpy.types`.  
However:

- `types/` is intended for low-level or abstract infrastructure.
- Field objects are primary, user-facing scientific objects.
- Future extensions may include vector and tensor fields.
- 2D and 3D cases are already treated as degenerate 4D arrays (axis length = 1).

Therefore, we will:
- Remove the explicit “4D” from user-facing class names.
- Introduce `ScalarField`, `VectorField`, and `TensorField`.
- Relocate field-related APIs into a dedicated `gwexpy.fields` namespace.
- Explicitly formalize **domain semantics** for time/frequency and space/real–Fourier axes.

Backward compatibility is **not required**, as the package is not yet publicly used.

---

## 2. High-Level Design Decisions (Fixed)

### 2.1 Public API Namespace
- **Canonical import path:** `gwexpy.fields`
- Example:
  ```python
  from gwexpy.fields import ScalarField
````

### 2.2 Field Classes

* `ScalarField`
* `VectorField`
* `TensorField`

These are user-facing scientific objects.

### 2.3 Shape Convention (Invariant)

#### ScalarField

```
(axis0, x, y, z)
```

#### VectorField / TensorField

```
(axis0, x, y, z, c)
```

* `axis0`: time or frequency
* `x, y, z`: spatial axes
* `c`: component axis (always last)

2D / 3D data are represented as **degenerate cases** with axis length = 1.
The number of axes is always preserved.

---

## 3. Domain Specification (Core Concept)

### 3.1 Axis Domains

Each coordinate axis carries an explicit domain state.

#### Axis 0 (temporal axis)

```
axis0_domain ∈ {"time", "frequency"}
```

* `"time"` → unit: seconds
* `"frequency"` → unit: Hz

#### Spatial axes (x, y, z)

```
space_domain[axis] ∈ {"position", "wavenumber"}
```

* `"position"` → unit: length (e.g. m)
* `"wavenumber"` → unit: 1/length (rad/m preferred)

This allows unambiguous representation of:

* `(t, x, y, z)`
* `(f, x, y, z)`
* `(t, kx, ky, z)`
* `(f, kx, ky, kz)`

### 3.2 Partial Fourier Transforms

* Fourier transforms may be applied to **any subset of axes**.
* Only transformed axes change domain:

  ```
  position → wavenumber
  time     → frequency
  ```
* Non-transformed axes remain unchanged.

### 3.3 Wavenumber Convention

* The canonical spatial Fourier variable is **k**.
* Relationship:

  ```
  k = 2π / λ
  ```
* `λ` (wavelength) is provided only as a **derived view**, not a primary axis.
* Spatial FFTs are **signed, two-sided**.

### 3.4 Axis Length = 1 (Slicing Semantics)

* Slicing does **not remove axes**.
* Sliced axes retain:

  * domain
  * coordinate value (Quantity of length 1)
* This preserves information such as:

  * “z = 0 m slice”
  * “kx = 0 mode”

---

## 4. Directory Structure (Target)

```
gwexpy/
├─ fields/
│  ├─ __init__.py
│  ├─ base.py          # shared base class / mixin
│  ├─ scalar.py        # ScalarField
│  ├─ vector.py        # VectorField
│  ├─ tensor.py        # TensorField
│  ├─ collections.py  # FieldList, FieldDict
│  ├─ signal.py        # PSD, xcorr, coherence (ScalarField only initially)
│  └─ plot.py          # plotting helpers
│
├─ types/
│  ├─ array4d.py       # low-level 4D array container
│  └─ ...
```

---

## 5. Implementation Phases

### Phase 1: Structural Refactoring

* Move Field-related code from `types/` to `fields/`.
* Rename:

  * `ScalarField` → `ScalarField`
  * `ScalarFieldList/Dict` → `FieldList/FieldDict`
* Update all internal imports.
* No compatibility aliases required.

### Phase 2: Base Class and Domain Metadata

* Introduce a shared base class (e.g. `FieldBase`).
* Add required metadata:

  * `axis0_domain`
  * `space_domain` (per axis)
* Enforce consistency:

  * domain ↔ coordinate unit checks
  * domain updates during FFT operations

### Phase 3: FFT and Domain Propagation

* Time FFT:

  * `time → frequency`
* Spatial FFT:

  * `position → wavenumber` per axis
* Partial FFT support.
* FFT ordering (e.g. `fftshift`) handled separately from domain.

### Phase 4: Signal Processing (ScalarField only)

* Implement / migrate:

  * PSD (Welch)
  * Cross-correlation
  * Time-delay maps
  * Coherence maps
* Vector/Tensor fields must be reduced to ScalarField first:

  * `component()`
  * `magnitude()`
  * other invariants (future)

### Phase 5: Plotting and Documentation

* Plotting functions operate on ScalarField.
* Domain-aware axis labeling.
* Documentation:

  * Field invariants
  * Domain examples
  * Partial FFT examples

---

## 6. Acceptance Criteria (Definition of Done)

* `from gwexpy.fields import ScalarField` works.
* ScalarField always has exactly 4 coordinate axes.
* Vector/Tensor fields have a final component axis.
* Partial FFT updates only the transformed axes’ domains.
* Domain and coordinate units are always consistent.
* 2D/3D cases are handled via axis length = 1.
* Tests exist for:

  * domain propagation
  * slicing behavior
  * partial FFT
  * unit consistency

---

## 7. Explicit Non-Goals (For Now)

* No backward compatibility with old `ScalarField` imports.
* No direct signal processing on Vector/Tensor fields.
* No non-Cartesian coordinate systems.
* No I/O or serialization changes.

---

## 8. Notes for AI Coding Tools

* This refactor is intentionally **breaking**.
* Favor clarity and explicit metadata over clever inference.
* Domain handling must be deterministic and testable.
* Keep component handling orthogonal to domain handling.