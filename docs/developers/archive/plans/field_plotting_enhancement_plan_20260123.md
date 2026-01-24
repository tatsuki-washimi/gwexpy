# Implementation Plan: Enhanced Plotting for Field Classes

Implementation of dimension-aware 2D/3D visualization for `ScalarField`, `VectorField`, and `TensorField`. This includes a new `FieldPlot` API and instance-level plotting methods.

## 1. Core Visualization Engine (2D Slicing)

- **Objective**: Provide a robust way to extract a 2D slice from a 4D field by providing coordinate values or indices.
- **Location**: `gwexpy/fields/base.py` (or a helper in `gwexpy/plot/utils.py`).
- **Feature**:
    - `field.get_slice(x_axis='x', y_axis='y', **fixed_coords)`: Returns a 2D slice.
    - Automatic handling of coordinate-to-index mapping using the axis metadata.

## 2. FieldPlot Class (API B)

- **Objective**: A dedicated class to manage spatial maps, handles overlays of multiple fields.
- **Location**: `gwexpy/plot/field.py`.
- **Key Methods**:
    - `__init__(field, ...)`: Sets up the default axes and projection.
    - `add_scalar(field, cmap='viridis', mode='pcolormesh', **kwargs)`: Renders a scalar field slice.
    - `add_vector(field, mode='quiver', color='black', **kwargs)`: Renders a vector field slice.
        - Supported modes: `quiver`, `streamline`, `magnitude_contour`.
    - `add_tensor(field, ...)`: TBD (likely shows ellipses or specific components).

## 3. Instance-level Plotting (API C)

- **Objective**: Quick-look methods directly on field objects.
- **ScalarField**:
    - `plot(**slice_params)`: Simple 2D map.
- **VectorField**:
    - `plot_magnitude(**slice_params)`
    - `quiver(**slice_params)`
    - `streamline(**slice_params)`
- **TensorField**:
    - `plot_components(**slice_params)`: $3 \times 3$ grid of component maps.

## 4. Animation and Multi-facet Support

- **Objective**: Visualize 4D dynamics.
- **Features**:
    - `animate(loop_axis='t', **fixed_coords)`: Uses `matplotlib.animation.FuncAnimation`.
    - `plot(col='t', col_wrap=4)`: Faceted plot similar to Xarray/Seaborn.

## 5. Implementation Roadmap

### Phase 1: Foundation & Scalar Plotting
- [ ] Implement slice extraction logic in `FieldBase`.
- [ ] Create `gwexpy/plot/field.py` with basic `FieldPlot`.
- [ ] Implement `ScalarField.plot()`.

### Phase 2: Vector Visualization
- [ ] Add `VectorField.quiver()` and `VectorField.streamline()`.
- [ ] Update `FieldPlot` to support `add_vector`.

### Phase 3: Advanced Features
- [ ] Implement `TensorField.plot_components()`.
- [ ] Implement `field.animate()`.
- [ ] Add unit tests and tutorial updates.

## 6. Verification Plan
- **Verification Script**: Create `tests/plot/test_field_plots.py` to verify:
    - Slice extraction accuracy.
    - Vector field plotting consistency (arrow directions).
    - Unit preservation in labels.
- **Visual Inspection**: Use a Jupyter Notebook to verify aesthetic quality and interactivity.
