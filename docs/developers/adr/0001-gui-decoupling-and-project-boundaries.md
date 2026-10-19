# ADR 001: Decoupling of GUI Layer and Project Boundaries

## Status

Accepted (2026-10-19, Release v0.2.6 preparation, Issue #645)

## Context

GWexpy originally included an in-tree graphical user interface (`gwexpy/gui/`) designed as a PyQt5/pyqtgraph-based diagnostic tool mimicking and extending LIGO's DTT (diaggui). However, bundling interactive GUI components with a scientific data analysis library introduced several architectural challenges:
1. **Dependency footprint**: GUI frameworks (PyQt5, pyqtgraph, qtpy, pyautogui, sounddevice) add substantial weight, complex display server / X11 dependencies, and cross-platform installation issues that do not apply to core headless scientific pipelines.
2. **Release cycles and stability**: Interactive GUI features require distinct UI lifecycle testing, visual stability validation, and release pacing compared to core scientific numerical routines and GWpy parity guarantees.
3. **Packaging confusion**: The in-tree GUI was never included in the published PyPI wheel/sdist distributions, creating discrepancy between repository source checkouts and installed package environments.

## Decision

1. **GWexpy is strictly a headless scientific analysis library**:
   - Focuses on core physical data models (`ScalarField`, `VectorField`, `TimeSeries`, `FrequencySeries`, `Spectrogram`), digital signal processing, GW detector data I/O (GWF, HDF5, NDS2, DiagGUI XML, etc.), and programmatic matplotlib visualization (`gwexpy.plot`).
   - Does NOT contain desktop GUI applications, PyQt/PySide event loops, or desktop window managers.

2. **Standalone Companion Projects**:
   - **`pyaggui`**: Dedicated, independent repository ([https://github.com/tatsuki-washimi/pyaggui](https://github.com/tatsuki-washimi/pyaggui)) maintaining the interactive diagnostic GUI application.
   - **`GWexpy Studio`**: Independent desktop analytical studio with its own lifecycle and product ownership.
   - Per maintainer decision on Issue #645 (comment 6258374097), `pyaggui` and `GWexpy Studio` maintain their own development, issues, and release schedules independently from GWexpy core maintenance releases.

3. **Repository Boundaries and Prevention**:
   - `gwexpy/gui/` and `tests/gui/` are permanently removed from GWexpy.
   - Repository hygiene checks and test fixtures enforce that no GUI source directories, active GUI references, or GUI-only markers (`@pytest.mark.gui`, `@pytest.mark.pyautogui`) enter the GWexpy repository.
   - Core scientific plotting (`gwexpy.plot`), CLI tools, and DiagGUI XML reading (`format="xml.diaggui"`) remain first-class core features of GWexpy.
