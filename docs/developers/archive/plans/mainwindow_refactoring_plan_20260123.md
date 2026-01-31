# Implementation Plan: MainWindow logic refactoring and modularization

**Date**: 2026-01-23
**Status**: Draft
**Reference**: `docs/developers/plans/repo_improvement_report_20260123.md`

## 1. Objectives & Goals
Refactor the monolithic `MainWindow.update_graphs` method in `gwexpy/gui/ui/main_window.py` to improve maintainability and extensibility.
*   **Decoupling**: Separate rendering and excitation logic into dedicated classes.
*   **Abstraction**: Enable support for diverse result types (Series, Spectrogram, SweptSine) without modifying the main loop.
*   **DTT Compatibility**: Introduce selectable normalization logic (Standard vs. DTT-compatible).
*   **Maintainability**: Reduce the size of `main_window.py` and improve readability.

## 2. Detailed Roadmap

### Phase 1: Shared Signal Logic
*   **Goal**: Implement normalization and ENBW correction logic.
*   **Tasks**:
    *   Create `gwexpy/signal/normalization.py`.
    *   Implement `get_normalization_factor(window_name, n_fft, fs, mode='standard')`.
    *   Implement `detect_dtt_normalization_ratio(window)` based on DTT analysis findings.

### Phase 2: Excitation Manager Modularization
*   **Goal**: Extract signal generation and injection logic.
*   **Tasks**:
    *   Create `gwexpy/gui/ui/excitation_manager.py`.
    *   Define `ExcitationManager` class to handle `GeneratorParams` creation and signal injection into `data_map`.
    *   Migrate logic from lines ~741-800 of `main_window.py`.

### Phase 3: Plot Renderer Modularization
*   **Goal**: Extract rendering logic for all plot types.
*   **Tasks**:
    *   Create `gwexpy/gui/ui/plot_renderer.py`.
    *   Implement `PlotRenderer` class with methods:
        *   `render_results(info_root, results)`: High-level entry point.
        *   `render_trace(trace_item, result, config)`: Abstract dispatcher.
        *   `_draw_series(...)`, `_draw_spectrogram(...)`: Low-level rendering.
    *   Handle unit conversions (dB, Phase, Magnitude) and coordinate transformations (Log-Y).
    *   Migrate logic from lines ~883-1143 of `main_window.py`.

### Phase 4: MainWindow Integration
*   **Goal**: Refactor the main loop to use new managers.
*   **Tasks**:
    *   Initialize `ExcitationManager` and `PlotRenderer` in `MainWindow.__init__`.
    *   Rewrite `MainWindow.update_graphs` into a pipeline of private calls:
        *   `_collect_data_map()`
        *   `_check_stop_conditions()`
        *   `self.excitation_manager.inject(...)`
        *   `self.plot_renderer.render(...)`
    *   Ensure thread-safety and performance (aim for < 20ms execution per cycle).

### Phase 5: Verification
*   **Goal**: Ensure no regressions in UI behavior.
*   **Tasks**:
    *   Run `pytest tests/gui`.
    *   Manual verification of Spectrogram log-scale rendering.
    *   Verify DTT normalization toggle (if UI checkbox added or as a hidden setting).

## 3. Recommended Models & Skills
*   **Model**: `Claude Opus 4.5 (Thinking)` or `Gemini 3 Pro (High)` for architectural integrity.
*   **Skills**: `lint`, `test_gui`, `visualize_fields` (for reference).
*   **Estimated Effort**:
    *   Total Time: 60 - 90 minutes (AI wall-clock time).
    *   Quota Consumption: **High** (Large file edits and multiple new files).

## 4. Risks & Concerns
*   **Performance**: Inter-object communication overhead (should be negligible in Python).
*   **State Sync**: Ensuring `PlotRenderer` has up-to-date config from `MainWindow` widgets.
*   **Unit Consistency**: Ensuring normalization factors are applied exactly once.
