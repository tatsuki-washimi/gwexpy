# Task: Refactor MainWindow and Address Technical Debt

## Objective
Refactor the `gwexpy/gui/ui/main_window.py` class to improve maintainability, specifically focusing on the monolithic `update_graphs` method, and address key technical debt items.

## Context
The `MainWindow` class has grown to 1800+ lines. The `update_graphs` method (~500 lines) is a bottleneck for understanding and modifying the UI logic. Additionally, there are `TODO`s and broad exception handling that need review.

## Steps

1.  **Refactor `MainWindow.update_graphs`**:
    *   Analyze the `update_graphs` method in `gwexpy/gui/ui/main_window.py`.
    *   Identify logical groupings (e.g., preparing data, updating specific plots, handling UI state).
    *   Extract these into helper methods (e.g., `_update_time_series_plots`, `_update_spectral_plots`) or a separate helper class if appropriate.
    *   Ensure no functionality is lost.

2.  **Code Cleanup**:
    *   Review `gwexpy/fitting/core.py` for `TODO` items. Resolve them if straightforward, or document why they remain.
    *   Search for `FIXME` and `XXX` in `gwexpy/gui/ui/main_window.py` and resolve them.

3.  **Safety Check**:
    *   Run GUI tests (`tests/gui/`) to ensure no regressions.
    *   Run `mypy` on the modified files to ensure type correctness is maintained or improved.

## Definition of Done
*   `gwexpy/gui/ui/main_window.py` is significantly cleaner; `update_graphs` is decomposed.
*   Relevant `TODO`/`FIXME` items are addressed.
*   All tests pass.
