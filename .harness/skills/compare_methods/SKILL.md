---
name: compare_methods
description: make_notebookの specialized mode。ライブラリ内の類似手法の技術的・物理的差分を notebook 用の比較セクションへ落とし込む時に使う
---

# Document Method Comparison

This skill is kept as a compatibility entry point, but the canonical notebook workflow now lives in `make_notebook`.

## Instructions

1.  **Technical Analysis**:
    *   Examine the source code of both methods to identify implementation differences.
        - **Averaging/Integration**: Does it downsample the data using a sliding window? (e.g., `gwpy.heterodyne`)
        - **Filtering**: Does it use a Low-Pass Filter (LPF) like Butterworth or FIR? (e.g., `gwexpy.lock_in`)
    *   Compare numerical characteristics like time resolution and frequency response (aliasing).

2.  **Physical/Conceptual Context**:
    *   Check if there are differences between engineering definitions and package conventions.
        - Example: `gwpy.heterodyne` effectively performing "Homodyne" detection relative to the carrier frequency.
    *   Identify the target use case for each (e.g., stationary signal vs transient analysis).

3.  **Construct Comparative Sample Code**:
    *   Create a synthetic signal that highlights the differences (e.g., a signal with rapid amplitude/phase changes).
    *   Execute both methods on the same input signal with comparable parameters (e.g., matching the averaging stride with the filter bandwidth).
    *   Visualize results in a single plot for direct comparison.
        - Use `plt.step(..., where='post')` for discrete averaged data.
        - Use `plt.plot(...)` for continuous filtered data.

4.  **Draft Markdown Explanation**:
    *   Use tables to summarize differences (Algorithm, Resolution, Post-Processing, Features).
    *   Provide a clear summary statement on "which method to choose" for specific user goals (e.g., "Use `.lock_in()` for control system transient response").

5.  **Integration**:
    *   Use Python scripts to inject these cells (Markdown and Code) into the target `.ipynb` file, typically appending to the end or inserting into a relevant section.

## Routing Note

- 新しく notebook を作る場合は `make_notebook` を優先する。
- この skill は既存の比較タスク名との互換性維持用に残す。
