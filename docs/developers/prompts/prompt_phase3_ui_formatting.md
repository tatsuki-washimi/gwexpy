# Prompt for GPT-5.1-Codex-Max: Phase 3 (UI, Formatting, Visualization)

**Context**:
We are in the final stage of hardening `gwexpy`.

* **Phase 0 (Exceptions)** and **Phase 2 (Math/Algo)** are COMPLETE.
* **Phase 1 (Core)** is COMPLETE. `gwexpy.numerics` module is LIVE and available for import.
* **Current State**: The math is now correct/safe for $10^{-21}$ inputs, BUT the user interface, logs, and plots still treat these values as zero or display "0.00".

**Your Mission**:
Fix the "Presentation Layer" to support High Dynamic Range (HDR) data. You will touch many files; prioritize precision and readability.

---

### Task 1: Semantic Formatting Fixes (The "0.00" Bug)

**Target**: Global Codebase (Scan for `f"{...:.2f}"`, `%.2f`, `.format`)
**Instructions**:

1. **Identify**: Find string formatting that truncates small floats.
2. **Context Check**:
    * **IS A GW VALUE?** (strain, psd, coherence, freq): Change to Scientific Notation (`.2e` or `.3e`) or High Precision (`.3g`).
    * **IS A UI LABEL?** (progress %, index, version): **KEEP AS IS**. Do not change `Version 1.0` to `Version 1.00e+00`.
3. **Action**:
    * `f"{val:.2f}"` -> `f"{val:.2e}"` (for physical values).

### Task 2: GUI & Plotting Logic

**Target**: `gwexpy/gui/ui/`, `gwexpy/types/hht_spectrogram.py`
**Instructions**:

1. **Log Plotting (Fixing the Flatline)**:
    * Locate: `np.log10(x + 1e-20)` or similar arbitrary offsets.
    * **Action**: Use `gwexpy.numerics.scaling.safe_log_scale` (if avail) or implement local dynamic logic:

        ```python
        # Dynamic Floor Logic
        floor = np.nanmax(data) * 1e-15 if np.any(data) else 1e-50
        y = 10 * np.log10(np.maximum(data, floor))
        ```

2. **Spectrogram Visuals**:
    * `vmin`/`vmax` must be dynamic. Remove `vmin=1e-10`.
    * Use percentile-based auto-ranging for defaults (e.g. 1st to 99th percentile of positive data).

### Task 3: Boolean Logic Hardening

**Target**: Implicit checks in UI logic.
**Instructions**:

1. **Find**: `if data:` or `if not signal:` where `data` is a numpy array or potentially small float.
2. **Fix**:
    * Arrays: `if data.size > 0:` or `if np.any(data):`.
    * Floats: `if value is not None:` (Explicit is better than implicit for 0.0 vs None).

---

**Execution Guidelines**:

* **Import Strategy**: You should use `from gwexpy.numerics import ...` where helpful, but for UI formatting, standard string manipulation is often sufficient.
* **Verification**: Ensure that `0.000000000000000000001` is displayed as `'1.00e-21'`, not `'0.00'`.
* **Safety**: If you are unsure if a variable is a physical quantity or a generic index, **err on the side of caution** (leave it or use `.3g` which handles both reasonably).

**Source Inventory**:
Refer to `docs/developers/analysis/step1_2_summary.md` for the list of formatting risks identified in the audit.
