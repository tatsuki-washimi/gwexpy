# Phase 0: Exception Swallowing Analysis

**Generated:** 2026-02-03
**Source:** `scripts/audit_numerical_risks.py` (Deep Scan)

## 1. Overview

The AST Deep Scan identified **17 specific locations** where exceptions are broadly caught (`except Exception:` or `except:`) and potentially suppressed (`pass` or minimal action). This pattern hides critical numerical failures (OverflowError, ZeroDivisionError) and logic errors, making "silent failure" the default behavior for low-amplitude GW analysis.

## 2. Identified Locations (Inventory)

| File | Line | Context | Risk Level |
| :--- | :--- | :--- | :--- |
| `gwexpy/__init__.py` | 22 | Module initialization | **High** (Hidden import errors) |
| `gwexpy/frequencyseries/collections.py` | 216, 229 | Collection accessors | **Critical** (Hides missing data/math errors in loops) |
| `gwexpy/frequencyseries/collections.py` | 729, 741 | Batch processing | **Critical** |
| `gwexpy/spectrogram/collections.py` | 84, 100 | Spectrogram dict access | **Critical** |
| `gwexpy/spectrogram/collections.py` | 428, 441 | Batch processing | **Critical** |
| `gwexpy/timeseries/collections.py` | 76, 89 | TimeSeries dict access | **Critical** |
| `gwexpy/timeseries/collections.py` | 1957, 1969 | Batch processing | **Critical** |
| `gwexpy/io/dttxml_common.py` | 280 | XML Parsing Fallback | **Med** (Hides parsing logic errors) |
| `gwexpy/timeseries/io/gbd.py` | 237, 315 | Big Data Loader | **Med** (Hides IO corruption) |
| `gwexpy/plot/plot.py` | 910 | Plotting Renderer | **High** (Hides rendering crashes, results in blank plots) |

## 3. Pattern Analysis & Remediation Strategy

### Pattern A: Dictionary/Collection Accessors

**Context**: `collections.py` often wraps dict access to return `None` or skip missing keys.
**Risk**: If a math operation inside the retrieval triggers `ValueError`, it is treated as "Missing Key".
**Fix**:

* Change `except Exception:` to `except KeyError:`.
* Let `ValueError`, `TypeError`, `FloatingPointError` propagate.

### Pattern B: Batch Processing Loops

**Context**: Processing a list of channels where one failure shouldn't stop others.
**Risk**: A bug in the processing logic (e.g. `1/0`) is swallowed.
**Fix**:

* Keep the try/except, BUT:
* **Must log the error**: `logger.warning(f"Failed to process {channel}: {e}", exc_info=True)`
* Do NOT use `pass`.

### Pattern C: IO Fallbacks (XML/GBD)

**Context**: attempting multiple parsing strategies.
**Risk**: Valid files failing due to code bugs are treated as "Invalid Format".
**Fix**:

* Catch specific `struct.error` or `ExpatError`.
* Log unexpected errors as Warnings.

### Pattern D: Plotting Safety

**Context**: Preventing GUI crash on bad data.
**Fix**:

* **Never swallow**. Display a "Render Error" placeholder in the GUI or log specifically.
* Swallowing here causes "The Blank Screen of Death" where users think data is zero.

## 4. Implementation Checklist (Phase 0)

* [ ] **Step 1**: Grep all `except Exception` and `except:` in `gwexpy/`.
* [ ] **Step 2**: For each location, identify the *intended* exception (e.g. `KeyError`).
* [ ] **Step 3**: Replace generic handler with specific handler + Logging.
* [ ] **Step 4**: Verify `test_phase0_exceptions.py` (to be created) ensures `1/0` is NOT caught.
