---
name: phase0_exception_sweep
description: Use when auditing or fixing broad exception handling, silent failures, or suspicious try/except fallback logic in gwexpy.
---

# Phase 0 Exception Sweep

Use this skill to systematically identify and fix "Silent Failures" and "Broad Exceptions" based on the Phase 0 auditing methodology.

## Overview
Silent failures (catching exceptions without logging or specific handling) are a major source of bugs in GW data processing. This skill helps you clean them up.

## When to Use
- You see `except Exception:` or `except:`.
- You see a `try...except` block with `pass` or only a print statement.
- You are refactoring core logic in `gwexpy/fields/` or `gwexpy/io/`.
- You are implementing a new collection accessor.

## Core Workflow
1. **Inventory**: Use `grep` to find all `except` patterns in the target file.
2. **Context**: Read the surrounding code to understand what the block is trying to protect.
3. **Categorize**:
   - **Remove**: If the protection is unnecessary or hides bugs.
   - **Narrow**: If specific errors (e.g., `KeyError`, `FileNotFoundError`) are expected.
   - **Log & Continue**: If the error is benign but should be recorded (use `logger.exception`).
4. **Verification**: Run `pytest` and ensure no new crashes occur under normal conditions.

## Decision Rules
- **Rule 1**: Never use bare `except:`.
- **Rule 2**: If you must use `except Exception:`, you MUST call `logger.exception()` or `warnings.warn()`.
- **Rule 3**: Differentiate between "expected misses" (e.g., missing metadata) and "software bugs".

## Verification
- `grep -E "except (Exception|:)" <file>`
- Ensure `logger.exception` is present in all remaining broad catches.

## Common Mistakes
- Confusing `KeyError` with numerical calculation errors.
- Assuming GUI code should always suppress errors (it should log them to a status bar or console).
