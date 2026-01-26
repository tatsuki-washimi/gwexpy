---
name: fix_legacy_python_compatibility
description: Python 3.9などのレガシー環境での互換性問題を修正する（特に実行時に評価されるユニオン型の修正など）
---

# Fix Legacy Python Compatibility

This skill helps resolve compatibility issues when running code on Python versions older than 3.10, even when `from __future__ import annotations` is present.

## Instructions

1. **Identify Runtime Evaluation Contexts**:
   - `from __future__ import annotations` only applies to annotations (function signatures and variable type hints).
   - The following contexts DO evaluate types at runtime and will fail with `TypeError` if using `|` (pipe operator) or other 3.10+ features in Python < 3.10:
     - `TypeAlias` assignments: `MyType: TypeAlias = A | B`
     - Global/Class variable assignments: `MyType = A | B`
     - Arguments to functions/classes: `cast(A | B, val)`, `MyGeneric[A | B]()`, etc.
   - *Note*: `isinstance(val, A | B)` is invalid in < 3.10. Use `isinstance(val, (A, B))`.

2. **Refactor to Union**:
   - Import `Union` (and `Literal`, `Optional` if needed) from `typing`.
   - Replace `A | B` with `Union[A, B]`.
   - Ensure `from __future__ import annotations` is present at the top of the file to allow `|` in actual annotations.

3. **Handle TypeAlias and Modern Typing**:
   - Support for `TypeAlias` in Python < 3.10 requires `typing-extensions`.
   - Use the following pattern for robust imports:

     ```python
     try:
         from typing import TypeAlias
     except ImportError:
         from typing_extensions import TypeAlias
     ```

   - Ensure `typing-extensions` is listed in the project dependencies (e.g., `pyproject.toml`).

4. **Verify**:
   - Run an import check to catch `TypeError` at module level:

     ```bash
     python -c "from your_package.module import your_symbol; print('Import OK')"
     ```

   - Run tests focusing on the modified modules.

## Best Practices

- Use `multi_replace_file_content` to add `Union` to the `typing` import block and update type definitions simultaneously.
- When string-quoting a TypeAlias for lazy evaluation (e.g., `MyType: TypeAlias = "A | B"`), ensure it's still a valid TypeAlias in the eyes of static analyzers. Prefer `Union` for runtime-evaluated assignments.
