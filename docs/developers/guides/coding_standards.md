# GWExPy Coding Standards & Developer Guidelines (v0.1)

**Last Updated**: 2026-01-27

This document outlines the coding standards, type safety practices, and testing guidelines for the `gwexpy` project. All contributions must adhere to these rules to ensure maintainability, robustness, and compatibility.

---

## 1. Python Version & Compatibility

- **Target Versions**: Python 3.9, 3.10, 3.11, 3.12.
- **Syntax Rule**: Code must be syntactically compatible with Python 3.9.

### 1.1 `from __future__ import annotations` (MUST)

All Python files (excluding empty `__init__.py`) **MUST** start with:

```python
from __future__ import annotations
```

**Why?**

- Enables forward references for type hints.
- Allows modern type syntax (`list[int]`, `dict[str, Any]`) to be parsed without runtime errors even in older Python versions (deferred evaluation).
- Resolves circular import issues in type hints.

### 1.2 Union Type Syntax (CRITICAL)

While `from __future__ import annotations` allows `int | None` in _annotations_, it **DOES NOT** allow `int | None` in runtime contexts (e.g., `isinstance` checks, `cast`).

**Rules**:

1.  **In Type Hints (Signatures/Variables)**: Preferred `TypeA | TypeB` (thanks to `annotations` future import).
    ```python
    def func(x: int | None) -> str | None: ...
    ```
2.  **In Runtime Expressions (isinstance, cast, TypeVar)**: **MUST USE** `typing.Union` or `typing.Optional` if supporting Python 3.9. The `|` operator for types was introduced in Python 3.10.

    ```python
    # BAD (Fails in Py3.9)
    if isinstance(x, (int | float)): ...

    # GOOD
    from typing import Union
    if isinstance(x, (int, float)): ...
    ```

---

## 2. Type Safety & MyPy

`gwexpy` strives for 100% type checking coverage.

### 2.1 Mixin Classes (Protocol Pattern)

Mixin classes often rely on attributes (`self.value`, `self.unit`) assumed to exist in the host class. To avoid MyPy errors (`has no attribute ...`), use **Protocols**.

**Pattern**:

```python
from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class HasSeriesData(Protocol):
    value: Any
    unit: Any
    def copy(self) -> Any: ...

class SignalAnalysisMixin:
    def smooth(self: HasSeriesData, width: int) -> Any:
        # MyPy knows 'self' has .value and .unit
        return self.value
```

### 2.2 `super()` in Mixins

When calling `super()` in a Mixin, MyPy often loses context. Use `cast` to hint the expected parent interface.

```python
from typing import cast

class MyMixin:
    def method(self):
        parent = cast(SomeProtocol, super())
        return parent.method()
```

### 2.3 Ignoring Errors

- **Prefer Fixes**: Try to fix the type error first.
- **`# type: ignore`**: Use sparingly. Always include the specific error code (e.g., `# type: ignore[override]`).
- **`pyproject.toml` Exclusions**: Do not add new modules to the `exclude` list. The goal is to reduce this list to zero.

---

## 3. Exception Handling

### 3.1 No Bare `except Exception`

**Forbidden**:

```python
try:
    do_something()
except Exception:  # BAD: Hides typos, interrupts, and logic errors.
    pass
```

**Allowed**:

1.  Catch specific exceptions (`ValueError`, `RuntimeError`).
2.  If a broad catch is absolutely necessary (e.g., top-level thread safety), **MUST** log the exception.

```python
import logging
logger = logging.getLogger(__name__)

try:
    do_something()
except Exception:
    logger.exception("Unexpected error in thread")
    # Clean up if needed
```

### 3.2 Warnings

Use `warnings.warn` for deprecations or non-fatal data issues (e.g., invalid metadata that can be ignored), rather than `print` or `logger.warning` if it's library usage feedback.

---

## 4. Testing Guidelines

### 4.1 Pytest & Coverage

- **Goal**: 90% coverage for new modules.
- **Command**: `pytest --cov=gwexpy tests/`

### 4.2 GUI Testing (Qt)

- **Use `qtbot`**: Inherit from `pytest-qt` fixtures.
- **`waitExposed`**: NEVER use `waitForWindowShown` (deprecated). Use `qtbot.waitExposed(window)`.
- **Resource Cleanup**: Ensure widgets are closed at the end of tests (often handled by `qtbot.addWidget(window)`).
- **Headless**: Tests must pass in Xvfb (CI environment). Avoid absolute screen coordinates if possible.

---

## 5. Documentation Style

- **Docstrings**: NumPy style is preferred.

  ```python
  """
  Short summary.

  Parameters
  ----------
  arg1 : type
      Description.

  Returns
  -------
  type
      Description.
  """
  ```

- **Type Hints in Docstrings**: Redundant if type annotations are present in the signature. Focusing on _meaning_ in docstrings is better.

---
