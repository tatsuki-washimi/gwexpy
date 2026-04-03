# GWexpy Optional Dependency Rules

This rule ensures that `gwexpy` remains lightweight and installable in diverse environments (LIGO/KAGRA clusters, local macOS, etc.) by correctly managing optional dependencies.

## Standard Import Guard Pattern

Always use the following pattern when importing from `optional-dependencies`:

```python
try:
    import PyQt6
    HAS_PYQT6 = True
except (ImportError, ModuleNotFoundError):
    HAS_PYQT6 = False

def my_gui_function():
    if not HAS_PYQT6:
        raise ImportError(
            "PyQt6 is required for this feature. "
            "Please install it with: pip install gwexpy[gui]"
        )
    # ... use PyQt6 here
```

## Forbidden Patterns
- **Direct top-level import**: Never use `import optional_lib` without a `try...except` at the module level.
- **Generic error message**: Always include the `[extra]` tag (e.g., `[gui]`, `[nds]`, `[cvmfs]`) in the `ImportError` message.
- **Silent failure**: Don't just `pass` and leave the variable as `None` if it will cause an obscure `AttributeError` later. Use boolean flags (`HAS_...`).

## Metadata Consistency
- All optional dependencies must be listed in the `[project.optional-dependencies]` section of `pyproject.toml`.
- New extra names should be added to the install documentation that actually exists in this repository (for example `README.md` and relevant files under `docs/`).

## Review Triggers
- Modification of `pyproject.toml`.
- Addition of files to `gwexpy/interop/` or `gwexpy/gui/`.
- Use of specialized libraries like `h5py`, `ndscope`, `ROOT`, `torch`, `xarray`.

## Testing Requirement
- For any new optional feature, at least one test must verify the "graceful failure" when the dependency is missing (if possible using mock).
- Standard CI tests should run both with and without these extras.
