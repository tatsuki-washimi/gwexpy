# GWexpy Exception Handling Rules

This rule is designed to prevent silent failures and ensure robust error tracking across the gwexpy project. It complements `physics.md` and `testing.md`.

## Forbidden Patterns

- **`except Exception:` without logging**: Catching all exceptions without recording the event makes debugging impossible.
- **Bare `except:`**: Equivalent to `except BaseException:`, which catches even system signals (SIGINT). Never use this.
- **`pass` in handlers**: Silently swallowing errors is strictly prohibited unless explicitly justified in comments.
- **Missing `exc_info`**: When using `logger.warning` in an exception block, always include the traceback.

## Allowed Narrow Exceptions

Use the narrowest possible exception type:
- **Collection Access**: `KeyError`, `IndexError`, `AttributeError`.
- **IO Fallback**: `FileNotFoundError`, `OSError`, `json.JSONDecodeError`.
- **GUI/Plotting**: `ImportError`, `RuntimeError` (specific to backend).

## Logging Requirements

- Use `logger.exception("Message")` (level ERROR) in `except` blocks to automatically include tracebacks.
- Use `logger.warning("Message", exc_info=True)` if the error is expected and you want to continue.
- Use `warnings.warn("Message", PhysicalWarning)` for physics-related issues that don't block execution.

## Review Triggers

- Changes in `gwexpy/io/` or `gwexpy/*/collections.py`.
- New `try...except` blocks in core physical logic or GUI fallbacks.

## Examples

### Bad
```python
try:
    data = fetch_gw_data()
except Exception:
    pass
```

### Better
```python
try:
    data = fetch_gw_data()
except ConnectionError:
    logger.exception("Failed to connect to data server.")
    data = None
```
