# GWexpy GWpy Compatibility Rules

GWexpy is designed to extend `gwpy`. Compatibility with both GWpy 3.x and 4.0 is essential during the transition.

## Banned or Legacy Patterns

- **`nproc=`**: Deprecated in many GWpy 4.0 functions. Use `parallel=` or standard `multiprocessing`.
- **`gwpy.io.mp`**: This module is removed in GWpy 4.0. Migrate to modern `io` or `multiprocessing`.
- **`gwpy.utils.gprint` and `verbose`**: Standardize on `logger` and `warnings`. Avoid `gprint`.
- **Direct Registry Import**: Import from `gwpy.io.registry` instead of legacy paths.

## Preferred Patterns

- **`default_registry`**: Use for all I/O registration to maintain compatibility.
- **`parallel=`**: Use this argument instead of `nproc`.
- **Standard Library `multiprocessing`**: For new implementations, use Python's built-in `multiprocessing` for parallelization.

## Compatibility Strategy

- Use `if version.parse(gwpy.__version__) >= version.parse("4.0.0")` for version-specific logic.
- Wrap new GWpy-dependent parameters in `try...except TypeError` if necessary.

## Required Tests

- **I/O Registration**: Ensure `ScalarField.read/write` works with `gwpy` backends.
- **`TimeSeries.fetch()`**: Verify `parallel` handling.
- **Metadata Inheritance**: Ensure `TimeSeries` to `ScalarField` conversion preserves all `gwpy` attributes.

## Migration Notes

- Document all compatibility-forced changes in `CHANGELOG.md` under a dedicated "Compatibility" section.
- Tag any breaking deviation from `gwpy` with `needs-physics-review`.
