# FrequencySeries I/O Registry Backends

> Developer note — audit-only, no behaviour change.
> Tracked by issue #438 (this document) and issue #444 (migration decision).

## Current State

GWexpy exposes two distinct I/O dispatch backends within the `frequencyseries`
package.  These backends are **intentionally separated** at present.

| Class | Read/Write backend | Where formats are registered |
|---|---|---|
| `FrequencySeries` | `gwpy.io.registry.default_registry` | `gwexpy/frequencyseries/io/` and stubs |
| `FrequencySeriesMatrix` | `gwpy.io.registry.default_registry` | `SeriesMatrixIOMixin` fallback |
| `FrequencySeriesBaseDict` / `FrequencySeriesDict` | `astropy.io.registry` (module-level functions) | astropy global registry — but `xml.diaggui` / `dttxml` are handled by an inline fast path *before* the astropy fallback, and are also registered in the gwpy registry (see observation 4) |
| `FrequencySeriesBaseList` / `FrequencySeriesList` | `astropy.io.registry` (module-level functions) | astropy global registry |

### Key observations

1. **`FrequencySeries` and `FrequencySeriesMatrix`** dispatch through
   `gwpy.io.registry.default_registry`.  Formats such as `xml.diaggui` and
   `dttxml` are registered there on import.

2. **Collection containers** (`FrequencySeriesBaseDict`, `FrequencySeriesBaseList`
   and their concrete subclasses) dispatch unknown formats through the module-level
   `astropy.io.registry.read` / `astropy.io.registry.write` functions.
   This means:

   - A format registered **only** in `gwpy.io.registry.default_registry` is
     **not** reachable via `FrequencySeriesDict.read()` / `FrequencySeriesList.read()`.
   - Conversely, a format must be registered in the astropy global registry to be
     reachable from collection read/write.

3. **`FrequencySeriesList` is not registered in `gwpy.io.registry.default_registry`
   at all** (confirmed by `get_formats` returning an empty table).

4. The fast-path dispatch in `FrequencySeriesBaseDict.read` for `xml.diaggui` /
   `dttxml` bypasses the registry entirely — these formats are handled inline
   before the astropy fallback.

## Why This Split Exists

The collection containers (`FrequencySeriesBaseDict`, `FrequencySeriesBaseList`)
pre-date the GWpy-aligned registry migration.  Their `read`/`write` methods
were written against the standard Astropy Unified I/O API rather than the GWpy
`default_registry` wrapper.  The two codepaths diverged organically and have not
yet been unified.

## No Behaviour Change Policy

This split is **preserved as-is** until a deliberate migration decision is made.
The audit tests in
`tests/io/test_frequencyseries_collections_registry.py` lock in the current
dispatch behaviour so that any future refactor can be verified against a known
baseline.

## Future Work

Issue **#444** tracks the decision of whether to:

- Migrate collection containers to use `gwpy.io.registry.default_registry`, or
- Keep the astropy-registry fallback and ensure all needed formats are
  cross-registered, or
- Introduce an explicit public API that abstracts the difference.

No action is required before #444 is resolved.
