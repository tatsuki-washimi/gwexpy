# GWpy 4 Legacy Proxy Compatibility Design

## Status

Approved direction: remove four obsolete developer-utility proxies, repair five runtime proxies against supported GWpy 4 public APIs, and make the FrameL proxy lazy.

This is a post-integration compatibility scope on `agent/v020-integration`. It does not change scientific formulas, release state, package version, remotes, GitHub state, SeriesMatrix B1 adoption, or median-mean scope.

## Problem

`pytest --doctest-modules gwexpy` imports every Python module below `gwexpy/`. Ten legacy proxy modules fail during collection with the supported GWpy 4.0.1 environment:

- five proxies import names that GWpy removed, moved, or made private;
- one FrameL proxy imports an unavailable optional backend eagerly;
- four proxies target utility modules that GWpy removed.

The same ten collection failures reproduce from integration base `6bca889a`, so they are not an approved-head integration regression. They nevertheless keep the required doctest gate red and expose import paths that cannot satisfy the declared `gwpy>=4.0.0,<5.0.0` dependency contract.

## Design goals

1. Every retained GWexpy module imports successfully with the supported base dependency set.
2. Runtime proxies expose a small, explicit, stable contract rather than GWpy implementation imports.
3. Existing GWexpy runtime users, especially LAL interop and table filtering, continue to work.
4. Optional FrameL support fails only when a caller requests the backend.
5. Removed developer utilities are documented as intentional v0.2.0 removals.
6. The full-module doctest command reaches execution without proxy collection errors.

## Non-goals

- Reimplement removed GWpy developer tools.
- Re-export every non-underscore name from GWpy modules.
- Depend on private GWpy names such as `_LAL_UNIT_INDEX`.
- Make FrameL a required dependency.
- Change GWF parallel-read behavior or backend selection.
- Repair unrelated doctest failures discovered after proxy collection succeeds without first recording and classifying them.

## Module disposition

### Immediate removals

Delete these files without deprecated stubs:

| Removed path | Reason | Migration |
|---|---|---|
| `gwexpy/utils/shell.py` | `gwpy.utils.shell` no longer exists and GWexpy has no internal caller. | Use `subprocess` and `shutil.which` directly. |
| `gwexpy/utils/sphinx/__init__.py` | The proxied GWpy developer package no longer exports the old tools. | Import maintained documentation tooling directly. |
| `gwexpy/utils/sphinx/ex2rst.py` | `gwpy.utils.sphinx.ex2rst` no longer exists. | No GWexpy replacement. |
| `gwexpy/utils/sphinx/zenodo.py` | `gwpy.utils.sphinx.zenodo` no longer exists. | Use a maintained Zenodo client or project release tooling. |

Remove the obsolete `gwexpy.utils.sphinx*` package inclusion from `pyproject.toml`. Add English and Japanese v0.2.0 migration notes naming the removed import paths and replacements. Do not create modules whose only behavior is to raise a deprecation error.

### Curated runtime proxies

Retain these modules, but replace stale static imports with explicit contracts:

| Retained path | Required contract |
|---|---|
| `gwexpy/table/filter.py` | Re-export filter constants and parsing/filtering callables used as API: `DELIM_REGEX`, `OPERATORS`, `OPERATORS_INV`, `QUOTE_REGEX`, `filter_table`, `generate_tokens`, `is_filter_tuple`, `parse_column_filter`, `parse_column_filters`, and `parse_operator`. Do not export imported implementation modules or old `OrderedDict`. |
| `gwexpy/table/table.py` | Re-export `DEFAULT_GWOSC_URL`, `TIME_LIKE_COLUMN_NAMES`, `EventTable`, `Table`, `filter_table`, and `parse_operator`. Do not recreate removed implementation helpers such as `attrgetter` or old registry internals. |
| `gwexpy/timeseries/core.py` | Preserve GWpy's declared `TimeSeriesBase`, `TimeSeriesBaseDict`, and `TimeSeriesBaseList` surface. Preserve useful stable compatibility aliases from their maintained owners, including `Channel`, `ChannelList`, `LIGOTimeGPS`, `SegmentList`, `Series`, `Time`, `to_gps`, and `units`. Drop `OrderedDict` and other leaked implementation helpers. |
| `gwexpy/utils/lal.py` | Preserve the LAL conversion constants and callables needed by `gwexpy.interop.lal_`: `LAL_DETECTORS`, type maps/regex/string maps, `find_typed_function`, `from_lal_type`, `from_lal_unit`, `gwpy_units`, `to_gps`, `to_lal_ligotimegps`, `to_lal_type_str`, and `to_lal_unit`. Do not alias private `_LAL_UNIT_INDEX` back to public `LAL_UNIT_INDEX`. |
| `gwexpy/utils/misc.py` | Expose only current semantic utilities `if_not_none`, `property_alias`, `round_to_power`, and `unique`. Drop old standard-library implementation leaks `OrderedDict` and `nullcontext`. |

Each module must define an explicit `__all__`. Avoid a generic `dir(module)` proxy because most relevant GWpy modules have no `__all__` and expose implementation imports such as `numpy`, `operator`, `math`, and typing helpers.

### Lazy FrameL proxy

Retain `gwexpy/timeseries/io/gwf/framel.py` as an optional compatibility path.

- Importing the GWexpy module must not import `framel` or fail when the optional backend is absent.
- Accessing a backend-dependent symbol must load `gwpy.timeseries.io.gwf.framel` lazily.
- Missing `framel` must raise the original actionable `ModuleNotFoundError`; do not replace symbols with `None` or swallow the error.
- `__all__` and `__dir__` must be deterministic and must not force backend loading.
- When a test-provided or installed FrameL backend is available, proxied objects must be the corresponding GWpy objects.
- The new lazy proxy must not alter the integrated `_gwf_io.py` spawn worker, source preflight, or merge behavior.

## Error and compatibility policy

- Deleted import paths fail normally with `ModuleNotFoundError`.
- Removed leaked names fail with `ImportError` from the retained module.
- Retained names preserve object identity with their stable GWpy owner where practical.
- Optional dependency errors occur at backend use, not package discovery or doctest collection.
- No fallback may import a private GWpy symbol merely to preserve an accidental legacy export.

## Testing strategy

Use test-driven development. Add the contract tests first and verify the expected RED state before production edits.

### Import and API tests

1. Import each of the five curated proxy modules under GWpy 4.0.1.
2. Assert exact `__all__` contents and retained object identities.
3. Assert removed leaked names are absent.
4. Exercise table filtering and LAL interop through real GWexpy entry points.
5. Verify the four deleted paths are absent from the package tree and packaging configuration.

### FrameL tests

1. Import the proxy while blocking or omitting `framel`; import must succeed.
2. Access a backend symbol; the optional dependency error must remain explicit.
3. Inject or use an available backend and verify lazy object forwarding.
4. Confirm `dir()` and `__all__` do not load the backend.

### Regression gates

- Focused proxy and LAL interop tests.
- GWF parallel and optional-backend contracts.
- Table tests.
- `pytest -q --doctest-modules gwexpy` with the existing bounded execution policy.
- Full deterministic test-file audit if the broad single process exceeds the harness bound.
- Ruff check and non-mutating format check.
- MyPy for changed production modules and then `mypy gwexpy/`.
- EN and JA Sphinx builds with bounded background execution.
- `git diff --check` and package-build content inspection.

An unrelated failure must be recorded with a baseline reproduction. It must not be relabeled as pass or hidden by collection exclusions.

## Documentation and audit

- Update existing English and Japanese v0.2.0 migration guidance.
- Record the four removals, five curated surfaces, and optional FrameL behavior.
- Add a dedicated YAML audit manifest with commands, RED/GREEN evidence, test results, optional-dependency conditions, and final clean status.
- Do not claim a version bump, release, remote mutation, or GitHub action.

## Review gates

1. Independent Luna specification review of this design and implementation.
2. Independent Sol quality/adversarial review after Luna approval.
3. Return Critical or Important findings to the same implementer and repeat review.
4. Preserve the existing human physics/data-model gates; this compatibility work introduces no new physics judgment.
