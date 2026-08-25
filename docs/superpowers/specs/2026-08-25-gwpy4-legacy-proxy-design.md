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

`pyproject.toml` already excludes `gwexpy.utils.sphinx*` from package discovery.
Preserve that exclusion and verify both the wheel and sdist contain none of the
four removed paths. Add English and Japanese v0.2.0 migration notes naming the
removed import paths and replacements. Do not create modules whose only
behavior is to raise a deprecation error.

### Curated runtime proxies

Retain these modules, but replace stale static imports with explicit contracts:

| Retained path | Required contract |
|---|---|
| `gwexpy/table/filter.py` | Re-export filter constants and parsing/filtering callables used as API: `DELIM_REGEX`, `OPERATORS`, `OPERATORS_INV`, `QUOTE_REGEX`, `filter_table`, `generate_tokens`, `is_filter_tuple`, `parse_column_filter`, `parse_column_filters`, and `parse_operator`. Do not export imported implementation modules or old `OrderedDict`. |
| `gwexpy/table/table.py` | Re-export `DEFAULT_GWOSC_URL`, `TIME_LIKE_COLUMN_NAMES`, `EventTable`, `Table`, `filter_table`, and `parse_operator`. Do not recreate removed implementation helpers such as `attrgetter` or old registry internals. |
| `gwexpy/timeseries/core.py` | Preserve the exact symbol/owner contract below. In particular, import `ChannelList` from `gwpy.detector.channel`, not from `gwpy.timeseries.core`. Drop `OrderedDict` and the implementation helpers listed below. |
| `gwexpy/utils/lal.py` | Preserve the LAL conversion constants and callables needed by `gwexpy.interop.lal_`: `LAL_DETECTORS`, type maps/regex/string maps, `find_typed_function`, `from_lal_type`, `from_lal_unit`, `gwpy_units`, `to_gps`, `to_lal_ligotimegps`, `to_lal_type_str`, and `to_lal_unit`. Do not alias private `_LAL_UNIT_INDEX` back to public `LAL_UNIT_INDEX`. |
| `gwexpy/utils/misc.py` | Expose only current semantic utilities `if_not_none`, `property_alias`, `round_to_power`, and `unique`. Drop old standard-library implementation leaks `OrderedDict` and `nullcontext`. |

Each module must define an explicit `__all__`. Avoid a generic `dir(module)` proxy because most relevant GWpy modules have no `__all__` and expose implementation imports such as `numpy`, `operator`, `math`, and typing helpers.

#### Authoritative `timeseries.core` symbol contract

GWpy 4.0.1 declares only three names in `gwpy.timeseries.core.__all__`.
GWexpy retains those names plus a deliberately enumerated compatibility set.
The owner column is normative and the tests must assert object identity (or,
for string/module aliases, value/module identity) against that owner.

| GWexpy name | Maintained owner | Decision |
|---|---|---|
| `TimeSeriesBase` | `gwpy.timeseries.core.TimeSeriesBase` | retain; declared by GWpy |
| `TimeSeriesBaseDict` | `gwpy.timeseries.core.TimeSeriesBaseDict` | retain; declared by GWpy |
| `TimeSeriesBaseList` | `gwpy.timeseries.core.TimeSeriesBaseList` | retain; declared by GWpy |
| `Channel` | `gwpy.detector.channel.Channel` | retain compatibility alias |
| `ChannelList` | `gwpy.detector.channel.ChannelList` | retain compatibility alias from its corrected owner |
| `LIGOTimeGPS` | `gwpy.time.LIGOTimeGPS` | retain compatibility alias |
| `SegmentList` | `gwpy.segments.SegmentList` | retain compatibility alias |
| `Series` | `gwpy.types.Series` | retain compatibility alias |
| `Time` | `gwpy.time.Time` | retain compatibility alias |
| `to_gps` | `gwpy.time.to_gps` | retain compatibility alias |
| `units` | `astropy.units` | retain compatibility alias |
| `GWOSC_DEFAULT_HOST` | `gwosc.api.DEFAULT_URL` | retain the existing renamed constant |

The exact `__all__` is the twelve names above. `as_series_dict_class` is a
GWpy implementation decorator rather than a declared public export and is
removed. `property_alias` is visible in the upstream module namespace but was
not part of the GWexpy proxy's prior `__all__`, so it is not added. The stale
or implementation-only names `OrderedDict`, `ceil`, `gps_types`, and
`io_registry` are removed. Tests must exercise both retained imports and
`ImportError` for every removed/leaked name.

#### Owners for the other curated proxies

- `gwexpy.table.filter` forwards its exact listed symbols from
  `gwpy.table.filter`.
- `gwexpy.table.table` takes `EventTable` and `TIME_LIKE_COLUMN_NAMES` from
  `gwpy.table.table`, `Table` from `astropy.table`, `DEFAULT_GWOSC_URL` from
  `gwosc.api.DEFAULT_URL`, and `filter_table`/`parse_operator` from
  `gwpy.table.filter`.
- `gwexpy.utils.lal` forwards its exact listed maps, regular expression, and
  conversion functions from `gwpy.utils.lal`; `to_gps` is owned by
  `gwpy.time`, and `gwpy_units` by `gwpy.detector.units`.
- `gwexpy.utils.misc` forwards its exact four listed callables from
  `gwpy.utils.misc`.

No upstream module namespace is used as an implicit source of public names.

### Lazy FrameL proxy

Retain `gwexpy/timeseries/io/gwf/framel.py` as an optional compatibility path.

- Importing the GWexpy module must not import `framel` or fail when the optional backend is absent.
- Accessing a backend-dependent symbol must load `gwpy.timeseries.io.gwf.framel` lazily.
- Missing `framel` must raise the original actionable `ModuleNotFoundError`; do not replace symbols with `None` or swallow the error.
- `__all__` and `__dir__` must be deterministic and must not force backend loading.
- When a test-provided or installed FrameL backend is available, proxied objects must be the corresponding GWpy objects.
- The new lazy proxy must not alter the integrated `_gwf_io.py` spawn worker, source preflight, or merge behavior.

The static FrameL proxy surface is `FRAME_LIBRARY`, `Segment`, `TimeSeries`,
`file_list`, `file_path`, `framel`, `read`, `warnings`, and `write`. Both
`__all__` and `__dir__` are computed from this tuple without importing the
upstream FrameL module. Attribute access imports
`gwpy.timeseries.io.gwf.framel` once and forwards the requested object.

The generic `gwexpy.io.gwf` proxy is explicitly outside the production-change
scope: it imports successfully today and is not one of the ten collection
failures. Its current public surface is nevertheless frozen by a boundary
test to the thirteen GWpy 4 names `BACKENDS`, `backend`, `channel_exists`,
`core`, `data_segments`, `get_backend`, `get_backend_function`,
`get_channel_names`, `get_channel_type`, `identify_gwf`, `import_backend`,
`iter_channel_names`, and `num_channels`. The minimum and latest-4.x checks
must fail if upstream namespace growth changes that surface, preventing an
unreviewed export from becoming part of this compatibility work.

### GWF availability contract

The public I/O contract distinguishes the format family from a selected
backend:

- canonical `format="gwf"` remains available in the base installation and
  retains `optional_dependencies: []`;
- `format="framel"` and `format="gwf.framel"` remain accepted backend aliases,
  but explicit dispatch requires the optional `python-framel` backend;
- absence of FrameL does not make canonical `gwf` unavailable and does not
  prevent importing the lazy compatibility proxy;
- default backend-matrix tests skip the explicit FrameL rows when unavailable;
  `GWEXPY_REQUIRE_GWF_FRAMEL=1` converts that condition into a required-gate
  failure, matching `audit-manifest-356-gwf-gwpy4-compat.yaml`;
- with `python-framel` available, the explicit FrameL read-dispatch rows and
  lazy object-identity tests must pass.

Update the GWF entry's notes in
`docs/developers/contracts/public_io_contract.json` and its generated/readable
contract documentation to state this per-alias distinction. Do not mark the
entire canonical GWF format optional.

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
2. Assert exact `__all__` contents, the per-symbol owners above, and retained
   object identities.
3. Assert every removed leaked name raises `ImportError` when imported from
   its retained proxy.
4. Exercise table filtering and LAL interop through real GWexpy entry points.
5. Verify the four deleted paths are absent from the package tree and packaging configuration.
6. Freeze `gwexpy.io.gwf.__all__` to the explicit boundary list above without
   changing that proxy's production implementation in this scope.

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

### Supported GWpy 4.x matrix

One frozen GWpy version cannot establish compatibility with the declared
`gwpy>=4.0.0,<5.0.0` range. The proxy contract therefore has two mandatory
lanes:

1. the repository's frozen/minimum qualification environment using GWpy
   4.0.1 (`requirements-dev.txt`), including the full doctest command; and
2. the existing `.github/workflows/test-compat-gwpy.yml` environment, which
   resolves the latest available GWpy 4.x and must run the focused proxy,
   table, LAL, and GWF boundary tests.

The implementation updates that existing workflow as follows; it must not
create a release workflow or mutate remote GitHub state.

- Add `gwexpy/table/**`, `tests/table/**`, `tests/interop/**`,
  `tests/test_gwpy4_proxy_contract.py`, and
  `docs/developers/contracts/public_io_contract.*` to the pull-request path
  filters. Existing `gwexpy/timeseries/**`, `gwexpy/io/**`, `gwexpy/utils/**`,
  `tests/timeseries/**`, dependency, and workflow-self triggers remain.
- Install `lalsuite` in the compatibility environment after the base/latest
  GWpy 4.x dependencies. LAL is an optional GWexpy `gw` dependency, but this
  lane must exercise the retained `gwexpy.utils.lal` proxy rather than skip it.
- Add this exact command to the focused compatibility step before the existing
  full `tests/timeseries` invocation:

  ```bash
  pytest -q \
    tests/test_gwpy4_proxy_contract.py \
    tests/table/test_table.py \
    tests/interop/test_interop_lal.py
  ```

  `tests/test_gwpy4_proxy_contract.py` contains the exact curated surfaces,
  corrected symbol owners, deleted-module behavior, generic-GWF boundary, and
  lazy FrameL absence/forwarding contracts. The existing full
  `tests/timeseries` command remains mandatory and supplies the surrounding
  GWF dispatch regression coverage.

Add a workflow contract regression that parses
`.github/workflows/test-compat-gwpy.yml` and requires the path filters,
`lalsuite` provisioning, and all three focused test paths above. This prevents
the declared latest-4.x lane from silently losing the new compatibility gate.

The audit records the exact resolved GWpy and LALSuite versions for each lane.
If the latest-4.x lane is not locally available, local evidence is marked
unavailable rather than inferred; the CI lane remains required before merge.

An unrelated failure must be recorded with a baseline reproduction. It must not be relabeled as pass or hidden by collection exclusions.

## Documentation and audit

- Update `CHANGELOG.md` under `[Unreleased]` with the breaking removals and
  retained-proxy behavior.
- Update the canonical paired migration guides
  `docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md` and
  `docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md`. They are already linked
  from the EN/JA navigation, so verify those links rather than adding a second
  migration page.
- Record the four removals, five curated surfaces, and optional FrameL behavior.
- Add a dedicated YAML audit manifest with commands, RED/GREEN evidence, test results, optional-dependency conditions, and final clean status.
- Validate the public-I/O JSON contract and run the relevant release/docs
  contract tests.
- Do not claim a version bump, release, remote mutation, or GitHub action.

Do not create `release_notes/v0.2.0.md` in this scope. Repository policy makes
`CHANGELOG.md` the source of truth and generates `release_notes/vX.Y.Z.md`
only from a dated/tagged release section; the release-note generator rejects
`Unreleased`. Creating that file now would contradict the retained
"v0.2.0 is Unreleased" boundary. At the later authorized release cut, the
normal generator must synchronize the v0.2.0 file from the finalized
changelog section.

## Review gates

1. Independent Luna specification review of this design and implementation.
2. Independent Sol quality/adversarial review after Luna approval.
3. Return Critical or Important findings to the same implementer and repeat review.
4. Preserve the existing human physics/data-model gates; this compatibility work introduces no new physics judgment.
