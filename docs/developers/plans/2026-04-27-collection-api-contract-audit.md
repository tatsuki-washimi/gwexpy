# Collection API Contract Audit

Date: 2026-04-27
Issue: #289, "Audit collection API ordering mapping and metadata contracts"
Mode: audit-first; docs and regression tests only.

## Constraints

- This pass records current collection behavior for release readiness. It does
  not change runtime behavior.
- `.agent/AGENTS.md` requirements on unit, axis, and metadata preservation were
  reviewed. Behavior that changes physics-facing metadata, axis comparison, or
  serialization semantics must be split into a follow-up with human review.
- #270 and #271 already cover family-specific `to_matrix()` and matrix
  round-trip persistence. This audit focuses on collection-level ordering,
  delegation, mutability, key mapping, and metadata boundaries outside broad
  matrix behavior changes.

## Inputs Reviewed

- `gwexpy/types/mixin/_collection_mixin.py`
- `gwexpy/timeseries/collections.py`
- `gwexpy/frequencyseries/collections.py`
- `gwexpy/spectrogram/collections.py`
- `gwexpy/histogram/collections.py`
- `gwexpy/io/hdf5_collection.py`
- `gwexpy/io/collection_dir.py`
- Existing collection tests under `tests/types/`, `tests/timeseries/`,
  `tests/frequencyseries/`, `tests/spectrogram/`, `tests/histogram/`, and
  `tests/io/`.

## Collection Contract Table

| Surface | Ordering contract | Delegation return contract | Mutability contract | Metadata and key contract |
| --- | --- | --- | --- | --- |
| Shared `DictMapMixin` helpers | Iterate in current mapping insertion order. | `_make_dict_map_method()` returns `self.__class__()` unless `result_class_path` is supplied; `_make_dict_plain_method()` returns native `dict`. | Delegated methods construct a new result container. Exceptions from element methods propagate. | Mixins do not inspect units, names, channels, axes, or metadata. Concrete collections own those guarantees. |
| Shared `ListMapMixin` helpers | Iterate in current list order. | `_make_list_map_method()` returns `self.__class__()` unless `result_class_path` is supplied; `_make_list_plain_method()` returns native `list`. | Delegated methods construct a new result container. Exceptions from element methods propagate. | Mixins do not inspect metadata. If a list map is used for scalar-returning methods, list validation may reject the scalar result. |
| `TimeSeriesDict` / `TimeSeriesList` | Dict keys and list order are preserved when mapping over entries. `to_matrix()` consumes that order. | Many statistical methods convert through `to_matrix()` and return arrays. Rolling and signal methods return collection types. | `crop()` returns a new collection. It delegates `copy` to each underlying series crop. | `TimeSeriesDict.to_matrix()` writes dict keys to `channel_names`, but row keys remain generated. Current conversion does not preserve per-element value units or channels in matrix metadata. |
| `FrequencySeriesDict` / `FrequencySeriesList` | Dict/list insertion order is preserved. `FrequencySeriesDict.to_matrix()` uses dict order for rows. | Dict map methods return `FrequencySeriesDict` or native `dict` depending on helper. List map methods return `FrequencySeriesList`. | `FrequencySeriesDict.crop()` mutates and returns `self`; `FrequencySeriesList.crop()` returns a new list collection. | `FrequencySeriesDict.to_matrix()` preserves row keys and per-element unit/name/channel metadata. `FrequencySeriesList` has no `to_matrix()` contract today. |
| `SpectrogramDict` / `SpectrogramList` | Dict/list insertion order is preserved. Dict `to_matrix()` uses keys as rows; list `to_matrix()` uses generated rows. | Batch methods return dict/list collection types except explicit tensor/backend conversion helpers, which return native dict/list values. | `crop()` defaults to copy and supports in-place operation through `copy=False` or legacy `inplace` forms. | `to_matrix()` preserves per-element unit/name/channel metadata. A global matrix unit is set only when all entries share the same value unit. |
| `HistogramDict` / `HistogramList` | Dict/list insertion order is preserved by map helpers and HDF5/list writes. | `HistogramDict` scalar/statistical helpers return native `dict`. `HistogramList` currently uses collection-map helpers for several scalar/statistical methods. | Map helpers construct new containers. HDF5 readers skip unreadable entries under the existing tolerant read policy. | Histogram collection manifests preserve collection order. List statistical helper behavior is a current limitation when scalar results fail list validation. |
| HDF5 collection manifests | Writers store `gwexpy_order`; readers use it before falling back to file keys. | Not applicable. | Reads skip unreadable entries for expected parsing exceptions. | Dict writes sanitize physical HDF5 object names and store original keys in `gwexpy_keymap`; list writes store positional string keys. |
| Directory collection manifests | Manifest entries preserve explicit order. No-manifest fallback sorts matching `*.csv` / `*.txt` filenames. | Not applicable. | Directory overwrite behavior is explicit. | Manifest entries preserve original keys and optional per-entry metadata. No-manifest fallback uses filename stems as keys. |

## Ordering And Key Semantics

Collection methods follow Python insertion/list order. That order is observable
in mixin-generated methods, matrix row construction, HDF5 manifest order, and
directory collection manifests.

Dict-style matrix conversion is uneven by family:

- `FrequencySeriesDict.to_matrix()` uses dict keys as matrix row keys and keeps
  the single column key as `"value"`.
- `SpectrogramDict.to_matrix()` uses dict keys as matrix row keys.
- `TimeSeriesDict.to_matrix()` writes dict keys into element names through
  `channel_names`, while matrix row keys remain generated labels such as
  `"row0"`, `"row1"`.

This pass treats those differences as current contracts to document, not as
runtime behavior to normalize.

## Delegation And Container Return Semantics

Shared mixin helpers have two explicit return families:

- map helpers return collection-like containers suitable for element methods
  that return another element of the same family;
- plain helpers return native `dict` or `list` for scalar, ndarray, backend, or
  summary outputs.

Concrete collections are not fully uniform in which helper they use. In
particular, `HistogramList` currently uses list map helpers for several
statistical methods whose element-level return values are quantities, and the
list validation path rejects those quantity values. That behavior is now
baselined as a limitation instead of being silently changed in this audit PR.

## Mutability Contracts

Mutability differs across families:

- `TimeSeriesDict.crop()` and `TimeSeriesList.crop()` return new collection
  containers.
- `FrequencySeriesDict.crop()` mutates and returns `self`, matching its current
  GWpy-like behavior.
- `FrequencySeriesList.crop()` returns a new list collection through the shared
  list map helper.
- `SpectrogramDict.crop()` and `SpectrogramList.crop()` default to copied
  collections and support in-place operation through `copy=False`; deprecated
  `inplace` compatibility is handled by `_resolve_crop_compat_args()`.

These differences are user-observable and should not be unified without a
separate behavior-change PR.

## Error Propagation And Tolerant Reads

Mixin-generated delegation does not catch element-method exceptions. The first
raised exception propagates to the caller.

Collection readers are different: HDF5 collection readers skip entries that
fail with expected parsing exceptions such as `KeyError`, `ValueError`,
`TypeError`, or `OSError`. Tightening tolerant reads into strict failure would
be a behavior change and should be split from this audit.

## Metadata Contracts

The shared mixins preserve ordering and container shape but do not preserve or
interpret physical metadata. Metadata policy belongs to concrete collection
implementations.

Current high-level metadata boundaries:

- `TimeSeries` collection matrix conversion preserves key-derived element names
  but not per-element value units or channels.
- `FrequencySeriesDict.to_matrix()` preserves row keys plus per-element
  unit/name/channel metadata.
- `SpectrogramDict/List.to_matrix()` preserves per-element unit/name/channel
  metadata and sets a global matrix unit only when all entries share the same
  value unit.
- HDF5 and directory collection manifests preserve collection-level keys and
  ordering. Per-entry serialization details remain owned by the entry reader and
  writer.

## Regression Test Plan

This audit adds focused regression coverage for:

- shared dict/list mixin ordering, return-class, plain-return, and exception
  propagation behavior;
- `TimeSeriesDict/List.to_matrix()` ordering and current metadata loss;
- `FrequencySeriesDict.to_matrix()` row/key and per-entry metadata behavior;
- current absence of `FrequencySeriesList.to_matrix()`;
- `SpectrogramDict/List` copy/matrix row-name behavior;
- current `HistogramList` scalar/statistical helper limitation;
- directory no-manifest fallback sorting.

Suggested focused command:

```bash
rtk pytest -q \
  tests/types/test_collection_mixin.py \
  tests/timeseries/test_collections_batch.py \
  tests/frequencyseries/test_frequencyseries_containers.py \
  tests/spectrogram/test_collections.py \
  tests/histogram/test_histogram_collections.py \
  tests/io/test_collection_dir.py \
  tests/io/test_hdf5_collection.py
```

## Behavior Changes Deferred From First PR

- Enforce frequency-axis equality in `FrequencySeriesDict.to_matrix()`.
- Add `FrequencySeriesList.to_matrix()`.
- Preserve `TimeSeries` value units/channels through `TimeSeriesDict/List` matrix
  conversion.
- Normalize crop mutability across collection families.
- Change HDF5 tolerant read behavior to strict failure.
- Change HDF5 key sanitization, collision suffixes, manifest names, or manifest
  schema.
- Fix `HistogramList` scalar/statistical return behavior before maintainers
  decide whether those methods should return native `list` values or another
  collection-like result.
