# Wave 3 Field Space Transform Contract Audit

Date: 2026-04-27
Issue: #285, "Audit time-frequency and space-transform contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This slice records the stable field and collection surface that can be asserted
without changing runtime behavior:

- `ScalarField.fft_space()` partial-axis renaming and domain bookkeeping.
- `ScalarField.ifft_space()` round-trip restoration on supported k axes.
- `ScalarField.wavelength()` units and k-domain-only contract.
- `ScalarField` space transforms currently leave the source field unchanged
  but drop gwpy `name` and `channel` metadata on the result.
- `FieldList` and `FieldDict` apply `fft_space_all()` and `ifft_space_all()`
  elementwise while preserving order and keys.
- `VectorField.fft_space_all()` reconstructs a `VectorField` with basis reset
  to default `cartesian`.
- `VectorField.ifft_space_all()` follows the same inherited reconstruction
  behavior.
- `TensorField.fft_space_all()` and `TensorField.ifft_space_all()` reconstruct
  through the inherited `FieldDict` wrapper and infer rank from tuple keys
  instead of preserving an explicit source rank argument.

This slice does not change runtime algorithms. It documents the current
contract for the public field-space transform surface and leaves broader
behavior changes to later slices.

## Contracts Recorded

### ScalarField Space Transforms

- Partial-axis `fft_space()` updates axis names from `x` to `kx` and from `z`
  to `kz` while leaving untouched axes in their original domains.
- `space_domains` reflect the current transformed axes only, with transformed
  axes marked `"k"` and untouched axes remaining `"real"`.
- The spatial k-axis uses angular wavenumber units of `1 / length`.
- `wavelength()` returns length units and yields `inf` at the zero-frequency
  k bin.
- `ifft_space()` restores supported axes back to their original names and
  real-space domains.
- The supported round trip `ifft_space(fft_space(field))` recovers the original
  array values for the axes exercised in this slice.
- gwpy `name` and `channel` metadata are currently not preserved by space
  transforms.
- Source fields keep their original values, public axis descriptors,
  `axis_names`, `axis0_domain`, `space_domains`, unit, name, channel, and
  epoch after scalar space transform calls.

### FieldList and FieldDict Passthrough

- `FieldList.fft_space_all()` forwards `axes` to every member field and returns
  a `FieldList`.
- `FieldList.ifft_space_all()` forwards `axes` to every member field and returns
  a `FieldList`.
- `FieldDict.fft_space_all()` forwards `axes` to every member field and returns
  a `FieldDict`.
- `FieldDict.ifft_space_all()` forwards `axes` to every member field and returns
  a `FieldDict`.
- `FieldList` preserves list order across batch transforms.
- `FieldDict` preserves insertion-order keys across batch transforms.
- Collection wrapper calls leave the source member fields unchanged by the same
  public value, axis, domain, and metadata snapshot used for scalar workflows.
- Inverse collection wrapper calls restore real-space axis names and
  `space_domains` for the exercised public axes.

### Stable Wrapper Behavior

- `VectorField.fft_space_all()` currently reconstructs a `VectorField` whose
  basis falls back to the default `cartesian` value.
- `VectorField.ifft_space_all()` currently follows the same inherited
  reconstruction path and basis fallback.
- `TensorField.fft_space_all()` and `TensorField.ifft_space_all()` currently
  infer rank from tuple-key length during reconstruction. A source tensor with
  an explicit rank that differs from its key length does not preserve that
  explicit rank.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Preserve gwpy `name` and `channel` metadata across space transforms.
- Retain a non-default `VectorField` basis across collection reconstruction.
- Preserve explicit `TensorField.rank` values across inherited collection
  reconstruction when rank differs from tuple-key length.
- Change Fourier normalization, angular wavenumber convention, or wavelength
  semantics.
- Broaden the public contract beyond the current scalar, vector, tensor, and
  collection wrappers listed above.
- Treat private implementation attributes as public contract surface.

## Follow-Up Slices For #285

1. Field-space behavior changes that require runtime decisions: gwpy
   `name`/`channel` preservation, non-default `VectorField` basis preservation,
   and explicit `TensorField.rank` preservation across inherited collection
   wrappers.
2. PyEMD-dependent HHT numerical behavior after optional dependency and
   physics-review expectations are explicit.

## Verification

Suggested focused command:

```bash
rtk proxy pytest tests/fields/test_space_transform_contracts.py -q
```

Suggested follow-up checks:

```bash
rtk proxy pytest tests/fields/test_space_transform_contracts.py \
  tests/fields/test_scalarfield_fft_space.py \
  tests/fields/test_scalarfield_collections.py \
  tests/fields/test_scalarfield_metadata.py \
  tests/fields/test_vectorfield.py \
  tests/fields/test_tensorfield.py -q
rtk ruff check tests/fields/test_space_transform_contracts.py
rtk ruff format --check tests/fields/test_space_transform_contracts.py
rtk git diff --check
```
