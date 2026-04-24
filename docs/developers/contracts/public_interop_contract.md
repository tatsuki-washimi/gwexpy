# Public Interop Contract

This document defines how `docs/developers/contracts/public_interop_contract.json`
should be interpreted.

## Purpose

The public interop contract keeps these surfaces aligned:

- the interop guide
- the `gwexpy.interop` top-level namespace
- the interop reference index
- CI checks that must fail when a documented bridge regresses

Unlike direct file I/O, the main failure mode here is not registry drift. It is
documentation claiming that a `to_*()` / `from_*()` path is publicly usable
when the top-level namespace or reference index does not actually expose it.
This contract also fixes the opposite failure mode: implemented-only bridges
silently drifting into or out of the docs without an explicit boundary decision.

## Schema

Each target entry contains these fields:

- `name`: stable contract key
- `module`: source module under `gwexpy.interop`, or `null` when the guide row
  intentionally has no dedicated helper module yet
- `status`: public presentation status used by the guide
- `guide_api`: `to_*()` / `from_*()` names published for that target
- `row_match_en`: marker string used to find the English guide row
- `row_match_ja`: marker string used to find the Japanese guide row
- `reference_indexed`: whether the module must appear in `reference/api/interop.rst`
- `reference_page`: expected API reference page filename when indexed
- `details_link`: expected guide-side API link when the row links directly to a page
- `notes`: short rationale for boundary decisions

## Rules

- Every function in `guide_api` must be importable from `gwexpy.interop`.
- Every function in `guide_api` must exist in `gwexpy.interop.<module>`.
- `status = public` requires `reference_indexed = true`.
- `reference_indexed = true` requires matching English and Japanese
  `reference/api/interop.rst` entries and matching generated module pages.
- The guide status label must match the contract in both English and Japanese.
- `reference_indexed = false` means the module must stay out of the interop
  reference index until an explicit publication decision is made.
- Every guide row with an interop status label must appear in this contract.

## Boundary Decisions

### Public surface

The contract currently fixes the high-value public surface:

- storage/container bridges such as `hdf5`, `json`, `dict`, `zarr`, `netcdf4`
- storage-adjacent bridges such as `sqlite`
- analysis-library bridges such as `pandas`, `polars`, `xarray`, `xarray-field`,
  `astropy`, `dask`
- array/tensor bridges such as `torch`, `tensorflow`, `jax`, `cupy`
- geophysics bridges such as `simpeg`
- biosignal and unit bridges such as `mne`, `neo`, `quantities`
- audio and acoustics bridges such as `pyroomacoustics`, `pydub`, `librosa`
- spectral, circuit, and RF bridges such as `specutils`, `pyspeckit`,
  `pyspice`, `skrf`
- modal and structural-dynamics bridges such as `pyoma`, `multitaper`,
  `mtspec`, `sdypy`, `sdynpy`
- field-simulation bridges such as `meep`, `openems`, `emg3d`
- mesh, geoscience, and simulation-import bridges such as `meshio`, `metpy`,
  `wrf`, `harmonica`, `exudyn`, `opensees`
- major domain bridges such as `obspy`, `lal`, `pycbc`, `gwinc`, `finesse`,
  `control`, and `mth5`

These are the entries where user-facing documentation already implies stable
public use.

### Implemented-only surface

There are currently no documented helper-backed bridges in this category.

If a future row uses `status = implemented` together with a non-empty
`guide_api`, it must remain out of the reference index until a deliberate
publication decision is made.

### Partial reference entries

Some targets remain reference-indexed but are not fully public.

- `root` stays `implemented_partial`

Reason: the guide explicitly records an incomplete conversion boundary, and the
contract must preserve that warning instead of silently upgrading the status.

### No-helper rows

Some guide rows intentionally have no dedicated top-level helper family yet.

- `numpy`
- `mtpy`
- `elephant`

These use `module = null` and `guide_api = []` so the docs can still be
tracked without pretending a public helper module exists.

## Execution Rule

When promoting an interop path to public, land these items together:

1. update the contract
2. expose the top-level namespace entry
3. add or wire the reference page
4. add docs-sync tests
5. put those tests behind a PR gate
