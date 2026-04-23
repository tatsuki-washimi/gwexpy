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

## Schema

Each target entry contains these fields:

- `name`: stable contract key
- `module`: source module under `gwexpy.interop`
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

## Boundary Decisions

### Public surface

The contract currently fixes the high-value public surface:

- storage/container bridges such as `hdf5`, `json`, `dict`, `zarr`, `netcdf4`
- analysis-library bridges such as `pandas`, `xarray`, `xarray-field`,
  `astropy`, `dask`
- major domain bridges such as `obspy`, `lal`, `pycbc`, `gwinc`, `finesse`,
  `control`, and `mth5`

These are the entries where user-facing documentation already implies stable
public use.

### Partial reference entries

Some targets remain reference-indexed but are not fully public.

- `root` stays `implemented_partial`

Reason: the guide explicitly records an incomplete conversion boundary, and the
contract must preserve that warning instead of silently upgrading the status.

### Out of scope for this slice

Implemented-only targets that are not yet reference-indexed remain outside this
contract slice for now.

Examples include:

- `sqlite`
- `polars`
- `simpeg`
- `mne`
- `neo`
- `quantities`
- `pyroomacoustics`
- `meshio`

They still have tests, but they are not yet governed by the same
guide/reference publication rule.

## Execution Rule

When promoting an interop path to public, land these items together:

1. update the contract
2. expose the top-level namespace entry
3. add or wire the reference page
4. add docs-sync tests
5. put those tests behind a PR gate
