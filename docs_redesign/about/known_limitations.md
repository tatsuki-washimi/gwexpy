---
myst:
  html_meta:
    description: "Current user-visible limitations, fail-closed boundaries, and safe alternatives in GWexpy v0.2.0."
---

# Known Limitations

This page records the high-impact user-visible limitations of the current stable
GWexpy v0.2.0 release. It is a curated compatibility guide, not a complete issue
backlog and not a promise that a limitation will be removed in the next release.

:::{important}
When GWexpy cannot define a safe physical or metadata-preserving result, the
supported contract is to fail explicitly rather than silently downgrade to a bare
`numpy.ndarray`, `astropy.units.Quantity`, or ambiguous timestamp. An explicit
`TypeError` or `ValueError` below is therefore often an intentional safety boundary.
:::

For format-specific and adapter-specific coverage that is not listed here, see
[I/O formats](../how-to/io_formats.md) and
[interoperability](../how-to/interop.md).

## Exact time-axis precision

GWexpy v0.2.0 preserves exact integer GPS-nanosecond origins on the supported
constructor, copy/slice, persistence, and interoperability paths covered by the
release contract. It does not yet define a complete exact rational time-axis
authority for every cadence and operation. Cadences such as 1024 Hz or 4096 Hz
have fractional-nanosecond sample steps, while GWpy-compatible `t0`, `dt`, `times`,
and `xindex` remain binary64 projections.

If one-nanosecond distinctions or exact rational sample boundaries are scientifically
significant, do not use the floating-point time axis as the sole authority across
unsupported transformations or persistence boundaries. Track
[#688](https://github.com/tatsuki-washimi/gwexpy/issues/688).

## SeriesMatrix direct NumPy ufuncs

The v0.2.0 B0 / Phase-A contract deliberately keeps
`SeriesMatrix.__array_ufunc__ = None`. Direct NumPy ufuncs such as
`np.sqrt(matrix)`, `np.log(matrix)`, `np.exp(matrix)`, `np.isfinite(matrix)`, and
`np.isnan(matrix)` therefore raise `TypeError` instead of risking a metadata- or
unit-dropping downgrade.

For square root, use `matrix ** 0.5`; this is contract-tested for the three
SeriesMatrix families and preserves the B0 surface. No metadata-preserving B0
alternative is currently defined for `log`, `exp`, `isfinite`, or `isnan`. Explicit
supported operators, including Quantity-left and Quantity-right multiplication,
remain part of B0. See [#637](https://github.com/tatsuki-washimi/gwexpy/issues/637)
and [#681](https://github.com/tatsuki-washimi/gwexpy/issues/681). The B1 composition
redesign has no assigned version or date.

## Histogram arithmetic

`Histogram` numeric arithmetic (`+`, `-`, `*`, `/`, `**`, reflected and in-place
variants) and direct NumPy ufunc routing are intentionally fail-closed with
`TypeError`. This protects bin geometry and histogram metadata until bin
compatibility, count/weighted/density semantics, and uncertainty propagation are
defined.

This is a current safety contract, not a silent-corruption bug in v0.2.0. Use the
existing histogram-specific operations; convert to raw values only when discarding
bin and uncertainty metadata is intentional. The safety hardening was recorded in
[#579](https://github.com/tatsuki-washimi/gwexpy/issues/579).

## Time interpretation in experimental I/O

### WIN/WIN32

WIN/WIN32 calendar fields are timezone-naive at the file-format level. The current
reader interprets them as UTC with an explicit warning; there is no public
`timezone=` contract yet. Data recorded in another civil timezone therefore requires
external knowledge and care. Track
[#632](https://github.com/tatsuki-washimi/gwexpy/issues/632).

### Numeric CSV timestamps

Numeric CSV timestamps retain the legacy GPS-seconds interpretation. The proposed
explicit `time_scale=` / `time_unit=` contract for GPS, Unix, and relative timestamps
is not implemented. Do not infer a different epoch or unit from numeric magnitude.
Track [#634](https://github.com/tatsuki-washimi/gwexpy/issues/634).

### Broadcast Wave timestamps

Ordinary WAV input has no absolute file timestamp unless the caller supplies
`epoch=`. GWexpy does not yet promote BWF `OriginationDate`, `OriginationTime`, and
`TimeReference` into an absolute epoch because those fields do not identify a
timezone. Track [#636](https://github.com/tatsuki-washimi/gwexpy/issues/636).

## GWF parallel-read scope

v0.2.0 provides spawn-safe `parallel=` reads for a single local frame path.
Multi-worker reads reject caches, URI/composite source spellings, and unsupported
nested execution rather than guessing at semantics. Multi-file streaming merge and
the remaining peak-memory/scalability work are still open. Track
[#588](https://github.com/tatsuki-washimi/gwexpy/issues/588).

For very large multi-file reads, do not assume `parallel>1` implies bounded-memory
streaming. `nproc=` remains a compatibility alias for `parallel=`.

## NDScope HDF5 dataset creation options

The NDScope HDF5 writer validates unsupported dataset-option paths fail-closed, but
v0.2.0 does not provide general configurable HDF5 chunking, compression, shuffle,
checksum, or arbitrary dataset-creation options through this writer. Do not assume
arbitrary writer kwargs are forwarded to `h5py.create_dataset()`. Track
[#590](https://github.com/tatsuki-washimi/gwexpy/issues/590).

## Intentional v0.2.0 boundaries

- The coupling v1 schema does not include `significance`.
- `import gwexpy` uses lazy bootstrap. This is not a missing feature:
  `gwexpy.register_all()` explicitly registers the full supported surface, while
  supported public I/O entry points can register required handlers on demand.
- Future Field I/O, eager/advanced SegmentTable workflows, spatial geometry, mesh
  support, and Fisher/modeling work are roadmap items, not regressions in v0.2.0.

## Reporting a regression

If behavior documented as supported by v0.2.0 produces wrong values, wrong units,
lost metadata/provenance, or a silent downgrade, report it as a bug. Patch releases
in the v0.2.x line are reserved for regressions and correctness defects; new public
APIs remain minor-release work.
