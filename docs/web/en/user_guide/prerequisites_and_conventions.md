# Prerequisites and Conventions

This page is the entry point for the **shared prerequisites** and **conventions** used across the GWexpy guides and tutorials.
Page-specific assumptions and mathematical details remain on their original pages; this guide is meant to show where to check first.

## At a Glance

| Item | Details |
| --- | --- |
| **Page Role** | Guide |
| **Audience** | Users who want the shared assumptions before diving into tutorials or API pages |
| **Prerequisites** | Basic Python and NumPy familiarity, with GWpy/GWexpy differences still being learned |
| **Use Cases** | Review GPS-time assumptions, FFT conventions, and what is preserved from GWpy |
| **Search Keywords** | prerequisites, conventions, GPS time, FFT, GWpy compatibility, Field API |

## On This Page

- [Environment Prerequisites](#1-environment-prerequisites)
- [Data and Time Assumptions](#2-data-and-time-assumptions)
- [FFT and Spectral Conventions](#3-fft-and-spectral-conventions)
- [GWpy Compatibility and GWexpy Extensions](#4-gwpy-compatibility-and-gwexpy-extensions)
- [Where to Go Next](#5-where-to-go-next)

## 1. Environment Prerequisites

- The basic user-facing environment assumes **Python 3.11+**.
- The minimum background is **basic Python**, **NumPy array handling**, and optionally **Matplotlib**.
- Optional dependencies unlock additional features. See the [Installation Guide](installation.md) for setup details.

If you want the shortest overall learning path first, start with [Getting Started](getting_started.md).

## 2. Data and Time Assumptions

- GWexpy is designed to stay compatible with `gwpy`-style time-series and frequency-series containers.
- Some APIs assume **GPS time** explicitly. In particular, forecast timestamps such as ARIMA outputs should not be confused with UTC-style time systems that include leap seconds.
- Some file formats do not preserve absolute timestamps. Audio formats are a representative case where `t0=0.0` may be used only as a convenience convention.
- Formats that only store local wall-clock time may require an explicit `timezone`. A common example is GBD in the [File I/O Supported Formats Guide](io_formats.md).

For algorithm-specific assumptions, see [Validated Algorithms](validated_algorithms.md).

## 3. FFT and Spectral Conventions

- GWexpy treats FFT **normalization**, **one-sided vs two-sided spectra**, and **sign conventions** explicitly.
- `fft_time` and `fft_space` follow different assumptions for target axes and normalization.
- `spectral_density` distinguishes PSD-style density from per-bin spectrum values.

For the mathematical details, see [FFT Specifications and Conventions](../reference/FFT_Conventions.md).

## 4. GWpy Compatibility and GWexpy Extensions

- GWexpy is built on top of GWpy and preserves the basic data model and workflow expectations where possible.
- At the same time, it adds Matrix containers, the Field API, broader I/O support, and extra analysis utilities that do not exist in GWpy.
- If you want a quick view of what is still "GWpy-like" and what is GWexpy-specific, the migration guide is the fastest entry point.

For migration-oriented guidance, see [Migration from GWpy](gwexpy_for_gwpy_users_en.md) and the [GWpy Difference API Index](gwpy_added_api_index_en.md).

## 5. Where to Go Next

- First-time users: [Getting Started](getting_started.md)
- Hands-on learning: [Tutorial Index](tutorials/index.rst)
- FFT mathematics and normalization details: [FFT Specifications and Conventions](../reference/FFT_Conventions.md)
- Algorithm assumptions and validation basis: [Validated Algorithms](validated_algorithms.md)
- GWpy migration: [Migration from GWpy](gwexpy_for_gwpy_users_en.md)

## Next to Read

- [Installation Guide](installation.md)
- [Getting Started](getting_started.md)
- [File I/O Supported Formats Guide](io_formats.md)
- [Validated Algorithms](validated_algorithms.md)
- [FFT Specifications and Conventions](../reference/FFT_Conventions.md)
