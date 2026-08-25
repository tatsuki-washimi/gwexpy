Coupling segment schema
=======================

.. currentmodule:: gwexpy.coupling.segment

``gwexpy.coupling.segment`` provides a strict, versioned interchange table for
finite coupling-factor measurements and upper limits. It does not replace the
existing ``CouplingResult`` CSV/TXT exporters.

Units and table conversion
--------------------------

The column names are the schema authority for time and frequency values:
``start_gps_ns`` and ``duration_ns`` are integer nanoseconds, while
``frequency_hz`` is in Hz. Unitless pandas columns and unitless Astropy columns
are interpreted in those canonical units. An explicit Astropy unit must match
exactly: ``ns`` for the two time columns and ``Hz`` for frequency. In
particular, raw values declared as seconds or days are rejected rather than
being relabelled as nanoseconds.

Use :func:`to_pandas` and :func:`to_astropy` for a full pandas/Astropy
round-trip. They preserve table metadata, arbitrary nested Astropy
``Column.meta`` mappings, descriptions, formats, canonical units, and optional
masks/nulls with mutation-independent copies. A strict, versioned carrier is
stored under the reserved ``DataFrame.attrs`` key
``gwexpy.coupling.segment.v1.astropy_metadata``; a malformed carrier is
rejected rather than partly applied. Native ``Table.from_pandas`` is valid for
unitless canonical columns but loses Astropy metadata and units. Native
``Table.to_pandas`` may convert masked optionals to floating ``NaN``; that
ambiguous representation is deliberately rejected by :func:`validate` and is
not a supported round-trip path. ``from_result`` and ``from_results`` emit
``None`` for non-applicable textual measurement ``limit_method`` cells, while
the nullable-binary64 pandas ``confidence_level`` column uses ``pd.NA``.
Astropy represents either absence as a mask and the JSON envelope as ``null``.
A legacy empty string is accepted only in those two measurement fields, then
canonicalized by the public adapters and JSON envelope; it cannot become a
string-valued upper-limit confidence column.
With neither optional factory argument, result factories retain the minimal
required-plus-``estimate_kind`` shape. Supplying ``limit_method`` requests that
column for every row, and supplying a valid ``confidence_level`` requests both;
measurement cells use the above explicit nulls and empty mappings use the same
deterministic shape.

The same per-column dtype contract applies to empty and populated tables:
times are signed ``int64``; frequency, coupling factor, and present confidence
are binary64; and textual columns are object-capable rather than fixed-width.
Thus an empty table can be safely populated or stacked later without string
truncation or dtype inference. The ordered JSON envelope recreates this
contract for zero-row tables.

JSON and frequency-grid normalization
-------------------------------------

:func:`to_json_envelope` first validates its input and then emits only JSON
native values. Schema integer columns normalize to signed int64 Python ints;
finite real-valued frequency, factor, and confidence values normalize to
binary64 Python floats. Values outside finite binary64 range and finite nonzero
values that would underflow to zero are rejected; accepted conversions preserve
sign and zero/nonzero status. Consequently, every validated table can be
passed directly to ``json.dumps``.

Upper-limit grids accept at most exactly 32 IEEE-754 binary64 ``nextafter``
steps per direction, independently upward and downward, including at powers of
two and near zero. Step 33 is rejected in either direction. This permits benign
Hz/kHz conversion roundoff without accepting a material bin mismatch.

``significance`` is intentionally absent from v1 pending approved physics
authority for its formula, witness source, normalization, and upper-limit
applicability.

API
---

.. automodule:: gwexpy.coupling.segment
   :members:
