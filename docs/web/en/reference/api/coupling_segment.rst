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
round-trip. They preserve Astropy table metadata, attach canonical units on the
Astropy result, and encode non-applicable optional values as explicit nulls.
Native ``Table.from_pandas`` is valid for unitless canonical columns but loses
Astropy metadata and units. Native ``Table.to_pandas`` may convert masked
optionals to floating ``NaN``; that ambiguous representation is deliberately
rejected by :func:`validate` and is not a supported round-trip path.

JSON and frequency-grid normalization
-------------------------------------

:func:`to_json_envelope` first validates its input and then emits only JSON
native values. Schema integer columns normalize to signed int64 Python ints;
finite real-valued frequency, factor, and confidence values normalize to
binary64 Python floats. Values that cannot meet those constraints are rejected.
Consequently, every validated table can be passed directly to ``json.dumps``.

Upper-limit grids accept the larger of exactly 32 IEEE-754 binary64 ULPs (using
adjacent representable values, including near zero) and one billionth of the
nearest positive bin spacing. This permits benign Hz/kHz conversion roundoff
without accepting a material bin mismatch.

``significance`` is intentionally absent from v1 pending approved physics
authority for its formula, witness source, normalization, and upper-limit
applicability.

API
---

.. automodule:: gwexpy.coupling.segment
   :members:
