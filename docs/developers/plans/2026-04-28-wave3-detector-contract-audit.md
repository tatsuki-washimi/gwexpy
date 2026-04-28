# Wave 3 Detector Contract Audit

Date: 2026-04-28
Issue: #281, "Audit detector channel proxy and unit alias contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #281 slice records deterministic detector passthrough contracts that
can be baselined without network, NDS, datafind, remote channel lists, or
runtime behavior changes:

- package-level `gwexpy.detector` proxy exposure for representative GWpy public
  names;
- module-level `gwexpy.detector.channel` re-export identity for representative
  channel helpers;
- module-level `gwexpy.detector.units` re-export identity for representative
  unit helpers and alias state;
- current `Channel` construction, string serialization, copy, and parsed
  metadata behavior for a local deterministic channel name;
- stable Astropy/GWpy unit parsing for non-ambiguous base units.

This slice does not change detector runtime modules, channel parsing, channel
metadata, unit aliases, unit failure modes, optional backend behavior, KAGRA
overrides, or unified unit registry behavior.

## Contracts Recorded

### Package Proxy Boundary

- `gwexpy.detector.Channel` and `gwexpy.detector.ChannelList` are current
  package-level passthrough attributes to `gwpy.detector`.
- `gwexpy.detector.__all__` and `dir(gwexpy.detector)` expose representative
  public names from GWpy, including `Channel`, `ChannelList`, `channel`, and
  `units`.
- Importing local shim modules such as `gwexpy.detector.channel` can bind the
  package's `channel` attribute to the gwexpy shim module. This is recorded as
  the current Python package boundary; the underlying shim still re-exports
  GWpy objects directly.

### Channel Module Passthrough

- `gwexpy.detector.channel.Channel` is the same object as
  `gwpy.detector.channel.Channel`.
- `ChannelList`, `parse_unit`, and `to_gps` are also direct GWpy re-exports in
  the covered module-level contract.
- Constructing `Channel("K1:PEM-EXAMPLE_TEST")` returns the GWpy `Channel`
  type. Its string form and `.name` remain the original channel name.
- For the covered deterministic channel name, parsed metadata currently reports
  `ifo="K1"`, `system="PEM"`, `subsystem="EXAMPLE"`, and `signal="TEST"`.
- `Channel.copy()` returns a distinct GWpy `Channel` instance that compares
  equal to the source and preserves the covered name/string/metadata fields.

### Unit Module Passthrough

- `gwexpy.detector.units.parse_unit`, `alias`, `aliases`, and
  `UNRECOGNIZED_UNITS` are direct re-exports from `gwpy.detector.units`.
- `gwexpy.detector.channel.parse_unit` and `gwexpy.detector.units.parse_unit`
  currently resolve the same stable base-unit aliases for the covered cases.
- The first slice intentionally asserts only stable public units:
  `"m"`/`"meter"`, `"s"`, and `"Hz"`.

## Optional I/O And Environment Caveats

This slice avoids contracts that would require:

- network-backed NDS access;
- remote detector metadata or channel list services;
- datafind, frame files, or site-specific channel databases;
- optional detector I/O backends beyond the installed GWpy import surface.

Future tests that exercise `ChannelList.read()`, NDS-backed channel discovery,
or detector metadata services should declare backend availability, environment
variables, and skip/failure behavior explicitly.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Add KAGRA-specific detector/channel overrides.
- Make detector package submodule attributes strict GWpy module identities after
  importing gwexpy shim modules.
- Replace direct GWpy re-exports with gwexpy-owned wrapper classes.
- Change channel construction, parsing, equality, copy, serialization, or
  metadata normalization behavior.
- Add a unified gwexpy unit registry or stronger pint/domain unit parsing.
- Change detector unit aliases, ambiguous aliases, or unrecognized-unit failure
  modes.
- Hard-code behavior for ambiguous detector units such as `count`, `ct`,
  `strain`, `1/Hz`, or `Hz^-1/2` before a public-version compatibility audit.

## Follow-Up Slices For #281

1. Channel construction and serialization matrix: trend types, sample rates,
   frametype handling, equality/hash behavior, and round trips that remain
   independent of remote metadata.
2. Channel list and optional I/O boundaries: local-only CLF/CIS fixtures,
   explicit optional-backend skip policy, and NDS/datafind isolation.
3. Unit alias audit: public alias surface, ambiguous detector-domain units,
   unrecognized-unit behavior, Astropy-version differences, and compatibility
   notes.
4. Detector package ownership decision: document which names remain direct
   GWpy passthrough and which, if any, become gwexpy-owned behavior.
5. KAGRA and domain-specific extensions: behavior-changing proposal with human
   unit/detector review before implementation.

## Verification

Focused detector contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/detector/test_detector_contracts.py -p no:cacheprovider
```

Related detector compatibility checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q \
  tests/detector/test_detector_contracts.py \
  tests/detector/test_channel.py \
  tests/detector/test_units.py \
  -p no:cacheprovider
```

Changed-file hygiene:

```bash
rtk ruff check tests/detector/test_detector_contracts.py
rtk ruff format --check tests/detector/test_detector_contracts.py
rtk python -c "import yaml; from pathlib import Path; yaml.safe_load(Path('docs/developers/plans/audit-manifest-281-detector-contracts.yaml').read_text())"
rtk git diff --check
```
