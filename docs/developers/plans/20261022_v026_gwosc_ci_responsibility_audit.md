# GWexpy v0.2.6 GWOSC CI responsibility audit

Status: **Phase B/C validation complete; Phase D source review pending.**

Audit base: approved Source S20 `da3391d12d97a793c9710107d03d78e9f41d8e3b`.
Candidate R evidence: `caa780f025443bccbce463e17577ef488e9b10e3`.
GWpy test module SHA-256 (both versions):
`74cb0aac347e3df06e8660dd3ace9f5c14954a66e0516ec2f97e154d9d251766`.

## A. Root cause

The candidate's GWpy 4.0.1 and 4.0.2 jobs ran the same upstream test module,
Python 3.11.15, and Certifi 2026.10.19. GWpy 4.0.1 reported 3,091 passed,
146 skipped, 5 xfailed, 65 errors, and 1 failed. The 65 errors were consumers
of the live `gw150914` fixture. Their tracebacks ended in TLS certificate-chain
verification failure at `https://gwosc.org/api/v2/event-versions...`. The
remaining failure was `test_fetch_open_data_error`: the upstream test expected
`ValueError("Cannot find a GWOSC dataset")`, but the request failed earlier
with the TLS error.

GWpy 4.0.2 reported 3,091 passed, 214 skipped, and 3 xfailed. The fixture's
TCP preflight skipped when DataFind was unavailable, so the 65 fixture
consumers did not reach the GWOSC HTTPS request. The skip is an availability
guard, not evidence that TLS validation or a GWOSC request succeeded. The
different outcomes are due to the point at which the external network path
failed; the audited upstream test module is byte-identical. Certifi is not
being downgraded and TLS checks remain enabled.

## B. Responsibility boundary

The inherited local class is `tests/timeseries/test_timeseries.py::TestTimeSeries`.
It does not override upstream `TEST_CLASS`; the upstream value resolves to
`gwpy.timeseries.TimeSeries`. Its inherited tests therefore exercise GWpy's
class and upstream behavior, not `gwexpy.timeseries.TimeSeries`. The local
`gw150914` fixture calls that same `TEST_CLASS.get(...)` after a TCP preflight.
The repository-wide autouse `_datafind_network_fallback` also monkeypatches
`gwpy.timeseries.TimeSeriesDict.find` for tests, translating selected network
errors to `RuntimeError`; this is test-harness behavior and is not a GWexpy
product change.

Direct GWOSC/DataFind calls are upstream responsibilities. GWexpy's own
overrides, I/O registration, dispatch, conversion, metadata handling, and
scientific behavior remain GWexpy responsibilities. In particular,
`tests/timeseries/test_io_losc.py::test_losc_reexports_gwpy` is an offline
identity/re-export assertion and remains in the required tests.

## C. Test inventory

Both GWpy 4.0.1 and 4.0.2 collected 3,308 IDs. Their fixture audit produced
the same 92 candidate live-fixture IDs (8 Category A and 84 Category C), with
no unknown `gw150914*` fixture names. The complete node-ID list, fixture
names, category, and reason are in
[`gwosc_timeseries_inventory.json`](../../../tests/docs/gwosc_timeseries_inventory.json).
The complete collection reports are in `$TMPDIR/gwosc-s20-audit/` as
`audit-final-4.0.1.json` and `audit-final-4.0.2.json`.

| Category | Collected IDs | Contents and treatment |
| --- | ---: | --- |
| A | 8 | Direct inherited GWOSC/DataFind tests: `test_fetch_open_data`, `test_fetch_open_data_error`, `test_find`, `test_find_best_frametype_in_find`, `test_find_datafind_httperror`, `test_find_datafind_runtimeerror`, `test_get_datafind`, and `test_get_gwosc_kwargs`. Four are already unconditionally skipped in the local subclass; they are not newly excluded. The other four are live API cases. |
| B | Not part of the 92 live-fixture set | GWexpy-owned tests and overrides, including the LOSC re-export identity check, focused compatibility checks, override inventory/evidence execution, and synthetic/offline scientific and I/O regressions. Keep as required gates. |
| C | 84 | Inherited upstream calculation/I/O cases whose inputs come from `gw150914`, `gw150914_16384`, `gw150914_h1_32`, or `gw150914_l1_32`; examples include FFT/PSD/ASD, spectrogram, filtering, resampling, Q-transform, and GWF interval assertions. Their numerical value is not dismissed. Their current `TEST_CLASS` is nevertheless GWpy's class. Phase A2 mapped all 84 cases individually: 83 are upstream-only; the multiprocessing harness affects one, now covered by an offline GWexpy regression. The three H1/L1-fixture cases remain selected in the required suite. |

The local `gw150914_h1_32` and `gw150914_l1_32` fixtures are unconditionally
skipped because the LAL/frame-backed fixtures segfault in the CI environment.
The three dependent collected IDs are `test_transfer_function`,
`test_coherence`, and `test_coherence_spectrogram`. They are classified C in
the inventory, but this audit does not change those existing fixture skips.

The baseline Candidate R summaries were:

| GWpy | Collected | Passed | Errors | Failed | Skipped | Xfailed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0.1 | 3,308 | 3,091 | 65 | 1 | 146 | 5 |
| 4.0.2 | 3,308 | 3,091 | 0 | 0 | 214 | 3 |

These are the original full-suite Candidate R results, not the post-change
execution set. Post-change results are recorded in Phase B/C below.

## Phase A2. Category C responsibility audit

The full 84-case audit is recorded in
[`gwosc_timeseries_phase_a2_category_c.json`](../../../tests/docs/gwosc_timeseries_phase_a2_category_c.json).
Both GWpy versions use `gwpy.timeseries.TimeSeries` as the effective test
class. The 84 cases classify as 83 `upstream-only`, one `gwexpy-influenced`,
and zero `unresolved`.

The single influenced case is
`TestTimeSeries::test_spectrogram_multiprocessing`. The autouse `_mp_fallback`
in `tests/conftest.py` forces GWpy's multiprocessing helper to one worker when
the platform lacks semlocks, so this inherited case cannot guarantee GWexpy's
parallel route in that environment. The existing GWexpy terminal compatibility
suite had only an `nproc=1` differential test. A synthetic-data GWexpy
`TimeSeries.spectrogram()` test now compares `nproc=1` and `nproc=2`, including
values, units, time/frequency axes, and metadata. It passes with GWpy 4.0.1 and
4.0.2.

For the other 83 Category C cases, the target is GWpy's base `TimeSeries`, with
no GWexpy class, override, I/O dispatch, or metadata conversion on the tested
path. GWexpy I/O registrations attach to GWexpy's subclass and do not change
dispatch on a base `gwpy.timeseries.TimeSeries`. The autouse DataFind fallback
wraps `TimeSeriesDict.find`; it does not participate in these science fixture
calculations. The inherited numerical examples remain useful upstream GWpy
science tests, but separating them from GWexpy's required gate removes no
GWexpy implementation assertion. The three H1/L1 cases remain selected in the
required suite and keep their existing local fixture skips; they are not live
marker exclusions.

## Phase B/C. Implementation and validation

The test marker is scoped to `tests/timeseries/test_timeseries.py::TestTimeSeries`
and only the audited live fixtures (`gw150914`, `gw150914_16384`) or eight
audited direct GWOSC/DataFind methods. The H1/L1 fixtures are excluded from
that predicate because their local fixture bodies unconditionally skip. No
general network marker is used.

Both GWpy environments collect 3,309 timeseries IDs: the original S20
collection of 3,308 plus the new offline multiprocessing regression. Each has
256 inherited `TestTimeSeries` IDs and the same 92 fixture/direct-call audit
cases (8 A, 84 C). Exactly 89 IDs receive `upstream_gwosc_live`: all 8 A and
81 C with an active live fixture. The other 3 C IDs depend only on the already
skipped H1/L1 fixtures and remain in the required set. The required selection
has 3,220 IDs. The full ID lists, category, fixtures, reasons, marker IDs,
upstream test-source SHA, and inherited-ID baseline are in
[`gwosc_timeseries_inventory.json`](../../../tests/docs/gwosc_timeseries_inventory.json).
The script fails on new or missing inherited test IDs, changed fixture cases,
marker-set drift, unknown `gw150914*` fixtures, unsupported GWpy versions, or
an audit source SHA mismatch. CI saves its JSON and text reports with the
existing JUnit, logs, and version evidence.

| GWpy | collected | required gate | deselected | passed | skipped | xfailed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0.1 | 3,309 | 3,220 | 89 | 3,092 | 125 | 3 |
| 4.0.2 | 3,309 | 3,220 | 89 | 3,173 | 44 | 3 |

Both required suites passed using Certifi 2026.10.19. The existing focused,
proxy, and CI contract tests passed on both versions; override inventory and
evidence execution passed with 396 cases on both versions. Ruff check,
format, and `git diff --check` passed. One initial 4.0.1 run used a long local
pytest temp path and failed to bind a Unix socket (`AF_UNIX path too long`);
rerunning with a short HDD-backed basetemp passed. The normal repository
environment was not modified; Certifi, LALSuite, and `iminuit` overlays were
installed under `$TMPDIR` for validation.

No `gwexpy/` product code, public API, scientific algorithm, Certifi policy, or
TLS verification behavior changed. Focused compatibility, proxy, override
inventory, artifact, and exact source checkout gates remain. S20, Candidate R,
`main`, `maint/0.2`, and PR #805 remain unchanged. The corrected S21 source is
frozen on this isolated S20-based branch. Its initial review attempts were
canceled and are not evidence; this exact source commit must receive fresh
three-lane reviews. Source Human Approval has not been requested.

### DOC-01 follow-up

After the first Documentation review, the independent audit-appendix crosscheck
was corrected to fail closed when its D ledger is absent or unreadable. An
explicit missing D path no longer falls back to the default ZIP. Four
regressions cover complete B↔D success, absent D, invalid explicit path, and a
B↔D mismatch. The README now makes the full B/C/D verification the standard
procedure; reruns write to a separate output so the historical result remains
unchanged. `SHA256SUMS` was regenerated and verified. This correction does not
change the GWOSC selection or its validation results. The resulting frozen
source requires new three-lane reviews; none of the prior review states carry
forward.

## Recommendation

**HOLD.** Phase B/C validation is complete and the corrected S21 source is
frozen on the isolated branch. The initial review attempts were canceled; the
exact frozen source SHA must pass fresh independent Documentation, Release-Security, and
Scientific-Data-Model reviews before Source Human Approval is requested. No
Candidate R, Candidate Qualification, Final GO, tag, release, or PyPI action is
authorized by this audit.
