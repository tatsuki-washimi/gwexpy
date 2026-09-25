# DiagGUI XML product/subtype compatibility audit

Date: 2026-09-25
Base: `origin/main` at `27ea739061bc0ac154d9655c5bf8943c67b214bb`
Reviewer status: approved
Umbrella issue: [#729](https://github.com/tatsuki-washimi/gwexpy/issues/729)

## Goal and boundaries

Keep the completed Issue #726 ASD/TF repair intact. Audit DiagGUI XML product normalization, reader construction, return types, data and coordinate metadata, registry registration, and class-specific auto-identification. Characterize the public `TimeSeriesDict.read()` contract and the registered frequency-reader surfaces without promoting them to public direct I/O. This audit creates evidence and regression characterization, not parser or contract fixes.

Multi-file concatenation, `start`/`end`, performance, writers, and general timezone policy are outside this audit. Include minimal `channels`, `rows`, `cols`, and `pairs` filters to check channel and pair mapping. Do not publish the real DiagGUI export, raw samples, or sensitive channel metadata.

## Critical path

1. Confirm `origin/main` against the base SHA before creating the issue. If DiagGUI-related code changed, revise this plan before any issue or worker action.
2. Open the umbrella audit issue immediately after the plan is saved, before worktrees or worker changes. Link Issue #726 without changing its closed state.
3. Freeze the cell-ID universe in a machine-readable matrix before A/B start. Include all 16 rows (`Spectrum/0..7`, `TransferFunction/0..7`), every backend and reader-surface candidate, and reasoned N/A cells. The row set may not silently grow or shrink. Track TS separately because it is a distinct XML Type and public reader contract.
4. Run A and B in parallel isolated worktrees; run C after B fixtures are verified; integrate A, B, and C patches separately; then report, independently review, update the issue, split confirmed defects into follow-up issues, and close the audit issue once its acceptance criteria are met.

Cell IDs use `TYPE/SUBTYPE:BACKEND:SURFACE:CLASS`, for example `Spectrum/4:external:normalize:-` or `Spectrum/4:native:explicit:FrequencySeries`. `BACKEND` is `external`, `native`, or `fallback`; `SURFACE` is `normalize`, `explicit`, or `auto`; `CLASS` is one of `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`, `FrequencySeries`, `FrequencySeriesDict`, `FrequencySeriesMatrix` for reader cells. A separate class-level table records reader registration and identifier registration. The fixed matrix includes all Type/Subtype × backend × candidate-surface combinations; `N/A` cells carry a reason, including the absence of a reader-level `native=` selector for TS.

The fixed cell set is in `docs/plans/2026-09-25-diaggui-xml-product-audit-matrix.csv` (663 unique cells: 16 frequency subtype rows plus TS, 39 cells each; initial SHA-256 `9aa46960e1c3efdb613257f2c3c4d730218ecec92bc2161ba3b7949d945a5889`). The six-class registration set is in `docs/plans/2026-09-25-diaggui-xml-product-audit-registry.csv` (initial SHA-256 `308a9cddb91a96f7a65a409afff74744ca0018c1aea3bb2e990db224ae583d37`). The matrix has 312 initial reasoned N/A cells; all other statuses remain unclassified until investigation. Preserve the cell-ID set when updating values.

For each cell, record the declared token, source-backed product meaning, Array Type, endianness, dimensions, axis layout, current behavior, output type, dtype, shape, frequency/time coordinates, epoch, pair direction, phase, warnings/errors, evidence, and one status: `verified`, `observed`, `unverified`, `source-backed unsupported/invalid`, or `N/A`. A source-backed invalid claim requires an identified authoritative source; native implementation gaps are not invalid XML.

## Lanes and model/effort

| Lane | Role and deliverable | Exclusive write scope | Dependencies and verification | Model / effort |
|---|---|---|---|---|
| A | Contract investigator: TS base-only trace and FFT/STF token reproductions | `tests/io/test_dttxml_base_contract_audit.py` | Confirm `find_spec('dttxml') is None` in a clean base process; trace contract → loader route → normalized payload → `TimeSeriesDict.read()` | `gpt-6-luna` / high |
| B | Serialization investigator: inventory all subtype candidates, create only source-grounded fixtures | `tests/io/dttxml_audit/conftest.py`, `tests/io/dttxml_audit/test_subtypes.py` | Compare real/authoritative evidence, installed parser source and decode, existing verified fixtures, then uniquely derived synthetic streams; non-autouse local fixtures only | `gpt-6-luna` / max |
| C | Reader investigator: CSD matrix, other applicable readers, filters, metadata, compression, class-level auto-identification | `tests/io/dttxml_audit/test_readers.py` | Starts after B verification; may apply B patch to its worktree for testing but exports only C-owned file | `gpt-6-luna` / high |
| I | Integrator: inspect and combine A/B/C evidence | Isolated integration worktree | Apply A → B → C as three individually inspected patches; run focused and I/O gates | `gpt-6-luna` / high |
| D | Reporter: completed matrix, red reproductions, issue update and follow-up drafts | This plan, its matrix/registry CSV files, and its `~/.claude/plans/` copy | Starts after I; verify each cell against evidence | `gpt-6-luna` / high |
| V | Independent evidence auditor | Read-only | Starts after D; inspect matrix, patch hashes, tests, and issue claims | `gpt-6-luna` / max |

For each worker patch, include new files using intent-to-add in that isolated worktree, export canonical `git diff --binary --no-ext-diff` output restricted to the lane's owned files, and record its SHA-256. C's exported patch must exclude B's dependency patch. I and V compare the recorded digest with the exact patch bytes before inspection and application. Do not create commits.

## Evidence and verification

Evidence priority: (1) real DiagGUI export or authoritative DTT serialization, (2) installed `dttxml` parser source plus actual decode output, (3) verified Issue #726 fixture and historical finite-value behavior, (4) an independent synthetic fixture uniquely derived from that evidence. Inventory a subtype even if no fixture can be justified; mark it `unverified` rather than inventing serialization.

Run the installed external parser, explicit native helper, and no-`dttxml` fallback in separately initialized processes/environments. Record Python and `dttxml` versions and prove package absence in the base-only process. TS `TimeSeriesDict.read()` has no explicit native-selection path, so that reader/backend cell is `N/A`; the low-level native loader remains a separate observation. For each class, check reader registration, identifier registration, explicit `format="xml.diaggui"` read, and `format=None` read separately. In particular, inspect `TimeSeriesMatrix` despite its reader being registered without a DiagGUI identifier.

Use source-grounded, finite raw samples to check values, dtype, shape, axes, epoch, `(ChannelB, ChannelA)` labels, complex phase, and return type. Cover applicable `float`/`double`/complex storage, both endian forms, 1-D/2-D dimensions, uniform/nonuniform frequencies, multiple `ChannelB[n]`, `Reference[n]`, missing `t0`, and actual `.xml.gz` decoding. Do not infer PSD→ASD square roots, FFT normalization, CSD conjugate convention, or a general nonfinite-value rejection policy without independent evidence.

Keep permanent audit tests green for verified positive behavior and parser/fixture characterization. Capture desired-but-unimplemented behavior as red commands and outputs in the report and follow-up issues; do not freeze current defects as desired contracts or leave failing/xfail tests in the merged audit set. Integration gates: targeted new and Issue #726 tests, `tests/io/`, `io-contract --no-fixtures`, `io-conformance`, Ruff, and `git diff --check`, using the project `gwexpy` conda environment for pytest/Ruff.

## Checklist and completion

- [x] **Step 1: Recheck the base SHA and relevant origin diff; save this plan and synchronized copy.** `origin/main` remained at the recorded SHA.
- [x] **Step 2: Create the umbrella issue and freeze the matrix cell IDs, including all 16 required subtype rows and reasoned N/A cells.** Issue #729 was opened before worker changes and closed after Step 7; the matrix has 663 unique cells and 312 initial reasoned N/A cells.
- [x] **Step 3: Run A and B in separate worktrees; inspect each patch and record its SHA-256.** A: `0dd3b58b271560ef50ce9887f6648091be2e7fd459cb568b6ee86e1695662dbc`; B: `3f80113fba4f97b9bddd14cbd76c2667ae5ffbc8629540b1c07e824fbb895816` (canonical `git diff --binary --no-ext-diff` patches).
- [x] **Step 4: Apply B as a dependency to C; inspect C-only patch and its SHA-256.** C: `6047f7c109359ee48b405199d693f7765c3a7d6f85353d3c190ba3a9c2643552` (canonical C-only `git diff --binary --no-ext-diff` patch).
- [x] **Step 5: Integrate A, B, C in order and run the specified gates.** Focused: 36 passed; `tests/io/`: 1290 passed, 23 skipped; `io-contract --no-fixtures`: 1545 passed, 24 skipped, 1 deselected; `io-conformance`: 71 passed, 7 skipped; Ruff and `git diff --check` passed.
- [x] **Step 6: Complete the matrix/report and synchronize the plan copy; obtain V's read-only verdict.** V approved the corrected canonical patch hashes, all 663 classified cells, six-class registry, and focused reproductions; 31 verified, 29 observed, 291 unverified, 312 N/A.
- [x] **Step 7: Update the umbrella issue, create or link distinct follow-up issues for each confirmed defect, and close the umbrella issue once every applicable cell is classified, every unverified cell identifies its evidence gap, and no confirmed defect lacks reproducible evidence and a tracking issue.** Follow-ups: #730 (TS base contract), #731 (FFT), #732 (STF), #733 (TF/6 phase), #734 (TimeSeriesMatrix auto-identification). Issue #729 was updated with the final summary and closed; #726 remains closed.

The issue is an audit tracker, not a promise to implement every product in this pass. Issue #726 remains closed. No PR, commit, release, or parser fix is part of this plan.

## Evidence report (lane D)

The frozen 663 IDs and header are unchanged. Every cell now has `product_claim`, one of the five allowed statuses, a reason, and an `evidence_ref`; the 312 original N/A cells retain their reasons. Final status counts: 31 `verified`, 29 `observed`, 291 `unverified`, 312 `N/A`, and 0 source-backed unsupported/invalid. No authoritative DiagGUI serialization source establishes any audited subtype as invalid. Native parser omissions are recorded as observed implementation behavior, not invalid XML. Each unverified cell names the missing route or variant evidence.

### Product claims and N A rationale

#### Subtype survey

The source survey treats Type/Subtype as the product selector and Array Type as the storage selector. `product_claim` records product, payload (Y or embedded `(f,Y)`), and surveyed sample precision. Inventory: Spectrum/0 FFT complex64 Y; /1 PSD float32 Y; /2 CSD complex64 Y; /3 COH float32 Y; /4 FFT complex64 `(f,Y)`; /5 PSD float32 `(f,Y)`; /6 CSD complex64 `(f,Y)`; /7 COH float32 `(f,Y)`; TransferFunction/0 TF complex64 Y; /1 STF complex64 Y; /2 COH float32 Y; /3 TF complex64 `(f,Y)`; /4 STF complex64 `(f,Y)`; /5 COH float32 `(f,Y)`; /6 TF with float64 frequency values and complex64 samples; /7 COH float64 `(f,Y)` observed. TS is a time-domain sample stream. The survey uses installed `dttxml` 1.1.8 source and actual decode, with verified #726 fixtures where noted; it does not establish every serialization variant.

Frequency XML cells targeting TimeSeries, TimeSeriesDict, or TimeSeriesMatrix remain N/A because they are time-domain classes. TS cells targeting frequency classes remain N/A because TS is time-domain. TS native-reader cells remain N/A because the public time-series reader has no `native=` selector; low-level native normalization is reported separately. No subtype/class mismatch was reclassified as N/A after freezing the matrix. N/A rows retain a product claim and reason.

### TS contract trace

The contract manifest records `TimeSeriesDict.read` as available in the base install and lists `dttxml` only in the `gw` extra (`xml.diaggui` optional-dependency list is empty). In clean base Python 3.11.16, `find_spec('dttxml') is None`. The public reader calls `load_dttxml_products`; without the package it invokes the native fallback. A TS-only Type XML is not normalized by that frequency-focused native parser, so normalization is `{}` and the public result is an empty TimeSeriesDict with a missing-package warning. Explicit external `dttxml==1.1.8` reading returns float32 `[1.5, -2, 0.25, 8]`, `dt=0.125`, and `t0=1234567890.25`. Positive characterization: `tests/io/test_dttxml_base_contract_audit.py::test_external_parser_public_timeseriesdict_read`; the base-only probe output and route trace are recorded in this report and A's investigation. This documents a base contract gap, not a general dependency policy.

### FFT STF and other subtypes

Installed parser source and decode identify Spectrum/0 as FFT Y and TransferFunction/1 as STF Y. The external parser produces those result objects, but `load_dttxml_products` does not add FFT or STF keys; native subtype tables also omit those tokens. Positive tests do not assert this omission as desired behavior. Spectrum/4–7 and TransferFunction/2,4,7 were inventoried from source and actual decode where independently available; cells without a direct class/surface assertion remain unverified. #726 fixtures characterize Spectrum/1–3 and TransferFunction/0,3,5. No PSD→ASD square root, FFT normalization, CSD conjugation rule, broad endian/dimension claim, or general nonfinite-value policy is inferred.

### Registry observations

The integrated CSD test exercises Spectrum/2 through native and external normalization and explicit/automatic FrequencySeriesMatrix reads, asserting shape `(1,1,3)`, complex64 values and phase, frequencies, epoch, row `K1:TEST-OUTPUT`, column `K1:TEST-INPUT`, and pair filtering. Native gzip XML and missing `t0` (epoch zero) are covered. An additional clean no-dttxml run confirms the default fallback on the same source-grounded CSD fixture for explicit format and format=None, preserving shape, complex64 phase, pair labels, frequency axis, and epoch while emitting the fallback warning. The class table distinguishes reader registration, identifier registration, and actual `format=None` reads. All six classes have xml.diaggui/dttxml readers. FrequencySeries, FrequencySeriesDict, and FrequencySeriesMatrix have xml.diaggui identifiers; TimeSeries and TimeSeriesDict have xml.diaggui identifiers; TimeSeriesMatrix has no identifier. FrequencySeriesMatrix CSD auto-read is verified under external, native, and implicit fallback routes. FrequencySeries and FrequencySeriesDict auto product reads were not directly exercised.

### Base-only CSD fallback probe

Using the fixed Spectrum/2 CSD fixture layout in the clean Python 3.11.16 environment (`HAS_DTTXML=False`), `FrequencySeriesMatrix.read(path, products="CSD")` succeeds both with `format="xml.diaggui"` and `format=None`, leaving `native` unspecified. Both routes return shape `(1,1,3)`, complex64 `[(1+2j),(-0.5+0.25j),(3-4j)]`, phases approximately `[1.1071488, 2.6779451, -0.9272952]`, frequencies `[17.5,20,22.5]`, epoch `1234567890.25`, row `K1:TEST-OUTPUT`, and column `K1:TEST-INPUT`. The implicit fallback emits the documented warning that dttxml is unavailable and the native parser is used. The matrix upgrades only those exact fallback normalize/reader cells to verified.

### TSM TS-only probe

A deterministic synthetic XML with `LIGO_LW Type="TimeSeries"`, `Subtype=0`, `N=4`, `dt=0.125`, channel `K1:AUDIT-TS`, `t0=1234567890.25`, Array Type `float`, and little-endian float32 values `[1.5,-2,0.25,8]` was read against the integrated checkout. `TimeSeriesMatrix.read(path, format="xml.diaggui", products="TS")` succeeded and returned shape `(1,1,4)`, float64 values, generic `row0`/`col0` labels, and the epoch. Dtype and labels are observations, not asserted contracts. With the same file, `format=None` failed with `ValueError: Could not identify format for /tmp/auditts_for_tsm.xml. Please specify 'format' argument.` The registry has a reader but no identifier, directly confirmed by this read failure.

Reproduce by using the TS-only fixture in `tests/io/test_dttxml_base_contract_audit.py` (or the equivalent fields above) and calling `TimeSeriesMatrix.read` once with explicit format and once with `format=None`. Explicit read returns the observations above; auto read raises the format-identification error. This does not establish a preferred matrix dtype or label policy.

Minimal synthetic fixture generation and read command (run from the integrated checkout):

```python
import base64, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from gwexpy.timeseries import TimeSeriesMatrix

root = ET.Element("LIGO_LW")
result = ET.SubElement(root, "LIGO_LW", {"Name": "Result[0]", "Type": "TimeSeries"})
for name, value in {"Subtype": "0", "N": "4", "dt": "0.125", "Channel": "K1:AUDIT-TS"}.items():
    ET.SubElement(result, "Param", {"Name": name}).text = value
ET.SubElement(result, "Time", {"Name": "t0"}).text = "1234567890.25"
array = ET.SubElement(result, "Array", {"Type": "float"})
ET.SubElement(array, "Dim").text = "4"
values = np.array([1.5, -2.0, 0.25, 8.0], dtype="<f4")
ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = base64.b64encode(values.tobytes()).decode()
path = Path(tempfile.gettempdir()) / "audit_tsm_ts.xml"
ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
matrix = TimeSeriesMatrix.read(path, format="xml.diaggui", products="TS")
print(matrix.shape, matrix.dtype, matrix.value.tolist(), list(matrix.rows), list(matrix.cols), float(matrix.t0.value))
TimeSeriesMatrix.read(path, format=None, products="TS")
```

Output before the expected exception: `(1, 1, 4) float64 [[[1.5, -2.0, 0.25, 8.0]]] ['row0'] ['col0'] 1234567890.25`; final call raises the format-identification `ValueError` above.

### Base-only TS auto probe

With that TS-only XML in the same no-dttxml environment, `TimeSeries.read(path, format=None, products="TS")` raises `ValueError: No channels found in xml.diaggui`; `TimeSeriesDict.read(path, format=None, products="TS")` raises `ValueError: cannot calculate span for empty TimeSeriesDict`. Both are preceded by the missing-dttxml fallback warning. The explicit `TimeSeriesMatrix.read(path, format="xml.diaggui", products="TS")` route raises `ValueError: No timeseries provided to align.` The matrix updates these specific fallback cells to observed; they are failures on synthetic compatible TS input, not XML-invalid claims.

### TF/6 phase

The installed parser source and actual decode show the mixed TF/6 stream: float64 frequency column followed by complex64 samples. Native decoding retains complex phase. Installed `dttxml` 1.1.8 returns float32 real-only samples for the surveyed stream, losing phase. Evidence is limited to the observed serialization and does not validate other mixed precision layouts. No permanent test blesses the phase loss.

### Reproducers and coverage gaps

All reproductions use finite synthetic values; no real export or raw channel data is included.

* Base-only TS: in clean Python 3.11.16, assert `find_spec('dttxml') is None`, then run the public TS reader with warnings enabled. Explicit TimeSeriesDict reading returns an empty mapping; `format=None` TimeSeries and TimeSeriesDict raise the errors recorded above; explicit TSM raises the alignment error. Each run warns that dttxml is unavailable and native fallback is used. External dttxml returns the four float32 samples, `dt=0.125`, and the stated epoch.
* Base-only CSD: run the Spectrum/2 fixture through `FrequencySeriesMatrix.read(path, products="CSD")` with omitted native selector. Both explicit xml.diaggui and format=None return phase-preserving complex64 matrix data and matching frequency, epoch, and pair coordinates as detailed above.
* FFT/STF: build Spectrum/0 and TransferFunction/1 blocks from the surveyed layouts, inspect `dttxml.DiagAccess(path).results`, then call `load_dttxml_products(path, native=False)` and with `native=True`. Observed: parser has FFT/STF results, while neither normalized mapping has those keys. This is red desired behavior, not a passing regression assertion.
* TF/6: use a synthetic mixed stream with an eight-byte float64 frequency prefix and interleaved complex64 samples. Compare parser-decoded values with external and native `load_dttxml_products`. Observed external values are float32 real parts while native values are complex64 with phase.
* TSM: the TS-only fixture succeeds with explicit format and fails with `format=None` as described above.

#### Coverage gaps

Unverified cells remain where the exact subtype × backend × class × surface or a declared dtype/dimension/endian/frequency-axis variant lacks direct evidence. In particular, most frequency fallback reader surfaces beyond Spectrum/2 CSD were not run in a clean no-dttxml environment; many subtype/class auto-read combinations and alternate class conversions lack direct probes; TF/6 has limited independent serialization; several `(f,Y)` layouts lack verified fixtures; and positive TimeSeries/TimeSeriesDict auto reads were not both exercised. The matrix records a specific reason and evidence reference for every such cell.

Tracked follow-ups: #730 covers the base-install TS contract; #731 covers FFT; #732 covers STF; #733 covers TF/6 phase preservation; and #734 covers TimeSeriesMatrix auto-identification. #415 is closed and has a different TS TypeError root cause, while #589 concerns performance. The umbrella issue #729 carries the final matrix summary and is closed.

### Integration evidence

Canonical binary patch SHA-256: A `0dd3b58b271560ef50ce9887f6648091be2e7fd459cb568b6ee86e1695662dbc`; B `3f80113fba4f97b9bddd14cbd76c2667ae5ffbc8629540b1c07e824fbb895816`; C `6047f7c109359ee48b405199d693f7765c3a7d6f85353d3c190ba3a9c2643552`.

Integrated gates reported by lane I: focused tests 36 passed; `tests/io/` 1290 passed, 23 skipped; `io-contract --no-fixtures` 1545 passed, 24 skipped, 1 deselected; `io-conformance` 71 passed, 7 skipped; Ruff and `git diff --check` passed. Issue #726 remains closed. The audit adds positive characterization and leaves observed defects as report evidence.
