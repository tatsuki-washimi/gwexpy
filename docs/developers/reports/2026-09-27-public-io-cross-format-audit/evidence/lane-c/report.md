# Lane C: HDF5 collection integrity audit

Source: `ae10234a1e37853508c54901bf7c9e80878f25aa` (unchanged worktree source). No commits or package/test/doc edits.

## Reproduction

From this worktree:

```bash
rtk proxy env PYTHONPATH=/home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/C-hdf5-collections /home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/envs/present/bin/python .audit_tmp/probe.py > .audit_tmp/present.jsonl 2> .audit_tmp/present.stderr
rtk proxy env PYTHONPATH=/home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/C-hdf5-collections /home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/envs/base/bin/python .audit_tmp/probe.py > .audit_tmp/base.jsonl 2> .audit_tmp/base.stderr
```

Both returned exit 0, 33 JSON lines (version header plus 32 scenarios), empty stderr. Both report Python 3.11.14, GWexpy 0.2.4, GWpy 4.0.2, h5py 3.16.0, NumPy 1.26.4, Astropy 6.1.7. Outcome signatures match in both environments.

## Fixture authority

The writer only seeds a valid A/B/C skeleton. h5py independently verifies the root manifest and physical entries, then rewrites C1 order to `[C,B,A]` or `[2,1,0]` and dict keymap to `logical-A/B/C`. The C1 control public read honors those independent manifest changes. C2 is a distinct copy with all five `gwexpy_*` collection manifest attributes removed. Each has an uncorrupted public control read except TS Dict C2 group. Finally h5py changes only B's `x0` from numeric to `audit-invalid-axis` for TS/FS/SPEC, or only B's `unit` to `audit-invalid-unit-@???` for HIST. A/C raw attributes remain equal; their values were never written after the seed. B's public single-entry read fails (`TypeError` for TS/FS/SPEC; `ValueError` for HIST). Thus no partial collection is expected for C1: its explicit manifest still lists B, which is unreadable.

## Result matrix

Notation: `C,A` means successful C1 dict result with exactly `logical-C,logical-A`; `C,A→0,1` means successful C1 list result with names C,A at indices 0,1. C1 controls have C,B,A in the same logical order. C2 dict `A,C` is physical discovery order; C2 list `A,C→0,1` has names A,C. `err` is an explicit exception. `D`/`G` are requested dataset/group layouts. HIST `D` actually stores root groups and marks the manifest dataset-per-entry, so it is not an independent physical dataset layout.

| Public entrypoint | C1 D | C1 G | C2 D | C2 G | Public route |
| --- | --- | --- | --- | --- | --- |
| `TimeSeriesDict.read(path, format="hdf5")` | C,A | C,A | err `TypeError` | err `ValueError` even control | C1 custom loop; C2 GWpy parent delegate |
| `TimeSeriesList.read(path, format="hdf5")` | C,A→0,1 | C,A→0,1 | A,C→0,1 | A,C→0,1 | Direct custom loop |
| `FrequencySeriesDict.read(path, format="hdf5")` | C,A | C,A | A,C | A,C | Direct custom loop |
| `FrequencySeriesList.read(path, format="hdf5")` | C,A→0,1 | C,A→0,1 | A,C→0,1 | A,C→0,1 | Direct custom loop |
| `SpectrogramDict().read(path, format="hdf5")` | C,A | C,A | A,C | A,C | Instance method custom loop |
| `SpectrogramList().read(path, format="hdf5")` | C,A→0,1 | C,A→0,1 | A,C→0,1 | A,C→0,1 | Instance method custom loop |
| `HistogramDict.read(path, format="hdf5")` | C,A | C,A | A,C | A,C | Direct custom loop |
| `HistogramList.read(path, format="hdf5")` | C,A→0,1 | C,A→0,1 | A,C→0,1 | A,C→0,1 | Direct custom loop |

`TimeSeriesDict.read(path)` auto route also returns logical-C,logical-A for both C1 layouts; C2 dataset raises `TypeError`; C2 group raises `ValueError`. C1 controls all returned C,B,A according to the independently rewritten manifest; all single-entry B reads failed after mutation. C2 controls returned A,B,C except TS Dict group, which raises `ValueError` (`cannot calculate span for empty TimeSeriesDict`), so that cell cannot establish a valid legacy discovery read.

## Classification and issue titles

- `HDF5-TS-COLL-001`: **CONFIRMED_CORRECTNESS_DEFECT**, `silent_entry_loss`, C1 dict/list, both layouts. Suggested: “Fail HDF5 manifest collection reads when a listed TimeSeries entry is unreadable.” C2 list skip is **CONFIRMED_COMPATIBILITY_BEHAVIOR**; C2 dict dataset fail-closed is **CONFIRMED_INTENTIONAL_BEHAVIOR** (GWpy route). C2 dict group entry-loss question is **BLOCKED**, `NO_VALID_FIXTURE` (uncorrupted control fails).
- `HDF5-FS-COLL-001`: **CONFIRMED_CORRECTNESS_DEFECT**, `silent_entry_loss`, C1 dict/list, both layouts. Suggested: “Report unreadable manifest-listed FrequencySeries entries instead of omitting them.” C2 skip is **CONFIRMED_COMPATIBILITY_BEHAVIOR** under historical tolerant discovery behavior.
- `HDF5-SPEC-COLL-001`: **CONFIRMED_CORRECTNESS_DEFECT**, `silent_entry_loss`, C1 dict/list, both layouts. Suggested: “Fail Spectrogram collection reads when a manifest-listed entry cannot be reconstructed.” C2 skip is **CONFIRMED_COMPATIBILITY_BEHAVIOR**; provenance sidecar errors have a separate explicit rethrow path.
- `HDF5-HIST-COLL-001`: **CONFIRMED_CORRECTNESS_DEFECT**, `silent_entry_loss`, C1 dict/list, physical group layouts. Suggested: “Reject partial Histogram collections when a manifest-listed histogram is unreadable.” C2 skip is **CONFIRMED_COMPATIBILITY_BEHAVIOR**. Requested `dataset` layout is physically a group, so no physical histogram dataset case was available; **BLOCKED**, `NO_VALID_FIXTURE` for a true dataset-per-entry Histogram layout.

## Root cause and authority

`gwexpy/io/hdf5_collection.py` provides shared layout/order/keymap helpers. It does not catch entry errors. Each concrete reader has its own `except (KeyError, ValueError, TypeError, OSError): continue` path; group layout also tries a fallback entry path before continuing. TimeSeriesDict first detects root collection attributes: C1 enters its custom loop; C2 delegates to GWpy. Spectrogram explicitly rethrows `ProvenanceSidecarError` before the generic skip. HIST's writer labels `layout="dataset"` in the manifest while the entry writer creates groups.

Authority 1: `docs/developers/contracts/public_io_contract.json` publishes direct HDF5 reads for all eight classes and requires collection manifest key maps and entry order to be preserved. Authority 2: raw HDF5 files retain B and their C1 manifests list B after mutation. Authority 3: GWpy-backed public TS Dict C2 dataset raises on B, but this is an error case, not finite correct output under the GWpy parity rule. Authority 5: historical `docs/developers/plans/archive/contract-audits/2026-04-27-collection-api-contract-audit.md` explicitly records tolerant reads as existing behavior, so C2 discovery is classified separately. Its lower-level historical policy does not authorize silently losing an explicit C1 manifest entry under the published contract.

No fix applied.
