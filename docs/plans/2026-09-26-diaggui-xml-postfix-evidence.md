# DiagGUI XML post-fix evidence report

**Purpose:** Evidence-only follow-up based on the merged #730–#734 fixes. It records reader behavior for the cited fixtures and the post-fix matrix classification.

## Baseline and immutable archive

- Archived matrix: `docs/plans/2026-09-25-diaggui-xml-product-audit-matrix.csv`
- Archived SHA-256: `eaa3890fc4038fbb776f8a535cbba5c203d3a5fdcd53de05314d46fb7ae01202`
- Post-fix base: `76c2a6c4b68c153f89ea807d2bfcea2f0512f581` (main after all five issue PRs merged).
- Merged issue PRs: #737 (#730), #739 (#731), #740 (#732), #741 (#733), #738 (#734).
- The archived CSV is unchanged. The post-fix matrix is `docs/plans/2026-09-26-diaggui-xml-product-audit-matrix-postfix.csv`.
- All 663 cell IDs and their order are preserved. Exactly 35 rows have updated status, reason, and evidence reference; every other field and row is unchanged.

## Status summary

| Status | Archived | Post-fix |
|---|---:|---:|
| Verified | 31 | 62 |
| Observed | 29 | 18 |
| Unverified | 291 | 271 |
| N/A | 312 | 312 |

The 35 transitions are 31 rows to `verified` and 4 rows to `observed`:

- **#730 (3 verified):** `TimeSeries/TS:native:normalize:-`; `TimeSeries/TS:fallback:explicit:TimeSeriesDict`; `TimeSeries/TS:fallback:auto:TimeSeriesDict`.
- **#734 (2 verified):** `TimeSeries/TS:fallback:explicit:TimeSeriesMatrix`; `TimeSeries/TS:fallback:auto:TimeSeriesMatrix`.
- **#731 (14 verified, 2 observed):** For each `Spectrum/0` and `Spectrum/4`, external/native normalize; external/native explicit `FrequencySeries` and `FrequencySeriesDict`; fallback explicit `FrequencySeriesDict` are verified. The two fallback explicit `FrequencySeries` cells are observed: `Spectrum/0:fallback:explicit:FrequencySeries` and `Spectrum/4:fallback:explicit:FrequencySeries`.
- **#732 (10 verified):** For each `TransferFunction/1` and `TransferFunction/4`, external/native normalize and external/native/fallback explicit `FrequencySeriesMatrix`.
- **#733 (2 verified, 2 observed):** `TransferFunction/6:external:normalize:-` and `TransferFunction/6:native:normalize:-` are verified for the characterized route. External and native explicit `FrequencySeriesMatrix` cells are observed only: `TransferFunction/6:external:explicit:FrequencySeriesMatrix` and `TransferFunction/6:native:explicit:FrequencySeriesMatrix`.

The status update means the cited reader behavior was asserted for the cited fixture and route. It does not establish the full serialization format or a physical convention.

## Evidence and limits by issue

- **#730:** `tests/io/test_dttxml_issue730.py::test_native_parser_reads_source_grounded_timeseries` and `::test_native_product_loader_reads_timeseries` cover native TS normalization. `::test_public_read_in_real_no_dttxml_process` covers explicit and automatic `TimeSeriesDict` reads in a separate process without `dttxml`, including values, dtype, sample interval, epoch, and channel. Claims are limited to this fixture and these reader routes.
- **#734:** `tests/io/test_dttxml_issue734.py::test_matrix_read_in_real_base_only_process` exercises the no-`dttxml` `TimeSeriesMatrix` route for the tested XML and XML.GZ cases, both explicit and automatic. Assertions cover matrix type, values, and time coordinates; they do not establish other input forms or class behavior.
- **#731:** `tests/io/test_dttxml_issue731.py::test_fft_frequency_readers_preserve_both_channel_layouts` and `::test_fft_one_bin_frequency_axis_is_preserved` cover external/native normalization, direct `FrequencySeries`, and `FrequencySeriesDict` for the tested Spectrum/0 and /4 fixtures. `::test_fft_fallback_in_separate_no_dttxml_interpreter` covers fallback dict and direct-series reads. The fallback direct-series cells remain `observed`; all statements are scoped to the tested values, axes, phase, epoch, and channels.
- **#732:** `tests/io/test_dttxml_issue732.py::test_stf_matrix_reader_preserves_labeled_complex_row_and_axis` covers external/native matrix routes. `::test_stf_matrix_reader_uses_real_no_dttxml_interpreter` covers the base-only route. `::test_stf_reader_preserves_one_bin_axis` adds the one-bin case and direct normalized spacing assertion for subtype 1. Evidence is specific to these synthetic fixtures and asserted matrix behavior.
- **#733:** `tests/io/test_dttxml_issue733.py::test_external_tf6_preserves_or_fails_before_returning_real_only_data`, `::test_native_tf6_preserves_complex_values_axis_epoch_and_pair`, and `::test_tf6_result_index_is_independent_of_subtype` support the loader cells. `::test_tf6_preserves_near_uniform_serialized_axis_through_matrix` supports only the observed matrix-axis classification. Recovery is limited to a unique matching TF/6 Result, one row, and the characterized little-endian float64 frequency plus complex64 sample layout. Ambiguity, collisions, mismatches, malformed/unsupported layouts, and TF/6 Reference are refused or fail closed as recorded by the tests.

## Excluded claims

These synthetic fixtures characterize tested reader behavior only. They do not establish authoritative DiagGUI serialization semantics, real-export compatibility, physical units, transfer-function direction, calibration, conjugation, normalization, or universal endianness, precision, or dimensions. TF/6 support does not extend to arbitrary row counts/layouts or Reference results. The four observed cells above remain partial reader observations, not full product-contract verification. Other archived cells keep their existing status.

## Reproducibility and invariants

The post-fix matrix was generated from the archived CSV and checked as follows:

1. The source SHA-256 matched `eaa3890fc4038fbb776f8a535cbba5c203d3a5fdcd53de05314d46fb7ae01202` before and after the transformation.
2. Source and post-fix matrix each contain 663 unique IDs; the complete ID sequence is identical.
3. Exactly 35 rows changed, and only `status`, `reason`, and `evidence_ref` changed on those rows. Each changed row has a direct test-node reference and a reader-scoped reason.
4. The changed rows classify as 31 verified and exactly these 4 observed: the two fallback FFT direct `FrequencySeries` cells and the two TF/6 external/native matrix cells.
5. Post-fix status totals equal 62 verified, 18 observed, 271 unverified, and 312 N/A.
6. Post-fix matrix SHA-256: `35d51c144c1bcbab07d03541a2e6cd9048ba630a9e025b68b652af7479a3a3ee`.

Focused regression command run against the merged main base: 110 passed, 12 skipped.

```sh
rtk conda run -n gwexpy python -m pytest -q tests/io/test_dttxml_issue733.py tests/io/test_dttxml_issue726.py tests/io/test_dttxml_issue730.py tests/io/test_dttxml_issue731.py tests/io/test_dttxml_issue732.py tests/io/test_dttxml_issue734.py
```

Matrix checks can be repeated with `rtk sha256sum docs/plans/2026-09-25-diaggui-xml-product-audit-matrix.csv docs/plans/2026-09-26-diaggui-xml-product-audit-matrix-postfix.csv` and Python csv/hashlib/Counter: assert the source hash above, 663 unique IDs per file, identical ordered IDs, 35 changed rows, only status/reason/evidence_ref changed, 31 transitions to verified and 4 to observed, and the stated final counts. Post-fix matrix SHA-256: `35d51c144c1bcbab07d03541a2e6cd9048ba630a9e025b68b652af7479a3a3ee`.

The cited regression tests and matrix invariants were rerun against the merged main base before this local evidence commit.
