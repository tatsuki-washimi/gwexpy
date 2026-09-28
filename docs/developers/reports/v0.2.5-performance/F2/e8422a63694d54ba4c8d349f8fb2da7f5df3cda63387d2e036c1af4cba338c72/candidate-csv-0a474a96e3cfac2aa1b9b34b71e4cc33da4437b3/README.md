# F2 CSV candidate evidence

This is the CSV portion of F2. SDB concurrent-WAL behavior remains HOLD until
the separate B-F2-SDB baseline-v2 fault matrix is frozen. B1 already makes zero
unselected TDMS `read_data()` calls, so this candidate makes no TDMS selection
performance claim.

## Provenance

- Frozen F2 CSV baseline commit: `258c17357b9253a6ec2d809fa0adfa0b1f0be5fb`.
- Frozen harness digest: `e8422a63694d54ba4c8d349f8fb2da7f5df3cda63387d2e036c1af4cba338c72`.
- Frozen fixture-manifest SHA-256: `2b29c7bccedfc8335beb5c59580b26aa1a211bd3959e142dcb2e99a7dbcab1f8`.
- B1 source: `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`; controlled local wheel SHA-256: `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- Candidate package source: `0a474a96e3cfac2aa1b9b34b71e4cc33da4437b3`; controlled local wheel SHA-256: `81270a6a73052b80b65fb4e03af81bf813c45ef12a1cb5c16f97b3a3a8af07ad`.
- Structural source test commits: `96a9255e3f3a4d36c358af09c3ad0f8362070653` and `9a804abe541f215d7d9dda5335e677fa0b3331b3`; `git diff 0a474a96e 9a804abe5 -- gwexpy` is empty.
- Both wheels were built with `python -m build --wheel --no-isolation`, installed in isolated `--system-site-packages` venvs with `pip install --no-deps --no-index`, and audited by the frozen harness. Captures check the imported path/version, all installed package files, and dependency-version equality. The ABBA run order and raw samples are stored below.

## Correctness and exact work

- The five normal routes (FrequencySeries explicit and `.csv` auto detection, all-channel TimeSeries CSV, selected large CSV, and the internal enhanced writer) plus 16 malformed CSV routes have **21/21 B1-equal fingerprints**. These include numerical values and axes, dtype, unit, metadata, warning category/message, exception type/message, and writer output-byte hash.
- Both #394 FrequencySeries routes retain the first CSV column as the nonuniform frequency axis and meet the small-input non-regression gate.
- Route ③ retains unselected-column validation and its old warning/exception order. The frozen B1 local-variable spy cannot see the candidate's renamed parser locals (`csv_parser_local_probe_covered=false`). A source-level return-frame test, `test_large_selected_parser_materializes_one_payload_column`, checks a 65,536-row, 17-column input: `selected_values` holds **one 65,536-value Python column**, the converted `selected` mapping holds **one 65,536-value NumPy column**, full-row buffers **0**, full-column matrices **0**. The list and NumPy array can coexist briefly at parser return; this is two copies of the selected column, not all input columns. Source audit confirms each row's unselected tokens are converted for validation and discarded before the next row. The frozen B1 spy observed 17 columns and 1,114,112 values retained. Linux PSS below corroborates the reduction. Route ② materializes all requested columns, so it makes a parser-speed claim only.
- The internal enhanced writer has full-output buffers **1 → 0**, maximum `write()` argument **26,214,510 → 50 bytes**, and byte-identical output on the frozen fixture. This route does not support a claim about public `TimeSeries.write(format="csv")`, which uses GWpy's writer.
- Parsed temporary matrix chunks are capped at 64 MiB. An indivisible final output row wider than 64 MiB necessarily creates a larger final output array; its conversion temporaries are split into bounded column slices. `test_wide_row_converts_in_bounded_temporary_chunks` checks this boundary.

## Noisy evidence

Nine interleaved warm samples per arm; medians and MADs are in nanoseconds:

| Route | Metric | B1 median / MAD | Candidate median / MAD | Relative result |
| --- | --- | ---: | ---: | ---: |
| ①a explicit | wall | 38,532,399 / 1,617,344 | 38,574,237 / 1,586,414 | 0.1% slower; non-regression green |
| ①b auto | wall | 40,845,414 / 2,174,634 | 40,035,216 / 499,577 | 2.0% faster; non-regression green |
| ② all channels | wall | 34,068,874 / 2,948,014 | 13,544,302 / 299,397 | 60.2% faster; `2 × relative MAD = 21.7%` |
| ③ selected large | wall | 654,145,551 / 57,864,531 | 386,922,841 / 3,907,197 | 40.9% faster; `2 × relative MAD = 19.7%` |
| internal writer | wall | 391,577,371 / 5,973,025 | 355,544,417 / 10,584,615 | 9.2% faster; `2 × relative MAD = 9.0%` |

Linux memory was sampled in separate runs, five samples per arm. Values are
simultaneous process-tree peak PSS in KiB (median):

| Route | B1 | Candidate | Result |
| --- | ---: | ---: | ---: |
| ② all channels | 314,314 | 303,561 | 3.4% lower; supporting only |
| ③ selected large | 502,353 | 326,227 | 35.1% lower; gate green |
| internal writer | 412,603 | 308,161 | 25.3% lower; gate green |

Cold measurements include import and process startup. They do not establish a
wall-time improvement for this CSV change; route ③ cold CPU is 26.2% lower.
Full raw samples, summaries, manifests, and file hashes are stored alongside
this report. Instrumented structural runs and uninstrumented resource runs
were separate. Independent CSV re-review found no remaining correctness
blocker and accepted the source-level selected-materialization probe with
the explicit final-output-array exception above.

Focused checks: 136 CSV/registry tests passed, including the large selected
materialization probe. Ruff and MyPy passed for the candidate source. The
final combined test gate is recorded in the candidate manifest after the
evidence bundle is assembled.
