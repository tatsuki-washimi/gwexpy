# SeriesMatrix #637 B1 decision packet

Status: deferred. `adopted: false`.

The candidate was evaluated in the isolated `v020-seriesmatrix-candidate`
worktree. No #637 composition candidate runtime was copied into integration.
Integration retains the approved Phase A `SpectrogramMatrix` dimensional
raw-ndarray add/sub atomic-`TypeError` runtime contract change. The integration
runtime is therefore not literally frozen B0; the B0 benchmark and fallback
semantics remain frozen, and the candidate remains `adopted: false`.

## Evidence and protocol

- Raw evidence: `series_matrix_b1.json`.
- Raw evidence SHA-256:
  `6b1fac847052d1e814f2f5501f9eed329d876a03cf67e6f637d65acc804bbd8e`.
- Fixed origin: `origin/main` at `6a13900672900551ccaf1b18fe78b9ce6f062e29`.
- Protocol: 3 warmups, 7 independent child processes, measured batches of at
  least 250 ms, and at most 3 attempts per operation.
- Environment: Python 3.12.12, NumPy 2.3.5, Astropy 7.2.0, GWpy 4.0.1,
  GWexpy 0.1.14, Linux 6.17.0-1030-oem x86_64.
- Candidate runtime files, sorted and authoritative against the fixed SHA:
  `gwexpy/spectrogram/matrix.py`,
  `gwexpy/types/series_matrix_indexing.py`,
  `gwexpy/types/seriesmatrix_base.py`.
- Candidate runtime-file SHA-256:
  `e5acf6ce7ce87fd1d0986c5cfa094f709cfff82fbbc2934fb7df080e1cab227f`.
- Frozen B0 SHA-256 confirmed:
  `ac856b9ffab86c702cb1d66a8cae7f8a826b6928eb2119a0fbf1ad73f87da01c`.

The B1 JSON retains the required machine-readable fields
`candidate_evidence.decision: pending` and
`candidate_evidence.issue_637: not evaluated in B1`. This Markdown file is the
decision record.

## Benchmark results

Values are final-attempt median and MAD in seconds. All B1 operations were
stable. The B0 `slice` operation remained unstable across its allowed attempts.

| Operation | B0 median | B0 MAD | B0 stability | B1 median | B1 MAD | B1 stability |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `asarray` | 2.02341685153e-7 | 9.99259517904e-9 | stable | 4.19973208137e-7 | 1.66484049372e-8 | stable |
| `construct` | 1.16424903110e-4 | 2.64983477461e-6 | stable | 1.17915355500e-4 | 1.21690222618e-6 | stable |
| `copy` | 1.49389410283e-4 | 2.37037008031e-6 | stable | 1.50360224820e-4 | 1.81540373774e-6 | stable |
| `multiply` | 3.66493961897e-4 | 1.10938101144e-5 | stable | 3.56910807339e-4 | 1.37363997548e-5 | stable |
| `quantity_left_multiply` | 4.38234501785e-4 | 1.02094935578e-5 | stable | 4.34800345743e-4 | 8.70892131994e-6 | stable |
| `slice` | 5.04560595835e-5 | 4.79479238366e-6 | **unstable** | 8.91179753497e-5 | 1.18778169940e-6 | stable |

The unchanged benchmark validator accepted the B1 JSON and the authoritative
runtime-file set. `adopt_candidate(B0, B1)` returned:

```text
passed=False
stability_gate_passed=False
unstable_operations=('slice',)
```

Because B0 has an unstable `slice`, adoption is false regardless of the
candidate timing. No numeric timing comparison is used as an adoption claim.

## Compatibility and test results

| Area | Result | Notes |
| --- | --- | --- |
| Candidate TDD tests | 13 passed | Composition identity, `sqrt`, Quantity-left multiplication, unit rules, unsupported ufunc forms, and atomic failure paths. |
| Candidate structure/indexing/math tests | 122 passed | Existing candidate tests. |
| Frozen Phase A manifest | 27 passed | Run with the candidate root on `PYTHONPATH`; all 318 ledger cells pass. |
| Filtered Spectrogram compatibility run | 389 passed, 2 failed, 5 skipped, 2 deselected, 1 xfailed | The two typed-view cases are excluded; the two failures are the genuine reflected Quantity-left addition limitations documented below. |
| Legacy operator tests | 339 passed, 33 failed | 30 direct-ufunc rejection assertions are intentionally superseded by B1 ufunc support; 3 `test_modulo_and_floor_divide_do_not_ignore_units` cases remain legacy reflected-operand failures. |
| Ruff | passed | Candidate runtime and candidate tests. |
| Scoped mypy | passed | Candidate runtime files. |

### Intentional typed-view incompatibilities

These are two intentional composition-vs-ndarray typed-view identity
incompatibilities. They are measured by a separate candidate probe and are
excluded from the filtered Spectrogram compatibility run.

Command:

```text
python -m pytest tests/spectrogram/test_sgm_matrix_coverage.py::test_sgm_attribute_propagation tests/spectrogram/test_sgm_matrix_ops.py::TestSgmToSeries1DList::test_ndim_less_than_3_value_error -q -p no:cacheprovider
```

Result: `2 failed`.

- `tests/spectrogram/test_sgm_matrix_coverage.py::test_sgm_attribute_propagation`
  assumes a typed `view(SpectrogramMatrix)`.
- `tests/spectrogram/test_sgm_matrix_ops.py::TestSgmToSeries1DList::test_ndim_less_than_3_value_error`
  assumes typed ndarray view construction.

### Reflected Quantity-left addition limitations

The filtered Spectrogram compatibility run keeps these as two genuine
reflected Quantity-left addition limitations; they are not the typed-view
cases above and do not imply that all Spectrogram tests pass.

Command:

```text
python -m pytest tests/spectrogram -q -p no:cacheprovider -k 'not test_sgm_attribute_propagation and not test_ndim_less_than_3_value_error'
```

Result: `389 passed, 2 failed, 5 skipped, 2 deselected, 1 xfailed`.

The two parameterized failures are
`tests/spectrogram/test_spectrogram_matrix_features.py::TestSpectrogramMatrixSeriesMatrixRules::test_add_quantity_converts_to_matrix_units`
(`matrix_unit0-addend0-1.01` and `matrix_unit1-addend1-101.0`). In both cases
the reflected form still returns a bare `Quantity` rather than a matrix.

### Legacy operator failures

Command:

```text
python -m pytest tests/types/test_series_matrix_operator_contract.py -q -p no:cacheprovider
```

Result: `339 passed, 33 failed`.

The 33 failures remain accurately grouped: 30 are
`test_direct_ufunc_application_raises` cases for the three matrix families,
and 3 are `test_modulo_and_floor_divide_do_not_ignore_units` cases for the
three matrix families. These are not silently reclassified as passes.

## Candidate behavior

The candidate replaces the matrix ndarray subclass with
`NDArrayOperatorsMixin` plus internal numeric storage. It preserves the
required public structure surface, raw `view(np.ndarray)`, per-cell metadata,
axes, labels, and independent result metadata. Ordinary single-output ufunc
calls dispatch through typed unit rules; `out=`, reductions/methods other than
`__call__`, multiple outputs, and unsupported `where` forms fail before
mutation. `np.sqrt(matrix)` and `(2 * u.s) * matrix` return the concrete matrix
family with per-cell units. Heterogeneous SpectrogramMatrix cells do not expose
a fabricated scalar `.unit`; a scalar is returned only for homogeneous cells
as a legacy compatibility view of `.units`.

## Fallback

Keep integration at its approved Phase A runtime state; do not copy the #637
composition candidate runtime. Preserve this B1 JSON and packet as
candidate-only evidence, with `adopted: false`, the frozen B0 benchmark and
fallback semantics, and the recorded hashes unchanged. Re-open adoption only
after a clean B0 benchmark freeze removes the unstable `slice` result and the
remaining compatibility failures receive an explicit contract decision.
