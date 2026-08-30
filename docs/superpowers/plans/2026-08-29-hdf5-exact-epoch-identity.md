# HDF5 Exact Epoch Identity Implementation Plan

> **For agentic workers:** REQUIRED: Use
> superpowers:subagent-driven-development (if subagents available) or
> superpowers:executing-plans to implement this plan. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Replace path-authoritative HDF5 exact-time metadata with a
GWpy-compatible, lineage-bound epoch marker and make every supported write target
transactional without duplicate native writes or unbounded wrapper memory.

**Architecture:** Put the canonical numeric marker and strict v2 sidecar schema in
one private identity module with no registry or transaction responsibilities. Keep
dataset discovery, native GWpy read/view integration, registry wrapping, and the
three target-specific transaction strategies in `hdf5.py`; pathname and file-like
targets use disposable staging, while caller-owned HDF5 handles use an
identity-preserving recovery hard link.

**Tech Stack:** Python 3.11, GWpy 4.x, Astropy units, h5py/HDF5, NumPy, pytest,
Ruff, MyPy, conda environment `gwexpy`.

**Status:** in-progress

Tasks 1–6 and Checkpoint A are completed. Tasks 6–8 were approved for the
local v0.2.1 P0 scope on 2026-08-30. Tasks 7–10 remain planned; P0 approval
still requires their verification and the independent reviews below.

**Specification:**
`docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md`

**Release boundary:** This plan performs local source, test, documentation, and
Git commits only. It must not dispatch a release workflow, publish an artifact,
change conda-forge or Zenodo, push a branch, or approve P0/P1 by itself.

---

## File map

### Production code

- Create `gwexpy/timeseries/io/_hdf5_exact_epoch.py`: canonical epoch marker,
  unit binding, strict v2 sidecar records, reconstruction, and bounded JSON
  parsing. It does not register I/O handlers or mutate HDF5 files.
- Modify `gwexpy/timeseries/io/hdf5.py`: raw dataset inspection, resolved-file
  authority, sidecar compaction/application, native GWpy-to-GWexpy read/view,
  reload-safe registration, path validation, and target-specific transactions.
- Preserve `gwexpy/timeseries/timeseries.py`: the public constructor is not
  changed to understand storage markers; HDF5 reads must avoid invoking it on a
  marker.

### Tests

- Create `tests/timeseries/test_hdf5_exact_epoch_codec.py`: pure marker and v2
  document tests, including numerical boundaries, canonicality, corruption, and
  schema limits.
- Create `tests/timeseries/test_hdf5_exact_t0_transactions.py`: native-call count,
  pathname, file-like, open-handle, failure-injection, cleanup, storage-growth,
  and memory tests.
- Modify `tests/timeseries/test_hdf5_exact_t0.py`: v2 public integration,
  authority truth table, aliases, copies, external links, native path forms,
  GWpy-only compatibility, reload, and migration from v1 expectations.

### Documentation

- Modify this plan only to check completed steps while executing it.
- Preserve the approved design specification except for separately reviewed
  factual corrections. Do not silently change implementation semantics in the
  plan or code.

## Locked implementation boundaries

- The marker decoder derives its only payload boundary from
  `float(marker)` and the canonical `Decimal.from_float` prefix. It does not scan
  payload-internal guard/magic occurrences as independent candidates.
- The exact integer is never projected back to `x0` for validation. Legitimate
  exact slices can differ by one ULP from a fresh integer-to-unit conversion.
- Every wrapper read asks the saved native reader for a base GWpy `TimeSeries`,
  then views it as the requested GWexpy class.
- A dataset marker is the only exact-time authority. A v2 sidecar corroborates
  and diagnoses; v1 and paths never authorize exact time.
- Pathname and file-like stages call the native dataset writer once and never
  create an in-file recovery hard link.
- Caller-owned `h5py.File` and `h5py.Group` writes retain the old physical object
  through an in-file recovery link; H5Ocopy is not a rollback substitute.
- File-like copying uses a fixed-size buffer and disk-backed working/backup
  files. It must not call `BytesIO.getvalue()` or retain a full Python `bytes`
  snapshot.

## Command policy

Prefix every shell invocation with RTK. Run Python tooling in the `gwexpy` conda
environment. Short RED/GREEN nodes may use direct `conda run`; longer suites use
the project job script and its log.

```bash
rtk conda run -n gwexpy pytest <node> -q
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start pytest <path> -q
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh tail <session-name>
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start ruff
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start mypy
```

After every pytest job, run `rtk git status --short` and preserve unrelated user
changes.

### Required per-node TDD micro-cycle

Every test node named below is one independent 2–5 minute micro-cycle. Do not
write a whole bullet list and then run it as one batch. For each node, repeat
these checkboxes before moving to the next node:

- [ ] Add exactly that test or one parameter row to the named test file.
- [ ] Run `rtk conda run -n gwexpy pytest <file>::<node> -q` and confirm the
  expected semantic assertion fails. Collection, import, and fixture errors are
  not acceptable RED results.
- [ ] Add only the helper or branch required by that RED result.
- [ ] Rerun the identical node and confirm `1 passed`, or the exact expected
  parameterized case count.
- [ ] Run the already-green nodes for the same task before committing.

Every failure-injection test also asserts that its intended injection seam was
called exactly once. A different exception cannot satisfy `pytest.raises` by
accident.

The task-level commands below are aggregation gates after those micro-cycles.
For a parameterized test, use its collected node ID while developing one row and
the unqualified function node for the task gate.

Task 1 Step 2 is the sole bootstrap exception: because the new private module does
not yet exist, its first run may fail during collection. Step 3 immediately creates
an importable API scaffold whose entry points raise `NotImplementedError`. From
that point onward, only semantic assertion failures count as RED.

---

### Task 1: Build the canonical epoch-marker codec

**Status:** completed (verified: `rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_epoch_codec.py -q`)

**Files:**

- Create: `gwexpy/timeseries/io/_hdf5_exact_epoch.py`
- Create: `tests/timeseries/test_hdf5_exact_epoch_codec.py`

- [x] **Step 1: Write the numerical-envelope RED tests**

Add deterministic tests for explicit `+`/`-`, negative zero, minimum subnormal,
minimum normal, maximum finite, and seeded random finite binary64 values. Test
the seven required units and positive, zero, and negative exact nanoseconds.

```python
@pytest.mark.parametrize(
    "bits",
    [
        0x0000000000000000,
        0x8000000000000000,
        0x0000000000000001,
        0x0010000000000000,
        0x7FEFFFFFFFFFFFFF,
        0xFFEFFFFFFFFFFFFF,
    ],
)
def test_v2_marker_envelope_preserves_binary64_boundaries(bits: int) -> None:
    raw_x0 = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
    marker = encode_epoch_marker(
        epoch_ns=1234567890123456789,
        raw_x0=raw_x0,
        xunit="s",
        token=bytes.fromhex("00" * 16),
    )
    assert float_bits(float(marker.text)) == bits
    assert decode_epoch_marker(
        marker.text,
        raw_x0=raw_x0,
        xunit="s",
    ).epoch_ns == 1234567890123456789
```

Also add `test_v2_marker_envelope_preserves_seeded_random_binary64_values`,
`test_v2_marker_binds_supported_axis_units`, and
`test_v2_marker_rejects_nonfinite_x0`.

The seeded property-style node uses a fixed published seed and at least 40,000
finite patterns stratified across exponent classes and signs. Add targeted cases
around powers of ten and decimal-representation transition boundaries. The fixed
seed and sample count remain reproducible release-gate inputs.

- [x] **Step 2: Run the envelope tests and verify RED**

Run:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_epoch_codec.py -q
```

Expected under the one-time bootstrap exception: collection fails because
`_hdf5_exact_epoch` is absent.

- [x] **Step 3: Create the importable scaffold, then obtain semantic RED**

Create the module, immutable value types, and public test-facing function
signatures. The codec entry points initially raise `NotImplementedError`. Rerun
the same node and require RED at that exception rather than collection.

- [x] **Step 4: Define the primitive encoders**

Implement focused immutable values and helpers. Keep the module private and do
not import h5py or the GWpy registry.

```python
@dataclass(frozen=True)
class AxisBinding:
    xunit: str
    xunit_to_ns_bits: str
    ns_to_xunit_bits: str


@dataclass(frozen=True)
class EpochMarker:
    text: str
    lineage_token: str
    epoch_ns: int
    x0_bits: str
    axis: AxisBinding
    marker_sha256: str
```

Add `_float_bits`, `_canonical_axis_binding`, `_canonical_x0_prefix`,
`_encode_magnitude`, `_decode_magnitude`, `_encode_triplets`, and
`_decode_triplets`. Enforce finite binary64, canonical UTF-8 unit length 255,
minimal magnitude length 512, and the unique zero representation.

- [x] **Step 5: Write payload canonicality and corruption RED tests**

Add these nodes before implementing the payload:

- `test_v2_marker_roundtrip_is_byte_canonical`
- `test_v2_marker_zero_magnitude_has_one_encoding`
- `test_v2_marker_rejects_bad_digest_and_trailing_bytes`
- `test_v2_marker_rejects_noncanonical_payload_encoding`
- `test_v2_marker_rejects_unit_and_factor_tampering`
- `test_v2_marker_recognizable_corruption_raises`
- `test_ordinary_epoch_metadata_is_not_claimed_as_v2`
- `test_v2_marker_payload_internal_magic_is_not_a_second_candidate`
- `test_v2_marker_enforces_4096_character_cap`

Parameterize `test_v2_marker_rejects_noncanonical_payload_encoding` over a
triplet above 255, non-multiple-of-three payload, invalid field length,
non-minimal magnitude, and negative-zero epoch payload.

Run the named nodes and verify that they fail for missing behavior, not for a
fixture or import error introduced by the scaffold.

- [x] **Step 6: Implement the complete payload and one-boundary decoder**

Implement `encode_epoch_marker`, `decode_epoch_marker`, and
`reconstruct_epoch_marker` with the exact field order in the specification.
The decoder must:

```python
def decode_epoch_marker(
    value: object,
    *,
    raw_x0: object,
    xunit: object,
) -> EpochMarker | None:
    text = _strict_epoch_text_or_none(value)
    if text is None:
        return None
    parsed_float = _parse_finite_marker_float(text)
    prefix = _canonical_x0_prefix(parsed_float)
    boundary = len(prefix) + MARKER_GUARD_DIGITS
    # Only this boundary is parsed. Payload-internal magic is ordinary payload.
    payload = _decode_canonical_payload_at(text, boundary)
    marker = _validate_payload_and_reencode(text, payload)
    _require_current_axis_binding(marker, raw_x0=raw_x0, xunit=xunit)
    return marker
```

An ordinary numeric epoch returns `None`; recognizable malformed v2 input raises
`ValueError`. The SHA-256 payload digest uses the fixed domain separator, while
`marker_sha256` hashes the complete ASCII marker without that separator.

- [x] **Step 7: Run codec tests GREEN and commit**

Run:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_epoch_codec.py -q
rtk conda run -n gwexpy ruff check gwexpy/timeseries/io/_hdf5_exact_epoch.py tests/timeseries/test_hdf5_exact_epoch_codec.py
rtk conda run -n gwexpy mypy gwexpy/timeseries/io/_hdf5_exact_epoch.py
rtk git diff --check
```

Expected: all commands pass.

Commit:

```bash
rtk git add gwexpy/timeseries/io/_hdf5_exact_epoch.py tests/timeseries/test_hdf5_exact_epoch_codec.py
rtk git commit -m "feat: add canonical HDF5 epoch marker codec"
```

---

### Task 2: Implement the strict v2 sidecar document

**Status:** completed (verified: `rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_epoch_codec.py -q`)

**Files:**

- Modify: `gwexpy/timeseries/io/_hdf5_exact_epoch.py`
- Modify: `tests/timeseries/test_hdf5_exact_epoch_codec.py`

- [x] **Step 1: Add whole-document and bound RED tests**

Create canonical test records from a fixed `EpochMarker`; do not handwave a
partially valid JSON fixture. Cover duplicate JSON keys, extra/missing fields,
bool-as-int, invalid UTF-8, NaN/Infinity, bad token/digest widths, nonempty
provenance, inconsistent metadata, unit/factor mismatch, 8 MiB JSON, 10,000
records, 16 paths, and 4,096-byte paths.

Name the parameterizations:

- `test_v2_sidecar_rejects_duplicate_or_noncanonical_json`;
- `test_v2_sidecar_rejects_bad_key_sets_and_bool_epoch`;
- `test_v2_sidecar_rejects_invalid_utf8`;
- `test_v2_sidecar_rejects_cross_field_mismatch`;
- `test_v2_sidecar_rejects_nonempty_provenance`;
- `test_v2_sidecar_rejects_schema_limits`.

```python
def test_v2_sidecar_rejects_one_invalid_unselected_record() -> None:
    selected = record_for(encode_fixture(token="00" * 16))
    invalid = record_for(encode_fixture(token="11" * 16))
    invalid["metadata"]["_gwexpy_t0_gps_state"]["_gwex_t0_gps_ns"] += 1
    raw = sidecar_json({"00" * 16: selected, "11" * 16: invalid})
    with pytest.raises(ValueError, match="sidecar"):
        parse_v2_sidecar(raw)
```

Add `test_v2_sidecar_reconstructs_complete_ascii_marker`,
`test_v2_sidecar_marker_sha256_covers_complete_marker`,
`test_v2_sidecar_paths_are_diagnostic_only`, and
`test_v1_sidecar_parser_never_returns_exact_authority`.

- [x] **Step 2: Run the sidecar nodes and verify RED**

Run each node through the required micro-cycle, then aggregate with:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_epoch_codec.py -k 'v2_sidecar or v1_sidecar' -q
```

Expected before Step 3: failures identify missing v2 parse, reconstruction, or
limit behavior. Expected after Step 3: every selected node passes.

- [x] **Step 3: Implement strict parse, reconstruction, and serialization**

Add immutable `SidecarRecord` and `SidecarDocument` values plus:

```python
def parse_v2_sidecar(raw: object) -> SidecarDocument: ...
def serialize_v2_sidecar(records: Iterable[SidecarRecord]) -> str: ...
def record_from_marker(marker: EpochMarker, paths: Iterable[str]) -> SidecarRecord: ...
def validate_marker_record(
    marker: EpochMarker,
    document: SidecarDocument | None,
) -> SidecarRecord | None: ...
```

Parse with `object_pairs_hook` and `parse_constant`. Validate every record before
returning the document, reconstruct its complete marker byte-for-byte, and compare
`marker_sha256`. Sort record keys and unique diagnostic paths at serialization.
An absent document is distinct from an empty valid document.

- [x] **Step 4: Run codec and document tests GREEN and commit**

Run:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_epoch_codec.py -q
rtk conda run -n gwexpy ruff check gwexpy/timeseries/io/_hdf5_exact_epoch.py tests/timeseries/test_hdf5_exact_epoch_codec.py
rtk conda run -n gwexpy mypy gwexpy/timeseries/io/_hdf5_exact_epoch.py
rtk git diff --check
```

Expected: all commands pass.

Commit:

```bash
rtk git add gwexpy/timeseries/io/_hdf5_exact_epoch.py tests/timeseries/test_hdf5_exact_epoch_codec.py
rtk git commit -m "feat: validate HDF5 exact epoch sidecar v2"
```

---

### Task 3: Integrate the marker with native GWpy read and write

**Status:** completed (verified: `rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py -q`)

**Files:**

- Modify: `gwexpy/timeseries/io/hdf5.py`
- Modify: `tests/timeseries/test_hdf5_exact_t0.py`

- [ ] **Step 1: Replace v1 test helpers with marker-token helpers**

Define test helpers that read `_gwexpy_sidecar_json_v2`, decode a dataset's raw
`epoch` marker, and select the record by `lineage_token`. Keep a separate helper
for deliberately constructing v1 fixtures.

- [ ] **Step 2: Write basic public integration RED tests**

Add or update:

- `test_hdf5_roundtrip_preserves_exact_t0_and_core_metadata`
- `test_hdf5_exact_t0_writes_v2_marker_and_token_record`
- `test_hdf5_exact_t0_roundtrips_standard_axis_units`
- `test_hdf5_reader_uses_native_gwpy_semantics_for_marker_states`
- `test_hdf5_marker_only_read_recovers_exact_t0`
- `test_hdf5_v1_sidecar_never_authorizes_exact_t0`
- `test_hdf5_successful_v2_write_removes_v1_attribute`
- `test_hdf5_malformed_marker_fails_before_native_reader`
- `test_hdf5_marker_read_crops_after_attaching_exact_authority`

The marker-state parameterization is `absent`, `ordinary`, and `v2`, crossed with
at least `s`, `ms`, `min`, and `day`.

- [ ] **Step 3: Write the complete caller-metadata RED matrix**

Execute one micro-cycle per parameter row in
`test_hdf5_write_metadata_policy_fails_before_mutation`. Cover:

- exact write with mismatching `attrs["x0"]` bits;
- exact write with equivalent but noncanonical or scaled `attrs["xunit"]`;
- exact write with ordinary `attrs["epoch"]` whose float bits mismatch;
- matching ordinary epoch, which is replaced by a canonical marker;
- matching v2 epoch, which retains its lineage token;
- v2 epoch with conflicting exact ns or fingerprint;
- exact and non-exact writes with a recognizable malformed v2 epoch;
- non-exact write with an ordinary epoch, which remains native metadata;
- exact output with `external=`, rejected before HDF5/raw-file mutation;
- non-exact `external=` with canonical or recognizable v2 caller epoch;
- non-exact `external=` replacement of a marked or sidecar-managed dataset.

For pathname, file-like, `File`, and nested `Group`, snapshot target bytes or
public links, both root sidecar attrs, raw external storage, and caller `attrs`.
Every rejected row must leave all snapshots unchanged.

- [ ] **Step 4: Write sidecar-compaction RED tests before traversal exists**

Execute one micro-cycle per node:

- `test_hdf5_compaction_adds_marker_only_copy_and_drops_stale_record`;
- `test_hdf5_compaction_merges_same_lineage_copies`;
- `test_hdf5_compaction_caps_deterministic_paths_at_sixteen`;
- `test_hdf5_compaction_handles_hard_group_alias_and_self_cycle`;
- `test_hdf5_compaction_excludes_rollback_namespace_before_dereference`;
- `test_hdf5_compaction_does_not_follow_soft_or_external_links`;
- `test_hdf5_compaction_rejects_unrelated_malformed_local_marker`;
- `test_hdf5_compaction_rejects_conflicting_same_token_objects`;
- `test_hdf5_compaction_refreshes_paths_without_using_them_for_authority`;
- `test_hdf5_sidecar_size_tracks_live_markers_not_operation_count`.

Use synthetic observations for 10,001-record and 8 MiB pure bounds; do not create
10,001 HDF5 datasets. Give the cycle node a short timeout so infinite traversal is
a deterministic failure. The size-stability node performs hundreds of
write/overwrite/copy/rename/delete cycles and asserts that serialized sidecar size
tracks live markers rather than operation count.

- [ ] **Step 5: Write mutation-seam RED tests for every current target**

Parameterize `test_hdf5_marker_mutation_failure_restores_all_current_targets`
over pathname, binary file-like, `h5py.File`, and nested `h5py.Group`, and over
native dataset creation, canonical `x0`/`xunit` reset, marker write, v2 payload
build, and v2/v1 apply.

For each row assert public values, raw `x0`, exact authority, both raw sidecar
attrs, created parent groups, file-like position, private recovery objects, error
state/recovery path, and unchanged caller `attrs`. These rows must all become
GREEN in this task using the current target envelopes; later transaction
refactors preserve them.

- [ ] **Step 6: Run every Task 3 node and verify semantic RED**

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_roundtrip_preserves_exact_t0_and_core_metadata tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_exact_t0_writes_v2_marker_and_token_record tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_exact_t0_roundtrips_standard_axis_units tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_reader_uses_native_gwpy_semantics_for_marker_states tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_marker_only_read_recovers_exact_t0 tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_v1_sidecar_never_authorizes_exact_t0 tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_successful_v2_write_removes_v1_attribute tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_malformed_marker_fails_before_native_reader tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_marker_read_crops_after_attaching_exact_authority tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_write_metadata_policy_fails_before_mutation tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_marker_mutation_failure_restores_all_current_targets -q
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py -k 'hdf5_compaction_' -q
```

Expected: failures identify absent v2 marker/sidecar, incorrect constructor
semantics, missing metadata validation/rollback, or missing compaction. No node
may fail from collection, fixture setup, or a `KeyError` in the assertion.

- [ ] **Step 7: Implement marker write, compaction, and atomic v2 application**

Replace v1 constants with explicit v1/v2 names. Validate authority and caller
`attrs` before mutation. After the native writer returns, reset and validate raw
`x0`/`xunit`, write the marker, collect live marker records, build v2, and remove
v1 as one logical commit.

Introduce `_build_v2_sidecar` and `_apply_sidecar_payload` as separate injectable
seams. The builder performs sorted, local-hard-link-only, cycle-safe traversal;
tracks group and dataset identities only during the scan; excludes recovery names
before dereference; validates every local marker; and caps deterministic paths.
The apply helper alone mutates the two root attrs. Extend each current transaction
envelope just enough to restore marker, v1, and v2 state at the Step 5 seams.

- [ ] **Step 8: Implement raw decode, base-GWpy read, and view conversion**

Resolve the dataset first. Read marker and sidecar authority from `dataset.file`,
not `source.file`. Decode before native construction, force the saved native
reader's `array_type` to base GWpy `TimeSeries`, view as the requested GWexpy
class, attach exact authority only after binding checks, then crop.

- [ ] **Step 9: Verify GWpy-only compatibility**

Update the isolated subprocess test to assert that GWpy 4 reads the long numeric
`epoch`, preserves raw `x0` bits in every required standard unit, and never imports
`gwexpy`.

- [ ] **Step 10: Run Task 3 GREEN and commit the basic integration**

Rerun both exact commands from Step 6, the GWpy-only subprocess node, and the full
codec file. Expected: all commands pass and no Task 3 test remains RED.

```bash
rtk git add gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0.py
rtk git commit -m "feat: integrate exact epoch markers with HDF5 I/O"
```

---

### Task 4: Make identity independent of paths and physical object reuse

**Status:** completed (verified: `rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py -q`)

**Files:**

- Modify: `gwexpy/timeseries/io/hdf5.py`
- Modify: `tests/timeseries/test_hdf5_exact_t0.py`

- [ ] **Step 1: Write `test_hdf5_v2_authority_truth_table` RED rows**

Cover absent marker, valid marker without sidecar, valid matching marker/record,
missing record, stale record after GWpy overwrite, marker/record fingerprint
conflict, malformed unselected record, duplicate lineage records, and ordinary
numeric epoch. Assert exact, native/quantized, or `ValueError` exactly as the spec
table states.

Include v1-only, malformed-v1-only, valid-v2-plus-malformed-v1, and
invalid-v2-plus-valid-v1 rows. V1 never authorizes exact time; invalid v2 never
falls back to v1.

- [ ] **Step 2: Write alias, move, and copy RED tests**

Add:

- `test_hdf5_v2_marker_survives_hard_and_soft_alias_reads`
- `test_hdf5_v2_marker_survives_move_and_rename`
- `test_hdf5_v2_marker_survives_same_file_h5ocopy`
- `test_hdf5_v2_marker_survives_cross_file_h5ocopy_without_sidecar`
- `test_hdf5_copy_without_attributes_loses_exact_authority`
- `test_hdf5_gwpy_overwrite_without_marker_ignores_stale_v2_record`
- `test_hdf5_recreated_object_cannot_inherit_stale_exact_authority`
- `test_hdf5_exact_slice_with_one_ulp_public_x0_difference_roundtrips`
- `test_hdf5_independent_equal_epochs_receive_distinct_lineage_tokens`

Add the approved custom scaled time unit to the axis-unit parameterization; the
seven standard units alone are not the complete binding gate.

- [ ] **Step 3: Verify RED against the current path-authoritative behavior**

Run each family separately. Save the failure summary in the implementation
handoff; do not weaken expected authority to match current behavior.

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_v2_authority_truth_table tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_v2_marker_survives_hard_and_soft_alias_reads tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_v2_marker_survives_move_and_rename tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_v2_marker_survives_same_file_h5ocopy tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_v2_marker_survives_cross_file_h5ocopy_without_sidecar tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_copy_without_attributes_loses_exact_authority tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_gwpy_overwrite_without_marker_ignores_stale_v2_record tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_recreated_object_cannot_inherit_stale_exact_authority tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_exact_slice_with_one_ulp_public_x0_difference_roundtrips tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_independent_equal_epochs_receive_distinct_lineage_tokens -q
```

Expected: current path-authoritative alias/copy/overwrite assertions fail while
the already-green Task 3 compaction tests stay green.

- [ ] **Step 4: Bind reads by marker lineage and full fingerprint**

Delete `_sidecar_alias_paths` authority logic. Match by decoded lineage token,
complete marker hash, `x0_bits`, unit, and both factor bits. Permit marker-only
authority for copied datasets; reject conflicts and never copy exact ns from a
sidecar into a dataset without a valid marker.

- [ ] **Step 5: Run identity tests GREEN and commit**

Rerun the exact Step 3 command, the complete codec file, and
`-k 'hdf5_compaction_'`. Expected: all commands pass.

Commit:

```bash
rtk git add gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0.py
rtk git commit -m "fix: bind HDF5 exact epochs to dataset lineage"
```

---

### Task 5: Restore native paths, link safety, and reload idempotence

**Status:** completed (verified: `rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py -q`)

**Files:**

- Modify: `gwexpy/timeseries/io/hdf5.py`
- Modify: `tests/timeseries/test_hdf5_exact_t0.py`

- [ ] **Step 1: Write native path compatibility RED tests**

Parameterize pathname, file-like, `h5py.File`, and `h5py.Group` writes over:

- relative and absolute `str` paths;
- relative and absolute UTF-8 `bytes` paths;
- non-ASCII UTF-8 paths;
- NUL, invalid UTF-8, empty, `.`, `..`, and empty components.

Name these parameterizations
`test_hdf5_native_path_matrix_preserves_original_object` and
`test_hdf5_invalid_native_path_fails_before_mutation`.

Assert the wrapper passes the original safe path object unchanged to the native
writer. Invalid paths fail before file, raw external storage, or handle mutation.

- [ ] **Step 2: Write link-boundary RED tests**

Require reads through `ExternalLink` to use the resolved external file's marker and
sidecar. Require writes to reject an `ExternalLink` leaf or ancestor, a soft link
that resolves externally, and a leaf `SoftLink`; allow a proven local soft-linked
ancestor. Preserve unrelated links and sidecars after every rejection.

Name the two parameterizations `test_hdf5_link_write_policy` and
`test_hdf5_external_link_read_uses_resolved_file_authority`.

- [ ] **Step 3: Write reload RED tests**

In isolated subprocesses, import the registry in both supported orders, reload
`gwexpy.timeseries.io.hdf5` twice, and assert one native read/write call with no
recursion. Inject half-wrapped and recursive wrapper states and require a clear
`RuntimeError` instead of silently storing `None` bases.

Name the subprocess nodes `test_hdf5_registry_reload_is_idempotent` and
`test_hdf5_registry_rejects_half_or_recursive_wrapper`.

Aggregate the completed micro-cycles with:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_native_path_matrix_preserves_original_object tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_invalid_native_path_fails_before_mutation tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_link_write_policy tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_external_link_read_uses_resolved_file_authority tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_registry_reload_is_idempotent tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_registry_rejects_half_or_recursive_wrapper -q
```

Expected before Steps 4–5: absolute/bytes inline paths and reload nodes fail at
the wrapper. Expected after them: every selected node passes.

- [ ] **Step 4: Implement native path normalization without changing native input**

Use decoded components only for validation and root-relative diagnostic lookup.
Keep the original `str | bytes` value for the native writer. Reject unsafe
components before mutation, but do not reject a leading slash.

- [ ] **Step 5: Implement reload-safe handler recovery**

Attach saved native callables to both registered wrapper functions. During
registration, unwrap only a complete marked pair, validate callability and
non-recursion, and repopulate module `_BASE_READER`/`_BASE_WRITER` after reload.
Reject a half-wrapped pair.

- [ ] **Step 6: Run path/link/reload tests GREEN and commit**

Rerun the exact aggregate command from Step 3, then run:

```bash
rtk conda run -n gwexpy ruff check gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0.py
rtk conda run -n gwexpy mypy gwexpy/timeseries/io/hdf5.py
```

Expected: all commands pass.

```bash
rtk git add gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0.py
rtk git commit -m "fix: preserve native HDF5 paths across reloads"
```

---

### Checkpoint A: Reassess the v0.2.1 transaction scope

**Status:** completed (verified: `rtk conda run -n gwexpy pytest tests/qualification/test_v020_release_claims.py::test_timeseries_hdf5_roundtrip_retains_exact_t0_gps_ns tests/timeseries/test_hdf5_exact_epoch_codec.py tests/timeseries/test_hdf5_exact_t0.py -q`; 705 passed, 1 skipped; Tasks 6–8 approved on 2026-08-30)

**Files:**

- Read-only qualification; do not modify production code during the decision.

- [ ] **Step 1: Run the official exact-time claim and focused contract**

```bash
rtk conda run -n gwexpy pytest tests/qualification/test_v020_release_claims.py::test_timeseries_hdf5_roundtrip_retains_exact_t0_gps_ns tests/timeseries/test_hdf5_exact_epoch_codec.py tests/timeseries/test_hdf5_exact_t0.py -q
```

Expected: the historical +165/+166 ns reproduction is now a 0 ns exact
round-trip; marker/sidecar authority and GWpy-only reads pass for every supported
target form covered by Tasks 1–5.

- [ ] **Step 2: Report the direct P0 evidence and stop for scope approval**

Report exact pass/fail/skip counts and evidence for:

- official v0.2.0 qualification claim;
- exact marker and v2 sidecar authority;
- marker-only recovery and stale-sidecar rejection;
- GWpy-only compatibility;
- aliases, move/copy, path forms, and reload idempotence.

Do not begin Task 6 in the same execution batch. Ask whether the observed native
writer duplication, pathname atomicity gap, file-like full-buffer duplication,
and open-handle rollback risks remain v0.2.1 blockers.

- [ ] **Step 3: Apply the scope decision**

- If the transaction wrapper is still required to close P0 safely, proceed with
  Tasks 6–8.
- If Tasks 1–5 close the corrective-patch contract without those risks, defer
  Tasks 6–8 to a separately approved maintenance/hardening series and proceed to
  adapted Tasks 9–10.
- Keep bootstrap/on-demand registration P1 out of this HDF5 series in either
  branch of the decision.

---

### Task 6: Introduce one-write disposable staging and atomic pathname writes

**Status:** completed (verified: `rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py -q`; 22 passed, and `rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py -q`; 590 passed)

**Files:**

- Modify: `gwexpy/timeseries/io/hdf5.py`
- Create: `tests/timeseries/test_hdf5_exact_t0_transactions.py`

- [ ] **Step 1: Write native-call and disposable-stage RED tests**

Add `test_hdf5_each_transaction_invokes_native_writer_once` for pathname,
file-like, `File`, and `Group`, on success and injected post-write failure. Add
`test_hdf5_disposable_stage_never_creates_recovery_group` and assert path/file-like
stages do not retain or write an old replacement dataset.

- [ ] **Step 2: Run the call-count tests and verify RED**

Expected: pathname, file-like, `File`, and `Group` overwrite cases all show the
existing preflight plus the real native writer. Path/file-like stages also show an
unnecessary in-file recovery object.

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py -k 'invokes_native_writer_once or disposable_stage' -q
```

- [ ] **Step 3: Factor the common write and disposable-stage helpers**

Remove `_preflight_core_write`. Introduce:

```python
def _write_dataset_once(
    array: Any,
    container: h5py.File | h5py.Group,
    path: str | bytes | None,
    exact_epoch: int | None,
    kwargs: Mapping[str, Any],
) -> h5py.Dataset: ...

def _write_disposable_stage(
    array: Any,
    stage: BinaryIO,
    path: str | bytes | None,
    exact_epoch: int | None,
    kwargs: Mapping[str, Any],
) -> None: ...
```

The first performs one native write plus marker/sidecar work for every target.
Remove `_preflight_core_write` from caller-owned handles as well as disposable
stages. The disposable stage opens the passed file object and calls it directly,
with no preflight and no recovery link.

- [ ] **Step 4: Write pathname target RED tests**

Add:

- `test_hdf5_path_rejects_nonregular_target_before_staging` for directory,
  symlink, FIFO, socket, and available device fixtures;
- `test_hdf5_path_rejects_multiply_linked_regular_target`;
- `test_hdf5_path_replace_failure_preserves_old_file_and_cleans_stage`;
- `test_hdf5_path_replace_and_unlink_failure_reports_old_state_and_stage`;
- `test_hdf5_path_append_preserves_unrelated_entries`;
- `test_hdf5_path_overwrite_without_append_starts_fresh`;
- `test_hdf5_path_repeated_overwrite_has_bounded_growth`;
- `test_hdf5_disposable_stage_does_not_duplicate_old_dataset_storage`.

The last node compares a representative 16 MiB replacement with native output
using a fixed marker/sidecar allowance and fails on the current factor-of-two
stage image. Run each node through the per-node micro-cycle before Step 5.

- [ ] **Step 5: Implement atomic pathname staging**

Use `os.lstat`; accept only nonexistent or regular targets with `st_nlink == 1`.
Create a same-directory `O_CREAT | O_EXCL` sibling. Copy the old file only for
`append=True`; start fresh for overwrite without append. Pass an opened stage file
object to `_write_disposable_stage`, preserve only permission mode, close and
flush before `os.replace`, and classify replace/cleanup failures with
`state="old"` and a retained stage path when cleanup fails.

- [ ] **Step 6: Run pathname tests GREEN and commit**

Run transaction call-count/path tests plus the existing path compatibility nodes.

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py -k 'native_writer_once or disposable_stage or path_' -q
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_marker_mutation_failure_restores_all_current_targets -q
```

Expected: every selected node passes. No tracked test is RED at commit.

Commit:

```bash
rtk git add gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0_transactions.py
rtk git commit -m "fix: stage HDF5 pathname writes atomically"
```

---

### Task 7: Make file-like transactions disk-backed and chunk-bounded

**Status:** planned

**Files:**

- Modify: `gwexpy/timeseries/io/hdf5.py`
- Modify: `tests/timeseries/test_hdf5_exact_t0_transactions.py`

- [ ] **Step 1: Write bounded-copy primitive RED tests**

Use adversarial file objects to add:

- `test_hdf5_filelike_copy_requests_are_chunk_bounded`;
- `test_hdf5_filelike_copy_retries_short_positive_writes`;
- `test_hdf5_filelike_copy_rejects_none_zero_negative_and_oversize_counts`;
- `test_hdf5_filelike_copy_truncates_to_exact_final_size`.

Every recorded `read(size)` must be at most the configured chunk. A short positive
write is retried; every non-progress or invalid count raises `OSError`.

- [ ] **Step 2: Implement the bounded copy helper**

```python
def _copy_filelike(
    source: Any,
    destination: Any,
    *,
    chunk_size: int = FILELIKE_COPY_CHUNK,
) -> int:
    """Copy from current positions and return bytes copied."""
```

Use one reusable `bytearray`/`memoryview`-sized buffer where supported. Loop on
short positive writes and explicitly truncate the destination to the returned
size at commit/rollback call sites.

- [ ] **Step 3: Write file-like envelope and cleanup RED tests**

Add failure injection for original `tell`, `seek`, `read`, backup create/write/
flush/fsync/close, working create/copy/open/write/marker/sidecar/close, commit
write/truncate/flush, rollback bytes, rollback position, and temp cleanup. Required
nodes include:

- `test_hdf5_filelike_precommit_failure_restores_position`;
- `test_hdf5_filelike_commit_failure_restores_bytes_and_position`;
- `test_hdf5_filelike_incomplete_rollback_retains_durable_backup`;
- `test_hdf5_filelike_classifies_byte_and_position_state_independently`;
- `test_hdf5_filelike_success_cleanup_failure_warns_new_and_returns`;
- `test_hdf5_filelike_complete_rollback_cleanup_failure_reports_old`;
- `test_hdf5_filelike_normal_paths_leak_no_tempfiles`;
- `test_hdf5_filelike_overwrite_preserves_native_existing_entry_semantics`;
- `test_hdf5_filelike_backup_is_mode_0600_and_fsynced_before_commit`;
- `test_hdf5_filelike_repeated_overwrite_has_bounded_growth`;
- `test_hdf5_filelike_large_write_has_bounded_wrapper_rss`.

Write the two resource nodes before implementation. The RSS node runs in a
subprocess with two sufficiently large preexisting sizes and a fixed chunk-scale
allowance, excluding the caller buffer, input array, and h5py cache. It must fail
while `_filelike_snapshot`, `BytesIO(snapshot)`, or `getvalue()` remains. The
growth node fails on the current retained old dataset image.

Develop each node through the required micro-cycle. Aggregate with:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py -k 'filelike or copy_' -q
```

Expected before Step 4: full-buffer duplication and missing cleanup-state
assertions fail. Expected after Step 5: every selected node passes.

- [ ] **Step 4: Implement disk-backed backup, working image, and commit**

Capture the original position first. Create a named mode-0600 backup, copy the
target in chunks, flush and `fsync` it, then create a disk-backed working file and
initialize it from the target in every mode. Pass the working file object, not its
path, to h5py so GWpy's file-object overwrite semantics remain unchanged.
Set the working file to the caller's original position before opening the HDF5
stage, and establish the committed position only after stage close.

Commit by chunk-copying working to target, truncating, and flushing. On failure,
restore bytes from the named backup and restore position. Retain the backup until
both are known restored. Normal success and complete rollback close and unlink
both temporaries.

Post-commit cleanup failure returns success and emits a classified
`ResourceWarning` with `state="new"`. Incomplete rollback raises `_RollbackError`
with byte state, position state, and durable `recovery_path`.

- [ ] **Step 5: Run the resource nodes after implementation**

Run each named growth/RSS node directly. Expected: both pass before the Task 7
aggregate gate and no size-dependent full Python snapshot remains.

- [ ] **Step 6: Run file-like tests GREEN and commit**

Run:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py -k 'filelike or copy_' -q
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_marker_mutation_failure_restores_all_current_targets -q
rtk conda run -n gwexpy ruff check gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0_transactions.py
rtk conda run -n gwexpy mypy gwexpy/timeseries/io/hdf5.py
```

Expected: all commands pass and no tracked test is RED.

```bash
rtk git add gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0_transactions.py
rtk git commit -m "fix: bound HDF5 file-like transaction memory"
```

---

### Task 8: Harden caller-owned handle rollback and recovery

**Status:** planned

**Files:**

- Modify: `gwexpy/timeseries/io/hdf5.py`
- Modify: `tests/timeseries/test_hdf5_exact_t0_transactions.py`
- Modify: `tests/timeseries/test_hdf5_exact_t0.py`

- [ ] **Step 1: Write rollback-setup RED tests**

Inject failure at recovery group create, old-dataset hard link, v1 snapshot, v2
snapshot, flush/verify, and partial setup cleanup. Require unchanged public data,
link identity, sidecars, and parent groups. A setup-cleanup failure reports
`state="old"` plus the surviving private path when one exists.

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_open_recovery_setup_failure_preserves_public_state -q
```

Expected before Step 3: v2 snapshot, verification, and setup-cleanup rows fail.

- [ ] **Step 2: Write operation and cleanup RED tests**

Add:

- `test_hdf5_handle_delete_before_raise_rolls_back`;
- `test_hdf5_handle_delete_after_raise_closes_id_then_recreates_recovery`;
- `test_hdf5_handle_delete_after_raise_and_relink_failure_survives_reopen`;
- `test_hdf5_handle_recovery_recreation_failure_with_complete_restore_reports_old`;
- `test_hdf5_handle_recreation_and_public_restore_failure_reports_indeterminate`;
- `test_hdf5_handle_restore_sidecars_reports_all_failures`;
- `test_hdf5_handle_success_leaves_no_private_recovery_link`;
- `test_hdf5_handle_rollback_preserves_address_alias_refs_and_scales`;
- `test_hdf5_handle_repeated_success_has_no_private_recovery_object`;
- `test_hdf5_handle_incomplete_rollback_keeps_at_most_one_recovery_object`.

The final test stores ordinary object references, region references, and a
dimension-scale attachment to prove that rollback restores the same old physical
object, not an H5Ocopy replacement.

- [ ] **Step 3: Define explicit rollback state and snapshots**

Extend `_RollbackError` with stable `state`, `recovery_path`, and optional
`byte_state`/`position_state`. Add a raw root-attribute snapshot for the presence
and exact values of both v1 and v2. Store both snapshots in the recovery group and
flush/verify them before calling the native writer.

- [ ] **Step 4: Factor setup, public restore, sidecar restore, and cleanup**

Use separate helpers so every injection point has one responsibility:

```python
def _prepare_handle_recovery(...) -> _HandleRecovery: ...
def _restore_public_dataset(...) -> tuple[BaseException, ...]: ...
def _restore_root_sidecars(...) -> tuple[BaseException, ...]: ...
def _remove_or_recreate_recovery(...) -> str | None: ...
```

On success, deleting the verified recovery group is the commit. If deletion raises
after unlinking it, close the unlinked group ID before recreating a durable group
through the still-open old dataset ID. If durable recreation also fails, report
`state="old"` when the public dataset and both sidecars are nevertheless fully
restored. Use `state="indeterminate"` and `recovery_path=None` only when durable
recreation and public restoration are both incomplete.

- [ ] **Step 5: Verify the prewritten recovery-object bounds**

Run the two repeated-operation nodes written in Step 2. Expected: no private link
after ordinary success and at most one wrapper-created recovery object after an
incomplete rollback. Do not assert physical byte-size shrinkage for caller-owned
HDF5 handles; HDF5 free-space retention is accepted for this target form.

- [ ] **Step 6: Run handle tests GREEN and commit**

Run new handle tests and all preexisting handle/group rollback nodes.

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py -k 'open_ or handle_' -q
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_marker_mutation_failure_restores_all_current_targets -q
```

Expected: every selected node passes and no tracked test is RED.

Commit:

```bash
rtk git add gwexpy/timeseries/io/hdf5.py tests/timeseries/test_hdf5_exact_t0.py tests/timeseries/test_hdf5_exact_t0_transactions.py
rtk git commit -m "fix: preserve HDF5 handle identity during rollback"
```

---

### Task 9: Reconcile the legacy suite and qualify resource invariants

**Status:** planned

**Files:**

- Modify: `tests/timeseries/test_hdf5_exact_t0.py`
- Modify: `tests/timeseries/test_hdf5_exact_t0_transactions.py`
- Modify: `gwexpy/timeseries/io/hdf5.py` only for failures exposed by this task
- Modify: `gwexpy/timeseries/io/_hdf5_exact_epoch.py` only for failures exposed
  by this task

- [ ] **Step 1: Remove obsolete v1/path-authority assertions**

Update old helpers and tests intentionally changed by v2:

- historical v1 schema becomes v2 marker/token schema;
- alias expectations become one lineage record with diagnostic paths;
- quantized v1 state becomes native fallback and is removed by successful v2
  commit;
- GWpy-only overwrite no longer inherits exact time;
- missing marker never receives exact authority from any sidecar path.

Do not delete unrelated compatibility, external-storage, permission, provenance,
crop, or import-order coverage.

- [ ] **Step 2: Rerun the prewritten resource qualification nodes**

The resource tests were written RED before their production changes in Tasks 6
and 7. Run them together now:

```bash
rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_each_transaction_invokes_native_writer_once tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_path_repeated_overwrite_has_bounded_growth tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_disposable_stage_does_not_duplicate_old_dataset_storage tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_filelike_repeated_overwrite_has_bounded_growth tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_filelike_large_write_has_bounded_wrapper_rss -q
```

Expected: all nodes pass. This covers repeated 1 MiB replacements, the 16 MiB
native-image comparison, bounded file-like RSS, and one native writer call per
target without adding any acceptance test after implementation.

- [ ] **Step 3: Run the complete focused HDF5 suite**

Run:

```bash
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start pytest tests/timeseries/test_hdf5_exact_epoch_codec.py tests/timeseries/test_hdf5_exact_t0.py tests/timeseries/test_hdf5_exact_t0_transactions.py -q
```

Expected: all focused tests pass. Inspect the saved log and then run
`rtk git status --short` for generated changes.

- [ ] **Step 4: Run surrounding compatibility selectors**

Run these explicit selectors:

```bash
rtk conda run -n gwexpy pytest tests/qualification/test_v020_release_claims.py::test_timeseries_hdf5_roundtrip_retains_exact_t0_gps_ns tests/io/test_gwpy_hdf5_compat.py tests/io/test_hdf5_timeseries_family.py tests/timeseries/test_hdf5_layouts.py tests/timeseries/test_exact_gps_epoch.py tests/test_import_order.py -q
```

Record exact pass/fail/skip counts; do not label an environmental skip as a pass.

- [ ] **Step 5: Commit reconciliation fixes**

If Task 9 changed tracked files, run focused Ruff/MyPy and commit:

```bash
rtk git add gwexpy/timeseries/io/hdf5.py gwexpy/timeseries/io/_hdf5_exact_epoch.py tests/timeseries/test_hdf5_exact_t0.py tests/timeseries/test_hdf5_exact_t0_transactions.py
rtk git commit -m "test: qualify HDF5 exact epoch transactions"
```

---

### Task 10: Run static gates and independent reviews

**Status:** planned

**Files:**

- Modify only files required to fix findings within the approved P0 scope.
- Do not modify release metadata or public infrastructure.

- [ ] **Step 1: Run changed-file static checks**

```bash
rtk conda run -n gwexpy ruff check gwexpy/timeseries/io/hdf5.py gwexpy/timeseries/io/_hdf5_exact_epoch.py tests/timeseries/test_hdf5_exact_epoch_codec.py tests/timeseries/test_hdf5_exact_t0.py tests/timeseries/test_hdf5_exact_t0_transactions.py
rtk conda run -n gwexpy ruff format --check gwexpy/timeseries/io/hdf5.py gwexpy/timeseries/io/_hdf5_exact_epoch.py tests/timeseries/test_hdf5_exact_epoch_codec.py tests/timeseries/test_hdf5_exact_t0.py tests/timeseries/test_hdf5_exact_t0_transactions.py
rtk conda run -n gwexpy mypy gwexpy/timeseries/io/hdf5.py gwexpy/timeseries/io/_hdf5_exact_epoch.py
rtk git diff --check
```

Expected: all pass.

- [ ] **Step 2: Run repository source/test gates**

Run these exact jobs, using the emitted session name with the `tail` command:

```bash
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start ruff
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start mypy
rtk bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start pytest tests/timeseries tests/io/test_gwpy_hdf5_compat.py tests/io/test_hdf5_timeseries_family.py tests/qualification/test_v020_release_claims.py -q
```

Expected: MyPy and pytest pass. Full-repository Ruff may report only the recorded,
unchanged `docs_redesign/conf.py:242` D103; `ruff check gwexpy tests` must pass.
Preserve logs under `.agent/tmp/gwexpy_conda_jobs/` and report exact counts.

- [ ] **Step 3: Dispatch independent specification review**

Give a fresh reviewer only the approved spec, this plan, the implementation diff,
and focused test logs. Require an explicit Critical/Important/Minor classification.
Fix every in-scope Critical or Important finding using RED-first tests, rerun the
review, and commit each reviewed fix.

- [ ] **Step 4: Dispatch independent code-quality review**

After specification compliance is approved, give a different fresh reviewer the
same diff and logs. Require review of exception state, cleanup, type safety,
private API cohesion, test validity, and memory/storage assertions. Fix and
re-review all Critical or Important findings.

- [ ] **Step 5: Verify the final local state**

```bash
rtk git status --short --branch
rtk git log --oneline -12
rtk git diff origin/main...HEAD --check
```

Expected: only intentional committed changes, no generated artifacts, and no
public operation. Report focused/static/surrounding results and remaining P0/P1
boundaries. Do not mark P0 approved merely because this implementation plan is
complete.
