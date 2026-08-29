# HDF5 Exact-Epoch Identity and Transaction Design

## Status

The direction approved on 2026-08-29 binds exact epoch authority to a
dataset-local, GWpy-compatible canonical marker and a versioned root sidecar.
Independent review found that the first unit-relative Decimal encoding was
not viable and required the digits-only payload specified below. This revised
encoding and the clarified transaction limits await user approval before
implementation. The sidecar path is diagnostic only.

This design resolves the five release-blocking P0 findings recorded in
'docs/developers/reports/report_Phase1QualificationHandover_20260829_171704.md'.
P0 remains unapproved until the implementation, regression tests, independent
spec review, code-quality review, and required local gates pass.

No part of this design authorizes a tag, push, GitHub release, package upload,
workflow dispatch, conda-forge or Zenodo update, candidate qualification, P1
work, or Phase 2 work.

## Context

GWexpy keeps an exact GPS epoch as an integer number of nanoseconds in
'_gwex_t0_gps_ns'. GWpy's generic HDF5 representation stores the public axis
origin as binary64 'x0'. At large GPS epochs that projection loses
nanoseconds. For example, the exact value '1234567890123456789 ns' can return
as '1234567890123456955 ns'.

The current P0 branch stores the exact integer in a root JSON sidecar indexed
by a dataset path. Direct round trips pass, but path authority is unsafe:

- hard-link and soft-link aliases reach the same dataset through another path
  and can lose exact authority;
- a GWpy-only overwrite can replace the dataset while leaving the old
  path-keyed exact epoch behind;
- HDF5 object addresses, tokens, and references are not durable replacement
  identities in the supported h5py/HDF5 stack;
- an unknown private dataset attribute is forwarded by the GWpy 4 reader to
  'TimeSeries(...)' and raises 'TypeError'.

The final review also found a reload registration defect, safe native path
regressions, a cleanup atomicity gap, full-data preflight duplication, full
buffer duplication for file-like targets, and file growth caused by retaining
rollback hard links inside disposable staging files.

## Goals

1. Recover the exact integer epoch through direct paths, group aliases,
   dataset hard links, internal soft links, moves, and attribute-preserving
   HDF5 object copies.
2. Never apply an old exact epoch to a replacement dataset.
3. Preserve GWpy 4 read compatibility for every dataset attribute written by
   GWexpy.
4. Preserve GWpy's accepted absolute string and UTF-8 byte path forms while
   retaining the existing NUL, raw-dot-component, and external-link checks.
5. Make an exception observable only when the caller-owned target has its old
   state restored, an explicit recovery artifact, or a structured
   indeterminate report that includes failure to persist that artifact.
6. Write the dataset once per successful pathname or file-like transaction.
7. Excluding caller-owned storage, the input array, and h5py caches, bound
   additional file-like wrapper heap to 'O(chunk)'.
8. Keep module reload idempotent without recursively wrapping registry
   handlers.

## Non-goals

- A globally unique physical HDF5 object identifier.
- Dataset-specific provenance resolution. The P0 sidecar keeps 'provenance'
  empty and must not use the exact-time marker to distinguish two datasets
  that have the same exact axis state.
- Exact-time support for HDF5 external storage.
- Changes to GWpy, h5py, other GWexpy formats, collection layouts, or public
  TimeSeries constructor semantics.
- P1 lazy-registration work or any release operation.

## Rejected identity carriers

### Unknown private dataset attribute

A UUID attribute would follow hard links and object copies, but GWpy 4 passes
every dataset attribute to its array constructor. An attribute such as
'_gwexpy_dataset_token' therefore breaks GWpy-only reads with an unexpected
keyword error.

### HDF5 object reference, token, or address

Aliases share these values, but delete-and-recreate operations reused the
address and token in local tests with h5py 3.16.0 and HDF5 1.14.6. A root
object reference also resolved to the replacement object after reuse. These
values can help diagnostics but cannot authorize exact metadata.

### Object comment

An object comment follows aliases and ordinary object copies, but HDF5 exposes
one user comment per object. Using it would overwrite caller metadata, and the
low-level comment API is deprecated or version-sensitive in the supported
stack.

### Hidden hard link

A hidden link would retain identity, but it also keeps overwritten dataset
storage alive, changes path-less dataset discovery, and recreates the file
growth defect already found in disposable staging.

## Dataset-local exact marker

### Carrier

The wrapper writes its marker to the standard dataset 'epoch' attribute.
GWpy 4 recognizes 'epoch' and converts its numeric string to the public axis
origin, so the attribute does not introduce an unknown constructor keyword.
The marker is written only after the native writer has created the dataset and
stored 'x0' and 'xunit'.

The numeric value must be the dataset's native 'x0' in its own 'xunit'.
Encoding the mathematically exact unit conversion does not work: Astropy's
integer-to-binary64 conversion and its two directional unit factors can
produce a native 'x0' one ULP away from the exact Decimal quotient. A
35,000-case probe across 's', 'ms', 'us', 'ns', 'min', 'ks', and 'day' found
15,624 such mismatches. The marker therefore stores exact nanoseconds in a
low-significance payload rather than in the visible numeric value.

### Canonical numeric prefix

For finite raw binary64 'x0', the encoder:

1. builds the exact decimal value with 'Decimal.from_float(raw_x0)';
2. formats it in fixed-point notation without an exponent;
3. preserves negative zero and writes an explicit sign;
4. ensures that the prefix contains a decimal point followed by at least one
   fractional digit, using '.0' for an integer;
5. appends exactly 400 zero guard digits;
6. appends a decimal-triplet payload.

If the canonical prefix has 'p >= 1' fractional digits, the appended tail
changes its magnitude by less than '10^-(p+400)', and therefore by less than
'10^-401'. For zero this is still far below half the minimum binary64
subnormal, approximately '2.47e-324'. The formatted marker therefore rounds
to the same binary64 for every finite 'x0', including positive and negative
zero, minimum subnormal, minimum normal, and maximum finite values. The
encoder and decoder still compare the resulting bits with raw 'x0'; the proof
is not used as a reason to skip the runtime check.

The marker uses decimal digits, one sign, and one decimal point only. It does
not depend on Python's underscore syntax or exponent parsing.

### Versioned payload

The binary payload has this field order:

~~~
magic = b"GWEXH5T0"
version = uint8(2)
lineage_token = 16 random bytes
x0_bits = 8 bytes, big-endian IEEE-754
xunit_length = unsigned big-endian uint16
xunit_utf8 = canonical Astropy unit
xunit_to_ns_bits = 8 bytes, big-endian IEEE-754 binary64
ns_to_xunit_bits = 8 bytes, big-endian IEEE-754 binary64
epoch_sign = uint8(0 or 1)
epoch_magnitude_length = unsigned big-endian uint16
epoch_magnitude = minimal unsigned big-endian bytes
digest = SHA-256(domain_separator || all preceding fields)
~~~

The domain separator is the fixed ASCII byte string
'gwexpy.hdf5.exact-epoch-marker.v2' followed by one NUL byte. SHA-256 detects
accidental corruption; it does not authenticate a file against a malicious
writer.

Each payload byte is encoded as exactly three decimal digits from '000'
through '255'. Values above 255, a non-multiple-of-three payload length,
trailing data, a non-minimal magnitude, negative zero, an invalid length, a
bad digest, or a non-canonical re-encoding is rejected.

'epoch_sign' is zero for non-negative values and one for negative values.
Zero has sign zero, magnitude length one, and the sole magnitude byte '0x00'.
Every other magnitude begins with a nonzero byte.

The limits are:

- marker: 4,096 ASCII characters;
- canonical unit: 255 UTF-8 bytes;
- epoch magnitude: 512 bytes.

The limits accommodate the longest finite-binary64 prefix together with every
schema-permitted unit and epoch magnitude. Exact output exceeding either
field limit is rejected before mutation. The maximum schema construction is
4,036 characters.

### Binding and decode

The payload carries exact integer nanoseconds directly. Decoding never derives
exact authority by rounding a decimal unit conversion.

The decoder performs these checks before calling the native reader:

1. Enforce the 4,096-character cap and ASCII-only grammar.
2. Parse the marker itself as one finite binary64 'f', without consulting the
   dataset's raw 'x0'.
3. Reconstruct the canonical signed fixed-point prefix from
   'Decimal.from_float(f)'. The only candidate boundary is immediately after
   that prefix and exactly 400 zero guard digits.
4. Require triplet-encoded 'magic' at that boundary, then enforce the payload
   digest, field-length, magnitude, and canonical byte rules. A guard-plus-
   magic pattern elsewhere can make a failed boundary recognizable as
   malformed, but is never parsed as another candidate; legal payload bytes
   may contain the same pattern.
5. Decode and canonicalize the dataset's actual 'xunit'.
6. Require the reconstructed prefix, embedded 'x0_bits', payload 'xunit', and both
   directional conversion-factor bits to equal the current dataset
   attributes and Astropy conversion factors.
7. Re-encode the complete marker byte-for-byte.
8. Require 'float(marker)' to have raw 'x0_bits'.

The decoder does not recompute 'x0' from exact nanoseconds. Exact slicing and
cropping update the integer authority exactly but can update public 'x0'
through binary64 addition, which may differ by one ULP from a fresh
integer-to-unit conversion. The marker digest binds exact nanoseconds to the
observed native 'x0'; the public projection invariant is the marker float-bit
check above.

This establishes:

~~~
decode(encode(epoch_ns, raw_x0, axis_unit)) == epoch_ns
float_bits(encode(epoch_ns, raw_x0, axis_unit)) == float_bits(raw_x0)
~~~

Non-finite 'x0', an absent or invalid time unit, a non-finite conversion
factor, or a marker-to-'x0' bit mismatch is rejected for exact output before
caller state is mutated.

### Meaning of identity

The random token is a lineage identifier, not a permanent physical HDF5
object ID. Hard-link aliases share it. An attribute-preserving object copy
also copies it, so the copy belongs to the same exact-time lineage even
though it has another HDF5 address. A new GWexpy exact write receives a new
token unless the caller supplies a fully valid, equivalent v2 marker
explicitly.

Two independent datasets may have the same exact epoch while retaining
different tokens. The reader must not use exact-time equality alone to attach
dataset-specific provenance or other future metadata. P0 keeps provenance
empty.

This semantic identity is sufficient for P0:

- aliases expose the same dataset attribute;
- a move preserves it;
- an attribute-preserving 'H5Ocopy' preserves it;
- a GWpy overwrite creates a new dataset without the marker;
- an attribute-free copy loses the marker and therefore loses exact
  authority rather than receiving stale authority.

## Sidecar schema v2

### Storage

The new root attribute is '_gwexpy_sidecar_json_v2'. It contains a strict JSON
document:

~~~json
{
  "schema": "gwexpy.hdf5.sidecar",
  "version": 2,
  "records": {
    "<32-lowercase-hex-token>": {
      "binding": {
        "marker_sha256": "<64-lowercase-hex-digits>",
        "x0_bits": "<16 lowercase hex digits>",
        "xunit": "<canonical unit string>",
        "xunit_to_ns_bits": "<16 lowercase hex digits>",
        "ns_to_xunit_bits": "<16 lowercase hex digits>"
      },
      "metadata": {
        "_gwexpy_t0_gps_state": {
          "_gwex_t0_gps_ns": 1234,
          "precision": "exact"
        }
      },
      "provenance": {},
      "paths": ["diagnostic/path"]
    }
  }
}
~~~

JSON validation remains strict:

- no duplicate members;
- no NaN or Infinity constants;
- exact key sets at every schema-owned level;
- integer type checks that reject booleans;
- UTF-8 decoding without replacement;
- canonical hex widths for binary64 fields;
- canonical HDF5 diagnostic paths;
- 'provenance' exactly equal to '{}'.

'x0' and conversion factors are stored as binary64 bit strings rather than
JSON numbers. 'xunit' is parsed with Astropy and serialized to one canonical
unit spelling. The record key is the lineage token carried by the marker.
The full binding, not a path, confirms an exact-time record.

'marker_sha256' is
'lowercase_hex(SHA-256(complete_marker.encode("ascii")))'. It is distinct
from the domain-separated digest inside the marker payload.

The schema limits are:

- root JSON attribute: 8 MiB of UTF-8;
- records: 10,000;
- record ID: exactly 32 lowercase hexadecimal characters;
- diagnostic paths per record: 16;
- one diagnostic path: 4,096 UTF-8 bytes;
- canonical unit: 255 UTF-8 bytes;
- marker digest: exactly 64 lowercase hexadecimal characters.

Serialized record keys and each record's unique diagnostic paths are sorted.
Paths are finite representatives from a cycle-safe traversal, not the
lexicographically smallest names from the alias-expanded graph. HDF5 groups
can contain hard-link cycles, so the traversal sorts link names, tracks
visited group and dataset identities, never revisits a group identity, and
records at most one simple representative path per physical dataset object.
If copied objects with one lineage produce more than 16 representatives, the
first 16 in that deterministic traversal are retained.

### Whole-document validation

The reader validates every v2 record before selecting a dataset:

1. Parse all fields and enforce the document limits.
2. Reconstruct the complete canonical ASCII marker, including numeric prefix,
   400-digit guard, and payload, from the token, exact integer, 'x0_bits',
   unit, and conversion-factor fields.
3. Require 'marker_sha256' to match that reconstructed marker.
4. Require both conversion-factor fields to match the canonical unit.
5. Require metadata exact nanoseconds to equal the payload value.
6. Require 'provenance' to be empty.

One inconsistent record invalidates the complete v2 document and raises
'ValueError'. Sidecar metadata never overwrites the integer decoded from the
dataset marker.

### Version 1 treatment

The existing '_gwexpy_sidecar_json_v1' layout is unpublished and path-bound.
It never authorizes an exact epoch under this design. A v1-only file reads
through the native/quantized path. A successful v2 transaction removes the
v1 root attribute as part of the same atomic sidecar update.

A malformed v2 document raises 'ValueError'. The reader does not fall back to
v1 after seeing an invalid v2 claim.

### Sidecar maintenance

Paths record where a binding was observed during a successful write. They can
be stale after an external move and are never consulted for authority.

On a successful GWexpy transaction, sidecar compaction rebuilds the live
record set from reachable local datasets. The scan:

- walks sorted local hard links with cycle detection, records finite
  representative paths, and decodes each object identity once;
- reads object attributes only;
- excludes the private rollback namespace;
- does not dereference soft or external links;
- adds a marker-only copied dataset to v2;
- removes records whose token is no longer reachable;
- requires every occurrence of one token to carry the same marker and
  fingerprint.

A recognizable malformed marker on any scanned local dataset aborts the
transaction and triggers rollback. Ordinary non-v2 'epoch' metadata is
ignored. The size and record limits are checked before the root attribute is
updated.

This prevents repeated replacements from growing the root JSON indefinitely.

## Read authority policy

The reader first resolves the dataset through GWpy-compatible native path
semantics. It then uses the resolved 'dataset.file', not the referring
container's file, for v2 sidecar validation. This distinction preserves reads
through an HDF5 ExternalLink without attaching metadata from the wrong file.

Raw 'epoch', 'x0', and 'xunit' attributes are inspected before exact authority
or the native constructor is invoked.

| Dataset marker | Matching v2 record | Result |
|---|---|---|
| absent or ordinary non-v2 'epoch' | any stale path record | native/quantized result |
| valid canonical v2 marker | token absent | recover exact nanoseconds from the marker |
| valid canonical v2 marker | token and full binding match | attach marker exact nanoseconds |
| valid canonical v2 marker | token record conflicts | raise 'ValueError' |
| recognizable but malformed v2 marker | any | raise 'ValueError' |

Marker-only recovery preserves exact time when an HDF5 object is copied to a
different file without its root sidecar. A user-supplied 'epoch' is treated as
an exact marker only if it satisfies the full canonical v2 grammar, payload,
digest, unit binding, marker-to-stored-'x0' float-bit projection, and
byte-for-byte re-encoding.

A missing marker is a loss of exact authority, not a read failure. This rule
is what makes a stale v1/v2 path after a GWpy-only overwrite safe.

### GWpy and GWexpy construction

GWexpy's public constructor interprets string 'epoch' values as GPS seconds
before normalizing to 'xunit'. Passing the dataset marker directly to that
constructor can overflow, fail, or scale non-second axes twice.
The HDF5 reader therefore:

1. validates and decodes candidate marker bytes from raw attributes;
2. always asks the saved native GWpy reader to construct a GWpy
   'TimeSeries', for marked, unmarked, and ordinary-'epoch' datasets;
3. lets GWpy consume the numeric 'epoch' in the dataset's axis unit;
4. views the result as the GWexpy target class without invoking its
   constructor again;
5. attaches the already validated '_gwex_t0_gps_ns';
6. applies 'start'/'end' cropping after exact authority is attached.

Direct construction as a GWexpy 'TimeSeries' is forbidden for a marker.
Depending on 'xunit', that route can overflow, reject the value, or scale the
axis twice. The marker's float projection equals raw 'x0', so native GWpy
construction does not change the public axis.

## Write metadata policy

Exact authority comes from the array's integer '_gwex_t0_gps_ns'. Before any
mutation, the wrapper validates the axis unit, builds the canonical marker,
and examines caller 'attrs'.

- 'attrs["xunit"]' must have the same canonical representation, scale, and
  directional factor bits as the array axis unit. Time equivalence alone is
  insufficient.
- 'attrs["x0"]' must be a finite scalar whose binary64 bits equal the expected
  native 'x0'.
- An ordinary numeric 'attrs["epoch"]' is accepted only when its float bits
  equal expected 'x0'; the wrapper replaces it with a new canonical marker.
- A valid v2 'attrs["epoch"]' is accepted only when exact nanoseconds and the
  complete fingerprint agree; its lineage token is retained.
- A recognizable malformed marker raises and is never reclassified as
  ordinary caller metadata.
- Native output is reset to canonical 'x0', 'xunit', and marker values and
  revalidated before sidecar commit.
- Non-exact writes preserve ordinary caller metadata and do not invent a
  marker.
- Non-exact writes reject recognizable v2 marker metadata because the source
  array has no exact authority for it.
- Exact output with HDF5 external storage remains unsupported and raises
  before mutation.

If a non-exact write replaces a marked dataset through the GWexpy wrapper, the
successful sidecar compaction removes the unreachable binding. External
storage retains the existing conservative rule: it cannot replace a
sidecar-managed or canonically marked dataset, and 'external=' rejects
canonical or recognizable v2 caller 'epoch' metadata even for a non-exact
array.

## Native path compatibility

User paths and sidecar coordinates are separate concepts.

### Native path

The wrapper accepts the same safe native forms as GWpy:

- relative 'str';
- absolute 'str';
- relative UTF-8 'bytes';
- absolute UTF-8 'bytes'.

It passes the original object unchanged to the native reader or writer.
Validation rejects NUL, invalid UTF-8, empty components, '.', and '..'. It
does not reject a leading slash.

Write resolution rejects an ExternalLink at the leaf or any ancestor,
including an internal soft-link chain that reaches an external object. It
also rejects a SoftLink at the leaf because dataset rollback cannot preserve
that link's type and target. A soft-linked local ancestor group remains
allowed after it is proven to resolve within the same file. The check
completes before the native writer or external raw file is touched.

Read resolution preserves GWpy behavior and may follow an ExternalLink. Exact
authority then comes from the resolved dataset marker and the resolved
dataset's file-side v2 document. A sidecar in the referring file is never
applied to an external dataset.

### Transaction coordinate

For relative paths, overwrite and rollback operations are rooted at the
caller-provided 'h5py.Group' or 'h5py.File'. For absolute paths, they are
rooted at 'container.file'. The canonical path returned by the created
dataset is used only for diagnostics.

The tests cover pathname, open 'h5py.File', nested 'h5py.Group', and binary
file-like sources and targets.

## Reload-safe registration

Registered wrapper functions retain private references to their native reader
and writer. Registration follows these rules:

1. Resolve the current registry handlers.
2. If both are marked wrappers, recover their saved native handlers into the
   reloaded module globals and keep the existing wrappers registered.
3. If neither is wrapped, resolve the native GWpy handlers, create one wrapper
   pair, and save the native references on each wrapper.
4. Reject a half-wrapped or recursively wrapped state as an invariant error.

Because 'importlib.reload' reuses the module dictionary, old wrappers see the
restored globals. Repeated imports and at least two consecutive reloads must
not change wrapper identity or nesting depth.

Every recovered native handler must be callable and unmarked. A wrapper is
never accepted as its own saved base handler.

## Transaction architecture

The implementation has three write layers.

### Validation

Validation checks exact-marker construction, caller metadata, path syntax,
external-link policy, target type, and existing sidecar syntax. It never
writes array data. The current HDF5 core-driver data-writing preflight is
removed.

Native dataset-creation errors are handled inside one of the two transactional
layers below.

### Disposable stage writer

Filesystem-path and file-like transactions write to a disposable staging
container. The stage writer:

1. calls the native writer exactly once;
2. writes and validates the dataset-local marker when exact authority exists;
3. writes or compacts sidecar v2;
4. closes the stage before external commit.

It does not create an in-file rollback hard link. The entire stage can be
discarded, so an inner rollback would retain the old dataset unnecessarily
and double the staged file size.

### Caller-owned open-container transaction

An open 'h5py.File' or 'h5py.Group' cannot be replaced externally. For an
existing dataset, the wrapper:

1. snapshots the presence and raw values of both v1 and v2 root sidecar
   attributes;
2. holds the original dataset handle open;
3. creates one private rollback group;
4. links the original dataset into that group;
5. stores both sidecar snapshots and their presence flags in that group;
6. verifies that the recovery artifact is complete;
7. invokes the native writer;
8. writes the marker and sidecar;
9. deletes the rollback group as the final commit operation.

Creation of the group, dataset link, each sidecar snapshot, and verification
belongs to the transaction envelope. A failure before the recovery artifact
is complete leaves the public dataset and sidecars unchanged and removes the
partial private group. If that cleanup also fails, the structured error
reports 'state="old"' and the partial recovery path.

Rollback-link deletion is part of success, not post-success cleanup. If it
raises, the wrapper first determines whether the original recovery link still
exists. If cleanup deleted the link before raising, the wrapper immediately
closes the now-unlinked rollback-group ID, then uses the still-open original
dataset ID to create a new private recovery group, re-link the old dataset,
and persist both v1 and v2 sidecar snapshots. This ordering keeps at most one
wrapper-created recovery link. Only then does it remove the new public link,
restore the old public link and sidecars, and remove newly created empty
parent groups.

The recovery link is not deleted until the public dataset, both sidecar
attributes, and parent-group state have all been restored. A recovery cleanup
failure after successful public restoration raises a structured error with
'state="old"' and a durable recovery path. If a durable recovery link cannot
be created and public restoration is also incomplete, the error uses
'state="indeterminate"' and 'recovery_path=None'.

'_RollbackError' reports:

- the triggering operation;
- every rollback failure;
- 'state' as 'old', 'new', or 'indeterminate';
- the retained recovery object path, or an explicit 'None' after recovery
  persistence itself failed.

The original dataset handle is not closed until commit or rollback has
finished.

Keeping the old HDF5 object alive is required to preserve incoming object
references, region references, dimension-scale attachments, committed-type
relationships, and object identity. It can force HDF5 to allocate the new
dataset elsewhere and leave freed space in the physical file. P0 therefore
does not promise physical file-size stability for caller-owned open
containers. It does require:

- no private rollback link after ordinary success;
- no reachable old object unless a pre-existing hard-link alias requires it;
- at most one wrapper-created old-object recovery link during the operation.

An external H5Ocopy backup is not used for this case because delete-and-copy
restoration changes object identity and can break incoming references and
dimension-scale relationships.

## Filesystem-path transaction

The wrapper creates a secure sibling staging file.

- 'lstat' accepts only a nonexistent target or an existing regular file.
  Directories, symbolic links, FIFOs, sockets, and devices are rejected before
  staging.
- An existing regular target with 'st_nlink > 1' is rejected because atomic
  replacement would split filesystem hard-link aliases, unlike native
  in-place truncation.
- An existing target is copied only for native 'append=True' semantics.
- 'overwrite=True, append=False' starts from an empty stage, matching
  pathname-mode GWpy behavior.
- The disposable stage writer performs one dataset write.
- File mode is preserved for an existing regular target.
- 'os.replace' commits the closed stage atomically.

A failed stage leaves the original path unchanged. If the primary operation
and stage cleanup both fail, a structured error reports both exceptions,
'state="old"', and the retained stage path.

Atomic replacement preserves file contents and the explicitly restored mode,
but it does not promise inode identity, owner, ACL, or platform-specific
extended metadata. This limitation is recorded rather than implying native
in-place metadata semantics.

## File-like transaction

The wrapper uses two disk-backed temporary files and a fixed-size binary copy
loop:

- backup: a secure named mode-0600 file completed, flushed, and fsynced before
  commit;
- working: a copy initialized from the original bytes for every mode.

The working temporary is passed to h5py as a file object, not opened by its
pathname. This preserves native h5py file-object behavior: in the supported
stack, 'overwrite=True, append=False' can preserve unrelated existing
datasets even though pathname mode truncates the file.

The complete envelope begins with the first target 'seek' and ends only after
the final position is established:

1. save the caller's original position;
2. copy target to backup and working in bounded chunks;
3. set the working file to the caller's original position;
4. run the disposable stage writer once on the working file object;
5. close the HDF5 stage and record the resulting working-file position;
6. copy working to target in bounded chunks;
7. call 'truncate(final_size)', flush, and set the committed position.

Any failure before target bytes change still restores the original caller
position. If commit fails, backup is copied back in bounded chunks and the
original position is restored. The error classifies byte state and position
state independently. An incomplete byte rollback retains the already durable
named backup and reports its path.

After successful commit and final position establishment, the wrapper closes
the working file, closes the backup, and unlinks the named backup. If working
or backup cleanup fails, it returns the successful write and emits a
'ResourceWarning' containing the cleanup exception, 'state="new"', and the
retained mode-0600 path when one exists; it does not raise an exception after
publishing new bytes.

After a complete byte and position rollback, both temporary files are closed
and the named backup is unlinked. Cleanup failure is added to
'_RollbackError' with 'state="old"' and the retained path when one exists. If
byte or position rollback is incomplete, the backup remains available and
its path is reported. An anonymous working file is never an authority; its
cleanup failure has 'recovery_path=None'. Normal success and complete
rollback leak no temporary file.

The copy helper accepts bytes-like reads of at most the requested chunk. It
loops over short positive writes. A zero, negative, 'None', or oversized
write, and an oversized or non-bytes read, is an error. EOF is the only empty
read. Truncation always receives the explicit final byte length.

Excluding the caller-owned buffer, input array, and h5py caches, the wrapper's
additional Python heap is 'O(chunk)'. Disk use is proportional to the
original and staged HDF5 images. 'BytesIO(snapshot)', full 'bytes' snapshots,
and 'getvalue()' are not used.

## Error policy

- Detect marker, metadata, path, sidecar, target, and unsupported-unit errors
  before mutating caller-owned state when possible.
- Once mutation begins, catch failures across native write, marker write,
  sidecar update, and final rollback cleanup.
- Re-raise the original operation error after a complete rollback.
- Raise '_RollbackError' when restoration or recovery cleanup is incomplete,
  state cannot be classified, or more than one error must be reported.
- Never apply exact authority after any marker/fingerprint ambiguity.
- Never hide an invalid v2 document by falling back to native data.

## Test-driven implementation order

Production edits start only after the corresponding focused tests demonstrate
the old failure.

### 1. Dataset marker and authority

Add RED regressions for:

- direct path, group hard alias, dataset hard alias, and internal soft alias;
- dataset move and rename;
- pathname, 'h5py.File', 'h5py.Group', and 'BytesIO' reads;
- GWpy-only overwrite with a different epoch and with the same native float
  projection;
- independent datasets with the same exact epoch;
- positive, negative, and zero epochs;
- exact slices and crops whose accumulated public 'x0' differs by one ULP
  from a fresh exact-nanosecond unit conversion;
- 's', 'ms', 'us', 'ns', 'min', 'ks', and 'day' axes;
- both binary64 conversion-factor directions and a custom scaled time unit;
- marker float-bit preservation at positive/negative zero, minimum
  subnormal, minimum normal, maximum finite, and representative random
  finite values;
- decimal guard, triplet range, payload length, magic, token, magnitude,
  digest, factor, unit, float-bit binding, re-encoding, and
  4,096-character cap;
- conflicting user 'epoch', 'x0', and 'xunit' attributes;
- default cross-file 'H5Ocopy' and copy-without-attributes;
- missing marker, malformed marker, tampered marker, tampered sidecar, bit
  fingerprint mismatch, token collision, and conflicting records;
- strict document key sets, cross-field validation, empty provenance, sorted
  bounded paths, JSON size, and record-count limits;
- compaction with marker-only copies, stale records, and an unrelated
  malformed local marker;
- compaction through a hard-linked group self-cycle, group aliases, and
  rollback-namespace names that sort before public names;
- v1-only and mixed v1/v2 files;
- GWpy-only readability of every file produced by the wrapper.

### 2. Reload and path compatibility

Use clean subprocesses to test:

- import followed by two module reloads;
- read and write after every reload;
- stable wrapper identity and one native invocation;
- alternate import order;
- relative and absolute string paths;
- relative and absolute UTF-8 byte paths;
- all source/target container forms;
- retained rejection of NUL, invalid UTF-8, raw '.', and raw '..';
- write rejection for direct external parents, external leaves, and
  soft-to-external chains;
- write rejection for a leaf SoftLink, while a proven local soft-linked
  ancestor group retains native behavior;
- native read resolution through those external forms, with authority taken
  only from the resolved dataset file;
- callable, unmarked saved native handlers and half-wrapped registry failure.

### 3. Transaction atomicity

Inject failures at:

- native dataset creation;
- metadata write;
- marker write;
- sidecar write;
- rollback-group creation, old-dataset linking, v1 snapshot, v2 snapshot,
  recovery verification, and partial-group cleanup;
- rollback-link deletion before deletion;
- rollback-link deletion after deletion;
- recovery-link recreation after delete-then-raise;
- simultaneous delete-then-raise and public dataset relink failure, followed
  by close/reopen verification that the durable recovery path resolves to the
  old dataset;
- dataset restoration;
- sidecar restoration;
- path replacement;
- stage unlink after path-replacement failure;
- path targets that are directories, symlinks, FIFOs, sockets, devices, and
  multiply-linked regular files;
- file-like initial seek/read, backup copy, working copy, stage open/close,
  native write, marker, sidecar, short write, truncate, flush, position
  restore, backup restore, working close, backup close, and backup unlink
  failures;
- no temporary-file leak after normal success or complete rollback, and a
  classified warning after commit-success cleanup failure.

Every raised operation must assert old/new public data, native 'x0', exact
authority, sidecar state, private rollback objects, parent groups, caller
position, independently classified byte/position state, and any recovery
path.

### 4. Memory and storage behavior

Assert:

- at most one native dataset-writer call for pathname, file-like, File, and
  Group success and failure paths;
- no core-driver preflight;
- no 'BytesIO' snapshot or 'getvalue()';
- bounded chunk sizes during backup, commit, and rollback;
- unrelated file-like datasets retain native behavior;
- repeated pathname and file-like overwrite does not grow the HDF5 image
  linearly or retain a rollback object;
- successful open-handle overwrite leaves no private recovery link and at
  most the old objects required by pre-existing aliases; physical file size
  is not asserted for this identity-preserving case;
- a representative large file-like write has bounded RSS relative to native
  behavior, using a separate process and a conservative non-flaky limit.

### 5. Existing regression surface

Keep every current exact-time, path, link, external-storage, sidecar,
transaction, cropping, and native-compatibility test unless the v1 expectation
is intentionally replaced by the v2 contract above.

## Verification gates

Run commands in the project's 'gwexpy' conda environment and preserve the
registry bootstrap required by the repository.

1. Focused 'tests/timeseries/test_hdf5_exact_t0.py'.
2. Qualification exact node and existing HDF5 compatibility selectors.
3. TimeSeries and I/O contract suites affected by registration.
4. 'ruff check' for changed production and test files, then the repository
   source/test scope.
5. Non-mutating formatting check.
6. MyPy for changed production files, then 'mypy gwexpy'.
7. 'git diff --check'.
8. Independent spec review against this document.
9. Separate code-quality review.
10. Fresh physics/maintainer review before P0 approval.

The existing unrelated 'docs_redesign/conf.py:242' D103 finding remains
outside P0 unless a separate scope decision changes that classification.

## Acceptance criteria

P0 can be proposed for approval only when all of the following are true:

- exact epoch survives every supported alias and copy case;
- no stale exact epoch is attached after a GWpy-only replacement;
- GWpy 4 reads wrapper-produced files without unknown metadata errors;
- the digits-only marker satisfies payload decode, native float-bit, unit
  binding, checksum, and canonical re-encoding invariants;
- reload and native path compatibility regressions pass;
- every injected cleanup failure restores old state, reports a retained
  recovery location, or explicitly reports artifact-persistence failure with
  'state="indeterminate"' and 'recovery_path=None', except that post-commit
  temporary cleanup reports committed 'state="new"' through the classified
  'ResourceWarning' and returns success;
- every target form invokes the native dataset writer at most once;
- excluding caller-owned storage, the input array, and h5py caches,
  file-like wrapper heap is 'O(chunk)';
- repeated pathname and file-like replacement neither retains nor writes an
  extra in-stage old dataset;
- successful open-handle replacement leaks no private link; its documented
  HDF5 free-space limitation is not misreported as a live-object leak;
- focused and surrounding regression gates pass;
- independent spec and code-quality reviewers approve.

Even after these criteria pass, candidate qualification and public release
remain blocked until P1 and the later handover gates are separately completed
and approved.
