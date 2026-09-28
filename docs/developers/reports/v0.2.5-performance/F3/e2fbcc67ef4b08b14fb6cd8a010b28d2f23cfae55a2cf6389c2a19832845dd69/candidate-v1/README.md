# F3 native PSD candidate evidence

This append-only package compares the frozen B1 old-R wheel with candidate
source commit `96f6c72f11ea318e8fc9f1ccddc32e29fc300d05`. It is candidate
evidence for review, not an integration or release approval. The baseline is
the immutable sibling `baseline-v1`, frozen at source commit
`000f1f4988fef13ed33eaf37661ca8deef4561d3` with harness digest
`e2fbcc67ef4b08b14fb6cd8a010b28d2f23cfae55a2cf6389c2a19832845dd69`.
The release controller loaded the frozen harness from a separate checkout at
that commit; later benchmark edits in the candidate worktree were not used.

## Build and installation identity

- B1 source is old R `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`; its
  previously frozen wheel SHA-256 is
  `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- Candidate wheel SHA-256 is
  `22ed237cbaa158bae08267d158eb9bf91e8bb0fd0ea835ef39447ed1efb690c0`.
  It was built from a clean source commit with
  `/home/washimi/miniforge3/envs/main/bin/python -m build --wheel --no-isolation --outdir /tmp/gwexpy-v025-f3-native-psd-wheel-release-v2`, then installed in a fresh
  `python -m venv --system-site-packages` environment with
  `python -m pip install --no-deps --no-index <wheel>`.
- Both arms use Python 3.12.12 and the same inherited package environment.
  The arm manifests record all distribution versions, `-I` import path,
  version 0.2.5, wheel hash, and exact audit of all 369 installed `gwexpy`
  package files against each wheel. An independent candidate wheel/source
  audit found 369/369 package bytes equal to the committed source tree.
  The wheel is reproducible from the source commit; the wheel itself is not
  checked into this evidence package.

## Correctness and structural gate

`correctness-public.json` covers 27 cases on native, external-requested, and
forced-fallback public routes: 81 route outcomes. The complete result,
warning, and log fingerprint equals B1 in 79 routes. The two deliberate
native-only differences are `psd_unselected_short_payload_warning` and
`unselected_payload_fault`: B1 warns about a fully unselected PSD payload,
while this candidate emits no warning. Those are the separately approved
#589 exception. Both outcome fingerprints and logs are unchanged. External,
TF/6, FFT, STF, TS, gzip, namespace, EOF, and selected-payload cases retain
B1 fingerprints. External and the other native product routes have no
performance claim here.

`metadata-extension` contains 15 additional B1/candidate exact public
comparisons, its deterministic fixture generator and capture source, fixture
manifest, and source XML. It covers nonfinite and negative metadata, invalid
encoding/BUnit, frequency-axis overflow, large N, and NumPy underflow modes.
All 15 outcome/warning/log fingerprints match. The manifest records fixture
bytes and SHA-256; paths reflect the capture host's `/tmp` roots.

`structure-large.json` is separate from timing and memory. The selected
native PSD decodes 65,536 bytes. B1 decoded 16,777,216 fully unselected bytes
in all five frozen structural samples; the candidate decodes zero, with zero
unknown decoder calls and zero unselected series constructions. Source review
confirms the candidate still uses the instrumented native `base64.b64decode`
call for selected data and skips the same helper for fully unselected data.

## Independent nine-sample runs

`memory`, `warm`, and `cold` are separate uninstrumented/measurement runs,
each with nine samples per arm in the predeclared
`ABBA · BAAB · ABBA · BAAB · AB` order. Each directory contains the controller
manifest, every raw sample, raw arm arrays, median/MAD summary, and a verified
artifact-hash map. `release-controller.py` is the exact controller bytes,
SHA-256 `4adfa1f8b86db62ed83b62dff76b69ab9b7e0665f05092d4df1e415547d607a3`.

| Metric | B1 median (MAD) | Candidate median (MAD) | Role |
| --- | ---: | ---: | --- |
| Linux process-tree peak PSS | 364,173 (86) KiB | 292,215 (339) KiB | Co-primary, 19.76% lower; required threshold 5% |
| Warm function wall | 102.748 (2.683) ms | 110.151 (3.092) ms | Supporting, 7.21% slower |
| Warm function CPU | 102.682 (2.749) ms | 110.091 (3.031) ms | Supporting |
| Cold fresh-process wall | 2,355.613 (22.337) ms | 2,351.300 (14.129) ms | Supporting |

Peak PSS is sampled on Linux and can miss a shorter instantaneous peak.
The memory run is distinct from uninstrumented warm/cold. The primary
relative-MAD threshold is `max(5%, 2*(MAD_B1/median_B1 + MAD_candidate/median_candidate))`;
it evaluates to 5% here. Warm wall is a supporting regression to review; this
package does not claim a wall improvement. No separate small-input timing
comparison is included in the frozen F3 scenario set; the focused public
tests cover small-input behavior, while any required small-input timing gate
needs a separately frozen and measured scenario.

`verification.json` records the machine-checked gate and the exact public
exception list. `candidate-manifest.json` hashes every other package file;
the manifest itself is identified by its SHA-256 and git blob after commit.
