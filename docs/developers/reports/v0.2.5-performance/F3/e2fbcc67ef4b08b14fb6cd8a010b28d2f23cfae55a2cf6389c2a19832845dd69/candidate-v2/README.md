# F3 native PSD size-guard candidate

This append-only package compares frozen B1 old R with candidate source
`ffba2e5e27472c098cc1fb45f47a21fffcf918a1`. It replaces `candidate-v1`
for release review; the earlier evidence and the confirmed small-input
regression in `candidate-small-diagnostic-v1` remain unchanged. This package
is still subject to independent review and integration approval.

The frozen B-F3 harness source commit is
`000f1f4988fef13ed33eaf37661ca8deef4561d3`, digest
`e2fbcc67ef4b08b14fb6cd8a010b28d2f23cfae55a2cf6389c2a19832845dd69`.
The controller loaded its modules from a separate checkout at that commit.
`release-controller.py` is the exact controller source, SHA-256
`e9275ac757fe5c4bd2b026eab03b40fd57ae7f065b13529aeef9036ea17feb11`.

## Wheel and installation

- B1 old-R commit: `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`; wheel
  SHA-256: `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- Candidate wheel SHA-256:
  `b3a9ff1ceeda7353c3aea972a6f7c71e18413a6f7adebfbf5a99c809f2d7eab4`.
  It was built from the clean source commit with
  `/home/washimi/miniforge3/envs/main/bin/python -m build --wheel --no-isolation --outdir /tmp/gwexpy-v025-f3-native-psd-wheel-release-v3`.
  A fresh Python 3.12.12 `venv --system-site-packages` received the wheel via
  `pip install --no-deps --no-index`. Both arms use the same dependency versions
  and install mode. Every run manifest records the `-I` import path, version
  0.2.5, dependency inventory, wheel hash, and verification of all 369
  installed `gwexpy` wheel package files. `wheel-source-audit.json` independently
  matches all 369 wheel package files to committed source bytes and records the
  build toolchain and exact commands. The wheel can be rebuilt from the source
  commit; wheel bytes are not stored in this package.

## Public behavior and structural scope

`correctness-public.json` covers 27 frozen small fixtures on three public
routes each: native, external requested, and forced fallback. All 81 result,
warning, and log fingerprints exactly match B1. The 1 MiB file-size guard
returns these small files to the established parser route.

`metadata-extension` records 15 further B1/candidate native public cases,
including nonfinite and negative metadata, invalid encoding/BUnit,
frequency-axis overflow, large N, and NumPy underflow modes. All 15 public
result/warning/log fingerprints match. The deterministic generator/capture
sources, XML bytes, manifest, and both raw captures are included.

`large-fault-extension` enlarges five frozen PSD faults past 1 MiB by adding
a deterministic XML comment before the root. It retains a fixture manifest,
generator/capture source, both raw public captures, and gzip-compressed exact
fixture bytes; decompression hashes match the manifest. Only two approved
#589 differences remain: warnings from fully unselected native PSD short
payload and malformed base64 disappear. Selected payload warning, unselected
metadata warning, and late XML structural warning/result exactly match B1.
No external or TF/6, FFT, STF, or TS performance claim is made.

`structure-large.json` is separate from timing and PSS. The frozen large
native PSD fixture has 256 unselected blocks. B1 decoded 16,777,216 bytes
from them in all five baseline structural samples. This candidate decodes
zero unselected bytes, with zero unknown decoder calls and zero unselected
series constructions. Selected decode remains 65,536 bytes. Source review
confirms that selected data still uses the instrumented native decoder.

## Nine samples per arm

The five measurement directories contain separate small warm batches, large
warm/cold, and large Linux PSS runs. Each has nine raw samples per arm in the
predeclared `ABBA · BAAB · ABBA · BAAB · AB` order, a complete wheel/import
audit, raw arm arrays, median/MAD, and verified artifact hashes. The small
gate is `regression > max(10%, 3×(MAD_B1/median_B1 + MAD_candidate/median_candidate))`
confirmed in a second independent batch; neither new batch exceeds its limit.

| Metric | B1 median (MAD) | Candidate median (MAD) | Result |
| --- | ---: | ---: | --- |
| Small warm wall, batch 1 | 683,562 (53,194) ns | 723,750 (26,401) ns | +5.88%; limit 34.29%; PASS |
| Small warm wall, batch 2 | 683,189 (31,580) ns | 660,588 (20,429) ns | −3.31%; limit 23.14%; PASS |
| Large Linux peak PSS | 363,940 (238) KiB | 292,491 (419) KiB | 19.63% lower; required 5%; PASS |
| Large warm wall | 96.120 (0.791) ms | 108.101 (1.822) ms | 12.46% slower; supporting |
| Large warm CPU | 96.013 (0.755) ms | 108.075 (1.827) ms | 12.56% slower; supporting |
| Large cold fresh-process wall | 2,367.410 (39.367) ms | 2,351.167 (22.998) ms | 0.69% faster; supporting |

The PSS improvement exceeds the predeclared relative-MAD gate
`max(5%, 2×(MAD_B1/median_B1 + MAD_candidate/median_candidate))`, which is
5% for this run. Large warm wall has a measured regression well above its
2.51% relative-noise sum, and no wall-speed claim is made. This is a resource
tradeoff for release review and disclosure. Linux PSS is sampled; a peak
shorter than the sample interval can be missed. PSS, uninstrumented timing,
and decoder spies are separate measurements.

`verification.json` records the machine-checked gates. `candidate-manifest.json`
hashes every other file in this package. Its own SHA-256 and git blob are
reported after commit. The five large-fault XML fixtures are stored as
deterministic gzip bytes; their uncompressed SHA-256 values are in their
fixture manifest.
