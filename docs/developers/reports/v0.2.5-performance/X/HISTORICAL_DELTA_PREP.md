# B-X historical-delta harness preparation

Status: `UNBASELINED`. No X runtime code or resource measurement was changed by
this preparation. Formal B-X captures must use a new integrated pre-X source
SHA that includes the frozen harness and oracle below.

The initial harness SHA-256
`074354b47e4fc1fbe35842b07342597df3bf6f23ec6446ac6731b0bc3e567115`
required B0/B1 cross-arm equality. Nine public cases passed before
`matrix_object_strings` exposed a real #751 difference: the published v0.2.4
wheel returns a Unicode matrix; old-R v0.2.5 raises an `isnan` `TypeError`.
The partial v1 capture remains at
`/tmp/gwexpy-v025-bx-baseline-036d9a-v1/` and is diagnostic only.

The intermediate v2 harness SHA-256
`bfbb002c724eb20910c18ff5f563f3dbcaad1433574a193b47e2bc0c5fb27855`
accepted that historical difference too broadly. Its paused public captures
remain at `/tmp/gwexpy-v025-bx-baseline-036d9a-v2/` and are diagnostic only.
Neither diagnostic series qualifies as a formal baseline.

The initial v3 diagnostic harness SHA-256 was
`c58974c1ced060bcd1bb5a3a1ce0d5f5e17f29677c640336f866221fe5d0f2bc`.
Its exact historical oracle SHA-256 is
`57fdb80b33f10a41d7ea6ac6ac5e55891e6ae967421764470f0416849b041729`.
The oracle binds the fixture, B0/B1 source and wheel SHAs, and complete
warn/raise public fingerprints, including output value hash, dtype, shape,
axis, units, warning order, and error message. Other historical differences
fail capture; B1/pre-X and pre-X/candidate remain strict cross-arm comparisons.
That diagnostic self-check passed 27 tests, including wrong source/wheel, value hash,
unit, warning, error-message, and within-arm drift cases.

Four direct installed-wheel public reads, their stderr, and the v1 B0 partial
sample are preserved in the append-only `oracle-prep-v1/` directory. Its raw
index manifest SHA-256 is
`7896458517530fafad8977a7aea45b5f2bfd1ad4235e51285299766f460da7aa`.
Direct warn/raise fingerprints agree within each arm. The B0 direct result
matches the v1 partial sample, and the B1 direct result matches the earlier
tracked `b1-probe-v1/matrix_object_strings-warn-public.json`. Published B0
wheel SHA-256 matches the v0.2.4 publication manifest's PyPI wheel SHA-256.

A v3 diagnostic 5/arm installed-wheel capture of the historical object case
completed with `within_arm_parity=true`, `cross_arm_parity=false`, and the
pinned reason/oracle SHA. It remains outside the formal baseline at
`/tmp/gwexpy-v025-bx-v3-diagnostic-object-warn/`, because its pre-X SHA
precedes integration of the v3 harness.

Before formal freeze, an additional arm-binding review found that equal
historical results could still be mislabeled. The final v3 harness therefore
pins B0/B1 source and wheel SHAs for **every** historical capture and pins
old-R/B1 as the `prex` arm A. Unsupported `numpy-seterr=ignore` for the object
oracle is an explicit validation error. Final v3 harness SHA-256:
`6c5cfca1239616a143237272fea9e679fa715e4e7e5f3118ab9f1a0ed3771eed`.
The oracle bytes remain unchanged. Focused tests pass 37 cases, including
same-output mislabeled arms. Formal captures must use this final digest and a
new integrated pre-X SHA; all preceding captures remain diagnostic.
