# GWexpy v0.2.4 publication status

This record captures the verified publication state as of 2026-09-26 13:35 UTC.
It is a partial status record; Zenodo archiving has not been confirmed complete,
and conda-forge packaging remains pending.

The release source is `522e52a082925da4dd37966d82a7616bdd2a5248`, tagged as
`v0.2.4` by annotated tag object `2bc0789918fbc0327114df0d1ac0840faf5ed2ba`.
The strict tag-triggered publish run passed all 33 checks. Its qualified wheel
and source distribution were published to PyPI and matched the qualified
payload byte for byte.

GitHub Release `397242347` was published at 2026-09-26 13:16:44 UTC. It is not
a draft or prerelease. Its substantive release notes content corresponds to
`release_notes/v0.2.4.md`; the body also includes release provenance and a
tag-pinned changelog entry.

At the readback time, conda-forge and the latest publicly visible Zenodo
archive were both v0.2.3. Zenodo acknowledged the `released` event with
HTTP 202, but its v0.2.4 record was not yet publicly visible; duplicate event
delivery returned HTTP 409, so no redelivery was attempted. No v0.2.4
conda-forge feedstock pull request existed. The public roadmap and installation
text therefore identify PyPI and GitHub Releases as the current v0.2.4 sources
and keep the older conda-forge and Zenodo state explicit.

The introductory downloadable examples were verified against v0.2.4 using
both the candidate documentation source and a clean R2 Git archive in an
isolated Python 3.12 environment with the published PyPI wheel. The optional
commissioner XML example also passed with `dttxml==1.1.8`, without a skip.
The helper SHA-256 was
`772b6d979b760c14796852482f0fb4428c6808ab4f82a81e1cf1b36b8f45cad3`, and the
published wheel SHA-256 was
`34afe8188c753cd9da0b182a5d2a88cce2f7ec36633730fe0bee15e2506df56d`.
Public documentation deployment readback is pending. This record does not
mark publication closure complete.

The machine-readable state is in
[`audit-manifest-v0.2.4-publication-status.yaml`](../plans/manifests/audit-manifest-v0.2.4-publication-status.yaml).
