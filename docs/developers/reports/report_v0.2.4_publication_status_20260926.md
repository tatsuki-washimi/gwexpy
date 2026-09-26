# GWexpy v0.2.4 publication status

This record captures the verified publication state as of 2026-09-26 14:13 UTC.
Zenodo API and archive metadata were read at 14:09 UTC; the direct record page
and DOI resolution were checked at 14:13 UTC. Zenodo publication is verified;
conda-forge packaging remains pending, so this is still a partial status record.

The release source is `522e52a082925da4dd37966d82a7616bdd2a5248`, tagged as
`v0.2.4` by annotated tag object `2bc0789918fbc0327114df0d1ac0840faf5ed2ba`.
The strict tag-triggered publish run passed all 33 checks. Its qualified wheel
and source distribution were published to PyPI and matched the qualified
payload byte for byte.

GitHub Release `397242347` was published at 2026-09-26 13:16:44 UTC. It is not
a draft or prerelease. Its substantive release notes content corresponds to
`release_notes/v0.2.4.md`; the body also includes release provenance and a
tag-pinned changelog entry.

The public Zenodo API reports record `22978439` as published, version 0.2.4,
publication date 2026-09-26, DOI `10.5281/zenodo.22978439`, resource type
Software. The direct [Zenodo record page](https://zenodo.org/records/22978439)
returned HTTP 200; the DOI resolver returned HTTP 404 at the readback time, so
public documentation links directly to the record until DOI resolution is
available. Its archive file is `tatsuki-washimi/gwexpy-v0.2.4.zip` (14,595,789
bytes); the API reports MD5 `c18f22ce6a63cc176a8cfb7cb81a5e4e`, and an
independent download has SHA-256
`67c224393870a0f6103e324fb4469e0c0064e3004e6efb9c8b64f3455b193515`. The
archive contains 2,590 paths whose names and file contents match `git archive`
of release source `522e52a082925da4dd37966d82a7616bdd2a5248`.

The earlier `released` webhook response was HTTP 202; duplicate delivery
returned HTTP 409, and no redelivery was attempted. At the readback time,
conda-forge remained at v0.2.3 and no v0.2.4 feedstock pull request existed.
The public roadmaps and changelogs now link the v0.2.4 Zenodo record and keep
the pending conda-forge state explicit.

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
