# Release notes

Standardized GitHub Release notes, one file per tagged version
(`vX.Y.Z.md`). **`CHANGELOG.md` is the source of truth** — these files are
generated from it, not edited by hand.

## Prepare a version's notes

Generate only the version being prepared, after its dated section has been
finalized in `CHANGELOG.md`:

```bash
python tools/gen_release_notes.py --version 0.2.7
```

The version must appear exactly once as a dated stable section in the
changelog. The command writes only `release_notes/vVERSION.md`; it does not
publish or edit a GitHub Release. Omitting `--version` is an error.

Review the generated file as release-note preparation only. Publication,
tag creation, and **Latest** changes remain governed by `RELEASING.md` and the
approved release decision.
