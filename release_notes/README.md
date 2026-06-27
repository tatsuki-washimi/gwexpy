# Release notes

Standardized GitHub Release notes, one file per tagged version
(`vX.Y.Z.md`). **`CHANGELOG.md` is the source of truth** — these files are
generated from it, not edited by hand.

## Workflow

1. Regenerate from CHANGELOG (after editing it or cutting a new release):

   ```bash
   python tools/gen_release_notes.py
   ```

   To add a new version, append it to `VERSIONS` in
   `tools/gen_release_notes.py`.

2. Publish/standardize the GitHub Releases (needs `gh` + write access):

   ```bash
   DRY_RUN=1 bash tools/publish_releases.sh   # preview
   bash tools/publish_releases.sh             # apply
   ```

   Existing releases are edited, missing ones (e.g. `v0.1.1`, `v0.1.5`) are
   created against the existing tag. Only `v0.1.6` is marked as `latest`.

Alternatively, paste a file's contents into the GitHub web UI
(*Releases → Draft a new release → choose the existing tag*), set the title to
`vX.Y.Z`, and leave *"Set as the latest release"* unchecked for anything older
than the newest version.
