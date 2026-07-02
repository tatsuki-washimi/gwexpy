#!/usr/bin/env bash
#
# Publish/standardize all GitHub Releases from the generated notes in
# release_notes/.  Run this from a machine where the GitHub CLI (`gh`) is
# installed and authenticated with write access to the repository.
#
# Prerequisites:
#   - gh installed:        https://cli.github.com/
#   - authenticated:       gh auth status   (needs repo write scope)
#   - notes generated:     python tools/gen_release_notes.py
#   - all tags pushed:     git push --tags   (tags v0.1.0..v0.1.7 must exist on origin)
#
# Behaviour:
#   - For each version, EDIT the release if it already exists, otherwise CREATE it
#     (bound to the existing tag via --verify-tag, never creating a new tag).
#   - Title is standardized to "vX.Y.Z".
#   - Only v0.1.7 is marked as the latest release.
#
# Usage:
#   bash tools/publish_releases.sh            # publish/update all
#   DRY_RUN=1 bash tools/publish_releases.sh  # print actions only

set -euo pipefail

REPO="tatsuki-washimi/gwexpy"
LATEST_TAG="v0.1.7"
VERSIONS=(0.1.0 0.1.1 0.1.2 0.1.3 0.1.4 0.1.5 0.1.6 0.1.7)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTES_DIR="$ROOT/release_notes"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI is not installed (see https://cli.github.com/)" >&2
  exit 1
fi
gh auth status --hostname github.com >/dev/null

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

for v in "${VERSIONS[@]}"; do
  tag="v$v"
  notes="$NOTES_DIR/$tag.md"
  if [[ ! -f "$notes" ]]; then
    echo "ERROR: missing $notes (run: python tools/gen_release_notes.py)" >&2
    exit 1
  fi

  # Note: `gh release edit` does not document --latest=false, so we never pass
  # --latest while editing. The latest release is set once, after the loop.
  if gh release view "$tag" --repo "$REPO" >/dev/null 2>&1; then
    echo ">> editing existing release $tag"
    run gh release edit "$tag" --repo "$REPO" \
      --title "$tag" --notes-file "$notes"
  elif [[ "$tag" == "$LATEST_TAG" ]]; then
    echo ">> creating new release $tag (latest=true)"
    run gh release create "$tag" --repo "$REPO" --verify-tag \
      --title "$tag" --notes-file "$notes" --latest
  else
    echo ">> creating new release $tag (latest=false)"
    run gh release create "$tag" --repo "$REPO" --verify-tag \
      --title "$tag" --notes-file "$notes" --latest=false
  fi
done

# Ensure exactly one Latest release, regardless of create/edit paths above.
echo ">> marking $LATEST_TAG as the latest release"
run gh release edit "$LATEST_TAG" --repo "$REPO" --latest

echo "Done. Verify with: gh release list --repo $REPO"
