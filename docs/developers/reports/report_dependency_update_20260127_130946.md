# Work Report: dependency update (analysis revert, gpstime/ligo.skymap required)

- Date: 2026-01-27
- Time: 13:09:46 (local)
- Model: GPT-5 (Codex CLI)
- Elapsed: ~30 minutes

## Summary
- Reverted the `analysis` extra to include broader time-frequency tooling.
- Promoted `gpstime` and `ligo.skymap` to required dependencies to ensure core GPS/skymap support.

## Changes
- Updated dependency groups in `pyproject.toml`:
  - Added `gpstime` and `ligo.skymap` to core `dependencies`.
  - Removed `gpstime` from `gw`, removed `ligo.skymap` from `plot`.
  - Restored `analysis` to include `librosa` and `obspy`.
  - Updated `all` to align with the above.

## Files Modified
- `pyproject.toml`

## Tests
- `ruff check .`
- `mypy .`
- `pytest` (full suite): 2473 passed, 222 skipped, 3 xfailed in 5:38

## Notes
- No code changes were made beyond dependency configuration.

## Skill Updates
- None.
