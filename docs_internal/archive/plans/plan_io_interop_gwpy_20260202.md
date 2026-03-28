# Plan: I/O Interop Improvements with GWpy (2026-02-02)

## Objectives & Goals

1. Add **CSV/TXT read/write** support for `gwexpy.timeseries.TimeSeriesDict` and `TimeSeriesList`.
2. Make `gwexpy.frequencyseries.FrequencySeriesDict/List` and `gwexpy.spectrogram.SpectrogramDict/List` writable in a way that **GWpy can read via its standard API** (i.e., `gwpy...FrequencySeries.read(..., format="hdf5", path=...)`, `gwpy...Spectrogram.read(..., format="hdf5", path=...)`).
3. Keep the design **pandas-friendly** and **metadata-preserving** (units, sampling, epoch where meaningful), with clear fallbacks.

Non-goals (for this iteration):
- Implementing new binary formats (GWF for frequency-domain objects, etc.).
- Changing GWpy itself (we will not patch site-packages).
- Guaranteeing cross-version `pickle`/`shelve` interchangeability.

## Detailed Roadmap (by Phase)

### Phase 0: Confirm expectations & constraints
- Confirm the desired semantics for CSV/TXT:
  - “wide” format (1 time column + N channel columns) vs “long/tidy”.
  - Time column encoding: GPS seconds (float) vs datetime string.
  - Unit handling: embed per-column units in comment metadata vs drop units.
- Confirm HDF5 interop target:
  - “GWpy reads each entry” is achieved by writing **one HDF5 dataset per entry** at a known `path` (because GWpy’s HDF5 reader requires `path` when multiple datasets exist).

### Phase 1: TimeSeriesDict/List CSV/TXT
- Implement `TimeSeriesDict.write(..., format="csv"/"txt")`:
  - Write a header block of `#`-prefixed metadata lines (JSON or simple `key: value`) so pandas can ignore it via `comment="#"`.
  - Write a tabular body with columns: `time,<ch1>,<ch2>,...`.
  - Store per-channel units (and optionally `t0`, `dt` if regular) in the metadata header.
- Implement `TimeSeriesDict.read(..., format="csv"/"txt")`:
  - Parse `#` metadata header if present.
  - Reconstruct `TimeSeries` per column with correct `t0/dt` (if inferable) and `unit`.
- Implement `TimeSeriesList.write/read` similarly:
  - Use `ts.name` (or `series_{i}` fallback) as column keys.
  - Preserve order on read via stored name list in metadata.

### Phase 2: FrequencySeriesDict/List HDF5 layout compatible with GWpy
- Implement `FrequencySeriesDict.write(..., format="hdf5")` and `FrequencySeriesList.write(..., format="hdf5")`:
  - Create a single HDF5 file where each entry is written as a **dataset directly under root** at `path=<safe_key>`.
  - Call the underlying GWpy writer by delegating to `FrequencySeries.write(h5file, format="hdf5", path=safe_key, overwrite=...)`.
  - Store an attribute mapping `safe_key -> original_key` for reversibility if keys are not HDF5-safe.
- Implement corresponding `.read(..., format="hdf5")`:
  - Enumerate datasets at file root, rebuild entries via `FrequencySeries.read(file, format="hdf5", path=...)`.
  - Re-apply original keys where mapping exists.
- Document GWpy usage:
  - `from gwpy.frequencyseries import FrequencySeries; fs = FrequencySeries.read("x.h5", path="H1_ASD")`

### Phase 3: SpectrogramDict/List HDF5 layout compatible with GWpy
- Replace current “group per entry” layout with “dataset per entry” layout:
  - Write each `Spectrogram` as a dataset at root `path=<safe_key>` via `Spectrogram.write(..., format="hdf5", path=...)`.
  - Keep optional compatibility mode if existing layout must remain supported (decide after quick survey of current users/tests).
- Implement `.read` that can read both:
  - New dataset-per-entry layout (preferred).
  - Legacy group-per-entry layout (best-effort) to avoid breaking existing files.
- Document GWpy usage:
  - `from gwpy.spectrogram import Spectrogram; sg = Spectrogram.read("sg.h5", path="0")`

### Phase 4: Tests & docs
- Add/extend tests that validate:
  - CSV round-trip: `TimeSeriesDict/List -> csv -> read` retains shape, time axis, and units.
  - HDF5 interop: files written by gwexpy can be read by **GWpy** with `path=...` for representative entries.
  - Backward compatibility (if we keep legacy spectrogram layout reader).
- Update user-facing docs (short “I/O interop” note) and relevant docstrings.

## Testing & Verification Plan

- Unit tests (pytest):
  - `TimeSeriesDict.write/read` for `csv` and `txt` with:
    - regular sampling (recover `t0/dt`)
    - irregular time stamps (store explicit time vector)
  - `TimeSeriesList.write/read` similarly.
  - `FrequencySeriesDict/List.write(hdf5)` then `gwpy.frequencyseries.FrequencySeries.read(..., path=...)`.
  - `SpectrogramDict/List.write(hdf5)` then `gwpy.spectrogram.Spectrogram.read(..., path=...)`.
- Manual smoke:
  - Create a small example file in `examples/` (optional) demonstrating GWpy read calls.

## Models, Recommended Skills, and Effort Estimates

### Suggested model / skills
- Model: `GPT-5.2` (good balance of refactor + tests + careful metadata handling).
- Skills to use after approval:
  - `run_tests` (pytest verification)
  - `lint_check` (ruff/mypy)
  - `archive_work` (work report)

### Effort estimate (wall-clock)
- Estimated total time: **35–60 minutes**
- Estimated quota consumption: **Medium–High**
- Breakdown:
  - Design decisions & format spec: 10–15 min
  - Implementation (collections + HDF5 writers): 15–25 min
  - Tests & fixes: 10–20 min
  - Docs touch-ups: 5–10 min
- Concerns:
  - Exact metadata fidelity for CSV/TXT (units + irregular time) needs agreement.
  - Spectrogram legacy-layout backward compatibility decision impacts scope.

