# Reproducibility Guide for the SoftwareX Manuscript

This document describes how to regenerate the figures and executable examples referenced by `docs/gwexpy-paper/main.tex`.

## Environment

Install the package with the extras used by the paper examples:

```bash
pip install "gwexpy[analysis,gw]"
pip install marimo
```

The repository CI is currently exercised on Linux (Ubuntu) with Python 3.11 and 3.12.

## Headless Reproduction Commands

The paper examples are stored as marimo-compatible Python scripts under `notebooks/`.

Synthetic workflow covered by default CI:

```bash
python -m marimo run notebooks/02_coherence_ranking.py
```

Equivalent direct execution:

```bash
python notebooks/02_coherence_ranking.py
```

Optional public-data workflow:

```bash
python -m marimo run notebooks/03_gwosc_case_study.py
```

Equivalent direct execution:

```bash
python notebooks/03_gwosc_case_study.py
```

If you have prepared a local cache artifact, you can force offline reuse with:

```bash
GWEXPY_GWOSC_CACHE=docs/repro/cache/gwosc_gw150914_h1_1024s.npz \
python notebooks/03_gwosc_case_study.py
```

## Generated Artifacts

Running the scripts above writes the figure files used by the manuscript to `docs/gwexpy-paper/`:

- `figure3_coherence_ranking.pdf`
- `figure3_coherence_ranking.png`
- `figure5_gwosc_case_study.pdf`
- `figure5_gwosc_case_study.png`

## Network-Dependent Example Policy

`notebooks/02_coherence_ranking.py` is synthetic and suitable for default CI smoke testing.

`notebooks/03_gwosc_case_study.py` downloads public GWOSC strain data. Because it depends on network access and remote service availability, it is treated as an optional reproduction path rather than a required default CI job. If you want to exercise it in automation, do so in a separate network-enabled job or with prepared local caches.

## Suggested Cache Handling

If you want to make the GWOSC workflow more repeatable locally:

- Keep the downloaded public-data cache under your standard Python/GWOSC cache location.
- Record the exact GPS interval used by the notebook.
- If strict offline reproduction is needed later, prepare a small cached artifact derived from the public data and document its provenance alongside the release.
- The notebook supports `GWEXPY_GWOSC_CACHE=/path/to/cache.npz` for loading a prepared local cache or writing one after a successful download.

## Manuscript Alignment

The SoftwareX manuscript points to:

- `notebooks/02_coherence_ranking.py` for the synthetic coherence-ranking example
- `notebooks/03_gwosc_case_study.py` for the public GWOSC example

The archived code record for the submitted version is the GitHub repository together with the Zenodo DOI `10.5281/zenodo.19059423`.
