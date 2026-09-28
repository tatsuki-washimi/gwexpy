# B-X formal baseline v1

Status: `RAW_UNQUALIFIED`, pending independent review and integration. This package is append-only evidence for the exact integrated pre-X source `7f13c4423addab4f92eafc2740da05faa4a444fa`. No X runtime code was edited. The harness and fixture generator are the integrated v3 bytes; the generated approximately 86 MB fixture payload remains at `/tmp/gwexpy-v025-bx-fixtures-v1` and is reproducible from the pinned generator and per-file SHA manifest. The fixture binary payload and wheel binaries are excluded from Git.

## Provenance

- Published PyPI B0 v0.2.4 wheel SHA256: `34afe8188c753cd9da0b182a5d2a88cce2f7ec36633730fe0bee15e2506df56d`; peeled source `522e52a082925da4dd37966d82a7616bdd2a5248`. It matches the publication-status audit manifest.
- Old-R B1 wheel SHA256: `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`; source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`.
- Exact pre-X wheel SHA256: `16fb103590c0310814d5d140a0a33f64de8893c1bd262ff9cc1887b18475aecd`; source `7f13c4423addab4f92eafc2740da05faa4a444fa`.
- All installed arms used Python 3.12.12, NumPy 1.26.4, GWpy 4.0.1, Astropy 7.2.0, xarray 2026.2.0, NetCDF4 1.7.4 and wheel-no-deps installation. Each capture audited all 369 installed GWexpy members against its wheel bytes.
- Harness SHA256: `6c5cfca1239616a143237272fea9e679fa715e4e7e5f3118ab9f1a0ed3771eed`; fixture generator: `0a49b9c57ebfa671f749f5903ed65a30e1913c06eb01d1f5664054b85f01c09d`; fixture manifest: `dade6bd54082ef7f771724b84064caeb252342317e0da5cdc81a966f5e916050`; historical object oracle: `57fdb80b33f10a41d7ea6ac6ac5e55891e6ae967421764470f0416849b041729`.

## Capture completeness and findings

There are 40 capture manifests and 496 raw samples: 22 public captures (5/arm), 6 separate structure captures (5/arm), 6 separate warm wall/parent CPU captures (9/arm ABBA), and 6 separate Linux tree PSS captures (9/arm ABBA, 10 ms sampling). Every capture has within-arm repeatability. B1 versus pre-X has exact cross-arm public parity in all 11 public cases, including `warn`/`raise` edges. Published B0 versus B1 has exact parity in nine cases. The two `matrix_object_strings` cases have the previously characterized #751 difference: B0 returns Unicode and B1 raises `TypeError`; both full warning/value/metadata fingerprints match the pinned oracle.

The structural spy counts full-payload `ndarray.astype` calls at the audited source sites: ATS int32 and int64 each have one call per read in B0, B1, and pre-X; the 16-cell homogeneous NetCDF matrix has 32 calls in B1 and pre-X (B0 has zero). These are site-specific structural counters, not a whole-process allocation or PSS estimate.

| Comparison | Scenario | Warm wall median A→B, ms | Parent CPU median A→B, ms | Tree PSS median A→B, MiB |
| --- | --- | ---: | ---: | ---: |
| historical | ats32 | 20.055→20.034 (-0.10%) | 20.056→19.973 | 374.63→374.45 (-0.05%) |
| historical | ats64 | 24.414→24.539 (+0.52%) | 24.415→24.541 | 390.53→390.47 (-0.02%) |
| historical | matrix | 30.116→52.425 (+74.08%) | 30.117→52.380 | 376.17→370.95 (-1.39%) |
| prex | ats32 | 20.388→21.182 (+3.90%) | 20.222→21.183 | 374.50→374.51 (+0.00%) |
| prex | ats64 | 24.720→24.831 (+0.45%) | 24.649→24.832 | 390.54→390.55 (+0.00%) |
| prex | matrix | 54.265→54.493 (+0.42%) | 54.266→54.494 | 384.01→390.79 (+1.77%) |

Warm wall/CPU measures only the public read call; fingerprint hashing is after the clocks. PSS measures `max_t ΣPSS_process(t)` of parent and live descendants at the same sample instant, after import/wheel audit and until the read result is retained; hashing follows STOP. This is a sampled lower bound. The NetCDF PSS samples show materially more variance than ATS, so a candidate needs its own exact-wheel, separately captured comparison before any memory claim. These are pre-optimization historical/baseline observations, not an X improvement claim.

The `raw/` tree preserves controller stdout/stderr, each JSON sample, each capture manifest, and execution indexes byte for byte. `protocol/` contains the exact controller scripts and pre-X wheel build/install logs. `baseline-manifest.json` hashes every file in this evidence package except itself; it also records all expected counts. Do not rewrite this package after review. Any correction or candidate result belongs in a new sibling directory.
