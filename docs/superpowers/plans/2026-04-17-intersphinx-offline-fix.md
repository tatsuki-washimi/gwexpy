# Intersphinx Offline-Safe Fix Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make external API cross-references to GWpy and other dependencies resolve reliably in local offline/sandboxed builds and in CI, so section 16-12 can be closed with evidence.

**Architecture:** Stop depending on live `objects.inv` fetches during normal docs builds. Keep official upstream documentation URLs as the click target, but resolve symbols from repo-local vendored inventories under `docs/_intersphinx/`, with a small refresh script for network-enabled maintenance.

**Tech Stack:** Sphinx `intersphinx`, Python docs tooling, GitHub Actions docs workflows

---

## Findings

- Current `intersphinx_mapping` in [docs/conf.py](/home/washimi/work/gwexpy/docs/conf.py:261) is remote-first for every dependency and explicitly hardcodes GWpy to `https://gwpy.readthedocs.io/en/stable/objects.inv`.
- A local docs build reproduced the actual failure mode: `sphinx-build -b html -D nbsphinx_execute=never docs docs/_build/html` succeeded, but emitted six intersphinx fetch warnings caused by DNS resolution failures for `python`, `numpy`, `scipy`, `astropy`, `matplotlib`, and `gwpy`.
- CI history is worse than “local DNS only”: [gwexpy-failure-report.md](/home/washimi/work/gwexpy/gwexpy-failure-report.md:97) records repeated `Sphinx: Intersphinx 404 (GWpy docs stable)` failures on April 9, 2026. That means the current GWpy inventory source is also operationally brittle on networked runners.
- The build currently suppresses `ref.intersphinx` in [docs/conf.py](/home/washimi/work/gwexpy/docs/conf.py:160), so missing external resolution is easy to miss unless someone inspects warnings carefully.
- The source tree does use real external Sphinx roles such as `:class:\`gwpy.plot.Plot\`` and `:meth:\`gwpy.spectrogram.Spectrogram.imshow\`` in code docstrings, so fixing `intersphinx_mapping` matters for more than cosmetic navigation.
- Some public docs pages already contain hardcoded GWpy links as a workaround, for example [docs/web/en/reference/FrequencySeries.md](/home/washimi/work/gwexpy/docs/web/en/reference/FrequencySeries.md:47). Those links reduce damage, but they do not solve unresolved docstring cross-references elsewhere.

## Recommended Solution

Use **vendored intersphinx inventories** committed into the repository:

1. Keep each dependency’s official documentation base URL as the outward-facing link target.
2. Store a local copy of each `objects.inv` in `docs/_intersphinx/`.
3. Point `intersphinx_mapping` to the local inventory path by default.
4. Allow an explicit opt-in to live upstream fetches only for maintenance or refresh operations.
5. Add a refresh script so inventories can be updated intentionally on a networked machine or in a scheduled workflow.

This solves both observed failure classes:

- local offline / sandbox / DNS-restricted builds
- CI instability or upstream `objects.inv` outages

## Scope Decision

- Fix `docs/conf.py`, inventory storage, and maintenance workflow in this pass.
- Do not rewrite every manual external GWpy link in reference pages as part of the core fix.
- Do update the audit report entry for 16-12 once offline-safe resolution is verified.
- Prefer `gwpy.readthedocs.io/en/stable/` as the canonical GWpy base URL unless a live verification step proves another canonical stable URL is required.

### Task 1: Make `docs/conf.py` Prefer Local Inventories

**Files:**
- Modify: `docs/conf.py`
- Create: `docs/_intersphinx/README.md`

- [ ] **Step 1: Add a docs-root-aware helper for inventory paths**

Introduce a helper near the existing config helpers:

```python
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
INTERSPHINX_DIR = DOCS_DIR / "_intersphinx"


def _intersphinx_inventory(name: str) -> str | None:
    path = INTERSPHINX_DIR / f"{name}.inv"
    return str(path) if path.exists() else None
```

- [ ] **Step 2: Add an explicit remote-override switch**

Use an env flag so maintainers can intentionally test live upstream inventories:

```python
def _prefer_remote_intersphinx() -> bool:
    return _env_flag("INTERSPHINX_USE_REMOTE")
```

- [ ] **Step 3: Rebuild `intersphinx_mapping` around local-first resolution**

Replace the current static mapping block with a small helper:

```python
def _intersphinx_target(name: str, base_url: str, remote_inventory: str | None = None) -> tuple[str, str | None]:
    if not _prefer_remote_intersphinx():
        local_inventory = _intersphinx_inventory(name)
        if local_inventory is not None:
            return (base_url, local_inventory)
    return (base_url, remote_inventory)


intersphinx_mapping = {
    "python": _intersphinx_target("python", "https://docs.python.org/3", "https://docs.python.org/3/objects.inv"),
    "numpy": _intersphinx_target("numpy", "https://numpy.org/doc/stable", "https://numpy.org/doc/stable/objects.inv"),
    "scipy": _intersphinx_target("scipy", "https://docs.scipy.org/doc/scipy", "https://docs.scipy.org/doc/scipy/objects.inv"),
    "astropy": _intersphinx_target("astropy", "https://docs.astropy.org/en/stable", "https://docs.astropy.org/en/stable/objects.inv"),
    "matplotlib": _intersphinx_target("matplotlib", "https://matplotlib.org/stable", "https://matplotlib.org/stable/objects.inv"),
    "gwpy": _intersphinx_target("gwpy", "https://gwpy.readthedocs.io/en/stable/", "https://gwpy.readthedocs.io/en/stable/objects.inv"),
}
```

- [ ] **Step 4: Stop hiding intersphinx resolution failures**

Remove `"ref.intersphinx"` from `suppress_warnings` once local inventories are committed. Keep `intersphinx.broken_domain` only if it is still needed for third-party inventory quirks.

Expected result:

- offline builds no longer attempt live fetches when vendored files exist
- broken inventory wiring becomes visible again in CI

### Task 2: Vendor the Required `objects.inv` Files

**Files:**
- Create: `docs/_intersphinx/python.inv`
- Create: `docs/_intersphinx/numpy.inv`
- Create: `docs/_intersphinx/scipy.inv`
- Create: `docs/_intersphinx/astropy.inv`
- Create: `docs/_intersphinx/matplotlib.inv`
- Create: `docs/_intersphinx/gwpy.inv`
- Create: `docs/_intersphinx/README.md`
- Optional create: `docs/_intersphinx/sources.json`

- [ ] **Step 1: Create the inventory directory and documentation**

`docs/_intersphinx/README.md` should document:

- why inventories are vendored
- which upstream URL each file came from
- how to refresh them
- that Sphinx still links users to the official external docs, not to local files

- [ ] **Step 2: Download and commit inventory snapshots**

Populate each `.inv` file from the official upstream `objects.inv` endpoint on a network-enabled machine.

Minimum set for this fix:

- `python`
- `numpy`
- `scipy`
- `astropy`
- `matplotlib`
- `gwpy`

- [ ] **Step 3: Record provenance**

Store a simple manifest, for example:

```json
{
  "gwpy": {
    "base_url": "https://gwpy.readthedocs.io/en/stable/",
    "inventory_url": "https://gwpy.readthedocs.io/en/stable/objects.inv"
  }
}
```

This avoids future “where did this binary file come from?” confusion.

### Task 3: Add a Refresh / Validation Script

**Files:**
- Create: `scripts/update_intersphinx_inventories.py`
- Optional modify: `.github/workflows/ops-weekly.yml`

- [ ] **Step 1: Add a deterministic downloader**

Implement a small script that:

- knows the authoritative inventory URLs
- downloads each file with `requests`
- writes them to `docs/_intersphinx/<name>.inv`
- optionally updates `sources.json`

Suggested interface:

```bash
python scripts/update_intersphinx_inventories.py --write
python scripts/update_intersphinx_inventories.py --check-upstream
```

- [ ] **Step 2: Fail clearly when upstream is unavailable in check mode**

`--check-upstream` should exit non-zero if an authoritative inventory URL fails, without changing the working tree.

- [ ] **Step 3: Consider wiring a scheduled upstream check**

Add a weekly or manual CI step that runs:

```bash
python scripts/update_intersphinx_inventories.py --check-upstream
```

This keeps vendored inventories from masking a long-term upstream URL migration.

### Task 4: Verify That Offline Builds Still Produce External Links

**Files:**
- Modify if needed: `.github/workflows/docs-pr.yml`
- Modify if needed: `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md`

- [ ] **Step 1: Run the offline-safe local docs build**

Run:

```bash
conda run -n gwexpy sphinx-build -b html -D nbsphinx_execute=never docs docs/_build/html
```

Expected:

- build succeeds
- no intersphinx inventory fetch warnings are emitted for the vendored dependencies

- [ ] **Step 2: Spot-check generated links**

Inspect at least these pages in generated HTML:

- `docs/_build/html/web/en/reference/FrequencySeries.html`
- `docs/_build/html/web/en/reference/TimeSeries.html`
- `docs/_build/html/web/en/reference/Spectrogram.html`

Confirm that GWpy cross-references still point to official external URLs.

- [ ] **Step 3: Validate a live-upstream maintenance path**

On a machine with network access, run:

```bash
INTERSPHINX_USE_REMOTE=1 conda run -n gwexpy sphinx-build -b html -D nbsphinx_execute=never docs docs/_build/html
python scripts/update_intersphinx_inventories.py --check-upstream
```

Expected:

- upstream inventories are reachable, or failures are reported explicitly
- local default builds do not depend on that reachability

- [ ] **Step 4: Update the audit report**

Change section 16-12 in [docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md](/home/washimi/work/gwexpy/docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md:529) from `要確認` to a resolved note such as:

```md
✅修正済み（intersphinx inventory を `docs/_intersphinx/` に同梱し、ローカル制約下でも GWpy / NumPy / SciPy / Astropy / Matplotlib への外部 API リンクを解決可能にした）
```

### Task 5: Keep the Change Tight

**Files:**
- Modify only the files above unless verification exposes a concrete mismatch

- [ ] **Step 1: Do not mass-edit manual GWpy links in this patch**

Manual links already added to overview/reference pages are workaround content, not the root cause. Keep the patch focused unless a broken canonical base URL is discovered during verification.

- [ ] **Step 2: Commit with a docs-scoped message**

```bash
git add docs/conf.py \
        docs/_intersphinx \
        scripts/update_intersphinx_inventories.py \
        .github/workflows/ops-weekly.yml \
        docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md
git commit -m "docs: vendor intersphinx inventories for offline-safe builds"
```

## Success Criteria

- `docs/conf.py` no longer depends on live network fetches during ordinary builds.
- GWpy and other mapped dependency references resolve to official external docs even in offline/sandboxed local builds.
- Docs PR CI no longer fails because `objects.inv` fetches 404 or DNS-resolve poorly.
- The audit report entry 16-12 can be closed with a concrete verification note instead of `要確認`.
