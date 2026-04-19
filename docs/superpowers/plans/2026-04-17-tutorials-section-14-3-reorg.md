# Tutorials Section 14-3 Reorganization Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve section 14-3 by removing the remaining I/O and interoperability ambiguity around `Advanced Signal Processing` and reorganizing tutorial categories into clearer user-facing buckets.

**Architecture:** Keep the change scoped to documentation taxonomy first: update the visible grouped list and the hidden toctree order in `docs/web/{ja,en}/user_guide/tutorials/index.rst`, without rewriting notebook bodies unless a moved item's framing becomes misleading. Use the already-established split between direct I/O (`io_formats`) and interop (`interop`) as the classification rule.

**Tech Stack:** Sphinx/reStructuredText, nbsphinx-generated tutorial pages, bilingual docs (`ja` and `en`)

---

## Findings

- `Advanced Signal Processing` is still oversized in both languages: 17 items in [docs/web/ja/user_guide/tutorials/index.rst](/home/washimi/work/gwexpy/docs/web/ja/user_guide/tutorials/index.rst:51) and [docs/web/en/user_guide/tutorials/index.rst](/home/washimi/work/gwexpy/docs/web/en/user_guide/tutorials/index.rst:51).
- Direct file I/O tutorials are no longer listed under that section. The previous worst offender, `case_gbd_format`, already lives under `Specialized Tools` at [JA](/home/washimi/work/gwexpy/docs/web/ja/user_guide/tutorials/index.rst:73) / [EN](/home/washimi/work/gwexpy/docs/web/en/user_guide/tutorials/index.rst:73).
- The remaining issue is classification drift, not a literal IV-section misplacement:
  - `intro_interop` is currently grouped under `Core Data Structures`, but its notebook is about external conversions and storage bridges, not core containers.
  - `case_gbd_format` is pure direct I/O.
  - `case_seismic_obspy` is mixed: it starts with MiniSEED/SAC/FDSN reading and `obspy` conversion, then proceeds to signal processing. It is I/O/interop-adjacent enough that users will look for it near data-ingest content.
- This drift conflicts with the rule already established in [docs/web/ja/user_guide/io_formats.md](/home/washimi/work/gwexpy/docs/web/ja/user_guide/io_formats.md:6) and [docs/web/ja/user_guide/interop.md](/home/washimi/work/gwexpy/docs/web/ja/user_guide/interop.md:1): direct I/O and interop are separate entry points and should not be hidden inside signal-processing buckets.

## Recommended Target Structure

Use the following visible sections in both language trees:

1. `Core Data Structures`
2. `Multi-channel & Matrix Containers`
3. `High-dimensional Fields`
4. `Advanced Signal Processing`
5. `Data I/O & Interoperability`
6. `Noise Hunting & Specialized Tools`
7. `Segment Analysis`

Place tutorials as follows:

- `Data I/O & Interoperability`
  - `intro_interop`
  - `case_gbd_format`
  - `case_seismic_obspy`
- `Noise Hunting & Specialized Tools`
  - `advanced_bruco`
  - `case_bruco_ica_denoising`
  - `case_bruco_advanced`
  - `case_violin_mode`
  - `case_schumann_resonance`

Rationale:

- This is the smallest change that makes the I/O/interop path visible without rewriting the signal-processing section itself.
- It matches the public-information architecture already introduced by `io_formats` and `interop`.
- It gives users a predictable place to find “how do I ingest or convert data?” tutorials.

## Scope Decision

- Do not move true signal-processing items out of section IV in this pass.
- Do not split `case_seismic_obspy` yet unless its tutorial body is later rewritten to separate ingestion from analysis; for now, reclassify it as an ingest-and-convert case study.
- Do not expose unlisted notebooks such as `case_hdf5_provenance.ipynb` in the same patch unless product direction explicitly expands tutorial coverage.

### Task 1: Update Japanese Tutorial Taxonomy

**Files:**
- Modify: `docs/web/ja/user_guide/tutorials/index.rst`
- Check: `docs/web/ja/user_guide/io_formats.md`
- Check: `docs/web/ja/user_guide/interop.md`

- [ ] **Step 1: Create the new visible section**

Insert a new section after `IV. 高度な信号処理` and before the current `V. 特殊ツール`:

```rst
V. データ I/O と相互運用
------------------------
ファイルの読み書き、外部ライブラリとの変換、観測データの取り込みを扱います。
```

- [ ] **Step 2: Move I/O-adjacent bullets into the new section**

Move these bullets out of their current categories and place them under the new section:

```rst
- :doc:`相互運用: 基本 <intro_interop>` ...
- :doc:`ケーススタディ: GBD 形式 I/O <case_gbd_format>` ...
- :doc:`ケーススタディ: ObsPy 連携による地震データ解析 <case_seismic_obspy>` ...
```

- [ ] **Step 3: Rename the old specialized-tools bucket**

Retitle the remaining tool-heavy section to emphasize diagnostics rather than generic miscellany:

```rst
VI. ノイズハンティングと特殊ツール
--------------------------------
```

- [ ] **Step 4: Renumber the segment section**

Change `VI. セグメント解析` to `VII. セグメント解析`.

- [ ] **Step 5: Reorder the hidden toctree to match visible grouping**

Ensure the hidden toctree order mirrors the new section order so Next/Previous navigation follows the visible taxonomy.

Expected moved block:

```rst
intro_histogram
matrix_timeseries
...
advanced_decomposition
intro_interop
case_gbd_format
case_seismic_obspy
advanced_bruco
...
```

### Task 2: Mirror the Same Taxonomy in English

**Files:**
- Modify: `docs/web/en/user_guide/tutorials/index.rst`
- Check: `docs/web/en/user_guide/io_formats.md`
- Check: `docs/web/en/user_guide/interop.md`

- [ ] **Step 1: Add the new English section**

```rst
V. Data I/O & Interoperability
------------------------------
Tutorials for file ingest, read/write workflows, and conversions with external libraries.
```

- [ ] **Step 2: Move the same three bullets**

```rst
- :doc:`Interoperability: Basics <intro_interop>` ...
- :doc:`Case Study: GBD Format I/O <case_gbd_format>` ...
- :doc:`Case Study: Seismic Analysis with ObsPy <case_seismic_obspy>` ...
```

- [ ] **Step 3: Rename the old tools section**

```rst
VI. Noise Hunting & Specialized Tools
------------------------------------
```

- [ ] **Step 4: Renumber `Segment Analysis` to section VII**

- [ ] **Step 5: Mirror the hidden toctree order**

Keep JA/EN structurally identical.

### Task 3: Validate Terminology and Cross-links

**Files:**
- Modify if needed: `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md`
- Check: `docs/web/{ja,en}/user_guide/io_formats.md`
- Check: `docs/web/{ja,en}/user_guide/interop.md`

- [ ] **Step 1: Confirm section descriptions match the established taxonomy**

Validation checklist:

- `io_formats` remains the direct-I/O landing page.
- `interop` remains the conversion/bridge landing page.
- Tutorial index does not describe these as signal-processing topics.

- [ ] **Step 2: Update the internal audit report status**

Change 14-3 from `要確認` to a resolved state only after both language indexes and the toctree order are updated.

Suggested note:

```md
✅修正済み（I/O/interop 系 tutorial を独立カテゴリへ分離し、Advanced Signal Processing から分類上の混入を解消）
```

### Task 4: Build and Review

**Files:**
- No new source files expected beyond the two tutorial index pages and the audit note

- [ ] **Step 1: Build the docs**

Run:

```bash
cd docs && sphinx-build -b html . _build/html
```

Expected:

- Build succeeds.
- `tutorials/index.html` in both languages shows the new category.
- The sidebar and Next/Previous flow follow the new grouping.

- [ ] **Step 2: Spot-check the three moved tutorials**

Confirm these pages still feel semantically aligned with their new bucket:

- `intro_interop`
- `case_gbd_format`
- `case_seismic_obspy`

- [ ] **Step 3: Commit**

```bash
git add docs/web/ja/user_guide/tutorials/index.rst \
        docs/web/en/user_guide/tutorials/index.rst \
        docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md
git commit -m "docs: separate tutorial I/O and interop category"
```

## Success Criteria

- No direct-I/O or interop-first tutorial is visually grouped under `Advanced Signal Processing`.
- JA and EN tutorial indexes remain structurally symmetric.
- Hidden toctree order matches the visible grouping.
- The audit record for 14-3 can be closed with a concrete rationale instead of `要確認`.
