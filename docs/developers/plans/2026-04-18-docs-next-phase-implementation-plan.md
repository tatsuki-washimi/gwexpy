# Web Docs Next Phase Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the highest-value documentation audit gaps left after the April IA rewrite, while avoiding a risky mix of content cleanup, theme migration, and docs-platform changes in the same implementation wave.

**Architecture:** Execute the next phase in layers. First, resolve product decisions that affect page placement and platform direction. Next, apply shared documentation scaffolding and cross-link patterns across high-traffic pages. Only after that, add visual assets and platform-level polish. Theme/search/output-format changes should be isolated behind a separate spike or dedicated phase so they do not destabilize content work.

**Tech Stack:** Sphinx, MyST Markdown, reStructuredText, nbsphinx, `sphinx_rtd_theme`, custom Jinja layout override, custom CSS/JS, GitHub Pages

---

## Objectives & Goals

1. Reduce the remaining high-visibility UX and trust gaps in user-facing docs.
2. Close pending audit items by grouped implementation waves rather than page-by-page churn.
3. Keep JA/EN structures synchronized.
4. Separate low-risk content fixes from high-risk platform changes.
5. Produce a plan that can be reviewed with the Consultant (Session 26) before execution.

## Risk Scale

- **Low:** Localized doc/content edits with low build or navigation risk.
- **Medium:** Cross-page structural edits, shared templates, or asset reuse within the current theme.
- **High:** Global template/theme/search/output changes, CI gate changes, or new maintenance burden.

## Decision Gates Before Implementation

These should be confirmed with the Consultant before coding begins, because they affect multiple later phases.

### Gate A: Distribution Messaging Policy

**Audit items:** `4-1`

**Decision needed:** Should `installation` continue to present GWexpy as development-install only, or should it expose a more explicit release-status matrix for PyPI/Conda plans?

**Preferred direction:** Keep the actual install path unchanged, but replace vague "coming soon" language with a concrete status block:

- source install: supported
- package index install: not yet supported
- recommended audience: contributors / early adopters

**Technical risk:** Low

### Gate B: Validated Algorithms Placement and Bibliography Strategy

**Audit items:** `12-11`, `12-12`

**Decision needed:** Should `validated_algorithms` remain in `user_guide`, move under `reference/topics`, or stay in place with stronger labeling as advanced/theory material? Also decide whether bibliography should be centralized through:

- a dedicated shared bibliography section/page, or
- per-page curated references with a shared citation source.

**Preferred direction:** Keep the page where it is for now, but add stronger page-role labeling and centralize citations through one shared source to avoid moving a stable public URL in the same phase.

**Technical risk:** Medium

### Gate C: Visual Examples vs Case Studies Canonical Source

**Audit items:** `15-3`

**Decision needed:** Should homepage "Visual Examples" and `examples/index` become one canonical visual catalog, or should they remain separate with explicit role separation?

**Preferred direction:** Keep the homepage as a teaser layer and make `examples/index` the canonical visual catalog backed by the same thumbnail assets.

**Technical risk:** Medium

### Gate D: Theme/Search Strategy

**Audit items:** `17-15`, `17-34`, `17-35`, `17-41`, `17-26`, `17-27`, `17-32`, `17-36`

**Decision needed:** Stay on hardened RTD theme for the next phase, or open a dedicated migration to Furo or another theme/search stack?

**Preferred direction:** Do **not** mix theme migration into the next content wave. Run a short spike first, then choose.

**Technical risk:** High

## File Families Affected

### Shared platform files

- `docs/conf.py`
- `docs/_templates/layout.html`
- `docs/_static/custom.css`
- `docs/_static/copybutton.js`

### High-traffic guide pages

- `docs/web/{ja,en}/user_guide/installation.md`
- `docs/web/{ja,en}/user_guide/quickstart.md`
- `docs/web/{ja,en}/user_guide/getting_started.md`
- `docs/web/{ja,en}/user_guide/io_formats.md`
- `docs/web/{ja,en}/user_guide/interop.md`
- `docs/web/{ja,en}/user_guide/time_utilities.md`
- `docs/web/{ja,en}/user_guide/numerical_stability.md`
- `docs/web/{ja,en}/user_guide/scalarfield_slicing.md`
- `docs/web/{ja,en}/user_guide/validated_algorithms.md`
- `docs/web/{ja,en}/user_guide/architecture.md`

### Index and navigation pages

- `docs/web/{ja,en}/index.rst`
- `docs/web/{ja,en}/user_guide/tutorials/index.rst`
- `docs/web/{ja,en}/examples/index.rst`
- `docs/web/{ja,en}/reference/index.rst`

### Tutorial and case-study notebooks/pages

- selected notebooks under `docs/web/{ja,en}/user_guide/tutorials/`

### Internal planning and audit records

- `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md`

## Detailed Roadmap

### Phase 0: Decision Packet and Scope Freeze

**Items:** `4-1`, `12-11`, `12-12`, `15-3`, `17-34`, `17-35`, `17-41`

**Goal:** Resolve the product and architecture choices that would otherwise cause rework.

**Deliverables:**

- one short decision memo for the Consultant
- accepted scope split between "current theme hardening" and "theme/search spike"
- accepted placement policy for `validated_algorithms`
- accepted canonical source policy for visual case-study thumbnails

**Recommended implementation note:** No source changes yet except the planning memo if needed.

**Technical risk:** Low technically, high coordination dependency

### Phase 1: Shared Page Contract Rollout

**Items:** `17-2`, `17-3`, `17-5`, `17-6`, `17-7`, `17-8`, `17-9`, `17-11`

**Goal:** Introduce a repeatable user-facing structure across the main guide pages before more page-specific edits.

**Grouped fixes:**

- terminology normalization:
  - resolve `Field / フィールド / 場` policy (`17-2`)
- standardized page intro contract:
  - audience / prerequisites / use cases (`17-3`)
  - page role label such as guide/theory/reference (`17-7`)
  - shortcut or "on this page" lead-in for long pages (`17-9`)
- standardized page footer contract:
  - "Next to read" sections (`17-5`)
- code example framing:
  - purpose / inputs / outputs blocks (`17-6`)
- JP readability cleanup:
  - reduce unnecessary English overload in JA pages (`17-8`)
- discoverability hints:
  - add search keywords or synonyms to strategic pages (`17-11`)

**Primary targets first:**

- `installation`
- `quickstart`
- `getting_started`
- `io_formats`
- `interop`
- `time_utilities`
- `numerical_stability`
- `validated_algorithms`
- `architecture`
- `tutorials/index`
- `examples/index`
- `reference/index`

**Execution rule:** Define the contract once, then roll it out in JA/EN pairs.

**Technical risk:** Medium

**Why it is early:** This phase reduces drift and makes later content additions cheaper.

### Phase 2: Advanced Guide and Reference Linkage Cleanup

**Items:** `12-11`, `12-12`, `17-4`

**Goal:** Make advanced/theory pages easier to place and navigate without changing the broader IA again.

**Grouped fixes:**

- central bibliography strategy for `validated_algorithms` (`12-11`)
- page placement/noise mitigation for `validated_algorithms` (`12-12`)
- stronger tutorial/API/reference back-links (`17-4`)

**Recommended shape:**

- add a stable citation source for algorithm validation references
- keep `validated_algorithms` public, but clearly badge it as advanced/theory material
- add explicit "related tutorial / related API / related theory" blocks instead of generic links

**Likely file families:**

- `docs/web/{ja,en}/user_guide/validated_algorithms.md`
- `docs/web/{ja,en}/reference/api/*.rst`
- `docs/web/{ja,en}/reference/index.rst`
- selected tutorial entry pages

**Technical risk:** Medium

**Why it is second:** It depends on the shared page contract, but should land before visual polish.

### Phase 3: Visual Navigation Completion

**Items:** `13-9`, `14-9`, `15-1`, `15-3`, optionally `17-37`

**Goal:** Finish the obvious missing visual and failure-mode guidance on the most visible content paths.

**Grouped fixes:**

- architecture data-flow diagram (`13-9`)
- failure-prone examples or "common mistakes" callouts in selected tutorials (`14-9`)
- thumbnail-backed `examples/index` (`15-1`)
- unify homepage visual teasers with canonical case-study visuals (`15-3`)
- optionally seed a lightweight visual index from the same assets (`17-37`)

**Recommended sequencing:**

1. wire existing thumbnail assets into `examples/index`
2. add the architecture flow diagram
3. add failure-mode sections only to high-value notebooks/pages, not to every tutorial
4. decide whether the same thumbnail registry can seed a later visual index

**Likely targets:**

- `docs/web/{ja,en}/examples/index.rst`
- `docs/web/{ja,en}/user_guide/architecture.md`
- selected notebooks under `docs/web/{ja,en}/user_guide/tutorials/`
- existing image assets under `docs/_static/images/`

**Technical risk:** Medium

**Why it is third:** High user impact, but should reuse the IA and copy conventions settled in earlier phases.

### Phase 4: Trust, Reliability, and Contributor-Readiness

**Items:** `4-1`, `17-21`, `17-24`, `17-28`, `17-29`, `17-30`, `17-31`

**Goal:** Address the remaining trust gaps that block wider adoption even if the pages read well.

**Grouped fixes:**

- release/install status transparency (`4-1`)
- code-example verification or smoke checks (`17-21`)
- public-facing coverage summary or coverage policy (`17-24`)
- runtime/memory guidance for expensive workflows (`17-28`)
- error-message reverse lookup or troubleshooting index (`17-29`)
- extension guide for developers (`17-30`)
- GitHub Discussion entry point next to Issues (`17-31`)

**Recommended execution split:**

- **content-only subphase:**
  - install status matrix
  - runtime/memory notes
  - troubleshooting/error index
  - extension guide
  - Discussion link
- **CI/process subphase:**
  - example verification strategy
  - coverage publication strategy

**Technical risk:** Medium to High

**Why it is fourth:** This phase benefits from the page contract and cross-link scaffolding from earlier phases, and some tasks touch CI/process rather than pure docs.

### Phase 5A: Safe Platform Polish Within Current Theme

**Items:** `17-17`, `17-18`, `17-19`, `17-38`, `17-39`

**Goal:** Land the highest-value global polish that can be done without a theme migration.

**Grouped fixes:**

- actual OGP/Twitter meta output (`17-17`)
- external-link behavior and security attributes (`17-18`)
- responsive tables and right-margin overflow cleanup (`17-19`)
- breadcrumb wording clarity (`17-38`)
- stronger visual hierarchy for admonitions (`17-39`)

**Likely file families:**

- `docs/conf.py`
- `docs/_templates/layout.html`
- `docs/_static/custom.css`
- possibly `docs/_static/copybutton.js`

**Technical risk:** Medium

**Why it is separate:** These are global changes, but still compatible with the current RTD-based stack.

### Phase 5B: Platform Spike for Theme/Search/Output Modernization

**Items:** `17-15`, `17-34`, `17-35`, `17-41`, `17-26`, `17-27`, `17-32`, `17-36`, `17-37`

**Goal:** Decide whether GWexpy should continue incremental RTD hardening or move to a more modern docs platform.

**Spike questions:**

- Is RTD theme sufficient if we add better CSS, right-side TOC, and darker color support?
- Would Furo or another theme reduce custom-template burden?
- What is the smallest viable Japanese-search improvement?
- Should PDF/ePub and cheat-sheet output be generated from the main docs source, or maintained separately?
- Are interactive plots worth the build and hosting complexity on GitHub Pages?

**Expected outputs of the spike:**

- comparison matrix: stay vs migrate
- proof-of-concept branch or screenshot set
- risk register for search, TOC, PDF/ePub, dark mode, interactive embeds

**Technical risk:** High

**Execution rule:** Do not combine this spike with the content phases above in the same PR stream.

### Separate Issue: Non-Docs API Follow-up

**Item:** `8-19`

**Goal:** Track the Zarr sampling-rate default issue separately from the docs wave.

**Reason:** This requires API behavior change, tests, and likely user-facing compatibility discussion.

**Technical risk:** Medium

## Priority Order Summary

1. **Phase 0** decision gates
2. **Phase 1** shared page contract
3. **Phase 2** advanced guide/reference linkage
4. **Phase 3** visual navigation completion
5. **Phase 4** trust/reliability/contributor readiness
6. **Phase 5A** safe platform polish
7. **Phase 5B** theme/search/output spike
8. **Separate Issue** for `8-19`

## Execution Rules

- Do not mix content standardization and theme migration in one pass.
- Do not start broad page rewrites before terminology and page-contract decisions are frozen.
- Apply changes to JA/EN in pairs.
- Use existing generated thumbnails and scripts before inventing new asset pipelines.
- Record unresolved release blockers separately from docs-only fixes.

## Testing & Verification Plan

### Baseline for every content phase

- [ ] `cd docs && make html`
- [ ] `sphinx-build -b linkcheck docs docs/_build/linkcheck`
- [ ] spot-check JA/EN parity on changed pages
- [ ] verify that modified index pages still expose the intended navigation

### Additional checks by phase

- **Phase 1:** confirm each target page now has an intro contract and a "next to read" block
- **Phase 2:** confirm tutorial/API/theory links are bidirectional where intended
- **Phase 3:** confirm images render and alt text is present
- **Phase 4:** confirm trust pages link cleanly from installation, troubleshooting, and footer/navigation
- **Phase 5A:** confirm meta tags, external-link behavior, and responsive table behavior in built HTML
- **Phase 5B:** compare screenshots and build warnings before and after the spike

### Release Gate Reminder

Use `web_docs_release_gates` before claiming completion of any implementation phase.

## Models, Recommended Skills, and Effort Estimates

### Recommended model usage

- **Planning / coordination:** `gpt-5.4`
- **Broad docs editing:** `gpt-5.4` or equivalent high-context model
- **Focused page rewrites:** lighter model acceptable once page contract is fixed

### Recommended skills

- `web_docs_ia_overhaul`
- `web_docs_page_rewrite`
- `manage_docs`
- `web_docs_release_gates`

### Estimated effort

- **Phase 0:** 0.5 day, quota low
- **Phase 1:** 1.5 to 2.5 days, quota medium
- **Phase 2:** 1.0 to 1.5 days, quota medium
- **Phase 3:** 1.0 to 2.0 days, quota medium
- **Phase 4:** 1.5 to 2.5 days, quota medium to high
- **Phase 5A:** 1.0 to 1.5 days, quota medium
- **Phase 5B spike:** 2.0 to 4.0 days, quota medium to high
- **Separate Issue `8-19`:** 0.5 to 1.5 days, quota medium

**Estimated total for next docs wave without theme migration:** 6.5 to 10.0 working days

**Estimated total including theme/search/output modernization after spike approval:** 10.0 to 15.0+ working days

## Concerns

- Theme migration can invalidate CSS/template work from earlier phases if started too soon.
- Bibliography centralization can expand unexpectedly if it tries to solve all citation hygiene across the site instead of `validated_algorithms` first.
- Tutorial failure-mode additions can balloon if applied to every notebook. Restrict to high-value tutorials in this phase.
- CI-based code-example verification can become a release-engineering task; keep the docs-facing requirements separate from infrastructure work.
