# Phase 4 Post-Release Roadmap

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the released docs set, separate internal-only artifacts safely, and ship the next low-regression UX and trust improvements without mixing them with theme migration.

**Architecture:** Treat Phase 4 as a post-release cleanup and trust-building wave, not as a continuation of release-critical remediation. Keep three lanes separate: `public docs stabilization`, `internal repo hygiene`, and `future platform modernization`.

**Tech Stack:** Sphinx, MyST/Markdown, reStructuredText, `conda run -n gwexpy sphinx-build`, docs-only Git workflow.

---

## Release Baseline

### Included in This Release

- Gateway / JA-EN hub refresh and navigation restructuring
- `io_formats` regrouping and `interop` split
- `validated_algorithms` strengthening and API backlink work
- tutorial/case-study expansion including Bruco / HHT failure-mode additions
- reference hub / topics / class-page enrichment
- site hygiene already closed in the audit: glossary, changelog, citation, copy button, favicon, alt text, license/disclaimer, roadmap, feedback button, external-link CI

### Explicit Leftovers Still Tracked in the Audit Report

- `4-1`: install trust gap around PyPI / Conda messaging
- `8-19`: Zarr default 1 Hz fallback, separate API issue
- `13-9`: architecture / data-flow diagram missing
- `15-1`: examples gallery still lacks thumbnails
- `15-3`: visual examples vs case-study gallery relation still pending
- `17-2`, `17-3`, `17-5`, `17-6`, `17-7`, `17-8`, `17-9`, `17-11`
- `17-15`, `17-17`, `17-18`, `17-19`
- `17-21`, `17-24`, `17-26`, `17-27`, `17-28`, `17-29`, `17-30`, `17-31`, `17-32`
- `17-34`, `17-35`, `17-36`, `17-37`, `17-38`, `17-39`, `17-41`

### Operational Recheck After Release

- `17-4` is marked complete in the audit report, but class-page backlink quality should be spot-checked one more time after the current reference sweep settles.

---

## Phase 4 Rules

- Do not mix Phase 4 with theme migration.
- Do not move internal files before the release build and link gates are clean.
- Freeze the current public-build exclusions through the release cut. Do not change `exclude_patterns` again unless a new release-blocking build leak is found.
- Keep JA / EN copy changes paired in the same PR.
- Keep URL paths stable during wording cleanup; adjust labels, headings, helper text, and navigation text only.
- Keep `docs/developers/**` and `docs/superpowers/plans/**` excluded from the public build until the physical migration is complete.

---

## Workstreams

### Workstream A: Public Docs Stabilization

**Purpose:** Ship low-regression improvements that increase trust and reduce user confusion without changing site architecture.

**Audit IDs:**
- `4-1`
- `17-21`
- `17-24`
- `17-28`
- `17-29`
- `17-30`
- `17-31`

**Primary Files:**
- `docs/web/{ja,en}/user_guide/installation.md`
- `docs/web/{ja,en}/reference/**`
- `docs/web/{ja,en}/user_guide/**`
- `docs/web/{ja,en}/info/**`
- `docs/web/{ja,en}/index.rst`

- [ ] Add a release/install status matrix to installation docs and clarify what is source-install only.
- [ ] Add a public troubleshooting / error-index page and connect it from installation and tutorials.
- [ ] Add runtime / memory guidance to advanced pages where cost surprises are likely.
- [ ] Publish a minimal coverage / example-verification policy page instead of promising hidden CI guarantees.
- [ ] Add a Discussion entry point near existing feedback / Issue surfaces.
- [ ] Draft an extension guide entry point for contributor-facing customization.
- [ ] Verify HTML build and linkcheck after each additive PR.

**Risk:** Medium-low. Mostly additive content and links.

### Workstream B: Internal Repo Hygiene

**Purpose:** Remove internal-only artifacts from `docs/` without destabilizing the released public site.

**Targets:**
- `docs/developers/**`
- `docs/superpowers/plans/**`
- later decision pass: `docs/NOTEBOOK_POLICY.md`, `docs/repro/**`

**Consultant Decision:**
- Do not move these trees before release.
- Execute the physical migration in a single Week 2 batch, not piecemeal.
- Keep the current exclusion-based containment in place until that migration lands.

**Primary Files:**
- `docs/conf.py`
- `docs_internal/**`
- any repo docs that still reference old paths

- [ ] Freeze the released public docs baseline first.
- [ ] Mirror `docs/developers/**` into `docs_internal/developers/**`.
- [ ] Mirror `docs/superpowers/plans/**` into `docs_internal/superpowers/plans/**`.
- [ ] Update references in one sweep, not piecemeal.
- [ ] Re-run Sphinx build to confirm public docs still exclude the intended trees.
- [ ] Delete old copies only after path and reference verification passes.
- [ ] Decide separately whether `NOTEBOOK_POLICY.md` and `repro/**` should remain repo-visible or also move.

**Risk:** Medium. Path churn can create hidden stale references if split into small moves.

### Workstream C: User-Facing Wording Refinement

**Purpose:** Replace dev-centric labels with user-facing copy once Gateway / Hub structure stabilizes.

**Audit IDs:**
- `17-2`
- `17-8`
- partial follow-up to `15-3`

**Primary Files:**
- `docs/web/{ja,en}/index.rst`
- `docs/web/{ja,en}/examples/index.rst`
- `docs/web/{ja,en}/user_guide/**/*.md`
- `docs/web/{ja,en}/reference/**/*.md`

- [ ] Finalize a terminology map for `Field / フィールド / 場` and related bilingual labels.
- [ ] Replace dev-centric surface labels such as `Hub` with user-facing navigation labels.
- [ ] Keep legacy terms as aliases in helper text or search keywords for one release cycle.
- [ ] Apply the wording pass in JA / EN together.
- [ ] Avoid structural changes in this pass: no URL moves, no toctree reshuffle, no theme work.

**Risk:** Medium. Easy to cause JA/EN drift or re-open IA debates if combined with structural edits.

### Workstream D: Visual Navigation Completion

**Purpose:** Finish the remaining visual/document-structure items that are still content-shaped rather than theme-shaped.

**Audit IDs:**
- `13-9`
- `15-1`
- `15-3`

**Primary Files:**
- `docs/web/{ja,en}/user_guide/architecture.md`
- `docs/web/{ja,en}/examples/index.rst`
- `docs/_static/images/phase3/**`

- [ ] Add the architecture / data-flow diagram.
- [ ] Add thumbnails and difficulty markers to the examples gallery.
- [ ] Reconcile `Visual Examples` and the case-study gallery only after Hub wording is stable.
- [ ] Keep asset generation scripts reproducible and checked into the same PR as the docs changes.

**Risk:** Medium. Mostly safe, but touches homepage/gallery surfaces and can conflict with ongoing Hub polish.

### Workstream E: Deferred Platform Modernization Spike

**Purpose:** Isolate high-risk platform changes from Phase 4 so the post-release stabilization window stays clean.

**Audit IDs:**
- `17-15`
- `17-17`
- `17-18`
- `17-19`
- `17-26`
- `17-27`
- `17-32`
- `17-34`
- `17-35`
- `17-36`
- `17-37`
- `17-38`
- `17-39`
- `17-41`

- [ ] Open a separate spike for theme, search, right-side TOC, PDF/ePub, cheat sheet, glossary tooltip, interactive plot, and broader visual index work.
- [ ] Evaluate RTD hardening versus Furo/theme migration as an explicit architecture decision.
- [ ] Do not start this spike until the release stabilization window is closed.

**Risk:** High. These items cross theme, layout, search, and packaging boundaries.

---

## Timing

### Week 1 After Release

- Workstream A first:
  - install/status transparency
  - troubleshooting / error index
  - runtime / memory guidance
- Begin Workstream B only after release gates are green and the release branch is frozen.

### Week 2 After Release

- Start Week 2 with the one-shot physical migration for clearly internal trees:
  - `docs/developers/**`
  - `docs/superpowers/plans/**`
- Keep `NOTEBOOK_POLICY.md` and `repro/**` out of this first sweep unless separately approved.
- Run Workstream C wording cleanup as a dedicated copy-edit wave.
- Start Workstream D only if Gateway / Hub live labels are no longer moving.

### After Stabilization Window

- Decide whether to open Workstream E as a separate modernization project.

---

## Verification

- `python3 ~/ai-harness/bin/verify-changed-files --changed`
- `conda run -n gwexpy sphinx-build -b html -W --keep-going -D nbsphinx_execute=never docs /tmp/gwexpy-phase4-html`
- `conda run -n gwexpy sphinx-build -b linkcheck -D nbsphinx_execute=never docs /tmp/gwexpy-phase4-linkcheck`

---

## Success Criteria

- Public release docs remain build-clean while internal-only docs are no longer mixed into the public tree.
- User-facing labels become less developer-centric without URL churn.
- The next trust-oriented doc gaps are closed before any theme/search rewrite begins.
- Theme/search/platform work is explicitly isolated as a later spike, not smuggled into Phase 4.
