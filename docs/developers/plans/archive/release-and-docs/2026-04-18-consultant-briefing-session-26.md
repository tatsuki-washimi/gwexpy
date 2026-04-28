# Consultant Briefing for Session 26

## Context

The integrated documentation audit is largely complete on structure and major IA work, but the remaining backlog is now concentrated in cross-cutting UX, advanced-page placement, visual navigation, and docs-platform polish.

Current tracked state from the audit report:

- 142 items done
- 33 items still open
- 3 items need policy confirmation
- 1 item is design-complete but not fully closed in public docs
- 1 item (`8-19`) requires a real API fix, not just documentation

## Recommended Framing

We should not treat the remaining backlog as one homogeneous docs sprint. It has three different classes of work:

1. **Content standardization and navigation**  
   Low-to-medium risk, high ROI, should be done first.

2. **Visual and trust improvements**  
   Medium risk, still compatible with the current Sphinx/RTD stack.

3. **Platform modernization**  
   High risk. This includes theme migration, Japanese search improvement, right-side TOC, dark mode strategy, PDF/ePub, cheat sheets, tooltips, and interactive plots. This should be handled as a separate spike or it will destabilize the content wave.

## Decisions Needed from Session 26

### 1. Should we keep the current RTD-based stack for the next wave?

**Decision:**  
Choose between:

- **Option A: Harden current RTD theme first**
- **Option B: Start theme/search migration now**

**Recommendation:** Option A

**Trade-off:**  
Option A closes more audit items faster and with less regression risk.  
Option B may produce a better long-term UX, but it will reopen layout, CSS, navigation, and verification work across many already-fixed pages.

### 2. Where should `validated_algorithms` live?

**Decision:**  
Choose between:

- **Option A: Keep it in `user_guide` but clearly mark it as advanced/theory**
- **Option B: Move it under `reference/topics` or another advanced area**

**Recommendation:** Option A for this phase

**Trade-off:**  
Keeping the page in place avoids URL churn and rework in existing links.  
Moving it may improve IA purity, but it adds migration work while the real user problem is page labeling and bibliography hygiene, not necessarily the current URL.

### 3. How should we resolve “Visual Examples” vs “Case Studies”?

**Decision:**  
Choose between:

- **Option A: Homepage visual block as teaser, `examples/index` as canonical gallery**
- **Option B: Merge both into one visual-first destination**

**Recommendation:** Option A

**Trade-off:**  
Option A reuses existing thumbnails and preserves the homepage as a landing surface.  
Option B is conceptually cleaner, but risks another IA rewrite when the current problem is mostly missing thumbnails and ambiguous labeling.

### 4. How explicit should installation messaging be before PyPI/Conda exist?

**Decision:**  
Choose between:

- **Option A: Keep “development install” as the primary path, but add a release-status matrix**
- **Option B: Continue vague future-package wording**

**Recommendation:** Option A

**Trade-off:**  
Option A improves trust immediately without changing packaging reality.  
Option B keeps the page shorter, but preserves an avoidable credibility gap.

## Proposed Execution Order

### Phase 1: Shared page contract

Close cross-cutting items like:

- page role labels
- audience/prerequisites/use-cases blocks
- “next to read” sections
- code example framing
- long-page shortcuts
- terminology normalization
- Japanese readability cleanup

**Risk:** Medium  
**Why first:** It reduces drift and makes later edits cheaper.

### Phase 2: Advanced-page linkage

Close:

- `12-11`, `12-12`, `17-4`

This means:

- bibliography strategy for `validated_algorithms`
- stronger placement cues for advanced content
- tutorial/API/theory bidirectional links

**Risk:** Medium

### Phase 3: Visual navigation

Close:

- `13-9`, `14-9`, `15-1`, `15-3`

This means:

- architecture data-flow diagram
- selected failure-mode examples in tutorials
- thumbnail-backed `examples/index`
- canonical relation between homepage visuals and case-study gallery

**Risk:** Medium

### Phase 4: Trust and contributor readiness

Close:

- `4-1`, `17-21`, `17-24`, `17-28`, `17-29`, `17-30`, `17-31`

This means:

- install status transparency
- example verification policy
- coverage publication or coverage policy
- runtime/memory guidance
- troubleshooting/error index
- developer extension guide
- Discussion entry point

**Risk:** Medium to High

### Phase 5: Safe platform polish within current theme

Close:

- `17-17`, `17-18`, `17-19`, `17-38`, `17-39`

This means:

- real OGP output
- external-link behavior
- responsive tables and overflow cleanup
- breadcrumb wording
- stronger admonition hierarchy

**Risk:** Medium

### Separate spike: Platform modernization

Investigate before implementing:

- `17-15`, `17-34`, `17-35`, `17-41`, `17-26`, `17-27`, `17-32`, `17-36`, `17-37`

This includes:

- dark mode strategy
- theme migration
- Japanese search
- right-side TOC
- PDF/ePub
- cheat sheet
- glossary tooltips
- interactive plots
- visual index

**Risk:** High

### Separate engineering issue

- `8-19` Zarr default sampling-rate behavior

This is not a pure docs task.

## Session Estimate

Assumption: one “worker session” is a focused implementation session similar in scope to Sessions 27-30, ending with local verification and audit-report updates.

### Best case

**6 sessions**

This is only realistic if:

- Session 26 approves “harden current theme first”
- theme migration and search modernization are deferred to a later spike
- `8-19` is tracked outside the docs wave

### Most likely

**8 sessions**

Suggested breakdown:

1. shared page contract, part 1
2. shared page contract, part 2
3. advanced-page linkage and bibliography cleanup
4. visual navigation and examples gallery
5. trust/readiness docs wave
6. current-theme platform polish
7. platform spike / decision branch
8. finalize remaining open items, verification, and audit closure

### Worst case

**10 to 11 sessions**

This happens if we decide to include real theme/search/output modernization in the same completion target, or if `8-19` is included in the same completion definition.

## Recommendation

For planning purposes, we should tell Session 26:

> **Plan for 8 additional worker sessions to reach practical full closure, with a realistic range of 6 to 11 depending on whether theme/search modernization is deferred or absorbed into the same program.**

The critical architectural recommendation is:

> **Do not mix content standardization and theme migration in the same execution wave.**

That is the main lever that determines whether this closes efficiently or sprawls.
